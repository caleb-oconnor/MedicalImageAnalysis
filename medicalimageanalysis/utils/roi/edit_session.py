"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Incremental Contour-to-Mesh Editing
===================================

Description:
    This module keeps the shape-interpolation field of a contour set in
    memory so an interactive edit of one slice only recomputes a thin slab
    of the field (slices k-1..k+1) and splices the re-meshed slab back into
    the kept mesh. It is the editing counterpart of the one-shot
    `contours_to_mesh` in `to_mesh`, built on the same `_contourmesh`
    engine.

Classes:
    * EditSession         one spatial region: field volume + raw mesh, local updates
    * RegionEditSession   one EditSession per separate region, routes each edit

Key Functionalities:
    1. **Local Update**: `EditSession.update_slices` changes, adds or
       removes slices (top, bottom or in between) by recomputing only the
       field and mesh next to them; the rest of the field and mesh is moved
       to its new place in the plane stack. Only a contour that leaves the
       padded lattice falls back to a rebuild of that region.
    2. **Region Routing**: `RegionEditSession.update_slices` decides, per
       slice, whether an edit updates one region, starts a new one, merges
       several, or empties / drops one, then updates each affected region
       once for the whole batch.
    3. **Final Mesh**: `RegionEditSession.finalize` recomputes the regions
       and their lattice spacing from scratch, the same as `contours_to_mesh`.

Accuracy Note:
    A local update gives the mesh a rebuild of the same contours gives, to
    float32 precision of the Flying Edges output. The one exception is rare
    ties in the nearest-sample distance query (`_Volume._query`), which can
    move a vertex by a few hundredths of a millimetre; the existing per-slice
    update had the same property.

Geometry Note:
    The sessions work in a contour-aligned `Frame` (u, v span the drawing
    plane, n is the slice normal) and key the slices by slice index.
    `image.matrix` stores the image axes in its rows; the frame needs them in
    columns, so pass it through `normalize_direction` first. To match
    `Roi.create_mesh` (interpolate=False), use the image spacing along the
    contour axis as `slice_spacing`.

Usage:
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi._contourmesh import Frame
    >>> from medicalimageanalysis.utils.roi._raster import normalize_direction
    >>> from medicalimageanalysis.utils.roi.edit_session import RegionEditSession
    >>> D = normalize_direction(image.matrix)               # rows -> columns
    >>> frame = Frame(np.asarray(image.origin, float), D[:, 0], D[:, 1], D[:, 2])   # axial contours
    >>> session = RegionEditSession(slices, frame, image.spacing[2])
    >>> session.update_slice(40, rings_on_slice_40)          # all rings now on slice 40
    >>> mesh = session.mesh(reduction=0)

Examples
--------
>>> import numpy as np
>>> from medicalimageanalysis.utils.roi._contourmesh import Frame
>>> from medicalimageanalysis.utils.roi.edit_session import RegionEditSession
>>> frame = Frame(np.zeros(3), np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.]))
>>> square = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=float)
>>> slices = {k: [np.c_[square, np.full(4, 2.0 * k)]] for k in range(5)}   # 2 mm slices
>>> session = RegionEditSession(slices, frame, 2.0)
>>> session.update_slice(2, [np.c_[1.5 * square, np.full(4, 4.0)]])      # widen slice 2
>>> session.last_route
'local(local)'
>>> mesh = session.mesh(reduction=0)
"""

import numpy as np
from scipy.spatial import cKDTree

from ._contourmesh import (BIG, _Volume, _dense, _finish, _inside, _prepare, _to_polydata, auto_h,
                           contours_to_mesh_regions, postprocess, split_regions)


class EditSession(object):
    """
    Field volume and raw mesh of one spatial region, kept for local updates.

    An edit only recomputes the field around the slices it touches and
    re-meshes those slabs; the rest of the field and mesh is kept and the new
    slabs are welded in. This holds for changed contours, for slices that are
    added (top, bottom or in between) and for slices that are removed: the
    kept part of the field and mesh is moved to its new place in the plane
    stack instead of being recomputed. A full rebuild only happens when a
    contour leaves the padded lattice. The result is the same mesh a rebuild
    of the same contours gives. The lattice spacing `h` is fixed for the life
    of the session.

    Attributes
    ----------
    frame : Frame
        Contour-aligned coordinate frame.

    slices : dict[int, list[np.ndarray]]
        Slice index -> world rings (n, 3) currently in the session.

    keys : list[int]
        Slice indices in field order (sorted by slice position).

    vol : _Volume | None
        Field volume. None when the session has no contours.

    P, T : np.ndarray
        Raw Flying Edges mesh in the field's (x, y, plane index) space:
        (m, 3) points and (t, 3) triangles.

    last : str
        How the last update ran: 'rebuild', 'local' or 'empty'.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi._contourmesh import Frame
    >>> from medicalimageanalysis.utils.roi.edit_session import EditSession
    >>> frame = Frame(np.zeros(3), np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.]))
    >>> square = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=float)
    >>> slices = {k: [np.c_[square, np.full(4, 2.0 * k)]] for k in range(5)}
    >>> session = EditSession(slices, frame, 2.0)
    >>> session.update_slices([0, 1], [[], []])        # erase the two bottom slices: still local
    >>> session.last
    'local'
    """

    _SLICE_ARRAYS = ("M", "L", "Aup", "Bdn", "nearI", "nearJ", "dM")

    def __init__(self, slices, frame, slice_spacing, h=None, m=1, c=None, gap_factor=1.5, pad_mm=15.0, workers=1,
                 cap_radius=None):
        """
        Build the field volume and the raw mesh of one region.

        Parameters
        ----------
        slices : dict[int, list[np.ndarray]]
            Slice index -> world rings (n, 3) of this region.

        frame : Frame
            Contour-aligned coordinate frame.

        slice_spacing : float
            Spacing between contour slices in mm. Slices more than
            `gap_factor` times this apart are capped instead of joined.

        h : float | None
            In-plane lattice spacing in mm. Automatic (`auto_h`) if None.

        m : int
            Sub-planes per half slab (1 = one mid plane).

        c : float | None
            Hat-field cap constant. Defaults to `h`.

        gap_factor : float
            Largest slice gap, in slice spacings, that is still joined.

        pad_mm : float
            Extra lattice margin around the contours, in mm, so edits that
            grow a contour stay local instead of forcing a rebuild.

        workers : int
            KD-tree query threads.

        cap_radius : float | None
            Rounded-cap radius in mm. Defaults to half the slice spacing.

        Examples
        --------
        >>> import numpy as np
        >>> from medicalimageanalysis.utils.roi._contourmesh import Frame
        >>> frame = Frame(np.zeros(3), np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.]))
        >>> square = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=float)
        >>> session = EditSession({0: [np.c_[square, np.zeros(4)]]}, frame, 2.0, pad_mm=25.0)
        >>> session.last
        'rebuild'
        """
        if h is None:
            h = auto_h(slices, frame)

        self.frame = frame
        self.delta = float(slice_spacing)
        self.h = float(h)
        self.m = int(m)
        self.c = self.h if c is None else float(c)
        self.margin = 3
        self.pad = int(np.ceil(pad_mm / self.h))
        self.gap_factor = float(gap_factor)
        self.workers = workers
        self.cap_radius = 0.5 * self.delta if cap_radius is None else float(cap_radius)

        self.slices = {k: list(v) for k, v in slices.items()}
        self.keys = []
        self.vol = None
        self.P = np.zeros((0, 3))
        self.T = np.zeros((0, 3), np.int64)
        self.last = None

        self._rebuild()

    @staticmethod
    def _runs(mask):
        """(start, stop) index pairs of the True runs of a 1D boolean array."""
        edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    @staticmethod
    def _weld(P, T, Pn, Tn, p0, p1):
        """Append a re-extracted block (planes p0..p1) to the kept mesh, merging the vertices on planes p0 and p1."""
        ob = np.flatnonzero((P[:, 2] == p0) | (P[:, 2] == p1))
        nb = np.flatnonzero((Pn[:, 2] == p0) | (Pn[:, 2] == p1))
        remap = np.arange(len(Pn)) + len(P)
        if ob.size and nb.size:
            keys = np.round(np.vstack([P[ob], Pn[nb]]) * 1e6).astype(np.int64)
            _, inv = np.unique(keys, axis=0, return_inverse=True)
            inv = inv.ravel()
            first = np.full(inv.max() + 1, -1)
            first[inv[:ob.size]] = ob
            hit = first[inv[ob.size:]]
            remap[nb[hit >= 0]] = hit[hit >= 0]

        keepn = np.ones(len(Pn), bool)
        keepn[nb[remap[nb] < len(P)]] = False
        shift = np.cumsum(keepn) - 1 + len(P)
        remap = np.where(keepn, shift, remap)

        return np.vstack([P, Pn[keepn]]), np.vstack([T, remap[Tn]])

    def _apply(self, new_local):
        """
        Local update after self.slices changed. new_local: key -> (slice position, local 2D rings) for every slice
        whose contours were replaced (added or changed); removed keys are simply absent from self.slices.

        Only the slices next to a change are re-classified and re-fielded, only the planes of the affected slabs are
        re-evaluated, and only those slabs are re-meshed. Everything else (field rows, planes, mesh) is copied to
        its new position in the plane stack.
        """
        v = self.vol
        old_keys = list(self.keys)
        old_index = {k: i for i, k in enumerate(old_keys)}
        old_link = v.link.copy()
        old_kind, old_k, old_t = v.pkind.copy(), v.pk.copy(), v.pt.copy()
        old_V = v.V

        position = {k: new_local[k][0] if k in new_local else float(v.s[old_index[k]]) for k in self.slices}
        new_keys = sorted(self.slices, key=lambda k: position[k])
        S = len(new_keys)
        s_new = np.array([position[k] for k in new_keys], dtype=float)
        link_new = np.diff(s_new) <= self.gap_factor * self.delta
        changed = [i for i, k in enumerate(new_keys) if k in new_local]
        copy_from = np.array([old_index[k] if k in old_index and k not in new_local else -1 for k in new_keys],
                             dtype=np.int64)

        # Affected slabs: slab i joins slices i and i+1; -1 owns the bottom cap and S-1 the top cap.
        slabs = set()
        for i in changed:
            slabs |= {i - 1, i}
        for i in range(S - 1):
            a, b = old_index.get(new_keys[i]), old_index.get(new_keys[i + 1])
            if a is None or b is None or b != a + 1 or link_new[i] != old_link[a]:
                slabs.add(i)
        if new_keys[0] != old_keys[0]:
            slabs.add(-1)
        if new_keys[-1] != old_keys[-1]:
            slabs.add(S - 1)

        refield = set(changed)
        for p in slabs:
            refield |= {q for q in (p, p + 1) if 0 <= q < S}

        # Per-slice state: same rows -> edit in place; added/removed slices -> copy kept rows to their new index.
        if new_keys != old_keys:
            keep = copy_from >= 0
            old_to_new = np.full(len(old_keys), -1, dtype=np.int64)
            old_to_new[copy_from[keep]] = np.flatnonzero(keep)
            for name in self._SLICE_ARRAYS:
                old = getattr(v, name)
                new = np.full((S,) + old.shape[1:], BIG if name == "dM" else 0, dtype=old.dtype)
                new[keep] = old[copy_from[keep]]
                setattr(v, name, new)

            sel = old_to_new[v.psl] >= 0          # whole slices are kept or dropped, so ring offsets stay valid
            v.pts, v.plab, v.prel, v.nrel = v.pts[sel], v.plab[sel], v.prel[sel], v.nrel[sel]
            v.psl = old_to_new[v.psl[sel]]
            v.sub = {(name, int(old_to_new[k])): f for (name, k), f in v.sub.items() if old_to_new[k] >= 0}

        v.S, v.s, v.link = S, s_new, link_new
        v.Aup[S - 1] = False                      # the end slices have no partner beyond them
        v.Bdn[0] = False
        if changed:
            v.set_contours(np.asarray(changed), [new_local[new_keys[i]][1] for i in changed])
        v.classify(sorted(p for p in slabs if 0 <= p < S - 1))
        v.fields(sorted(refield))

        # Planes: re-evaluate those owned by an affected slab (and the real planes of re-fielded slices), copy the
        # rest from their old position.
        v._layout()
        n_new = len(v.pz)
        owner = np.where(v.pkind == 3, v.pk - 1, v.pk)
        dirty = np.where(v.pkind == 0, np.isin(v.pk, sorted(refield)), np.isin(owner, sorted(slabs)))
        old_plane = {(int(kd), int(k), float(t)): q for q, (kd, k, t) in enumerate(zip(old_kind, old_k, old_t))}
        src = np.full(n_new, -1, dtype=np.int64)
        for q in np.flatnonzero(~dirty):
            src[q] = old_plane.get((int(v.pkind[q]), old_index[new_keys[v.pk[q]]], float(v.pt[q])), -1)
        dirty |= src < 0

        same_planes = n_new == len(old_kind) and np.array_equal(src[~dirty], np.flatnonzero(~dirty))
        if not same_planes:
            v.V = np.empty((n_new,) + old_V.shape[1:], dtype=old_V.dtype)
            v.V[~dirty] = old_V[src[~dirty]]
        for p0, p1 in self._runs(dirty):
            v.planes(p0, p1 - 1)

        old_to_new_plane = np.full(len(old_kind), -1, dtype=np.int64)
        old_to_new_plane[src[src >= 0]] = np.flatnonzero(src >= 0)
        self._remesh(old_to_new_plane, n_new)

        self.keys = new_keys
        self.last = "local"

    def _rebuild(self):
        """Recompute the whole field and mesh of this region from `self.slices`."""
        keys, s, loc = _prepare(self.slices, self.frame)
        self.keys = keys
        if not keys:
            self.vol = None
            self.P, self.T = np.zeros((0, 3)), np.zeros((0, 3), np.int64)
            self.last = "empty"
            return

        self.vol = _Volume(loc, s, self.delta, self.h, self.m, self.c, self.margin, self.pad, self.gap_factor,
                           self.workers, True, False, self.cap_radius)       # mid=True, flat_caps=False
        self.P, self.T = self.vol.extract(0, len(self.vol.pz) - 1)
        self.last = "rebuild"

    def _remesh(self, old_to_new_plane, n_new):
        """Keep the mesh of every field cell whose two planes were copied unchanged (moved to their new plane
        index), drop the rest, and re-extract the uncovered cells block by block."""
        P, T = self.P, self.T
        n_old = len(old_to_new_plane)
        cell_keep = (old_to_new_plane[:-1] >= 0) & (old_to_new_plane[1:] == old_to_new_plane[:-1] + 1)

        if len(T):
            layer = np.clip(np.floor(P[T, 2].mean(axis=1)).astype(np.int64), 0, n_old - 2)
            T = T[cell_keep[layer]]
        used = np.zeros(len(P), bool)
        used[T.ravel()] = True
        newidx = np.cumsum(used) - 1
        P, T = P[used].copy(), newidx[T]
        base = np.clip(np.floor(P[:, 2]).astype(np.int64), 0, n_old - 1)
        P[:, 2] = old_to_new_plane[base] + (P[:, 2] - base)

        covered = np.zeros(max(n_new - 1, 0), bool)
        covered[old_to_new_plane[:-1][cell_keep]] = True
        for c0, c1 in self._runs(~covered):
            Pn, Tn = self.vol.extract(c0, c1)
            if len(Pn):
                P, T = self._weld(P, T, Pn, Tn, c0, c1)

        self.P, self.T = P, T

    def mesh(self, post=True, reduction=0.5, smooth_iters=0, method="auto", as_polydata=True):
        """
        Mesh of this region in world coordinates, optionally decimated and smoothed.

        Parameters
        ----------
        post : bool
            Run `postprocess` (decimation / smoothing).

        reduction : float
            Decimation fraction, 0 to 1.

        smooth_iters : int
            Windowed-sinc smoothing iterations (0 = none).

        method : str
            Decimation method passed to `postprocess`.

        as_polydata : bool
            True returns pv.PolyData, False returns (points, triangles).

        Returns
        -------
        pv.PolyData | tuple[np.ndarray, np.ndarray] | None
            None if the region has no contours.

        Examples
        --------
        >>> poly = session.mesh(reduction=0.5)
        >>> P, T = session.mesh(post=False, as_polydata=False)
        """
        if self.vol is None:
            return None

        P, T = self.raw_mesh()
        if post:
            P, T = postprocess(P, T, reduction, smooth_iters, method=method)

        return _to_polydata(P, T) if as_polydata else (P, T)

    def raw_mesh(self):
        """
        Undecimated welded mesh in world coordinates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (m, 3) world points and (t, 3) triangles. Empty arrays if the
            region has no contours.

        Examples
        --------
        >>> P, T = session.raw_mesh()
        >>> P.shape[1], T.shape[1]
        (3, 3)
        """
        if self.vol is None:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)

        Pw = self.frame.to_world(self.vol.to_local_xyz(self.P))
        return _finish(self.frame, Pw, self.T)

    def update_slice(self, key, rings_world):
        """
        Replace the contours of one slice (see `update_slices`).

        Parameters
        ----------
        key : int
            Slice index.

        rings_world : list[np.ndarray]
            All world rings (n, 3) now on that slice. An empty list removes
            the slice.

        Returns
        -------
        None

        Examples
        --------
        >>> session.update_slice(2, [np.c_[1.5 * square, np.full(4, 4.0)]])
        >>> session.last
        'local'
        """
        self.update_slices([key], [rings_world])

    def update_slices(self, keys, rings_per_key):
        """
        Replace the contours of several slices and update the field and mesh around them in one pass.

        Slices can be changed, added (above, below or between the current
        slices) or removed (empty ring list); all of it stays local. Only a
        contour that leaves the padded lattice forces a rebuild of the region.

        Parameters
        ----------
        keys : list[int]
            Slice indices.

        rings_per_key : list[list[np.ndarray]]
            For each key, all world rings (n, 3) now on that slice. An empty
            list removes the slice.

        Returns
        -------
        None

        Examples
        --------
        Erase the bottom two slices and widen the top one, in one update:

        >>> session.update_slices([0, 1, 4], [[], [], [np.c_[1.5 * square, np.full(4, 8.0)]]])
        >>> session.last
        'local'

        Add a slice above the top:

        >>> session.update_slices([5], [[np.c_[square, np.full(4, 10.0)]]])
        >>> sorted(session.slices)
        [2, 3, 4, 5]
        """
        new_local = {}
        for key, rings in zip(keys, rings_per_key):
            rings = [np.asarray(r, float) for r in rings if len(r) >= 3]
            local_keys, local_s, local_rings = _prepare({key: rings}, self.frame) if rings else ([], [], [])
            if local_keys:
                self.slices[key] = list(rings)
                new_local[key] = (float(local_s[0]), local_rings[0])
            else:
                self.slices.pop(key, None)
                new_local.pop(key, None)

        v = self.vol
        if v is None or not self.slices:
            return self._rebuild()

        lim_lo = np.array([(v.i0 + v.margin) * v.h, (v.j0 + v.margin) * v.h])
        lim_hi = np.array([(v.i0 + v.nx - 1 - v.margin) * v.h, (v.j0 + v.ny - 1 - v.margin) * v.h])
        for _, local_rings in new_local.values():
            allp = np.vstack(local_rings)
            if (allp.min(0) < lim_lo).any() or (allp.max(0) > lim_hi).any():
                return self._rebuild()

        self._apply(new_local)


class _Region(object):
    """One spatially separate region of a RegionEditSession: its slices, their rings in the local frame (cached
    for routing), session, lattice spacing, and the slice changes routed to it but not yet applied to its session
    (applied together at the end of a batch)."""

    def __init__(self, slices, local, session, h):
        self.slices = slices
        self.local = local
        self.session = session
        self.h = h
        self.pending = {}


class RegionEditSession(object):
    """
    One EditSession per spatially separate region, each with its own lattice spacing.

    `update_slice(key, rings)` routes the edit:

    * touches one region -> that region's `EditSession.update_slice` (local)
    * touches no region -> a new region
    * touches several regions -> they are merged into one new region
    * a region lost its rings on `key` -> that region is updated, or dropped
      when it has no slices left

    A region's `h` is fixed while editing; call `finalize` for the
    from-scratch result (regions and `h` recomputed).

    Attributes
    ----------
    frame : Frame
        Contour-aligned coordinate frame.

    dz : float
        Slice spacing in mm.

    regions : list[_Region]
        Current regions.

    last_route : str
        Comma-separated routes taken by the last update, e.g. 'local(local)',
        'new', 'merge(2)', 'emptied-slice', 'drop' or 'no-op'.

    Examples
    --------
    Start from an ROI's world contours, grouped by axial slice index:

    >>> import numpy as np
    >>> import medicalimageanalysis as mia
    >>> from medicalimageanalysis.utils.roi._contourmesh import Frame
    >>> from medicalimageanalysis.utils.roi._raster import normalize_direction
    >>> from medicalimageanalysis.utils.roi.edit_session import RegionEditSession
    >>> image = mia.Data.image['CT 01']
    >>> roi = image.roi['Liver']
    >>> slices = {}
    >>> for pixel, position in zip(roi.contour_pixel, roi.contour_position):
    ...     slices.setdefault(int(np.round(pixel[0, 2])), []).append(position)
    >>> D = normalize_direction(image.matrix)
    >>> frame = Frame(np.asarray(image.origin, float), D[:, 0], D[:, 1], D[:, 2])
    >>> session = RegionEditSession(slices, frame, image.spacing[2])
    >>> roi.mesh = session.mesh(reduction=0)
    """

    def __init__(self, slices, frame, slice_spacing, pad=0.5, gap_factor=1.5, step=0.25, **kw):
        """
        Split the contours into regions and start one EditSession per region.

        Parameters
        ----------
        slices : dict[int, list[np.ndarray]]
            Slice index -> world rings (n, 3). May be empty (a new ROI).

        frame : Frame
            Contour-aligned coordinate frame.

        slice_spacing : float
            Spacing between contour slices in mm.

        pad : float
            Distance in mm within which rings on linked slices count as
            touching (same region).

        gap_factor : float
            Largest slice gap, in slice spacings, that is still joined.

        step : float
            Ring resampling step in mm for the touch test.

        **kw
            Passed to every `EditSession` (m, c, pad_mm, workers, cap_radius).

        Examples
        --------
        An empty session for a new ROI; the first edit creates a region:

        >>> session = RegionEditSession({}, frame, 2.0)
        >>> session.mesh() is None
        True
        >>> session.update_slice(0, [np.c_[square, np.zeros(4)]])
        >>> session.last_route
        'new'
        """
        self.frame = frame
        self.dz = float(slice_spacing)
        self.pad = pad
        self.gap = gap_factor
        self.step = step
        self.kw = kw
        self.regions = []
        self.last_route = "init"

        if slices:
            for group in split_regions(slices, frame, slice_spacing, gap_factor, pad):
                self._new_region(group)

    def _local(self, ring):
        """(mean slice position, (n, 2) in-plane points) of a world ring in the local frame."""
        q = self.frame.to_local(np.asarray(ring, float))
        return q[:, 2].mean(), q[:, :2]

    def _near_rings(self, region, z0, skip_key):
        """Local 2D rings of a region on slices within link reach of z0, excluding slice skip_key."""
        reach = self.gap * self.dz + 1e-6
        out = []
        for k, rings in region.local.items():
            if k == skip_key:
                continue
            for z, q in rings:
                if abs(z - z0) <= reach:
                    out.append(q)

        return out

    def _new_region(self, group):
        """Start an EditSession for a group of slices, with its own automatic h."""
        h = auto_h(group, self.frame)
        session = EditSession(group, self.frame, self.dz, h=h, gap_factor=self.gap, **self.kw)
        local = {k: [self._local(r) for r in v] for k, v in group.items()}
        self.regions.append(_Region({k: list(v) for k, v in group.items()}, local, session, h))

    def _route(self, key, rings_world):
        """Route one slice's new rings to the regions (see `update_slices`); returns the routes taken. Local changes
        are only recorded in each region's `pending`, so a batch touching one region updates its session once."""
        rings_world = [np.asarray(r, float) for r in rings_world if len(r) >= 3]
        new = [self._local(r) for r in rings_world]
        n_new, n_reg = len(new), len(self.regions)
        parent = list(range(n_new + n_reg))  # nodes: new rings, then regions

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n_new):  # new rings among themselves (same slice)
            for j in range(i + 1, n_new):
                if self._touch(new[i][1], new[j][1]):
                    union(i, j)

        for i, (z, q) in enumerate(new):  # new rings vs each region's neighbours
            for rr, region in enumerate(self.regions):
                if any(self._touch(q, p) for p in self._near_rings(region, z, key)):
                    union(i, n_new + rr)

        # Rings still without a region are matched against the rings being replaced on this slice, so a region
        # that only lives on `key` (one-slice fragment) is updated in place instead of dropped and re-created.
        claimed = {find(n_new + rr) for rr in range(n_reg)}
        for i, (_, q) in enumerate(new):
            if find(i) in claimed:
                continue
            for rr, region in enumerate(self.regions):
                if key in region.local and any(self._touch(q, p) for _, p in region.local[key]):
                    union(i, n_new + rr)

        comps = {}
        for i in range(n_new):
            comps.setdefault(find(i), []).append(i)

        routes, handled, remove, add = [], set(), [], []
        for root, ring_ids in comps.items():
            regs = [rr for rr in range(n_reg) if find(n_new + rr) == root]
            rings = [rings_world[i] for i in ring_ids]
            if len(regs) == 1:  # case 1: local update (applied to the session at the end of the batch)
                region = self.regions[regs[0]]
                region.slices[key] = list(rings)
                region.local[key] = [new[i] for i in ring_ids]
                region.pending[key] = list(rings)
                routes.append(region)
                handled.add(regs[0])

            elif not regs:  # case 2: new region
                add.append({key: list(rings)})
                routes.append("new")

            else:  # case 3: merge
                merged = {}
                for rr in regs:
                    for k, v in self.regions[rr].slices.items():
                        if k != key:
                            merged.setdefault(k, []).extend(v)
                    handled.add(rr)
                merged[key] = list(rings)
                add.append(merged)
                routes.append(f"merge({len(regs)})")
                remove.extend(regs)

        for rr, region in enumerate(self.regions):  # case 4: region lost its rings on key
            if rr not in handled and key in region.slices:
                region.slices.pop(key)
                region.local.pop(key, None)
                if region.slices:
                    region.pending[key] = []
                    routes.append("emptied-slice")
                else:
                    remove.append(rr)
                    routes.append("drop")

        remove = set(remove)
        self.regions = [r for i, r in enumerate(self.regions) if i not in remove]
        for group in add:
            self._new_region(group)

        return routes

    def _touch(self, a, b):
        """True if two local 2D rings overlap or come within `pad` of each other."""
        if _inside(a[0], b) or _inside(b[0], a):
            return True

        da, db = _dense(a, self.step), _dense(b, self.step)
        d, _ = cKDTree(db).query(da, distance_upper_bound=self.pad + self.step)
        return bool(np.isfinite(d).any())

    def contours(self):
        """
        All world rings currently in the session, region by region.

        Returns
        -------
        list[np.ndarray]
            World rings (n, 3).

        Examples
        --------
        >>> rings = session.contours()
        """
        return [r for region in self.regions for rings in region.slices.values() for r in rings]

    def finalize(self, reduction=0.5, **kw):
        """
        From-scratch mesh of the current contours: regions and h recomputed.

        Same result as `contours_to_mesh` with interpolate=False when the
        frame and slice spacing come from the image.

        Parameters
        ----------
        reduction : float
            Decimation fraction, 0 to 1.

        **kw
            `contours_to_mesh_regions` options; override the session's.

        Returns
        -------
        pv.PolyData | None
            None if the session has no contours.

        Examples
        --------
        >>> roi.mesh = session.finalize(reduction=0)
        """
        slices = {}
        for region in self.regions:
            for k, v in region.slices.items():
                slices.setdefault(k, []).extend(v)

        if not slices:
            return None

        return contours_to_mesh_regions(slices, self.frame, self.dz, reduction=reduction, pad=self.pad,
                                        gap_factor=self.gap, **{**self.kw, **kw})

    def mesh(self, reduction=0.5, as_polydata=True):
        """
        Stack the regions' kept meshes and decimate once.

        Parameters
        ----------
        reduction : float
            Decimation fraction, 0 to 1. 0 returns the raw welded meshes
            (fastest, and the most accurate to slice while editing).

        as_polydata : bool
            True returns pv.PolyData, False returns (points, triangles).

        Returns
        -------
        pv.PolyData | tuple[np.ndarray, np.ndarray] | None
            None when the session has no contours (e.g. everything erased).

        Examples
        --------
        >>> roi.mesh = session.mesh(reduction=0)
        >>> points, triangles = session.mesh(reduction=0.5, as_polydata=False)
        """
        Ps, Ts, n = [], [], 0
        for region in self.regions:
            P, T = region.session.raw_mesh()
            if len(P) == 0:
                continue
            Ps.append(P)
            Ts.append(T + n)
            n += len(P)

        if not Ps:
            return None

        P, T = np.vstack(Ps), np.vstack(Ts)
        if reduction > 0:
            P, T = postprocess(P, T, reduction, 0)

        return _to_polydata(P, T) if as_polydata else (P, T)

    def update_slice(self, key, rings_world):
        """
        Replace all contours of one slice and route the edit to the affected regions (see `update_slices`).

        Parameters
        ----------
        key : int
            Slice index.

        rings_world : list[np.ndarray]
            All world rings (n, 3) now on that slice, from every region. An
            empty list removes the slice.

        Returns
        -------
        None

        Examples
        --------
        Edit a slice inside an existing region (local update):

        >>> session.update_slice(2, [np.c_[1.5 * square, np.full(4, 4.0)]])
        >>> session.last_route
        'local(local)'

        Draw a separate island far from it (new region):

        >>> session.update_slice(2, [np.c_[1.5 * square, np.full(4, 4.0)],
        ...                          np.c_[square + 100, np.full(4, 4.0)]])
        >>> session.last_route
        'local(local), new'
        """
        self.update_slices([key], [rings_world])

    def update_slices(self, keys, rings_per_key):
        """
        Replace all contours of several slices and route the edits to the affected regions.

        Each slice is routed like a single edit:

        * touches one region -> recorded for that region
        * touches no region -> a new region
        * touches several regions -> they are merged into one new region
        * a region lost its rings on a slice -> recorded for that region, or
          the region is dropped when it has no slices left

        Every region with recorded changes then gets one
        `EditSession.update_slices` call, so erasing or adding several slices
        costs one local update per region, not one per slice.

        Parameters
        ----------
        keys : list[int]
            Slice indices.

        rings_per_key : list[list[np.ndarray]]
            For each key, all world rings (n, 3) now on that slice, from every
            region. An empty list removes the slice.

        Returns
        -------
        None

        Examples
        --------
        Erase the three bottom slices of an ROI in one update:

        >>> session.update_slices([10, 11, 12], [[], [], []])
        >>> session.last_route
        'emptied-slice, emptied-slice, emptied-slice'
        """
        routes = []
        for key, rings in zip(keys, rings_per_key):
            routes.extend(self._route(key, rings))

        for region in self.regions:
            if region.pending:
                pending_keys = list(region.pending)
                region.session.update_slices(pending_keys, [region.pending[k] for k in pending_keys])
                region.pending = {}

        names = [f"local({r.session.last})" if isinstance(r, _Region) else r for r in routes]
        self.last_route = ", ".join(names) if names else "no-op"
