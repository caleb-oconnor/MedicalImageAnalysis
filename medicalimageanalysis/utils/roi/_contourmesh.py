"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Contour-to-mesh reconstruction using a RayStation-style shape-based interpolation field.

This module turns sparse stacks of closed contours into a watertight surface mesh. The
core idea is:

1. Work in a contour-aligned local coordinate frame so every contour slice becomes a 2D
   problem plus a scalar slice position.
2. For each slice, build an exact signed distance field to the contour polylines.
3. For each pair of neighbouring slices, classify which contour regions correspond across
   slices (overlap) and which appear/disappear (isolated components).
4. Interpolate overlapping components linearly, but handle appearing/disappearing pieces
   with a "hat" field so they close smoothly instead of collapsing unnaturally.
5. Store all real slice planes and interpolation sub-planes in one 3D scalar volume.
6. Extract the zero level-set once with ``vtkFlyingEdges3D`` to obtain the mesh.

Why this approach exists
------------------------
Naive contour lofting often fails when topology changes between slices: one loop can split
into two, holes can appear/disappear, and disconnected components may start or end. Purely
linear interpolation between signed distance fields also produces undesirable bridges or
shrinking artefacts in these cases. The overlap/isolated split used here reproduces the
behaviour of commercial shape-based interpolation more faithfully.

Two main usage modes
--------------------
``contours_to_mesh(...)``
    One-shot pipeline. Build the field, extract a mesh, optionally post-process it, and
    discard the temporary state.

``EditSession(...)``
    Incremental pipeline for interactive editing. The field volume and raw mesh stay in
    memory so that editing one slice only recomputes a small local slab and splices the
    result back into the kept mesh.

Field definition
----------------
For each contour slice we compute an exact 2D signed distance to the contour polylines
(negative inside, positive outside, even-odd fill to support holes/nesting).

For linked neighbouring slices ``k`` and ``k+1`` at interpolation fraction ``t``:

    F = min((1-t) * dA_k + t * dB_k+1, hat_I(t), hat_J(t))

where:

``dA_k``
    Distance field restricted to components on slice ``k`` that overlap slice ``k+1``.

``dB_k+1``
    Distance field restricted to components on slice ``k+1`` that overlap slice ``k``.

``hat_I(t)``, ``hat_J(t)``
    Cap-like fields used for components that do *not* overlap. These let components start
    or end smoothly and reach exactly half-way between slices.

Unlinked gaps and the first/last slice use the same cap logic to close the surface.

Performance strategy
--------------------
The implementation is tuned to avoid doing expensive work everywhere:

* Scanline filling and connected-component labelling are vectorized across slices.
* Exact point-to-polyline distances are only evaluated in narrow bands where the zero
  isosurface could actually pass.
* A single KD-tree query can serve many slices by separating slices in an artificial z
  dimension (``ZSEP``).
* The final meshing step is one ``vtkFlyingEdges3D`` call on one volume, which is much
  faster than meshing slabs independently and stitching them afterwards.
* Geometry is computed in a local frame and only rotated back to world coordinates at the
  very end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtk
import fast_simplification
from scipy import ndimage
from scipy.spatial import cKDTree
from vtk.util import numpy_support as nps

BIG = np.float32(1e4)
ZSEP = 1e6                                   # slice separation inside one KD-tree
_S2D = np.zeros((3, 3, 3), bool)
_S2D[1] = True                               # 8-connected in-plane, no z connectivity


# =============================================================================== frame
@dataclass
class Frame:
    """Local orthonormal-ish coordinate frame used by the whole pipeline.

    ``u`` and ``v`` span the contour drawing plane, while ``n`` is the direction along
    which slices are stacked. Contours are converted into this frame immediately so the
    algorithm can treat each slice as a 2D problem. That greatly simplifies filling,
    distance calculations, overlap classification, and interpolation.
    """
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray

    @classmethod
    def standard(cls, plane: str, origin=(0.0, 0.0, 0.0)):
        """Construct a standard axial/coronal/sagittal frame.

        This is a convenience helper for the common case where contours already align with
        a standard anatomical plane.
        """
        e = np.eye(3)
        u, v, n = {"axial": (e[0], e[1], e[2]),
                   "coronal": (e[0], e[2], e[1]),
                   "sagittal": (e[1], e[2], e[0])}[plane]
        return cls(np.asarray(origin, float), u, v, n)

    @classmethod
    def from_direction(cls, origin, direction):
        """direction: 3x3, columns = image i, j, k axes (contours drawn in the i-j plane)."""
        d = np.asarray(direction, float)
        return cls(np.asarray(origin, float), d[:, 0], d[:, 1], d[:, 2])

    def to_local(self, p):
        """World -> local coordinates.

        The result has columns ``(u, v, n)``. Converting to local coordinates lets the
        rest of the code operate in a slice-aligned coordinate system independent of the
        original image orientation.
        """
        q = np.asarray(p, float) - self.origin
        return np.stack([q @ self.u, q @ self.v, q @ self.n], axis=-1)

    def to_world(self, q):
        """Local -> world coordinates for a stack of points ``q`` with columns ``u,v,n``."""
        return self.origin + q[:, :1] * self.u + q[:, 1:2] * self.v + q[:, 2:3] * self.n

    @property
    def flips(self) -> bool:
        """Whether the basis is left-handed.

        If true, triangle winding must be reversed after meshing so normals/orientation are
        consistent in world space.
        """
        return float(np.dot(np.cross(self.u, self.v), self.n)) < 0


# =============================================================================== helpers
def _links(ring):
    """Return previous/next vertex indices for many closed rings stored contiguously.

    ``ring`` assigns each point to a ring id. The returned arrays let the code walk along
    each ring without ever building Python lists of edge tuples, which keeps later
    resampling and segment-distance code vectorized.
    """
    n = len(ring)
    start = np.flatnonzero(np.r_[True, ring[1:] != ring[:-1]])
    end = np.r_[start[1:], n] - 1
    nxt = np.arange(1, n + 1)
    nxt[end] = start
    prv = np.arange(-1, n - 1)
    prv[start] = end
    return prv, nxt


def _dil(x):
    """4-connected in-plane dilation of a (..., ny, nx) bool stack. 4-connectivity is
    enough: Flying Edges only interpolates along axis-aligned lattice edges."""
    out = x.copy()
    out[..., 1:, :] |= x[..., :-1, :]
    out[..., :-1, :] |= x[..., 1:, :]
    out[..., :, 1:] |= x[..., :, :-1]
    out[..., :, :-1] |= x[..., :, 1:]
    return out


def _shell(x):
    """One-pixel shell around a boolean mask.

    This is used to restrict exact distance evaluations to where the zero crossing might
    matter. Far from the contour we can safely keep the field at +/- BIG.
    """
    return _dil(x) & _dil(~x)


def _seg_dist(q, a, b):
    """Point-to-segment distance for many query points/segments in parallel."""
    ab, aq = b - a, q - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", aq, ab) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    d = aq - t[:, None] * ab
    return np.sqrt(np.einsum("ij,ij->i", d, d))


# =============================================================================== state
class _Volume:
    """Internal state for the scalar field volume and all metadata derived from contours.

    This class is the engine of the reconstruction. It owns:

    * slice masks and connected-component labels,
    * overlap classification between neighbouring slices,
    * exact/narrow-band signed distance fields,
    * the layout of real slice planes and interpolated sub-planes,
    * the dense 3D scalar volume consumed by Flying Edges.

    ``EditSession`` keeps an instance of this class alive specifically so only the parts
    affected by a local contour edit need to be recomputed.
    """

    def __init__(self, slices_local, s, delta, h, m, c, margin, pad, gap_factor, workers,
                 mid=True, flat_caps=False, cap_radius=None):
        # ``slices_local`` is already sorted by slice position ``s`` and each slice holds a
        # list of 2D rings in the local frame. From here onward everything happens in local
        # coordinates on a regular x/y lattice with spacing ``h``.
        self.h, self.m, self.c, self.delta = float(h), int(m), float(c), float(delta)
        self.workers = workers
        self.s = np.asarray(s, float)
        S = self.S = len(slices_local)
        # ``link[k]`` says whether slices k and k+1 are close enough to interpolate through.
        # Larger gaps are treated as separate capped objects instead of bridged slabs.
        self.link = np.diff(self.s) <= gap_factor * self.delta          # (S-1,)
        allp = np.vstack([r for rs in slices_local for r in rs])
        pm = margin + pad
        self.i0 = int(np.floor(allp[:, 0].min() / h)) - pm
        self.j0 = int(np.floor(allp[:, 1].min() / h)) - pm
        self.nx = int(np.ceil(allp[:, 0].max() / h)) + pm - self.i0 + 1
        self.ny = int(np.ceil(allp[:, 1].max() / h)) + pm - self.j0 + 1
        self.margin = margin
        shp = (S, self.ny, self.nx)
        self.M = np.zeros(shp, bool)        # slice interior masks from even-odd fill
        self.L = np.zeros(shp, np.int32)    # connected-component labels per slice
        self.Aup = np.zeros(shp, bool)      # components on k overlapping slice k+1
        self.Bdn = np.zeros(shp, bool)      # components on k overlapping slice k-1
        self.nearI = np.zeros(shp, bool)    # narrow band for isolated-up components
        self.nearJ = np.zeros(shp, bool)    # narrow band for isolated-down components
        self.dM = np.full(shp, BIG, np.float32)  # per-slice full signed distance field
        self.sub = {}          # (kind, k) -> subset field, only when overlap/isolated mix
        self.next_label = 1
        self.mid = mid
        self.flat_caps = flat_caps
        self.cap_R = None if cap_radius is None else float(cap_radius)

        # Resampled contour points used for exact point-to-polyline distance queries.
        self.pts = np.zeros((0, 2))
        self.psl = np.zeros(0, np.int64)    # owning slice index per resampled point
        self.plab = np.zeros(0, np.int32)   # owning component label per point
        self.prel = np.zeros(0, np.int64)   # relative index to previous point on ring
        self.nrel = np.zeros(0, np.int64)   # relative index to next point on ring
        ks = np.arange(S)
        self.set_contours(ks, slices_local)
        self.classify(np.arange(S - 1))
        self.fields(ks)
        self._layout()
        self.V = np.empty((len(self.pz),) + shp[1:], np.float32)
        self.planes(0, len(self.pz) - 1)

    # --------------------------------------------------------------- plane layout
    def _layout(self):
        """Define the stack of scalar-field planes used by the 3D meshing step.

        The reconstructed field is piecewise linear between real contour slices. Instead of
        meshing each slab independently, we place all real slices, interpolated sub-planes,
        and cap planes into one global plane index space. Later, ``to_local_xyz`` maps the
        integer plane index back to the true physical z coordinate exactly.
        """
        m, S, s, d = self.m, self.S, self.s, self.delta
        kind, kk, tt, zz = [], [], [], []

        def add(kd, k, t, z):
            kind.append(kd); kk.append(k); tt.append(t); zz.append(z)

        # Optimization: if a slab is purely linear (no hat terms) and m == 1, Flying Edges
        # already reproduces the exact zero crossing between the two real slice planes. In
        # that case we can omit the middle sub-plane entirely.
        if self.mid == "auto" and m == 1:
            has_hat = self.nearI[:-1].any(axis=(1, 2)) | self.nearJ[1:].any(axis=(1, 2))
        else:
            has_hat = np.ones(max(S - 1, 0), bool)

        flat = self.flat_caps
        CAP_EPS = 0.05                     # BIG plane just beyond a flat cap (fraction of d)

        def cap_up(k):                     # top cap of slice k
            if flat:
                # Flat cap = keep the end-slice field half a slice beyond the contour, then
                # jump to an all-BIG plane. The zero set closes as a planar lid.
                add(2, k, 0.0, s[k] + 0.5 * d)             # plain SDF of the end contour
                add(4, k, 0.0, s[k] + (0.5 + CAP_EPS) * d) # all BIG -> closes the lid
            else:
                # Rounded cap = evaluate the disappearing-component hat on intermediate planes.
                for j in range(1, m + 1):
                    add(2, k, j / (2 * m), s[k] + j / (2 * m) * d)

        def cap_dn(k):                     # bottom cap of slice k
            if flat:
                add(4, k, 0.0, s[k] - (0.5 + CAP_EPS) * d)
                add(3, k, 0.0, s[k] - 0.5 * d)
            else:
                for j in range(m, 0, -1):
                    add(3, k, j / (2 * m), s[k] - j / (2 * m) * d)

        cap_dn(0)                                                   # bottom cap of 0
        self.splane = np.zeros(S, np.int64)
        for k in range(S):
            self.splane[k] = len(kind)
            add(0, k, 0.0, s[k])
            if k == S - 1:
                break
            if self.link[k]:
                if not has_hat[k]:
                    continue
                for j in range(1, 2 * m):
                    t = j / (2 * m)
                    add(1, k, t, (1 - t) * s[k] + t * s[k + 1])
            else:
                cap_up(k)
                cap_dn(k + 1)
        cap_up(S - 1)                                               # top cap of S-1

        self.pkind = np.array(kind)
        self.pk = np.array(kk)
        self.pt = np.array(tt)
        self.pz = np.array(zz)

    # --------------------------------------------------------------- step 1-2: rings, fill, labels
    def set_contours(self, ks, rings_per_slice):
        """Rasterize contours for selected slices and rebuild their local geometric state.

        For each target slice this method:

        1. vectorizes all ring edges,
        2. fills interiors with an even-odd scanline rule,
        3. labels connected components within the slice,
        4. assigns each ring to its filled component,
        5. resamples ring polylines densely enough for exact distance queries.

        The expensive parts are batched across the requested slices to minimize Python-level
        overhead.
        """
        h, i0, j0, ny, nx = self.h, self.i0, self.j0, self.ny, self.nx
        nb = len(ks)
        pts, ring, bix = [], [], []
        rid = 0
        for b, rings in enumerate(rings_per_slice):
            for r in rings:
                pts.append(r)
                ring.append(np.full(len(r), rid))
                bix.append(np.full(len(r), b))
                rid += 1
        P = np.concatenate(pts)
        ring = np.concatenate(ring)
        bix = np.concatenate(bix)
        _, nxt = _links(ring)

        # Vectorized scanline filling:
        # each edge contributes crossings for the rows it spans, then a cumulative XOR over
        # columns converts crossing parity into an inside/outside mask.
        a = P / h - (i0, j0)
        bq = a[nxt]
        ya, yb = a[:, 1], bq[:, 1]
        r0 = np.ceil(np.minimum(ya, yb)).astype(np.int64)
        n = np.ceil(np.maximum(ya, yb)).astype(np.int64) - r0
        keep = n > 0
        nk = n[keep]
        e = np.repeat(np.flatnonzero(keep), nk)
        rows = np.repeat(r0[keep], nk) + (np.arange(e.size) - np.repeat(np.cumsum(nk) - nk, nk))
        t = (rows - ya[e]) / (yb[e] - ya[e])
        x = a[e, 0] + t * (bq[e, 0] - a[e, 0])
        col = np.clip(np.floor(x).astype(np.int64) + 1, 0, nx)
        lin = (bix[e] * ny + rows) * (nx + 1) + col
        par = (np.bincount(lin, minlength=nb * ny * (nx + 1)) & 1).astype(np.uint8)
        Mb = np.bitwise_xor.accumulate(par.reshape(nb, ny, nx + 1), axis=2)[:, :, :nx].astype(bool)
        Lb, nlab = ndimage.label(Mb, structure=_S2D)

        tvec = bq - a
        Ln = np.hypot(tvec[:, 0], tvec[:, 1])
        good = Ln > 1e-9

        def ring_labels(ok):
            # Associate each ring with the connected component that lies on one side of its
            # edges. We probe offset points near the edges and vote for the most common label.
            mid = 0.5 * (a + bq)[ok]
            nrm = np.stack([-tvec[ok, 1], tvec[ok, 0]], 1) / Ln[ok, None]
            pr = np.concatenate([mid + 0.75 * nrm, mid - 0.75 * nrm])
            pr_ring = np.tile(ring[ok], 2)
            pr_b = np.tile(bix[ok], 2)
            ci = np.clip(np.rint(pr[:, 0]).astype(np.int64), 0, nx - 1)
            cj = np.clip(np.rint(pr[:, 1]).astype(np.int64), 0, ny - 1)
            lab = Lb[pr_b, cj, ci]
            sel = lab > 0
            key = pr_ring[sel].astype(np.int64) * (nlab + 1) + lab[sel]
            out = np.zeros(rid, np.int64)
            if key.size == 0:
                return out

            uk, cnt = np.unique(key, return_counts=True)
            rr, ll = uk // (nlab + 1), uk % (nlab + 1)
            o = np.lexsort((cnt, rr))
            last = np.r_[rr[o][1:] != rr[o][:-1], True]
            out[rr[o][last]] = ll[o][last]
            return out

        # Probe only a subset of edges first for speed; rings that remain unresolved fall back
        # to using all non-degenerate edges.
        gi = np.flatnonzero(good)
        gr = ring[gi]
        glen = np.bincount(gr, minlength=rid)
        gpos = np.arange(gi.size) - np.repeat(np.cumsum(glen) - glen, glen)
        ok = np.zeros(len(ring), bool)
        ok[gi[gpos % np.maximum(1, glen // 32)[gr] == 0]] = True
        ring_lab = ring_labels(ok)
        miss = ring_lab == 0
        if miss.any():
            ring_lab[miss] = ring_labels(good & miss[ring])[miss]

        # Resample each ring so segment length is <= h. This gives enough point density that
        # the nearest resampled point plus its adjacent segments can recover the exact local
        # point-to-polyline distance cheaply.
        k = np.maximum(1, np.ceil(np.hypot(*(P[nxt] - P).T) / h).astype(np.int64))
        if (k == 1).all():
            e = np.arange(len(P)); Q = P; qring = ring
            qprv, qnxt = _links(ring)
        else:
            e = np.repeat(np.arange(len(P)), k)
            frac = (np.arange(e.size) - np.repeat(np.cumsum(k) - k, k)) / k[e]
            Q = P[e] + (P[nxt][e] - P[e]) * frac[:, None]
            qring = ring[e]
            qprv, qnxt = _links(qring)

        qlab = ring_lab[qring]
        qlab = np.where(qlab > 0, qlab + self.next_label - 1, 0).astype(np.int32)

        ks = np.asarray(ks)
        self.M[ks] = Mb
        if self.next_label == 1:
            self.L[ks] = Lb
        else:
            self.L[ks] = np.where(Lb > 0, Lb + self.next_label - 1, 0)

        self.next_label += nlab
        keep = ~np.isin(self.psl, ks)
        idx = np.arange(len(Q))
        self.pts = np.concatenate([self.pts[keep], Q])
        self.psl = np.concatenate([self.psl[keep], ks[bix[e]]])
        self.plab = np.concatenate([self.plab[keep], qlab])
        self.prel = np.concatenate([self.prel[keep], qprv - idx])
        self.nrel = np.concatenate([self.nrel[keep], qnxt - idx])

    # --------------------------------------------------------------- overlap classification
    def classify(self, slabs):
        """Classify overlapping components between linked neighbouring slices.

        ``Aup[k]`` marks components on slice ``k`` that overlap slice ``k+1``.
        ``Bdn[k+1]`` marks components on slice ``k+1`` that overlap slice ``k``.

        Components present on one slice but not overlapping the next are later handled by the
        hat fields so they can appear/disappear cleanly.
        """
        slabs = np.asarray(slabs, np.int64)
        slabs = slabs[(slabs >= 0) & (slabs < self.S - 1)]
        self.Aup[slabs] = False
        self.Bdn[slabs + 1] = False
        lk = slabs[self.link[slabs]]
        if lk.size:
            lut = np.zeros(self.next_label, bool)
            lut[self.L[lk][self.M[lk + 1]]] = True
            lut[0] = False
            self.Aup[lk] = lut[self.L[lk]]
            lut[:] = False
            lut[self.L[lk + 1][self.M[lk]]] = True
            lut[0] = False
            self.Bdn[lk + 1] = lut[self.L[lk + 1]]

    # --------------------------------------------------------------- exact distances
    def _query(self, sel_pts, mask, inside):
        """Signed distance at pixels `mask` (S', ny, nx over slices `ks`) to points sel_pts."""
        ks, m = mask
        kk, jj, ii = np.nonzero(m)
        if kk.size == 0:
            return kk, jj, ii, np.zeros(0, np.float32)
        if sel_pts.size == 0:
            return kk, jj, ii, np.where(inside[kk, jj, ii], -BIG, BIG).astype(np.float32)
        # Embed all queried slices into a shared 3D KD-tree by separating them massively in
        # z. This lets one tree handle many 2D slices without accidental cross-slice matches.
        P3 = np.c_[self.pts[sel_pts], self.psl[sel_pts] * ZSEP]
        q = np.c_[(ii + self.i0) * self.h, (jj + self.j0) * self.h]
        dd, nn = cKDTree(P3, balanced_tree=False, compact_nodes=False).query(
            np.c_[q, ks[kk] * ZSEP], workers=self.workers)
        j = sel_pts[nn]
        pa = self.pts[j + self.prel[j]]
        pb = self.pts[j + self.nrel[j]]
        # The nearest resampled point tells us which two ring segments might be closest; the
        # exact distance is the minimum to the previous and next segment around that point.
        d = np.minimum(_seg_dist(q, pa, self.pts[j]), _seg_dist(q, self.pts[j], pb))
        d = np.where(dd > 0.5 * ZSEP, BIG, d)          # nearest point was on another slice
        d = np.where(inside[kk, jj, ii], -d, d).astype(np.float32)
        return kk, jj, ii, d

    def fields(self, ks):
        """Build per-slice signed distance fields and subset fields for requested slices.

        The full field ``dM`` is the signed distance to the whole slice contour. Additional
        subset fields are only created for slices where overlapping and isolated components
        coexist, because only then do we need separate fields for the linear part versus the
        hat part of the interpolation.
        """
        ks = np.unique(np.asarray(ks, np.int64))
        S = self.S
        M, Aup, Bdn = self.M[ks], self.Aup[ks], self.Bdn[ks]
        Iup, Jdn = M & ~Aup, M & ~Bdn
        nearI, nearJ = _dil(Iup), _dil(Jdn)
        self.nearI[ks], self.nearJ[ks] = nearI, nearJ
        needA = np.zeros_like(M)
        needB = np.zeros_like(M)
        slabs = np.union1d(ks, ks - 1)
        slabs = slabs[(slabs >= 0) & (slabs < S - 1)]
        slabs = slabs[self.link[slabs]]
        if slabs.size:
            A, B = self.Aup[slabs], self.Bdn[slabs + 1]
            band = _dil(A ^ B) | _shell(A) | _shell(B)
            pos = {k: n for n, k in enumerate(ks)}
            for b, k in enumerate(slabs):
                if k in pos:
                    needA[pos[k]] = band[b]
                if k + 1 in pos:
                    needB[pos[k + 1]] = band[b]
        # Only pixels in these bands can affect the zero isosurface. Everywhere else the
        # field can safely stay at +/- BIG and does not need an exact distance query.
        need = needA | needB | nearI | nearJ

        in_ks = np.isin(self.psl, ks)
        allp = np.flatnonzero(in_ks & (self.plab > 0))
        dM = np.where(M, -BIG, BIG).astype(np.float32)
        kk, jj, ii, d = self._query(allp, (ks, need), M)
        dM[kk, jj, ii] = d
        self.dM[ks] = dM

        # Subset fields are only needed for mixed slices. Purely overlapping or purely
        # isolated slices can reuse the full-field data directly.
        for k in ks:
            for kind in ("A", "I", "B", "J"):
                self.sub.pop((kind, k), None)
        # batched: one KD query per field kind over all "mixed" slices at once
        # (slices holding both overlapping and isolated components, e.g. small fragments)
        flat = lambda x: x.any(axis=(1, 2))
        for up in (True, False):
            over = Aup if up else Bdn
            iso = Iup if up else Jdn
            mixed = np.flatnonzero(flat(over) & flat(iso))
            if mixed.size == 0:
                continue
            km = ks[mixed]
            lut = np.zeros(self.next_label, bool)
            lut[self.L[km][over[mixed]]] = True
            lut[0] = False
            ptk = np.flatnonzero(in_ks & (self.plab > 0))
            ptk = ptk[np.isin(self.psl[ptk], km)]
            is_over = lut[self.plab[ptk]]
            for name, subset, region, band in (
                    ("A" if up else "B", ptk[is_over], over[mixed],
                     (needA if up else needB)[mixed]),
                    ("I" if up else "J", ptk[~is_over], iso[mixed],
                     (nearI if up else nearJ)[mixed])):
                f = np.where(region, -BIG, BIG).astype(np.float32)
                kk, jj, ii, d = self._query(subset, (km, band), region)
                f[kk, jj, ii] = d
                for n, k in enumerate(km):
                    self.sub[(name, k)] = f[n]

    def _get(self, name, ks):
        """Return the field stack needed for a given interpolation term.

        ``A`` / ``B`` are the overlapping-component fields used in linear interpolation.
        ``I`` / ``J`` are the isolated-component fields used in the hat terms.
        """
        out = self.dM[ks].copy()
        if name in ("A", "B"):
            region = self.Aup if name == "A" else self.Bdn
            empty = ~region[ks].any(axis=(1, 2))
            out[empty] = BIG
        for n, k in enumerate(ks):
            f = self.sub.get((name, k))
            if f is not None:
                out[n] = f
        return out

    # --------------------------------------------------------------- volume planes
    def planes(self, p0, p1):
        """Evaluate the scalar field on plane indices ``p0..p1``.

        Each plane kind corresponds to a different formula:

        * kind 0: real contour slice, use its full signed distance field.
        * kind 1: linked slab interpolation between two slices.
        * kind 2/3: top/bottom caps or isolated-component hats.
        * kind 4: all-BIG plane used to close flat lids.
        """
        c = self.c
        idx = np.arange(p0, p1 + 1)
        for kind in (0, 1, 2, 3, 4):
            ik = idx[self.pkind[idx] == kind]
            for t in np.unique(self.pt[ik]):
                ip = ik[self.pt[ik] == t]
                ks = self.pk[ip]
                if kind == 0:
                    f = self.dM[ks]
                elif kind == 1:
                    # Main interpolation term for linked slices. Overlapping regions are
                    # linearly interpolated; isolated regions are clipped by hat fields.
                    f = (1.0 - t) * self._get("A", ks) + t * self._get("B", ks + 1)
                    for use, hk, near, name, a in ((t <= 0.5, ks, self.nearI, "I", 1 - 2 * t),
                                                   (t >= 0.5, ks + 1, self.nearJ, "J", 2 * t - 1)):
                        rows = np.flatnonzero(near[hk].any(axis=(1, 2))) if use else []
                        if len(rows):
                            hr = hk[rows]
                            G = self._get(name, hr)
                            hat = self._cap(G) if (self.cap_R and t == 0.5) else a * G + (1 - a) * c
                            f[rows] = np.minimum(f[rows], np.where(near[hr], hat, BIG))

                elif kind in (2, 3):
                    # Cap planes use only one side's isolated-component field because there
                    # is no linked partner slice to interpolate with.
                    near, name = (self.nearI, "I") if kind == 2 else (self.nearJ, "J")
                    G = self._get(name, ks)
                    hat = self._cap(G) if (self.cap_R and t == 0.5) else (1 - 2 * t) * G + 2 * t * c
                    f = np.where(near[ks], hat, BIG)
                else:  # kind 4: lid-closing plane
                    f = np.full((len(ks), self.ny, self.nx), BIG, np.float32)
                self.V[ip] = f
        blk = self.V[p0:p1 + 1]
        # Avoid exact zeros on lattice points: that can create ambiguous topology in some
        # contouring algorithms.
        blk[blk == 0] = np.float32(1e-6)

    # --------------------------------------------------------------- meshing
    def extract(self, p0, p1):
        """Run ``vtkFlyingEdges3D`` on a subset of the field volume.

        The returned z coordinate is a *plane index*, not yet the true physical slice
        position. ``to_local_xyz`` performs the exact plane-index -> z mapping afterwards.
        """
        img = vtk.vtkImageData()
        img.SetExtent(self.i0, self.i0 + self.nx - 1, self.j0, self.j0 + self.ny - 1, p0, p1)
        img.SetSpacing(self.h, self.h, 1.0)
        img.SetOrigin(0.0, 0.0, 0.0)
        blk = np.ascontiguousarray(self.V[p0:p1 + 1])
        img.GetPointData().SetScalars(nps.numpy_to_vtk(blk.ravel(), deep=False,
                                                       array_type=vtk.VTK_FLOAT))
        fe = vtk.vtkFlyingEdges3D()
        fe.SetInputData(img)
        fe.SetValue(0, 0.0)
        fe.ComputeNormalsOff(); fe.ComputeGradientsOff(); fe.ComputeScalarsOff()
        fe.Update()
        out = fe.GetOutput()
        if out.GetNumberOfPoints() == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
        P = nps.vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
        T = nps.vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
        return P, T

    def to_local_xyz(self, P):
        """Convert Flying Edges plane indices into true local-frame z coordinates."""
        Q = P.copy()
        Q[:, 2] = np.interp(P[:, 2], np.arange(len(self.pz)), self.pz)
        return Q

    def _cap(self, F):
        """Rounded half-way cap field.

        ``F`` is a signed distance field on a slice. For points inside the contour, this
        converts depth-inside-contour into a cap value whose zero isosurface follows a
        quarter-circle profile of radius ``R`` near the boundary, then flattens deeper in.
        This produces less pinched caps than a purely linear hat.
        """
        R = self.cap_R
        u = np.maximum(-F, 0.0)
        p = np.sqrt(np.clip(1.0 - (1.0 - np.minimum(u, R) / R) ** 2, 1e-6, 1.0))
        g = u * (1.0 / p - 1.0) + 1e-4
        return np.where(F > 0, F + 1e-4, g).astype(np.float32)


# =============================================================================== post-processing
def postprocess(P, T, reduction=0.5, smooth_iters=0, passband=0.1, method="auto"):
    """Optional mesh simplification and smoothing stage.

    Why this exists:
    the raw Flying Edges mesh is faithful to the field but often denser than needed for
    storage or downstream interactive rendering. The decimation step reduces triangle count,
    and optional windowed-sinc smoothing can soften voxel-grid artefacts while preserving
    overall shape.
    """
    if method == "auto":
        try: # noqa: F401
            method = "fast"
        except ImportError:
            method = "cluster"
    if reduction > 0 and method == "fast":
        P, T = fast_simplification.simplify(P.astype(np.float32), T.astype(np.int32),
                                            target_reduction=reduction)
        pd = _to_polydata(P, T)
    else:
        pd = _to_polydata(P, T)
        if reduction > 0 and method in ("cluster", "vtk"):
            if method == "cluster":
                f = vtk.vtkQuadricClustering()
                b = np.ptp(P, axis=0)
                area = 0.5 * np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]],
                                                     P[T[:, 2]] - P[T[:, 0]]), axis=1).sum()
                cell = np.sqrt(area / max((1 - reduction) * len(P), 1))
                f.SetNumberOfDivisions(*[max(2, int(x / cell)) for x in b])
                f.AutoAdjustNumberOfDivisionsOff()
            else:
                f = vtk.vtkQuadricDecimation()
                f.SetTargetReduction(reduction)
                f.VolumePreservationOn()
            f.SetInputData(pd); f.Update(); pd = f.GetOutput()
    if smooth_iters > 0:
        f = vtk.vtkWindowedSincPolyDataFilter()
        f.SetInputData(pd)
        f.SetNumberOfIterations(smooth_iters)
        f.SetPassBand(passband)
        f.NormalizeCoordinatesOn()
        f.BoundarySmoothingOff(); f.FeatureEdgeSmoothingOff(); f.NonManifoldSmoothingOn()
        f.Update()
        pd = f.GetOutput()
    return _from_polydata(pd)


def _to_polydata(P, T):
    """Convert NumPy point/triangle arrays into ``vtkPolyData``."""
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(np.ascontiguousarray(P, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * len(T) + 1, 3, dtype=np.int64)
    cells.SetData(nps.numpy_to_vtkIdTypeArray(offsets, deep=True),
                  nps.numpy_to_vtkIdTypeArray(np.ascontiguousarray(T, dtype=np.int64).ravel(), deep=True))
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def _from_polydata(pd):
    """Convert ``vtkPolyData`` back into NumPy point/triangle arrays."""
    P = nps.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    T = nps.vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    return P, T


# =============================================================================== public API
class AdaptiveContourToMesh:
    """User-facing builder for the contour-to-mesh workflow.

    This class mostly packages configuration and exposes the pipeline in smaller logical
    pieces: grouping/normalization, volume construction, raw meshing, post-processing, and
    region-wise meshing.
    """

    def __init__(self, frame, slice_spacing, m=1, h=None, c=None, gap_factor=1.5,
                 post=True, reduction=0.5, smooth_iters=0, method="auto",
                 workers=-1, as_polydata=True, mid='auto', flat_caps=False,
                 cap_radius=None):
        self.frame = frame
        self.slice_spacing = float(slice_spacing)
        self.m = int(m)
        self.h = h
        self.c = c
        self.gap_factor = float(gap_factor)
        self.post = bool(post)
        self.reduction = float(reduction)
        self.smooth_iters = int(smooth_iters)
        self.method = method
        self.workers = workers
        self.as_polydata = bool(as_polydata)
        self.mid = mid
        self.flat_caps = bool(flat_caps)
        self.cap_radius = cap_radius

    @staticmethod
    def finish(frame, Pw, T):
        """Apply final triangle winding correction for left-handed frames."""
        if frame.flips:
            T = T[:, ::-1]
        return Pw, T

    @staticmethod
    def inside(p, poly):
        """Even-odd point-in-polygon: point p (2,), ring poly (n, 2)."""
        x, y = p
        a, b = poly, np.roll(poly, -1, 0)
        cross = (a[:, 1] > y) != (b[:, 1] > y)
        dy = np.where(b[:, 1] != a[:, 1], b[:, 1] - a[:, 1], 1.0)
        xi = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / dy
        return bool(np.count_nonzero(cross & (x < xi)) & 1)

    @staticmethod
    def dense(r, step):
        """Resample a closed ring so neighbouring points are at most ``step`` apart."""
        b = np.roll(r, -1, 0)
        n = np.maximum(1, np.ceil(np.linalg.norm(b - r, axis=1) / step).astype(int))
        e = np.repeat(np.arange(len(r)), n)
        t = (np.arange(e.size) - np.repeat(np.cumsum(n) - n, n)) / n[e]
        return r[e] + (b[e] - r[e]) * t[:, None]

    @staticmethod
    def split_keyholes(r, tol=1e-3):
        """Split a self-touching "keyhole" contour into separate simple loops.

        Some contouring systems represent holes using a contour that touches itself along a
        zero-width channel. That encoding is awkward for filling and distance queries, so this
        function removes the duplicated channel edges and extracts the remaining simple loops.
        """
        k = np.round(np.asarray(r)[:, :2] / tol).astype(np.int64)
        key = k[:, 0] * (1 << 31) + (k[:, 1] + (1 << 30))
        s = np.sort(key)
        if not (s[1:] == s[:-1]).any():
            return [r]
        _, first, vid = np.unique(key, return_index=True, return_inverse=True)
        a, b = vid, np.roll(vid, -1)
        ok = a != b
        a, b = a[ok], b[ok]
        N = len(first)
        e, rev = a * N + b, b * N + a
        if not np.isin(rev, e).any():
            return [r]
        ue, cnt = np.unique(e, return_counts=True)
        left = dict(zip(ue.tolist(), cnt.tolist()))
        for x in e.tolist():
            y = (x % N) * N + x // N
            if left.get(x, 0) > 0 and left.get(y, 0) > 0:
                left[x] -= 1
                left[y] -= 1
        out = {}
        for x, c in left.items():
            for _ in range(c):
                out.setdefault(x // N, []).append(x % N)
        rings = []
        while out:
            start = next(iter(out))
            loop, v = [start], start
            while True:
                nxt = out[v].pop()
                if not out[v]:
                    del out[v]
                if nxt == start or nxt not in out:
                    break
                loop.append(nxt)
                v = nxt
            if len(loop) >= 3:
                rings.append(np.asarray(r)[first[loop]])
        return rings if rings else [r]

    @classmethod
    def prepare_for_frame(cls, slices, frame):
        """Normalize input contours into the local frame and sort them by slice position.

        Output is:

        * original slice keys in sorted order,
        * slice positions along the local ``n`` axis,
        * 2D rings (local ``u,v`` only) for each slice.
        """
        items = []
        for key, rings in slices.items():
            loc = []
            for r in rings:
                r = np.asarray(r, float)
                if len(r) < 3:
                    continue
                r = r[np.any(r != np.roll(r, -1, axis=0), axis=1)]
                if len(r) < 3:
                    continue
                loc.extend(p for p in cls.split_keyholes(frame.to_local(r)) if len(p) >= 3)
            if loc:
                s = float(np.mean(np.concatenate([q[:, 2] for q in loc])))
                items.append((s, key, [q[:, :2] for q in loc]))
        items.sort(key=lambda x: x[0])
        return [i[1] for i in items], [i[0] for i in items], [i[2] for i in items]

    @classmethod
    def split_regions(cls, slices, frame, slice_spacing, gap_factor=1.5, pad=0.5, step=0.25):
        """Split a contour set into disconnected spatial regions.

        This is useful when the same ROI contains separate islands. Meshing them as separate
        regions lets each region choose its own grid spacing ``h`` and avoids unnecessary work
        in large empty space between components.
        """
        items = []
        for key, rings in slices.items():
            for r in rings:
                q = frame.to_local(np.asarray(r, float))
                items.append((key, r, q[:, 2].mean(), q[:, :2]))
        n = len(items)
        z = np.array([it[2] for it in items])
        xy = [it[3] for it in items]
        box = np.array([np.r_[p.min(0) - pad, p.max(0) + pad] for p in xy])
        parent = np.arange(n)
        trees, dense = {}, {}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def tree(i):
            if i not in trees:
                dense[i] = cls.dense(xy[i], step)
                trees[i] = cKDTree(dense[i])
            return trees[i]

        def touches(i, j):
            if cls.inside(xy[i][0], xy[j]) or cls.inside(xy[j][0], xy[i]):
                return True
            a, b = (i, j) if len(xy[i]) <= len(xy[j]) else (j, i)
            tree(a)
            d, _ = tree(b).query(dense[a], distance_upper_bound=pad + step)
            return bool(np.isfinite(d).any())

        order = np.argsort(z)
        reach = gap_factor * slice_spacing + 1e-6
        for ia in range(n):
            a = order[ia]
            ib = ia + 1
            while ib < n and z[order[ib]] - z[a] <= reach:
                b = order[ib]
                if (box[a, 0] <= box[b, 2] and box[b, 0] <= box[a, 2] and
                        box[a, 1] <= box[b, 3] and box[b, 1] <= box[a, 3]):
                    ra, rb = find(a), find(b)
                    if ra != rb and touches(a, b):
                        parent[ra] = rb
                ib += 1
        groups = {}
        for i, it in enumerate(items):
            groups.setdefault(find(i), {}).setdefault(it[0], []).append(it[1])
        return list(groups.values())

    @staticmethod
    def auto_h(slices, frame, k=0.013, h_min=0.5, h_max=2.5, max_voxels=4e6):
        """Choose a working in-plane lattice spacing ``h`` automatically.

        The heuristic balances two competing goals:

        * smaller ``h`` improves geometric fidelity,
        * larger ``h`` keeps the field volume computationally manageable.

        One term scales with the cube root of approximate object volume; another enforces a
        rough voxel-count budget. The final value is clamped to a practical range.
        """
        area, zs, allp = 0.0, [], []
        for rings in slices.values():
            for r in rings:
                q = frame.to_local(np.asarray(r, float))
                x, y = q[:, 0], q[:, 1]
                area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
                zs.append(q[:, 2].mean())
                allp.append(q[:, :2])
        zs = np.unique(np.round(zs, 3))
        dz = float(np.median(np.diff(zs))) if len(zs) > 1 else 1.0
        h_size = k * np.cbrt(area * dz)
        w, d = np.ptp(np.vstack(allp), axis=0)
        h_vox = np.sqrt(w * d * (len(zs) + 2) / max_voxels)
        return float(np.clip(max(h_size, h_vox), h_min, h_max))

    @staticmethod
    def group_contours(contours, direction, origin=(0, 0, 0), spacing=(1, 1, 1),
                       index_space=False, index_order="xyz", slice_spacing=None):
        """Infer a frame from contour/image orientation and group contours by slice.

        This is the usual entry point when starting from a flat list of world-coordinate
        contour rings. It determines which image axis acts as the slice axis, groups contours
        by rounded slice index along that axis, and returns the slice dictionary plus frame
        metadata required by the meshing pipeline.
        """
        D = np.asarray(direction, float)
        origin = np.asarray(origin, float)
        sp = np.asarray(spacing, float)
        pts = [np.asarray(c, float) for c in contours if len(c) >= 3]
        if index_space:
            if index_order == "zyx":
                pts = [c[:, ::-1] for c in pts]
            pts = [origin + (c * sp) @ D.T for c in pts]

        normals = []
        for c in pts:
            q = c - c.mean(0)
            nrm = np.linalg.svd(q, full_matrices=False)[2][2]
            normals.append(nrm * np.sign(nrm @ D[:, np.argmax(np.abs(D.T @ nrm))]))
        a = int(np.argmax(np.abs(D.T @ np.median(normals, axis=0))))
        u, v = D[:, (a + 1) % 3], D[:, (a + 2) % 3]
        frame = Frame(origin, u, v, D[:, a])

        s = np.array([(c - origin).mean(0) @ frame.n for c in pts])
        step = sp[a]
        keys = np.rint(s / step).astype(int)
        slices = {}
        for k, c in zip(keys, pts):
            slices.setdefault(int(k), []).append(c)

        if slice_spacing is None:
            d = np.diff(np.unique(keys) * step)
            slice_spacing = float(np.median(d)) if d.size else step
        return slices, frame, slice_spacing

    def prepare(self, slices):
        """Instance wrapper around :meth:`prepare_for_frame`."""
        return self.prepare_for_frame(slices, self.frame)

    def resolve_h(self, slices):
        """Use user-specified ``h`` or fall back to the automatic heuristic."""
        if self.h is not None:
            return self.h
        return self.auto_h(slices, self.frame)

    def build_volume(self, slices):
        """Construct the internal field volume object for a contour set."""
        h = self.resolve_h(slices)
        _, s, loc = self.prepare(slices)
        return _Volume(loc, s, self.slice_spacing, h, self.m, h if self.c is None else self.c,
                       3, 0, self.gap_factor, self.workers, self.mid,
                       self.flat_caps, self.cap_radius)

    def build_raw_mesh(self, slices):
        """Build the unsmoothed, undecimated mesh in world coordinates."""
        vol = self.build_volume(slices)
        P, T = vol.extract(0, len(vol.pz) - 1)
        P = self.frame.to_world(vol.to_local_xyz(P))
        P, T = self.finish(self.frame, P, T)
        return P, T

    def build_mesh(self, slices):
        """Build the final mesh, optionally applying post-processing."""
        P, T = self.build_raw_mesh(slices)
        if self.post:
            P, T = postprocess(P, T, self.reduction, self.smooth_iters, method=self.method)
        return _to_polydata(P, T) if self.as_polydata else (P, T)

    def build_region_mesh(self, slices, reduction=None, pad=0.5, gap_factor=None,
                          as_polydata=None, **kw):
        """Mesh each disconnected spatial region separately, then merge the results."""
        reduction = self.reduction if reduction is None else reduction
        gap_factor = self.gap_factor if gap_factor is None else gap_factor
        as_polydata = self.as_polydata if as_polydata is None else as_polydata
        Ps, Ts, n = [], [], 0
        for g in self.split_regions(slices, self.frame, self.slice_spacing, gap_factor, pad):
            builder = AdaptiveContourToMesh(
                self.frame,
                self.slice_spacing,
                m=kw.get("m", self.m),
                h=self.auto_h(g, self.frame),
                c=kw.get("c", self.c),
                gap_factor=gap_factor,
                post=False,
                reduction=reduction,
                smooth_iters=kw.get("smooth_iters", self.smooth_iters),
                method=kw.get("method", self.method),
                workers=kw.get("workers", self.workers),
                as_polydata=False,
                mid=kw.get("mid", self.mid),
                flat_caps=kw.get("flat_caps", self.flat_caps),
                cap_radius=kw.get("cap_radius", self.cap_radius),
            )
            P, T = builder.build_mesh(g)
            Ps.append(P)
            Ts.append(T + n)
            n += len(P)
        P, T = np.vstack(Ps), np.vstack(Ts)
        if reduction > 0:
            P, T = postprocess(P, T, reduction, 0)
        return _to_polydata(P, T) if as_polydata else (P, T)


def _prepare(slices, frame):
    """Backwards-compatible function alias for older code."""
    return AdaptiveContourToMesh.prepare_for_frame(slices, frame)


def _finish(frame, Pw, T):
    """Backwards-compatible alias for :meth:`AdaptiveContourToMesh.finish`."""
    return AdaptiveContourToMesh.finish(frame, Pw, T)


def _inside(p, poly):
    """Backwards-compatible alias for :meth:`AdaptiveContourToMesh.inside`."""
    return AdaptiveContourToMesh.inside(p, poly)


def _dense(r, step):
    """Backwards-compatible alias for :meth:`AdaptiveContourToMesh.dense`."""
    return AdaptiveContourToMesh.dense(r, step)


def split_keyholes(r, tol=1e-3):
    """Functional wrapper for splitting self-touching keyhole contours."""
    return AdaptiveContourToMesh.split_keyholes(r, tol=tol)


def split_regions(slices, frame, slice_spacing, gap_factor=1.5, pad=0.5, step=0.25):
    """Functional wrapper for dividing contours into disconnected spatial regions."""
    return AdaptiveContourToMesh.split_regions(
        slices, frame, slice_spacing, gap_factor=gap_factor, pad=pad, step=step
    )


def contours_to_mesh_regions(slices, frame, slice_spacing, reduction=0.5, pad=0.5,
                             gap_factor=1.5, as_polydata=True, **kw):
    """Convenience wrapper for region-wise meshing with one final decimation pass."""
    builder = AdaptiveContourToMesh(frame, slice_spacing, gap_factor=gap_factor,
                                    reduction=reduction, as_polydata=as_polydata, **kw)
    return builder.build_region_mesh(slices, reduction=reduction, pad=pad,
                                     gap_factor=gap_factor, as_polydata=as_polydata, **kw)


def auto_h(slices, frame, k=0.013, h_min=0.5, h_max=2.5, max_voxels=4e6):
    """Functional wrapper for the automatic in-plane grid-spacing heuristic."""
    return AdaptiveContourToMesh.auto_h(
        slices, frame, k=k, h_min=h_min, h_max=h_max, max_voxels=max_voxels
    )


def contours_to_mesh(slices, frame, slice_spacing, m=1, h=None, c=None, gap_factor=1.5,
                     post=True, reduction=0.5, smooth_iters=0, method="auto",
                     workers=-1, as_polydata=True, mid='auto', flat_caps=False, cap_radius=None):
    """One-shot user API for contour -> mesh.

    Parameters are intentionally close to the underlying algorithm:

    * ``m`` controls how many sub-planes are inserted between neighbouring slices,
    * ``h`` is the in-plane lattice spacing,
    * ``c`` is the hat-field cap constant,
    * ``gap_factor`` decides when neighbouring slices are considered linked,
    * post-processing options control decimation and smoothing.
    """
    builder = AdaptiveContourToMesh(frame, slice_spacing, m=m, h=h, c=c,
                                    gap_factor=gap_factor, post=post,
                                    reduction=reduction, smooth_iters=smooth_iters,
                                    method=method, workers=workers,
                                    as_polydata=as_polydata, mid=mid,
                                    flat_caps=flat_caps, cap_radius=cap_radius)
    return builder.build_mesh(slices)


def group_contours(contours, direction, origin=(0, 0, 0), spacing=(1, 1, 1),
                   index_space=False, index_order="xyz", slice_spacing=None):
    """Functional wrapper for contour grouping and frame inference."""
    return AdaptiveContourToMesh.group_contours(
        contours,
        direction,
        origin=origin,
        spacing=spacing,
        index_space=index_space,
        index_order=index_order,
        slice_spacing=slice_spacing,
    )