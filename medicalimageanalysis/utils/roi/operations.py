"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description: dilation / erosion and boolean operations (union, intersect, subtract) on ROIs.

An ROI can be any of
    contours : list of (n, 3) world rings (or a {slice: [rings]} dict)
    mesh     : closed vtkPolyData
    mask     : uint8 (nz, ny, nx), 0..255, >= 128 inside (same encoding as to_mask)

Everything is done on masks: each ROI is converted to a mask on a common CT grid, combined,
then converted to the requested output ("mesh", "contours" or "mask") at the end.

Examples
    grid = Grid(shape_zyx, origin, spacing, direction)

    dilate(roi, grid, 5)                            # 5 mm everywhere
    dilate(roi, grid, (5, 5, 10))                   # 5 mm in x / y, 10 mm in z
    erode(roi, grid, 3)

    union([a, b, c], grid)                          # a | b | c
    union([a, b], grid, margins=[5, 0])             # (a + 5 mm) | b
    intersect([a, b], grid, margins=[0, -2])        # a & (b - 2 mm)
    subtract(a, [b, c], grid, margins=[0, 3, 3])    # a minus ((b + 3 mm) | (c + 3 mm))

    # chain operations by asking for a mask, e.g. PTV = (CTV1 | CTV2) + 5 mm
    ctv = union([ctv1, ctv2], grid, output="mask")
    ptv = dilate(ctv, grid, 5)

Margins are in mm along world x, y, z: a scalar, or (x, y, z) for anisotropic (ellipsoidal)
margins. Positive = dilate, negative = erode, 0 = unchanged.

How it works
    Dilation / erosion: a 3D distance transform (scipy EDT) gives an approximate distance to
    the ROI everywhere; the voxels right at the new boundary are then corrected with the exact
    distance to the ROI's own surface mesh (sub-0.05 mm on test shapes).

    Booleans: the 0..255 masks are signed-distance ramps across each boundary, so
        union = max(a, b),   intersect = min(a, b),   a minus b = min(a, 255 - b)
    keeps each ROI's sub-pixel boundary instead of snapping to voxels.
"""

from dataclasses import dataclass

import vtk
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from vtk.util import numpy_support as nps

from ._raster import _to_u8
from .to_contours import mask_to_contours
from .to_mask import contours_to_mask, mesh_to_mask
from .to_mesh import contours_to_mask_and_mesh, contours_to_mesh


@dataclass
class Grid:
    """The CT grid all ROIs are put on (same conventions as the to_* conversions)."""
    shape_zyx: tuple            # (nz, ny, nx)
    origin: np.ndarray          # world (x, y, z) of voxel (0, 0, 0)
    spacing: np.ndarray         # (sx, sy, sz) mm
    direction: np.ndarray       # 3x3, columns = image i, j, k axes in world

    def __post_init__(self):
        self.shape_zyx = tuple(int(v) for v in self.shape_zyx)
        self.origin = np.asarray(self.origin, float)
        self.spacing = np.asarray(self.spacing, float)
        self.direction = np.asarray(self.direction, float)

    @property
    def ramp(self):
        # Width (mm) of the 0..255 ramp for masks made here. Two in-plane pixels keeps both
        # pixels either side of a boundary unclipped, so mask -> contours interpolates the
        # true sub-pixel boundary. Only the 127.5 level matters downstream.
        return 2.0 * max(self.spacing[0], self.spacing[1])

    @property
    def geom(self):
        # (origin, spacing, direction): the positional args every conversion takes
        return self.origin, self.spacing, self.direction


# =============================================================================== public API
def dilate(roi, grid, mm, output="mesh", reduction=0.5, **kw):
    """
    Grow one ROI by mm (scalar or (x, y, z), >= 0).
    """
    if np.any(np.asarray(mm) < 0):
        raise ValueError("dilate takes a positive margin; use erode to shrink")

    return _output(_roi_mask(roi, grid, mm, **kw), grid, output, reduction, **kw)


def erode(roi, grid, mm, output="mesh", reduction=0.5, **kw):
    """
    Shrink one ROI by mm (scalar or (x, y, z), given as a positive number).
    """
    if np.any(np.asarray(mm) < 0):
        raise ValueError("erode takes a positive margin (the amount to shrink by)")

    return _output(_roi_mask(roi, grid, -np.asarray(mm, float), **kw), grid, output, reduction, **kw)


def intersect(rois, grid, margins=None, output="mesh", reduction=0.5, **kw):
    """
    rois[0] & rois[1] & ...  margins: None or one margin per ROI, applied first.
    """
    masks = _roi_masks(rois, grid, margins, **kw)
    out = masks[0].copy()
    for m in masks[1:]:
        np.minimum(out, m, out=out)

    return _output(out, grid, output, reduction, **kw)


def subtract(main, others, grid, margins=None, output="mesh", reduction=0.5, **kw):
    """
    main minus (others[0] | others[1] | ...).
    margins: None or one margin per ROI in the order [main, *others], applied first.
    `others` may also be a single ROI.
    """
    if not isinstance(others, (list, tuple)) or _is_ring(others[0]):
        others = [others]                      # a single ROI (contour list counts as one ROI)

    masks = _roi_masks([main, *others], grid, margins, **kw)
    cut = masks[1].copy()
    for m in masks[2:]:
        np.maximum(cut, m, out=cut)            # union of everything being removed

    out = np.minimum(masks[0], 255 - cut)      # inside main AND outside the cut

    return _output(out, grid, output, reduction, **kw)


def union(rois, grid, margins=None, output="mesh", reduction=0.5, **kw):
    """
    rois[0] | rois[1] | ...  margins: None or one margin per ROI, applied first.
    """
    masks = _roi_masks(rois, grid, margins, **kw)
    out = masks[0].copy()
    for m in masks[1:]:
        np.maximum(out, m, out=out)

    return _output(out, grid, output, reduction, **kw)


# =============================================================================== ROI -> mask
def _is_ring(x):
    """
    True for a single (n, 3) contour ring (so a list of them is ONE contour ROI).
    """
    return isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 3


def _roi_masks(rois, grid, margins, **kw):
    """
    Convert a list of ROIs (each with an optional margin) to masks on `grid`.
    """
    if margins is None:
        margins = [0] * len(rois)

    if len(margins) != len(rois):
        raise ValueError(f"got {len(margins)} margins for {len(rois)} ROIs (need one per ROI)")

    return [_roi_mask(r, grid, m, **kw) for r, m in zip(rois, margins)]


def _roi_mask(roi, grid, mm=0, **kw):
    """
    One ROI (contours / mesh / mask) -> uint8 mask on `grid`, dilated or eroded by mm.
    """
    move = np.any(np.asarray(mm, float) != 0)

    if isinstance(roi, np.ndarray) and roi.ndim == 3:
        if roi.shape != grid.shape_zyx:
            raise ValueError(f"mask shape {roi.shape} does not match grid {grid.shape_zyx}")
        mask, surface = roi, None              # surface is rebuilt from the mask if needed

    elif isinstance(roi, vtk.vtkPolyData):
        mask = mesh_to_mask(roi, grid.shape_zyx, *grid.geom, ramp=grid.ramp)
        surface = roi

    else:                                      # contours
        if isinstance(roi, dict):              # {slice: [rings]} from mesh/mask_to_contours
            roi = [r for k in sorted(roi) for r in roi[k]]

        if move:                               # one field build gives mask + exact surface
            mask, surface = contours_to_mask_and_mesh(roi, grid.shape_zyx, *grid.geom, reduction=0, ramp=grid.ramp, **kw)

        else:
            mask = contours_to_mask(roi, grid.shape_zyx, *grid.geom, ramp=grid.ramp, **kw)
            surface = None

    return _move_boundary(mask, grid, mm, surface, **kw) if move else mask


def _output(mask, grid, output, reduction, **kw):
    """
    Mask -> requested output. Meshes go through contours like every other mesh.
    """
    if output == "mask":
        return mask
    rings = mask_to_contours(mask, *grid.geom, as_list=True)

    if output == "contours":
        return rings

    if output == "mesh":
        return contours_to_mesh(rings, *grid.geom, reduction=reduction, **kw)

    raise ValueError(f"output must be 'mesh', 'contours' or 'mask', not {output!r}")


# =============================================================================== dilate / erode
def _move_boundary(mask, grid, mm, surface=None, **kw):
    """
    Dilate (mm > 0) or erode (mm < 0) a uint8 mask. Returns a uint8 mask on the same grid,
    clipped to the image. surface: the ROI's closed mesh for the exact-distance step
    (rebuilt from the mask when None).
    """
    O, sp, D = grid.geom

    # --- margin per axis. Work in "margin units": divide each axis by its margin, then an
    #     ellipsoidal margin is just a distance of 1 in every direction.
    m = np.broadcast_to(np.asarray(mm, float), (3,)).copy()          # world x, y, z
    if np.any(m > 0) and np.any(m < 0):
        raise ValueError("mixed dilate/erode in one margin: do it as two steps")
    sign = 1.0 if np.any(m > 0) else -1.0
    m = np.maximum(np.abs(m), 0.1 * sp.min())    # 0 on an axis -> negligibly small, not 0
    m_ijk = np.abs(D).T @ m                      # world margin -> image (i, j, k) axes

    out = np.zeros(mask.shape, np.uint8)
    inside_full = mask >= 128
    if not inside_full.any():
        return out

    # --- crop to the ROI plus room to grow, and pad by 1 voxel (outside the image = outside)
    zyx = np.nonzero(inside_full)
    room = (np.ceil(m_ijk / sp).astype(int)[::-1] if sign > 0 else 0) + 2
    lo = np.maximum(np.array([a.min() for a in zyx]) - room, 0)
    hi = np.minimum(np.array([a.max() for a in zyx]) + room + 1, mask.shape)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    inside = np.pad(inside_full[crop], 1)
    to_full_ijk = (lo - 1)[::-1]                 # cropped (i, j, k) + this = full-grid index

    # --- approximate signed distance (negative inside) in margin units.
    #     The new boundary is where it equals `sign` (+1 dilate, -1 erode).
    s = (sp / m_ijk)[::-1]                       # voxel size in margin units, (z, y, x)
    sdf = np.where(inside,
                   -ndimage.distance_transform_edt(inside, sampling=s),
                   ndimage.distance_transform_edt(~inside, sampling=s))

    # --- exact distance for the voxels around the new boundary. The EDT measures to voxel
    #     centres (up to ~half a voxel off); the true surface fixes that where it matters.
    if surface is None:
        surface = _mask_to_mesh(mask, grid, **kw)

    if surface.GetNumberOfPoints():
        new_in = sdf < sign
        band = ndimage.binary_dilation(new_in) & ~ndimage.binary_erosion(new_in, border_value=1)
        band = ndimage.binary_dilation(band)     # 1 voxel of slack either side
        kk, jj, ii = np.nonzero(band)
        pts = O + ((np.c_[ii, jj, kk] + to_full_ijk) * sp) @ D.T      # world coordinates
        d = _surface_distance(surface, pts, m)
        sdf[kk, jj, ii] = np.where(inside[kk, jj, ii], -d, d)        # sign from the mask

    # --- back to the 0..255 encoding: distance from the new boundary in (in-plane) mm
    f = (sdf[1:-1, 1:-1, 1:-1] - sign) * min(m_ijk[0], m_ijk[1])
    out[crop] = _to_u8(f, grid.ramp)

    return out


def _mask_to_mesh(mask, grid, **kw):
    """
    Undecimated mesh of a mask (via contours), used as the exact-distance reference.
    """
    rings = mask_to_contours(mask, *grid.geom, as_list=True)

    return contours_to_mesh(rings, *grid.geom, reduction=0, **kw)


# =============================================================================== exact distance
def _surface_distance(poly, pts, m, k=8, chunk=50000):
    """
    Unsigned distance from world points to the triangles of `poly`, with every axis
    divided by its margin m (so the result is in margin units). Each point is tested
    against the k triangles whose centres are nearest; k = 8 matched brute force.
    """
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    pd = tri.GetOutput()
    V = nps.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64) / m
    F = nps.vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    Q = np.asarray(pts, np.float64) / m

    # per-triangle terms, computed once
    A = V[F[:, 0]]
    AB = V[F[:, 1]] - A
    AC = V[F[:, 2]] - A
    N = np.cross(AB, AC)
    nn = (N * N).sum(1)
    INV = np.where(nn > 1e-24, 1.0 / np.maximum(nn, 1e-24), 0.0)     # 0 = degenerate

    _, cand = cKDTree(A + (AB + AC) / 3.0).query(Q, k=k, workers=-1)
    out = np.empty(len(Q))
    for s in range(0, len(Q), chunk):            # chunked to bound memory
        t = cand[s:s + chunk].ravel()
        q = np.repeat(Q[s:s + chunk], k, axis=0)
        out[s:s + chunk] = _tri_dist(q, A[t], AB[t], AC[t], N[t], INV[t]).reshape(-1, k).min(1)

    return out


def _tri_dist(p, a, ab, ac, n, inv):
    """
    Exact point-to-triangle distance, vectorized. Triangle = a + v*ab + w*ac,
    n = ab x ac, inv = 1/|n|^2 (0 for degenerate triangles -> edges only).
    """
    ap = p - a
    # barycentric coordinates of p projected onto the triangle's plane
    v = (np.cross(ap, ac) * n).sum(1) * inv
    w = (np.cross(ab, ap) * n).sum(1) * inv
    on_face = (inv > 0) & (v >= 0) & (w >= 0) & (v + w <= 1)

    # projection lands inside the triangle: distance to the plane
    d = np.abs((ap * n).sum(1)) * np.sqrt(inv)

    # otherwise: nearest of the three edges
    e = ~on_face
    if e.any():
        ap_, ab_, ac_ = ap[e], ab[e], ac[e]

        def seg(q, s):                           # distance from q to segment 0 -> s
            t = np.clip((q * s).sum(1) / np.maximum((s * s).sum(1), 1e-24), 0.0, 1.0)
            return np.linalg.norm(q - t[:, None] * s, axis=1)

        d[e] = np.minimum(np.minimum(seg(ap_, ab_), seg(ap_, ac_)), seg(ap_ - ab_, ac_ - ab_))

    return d