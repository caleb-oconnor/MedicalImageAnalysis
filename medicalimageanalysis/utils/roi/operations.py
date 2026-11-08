"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

ROI Margin and Boolean Operations
=================================

Description:
    This module provides dilation / erosion and boolean operations (union,
    intersect, subtract) on ROIs. Every ROI is converted to a mask on a
    common CT grid, combined, then converted to the requested output
    ("mesh", "contours" or "mask") at the end.

Supported ROI Inputs:
    * contours : list of (n, 3) world rings (or a {slice: [rings]} dict)
    * mesh     : closed vtkPolyData
    * mask     : uint8 (nz, ny, nx), 0..255, >= 128 inside (same encoding as to_mask)

Key Functionalities:
    1. **Margins**: `dilate` / `erode` grow or shrink an ROI by a scalar or
       anisotropic (x, y, z) margin in mm.
    2. **Booleans**: `union`, `intersect` and `subtract`, each with an
       optional margin per ROI applied first.
    3. **Chaining**: Ask for output="mask" to feed one result into the next
       operation without a mesh round trip.

Margins:
    Margins are in mm along world x, y, z: a scalar, or (x, y, z) for
    anisotropic (ellipsoidal) margins. Positive = dilate, negative = erode,
    0 = unchanged.

How It Works:
    Dilation / erosion: a 3D distance transform (scipy EDT) gives an
    approximate distance to the ROI everywhere; the voxels right at the new
    boundary are then corrected with the exact distance to the ROI's own
    surface mesh (sub-0.05 mm on test shapes).

    Booleans: the 0..255 masks are signed-distance ramps across each
    boundary, so
        union = max(a, b),   intersect = min(a, b),   a minus b = min(a, 255 - b)
    keeps each ROI's sub-pixel boundary instead of snapping to voxels.

Usage:
    >>> from medicalimageanalysis.utils.roi.operations import Grid, dilate, union
    >>> grid = Grid(shape_zyx, origin, spacing, direction)
    >>> # PTV = (CTV1 | CTV2) + 5 mm
    >>> ctv = union([ctv1, ctv2], grid, output="mask")
    >>> ptv = dilate(ctv, grid, 5)

Examples
--------
>>> import numpy as np
>>> from medicalimageanalysis.utils.roi.operations import (
...     Grid, dilate, erode, intersect, subtract, union,
... )
>>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))
>>> a = np.zeros((16, 64, 64), dtype=np.uint8)
>>> a[4:12, 16:40, 16:40] = 255
>>> b = np.zeros((16, 64, 64), dtype=np.uint8)
>>> b[4:12, 28:52, 28:52] = 255
>>> dilate(a, grid, 5)                              # 5 mm everywhere
>>> dilate(a, grid, (5, 5, 10))                     # 5 mm in x / y, 10 mm in z
>>> erode(a, grid, 3)
>>> union([a, b], grid)                             # a | b
>>> union([a, b], grid, margins=[5, 0])             # (a + 5 mm) | b
>>> intersect([a, b], grid, margins=[0, -2])        # a & (b - 2 mm)
>>> subtract(a, b, grid, margins=[0, 3])            # a minus (b + 3 mm)
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
    """
    The CT grid all ROIs are put on (same conventions as the to_* conversions).

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Mask shape (nz, ny, nx).

    origin : np.ndarray
        World (x, y, z) of voxel (0, 0, 0).

    spacing : np.ndarray
        Voxel spacing (sx, sy, sz) in mm.

    direction : np.ndarray
        3x3 matrix; columns are the image i, j, k axes in world.

    Examples
    --------
    A plain 1 mm grid:

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.operations import Grid
    >>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))

    The grid of a loaded CT:

    >>> from medicalimageanalysis import Data
    >>> ct = Data.image['CT 01']
    >>> grid = Grid(ct.array.shape, ct.origin, ct.spacing, ct.matrix)
    """
    shape_zyx: tuple            # (nz, ny, nx)
    origin: np.ndarray          # world (x, y, z) of voxel (0, 0, 0)
    spacing: np.ndarray         # (sx, sy, sz) mm
    direction: np.ndarray       # 3x3, columns = image i, j, k axes in world

    def __post_init__(self):
        """
        Normalize the grid metadata to an int tuple and float NumPy arrays.

        Examples
        --------
        Lists and tuples are accepted and converted:

        >>> grid = Grid([16, 64, 64], [0, 0, 0], [0.98, 0.98, 2.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> grid.shape_zyx, grid.spacing.dtype
        ((16, 64, 64), dtype('float64'))
        """
        self.shape_zyx = tuple(int(v) for v in self.shape_zyx)
        self.origin = np.asarray(self.origin, float)
        self.spacing = np.asarray(self.spacing, float)
        self.direction = np.asarray(self.direction, float)

    @property
    def ramp(self):
        """
        Width in mm of the 0..255 boundary ramp for masks made on this grid.

        Two in-plane pixels, so both pixels either side of a boundary stay
        unclipped and mask -> contours can interpolate the true sub-pixel
        boundary.

        Returns
        -------
        float
            2 x the larger in-plane spacing.

        Examples
        --------
        >>> import numpy as np
        >>> grid = Grid((16, 64, 64), np.zeros(3), (0.8, 0.8, 3.0), np.eye(3))
        >>> grid.ramp
        1.6
        """
        # Width (mm) of the 0..255 ramp for masks made here. Two in-plane pixels keeps both
        # pixels either side of a boundary unclipped, so mask -> contours interpolates the
        # true sub-pixel boundary. Only the 127.5 level matters downstream.
        return 2.0 * max(self.spacing[0], self.spacing[1])

    @property
    def geom(self):
        """
        The positional geometry arguments every conversion takes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (origin, spacing, direction).

        Examples
        --------
        >>> import numpy as np
        >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
        >>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))
        >>> origin, spacing, direction = grid.geom
        >>> rings = mask_to_contours(mask, *grid.geom, as_list=True)
        """
        # (origin, spacing, direction): the positional args every conversion takes
        return self.origin, self.spacing, self.direction


# =============================================================================== public API
def dilate(roi, grid, mm, output="mesh", reduction=0.5, **kw):
    """
    Grow one ROI by a margin in mm.

    Parameters
    ----------
    roi : list[np.ndarray] | dict | vtk.vtkPolyData | np.ndarray
        Contours, closed mesh, or uint8 mask on `grid`.

    grid : Grid
        Common CT grid.

    mm : float | tuple[float, float, float]
        Margin in mm, scalar or (x, y, z). Must be >= 0.

    output : str
        'mesh', 'contours' or 'mask'.

    reduction : float
        Decimation fraction when output='mesh'.

    **kw
        Extra options passed to the contour / mesh conversions.

    Returns
    -------
    vtk.vtkPolyData | list[np.ndarray] | np.ndarray
        The grown ROI in the requested format.

    Raises
    ------
    ValueError
        If any margin is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.operations import Grid, dilate
    >>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))
    >>> ctv = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> ctv[4:12, 24:40, 24:40] = 255

    5 mm everywhere, returned as a mesh:

    >>> ptv = dilate(ctv, grid, 5)

    Anisotropic margin (3 mm in x / y, 6 mm in z), kept as a mask:

    >>> ptv_mask = dilate(ctv, grid, (3, 3, 6), output="mask")
    >>> int((ptv_mask >= 128).sum()) > int((ctv >= 128).sum())
    True

    Dilate RTSTRUCT contours and get contours back:

    >>> ptv_rings = dilate(ctv_rings, grid, 5, output="contours")
    """
    if np.any(np.asarray(mm) < 0):
        raise ValueError("dilate takes a positive margin; use erode to shrink")

    return _output(_roi_mask(roi, grid, mm, **kw), grid, output, reduction, **kw)


def erode(roi, grid, mm, output="mesh", reduction=0.5, **kw):
    """
    Shrink one ROI by a margin in mm (given as a positive number).

    Parameters
    ----------
    roi : list[np.ndarray] | dict | vtk.vtkPolyData | np.ndarray
        Contours, closed mesh, or uint8 mask on `grid`.

    grid : Grid
        Common CT grid.

    mm : float | tuple[float, float, float]
        Amount to shrink by in mm, scalar or (x, y, z). Must be >= 0.

    output : str
        'mesh', 'contours' or 'mask'.

    reduction : float
        Decimation fraction when output='mesh'.

    **kw
        Extra options passed to the contour / mesh conversions.

    Returns
    -------
    vtk.vtkPolyData | list[np.ndarray] | np.ndarray
        The shrunk ROI in the requested format.

    Raises
    ------
    ValueError
        If any margin is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.operations import Grid, erode
    >>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))
    >>> organ = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> organ[2:14, 16:48, 16:48] = 255

    3 mm everywhere:

    >>> inner = erode(organ, grid, 3, output="mask")
    >>> bool(inner[8, 18, 32] >= 128)          # 2 mm from the old edge
    False

    Shrink a mesh in-plane only:

    >>> inner_mesh = erode(organ_mesh, grid, (3, 3, 0))
    """
    if np.any(np.asarray(mm) < 0):
        raise ValueError("erode takes a positive margin (the amount to shrink by)")

    return _output(_roi_mask(roi, grid, -np.asarray(mm, float), **kw), grid, output, reduction, **kw)


def intersect(rois, grid, margins=None, output="mesh", reduction=0.5, **kw):
    """
    Intersection of several ROIs: rois[0] & rois[1] & ...

    Parameters
    ----------
    rois : list
        ROIs (contours, meshes or masks; types can be mixed).

    grid : Grid
        Common CT grid.

    margins : list | None
        One margin per ROI (mm, scalar or (x, y, z)), applied before
        intersecting. None for no margins.

    output : str
        'mesh', 'contours' or 'mask'.

    reduction : float
        Decimation fraction when output='mesh'.

    **kw
        Extra options passed to the contour / mesh conversions.

    Returns
    -------
    vtk.vtkPolyData | list[np.ndarray] | np.ndarray
        The intersection in the requested format.

    Raises
    ------
    ValueError
        If the number of margins does not match the number of ROIs.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.operations import Grid, intersect
    >>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))
    >>> a = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> a[4:12, 16:40, 16:40] = 255
    >>> b = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> b[4:12, 28:52, 28:52] = 255

    Overlap of two ROIs:

    >>> overlap = intersect([a, b], grid, output="mask")
    >>> bool(overlap[8, 34, 34] >= 128), bool(overlap[8, 20, 20] >= 128)
    (True, False)

    Part of the PTV within 2 mm inside the liver:

    >>> ptv_in_liver = intersect([ptv, liver], grid, margins=[0, -2])
    """
    masks = _roi_masks(rois, grid, margins, **kw)
    out = masks[0].copy()
    for m in masks[1:]:
        np.minimum(out, m, out=out)

    return _output(out, grid, output, reduction, **kw)


def subtract(main, others, grid, margins=None, output="mesh", reduction=0.5, **kw):
    """
    Subtract ROIs: main minus (others[0] | others[1] | ...).

    Parameters
    ----------
    main : list[np.ndarray] | dict | vtk.vtkPolyData | np.ndarray
        ROI to subtract from.

    others : list | ROI
        ROIs to remove. A single ROI is also accepted (a contour list
        counts as one ROI).

    grid : Grid
        Common CT grid.

    margins : list | None
        One margin per ROI in the order [main, *others], applied first.
        None for no margins.

    output : str
        'mesh', 'contours' or 'mask'.

    reduction : float
        Decimation fraction when output='mesh'.

    **kw
        Extra options passed to the contour / mesh conversions.

    Returns
    -------
    vtk.vtkPolyData | list[np.ndarray] | np.ndarray
        The remaining ROI in the requested format.

    Raises
    ------
    ValueError
        If the number of margins does not match the number of ROIs.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.operations import Grid, subtract
    >>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))
    >>> a = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> a[4:12, 16:40, 16:40] = 255
    >>> b = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> b[4:12, 28:52, 28:52] = 255

    a minus b:

    >>> rest = subtract(a, b, grid, output="mask")
    >>> bool(rest[8, 20, 20] >= 128), bool(rest[8, 34, 34] >= 128)
    (True, False)

    Optimization structure: PTV minus (rectum + 3 mm) minus (bladder + 3 mm):

    >>> ptv_opt = subtract(ptv, [rectum, bladder], grid, margins=[0, 3, 3])
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
    Union of several ROIs: rois[0] | rois[1] | ...

    Parameters
    ----------
    rois : list
        ROIs (contours, meshes or masks; types can be mixed).

    grid : Grid
        Common CT grid.

    margins : list | None
        One margin per ROI (mm, scalar or (x, y, z)), applied before the
        union. None for no margins.

    output : str
        'mesh', 'contours' or 'mask'.

    reduction : float
        Decimation fraction when output='mesh'.

    **kw
        Extra options passed to the contour / mesh conversions.

    Returns
    -------
    vtk.vtkPolyData | list[np.ndarray] | np.ndarray
        The union in the requested format.

    Raises
    ------
    ValueError
        If the number of margins does not match the number of ROIs.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.operations import Grid, dilate, union
    >>> grid = Grid((16, 64, 64), np.zeros(3), np.ones(3), np.eye(3))
    >>> a = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> a[4:12, 16:40, 16:40] = 255
    >>> b = np.zeros((16, 64, 64), dtype=np.uint8)
    >>> b[4:12, 28:52, 28:52] = 255

    Combine two ROIs:

    >>> both = union([a, b], grid, output="mask")
    >>> bool(both[8, 20, 20] >= 128), bool(both[8, 48, 48] >= 128)
    (True, True)

    Margin on the first ROI only:

    >>> mesh = union([a, b], grid, margins=[5, 0])

    Chain operations: PTV = (CTV1 | CTV2) + 5 mm

    >>> ctv = union([ctv1, ctv2], grid, output="mask")
    >>> ptv = dilate(ctv, grid, 5)
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

    Parameters
    ----------
    x : object
        Candidate ring.

    Returns
    -------
    bool
        True if `x` is a 2D NumPy array with 3 columns.
    """
    return isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 3


def _roi_masks(rois, grid, margins, **kw):
    """
    Convert a list of ROIs (each with an optional margin) to masks on `grid`.

    Parameters
    ----------
    rois : list
        ROIs (contours, meshes or masks).

    grid : Grid
        Common CT grid.

    margins : list | None
        One margin per ROI, or None for no margins.

    Returns
    -------
    list[np.ndarray]
        uint8 masks, one per ROI.

    Raises
    ------
    ValueError
        If the number of margins does not match the number of ROIs.
    """
    if margins is None:
        margins = [0] * len(rois)

    if len(margins) != len(rois):
        raise ValueError(f"got {len(margins)} margins for {len(rois)} ROIs (need one per ROI)")

    return [_roi_mask(r, grid, m, **kw) for r, m in zip(rois, margins)]


def _roi_mask(roi, grid, mm=0, **kw):
    """
    Convert one ROI (contours / mesh / mask) to a uint8 mask on `grid`, dilated or eroded by mm.

    Parameters
    ----------
    roi : list[np.ndarray] | dict | vtk.vtkPolyData | np.ndarray
        Contours, closed mesh, or uint8 mask.

    grid : Grid
        Common CT grid.

    mm : float | tuple[float, float, float] | np.ndarray
        Margin in mm (positive = dilate, negative = erode, 0 = unchanged).

    Returns
    -------
    np.ndarray
        uint8 mask (nz, ny, nx), 0..255, >= 128 inside.

    Raises
    ------
    ValueError
        If a mask ROI does not match the grid shape.
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
    Convert a mask to the requested output. Meshes go through contours like every other mesh.

    Parameters
    ----------
    mask : np.ndarray
        uint8 mask on `grid`.

    grid : Grid
        Common CT grid.

    output : str
        'mesh', 'contours' or 'mask'.

    reduction : float
        Decimation fraction when output='mesh'.

    Returns
    -------
    vtk.vtkPolyData | list[np.ndarray] | np.ndarray
        The ROI in the requested format.

    Raises
    ------
    ValueError
        If `output` is not 'mesh', 'contours' or 'mask'.
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
    Dilate (mm > 0) or erode (mm < 0) a uint8 mask.

    Parameters
    ----------
    mask : np.ndarray
        uint8 mask on `grid`.

    grid : Grid
        Common CT grid.

    mm : float | tuple[float, float, float] | np.ndarray
        Margin in mm. All axes must have the same sign.

    surface : vtk.vtkPolyData | None
        The ROI's closed mesh for the exact-distance step. Rebuilt from the
        mask when None.

    Returns
    -------
    np.ndarray
        uint8 mask on the same grid, clipped to the image.

    Raises
    ------
    ValueError
        If the margin mixes dilation and erosion across axes.
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

    Parameters
    ----------
    mask : np.ndarray
        uint8 mask on `grid`.

    grid : Grid
        Common CT grid.

    Returns
    -------
    vtk.vtkPolyData
        Closed surface mesh.
    """
    rings = mask_to_contours(mask, *grid.geom, as_list=True)

    return contours_to_mesh(rings, *grid.geom, reduction=0, **kw)


# =============================================================================== exact distance
def _surface_distance(poly, pts, m, k=8, chunk=50000):
    """
    Unsigned distance from world points to the triangles of `poly`, in margin units.

    Every axis is divided by its margin m, so the result is in margin units.
    Each point is tested against the k triangles whose centres are nearest;
    k = 8 matched brute force.

    Parameters
    ----------
    poly : vtk.vtkPolyData
        Closed surface mesh.

    pts : np.ndarray
        (n, 3) world points.

    m : np.ndarray
        Margin per world axis (x, y, z) in mm.

    k : int
        Number of nearest candidate triangles tested per point.

    chunk : int
        Points processed per batch, to bound memory.

    Returns
    -------
    np.ndarray
        (n,) distances in margin units.
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
    Exact point-to-triangle distance, vectorized.

    Triangle = a + v*ab + w*ac, n = ab x ac, inv = 1/|n|^2 (0 for degenerate
    triangles -> edges only).

    Parameters
    ----------
    p : np.ndarray
        (n, 3) query points.

    a, ab, ac : np.ndarray
        (n, 3) triangle origin and edge vectors.

    n : np.ndarray
        (n, 3) triangle normals (ab x ac).

    inv : np.ndarray
        (n,) 1/|n|^2, or 0 for degenerate triangles.

    Returns
    -------
    np.ndarray
        (n,) unsigned distances.
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