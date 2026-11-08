"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Mask Conversion Utilities
=========================

Description:
    This module holds the conversions that produce a uint8 0..255 mask
    (>= 128 inside). The 0..255 values are a signed-distance ramp across the
    boundary, so the mask keeps the ROI's sub-pixel edge.

Conversions:
    * contours -> mask   contours_to_mask(contours, shape_zyx, origin, spacing, direction)
    * contours -> mask   contours_to_mask_direct(contours, shape_zyx, origin, spacing, direction)
    * mesh     -> mask   mesh_to_mask(poly, shape_zyx, origin, spacing, direction)

Key Functionalities:
    1. **Shape Interpolation**: `contours_to_mask` samples the
       shape-interpolation field that Flying Edges meshes (no mesh is built),
       so it handles contours that skip CT slices and sagittal / coronal
       contours on an axial grid.
    2. **Direct Fill**: `contours_to_mask_direct` fills contours that already
       sit on every CT slice. Fastest, but no interpolation.
    3. **Mesh Fill**: `mesh_to_mask` cuts the mesh at the CT slice centres
       and uses the signed distance to the cuts.

Choosing a Function:
    Contours on every CT slice      -> contours_to_mask_direct
    Skipped slices / other planes   -> contours_to_mask
    Closed surface mesh             -> mesh_to_mask

Usage:
    >>> from medicalimageanalysis.utils.roi.to_mask import contours_to_mask
    >>> # Rasterize contours onto the CT grid
    >>> mask = contours_to_mask(contours, ct.array.shape, origin, spacing, direction)
    >>> inside = mask >= 128

Examples
--------
>>> import numpy as np
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.roi.to_mask import mesh_to_mask
>>> sphere = pv.Sphere(radius=8.0, center=(16.0, 16.0, 16.0))
>>> mask = mesh_to_mask(sphere, (16, 32, 32), (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3))
>>> mask.dtype, mask.shape
(dtype('uint8'), (16, 32, 32))
>>> bool(mask[8, 16, 16] >= 128)
True
"""
import numpy as np

from ._raster import (_cut, _group, _region_volumes, normalize_direction,
                      rings_to_mask, volume_to_grid)


def contours_to_mask(contours, shape_zyx, origin, spacing, direction, ramp=None,
                     direction_rows=True, **kw):
    """
    Rasterize contours from the shape-based-interpolation field (no mesh is built).

    Works when contours skip CT slices, and for sagittal / coronal contours
    on an axial grid. Each separate region is interpolated on its own.

    Parameters
    ----------
    contours : list[np.ndarray]
        World-space (n, 3) rings.

    shape_zyx : tuple[int, int, int]
        Output mask shape (nz, ny, nx).

    origin : tuple[float, float, float]
        World (x, y, z) of voxel (0, 0, 0).

    spacing : tuple[float, float, float]
        Voxel spacing (sx, sy, sz) in mm.

    direction : np.ndarray
        3x3 image direction matrix.

    ramp : float | None
        Width in mm of the 0..255 boundary ramp. None uses the default.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    **kw
        Extra options passed to the region field build (`_region_volumes`).

    Returns
    -------
    np.ndarray
        uint8 mask (nz, ny, nx), 0..255, >= 128 inside.

    Examples
    --------
    Contours drawn on every other slice, filled in between:

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
    >>> from medicalimageanalysis.utils.roi.to_mask import contours_to_mask
    >>> origin, spacing, direction = (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3)
    >>> box = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> box[2:7, 8:24, 8:24] = 1
    >>> by_slice = mask_to_contours(box, origin, spacing, direction)
    >>> sparse = by_slice[2] + by_slice[4] + by_slice[6]
    >>> mask = contours_to_mask(sparse, (8, 32, 32), origin, spacing, direction)
    >>> bool(mask[3, 16, 16] >= 128)          # slice 3 had no contour
    True

    Use the result as a boolean mask:

    >>> inside = mask >= 128
    >>> volume_cc = inside.sum() * np.prod(spacing) / 1000
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    slices, frame, dz = _group(contours, origin, spacing, direction)
    mask = np.zeros(shape_zyx, np.uint8)
    for vol in _region_volumes(slices, frame, dz, **kw):
        volume_to_grid(vol, frame, shape_zyx, origin, spacing, direction, ramp, out=mask)
    return mask


def contours_to_mask_direct(contours, shape_zyx, origin, spacing, direction, ramp=None,
                            direction_rows=True):
    """
    Fill contours drawn on the CT slices themselves (no field or mesh).

    Each ring is assigned to the nearest CT slice and filled there. Only
    valid when every CT slice inside the structure has its own contour.

    Parameters
    ----------
    contours : list[np.ndarray]
        World-space (n, 3) rings, each lying on a CT slice.

    shape_zyx : tuple[int, int, int]
        Output mask shape (nz, ny, nx).

    origin : tuple[float, float, float]
        World (x, y, z) of voxel (0, 0, 0).

    spacing : tuple[float, float, float]
        Voxel spacing (sx, sy, sz) in mm.

    direction : np.ndarray
        3x3 image direction matrix.

    ramp : float | None
        Width in mm of the 0..255 boundary ramp. None uses the default.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    Returns
    -------
    np.ndarray
        uint8 mask (nz, ny, nx), 0..255, >= 128 inside.

    Examples
    --------
    Round-trip an RTSTRUCT-style contour set back to a mask:

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
    >>> from medicalimageanalysis.utils.roi.to_mask import contours_to_mask_direct
    >>> origin, spacing, direction = (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3)
    >>> box = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> box[2:6, 8:24, 8:24] = 1
    >>> rings = mask_to_contours(box, origin, spacing, direction, as_list=True)
    >>> mask = contours_to_mask_direct(rings, (8, 32, 32), origin, spacing, direction)
    >>> bool(mask[3, 16, 16] >= 128), bool(mask[0, 16, 16] >= 128)
    (True, False)
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    D, O = np.asarray(direction, float), np.asarray(origin, float)
    by_k = {}
    for r in contours:
        r = np.asarray(r, float)
        k = int(np.rint(((r - O) @ D[:, 2]).mean() / spacing[2]))
        by_k.setdefault(k, []).append(r)
    return rings_to_mask(by_k, shape_zyx, origin, spacing, direction, ramp)


def mesh_to_mask(poly, shape_zyx, origin, spacing, direction, ramp=None,
                 direction_rows=True):
    """
    Rasterize a closed mesh by cutting it at the CT slice centres.

    The mask values come from the signed distance to the cut rings on each
    slice.

    Parameters
    ----------
    poly : vtk.vtkPolyData
        Closed surface mesh in world coordinates (a pv.PolyData works too).

    shape_zyx : tuple[int, int, int]
        Output mask shape (nz, ny, nx).

    origin : tuple[float, float, float]
        World (x, y, z) of voxel (0, 0, 0).

    spacing : tuple[float, float, float]
        Voxel spacing (sx, sy, sz) in mm.

    direction : np.ndarray
        3x3 image direction matrix.

    ramp : float | None
        Width in mm of the 0..255 boundary ramp. None uses the default.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    Returns
    -------
    np.ndarray
        uint8 mask (nz, ny, nx), 0..255, >= 128 inside.

    Examples
    --------
    Rasterize a sphere onto a 2 mm slice grid:

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.roi.to_mask import mesh_to_mask
    >>> sphere = pv.Sphere(radius=8.0, center=(16.0, 16.0, 16.0))
    >>> mask = mesh_to_mask(sphere, (16, 32, 32), (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3))
    >>> bool(mask[8, 16, 16] >= 128)          # centre, z = 16 mm
    True
    >>> bool(mask[0, 16, 16] >= 128)          # z = 0 mm, outside
    False

    Rasterize a mesh onto a loaded image's grid:

    >>> from medicalimageanalysis import Data
    >>> ct = Data.image['CT 01']
    >>> mask = mesh_to_mask(mesh, ct.array.shape, ct.origin, ct.spacing, ct.matrix)
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    rc = _cut(poly, origin, direction, np.arange(shape_zyx[0]) * spacing[2])  # key = slice index
    return rings_to_mask(rc, shape_zyx, origin, spacing, direction, ramp)