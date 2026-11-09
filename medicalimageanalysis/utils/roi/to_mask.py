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
    * mesh     -> mask   mesh_to_mask(poly, shape_zyx, origin, spacing, direction)

Key Functionalities:
    1. **Contour Fill**: `contours_to_mask` picks its method from the
       contours. Contours that lie on the image's slices are filled directly
       on their slice (exact at image resolution, fastest). Any other
       contours (sagittal, coronal, oblique, or drawn on a different grid)
       are sampled from the shape-interpolation field that Flying Edges
       meshes, without building a mesh.
    2. **Missing Slices**: by default missing slices stay empty, on both
       paths. Pass `interpolate=True` to fill them from the neighbouring
       contours; this always uses the field.
    3. **Mesh Fill**: `mesh_to_mask` cuts the mesh at the image slice centres
       and uses the signed distance to the cuts.

Choosing a Function:
    Contours, keep gaps as drawn          -> contours_to_mask
    Contours, fill missing slices         -> contours_to_mask(..., interpolate=True)
    Sagittal / coronal / other-grid       -> contours_to_mask(..., interpolate=True)
    Closed surface mesh                   -> mesh_to_mask

Usage:
    >>> from medicalimageanalysis.utils.roi.to_mask import contours_to_mask
    >>> # Rasterize contours onto the image grid
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
                      rings_to_mask, rings_by_slice, volume_to_grid)


def contours_to_mask(contours, shape_zyx, origin, spacing, direction, ramp=None,
                     interpolate=False, direction_rows=True, **kw):
    """
    Convert contours to a mask on the given image grid.

    Two paths are chosen automatically:

    * Contours that lie on this image's slices are filled directly on
      their slice. This is exact at image resolution and the fastest path.
      It is the usual case for an RTSTRUCT on its referenced image, or
      contours from `mask_to_contours` on the same grid.
    * Any other contours (sagittal, coronal, oblique, or drawn on a
      different grid) go through the shape-interpolation field, which is
      built per region and sampled at every voxel centre.

    By default missing slices stay empty on both paths. Pass
    ``interpolate=True`` to fill them from the neighbouring contours; this
    always uses the field.

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

    interpolate : bool
        How far apart contour planes can be and still be joined.
        False (default): only contours on adjacent image slices are joined,
        using the slice spacing from `spacing` and `direction`. A missing
        slice is left empty, and the structure ends half a slice past its
        first and last contours.
        True: the slice spacing is the median spacing between contoured
        slices, so missing slices are filled by shape interpolation. Use this
        for deliberately sparse contours, contours drawn on a different grid
        than this one, or sagittal / coronal contours on an axial grid. Gaps
        larger than 1.5x that spacing are still left empty.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    **kw
        Extra options passed to the region field build (`_region_volumes`).
        Ignored when the contours are filled directly on their slices.

    Returns
    -------
    np.ndarray
        uint8 mask (nz, ny, nx), 0..255, >= 128 inside.

    See Also
    --------
    contours_to_mask_and_mesh : Field-based mask and mesh from one build.

    Examples
    --------
    Round-trip contours on every slice (filled directly on each slice):

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
    >>> from medicalimageanalysis.utils.roi.to_mask import contours_to_mask
    >>> origin, spacing, direction = (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3)
    >>> box = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> box[2:6, 8:24, 8:24] = 1
    >>> rings = mask_to_contours(box, origin, spacing, direction, as_list=True)
    >>> mask = contours_to_mask(rings, (8, 32, 32), origin, spacing, direction)
    >>> bool(mask[3, 16, 16] >= 128), bool(mask[0, 16, 16] >= 128)
    (True, False)

    Contours on every other slice. Gaps stay empty by default, and
    interpolate=True fills them:

    >>> box = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> box[2:7, 8:24, 8:24] = 1
    >>> by_slice = mask_to_contours(box, origin, spacing, direction)
    >>> sparse = by_slice[2] + by_slice[4] + by_slice[6]
    >>> gapped = contours_to_mask(sparse, (8, 32, 32), origin, spacing, direction)
    >>> filled = contours_to_mask(sparse, (8, 32, 32), origin, spacing, direction,
    ...                           interpolate=True)
    >>> bool(gapped[3, 16, 16] >= 128), bool(filled[3, 16, 16] >= 128)
    (False, True)

    Use the result as a boolean mask:

    >>> inside = mask >= 128
    >>> volume_cc = inside.sum() * np.prod(spacing) / 1000
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)

    if not interpolate:
        by_k = rings_by_slice(contours, origin, spacing, direction)
        if by_k is not None:
            return rings_to_mask(by_k, shape_zyx, origin, spacing, direction, ramp)

    slices, frame, dz = _group(contours, origin, spacing, direction, interpolate=interpolate)
    mask = np.zeros(shape_zyx, np.uint8)
    for vol in _region_volumes(slices, frame, dz, **kw):
        volume_to_grid(vol, frame, shape_zyx, origin, spacing, direction, ramp, out=mask)
    return mask

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