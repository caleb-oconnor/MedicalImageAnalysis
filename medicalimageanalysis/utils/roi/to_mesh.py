"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Mesh Conversion Utilities
=========================

Description:
    This module holds the conversions that produce a closed surface mesh
    from contours. Contours are grouped into separate regions, each region
    gets its own shape-interpolation field, and the field is meshed with
    Flying Edges and decimated once at the end.

Conversions:
    * contours -> mesh          contours_to_mesh(contours, origin, spacing, direction)
    * contours -> mask + mesh   contours_to_mask_and_mesh(contours, shape_zyx, origin, spacing, direction)

Key Functionalities:
    1. **Contour Meshing**: `contours_to_mesh` handles skipped slices,
       multiple disconnected regions, and sagittal / coronal contours.
    2. **Single Field Build**: `contours_to_mask_and_mesh` returns the mask
       and the mesh from one field build per region, so both agree exactly
       and the field is not computed twice.

Usage:
    >>> from medicalimageanalysis.utils.roi.to_mesh import contours_to_mesh
    >>> # Build a surface from RTSTRUCT contours
    >>> mesh = contours_to_mesh(contours, origin, spacing, direction)

Examples
--------
>>> import numpy as np
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
>>> from medicalimageanalysis.utils.roi.to_mesh import contours_to_mesh
>>> origin, spacing, direction = (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3)
>>> box = np.zeros((8, 32, 32), dtype=np.uint8)
>>> box[2:6, 8:24, 8:24] = 1
>>> rings = mask_to_contours(box, origin, spacing, direction, as_list=True)
>>> mesh = pv.wrap(contours_to_mesh(rings, origin, spacing, direction))
>>> mesh.n_cells > 0
True
"""
import numpy as np

from ._contourmesh import _finish, _to_polydata, postprocess
from ._contourmesh import contours_to_mesh_regions as _mesh_regions
from ._raster import _group, _region_volumes, normalize_direction, volume_to_grid


def _empty_mesh():
    """
    Return empty point / triangle arrays with the expected shapes and dtypes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (0, 3) float64 points and (0, 3) int64 triangles.
    """
    return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.int64)


def contours_to_mesh(contours, origin, spacing, direction, reduction=0.5,
                     as_polydata=True, direction_rows=True, **kw):
    """
    Build a closed surface mesh from contours.

    Each separate region gets its own field (and its own h); the combined
    mesh is decimated once at the end.

    Parameters
    ----------
    contours : list[np.ndarray]
        World-space (n, 3) rings.

    origin : tuple[float, float, float]
        World (x, y, z) of voxel (0, 0, 0).

    spacing : tuple[float, float, float]
        Voxel spacing (sx, sy, sz) in mm.

    direction : np.ndarray
        3x3 image direction matrix.

    reduction : float
        Target decimation fraction (0 = no decimation, 0.5 = remove ~half
        the triangles).

    as_polydata : bool
        If True, return vtkPolyData. If False, return (points, triangles)
        arrays.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    **kw
        `_contourmesh` options: m, c, gap_factor, pad, workers, mid,
        flat_caps, cap_radius.

    Returns
    -------
    vtk.vtkPolyData | tuple[np.ndarray, np.ndarray]
        Closed surface mesh, or (points, triangles) when as_polydata=False.

    Examples
    --------
    Mesh a stack of contours:

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
    >>> from medicalimageanalysis.utils.roi.to_mesh import contours_to_mesh
    >>> origin, spacing, direction = (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3)
    >>> box = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> box[2:6, 8:24, 8:24] = 1
    >>> rings = mask_to_contours(box, origin, spacing, direction, as_list=True)
    >>> mesh = pv.wrap(contours_to_mesh(rings, origin, spacing, direction))
    >>> mesh.is_manifold
    True

    Keep every triangle (no decimation) and get raw arrays:

    >>> points, triangles = contours_to_mesh(
    ...     rings, origin, spacing, direction,
    ...     reduction=0,
    ...     as_polydata=False,
    ... )
    >>> points.shape[1], triangles.shape[1]
    (3, 3)

    Use multiple worker threads for large structures:

    >>> mesh = contours_to_mesh(rings, origin, spacing, direction, workers=8)
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    slices, frame, dz = _group(contours, origin, spacing, direction)
    return _mesh_regions(slices, frame, dz, reduction=reduction, as_polydata=as_polydata, **kw)


def contours_to_mask_and_mesh(contours, shape_zyx, origin, spacing, direction,
                              reduction=0.5, ramp=None, as_polydata=True,
                              direction_rows=True, **kw):
    """
    Build a mask and a surface mesh from one field build per region.

    Cheaper than calling `contours_to_mask` and `contours_to_mesh`
    separately, and the two outputs come from the same field.

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

    reduction : float
        Target decimation fraction for the mesh (0 = no decimation).

    ramp : float | None
        Width in mm of the mask's 0..255 boundary ramp. None uses the default.

    as_polydata : bool
        If True, the mesh is vtkPolyData. If False, (points, triangles).

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    **kw
        Extra options passed to the region field build (`_region_volumes`).

    Returns
    -------
    tuple[np.ndarray, vtk.vtkPolyData | tuple[np.ndarray, np.ndarray]]
        (mask, mesh). The mask is uint8 (nz, ny, nx), 0..255, >= 128 inside.

    Examples
    --------
    Get both outputs for one ROI:

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
    >>> from medicalimageanalysis.utils.roi.to_mesh import contours_to_mask_and_mesh
    >>> origin, spacing, direction = (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3)
    >>> box = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> box[2:6, 8:24, 8:24] = 1
    >>> rings = mask_to_contours(box, origin, spacing, direction, as_list=True)
    >>> mask, mesh = contours_to_mask_and_mesh(
    ...     rings, (8, 32, 32), origin, spacing, direction,
    ... )
    >>> mask.dtype, mask.shape
    (dtype('uint8'), (8, 32, 32))

    Undecimated mesh as raw arrays:

    >>> mask, (points, triangles) = contours_to_mask_and_mesh(
    ...     rings, (8, 32, 32), origin, spacing, direction,
    ...     reduction=0,
    ...     as_polydata=False,
    ... )
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    slices, frame, dz = _group(contours, origin, spacing, direction)
    mask = np.zeros(shape_zyx, np.uint8)
    Ps, Ts, n = [], [], 0
    for vol in _region_volumes(slices, frame, dz, **kw):
        P, T = vol.extract(0, len(vol.pz) - 1)
        if len(P):
            P, T = _finish(frame, frame.to_world(vol.to_local_xyz(P)), T)
            Ps.append(P); Ts.append(T + n); n += len(P)
        volume_to_grid(vol, frame, shape_zyx, origin, spacing, direction, ramp, out=mask)
    if Ps:
        P, T = np.vstack(Ps), np.vstack(Ts)
    else:
        P, T = _empty_mesh()

    if reduction > 0 and len(T):
        P, T = postprocess(P, T, reduction, 0)

    return mask, (_to_polydata(P, T) if as_polydata else (P, T))