"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Contour Conversion Utilities
============================

Description:
    This module holds the conversions that produce contours: closed meshes
    are cut with exact planes normal to the image k axis, and masks are
    traced slice by slice with marching squares (VTK Flying Edges 2D).

Conversions:
    * mesh -> contours   mesh_to_contours(poly, origin, spacing, direction, positions=None)
    * mask -> contours   mask_to_contours(mask, origin, spacing, direction)
    * helper             contour_positions(contours, origin, direction)

Key Functionalities:
    1. **Mesh Cutting**: `mesh_to_contours` cuts at every CT slice centre the
       mesh spans, or at caller-supplied positions.
    2. **Mask Tracing**: `mask_to_contours` gives sub-pixel boundaries for
       uint8 0..255 ramp masks and pixel-edge boundaries for binary masks.
    3. **Slice Positions**: `contour_positions` recovers which planes a
       contour list lies on, e.g. to cut a mesh back at the original slices.

Output Format:
    Both conversions return {slice_index: [(n, 3) world rings]} keyed by the
    nearest CT slice, or a flat list of rings with as_list=True.

Usage:
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
    >>> # Trace a mask into world-space contours
    >>> contours = mask_to_contours(mask, origin, spacing, direction)

Examples
--------
>>> import numpy as np
>>> from medicalimageanalysis.utils.roi.to_contours import (
...     contour_positions,
...     mask_to_contours,
... )
>>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
>>> mask[2:6, 8:24, 8:24] = 1
>>> origin, spacing, direction = (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3)
>>> contours = mask_to_contours(mask, origin, spacing, direction)
>>> sorted(contours.keys())
[2, 3, 4, 5]
>>> rings = mask_to_contours(mask, origin, spacing, direction, as_list=True)
>>> contour_positions(rings, origin, direction)
array([ 4.,  6.,  8., 10.])
"""
import numpy as np
import vtk
from vtk.util import numpy_support as nps

from ._raster import _cut, _flat, _loops, normalize_direction


def contour_positions(contours, origin, direction, direction_rows=True):
    """
    Positions of the planes a contour list lies on, in mm along the image k axis.

    Each ring's mean distance from `origin` along the k axis is rounded to
    0.001 mm, de-duplicated and sorted. Typical use is cutting a mesh back
    at the originally contoured slices with `mesh_to_contours`.

    Parameters
    ----------
    contours : list[np.ndarray]
        World-space (n, 3) rings. Rings with fewer than 3 points are ignored.

    origin : tuple[float, float, float]
        World (x, y, z) of voxel (0, 0, 0).

    direction : np.ndarray
        3x3 image direction matrix.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    Returns
    -------
    np.ndarray
        Sorted unique plane positions in mm from the origin along k.

    Examples
    --------
    Contours traced from a mask with 2 mm slices:

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.to_contours import (
    ...     contour_positions,
    ...     mask_to_contours,
    ... )
    >>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> mask[2:6, 8:24, 8:24] = 1
    >>> rings = mask_to_contours(
    ...     mask, (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3), as_list=True,
    ... )
    >>> contour_positions(rings, (0.0, 0.0, 0.0), np.eye(3))
    array([ 4.,  6.,  8., 10.])

    Hand-made rings, two on the same plane:

    >>> square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    >>> rings = [np.c_[square, np.full(4, z)] for z in (3.0, 3.0, 6.0)]
    >>> contour_positions(rings, (0.0, 0.0, 0.0), np.eye(3))
    array([3., 6.])
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    n, O = np.asarray(direction, float)[:, 2], np.asarray(origin, float)
    return np.array(sorted({round(float(((np.asarray(c, float) - O) @ n).mean()), 3)
                            for c in contours if len(c) >= 3}))


def mesh_to_contours(poly, origin, spacing, direction, positions=None, as_list=False,
                     direction_rows=True):
    """
    Cut a closed mesh into contours with exact planes normal to the image k axis.

    Parameters
    ----------
    poly : vtk.vtkPolyData
        Closed surface mesh in world coordinates (a pv.PolyData works too).

    origin : tuple[float, float, float]
        World (x, y, z) of voxel (0, 0, 0).

    spacing : tuple[float, float, float]
        Voxel spacing (sx, sy, sz) in mm. Only sz is used.

    direction : np.ndarray
        3x3 image direction matrix.

    positions : array-like | None
        Cut positions in mm along k from `origin`. If None, every CT slice
        centre the mesh spans is cut. Use `contour_positions` to cut at the
        planes of an existing contour list.

    as_list : bool
        If True, return a flat list of rings instead of a dictionary.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    Returns
    -------
    dict[int, list[np.ndarray]] | list[np.ndarray]
        {slice_index: [(n, 3) world rings]} keyed by the nearest CT slice,
        or a flat list of rings when as_list=True.

    Examples
    --------
    Cut a sphere at every CT slice it spans:

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.roi.to_contours import mesh_to_contours
    >>> mesh = pv.Sphere(radius=10.0, center=(16.0, 16.0, 16.0))
    >>> contours = mesh_to_contours(
    ...     mesh,
    ...     origin=(0.0, 0.0, 0.0),
    ...     spacing=(1.0, 1.0, 2.0),
    ...     direction=np.eye(3),
    ... )
    >>> ring = contours[8][0]          # slice 8 -> z = 16 mm
    >>> ring.shape[1]
    3

    Cut back at the planes the original contours were drawn on:

    >>> from medicalimageanalysis.utils.roi.to_contours import contour_positions
    >>> positions = contour_positions(original_rings, origin, direction)
    >>> rings = mesh_to_contours(
    ...     mesh, origin, spacing, direction,
    ...     positions=positions,
    ...     as_list=True,
    ... )
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    n, O, sz = np.asarray(direction, float)[:, 2], np.asarray(origin, float), float(spacing[2])
    if positions is None:
        d = (nps.vtk_to_numpy(poly.GetPoints().GetData()) - O) @ n
        positions = np.arange(np.ceil(d.min() / sz), np.floor(d.max() / sz) + 1) * sz
    positions = np.asarray(positions, float)
    cut = _cut(poly, O, direction, positions)
    out = {}
    for i, rings in cut.items():
        out.setdefault(int(np.rint(positions[i] / sz)), []).extend(rings)
    return _flat(out) if as_list else out


def mask_to_contours(mask, origin, spacing, direction, level=None, as_list=False,
                     direction_rows=True):
    """
    Trace a mask into world-space contours with marching squares on each slice.

    The mask is padded by one voxel so contours touching the image edge
    still close.

    Parameters
    ----------
    mask : np.ndarray
        Mask in z-y-x order, (nz, ny, nx). Either binary (0/1) or uint8
        0..255 with >= 128 inside.

    origin : tuple[float, float, float]
        World (x, y, z) of voxel (0, 0, 0).

    spacing : tuple[float, float, float]
        Voxel spacing (sx, sy, sz) in mm.

    direction : np.ndarray
        3x3 image direction matrix.

    level : float | None
        Iso-level to trace. If None, 127.5 for a uint8 0..255 mask
        (sub-pixel boundary) and 0.5 for a binary mask.

    as_list : bool
        If True, return a flat list of rings instead of a dictionary.

    direction_rows : bool
        Layout of `direction`, forwarded to `normalize_direction`.

    Returns
    -------
    dict[int, list[np.ndarray]] | list[np.ndarray]
        {slice_index: [(n, 3) world rings]} for every non-empty slice, or a
        flat list of rings when as_list=True.

    Examples
    --------
    Trace a binary box mask (2 mm slices):

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.to_contours import mask_to_contours
    >>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
    >>> mask[2:6, 8:24, 8:24] = 1
    >>> contours = mask_to_contours(mask, (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3))
    >>> sorted(contours.keys())
    [2, 3, 4, 5]
    >>> float(contours[2][0][:, 2].mean())      # slice 2 -> z = 4 mm
    4.0

    Get a flat list of rings instead:

    >>> rings = mask_to_contours(
    ...     mask, (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3), as_list=True,
    ... )
    >>> len(rings)
    4

    Trace a probability map at a custom level:

    >>> probability = (np.random.rand(8, 32, 32) * 255).astype(np.uint8)
    >>> contours = mask_to_contours(
    ...     probability, (0.0, 0.0, 0.0), (1.0, 1.0, 2.0), np.eye(3), level=200,
    ... )
    """
    direction = normalize_direction(direction, direction_rows=direction_rows)
    if level is None:
        level = 127.5 if mask.dtype == np.uint8 and mask.max() > 1 else 0.5
    D = np.asarray(direction, float)
    sp = np.asarray(spacing, float)
    origin = np.asarray(origin, float)
    out = {}
    zs = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(1))
    for k in zs:
        sl = np.pad(mask[k].astype(np.float32), 1)
        img = vtk.vtkImageData()
        img.SetDimensions(sl.shape[1], sl.shape[0], 1)
        img.SetOrigin(-1.0, -1.0, 0.0)                 # pixel index space, incl. pad
        img.GetPointData().SetScalars(nps.numpy_to_vtk(sl.ravel(), deep=True))
        fe = vtk.vtkFlyingEdges2D()
        fe.SetInputData(img)
        fe.SetValue(0, level)
        fe.Update()
        for ring in _loops(fe.GetOutput()):
            ijk = np.c_[ring[:, :2], np.full(len(ring), k)]
            out.setdefault(int(k), []).append(origin + (ijk * sp) @ D.T)
    return _flat(out) if as_list else out