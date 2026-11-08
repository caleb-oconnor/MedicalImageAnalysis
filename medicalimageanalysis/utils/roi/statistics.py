"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Geometric Measurement Utilities for Contours and Meshes
=======================================================

Description:
    This module estimates the longest internal distance of an ROI, either
    from a stack of planar contours or from a closed surface mesh. The
    distance is measured only along segments that stay inside the ROI, so
    concave shapes are not overestimated.

Supported Inputs:
    * Contours: list of (N, 2) or (N, 3) arrays, in mm
    * Meshes: closed PyVista / vtkPolyData surfaces, in mm

Key Functionalities:
    1. **Contour Distance**: `contour_longest_distance` returns the longest
       in-plane distance across a stack of contours.
    2. **Mesh Distance**: `mesh_longest_distance` returns the longest
       distance in full 3D, or per axial / sagittal / coronal slice using the
       actual image slice positions.

Coordinate Note:
    `mesh_longest_distance` takes `array_shape`, `origin`, `spacing` and
    `direction` in x-y-z order (axis 0 = x / sagittal, axis 1 = y / coronal,
    axis 2 = z / axial), not the z-y-x order used for image arrays.

Usage:
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.roi.statistics import mesh_longest_distance
    >>> # Longest 3D distance of a 10 mm radius sphere
    >>> mesh_longest_distance(pv.Sphere(radius=10.0), mode='3d')

Examples
--------
>>> import numpy as np
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.roi.statistics import (
...     contour_longest_distance,
...     mesh_longest_distance,
... )
>>> rectangle = np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
>>> round(contour_longest_distance([rectangle], exact=True), 2)
22.36
>>> round(mesh_longest_distance(pv.Sphere(radius=10.0), mode='3d'), 1)
20.0
"""

from itertools import combinations

import numpy as np
import pyvista as pv

from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon


def contour_longest_distance(contours, exact=False):
    """
    Longest in-plane distance across a stack of contours, in mm.

    Each contour is handled on its own and the largest result is returned.
    A candidate segment only counts if it lies inside the contour polygon,
    so the distance never crosses a concave notch. (N, 3) contours are
    projected onto their best-fit plane first.

    Parameters
    ----------
    contours : list[np.ndarray]
        Contours as (N, 2) or (N, 3) point arrays in mm.

    exact : bool
        If False, only convex hull vertices are tested (fast, can undercount
        on concave contours). If True, every contour point is tested
        (slower, exact).

    Returns
    -------
    float
        Longest internal distance in mm, or 0.0 if no valid segment is found.

    Examples
    --------
    Diagonal of a 20 x 10 mm rectangle:

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.statistics import contour_longest_distance
    >>> rectangle = np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
    >>> round(contour_longest_distance([rectangle], exact=True), 2)
    22.36

    The same rectangle as a 3D contour lying on z = 5 mm:

    >>> rectangle_3d = np.c_[rectangle, np.full(4, 5.0)]
    >>> round(contour_longest_distance([rectangle_3d]), 2)
    22.36

    Largest value across a stack of slices:

    >>> small = rectangle * 0.5
    >>> round(contour_longest_distance([small, rectangle]), 2)
    22.36
    """
    best = 0.0
    for c in contours:
        c = np.asarray(c, dtype=float)
        if c.shape[1] == 3:
            centered = c - c.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            c = centered @ vt[:2].T

        poly = Polygon(c)
        if not poly.is_valid:
            poly = poly.buffer(0)

        pts = c if exact else c[ConvexHull(c).vertices]
        pairs = list(combinations(range(len(pts)), 2))
        dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        for k in np.argsort(dists)[::-1]:
            i, j = pairs[k]
            line = LineString([pts[i], pts[j]])
            if poly.contains(line) or poly.boundary.contains(line):
                best = max(best, dists[k])
                break

    return best


def mesh_longest_distance(mesh, mode='all', array_shape=None, origin=None, spacing=None,
                           direction=None, n_line_samples=20, exact_2d=False):
    """
    Longest internal distance of a closed mesh, in 3D or per slice plane.

    In '3d' mode, hull vertex pairs are tested from longest to shortest and
    the first segment whose sample points all lie inside the mesh is
    returned. In the slice modes, the mesh is cut at every image slice
    position along that axis and `contour_longest_distance` is run on each
    cut.

    Parameters
    ----------
    mesh : pv.PolyData
        Closed surface mesh, already in mm.

    mode : str
        '3d', 'axial', 'sagittal', 'coronal', or 'all'.

    array_shape : tuple[int, int, int] | None
        Image size in x-y-z order. Required for every mode except '3d'.

    origin : tuple[float, float, float] | None
        Image origin in mm (x-y-z). Required for every mode except '3d'.

    spacing : tuple[float, float, float] | None
        Voxel spacing in mm (x-y-z). Required for every mode except '3d'.

    direction : np.ndarray | None
        3x3 matrix whose columns are the world direction of each array axis:
        physical = origin + direction @ (spacing * index). Identity if None.

    n_line_samples : int
        Number of points sampled along a 3D candidate segment for the
        inside test.

    exact_2d : bool
        Passed to `contour_longest_distance` as `exact` for the slice modes.

    Returns
    -------
    float | dict[str, float]
        Longest distance in mm. For mode='all', a dictionary with keys
        '3d', 'axial', 'sagittal' and 'coronal'.

    Raises
    ------
    ValueError
        If the image geometry is missing for a slice mode, or `mode` is not
        recognized.

    Examples
    --------
    Longest 3D distance of a 10 mm radius sphere:

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.roi.statistics import mesh_longest_distance
    >>> mesh = pv.Sphere(radius=10.0)
    >>> round(mesh_longest_distance(mesh, mode='3d'), 1)
    20.0

    Longest axial distance, cutting at the real CT slice positions
    (geometry in x-y-z order):

    >>> axial = mesh_longest_distance(
    ...     mesh,
    ...     mode='axial',
    ...     array_shape=(32, 32, 16),
    ...     origin=(-16.0, -16.0, -16.0),
    ...     spacing=(1.0, 1.0, 2.0),
    ... )

    Every measurement at once:

    >>> result = mesh_longest_distance(
    ...     mesh,
    ...     mode='all',
    ...     array_shape=(32, 32, 16),
    ...     origin=(-16.0, -16.0, -16.0),
    ...     spacing=(1.0, 1.0, 2.0),
    ...     direction=np.eye(3),
    ... )
    >>> sorted(result.keys())
    ['3d', 'axial', 'coronal', 'sagittal']
    """
    def slice_plane_max(normal, point):
        sl = mesh.slice(normal=normal, origin=point).strip()
        lines, pts, i, best = sl.lines, sl.points, 0, 0.0
        while i < len(lines):
            n = lines[i]
            loop = pts[lines[i + 1:i + 1 + n]]
            i += n + 1
            if len(loop) >= 3:
                best = max(best, contour_longest_distance([loop], exact=exact_2d))

        return best

    def axis_scan(mode_idx):
        normal = direction[:, mode_idx]
        normal = normal / np.linalg.norm(normal)
        best = 0.0
        for k in range(array_shape[mode_idx]):
            idx = np.zeros(3)
            idx[mode_idx] = k
            point = direction @ (np.asarray(spacing) * idx) + np.asarray(origin)
            best = max(best, slice_plane_max(normal, point))

        return best

    def distance_3d():
        hull_pts = mesh.points[ConvexHull(mesh.points).vertices]
        pairs = list(combinations(range(len(hull_pts)), 2))
        dists = np.array([np.linalg.norm(hull_pts[i] - hull_pts[j]) for i, j in pairs])
        for k in np.argsort(dists)[::-1]:
            i, j = pairs[k]
            p1, p2 = hull_pts[i], hull_pts[j]
            t = np.linspace(0, 1, n_line_samples)
            seg = p1 + (p2 - p1) * t[:, None]
            cloud = pv.PolyData(seg)
            if hasattr(cloud, 'select_interior_points'):
                r = cloud.select_interior_points(mesh, method='cell_locator', locator_tolerance=1e-6)
                inside = r['selected_points'].all()
            else:
                r = cloud.select_enclosed_points(mesh, tolerance=1e-6)
                inside = r['SelectedPoints'].view(bool).all()
            if inside:
                return float(dists[k])

        return 0.0

    if mode == '3d':
        return distance_3d()

    if array_shape is None or origin is None or spacing is None:
        raise ValueError("array_shape, origin, and spacing are required for axial/sagittal/coronal/all")

    direction = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    axis_index = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    if mode in ('axial', 'sagittal', 'coronal'):
        return axis_scan(axis_index[mode])

    if mode == 'all':
        return {
            '3d': distance_3d(),
            'axial': axis_scan(2),
            'sagittal': axis_scan(0),
            'coronal': axis_scan(1),
        }
    raise ValueError("mode must be '3d', 'axial', 'sagittal', 'coronal', or 'all'")