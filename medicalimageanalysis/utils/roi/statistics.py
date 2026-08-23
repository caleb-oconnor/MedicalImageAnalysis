from itertools import combinations

import numpy as np
import pyvista as pv

from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon


def contour_longest_distance(contours, exact=False):
    """
    Longest in-plane distance across a stack of (N,2)/(N,3) contours, in mm.
    exact=False: only hull vertices considered (fast, can undercount on
    concave contours). exact=True: all points considered (slower, exact).
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
    mesh: pyvista mesh, already in mm.
    mode: '3d', 'axial', 'sagittal', 'coronal', or 'all' (returns each's own max).

    array_shape, origin, spacing, direction: all in xyz order (axis 0 = x/
    sagittal, axis 1 = y/coronal, axis 2 = z/axial). Required for any mode
    besides '3d' -- used to find the actual image slice positions to cut
    through. direction columns are the world-space direction of each array
    axis: physical = origin + direction @ (spacing * index).
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
