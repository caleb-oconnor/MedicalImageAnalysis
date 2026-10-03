"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

to_mesh.py - conversions that produce a surface mesh.

  contours -> mesh          contours_to_mesh(contours, origin, spacing, direction)
  contours -> mesh + mask   contours_to_mesh_and_mask(contours, shape_zyx, origin, spacing, direction)
"""
import numpy as np

from ._contourmesh import _finish, _to_polydata, postprocess
from ._contourmesh import contours_to_mesh_regions as _mesh_regions
from ._raster import _group, _region_volumes, volume_to_grid


def _empty_mesh():
    """Return empty point/triangle arrays with the expected shapes/dtypes."""
    return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.int64)


def contours_to_mesh(contours, origin, spacing, direction, reduction=0.5, as_polydata=True, **kw):
    """Contours (list of (n,3) world arrays) -> closed surface mesh (vtkPolyData).
    Each separate region gets its own h; decimated once at the end.
    kw: _contourmesh options (m, c, gap_factor, pad, workers, mid, flat_caps, cap_radius)."""
    slices, frame, dz = _group(contours, origin, spacing, direction)
    return _mesh_regions(slices, frame, dz, reduction=reduction, as_polydata=as_polydata, **kw)


def contours_to_mask_and_mesh(contours, shape_zyx, origin, spacing, direction,
                              reduction=0.5, ramp=None, as_polydata=True, **kw):
    """Contours -> (mesh, uint8 mask) from one field build per region."""
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