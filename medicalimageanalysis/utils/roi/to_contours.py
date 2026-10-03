"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

to_contours.py - conversions that produce contours.

  mesh -> contours   mesh_to_contours(poly, origin, spacing, direction, positions=None)
  mask -> contours   mask_to_contours(mask, origin, spacing, direction)
  helper             contour_positions(contours, origin, direction)
"""
import numpy as np
import vtk
from vtk.util import numpy_support as nps

from ._raster import _cut, _flat, _loops


def contour_positions(contours, origin, direction):
    """Positions (mm along the image k axis from origin) of the planes a contour list lies on,
    e.g. to cut a mesh back at the originally contoured slices."""
    n, O = np.asarray(direction, float)[:, 2], np.asarray(origin, float)
    return np.array(sorted({round(float(((np.asarray(c, float) - O) @ n).mean()), 3)
                            for c in contours if len(c) >= 3}))


def mesh_to_contours(poly, origin, spacing, direction, positions=None, as_list=False):
    """Closed mesh -> contours, exact plane cuts normal to the image k axis.
    positions=None: every CT slice centre the mesh spans; otherwise mm along k from origin
    (e.g. contour_positions(original_contours, origin, direction)).
    Returns {slice_index: [(n,3) world rings]} (nearest CT slice), or a flat list of rings
    like contour_position with as_list=True."""
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


def mask_to_contours(mask, origin, spacing, direction, level=None, as_list=False):
    """Mask (nz, ny, nx) -> {slice_index: [(n,3) world rings]} by marching squares per slice.
    level=None: 127.5 for a uint8 0..255 mask (sub-pixel boundary), 0.5 for a binary mask.
    The mask is padded by one voxel so contours touching the image edge still close.
    as_list=True returns a flat list of rings like contour_position."""
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