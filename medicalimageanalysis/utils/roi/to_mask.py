"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

to_mask.py - conversions that produce a uint8 0..255 mask (>= 128 inside).

  contours -> mask   contours_to_mask(contours, shape_zyx, origin, spacing, direction)
  contours -> mask   contours_to_mask_direct(contours, shape_zyx, origin, spacing, direction)
  mesh     -> mask   mesh_to_mask(poly, shape_zyx, origin, spacing, direction)

contours_to_mask samples the shape-interpolation field that Flying Edges meshes (no mesh is
built); contours_to_mask_direct fills contours that already sit on every CT slice;
mesh_to_mask cuts the mesh at the CT slice centres and uses the signed distance to the cuts.
"""
import numpy as np

from ._raster import _cut, _group, _region_volumes, rings_to_mask, volume_to_grid


def contours_to_mask(contours, shape_zyx, origin, spacing, direction, ramp=None, **kw):
    """Contours (list of (n,3) world arrays) -> uint8 mask (0..255) sampled from the
    shape-based-interpolation field. No mesh is built. Works when contours skip CT slices
    and for sagittal / coronal contours on an axial grid."""
    slices, frame, dz = _group(contours, origin, spacing, direction)
    mask = np.zeros(shape_zyx, np.uint8)
    for vol in _region_volumes(slices, frame, dz, **kw):
        volume_to_grid(vol, frame, shape_zyx, origin, spacing, direction, ramp, out=mask)
    return mask


def contours_to_mask_direct(contours, shape_zyx, origin, spacing, direction, ramp=None):
    """Contours drawn on the CT slices themselves -> uint8 mask (0..255), no field or mesh.
    Only valid when every CT slice inside the structure has its own contour."""
    D, O = np.asarray(direction, float), np.asarray(origin, float)
    by_k = {}
    for r in contours:
        r = np.asarray(r, float)
        k = int(np.rint(((r - O) @ D[:, 2]).mean() / spacing[2]))
        by_k.setdefault(k, []).append(r)
    return rings_to_mask(by_k, shape_zyx, origin, spacing, direction, ramp)


def mesh_to_mask(poly, shape_zyx, origin, spacing, direction, ramp=None):
    """Closed mesh -> uint8 mask (0..255): cut at CT slice centres, signed distance to the cuts."""
    rc = _cut(poly, origin, direction, np.arange(shape_zyx[0]) * spacing[2])  # key = slice index
    return rings_to_mask(rc, shape_zyx, origin, spacing, direction, ramp)