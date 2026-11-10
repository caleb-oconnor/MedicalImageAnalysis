"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Deformable registration between the frames of reference of a reference and a moving image: the moving image (e.g.
    CT 2) is deformed onto the reference (e.g. CT 1). Computing the displacement vector field (DVF) with SimpleITK /
    Elastix or ANTs through one call (:meth:`Deformable.compute_registration`), display reslicing of any image in
    either frame deformed onto any image in the other, display of the DVF itself, warping ROI meshes, and a
    single-file save. The rigid pre-alignment comes from outside (e.g. a Rigid).

    Conventions
    -----------
    * Every image array is (z, y, x), including 2D X-ray (z = 1) and ultrasound frames. Origin, spacing and world
      points are (x, y, z). ``image.matrix`` rows are the image axes.
    * ``dvf`` is (z, y, x, 3), ordered (dx, dy, dz) in mm, identity direction, on (``origin``, ``spacing``).
    * ``rigid_matrix`` maps a reference-world point to the matching moving-world point, the same convention as
      ``Rigid.get_combined_matrix()``. The "rigid frame" is the moving image carried into the reference world by it.
    * The moving image is deformed onto the reference. There is ONE stored field, ``dvf``, and its vectors run from a
      reference point to the matching moving point, because that is what pulls the moving image onto the reference
      grid. For a point r in the reference world, the matching point of the moving image is

          m = rigid_matrix @ (r + p * dvf(r)),          p = percent / 100

      ``dvf`` lives on a grid in the reference world (normally the reference image's grid). It is exactly what
      SimpleITK, Elastix, ANTs and a DICOM Deformable Spatial Registration give with the reference as the fixed /
      registered image, so it is stored as is.
    * Every operation uses that relation. When r is known it is one lookup ("direct"); when only m is known, r is
      solved from  r + p * dvf(r) = rigid^-1 @ m  by the fixed-point iteration  r <- a - p * dvf(r)  ("solve"; a few
      lookups for any smooth, fold-free field). The solve runs only for the points actually needed (the pixels on
      screen, the mesh vertices), so no inverted field is ever stored.

          moving-side image on a reference-side view (the usual view)    pixel r          -> direct
          reference-side image on a moving-side view                     pixel m          -> solve
          reference-side ROI -> moving world                             vertex r         -> direct
          moving-side ROI -> reference world (contours, 3D)              vertex m         -> solve

    * Exactness: the direct cases are exact. The solve converges wherever the field does not fold; where it folds
      (not diffeomorphic) no inverse exists, so those points (and contours through them) are only approximate.
      :meth:`Deformable.compute_folding` measures how much of the field folds.
    * A full inverted field is only built when the user asks for one (:meth:`Deformable.compute_inverted_dvf`).
      It is returned, never stored.
    * Display calls name their own images, like Rigid: ``base_name`` is the image whose view is shown, ``image_name``
      the image deformed onto it. Each view is its own call, so any number of views (in either direction) can share
      one registration. Leaving both out shows the moving image deformed onto the reference image.
    * Any image in either frame can be displayed (e.g. an RTDose in the moving image's frame shown deformed on the
      reference). Display paths look images up with ``Data.volume`` (image or dose); registration and ROI warping use
      ``Data.image`` only. An image with no usable Frame of Reference UID is its own frame. Two images on the same side
      are in the same world and are shown with no deformation.
    * Display follows the same contract as Image.Display / Rigid: the caller passes (plane, position), where position
      is the widget's position on the base image's view. The output grid comes from the base image's Display (so its
      shared oblique rotation, wheel scrolling and crosshair all just work), and the deformed image is sampled
      straight onto that grid in one fused numba pass (no full-volume resample on screen).
    * ``spacing`` works as in Rigid: None = base image grid, 'native' = the displayed image's spacing, or any spacing.
      Use the same value in :meth:`Deformable.compute_array`, :meth:`Deformable.compute_grid`,
      :meth:`Deformable.compute_mesh_slice` and :meth:`Deformable.compute_aspect` so image, field and contours line up.
    * ``percent`` scales the displacement: 0 = rigid only, 100 = full deformation, the same way in every direction.
      The default lives in ``visual.percent`` (set it with :meth:`Deformable.set_percent`).
    * ROIs: :meth:`Deformable.compute_mesh_slice` gives 2D contours for the views; ``roi`` (built by
      :meth:`Deformable.update_rois`) holds the moving ROIs deformed onto the reference for the 3D view.
    * :meth:`Deformable.save` writes one .npz: the JSON state (names, frames, roi_names, rigid matrix, DVF grid), the
      DVF and the reference / moving meshes, so the meshes stay viewable if the images are deleted. Warped ROI
      meshes, display settings and caches are never saved; they are rebuilt on demand.

Structure:
    - _deformable_sample_kernel: numba kernel, deformed image sampled onto a view grid.
    - _field_component_kernel: numba kernel, one DVF component or its magnitude sampled onto a view grid.
    - _field_corner: numba helper, clamped trilinear lookup into the DVF.
    - _solve_field: numba helper, solves r + ratio * dvf(r) = a for one point (the "solve" case above).
    - DeformableSide: image name, frame and registration surface for one side. Built by Deformable.
    - DeformableVisual: display settings (opacity, DVF overlay, default deformation percent).
    - Deformable: DVF, rigid pre-alignment, registration (one call for SimpleITK / Elastix / ANTs), display, ROI
      warping and saving.

Examples
--------
>>> import medicalimageanalysis as mia
>>> rigid = mia.Rigid('CT 01', 'CBCT 01')
>>> rigid.compute_registration('itk', metric='MI')
>>> deform = mia.Deformable(reference_name='CT 01', moving_name='CBCT 01', roi_names=['Liver'],
...                         rigid_matrix=rigid.get_combined_matrix())
>>> deform.compute_registration('itk_fast_demons')
>>> deform.compute_registration('ants', transform='SyN', metric='CC', label_weight=1.0)
>>> deform.deformable_name
'DVF_CT 01_CBCT 01'

One registration, several views (each its own call)

>>> res = deform.compute_array('Axial', ct_position)                                          # CBCT on the CT
>>> res = deform.compute_array('Axial', cbct_position, base_name='CBCT 01', image_name='CT 01')  # CT on the CBCT
>>> res = deform.compute_array('Axial', ct_position, image_name='Dose 01')     # dose in the CBCT frame, on the CT
>>> contours = deform.compute_mesh_slice('Liver', 'Axial', ct_position)
>>> magnitude = deform.compute_grid('Axial', ct_position, vector='magnitude')                 # DVF overlay in mm
>>> mia.Deformable.find_deformables('Dose 01', 'CT 01')
['DVF_CT 01_CBCT 01']

Step animation (slider from rigid only to full deformation)

>>> deform.set_percent(0)       # rigid only
>>> deform.set_percent(100)     # full deformation
>>> res = deform.compute_array('Axial', ct_position)
>>> deform.update_rois()        # refresh the 3D meshes (deform.roi) at the new percent

How trustworthy the solved direction is

>>> deform.compute_folding()                # fraction of DVF voxels that fold, 0.0 = invertible everywhere

Existing DVF (e.g. from a DICOM deformable registration) with a non-identity direction

>>> deform = mia.Deformable(dvf=dvf, origin=origin, spacing=spacing, dimensions=dvf.shape[:3], dvf_matrix=direction,
...                         reference_name='CT 01', moving_name='CBCT 01', rigid_matrix=pre_registration)
"""

import os
import copy
import json

from collections import OrderedDict
from dataclasses import dataclass

import vtk
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from numba import njit, prange
from vtk.util import numpy_support
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

from ..data import Data
from ..utils.mesh.surface import compute_components
from ..utils.deformable.simpleitk import DeformableITK


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _deformable_sample_kernel(volume, field, h, w, sx, sy, origin, x_axis, y_axis,
                              pre, field_origin, field_spacing, ratio, post_index, background, solve, iterations,
                              tolerance):
    """
    One pass per output pixel:
        w      = origin + u * x_axis + v * y_axis                (output-view world point)
        a      = pre @ [w, 1]                                     (into the field's frame)
        q      = a + ratio * field(a)                             (solve=False: a is a reference point, direct)
               = r with r + ratio * field(r) = a                  (solve=True: a is a moving point, r solved)
        index  = post_index @ [q, 1]                              (into the sampled volume's (x, y, z) voxel index;
                                                                    rigid, direction and spacing are all folded in)
    then trilinear sample of the volume. Pixels that land outside the volume get `background`.

    An axis with a single slice (2D X-ray, one US frame) has no thickness, so it accepts +/- half a voxel and snaps
    onto the slice (same as Image.Display's kernel).
    """
    out = np.empty((h, w), dtype=np.float32)
    nz, ny, nx = volume.shape

    tz = 0.5 if nz == 1 else 0.0
    ty = 0.5 if ny == 1 else 0.0
    tx = 0.5 if nx == 1 else 0.0

    for i in prange(h):
        v = i * sy
        for j in range(w):
            u = j * sx
            wx = origin[0] + u * x_axis[0] + v * y_axis[0]
            wy = origin[1] + u * x_axis[1] + v * y_axis[1]
            wz = origin[2] + u * x_axis[2] + v * y_axis[2]

            ax = pre[0, 0] * wx + pre[0, 1] * wy + pre[0, 2] * wz + pre[0, 3]
            ay = pre[1, 0] * wx + pre[1, 1] * wy + pre[1, 2] * wz + pre[1, 3]
            az = pre[2, 0] * wx + pre[2, 1] * wy + pre[2, 2] * wz + pre[2, 3]

            if solve:
                qx, qy, qz = _solve_field(field, ax, ay, az, field_origin, field_spacing, ratio, iterations,
                                          tolerance)
            else:
                d0, d1, d2 = _field_corner(field,
                                           (ax - field_origin[0]) / field_spacing[0],
                                           (ay - field_origin[1]) / field_spacing[1],
                                           (az - field_origin[2]) / field_spacing[2])
                qx = ax + ratio * d0
                qy = ay + ratio * d1
                qz = az + ratio * d2

            ii = post_index[0, 0] * qx + post_index[0, 1] * qy + post_index[0, 2] * qz + post_index[0, 3]
            jj = post_index[1, 0] * qx + post_index[1, 1] * qy + post_index[1, 2] * qz + post_index[1, 3]
            kk = post_index[2, 0] * qx + post_index[2, 1] * qy + post_index[2, 2] * qz + post_index[2, 3]

            if (kk < -tz or kk > nz - 1 + tz or jj < -ty or jj > ny - 1 + ty or
                    ii < -tx or ii > nx - 1 + tx):
                out[i, j] = background
                continue

            # no-op for normal axes; snaps a single-slice axis onto its slice
            kk = min(max(kk, 0.0), nz - 1.0)
            jj = min(max(jj, 0.0), ny - 1.0)
            ii = min(max(ii, 0.0), nx - 1.0)

            # max(..., 0) and min(i0 + 1, n - 1) only matter for single-slice axes
            z0 = max(min(int(kk), nz - 2), 0)
            y0 = max(min(int(jj), ny - 2), 0)
            x0 = max(min(int(ii), nx - 2), 0)
            z1, y1, x1 = min(z0 + 1, nz - 1), min(y0 + 1, ny - 1), min(x0 + 1, nx - 1)
            fz, fy, fx = kk - z0, jj - y0, ii - x0

            c00 = volume[z0, y0, x0] * (1 - fx) + volume[z0, y0, x1] * fx
            c01 = volume[z0, y1, x0] * (1 - fx) + volume[z0, y1, x1] * fx
            c10 = volume[z1, y0, x0] * (1 - fx) + volume[z1, y0, x1] * fx
            c11 = volume[z1, y1, x0] * (1 - fx) + volume[z1, y1, x1] * fx
            c0 = c00 * (1 - fy) + c01 * fy
            c1 = c10 * (1 - fy) + c11 * fy
            out[i, j] = c0 * (1 - fz) + c1 * fz

    return out


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _field_component_kernel(field, h, w, sx, sy, origin, x_axis, y_axis, pre, field_origin, field_spacing,
                            component, outside, solve, iterations, tolerance):
    """
    Samples one DVF component (0=x, 1=y, 2=z) or its magnitude (3) onto the output view, always at the reference point
    r of the tissue shown at the pixel: r = pre @ w (solve=False), or the r with r + field(r) = pre @ w (solve=True).
    Pixels whose r is outside the field's grid get `outside` instead of a clamped edge value, so the field's footprint
    is visible.
    """
    out = np.empty((h, w), dtype=np.float32)
    fnz, fny, fnx = field.shape[0], field.shape[1], field.shape[2]

    tz = 0.5 if fnz == 1 else 0.0
    ty = 0.5 if fny == 1 else 0.0
    tx = 0.5 if fnx == 1 else 0.0

    for i in prange(h):
        v = i * sy
        for j in range(w):
            u = j * sx
            wx = origin[0] + u * x_axis[0] + v * y_axis[0]
            wy = origin[1] + u * x_axis[1] + v * y_axis[1]
            wz = origin[2] + u * x_axis[2] + v * y_axis[2]

            ax = pre[0, 0] * wx + pre[0, 1] * wy + pre[0, 2] * wz + pre[0, 3]
            ay = pre[1, 0] * wx + pre[1, 1] * wy + pre[1, 2] * wz + pre[1, 3]
            az = pre[2, 0] * wx + pre[2, 1] * wy + pre[2, 2] * wz + pre[2, 3]
            if solve:
                ax, ay, az = _solve_field(field, ax, ay, az, field_origin, field_spacing, 1.0, iterations, tolerance)

            fi = (ax - field_origin[0]) / field_spacing[0]
            fj = (ay - field_origin[1]) / field_spacing[1]
            fk = (az - field_origin[2]) / field_spacing[2]
            if (fk < -tz or fk > fnz - 1 + tz or fj < -ty or fj > fny - 1 + ty or
                    fi < -tx or fi > fnx - 1 + tx):
                out[i, j] = outside
                continue

            d0, d1, d2 = _field_corner(field, fi, fj, fk)
            if component == 0:
                out[i, j] = d0
            elif component == 1:
                out[i, j] = d1
            elif component == 2:
                out[i, j] = d2
            else:
                out[i, j] = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

    return out

@njit(inline='always', fastmath=True, nogil=True, cache=True)
def _field_corner(field, fi, fj, fk):
    """
    Clamped trilinear lookup into a (Z, Y, X, 3) field at continuous index (fi=x, fj=y, fk=z). Clamping matches the
    mode='nearest' behaviour the ROI warping uses, so meshes and image stay consistent at the field edges. A
    single-slice axis is handled (no neighbor past the last slice).
    """
    fnz, fny, fnx = field.shape[0], field.shape[1], field.shape[2]
    fi = min(max(fi, 0.0), fnx - 1.0)
    fj = min(max(fj, 0.0), fny - 1.0)
    fk = min(max(fk, 0.0), fnz - 1.0)

    x0 = max(min(int(fi), fnx - 2), 0)
    y0 = max(min(int(fj), fny - 2), 0)
    z0 = max(min(int(fk), fnz - 2), 0)
    x1, y1, z1 = min(x0 + 1, fnx - 1), min(y0 + 1, fny - 1), min(z0 + 1, fnz - 1)
    tx, ty, tz = fi - x0, fj - y0, fk - z0

    d0 = 0.0
    d1 = 0.0
    d2 = 0.0
    for c in range(3):
        c00 = field[z0, y0, x0, c] * (1 - tx) + field[z0, y0, x1, c] * tx
        c01 = field[z0, y1, x0, c] * (1 - tx) + field[z0, y1, x1, c] * tx
        c10 = field[z1, y0, x0, c] * (1 - tx) + field[z1, y0, x1, c] * tx
        c11 = field[z1, y1, x0, c] * (1 - tx) + field[z1, y1, x1, c] * tx
        v = (c00 * (1 - ty) + c01 * ty) * (1 - tz) + (c10 * (1 - ty) + c11 * ty) * tz
        if c == 0:
            d0 = v
        elif c == 1:
            d1 = v
        else:
            d2 = v

    return d0, d1, d2


@njit(inline='always', fastmath=True, nogil=True, cache=True)
def _solve_field(field, wx, wy, wz, field_origin, field_spacing, ratio, iterations, tolerance):
    """
    Reference point r with  r + ratio * field(r) = w  for a known (rigid-frame) moving point w, by the fixed-point
    iteration  r <- w - ratio * field(r), stopping when r moves less than `tolerance` (mm).
    """
    qx, qy, qz = wx, wy, wz
    if ratio == 0.0:
        return qx, qy, qz

    for _ in range(iterations):
        d0, d1, d2 = _field_corner(field,
                                   (qx - field_origin[0]) / field_spacing[0],
                                   (qy - field_origin[1]) / field_spacing[1],
                                   (qz - field_origin[2]) / field_spacing[2])
        nx = wx - ratio * d0
        ny = wy - ratio * d1
        nz = wz - ratio * d2
        change = abs(nx - qx) + abs(ny - qy) + abs(nz - qz)
        qx, qy, qz = nx, ny, nz
        if change < tolerance:
            break

    return qx, qy, qz


@dataclass
class DeformableSide:
    """One side (reference or moving) of a deformable registration. Built by :class:`Deformable`, not created directly.

    Attributes
    ----------
    image_name : str or None
        The image this side was registered with (key into ``Data.image``).
    frame : str or None
        Frame of Reference UID shared by every image on this side; the image name when the image has none.
    mesh : pyvista.PolyData, optional
        Surface used for the registration on this side, if any. Saved, so it stays viewable if the image is deleted.
    """
    image_name: str | None
    frame: str | None = None
    mesh: object = None


@dataclass
class DeformableVisual:
    """Display settings for a deformable registration.

    Attributes
    ----------
    opacity : float
        Blend opacity of the deformed image.
    vector : str or None
        DVF overlay shown: None, 'x', 'y', 'z' or 'magnitude' (see :meth:`Deformable.compute_grid`).
    percent : float
        Deformation level used when a call passes no percent: 0 = rigid only, 100 = full. Set it with
        :meth:`Deformable.set_percent` so stale warped meshes are dropped.
    """
    opacity: float = 0.5
    vector: object = None
    percent: float = 100.0


class Deformable(object):
    """Deformable registration between the frames of reference of a reference and a moving image: the moving image is
    deformed onto the reference by a DVF on top of a rigid pre-alignment.

    The display API mirrors Rigid so the same viewer code can drive either: :meth:`compute_array`,
    :meth:`compute_mesh_slice`, :meth:`compute_aspect` and :meth:`get_view_center` take (plane, position) of the base
    image's view and name their images with ``base_name`` / ``image_name``. :meth:`compute_grid` adds the DVF itself
    on the same grid.

    Parameters
    ----------
    dvf : (Z, Y, X, 3) ndarray, optional
        Reference -> moving displacement field, (dx, dy, dz) in mm, on a grid in the reference world (see the module
        conventions). None until :meth:`compute_registration`.
    origin, spacing : array_like, optional
        DVF grid origin and spacing, (x, y, z).
    dimensions : tuple, optional
        DVF grid dimensions, (z, y, x).
    roi_names : list of str, optional
        Default ROIs whose masks guide :meth:`compute_registration` (must exist on both images).
    rigid_matrix : (4, 4) array_like, optional
        Rigid pre-alignment, reference world -> moving world (``Rigid.get_combined_matrix()``). Defaults to identity.
    dvf_matrix : (3, 3) array_like, optional
        Direction of the supplied DVF, rows = index axes in world (same as ``image.matrix``). Anything other than
        identity is resampled onto an identity grid with :meth:`correct_dvf_direction`.
    registration_name : str, optional
        Lookup key in ``Data.deformable``. None builds one from the image names.
    reference_name, moving_name : str
        Images in ``Data.image`` used to compute the registration: the moving image is deformed onto the reference.
        Their frames define what the registration applies to.
    reference_meshes, moving_meshes : pyvista.PolyData, optional
        Surfaces used for registration (stored as ``reference.mesh`` / ``moving.mesh``).
    local_uid : str, optional
        Existing uid when loading (:meth:`load`). None generates a new one.

    Attributes
    ----------
    reference, moving : DeformableSide
        Image name, frame and registration surface of each side.
    visual : DeformableVisual
        Display settings, including the default deformation ``percent``.
    roi : dict
        Moving ROI meshes deformed into the reference world at ``visual.percent``, for the 3D view (built by
        :meth:`update_rois`). The 2D views use :meth:`compute_mesh_slice` instead.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> deform = mia.Deformable(reference_name='CT 01', moving_name='CBCT 01', roi_names=['Liver'],
    ...                         rigid_matrix=mia.Data.rigid['CT 01_CBCT 01'].get_combined_matrix())
    >>> deform.compute_registration('itk_diffeomorphic')
    >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
    """
    SCHEMA_VERSION = 1

    # compute_registration method name -> DeformableITK method, for the image-based SimpleITK / Elastix methods
    ITK_METHODS = {'itk_bspline': 'bspline', 'itk_field': 'displacement_field', 'itk_demons': 'demons',
                   'itk_symmetric_demons': 'symmetric_demons', 'itk_fast_demons': 'fast_demons',
                   'itk_diffeomorphic': 'diffeomorphic', 'itk_level_set': 'level_set_motion', 'elastix': 'elastix'}

    # "solve" case (r + p * dvf(r) = a for a known moving point): maximum iterations and stopping change in mm
    INVERT_ITERATIONS = 10
    INVERT_TOLERANCE = 0.01

    def __init__(self, dvf=None, origin=None, spacing=None, dimensions=None, roi_names=None,
                 rigid_matrix=None, dvf_matrix=None, registration_name=None, reference_name=None, moving_name=None,
                 reference_meshes=None, moving_meshes=None, local_uid=None):
        self.reference = DeformableSide(reference_name, self._frame_of(reference_name), reference_meshes)
        self.moving = DeformableSide(moving_name, self._frame_of(moving_name), moving_meshes)
        self.roi_names = roi_names
        self.roi = dict.fromkeys(Data.roi_list())
        self.local_uid = generate_uid() if local_uid is None else local_uid

        if dvf_matrix is not None and not np.allclose(dvf_matrix, np.identity(3), atol=1e-3):
            self.dvf, self.spacing, self.origin, self.dimensions = self.correct_dvf_direction(dvf, spacing, origin,
                                                                                              dvf_matrix)
        else:
            self.dvf = dvf
            self.origin = origin
            self.spacing = spacing
            self.dimensions = dimensions

        # float32, C-contiguous: the numba kernels use it as is (no second copy), half the memory of float64
        if self.dvf is not None:
            self.dvf = np.ascontiguousarray(self.dvf, dtype=np.float32)

        self.rigid_matrix = np.identity(4) if rigid_matrix is None else np.asarray(rigid_matrix, dtype=float)

        self.visual = DeformableVisual()

        # Everything display-side that is derived from dvf / rigid_matrix lives here, so it is never saved and it
        # rebuilds itself when dvf is replaced (checked by identity, not by a manual flag).
        self._display_cache = {}
        self.mesh_slice_cache_size = 2000  # per view, all ROIs together; least-recently-used entries dropped first

        self.deformable_name = self.add_deformable(registration_name)

        # the moving image may not be loaded (or was deleted) when a saved registration is loaded
        if self.dvf is not None and self.moving.image_name in Data.image:
            self.update_rois()

    @staticmethod
    def _apply_affine(matrix, points):
        """4x4 affine applied to (N, 3) points."""
        points = np.asarray(points, dtype=float)
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    @staticmethod
    def _frame_of(image_name):
        """Frame of Reference UID of an image, or its own name when the UID is missing or a placeholder.

        A placeholder (e.g. '00000.00000' from Image) must never make unrelated images share a frame.
        """
        try:
            volume = Data.volume(image_name)
        except KeyError:
            volume = None

        frame = None if volume is None else getattr(volume, 'frame_ref', None)
        if frame is None or not str(frame).strip(' .0'):
            return image_name

        return str(frame)

    @staticmethod
    def _invert_field(field, origin, spacing, iterations=50, mean_tolerance=1e-3, max_tolerance=1e-1):
        """Invert a (Z, Y, X, 3) identity-direction field with SimpleITK; same shape and (dx, dy, dz) ordering.

        Returned as a float32, C-contiguous array. The float64 copies SimpleITK needs only exist during the call.
        """
        # InvertDisplacementFieldImageFilter only accepts a float64 vector image
        field_sitk = sitk.GetImageFromArray(np.asarray(field, dtype=np.float64), isVector=True)
        field_sitk.SetSpacing([float(v) for v in spacing])
        field_sitk.SetOrigin([float(v) for v in origin])

        inverter = sitk.InvertDisplacementFieldImageFilter()
        inverter.SetMaximumNumberOfIterations(iterations)
        inverter.SetMeanErrorToleranceThreshold(mean_tolerance)
        inverter.SetMaxErrorToleranceThreshold(max_tolerance)

        inverted = inverter.Execute(field_sitk)
        del field_sitk      # free the float64 input before the float64 output is converted

        return np.ascontiguousarray(sitk.GetArrayViewFromImage(inverted), dtype=np.float32)

    @staticmethod
    def _mask_image(mask, like):
        """(z, y, x) mask array to a binary UInt8 SimpleITK image with the geometry of `like`."""
        image = sitk.GetImageFromArray((np.asarray(mask) != 0).astype(np.uint8))
        image.CopyInformation(like)
        return image

    @staticmethod
    def _rigid_inv(mat):
        """Exact inverse of a rigid 4x4 (avoids drift from repeated np.linalg.inv)."""
        inv = np.identity(4)
        inv[:3, :3] = mat[:3, :3].T
        inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
        return inv

    @staticmethod
    def correct_dvf_direction(dvf, spacing, origin, matrix):
        """Resample a DVF stored on a rotated grid onto an identity-direction grid covering the same region.

        The new grid keeps the spacing and is the axis-aligned bounding box of the old grid's voxel centers. Each new
        voxel takes the trilinearly interpolated displacement at its world point; voxels the old grid doesn't cover get
        zero displacement. The vectors themselves are not rotated: DICOM, SimpleITK and ANTs store displacements in
        world (patient) coordinates, independent of the grid direction.

        Called by ``__init__`` when ``dvf_matrix`` is not identity, and by :meth:`compute_registration` when the
        reference image is oblique.

        Parameters
        ----------
        dvf : (Z, Y, X, 3) ndarray
        spacing : array_like
            (x, y, z).
        origin : array_like
            (x, y, z).
        matrix : (3, 3) array_like
            Direction of the input DVF, rows = index axes in world (same as ``image.matrix``):
            world = origin + matrix.T @ (index * spacing).

        Returns
        -------
        tuple
            (dvf, spacing, origin, dimensions) for the identity-direction grid.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> dvf, spacing, origin, dimensions = mia.Deformable.correct_dvf_direction(dvf, spacing, origin, direction)
        """
        dvf = np.asarray(dvf)
        matrix = np.asarray(matrix, dtype=float)
        spacing = np.asarray(spacing, dtype=float)
        origin = np.asarray(origin, dtype=float)

        nz, ny, nx = dvf.shape[:3]
        corner_index = np.array([[i, j, k] for i in (0, nx - 1) for j in (0, ny - 1) for k in (0, nz - 1)], dtype=float)
        corners = origin + (corner_index * spacing) @ matrix                 # rows = axes
        low, high = corners.min(axis=0), corners.max(axis=0)
        size = np.floor((high - low) / spacing + 1e-6).astype(int) + 1        # (x, y, z)

        xs = low[0] + np.arange(size[0]) * spacing[0]
        ys = low[1] + np.arange(size[1]) * spacing[1]
        zs = low[2] + np.arange(size[2]) * spacing[2]
        grid_x, grid_y = np.meshgrid(xs, ys)                                    # (y, x)

        dvf_new = np.zeros((size[2], size[1], size[0], 3), dtype=dvf.dtype)
        for k, z in enumerate(zs):        # one z slab at a time keeps memory to a single slice of coordinates
            rel = np.stack((grid_x - origin[0], grid_y - origin[1], np.full_like(grid_x, z - origin[2])), axis=-1)
            index = (rel @ matrix.T) / spacing             # (y, x, 3) fractional (x, y, z) index; orthonormal rows
            coords = [index[..., 2], index[..., 1], index[..., 0]]
            for c in range(3):
                dvf_new[k, ..., c] = map_coordinates(dvf[..., c], coords, order=1, mode='constant', cval=0.0)

        return dvf_new, tuple(float(v) for v in spacing), low, dvf_new.shape[:3]

    @staticmethod
    def find_deformables(image_a, image_b):
        """Names of every registration in ``Data.deformable`` that connects the frames of two images (either
        direction).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Deformable.find_deformables('Dose 01', 'CT 01')
        ['DVF_CT 01_CBCT 01']
        """
        return [name for name, deform in Data.deformable.items() if deform.connects(image_a, image_b)]

    @classmethod
    def load(cls, path):
        """Rebuild a registration from :meth:`save` output and register it in ``Data.deformable``.

        Works before (or without) its images being loaded: the saved frames are kept, the reference and moving meshes
        come from the file, and the warped ROIs (``roi``) are only built when the moving image is present.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Deformable.load('C:/Data/ct_cbct.npz')
        >>> deform.reference.mesh                   # still viewable if 'CT 01' was deleted
        """
        with np.load(path, allow_pickle=False) as data:
            record = json.loads(str(data['record']))
            if record.get('schema', 1) > cls.SCHEMA_VERSION:
                raise ValueError(f"Record schema {record['schema']} is newer than supported ({cls.SCHEMA_VERSION}).")

            dvf = data['dvf'] if 'dvf' in data else None
            meshes = {}
            for side in ('reference', 'moving'):
                meshes[side] = None
                if side + '_points' in data:
                    faces = data[side + '_faces']
                    meshes[side] = pv.PolyData(data[side + '_points'], faces if faces.size else None)

        deform = cls(dvf=dvf, origin=record['origin'], spacing=record['spacing'],
                     dimensions=None if dvf is None else dvf.shape[:3], roi_names=record['roi_names'],
                     rigid_matrix=np.reshape(record['rigid_matrix'], (4, 4)), registration_name=record['name'],
                     reference_name=record['reference']['image_name'], moving_name=record['moving']['image_name'],
                     reference_meshes=meshes['reference'], moving_meshes=meshes['moving'], local_uid=record['uid'])

        # the images may not be loaded, so keep the saved frames instead of the placeholders __init__ derived
        deform.reference.frame = record['reference'].get('frame', deform.reference.frame)
        deform.moving.frame = record['moving'].get('frame', deform.moving.frame)

        return deform

    def _collect_masks(self, reference_rois=None, moving_rois=None):
        """(reference mask, moving mask) as (z, y, x) bool arrays on each image's own grid: the union of the ROI pairs
        (paired by position) that have a shape on both images, or (None, None) when there are none.

        reference_rois None = ``roi_names``; moving_rois None = the same names as reference_rois.
        """
        reference_rois = self.roi_names if reference_rois is None else reference_rois
        if reference_rois is None:
            return None, None

        reference_rois = [reference_rois] if isinstance(reference_rois, str) else list(reference_rois)
        if moving_rois is None:
            moving_rois = list(reference_rois)
        else:
            moving_rois = [moving_rois] if isinstance(moving_rois, str) else list(moving_rois)

        if len(reference_rois) != len(moving_rois):
            raise ValueError(f"reference_rois ({len(reference_rois)}) and moving_rois ({len(moving_rois)}) must "
                             f"pair one-to-one.")

        ref_image = Data.image[self.reference.image_name]
        mov_image = Data.image[self.moving.image_name]

        ref_mask = None
        mov_mask = None
        for ref_name, mov_name in zip(reference_rois, moving_rois):
            ref_roi = ref_image.roi.get(ref_name)
            mov_roi = mov_image.roi.get(mov_name)
            if ref_roi is None or (ref_roi.mesh is None and ref_roi.contour_pixel is None):
                continue
            if mov_roi is None or (mov_roi.mesh is None and mov_roi.contour_pixel is None):
                continue

            ref_part = np.asarray(ref_roi.compute_mask()) != 0
            mov_part = np.asarray(mov_roi.compute_mask()) != 0
            ref_mask = ref_part if ref_mask is None else ref_mask | ref_part
            mov_mask = mov_part if mov_mask is None else mov_mask | mov_part

        return ref_mask, mov_mask

    def _field_grid(self):
        """DVF origin and spacing as float32 arrays for the numba kernels."""
        return np.asarray(self.origin, dtype=np.float32), np.asarray(self.spacing, dtype=np.float32)

    def _get_field(self):
        """
        The DVF as float32, C-contiguous for the numba kernels. ``dvf`` is already stored that way, so this is the same
        array (no copy) unless dvf was replaced with another dtype from outside; then a converted copy is cached and
        rebuilt when dvf is replaced again.
        """
        if self.dvf is None:
            return None

        cached = self._display_cache.get('field')
        if cached is None or cached[0] is not self.dvf:
            cached = (self.dvf, np.ascontiguousarray(self.dvf, dtype=np.float32))
            self._display_cache['field'] = cached

        return cached[1]

    def _record(self):
        """Registration state as plain JSON types. The DVF and the meshes are stored as arrays next to it by save."""
        return {'schema': self.SCHEMA_VERSION, 'uid': self.local_uid, 'name': self.deformable_name,
                'reference': {'image_name': self.reference.image_name, 'frame': self.reference.frame},
                'moving': {'image_name': self.moving.image_name, 'frame': self.moving.frame},
                'roi_names': None if self.roi_names is None else list(self.roi_names),
                'rigid_matrix': np.asarray(self.rigid_matrix, dtype=float).ravel().tolist(),
                'origin': None if self.origin is None else [float(v) for v in self.origin],
                'spacing': None if self.spacing is None else [float(v) for v in self.spacing]}

    def _registration_images(self, reference_rois=None, moving_rois=None, reference_mask=None, moving_mask=None,
                             images=True):
        """(reference, moving, reference_mask, moving_mask) SimpleITK images for a registration, all on the reference
        grid.

        moving is the moving image (and moving_mask its mask) resampled into the rigid frame through ``rigid_matrix``,
        so the backends only have to find the deformation. Explicit masks ((z, y, x) on each image's own grid) take
        priority over ROI masks. images=False returns only the reference image (its grid is all a landmark fit needs).
        """
        ref = Data.image[self.reference.image_name]
        fixed = sitk.Cast(ref.create_sitk_image(), sitk.sitkFloat32)
        if not images:
            return fixed, None, None, None

        mov = Data.image[self.moving.image_name]
        mov_native = sitk.Cast(mov.create_sitk_image(), sitk.sitkFloat32)

        # sitk transforms map output (reference) points to input (moving) points: exactly rigid_matrix.
        # AffineTransform skips the orthogonality check that trips Euler3DTransform on accumulated float drift.
        rigid = sitk.AffineTransform(3)
        rigid.SetMatrix(np.asarray(self.rigid_matrix[:3, :3], dtype=float).ravel().tolist())
        rigid.SetTranslation(np.asarray(self.rigid_matrix[:3, 3], dtype=float).tolist())

        # Outside the moving field of view reads as the moving image's lowest value (its air / padding) instead of 0
        background = float(np.min(sitk.GetArrayViewFromImage(mov_native)))
        moving = sitk.Resample(mov_native, fixed, rigid, sitk.sitkLinear, background, sitk.sitkFloat32)

        if reference_mask is None and moving_mask is None:
            reference_mask, moving_mask = self._collect_masks(reference_rois, moving_rois)

        fixed_mask = None if reference_mask is None else self._mask_image(reference_mask, fixed)
        moving_mask_image = None
        if moving_mask is not None:
            moving_mask_image = sitk.Resample(self._mask_image(moving_mask, mov_native), fixed, rigid,
                                              sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

        return fixed, moving, fixed_mask, moving_mask_image

    def _sample_field_points(self, field, points):
        """Clamped trilinear lookup of a (Z, Y, X, 3) field at world points (N, 3) -> (N, 3) displacements."""
        voxel = (np.asarray(points, dtype=float) - np.asarray(self.origin)) / np.asarray(self.spacing)
        coords = [voxel[:, 2], voxel[:, 1], voxel[:, 0]]
        disp = np.empty((voxel.shape[0], 3))
        for i in range(3):
            disp[:, i] = map_coordinates(field[..., i], coords, order=1, mode='nearest')

        return disp

    def _sampling_setup(self, image_name, to_reference):
        """
        (volume, pre[3x4], post_index[3x4], solve) for sampling image_name onto a view on the other side.

        to_reference (moving-side image on a reference-side view): pixel w is a reference point r (direct), sample
            the image at rigid @ (r + p * dvf(r)).
        otherwise (reference-side image on a moving-side view): pixel w is a moving point, a = rigid^-1 @ w, solve
            r + p * dvf(r) = a, sample the image at r.
        """
        rigid = np.asarray(self.rigid_matrix, dtype=float)
        if to_reference:
            pre = np.identity(4)
            post = rigid
        else:
            pre = self._rigid_inv(rigid)
            post = np.identity(4)

        image = Data.volume(image_name)
        M = np.asarray(image.matrix, dtype=float)              # rows = index axes in world (same as Image.Display)
        o = np.asarray(image.origin, dtype=float)
        s = np.asarray(image.spacing, dtype=float)

        # index_k = M[k] . (post_rot q + post_t - o) / s_k   ->   one 3x4 affine
        post_index = np.empty((3, 4))
        post_index[:, :3] = (M @ post[:3, :3]) / s[:, None]
        post_index[:, 3] = (M @ (post[:3, 3] - o)) / s

        return image.array, pre[:3].astype(np.float32), post_index.astype(np.float32), not to_reference

    def _side_of(self, image_name):
        """The side an image belongs to: by name for the registered images, otherwise by frame of reference.

        None when the image is in neither frame, or when both registered images share one frame (then only those two
        images can tell the sides apart).
        """
        if image_name == self.reference.image_name:
            return self.reference
        if image_name == self.moving.image_name:
            return self.moving
        if self.reference.frame == self.moving.frame:
            return None

        frame = self._frame_of(image_name)
        if frame == self.reference.frame:
            return self.reference
        if frame == self.moving.frame:
            return self.moving

        return None

    def _sides(self, base_name, image_name):
        """(base_name, image_name, base side, moved side) for a display call; None names default to the registered
        reference (base) / moving (image) image, i.e. the moving image deformed onto the reference."""
        base_name = self.reference.image_name if base_name is None else base_name
        image_name = self.moving.image_name if image_name is None else image_name

        base_side, moved_side = self._side_of(base_name), self._side_of(image_name)
        if base_side is None or moved_side is None:
            missing = base_name if base_side is None else image_name
            raise ValueError(f"'{missing}' is not in a frame of reference connected by '{self.deformable_name}'.")

        return base_name, image_name, base_side, moved_side

    def _slice_cache_for(self, plane, base_name, grid):
        """
        Slice cache for one view (plane on one base image), so views on different base images keep their own entries.
        Its state is the view's in-plane geometry (axes, spacing); if that changes (rotation of the shared display
        matrix, different spacing) the view's cache is cleared. Scrolling only moves position along the normal, so it
        never clears anything.
        """
        state = (np.round(np.asarray(grid['x_axis'], dtype=float), 9).tobytes(),
                 np.round(np.asarray(grid['y_axis'], dtype=float), 9).tobytes(),
                 np.round(np.asarray(grid['normal'], dtype=float), 9).tobytes(),
                 float(grid['sx']), float(grid['sy']))

        caches = self._display_cache.setdefault('mesh_slices', {})
        cache = caches.get((plane, base_name))
        if cache is None or cache['state'] != state:
            cache = {'state': state, 'entries': OrderedDict()}
            caches[(plane, base_name)] = cache

        return cache['entries']

    def _solve_points(self, points, ratio):
        """Reference points r with  r + ratio * dvf(r) = a  for known rigid-frame moving points a (N, 3).

        Same fixed-point iteration as the display kernels (_solve_field), vectorized over the points.
        """
        a = np.asarray(points, dtype=float)
        r = a.copy()
        if ratio == 0:
            return r

        for _ in range(self.INVERT_ITERATIONS):
            updated = a - ratio * self._sample_field_points(self.dvf, r)
            change = np.abs(updated - r).sum(axis=1).max(initial=0.0)
            r = updated
            if change < self.INVERT_TOLERANCE:
                break

        return r

    def _store_field(self, field, origin, spacing, direction):
        """Store a backend's field (reference -> rigid-frame moving, i.e. exactly ``dvf``) as float32.

        direction is the field grid's 3x3 direction with columns = axes (SimpleITK / ANTs convention); an oblique grid
        is resampled onto an identity grid first.
        """
        field = np.asarray(field)
        matrix = np.asarray(direction, dtype=float).reshape(3, 3).T          # rows = axes
        if not np.allclose(matrix, np.identity(3), atol=1e-3):
            field, spacing, origin, _ = self.correct_dvf_direction(field, spacing, origin, matrix)

        origin = tuple(float(v) for v in origin)
        spacing = tuple(float(v) for v in spacing)

        self.invalidate_display()
        self.origin = origin
        self.spacing = spacing
        self.dvf = np.ascontiguousarray(field, dtype=np.float32)
        self.dimensions = self.dvf.shape[:3]

    def _view_grid(self, plane, position, base_name, image_name, spacing=None):
        """(base_grid, grid): the base image's own grid for this view and the grid to sample on (see _view_spacing)."""
        base = Data.volume(base_name).display
        base_grid = base.get_grid(plane, position)
        if spacing is None:
            return base_grid, base_grid

        sx, sy = self._view_spacing(plane, base_name, image_name, spacing)
        return base_grid, base.grid_with_spacing(base_grid, sx, sy)

    def _view_spacing(self, plane, base_name, image_name, spacing=None):
        """In-plane (sx, sy): None = base image, 'native' = displayed image, float or (sx, sy)."""
        if spacing is None or isinstance(spacing, str):
            if spacing not in (None, 'native'):
                raise ValueError(f"spacing must be None, 'native', a number or (sx, sy); got {spacing!r}")
            display = Data.volume(base_name if spacing is None else image_name).display
            xi, yi, _ = display.AXES[plane]
            return float(display.spacing[xi]), float(display.spacing[yi])

        if np.isscalar(spacing):
            return float(spacing), float(spacing)

        return float(spacing[0]), float(spacing[1])

    def _warp_to_moving(self, points, percent=100):
        """Reference world -> moving native world (direct: + dvf, then rigid)."""
        r = np.asarray(points, dtype=float)
        r = r + (percent / 100) * self._sample_field_points(self.dvf, r)
        return self._apply_affine(self.rigid_matrix, r)

    def _warp_to_reference(self, points, percent=100):
        """Moving native world -> reference world (rigid^-1, then solve r + dvf(r) = a)."""
        a = self._apply_affine(self._rigid_inv(self.rigid_matrix), points)
        return self._solve_points(a, percent / 100)

    def add_deformable(self, deformable_name):
        """Store this registration in ``Data.deformable``; None builds a unique name from the image names.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Deformable(reference_name='CT 01', moving_name='CBCT 01').deformable_name
        'DVF_CT 01_CBCT 01'
        >>> mia.Deformable(reference_name='CT 01', moving_name='CBCT 01').deformable_name     # second one
        'DVF_CT 01_CBCT 01_1'
        """
        if deformable_name is None:
            if self.reference.image_name is None and self.moving.image_name is None:
                deformable_name = 'DVF_Unknown'
            else:
                deformable_name = 'DVF_' + self.reference.image_name + '_' + self.moving.image_name

            if deformable_name in Data.deformable_list():
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(deformable_name + '_' + str(n))
                    if new_name not in Data.deformable_list():
                        deformable_name = new_name
                        n = -100

        Data.deformable[deformable_name] = self

        return deformable_name

    def clear_mesh_slice_cache(self, plane=None):
        """Drop cached mesh slices for one plane, or all planes when plane is None.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> deform.clear_mesh_slice_cache('Axial')
        >>> deform.clear_mesh_slice_cache()
        """
        caches = self._display_cache.get('mesh_slices', {})
        for key in [k for k in caches if plane is None or k[0] == plane]:
            del caches[key]

    def compute_array(self, plane, position, base_name=None, image_name=None, spacing=None, percent=None,
                      background=None):
        """image_name rigidly + deformably mapped onto base_name's view. Same call as Rigid.compute_array.

        Parameters
        ----------
        plane : str
            View plane, e.g. 'Axial'.
        position : array_like
            The widget's position (base image's view).
        base_name : str, optional
            Image whose view is shown. Any image in either frame. None = the registered reference image.
        image_name : str, optional
            Image deformed onto the view. Any image in either frame. None = the registered moving image. An image on
            the same side as the base is in the same world, so it is sampled with no deformation.
        spacing : None, 'native', float or (sx, sy), optional
            None = base image grid, 'native' = image_name's spacing, or any spacing. Use the same value in
            :meth:`compute_grid`, :meth:`compute_mesh_slice` and :meth:`compute_aspect`.
        percent : float, optional
            Deformation level (0 = rigid only, 100 = full). None = ``self.visual.percent``. Scales ``dvf`` the same
            way in both directions, so the image and the contours (:meth:`compute_mesh_slice`) agree at any level.
        background : float, optional
            Value for pixels that fall outside image_name. None = image_name's ``background`` (-3001 for images,
            0 Gy for a dose).

        Returns
        -------
        dict or None
            'array' (h, w) float32, 'dimensions' (w, h), 'spacing', 'pixel_spacing' (sx, sy), 'x_axis', 'y_axis',
            'normal', 'origin', 'grid', 'scale' (sample spacing / base spacing) and 'pixel_offset' (always (0, 0)).
            None when the images are on opposite sides and there is no DVF yet.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> res = deform.compute_array('Axial', ct_position)                         # CBCT deformed onto the CT
        >>> res = deform.compute_array('Axial', ct_position, percent=0)              # rigid only
        >>> res = deform.compute_array('Axial', cbct_position, base_name='CBCT 01', image_name='CT 01')
        >>> res = deform.compute_array('Sagittal', ct_position, spacing='native')    # at the CBCT's spacing
        >>> res = deform.compute_array('Axial', ct_position, image_name='Dose 01')   # moving-frame dose on the CT
        """
        percent = self.visual.percent if percent is None else percent
        base_name, image_name, base_side, moved_side = self._sides(base_name, image_name)
        volume = Data.volume(image_name)
        background = volume.background if background is None else background
        base_grid, grid = self._view_grid(plane, position, base_name, image_name, spacing)
        scale = (grid['sx'] / base_grid['sx'], grid['sy'] / base_grid['sy'])

        if base_side is moved_side:
            res = volume.display.compute_array(plane, position, grid=grid, background=background)
            res['scale'] = scale
            res['pixel_offset'] = (0.0, 0.0)
            return res

        if self.dvf is None:
            return None

        array, pre, post_index, solve = self._sampling_setup(image_name, to_reference=moved_side is self.moving)
        field_origin, field_spacing = self._field_grid()

        x_axis = np.asarray(grid['x_axis'], dtype=np.float32)
        y_axis = np.asarray(grid['y_axis'], dtype=np.float32)
        origin = np.asarray(grid['origin'], dtype=np.float32)
        h, w = grid['shape']

        arr = _deformable_sample_kernel(array, self._get_field(), h, w, np.float32(grid['sx']),
                                        np.float32(grid['sy']), origin, x_axis, y_axis, pre, field_origin,
                                        field_spacing, np.float32(percent / 100), post_index, np.float32(background),
                                        solve, self.INVERT_ITERATIONS, np.float32(self.INVERT_TOLERANCE))

        return {'array': arr,
                'dimensions': (w, h),
                'spacing': volume.display.spacing,
                'pixel_spacing': (float(grid['sx']), float(grid['sy'])),
                'x_axis': x_axis, 'y_axis': y_axis, 'normal': grid['normal'], 'origin': origin, 'grid': grid,
                'scale': scale,
                'pixel_offset': (0.0, 0.0)}

    def compute_aspect(self, plane, base_name=None, image_name=None, spacing=None):
        """Pixel aspect (sx / sy) of the view, matching compute_array with the same images and spacing. The default
        keeps the base image's aspect.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.deformable['DVF_CT 01_CBCT 01'].compute_aspect('Sagittal')
        """
        base_name, image_name, _, _ = self._sides(base_name, image_name)
        sx, sy = self._view_spacing(plane, base_name, image_name, spacing)

        return np.round(sx / sy, 2)

    def compute_components(self, roi_name):
        """Connected components of a deformed ROI mesh (``self.roi``), or None if it has not been warped.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> parts = mia.Data.deformable['DVF_CT 01_CBCT 01'].compute_components('Liver')
        """
        if self.roi[roi_name] is not None:
            return compute_components(self.roi[roi_name])

        else:
            return None

    def compute_folding(self):
        """Fraction of DVF voxels where the map r -> r + dvf(r) folds (Jacobian determinant <= 0).

        0.0 means the field is invertible everywhere, so every "solve" (moving -> reference: reference-side images on
        a moving-side view, moving-side contours and 3D meshes) converges to the exact point. Anything above 0 marks
        regions where those are only approximate. The direct direction (the moving image deformed onto the reference)
        is exact either way.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.deformable['DVF_CT 01_CBCT 01'].compute_folding()
        0.0
        """
        if self.dvf is None:
            return None

        field = sitk.GetImageFromArray(self.dvf, isVector=True)
        field.SetSpacing([float(v) for v in self.spacing])
        field.SetOrigin([float(v) for v in self.origin])
        # keep the image referenced: an array view of a temporary image reads freed memory
        determinant_image = sitk.DisplacementFieldJacobianDeterminant(field)
        determinant = sitk.GetArrayViewFromImage(determinant_image)

        return float(np.count_nonzero(determinant <= 0) / determinant.size)

    def compute_grid(self, plane, position, base_name=None, image_name=None, vector='magnitude', spacing=None,
                     outside=0.0):
        """One DVF component or its magnitude (mm) on the same view grid as compute_array, for overlaying the field on
        base_name's view.

        Each pixel shows ``dvf`` (reference -> moving, in world axes, full deformation) at the reference point of the
        tissue it shows, so the same tissue has the same value whichever side the base image is on. On a
        reference-side view that point is the pixel itself (direct); on a moving-side view it is solved from the
        pixel. image_name only matters for spacing='native'.

        Parameters
        ----------
        plane : str
            View plane, e.g. 'Axial'.
        position : array_like
            The widget's position (base image's view).
        base_name, image_name : str, optional
            Same meaning as in :meth:`compute_array`.
        vector : {'x', 'y', 'z', 'magnitude'}, default 'magnitude'
        spacing : None, 'native', float or (sx, sy), optional
            Same meaning as in :meth:`compute_array`.
        outside : float, default 0.0
            Value for pixels outside the DVF grid.

        Returns
        -------
        dict or None
            'array', 'dimensions', 'pixel_spacing', 'grid', 'scale', 'pixel_offset'. None when there is no DVF yet.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> magnitude = deform.compute_grid('Axial', ct_position)
        >>> dz = deform.compute_grid('Coronal', cbct_position, base_name='CBCT 01', vector='z', outside=np.nan)
        """
        if self.dvf is None:
            return None

        base_name, image_name, base_side, _ = self._sides(base_name, image_name)
        base_grid, grid = self._view_grid(plane, position, base_name, image_name, spacing)

        # reference-side view: the pixel is the reference point (direct); moving-side view: solve it from the pixel
        solve = base_side is self.moving
        pre = self._rigid_inv(self.rigid_matrix) if solve else np.identity(4)
        pre = pre[:3].astype(np.float32)
        field_origin, field_spacing = self._field_grid()
        h, w = grid['shape']

        _VECTOR_INDEX = {'x': 0, 'y': 1, 'z': 2, 'magnitude': 3}
        arr = _field_component_kernel(self._get_field(), h, w,
                                      np.float32(grid['sx']), np.float32(grid['sy']),
                                      np.asarray(grid['origin'], dtype=np.float32),
                                      np.asarray(grid['x_axis'], dtype=np.float32),
                                      np.asarray(grid['y_axis'], dtype=np.float32),
                                      pre, field_origin, field_spacing,
                                      _VECTOR_INDEX[vector], np.float32(outside),
                                      solve, self.INVERT_ITERATIONS, np.float32(self.INVERT_TOLERANCE))

        return {'array': arr, 'dimensions': (w, h), 'pixel_spacing': (float(grid['sx']), float(grid['sy'])),
                'grid': grid, 'scale': (grid['sx'] / base_grid['sx'], grid['sy'] / base_grid['sy']),
                'pixel_offset': (0.0, 0.0)}

    def compute_inverted_dvf(self, iterations=50, mean_tolerance=1e-3, max_tolerance=1e-1):
        """Full moving -> reference field (the inverse of ``dvf``, same grid, (Z, Y, X, 3), (dx, dy, dz)), for export.

        Returned and never stored: nothing in this class needs it, since the moving -> reference direction is solved
        per point where it is used.

        Parameters
        ----------
        iterations : int, default 50
        mean_tolerance, max_tolerance : float
            Mean and maximum inverse-consistency error (mm) at which the inversion stops.

        Returns
        -------
        (Z, Y, X, 3) float32 ndarray

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> inverted = mia.Data.deformable['DVF_CT 01_CBCT 01'].compute_inverted_dvf()
        """
        return self._invert_field(self.dvf, self.origin, self.spacing, iterations=iterations,
                                  mean_tolerance=mean_tolerance, max_tolerance=max_tolerance)

    def compute_mesh_slice(self, roi_name, plane, position, base_name=None, image_name=None, percent=None,
                           return_pixel=True, use_cache=True, spacing=None):
        """Slice one of image_name's ROIs, deformed onto base_name's view, on the same grid compute_array samples, so
        image and contour line up.

        Pixel output is (col, row) floats relative to the view's top-left corner, pixel centers at integers; each
        contour is closed (first point repeated).

        Results are cached per view (plane + base image), keyed by (image, roi, percent, position rounded to 1 um,
        return_pixel). An entry is reused only if it was cut from the same deformed mesh object, so a new DVF / rigid /
        percent / ROI edit (which builds a new mesh in get_display_mesh) is picked up automatically. The returned list
        is the cached object; copy it before modifying it in place.

        Parameters
        ----------
        roi_name : str
            ROI on image_name.
        plane : str
            View plane, e.g. 'Axial'.
        position : array_like
            The widget's position (base image's view).
        base_name, image_name : str, optional
            Same meaning as in :meth:`compute_array`.
        percent : float, optional
            Deformation level (0 = rigid only, 100 = full). None = ``self.visual.percent``.
        return_pixel : bool, default True
            False returns the raw pyvista slice (world coordinates).
        use_cache : bool, default True
            Pass False while dragging or animating.
        spacing : None, 'native', float or (sx, sy), optional
            Same value as in :meth:`compute_array`.

        Returns
        -------
        list of (N, 2) ndarray, or pyvista.PolyData when return_pixel=False

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> contours = deform.compute_mesh_slice('Liver', 'Axial', ct_position)          # CBCT Liver on the CT
        >>> contours = deform.compute_mesh_slice('Liver', 'Axial', cbct_position, base_name='CBCT 01',
        ...                                      image_name='CT 01')                         # CT Liver on the CBCT
        >>> world = deform.compute_mesh_slice('Liver', 'Axial', ct_position, percent=50, return_pixel=False)
        """
        percent = self.visual.percent if percent is None else percent
        base_name, image_name, _, _ = self._sides(base_name, image_name)

        mesh = self.get_display_mesh(roi_name, base_name, image_name, percent=percent)
        if mesh is None:
            return [] if return_pixel else None

        _, grid = self._view_grid(plane, position, base_name, image_name, spacing)

        if use_cache:
            entries = self._slice_cache_for(plane, base_name, grid)
            key = (image_name, roi_name, float(percent), np.round(np.asarray(position, dtype=float), 3).tobytes(),
                   bool(return_pixel))
            hit = entries.get(key)
            if hit is not None and hit[0] is mesh:
                entries.move_to_end(key)
                return hit[1]

        roi_slice = mesh.slice(normal=np.asarray(grid['normal'], dtype=float), origin=np.asarray(position, float))
        if not return_pixel:
            result = roi_slice

        elif roi_slice.n_points == 0:
            result = []

        else:
            origin = np.asarray(grid['origin'], dtype=float)
            x_axis = np.asarray(grid['x_axis'], dtype=float)
            y_axis = np.asarray(grid['y_axis'], dtype=float)

            result = []
            for cell in roi_slice.strip(max_length=10000000).cell:
                d = np.asarray(cell.points) - origin
                pixel = np.column_stack((d @ x_axis / grid['sx'], d @ y_axis / grid['sy']))
                result.append(np.vstack((pixel, pixel[:1])))

        if use_cache:
            entries[key] = (mesh, result)
            while len(entries) > self.mesh_slice_cache_size:
                entries.popitem(last=False)

        return result

    def compute_registration(self, method='itk_fast_demons', reference_rois=None, moving_rois=None,
                             reference_mask=None, moving_mask=None, reference_points=None, moving_points=None,
                             **kwargs):
        """Run a deformable registration on the registered images and store the result in ``dvf``.

        The moving image (and its mask) is first resampled into the rigid frame with ``rigid_matrix``, so every method
        only has to find the deformation on top of the rigid pre-alignment. The reference is the fixed image, so the
        backend's field (reference point r -> moving point r + f(r)) is ``dvf`` exactly and is stored as is. The 3D
        ROI meshes (``roi``) are rebuilt afterwards.

        Parameters
        ----------
        method : str, default 'itk_fast_demons'
            SimpleITK / Elastix (:class:`DeformableITK`):

            * 'itk_bspline'          : multi-resolution B-spline, ``DeformableITK.bspline``.
            * 'itk_field'            : dense displacement field, any metric, ``DeformableITK.displacement_field``.
            * 'itk_demons'           : Thirion Demons, ``DeformableITK.demons``.
            * 'itk_symmetric_demons' : symmetric forces Demons, ``DeformableITK.symmetric_demons``.
            * 'itk_fast_demons'      : fast symmetric forces Demons, ``DeformableITK.fast_demons``.
            * 'itk_diffeomorphic'    : diffeomorphic Demons, ``DeformableITK.diffeomorphic``.
            * 'itk_level_set'        : level set motion, ``DeformableITK.level_set_motion``.
            * 'elastix'              : Elastix B-spline / spline kernel, ``DeformableITK.elastix``. Uses the points
              when given.
            * 'itk_landmark'         : B-spline fitted to the points only, ``DeformableITK.landmark``.

            ANTs (:class:`DeformableANTs`, needs ``antspyx``):

            * 'ants'          : any ANTs deformable transform / metric / preset, ``DeformableANTs.registration``
              (e.g. ``transform='SyN'``, ``metric='CC'``, ``label_weight=1.0``, ``preset='SyNOnly'``).
            * 'ants_landmark' : field fitted to the points only, ``DeformableANTs.landmark``.
        reference_rois, moving_rois : str or list of str, optional
            ROIs whose masks restrict the metric (ANTs: also the default labels for ``label_weight``), paired by
            position. reference_rois None = ``roi_names``; moving_rois None = the same names. Pairs missing on either
            image are skipped.
        reference_mask, moving_mask : numpy.ndarray, optional
            Binary (z, y, x) masks on each image's own grid; override the ROI masks.
        reference_points, moving_points : (N, 3) array_like, optional
            Paired world points (mm): reference points in the reference world, moving points in the moving image's
            native world. Required for the landmark methods; optional for 'elastix'.
        **kwargs
            Passed straight to the backend method (e.g. ``iterations``, ``metric``, ``crop``, ``cross_modality``,
            ``control_spacing``, ``transform``).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Deformable(reference_name='CT 01', moving_name='CBCT 01', roi_names=['Liver'],
        ...                         rigid_matrix=mia.Data.rigid['CT 01_CBCT 01'].get_combined_matrix())
        >>> deform.compute_registration()                                               # fast Demons, Liver masks
        >>> deform.compute_registration('itk_bspline', control_spacing=(15, 15, 15), metric='ANTS')
        >>> deform.compute_registration('itk_diffeomorphic', iterations=200, crop=20)
        >>> deform.compute_registration('elastix', transform='RecursiveBSpline', bending_weight=0.1)
        >>> deform.compute_registration('ants', transform='SyN', metric='CC', label_weight=1.0)
        >>> deform.compute_registration('ants', preset='antsRegistrationSyNQuick[so]')
        >>> deform.compute_registration('ants_landmark', reference_points=ct_points, moving_points=cbct_points,
        ...                             transform='tps')
        """
        landmark_methods = ('itk_landmark', 'ants_landmark')
        point_methods = landmark_methods + ('elastix',)
        if method not in self.ITK_METHODS and method not in landmark_methods and method != 'ants':
            raise ValueError(f"Unknown method: {method}")

        for name in (self.reference.image_name, self.moving.image_name):
            if name not in Data.image:
                raise ValueError(f"'{name}' must be loaded to compute a registration.")

        if (reference_points is None) != (moving_points is None):
            raise ValueError("reference_points and moving_points must be given together.")
        if reference_points is not None and method not in point_methods:
            raise ValueError(f"'{method}' does not use points; use one of {point_methods}.")
        if reference_points is None and method in landmark_methods:
            raise ValueError(f"'{method}' needs reference_points and moving_points.")

        # Moving points into the rigid frame, where the field is computed
        if moving_points is not None:
            reference_points = np.asarray(reference_points, dtype=float).reshape(-1, 3)
            moving_points = self._apply_affine(self._rigid_inv(self.rigid_matrix), np.reshape(moving_points, (-1, 3)))

        fixed, moving, fixed_mask, moving_mask = self._registration_images(reference_rois, moving_rois,
                                                                           reference_mask, moving_mask,
                                                                           images=method not in landmark_methods)

        if method in ('ants', 'ants_landmark'):
            # antspyx is only imported when an ANTs method is used, so the package works without it
            from ..utils.deformable.ants import DeformableANTs

            helper = DeformableANTs()
            direction = np.reshape(fixed.GetDirection(), (3, 3))          # columns = axes, as create_ants_image wants
            for image, is_reference, is_mask in ((fixed, True, False), (moving, False, False),
                                                 (fixed_mask, True, True), (moving_mask, False, True)):
                if image is not None:
                    helper.create_ants_image(sitk.GetArrayFromImage(image), fixed.GetOrigin(), fixed.GetSpacing(),
                                             direction, reference=is_reference, mask=is_mask)

            if method == 'ants_landmark':
                dvf_image = helper.landmark(reference_points, moving_points, **kwargs)
            else:
                dvf_image = helper.registration(**kwargs)

            # ANTsPy arrays are (x, y, z, 3)
            field = dvf_image.numpy().transpose(2, 1, 0, 3)
            self._store_field(field, dvf_image.origin, dvf_image.spacing, dvf_image.direction)

        else:
            helper = DeformableITK(reference_image=fixed, moving_image=moving, reference_mask=fixed_mask,
                                   moving_mask=moving_mask)

            if method == 'itk_landmark':
                dvf_image = helper.landmark(reference_points, moving_points, **kwargs)
            elif method == 'elastix':
                dvf_image = helper.elastix(reference_points=reference_points, moving_points=moving_points, **kwargs)
            else:
                dvf_image = getattr(helper, self.ITK_METHODS[method])(**kwargs)

            self._store_field(sitk.GetArrayFromImage(dvf_image), dvf_image.GetOrigin(), dvf_image.GetSpacing(),
                              dvf_image.GetDirection())

        self.update_rois()

    def connects(self, image_a, image_b):
        """True when the two images are on opposite sides of this registration (by name or frame of reference).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.deformable['DVF_CT 01_CBCT 01'].connects('Dose 01', 'CT 01')
        True
        """
        side_a, side_b = self._side_of(image_a), self._side_of(image_b)

        return side_a is not None and side_b is not None and side_a is not side_b

    def create_image(self, ratio=1):
        """Moving image rigidly + deformably resampled onto the reference grid (full 3D, for export).

        Runs the same kernel as the display (direct case) one reference slice at a time, so the export matches what is
        shown and needs no extra field in memory.

        Parameters
        ----------
        ratio : float, default 1
            Displacement scale (0 = rigid only, 1 = full).

        Returns
        -------
        SimpleITK.Image
            Float32 image with the reference image's geometry; the moving image's ``background`` (-3001) outside it.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> image = deform.create_image()
        >>> halfway = deform.create_image(ratio=0.5)
        """
        ref = Data.image[self.reference.image_name]
        volume, pre, post_index, _ = self._sampling_setup(self.moving.image_name, to_reference=True)
        field = self._get_field()
        field_origin, field_spacing = self._field_grid()

        M = np.asarray(ref.matrix, dtype=float)                 # rows = index axes in world
        origin = np.asarray(ref.origin, dtype=float)
        sx, sy, sz = (float(v) for v in ref.spacing)
        nz, ny, nx = np.shape(ref.array)[:3]
        x_axis, y_axis = M[0].astype(np.float32), M[1].astype(np.float32)

        array = np.empty((nz, ny, nx), dtype=np.float32)
        for k in range(nz):
            slice_origin = (origin + k * sz * M[2]).astype(np.float32)
            array[k] = _deformable_sample_kernel(volume, field, ny, nx, np.float32(sx), np.float32(sy), slice_origin,
                                                 x_axis, y_axis, pre, field_origin, field_spacing, np.float32(ratio),
                                                 post_index, np.float32(Data.image[self.moving.image_name].background),
                                                 False, self.INVERT_ITERATIONS,
                                                 np.float32(self.INVERT_TOLERANCE))

        image = sitk.GetImageFromArray(array)
        image.SetOrigin([float(v) for v in origin])
        image.SetSpacing([sx, sy, sz])
        image.SetDirection(M.T.ravel().tolist())               # sitk direction columns = axes

        return image

    def create_roi_image(self, roi_name, sitk_image=True):
        """Binary mask of a deformed ROI (``self.roi``) on the reference image grid.

        Parameters
        ----------
        roi_name : str
        sitk_image : bool, default True
            True returns a SimpleITK image with the reference geometry, False a (z, y, x) uint8 array.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> mask_image = deform.create_roi_image('Liver')
        >>> mask = deform.create_roi_image('Liver', sitk_image=False)
        """
        mesh = self.roi[roi_name].triangulate()

        image = vtk.vtkImageData()
        image.SetDimensions(np.flip(Data.image[self.reference.image_name].dimensions))
        image.SetSpacing(Data.image[self.reference.image_name].spacing)
        image.SetOrigin(Data.image[self.reference.image_name].origin)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        image.GetPointData().GetScalars().Fill(0)      # AllocateScalars leaves the memory uninitialized

        pdts = vtk.vtkPolyDataToImageStencil()
        pdts.SetInputData(mesh)
        pdts.SetOutputOrigin(image.GetOrigin())
        pdts.SetOutputSpacing(image.GetSpacing())
        pdts.SetOutputWholeExtent(image.GetExtent())
        pdts.Update()

        stencil = vtk.vtkImageStencil()
        stencil.SetInputData(image)
        stencil.SetStencilConnection(pdts.GetOutputPort())
        stencil.ReverseStencilOn()
        stencil.SetBackgroundValue(1)
        stencil.Update()
        out = stencil.GetOutput()
        array = numpy_support.vtk_to_numpy(out.GetPointData().GetScalars())
        mask = array.reshape(Data.image[self.reference.image_name].dimensions)  # z,y,x

        if sitk_image:
            ref_image = Data.image[self.reference.image_name].create_sitk_image()
            mask_image = sitk.GetImageFromArray(mask)
            mask_image.CopyInformation(ref_image)

            return mask_image

        else:
            return mask

    def export_image(self, path=None):
        """Write the deformed moving image (create_image) to any format SimpleITK can write. Nothing is written when
        path is None.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.deformable['DVF_CT 01_CBCT 01'].export_image('C:/Data/cbct_deformed.mhd')
        """
        if self.moving.image_name is not None and path is not None:
            image = self.create_image()

            sitk.WriteImage(image, path)

    def get_display_mesh(self, roi_name, base_name=None, image_name=None, percent=None):
        """image_name's ROI warped into base_name's world, or None (no mesh, or no DVF yet).

        A moving-side ROI on a reference-side view is solved per vertex; a reference-side ROI on a moving-side view is
        direct. Same side: the ROI's own mesh. Cached per direction / image / percent / dvf / ROI mesh / rigid, so
        repeated calls are free.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> mesh = deform.get_display_mesh('Liver')                       # CBCT Liver on the CT, for a 3D view
        >>> halfway = deform.get_display_mesh('Liver', percent=50)
        >>> on_cbct = deform.get_display_mesh('Liver', base_name='CBCT 01', image_name='CT 01')
        """
        percent = self.visual.percent if percent is None else percent
        base_name, image_name, base_side, moved_side = self._sides(base_name, image_name)

        source = Data.volume(image_name).roi.get(roi_name)     # a dose has no ROIs -> None
        if source is None or source.mesh is None:
            return None

        if base_side is moved_side:
            return source.mesh

        if self.dvf is None:
            return None

        to_reference = moved_side is self.moving
        key = ('mesh', to_reference, image_name, roi_name, float(percent), id(self.dvf), id(source.mesh),
               np.asarray(self.rigid_matrix, dtype=float).tobytes())
        mesh = self._display_cache.get(key)
        if mesh is None:
            mesh = source.mesh.copy()
            if to_reference:
                mesh.points = self._warp_to_reference(mesh.points, percent=percent)
            else:
                mesh.points = self._warp_to_moving(mesh.points, percent=percent)
            self._display_cache[key] = mesh

        return mesh

    def get_view_center(self, plane, position, base_name=None):
        """base_name's crosshair in view pixel coords (pixel centers at integers), for crosshair / zoom anchors.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> col, row = mia.Data.deformable['DVF_CT 01_CBCT 01'].get_view_center('Axial', ct_position)
        """
        base_name = self.reference.image_name if base_name is None else base_name
        base = Data.volume(base_name).display
        grid = base.get_grid(plane, position)
        d = base.crosshair - grid['origin']

        return float(d @ grid['x_axis']) / grid['sx'], float(d @ grid['y_axis']) / grid['sy']

    def invalidate_display(self):
        """Drop cached display fields and warped meshes. Only needed after editing ``dvf`` or ``rigid_matrix`` in place
        (replacing them is detected automatically).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> deform.dvf *= 0.5
        >>> deform.invalidate_display()
        """
        self._display_cache = {}

    def map_mesh(self, roi_name, set_moving=True, return_mesh=False):
        """Map a deformed ROI (``self.roi``, reference world) back into the moving image's native world.

        Use after editing the deformed contour on the reference, so the edit lands on the moving image.

        Parameters
        ----------
        roi_name : str
        set_moving : bool, default True
            Replace the moving image's ROI mesh with the mapped one.
        return_mesh : bool, default False
            Return the mapped mesh.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> deform.map_mesh('Liver')                                                   # update the CBCT's Liver
        >>> mesh = deform.map_mesh('Liver', set_moving=False, return_mesh=True)       # just look at it
        """
        if self.roi[roi_name] is None:
            self.update_rois(roi_name)

        if self.roi[roi_name] is None or self.dvf is None:
            return None

        # reference world -> moving native world is the direct direction of dvf (no solve needed)
        mapped_mesh = copy.deepcopy(self.roi[roi_name])
        mapped_mesh.points = self._warp_to_moving(self.roi[roi_name].points, percent=self.visual.percent)
        if set_moving:
            Data.image[self.moving.image_name].roi[roi_name].update_mesh(mapped_mesh)

        if return_mesh:
            return mapped_mesh
        else:
            return None

    def save(self, path):
        """Write the registration to one .npz file: the JSON state, the DVF and the reference / moving meshes (points
        and faces). Written to a temporary name and swapped in, so a crash never leaves a half-written file. No
        pickle, so loading never runs code from the file.

        Warped ROIs (``roi``), display settings and caches are not saved; they are rebuilt on demand.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.deformable['DVF_CT 01_CBCT 01'].save('C:/Data/ct_cbct.npz')
        """
        arrays = {'record': np.array(json.dumps(self._record(), separators=(',', ':')))}
        if self.dvf is not None:
            arrays['dvf'] = np.asarray(self.dvf)

        for name, side in (('reference', self.reference), ('moving', self.moving)):
            if side.mesh is not None:
                arrays[name + '_points'] = np.asarray(side.mesh.points)
                arrays[name + '_faces'] = np.asarray(side.mesh.faces)

        temp = path + '.tmp'
        with open(temp, 'wb') as f:      # a file object stops numpy from appending .npz to the temporary name
            np.savez(f, **arrays)
        os.replace(temp, path)

    def set_percent(self, percent):
        """Deformation level used by compute_array / compute_mesh_slice / get_display_mesh when no percent is passed
        (0-100).

        Warped meshes cached for other levels are dropped, so dragging a slider doesn't pile up mesh copies.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> deform.set_percent(50)
        >>> res = deform.compute_array('Axial', ct_position)        # halfway between rigid and full deformation
        """
        percent = float(percent)
        if percent == self.visual.percent:
            return
        self.visual.percent = percent
        # drop warped meshes for other levels so dragging a slider doesn't pile up mesh copies
        for key in [k for k in self._display_cache if isinstance(k, tuple) and k[0] == 'mesh' and k[4] != percent]:
            del self._display_cache[key]

    def update_rois(self, roi_name=None):
        """Warp the visible moving ROI meshes into the reference world at ``visual.percent`` (rigid^-1, then solve
        r + dvf(r) per vertex) and store them in ``self.roi`` for the 3D view. Call again after the DVF, rigid or
        percent changes.

        Parameters
        ----------
        roi_name : str, optional
            Only this ROI. None = every ROI.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> deform = mia.Data.deformable['DVF_CT 01_CBCT 01']
        >>> deform.update_rois()
        >>> deform.update_rois('Liver')
        >>> liver_on_ct = deform.roi['Liver']
        """
        for name in list(self.roi.keys()):
            if name not in Data.roi_list():
                del self.roi[name]

        for name in Data.roi_list():
            if name not in list(self.roi.keys()):
                self.roi[name] = None

        for name in Data.roi_list():
            if roi_name is None or name == roi_name:
                roi = Data.image[self.moving.image_name].roi[name]
                if roi.mesh is not None and roi.visible:
                    mesh = roi.mesh.copy()
                    mesh.points = self._warp_to_reference(roi.mesh.points, percent=self.visual.percent)
                    self.roi[name] = mesh