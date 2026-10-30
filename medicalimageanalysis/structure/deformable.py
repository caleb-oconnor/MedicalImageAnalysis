"""
Morfeus Lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Deformable image registration management and multi-planar display for medical imaging datasets (CT/MR).
    This module handles displacement vector fields (DVFs), rigid + deformable resampling for display, and ROI
    mesh warping workflows.

    Display follows the same contract as Image.Display / Rigid: the caller passes (plane, position) where
    position is the widget's top-left reslice corner, the output grid comes from the *reference* image's
    Display (so its shared oblique rotation, wheel scrolling and crosshair all just work), and the deformed
    image is sampled straight onto that grid in one fused numba pass (no full-volume resample on screen).

Conventions (unchanged from the registration code):
    * self.dvf is (Z, Y, X, 3), ordered (dx, dy, dz) in mm, identity direction, on (self.origin, self.spacing).
    * It is a FORWARD field defined in the rigid frame:  x_ref = x + dvf(x),  where  x = rigid_matrix^-1 @ m
      and m is a point in the moving image's native world.
    * Forward display (inverse=False): pixel at reference-world w samples moving at  rigid @ (w + inv_dvf(w)).
    * Inverse display (inverse=True):  pixel at moving-world w samples reference at  x + dvf(x),  x = rigid^-1 @ w.

Structure:
    * Deformable: Manages DVF computations, registration tracking, display sampling and ROI transformations.
"""

import os
import copy

from collections import OrderedDict

import vtk
import numpy as np
import pandas as pd
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
                              pre, field_origin, field_spacing, ratio, post_index, background):
    """
    One pass per output pixel:
        w      = origin + u * x_axis + v * y_axis                (output-view world point)
        a      = pre @ [w, 1]                                     (into the field's frame)
        q      = a + ratio * field(a)                             (displace)
        index  = post_index @ [q, 1]                              (into the sampled volume's (x, y, z) voxel index;
                                                                    rigid, direction and spacing are all folded in)
    then trilinear sample of the volume.
    """
    out = np.empty((h, w), dtype=np.float32)
    nz, ny, nx = volume.shape
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

            if kk < 0 or kk > nz - 1 or jj < 0 or jj > ny - 1 or ii < 0 or ii > nx - 1:
                out[i, j] = background
                continue

            z0 = min(int(kk), nz - 2)
            y0 = min(int(jj), ny - 2)
            x0 = min(int(ii), nx - 2)
            fz, fy, fx = kk - z0, jj - y0, ii - x0

            c00 = volume[z0, y0, x0] * (1 - fx) + volume[z0, y0, x0 + 1] * fx
            c01 = volume[z0, y0 + 1, x0] * (1 - fx) + volume[z0, y0 + 1, x0 + 1] * fx
            c10 = volume[z0 + 1, y0, x0] * (1 - fx) + volume[z0 + 1, y0, x0 + 1] * fx
            c11 = volume[z0 + 1, y0 + 1, x0] * (1 - fx) + volume[z0 + 1, y0 + 1, x0 + 1] * fx
            c0 = c00 * (1 - fy) + c01 * fy
            c1 = c10 * (1 - fy) + c11 * fy
            out[i, j] = c0 * (1 - fz) + c1 * fz

    return out


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _field_component_kernel(field, h, w, sx, sy, origin, x_axis, y_axis, pre, field_origin, field_spacing,
                            component, outside):
    """
    Samples one DVF component (0=x, 1=y, 2=z) or its magnitude (3) onto the output view. Pixels outside the field's
    grid get `outside` instead of a clamped edge value, so the field's footprint is visible.
    """
    out = np.empty((h, w), dtype=np.float32)
    fnz, fny, fnx = field.shape[0], field.shape[1], field.shape[2]
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

            fi = (ax - field_origin[0]) / field_spacing[0]
            fj = (ay - field_origin[1]) / field_spacing[1]
            fk = (az - field_origin[2]) / field_spacing[2]
            if fk < 0 or fk > fnz - 1 or fj < 0 or fj > fny - 1 or fi < 0 or fi > fnx - 1:
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
    mode='nearest' behaviour the ROI warping uses, so meshes and image stay consistent at the field edges.
    """
    fnz, fny, fnx = field.shape[0], field.shape[1], field.shape[2]
    fi = min(max(fi, 0.0), fnx - 1.0)
    fj = min(max(fj, 0.0), fny - 1.0)
    fk = min(max(fk, 0.0), fnz - 1.0)

    x0 = min(int(fi), fnx - 2)
    y0 = min(int(fj), fny - 2)
    z0 = min(int(fk), fnz - 2)
    tx, ty, tz = fi - x0, fj - y0, fk - z0

    d0 = 0.0
    d1 = 0.0
    d2 = 0.0
    for c in range(3):
        c00 = field[z0, y0, x0, c] * (1 - tx) + field[z0, y0, x0 + 1, c] * tx
        c01 = field[z0, y0 + 1, x0, c] * (1 - tx) + field[z0, y0 + 1, x0 + 1, c] * tx
        c10 = field[z0 + 1, y0, x0, c] * (1 - tx) + field[z0 + 1, y0, x0 + 1, c] * tx
        c11 = field[z0 + 1, y0 + 1, x0, c] * (1 - tx) + field[z0 + 1, y0 + 1, x0 + 1, c] * tx
        v = (c00 * (1 - ty) + c01 * ty) * (1 - tz) + (c10 * (1 - ty) + c11 * ty) * tz
        if c == 0:
            d0 = v
        elif c == 1:
            d1 = v
        else:
            d2 = v

    return d0, d1, d2


class Deformable(object):
    """
    Manages non-rigid transformation workflows, holding non-rigid fields, parameters, and morph functions.

    Display API mirrors Rigid so the same viewer code can drive either:
        compute_array(plane, position)          -> dict with 'array', 'grid', 'pixel_spacing', 'scale', ...
        compute_grid(plane, position, vector)   -> DVF component / magnitude on the same grid
        compute_mesh_slice(roi_name, plane, position)
        compute_aspect(plane)
        set_current_order(inverse)              -> flip which image is shown deformed onto which
        current_ref / current_mov               -> image names the view is built on / being deformed

    Attributes
    ----------
    reference_name : str
        Key label matching the static reference/fixed baseline scan profile.
    reference_sops : list
        DICOM Service-Object Pair Instance UIDs identifying source reference items.
    moving_name : str
        Key label identifying the structural target image volume undergoing registration adjustments.
    moving_sops : list
        DICOM Instance UIDs mapping back to components within the moving scan profile.
    roi_names : list of str
        Descriptive structural ROI names associated with the target datasets.
    rigid_rois : dict
        Intermediary cache tracking rigid positional adjustments prior to applying non-rigid deformations.
    rois : dict
        Moving ROI meshes deformed into the reference world (100 %).
    reference_mesh : PolyData
        Master structural surface reference asset definition tracking target anatomy elements.
    moving_mesh : PolyData
        Deformable structural mesh surface parameters tracking moving anatomical targets.
    local_uid : str
        Process UID token generated for system pipeline identification tasks.
    modality : str
        Medical imaging scan parameter type (e.g., 'CT', 'MR').
    dvf : np.ndarray
        Forward displacement field (Z, Y, X, 3), (dx, dy, dz) in mm.
    inverted_dvf : np.ndarray
        Inverse of dvf on the same grid, computed lazily.
    origin : list or tuple
        DVF grid origin (world).
    spacing : list or tuple
        DVF grid spacing (x, y, z).
    dimensions : tuple or list
        DVF grid dimensions (z, y, x).
    rigid_matrix : np.ndarray
        4x4 rigid pre-alignment; moving native world = rigid_matrix @ rigid frame.
    inverse : bool
        False: moving deformed onto reference. True: reference deformed onto moving.
    deformable_name : str
        Registry ID tracking active parameters in runtime contexts.
    """

    def __init__(self, dvf=None, origin=None, spacing=None, dimensions=None, roi_names=None,
                 rigid_matrix=None, dvf_matrix=None, registration_name=None, reference_name=None, moving_name=None,
                 reference_sops=None, moving_sops=None, reference_meshes=None, moving_meshes=None, inverse=False):
        """
        Initializes a Deformable registration record holding transformation parameters.
        """
        self.reference_name = reference_name
        self.reference_sops = reference_sops
        self.moving_name = moving_name
        self.moving_sops = moving_sops
        self.roi_names = roi_names
        self.rigid_rois = dict.fromkeys(Data.roi_list)
        self.rois = dict.fromkeys(Data.roi_list)
        self.reference_mesh = reference_meshes
        self.moving_mesh = moving_meshes
        self.local_uid = generate_uid()

        self.modality = None
        if dvf_matrix is not None and not np.allclose(dvf_matrix, np.identity(3), atol=1e-3):
            self.dvf, self.spacing, self.origin, self.dimensions = self.correct_dvf_direction(dvf, spacing, origin,
                                                                                              dvf_matrix)
        else:
            self.dvf = dvf
            self.origin = origin
            self.spacing = spacing
            self.dimensions = dimensions
        self.inverted_dvf = None

        self.rigid_matrix = np.identity(4) if rigid_matrix is None else np.asarray(rigid_matrix, dtype=float)

        self.inverse = False
        self.current_ref = None
        self.current_mov = None
        self.set_current_order(inverse)

        self.visual = {'opacity': 0.5, 'vector': None}
        self.misc = {}

        # Everything display-side that is derived from dvf / rigid_matrix lives here, so save_deformable can skip it
        # and so it rebuilds itself when dvf is replaced (checked by identity, not by a manual flag).
        self._display_cache = {}
        self.mesh_slice_cache_size = 2000  # per plane, all ROIs together; least-recently-used entries dropped first

        self.deformable_name = self.add_deformable(registration_name)

        if self.dvf is not None:
            self.update_rois()

    @staticmethod
    def _apply_affine(matrix, points):
        points = np.asarray(points, dtype=float)
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    def _collect_masks(self):
        ref_mask = None
        mov_mask = None
        for roi_name in self.roi_names:
            ref_roi = Data.image[self.reference_name].rois[roi_name]
            mov_roi = Data.image[self.moving_name].rois[roi_name]
            if ref_roi.mesh is not None or ref_roi.contour_pixel is not None:
                if mov_roi.mesh is not None or mov_roi.contour_pixel is not None:
                    ref_mask = ref_roi.compute_mask() if ref_mask is None else ref_mask + ref_roi.compute_mask()
                    mov_mask = mov_roi.compute_mask() if mov_mask is None else mov_mask + mov_roi.compute_mask()

        return ref_mask, mov_mask

    def _field_grid(self):
        return np.asarray(self.origin, dtype=np.float32), np.asarray(self.spacing, dtype=np.float32)

    def _get_field(self, which='forward'):
        """
        float32, C-contiguous copy of the forward or inverted DVF for the numba kernels, rebuilt automatically when
        self.dvf is replaced (compute_bspline, compute_demons, loading, ...).
        """
        if self.dvf is None:
            return None

        if which == 'inverse':
            source = self.get_inverted_dvf()
        else:
            source = self.dvf

        key = 'field_' + which
        cached = self._display_cache.get(key)
        if cached is None or cached[0] is not source:
            cached = (source, np.ascontiguousarray(source, dtype=np.float32))
            self._display_cache[key] = cached

        return cached[1]

    @staticmethod
    def _rigid_inv(mat):
        """Exact inverse of a rigid 4x4."""
        inv = np.identity(4)
        inv[:3, :3] = mat[:3, :3].T
        inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
        return inv

    def _sample_field_points(self, field, points):
        """Clamped trilinear lookup of a (Z, Y, X, 3) field at world points (N, 3) -> (N, 3) displacements."""
        voxel = (np.asarray(points, dtype=float) - np.asarray(self.origin)) / np.asarray(self.spacing)
        coords = [voxel[:, 2], voxel[:, 1], voxel[:, 0]]
        disp = np.empty((voxel.shape[0], 3))
        for i in range(3):
            disp[:, i] = map_coordinates(field[..., i], coords, order=1, mode='nearest')

        return disp

    def _sampling_setup(self):
        """
        Returns (volume, field, pre[3x4], post_index[3x4]) for the current order.

        Forward  (inverse=False): w in reference world. q = w + inv_dvf(w) (rigid frame), sample moving at rigid @ q.
        Inverse  (inverse=True):  w in moving world.    a = rigid^-1 @ w, q = a + dvf(a), sample reference at q.
        """
        rigid = np.asarray(self.rigid_matrix, dtype=float)
        if self.inverse:
            pre = self._rigid_inv(rigid)
            post = np.identity(4)
            field = self._get_field('forward')
        else:
            pre = np.identity(4)
            post = rigid
            field = self._get_field('inverse')

        image = Data.image[self.current_mov]
        M = np.asarray(image.matrix, dtype=float)              # rows = index axes in world (same as Image.Display)
        o = np.asarray(image.origin, dtype=float)
        s = np.asarray(image.spacing, dtype=float)

        # index_k = M[k] . (post_rot q + post_t - o) / s_k   ->   one 3x4 affine
        post_index = np.empty((3, 4))
        post_index[:, :3] = (M @ post[:3, :3]) / s[:, None]
        post_index[:, 3] = (M @ (post[:3, 3] - o)) / s

        return image.array, field, pre[:3].astype(np.float32), post_index.astype(np.float32)

    def _slice_cache_for(self, plane, grid):
        """
        Per-plane slice cache. Its state is the view's in-plane geometry (current_ref, axes, spacing); if any of that
        changes (rotation of the shared display matrix, flipping current order, different spacing) the plane's cache is
        cleared. Scrolling only moves position along the normal, so it never clears anything.
        """
        state = (self.current_ref,
                 np.round(np.asarray(grid['x_axis'], dtype=float), 9).tobytes(),
                 np.round(np.asarray(grid['y_axis'], dtype=float), 9).tobytes(),
                 np.round(np.asarray(grid['normal'], dtype=float), 9).tobytes(),
                 float(grid['sx']), float(grid['sy']))

        caches = self._display_cache.setdefault('mesh_slices', {})
        cache = caches.get(plane)
        if cache is None or cache['state'] != state:
            cache = {'state': state, 'entries': OrderedDict()}
            caches[plane] = cache

        return cache['entries']

    def _view_grid(self, plane, position, use_moving_spacing=False):
        base = Data.image[self.current_ref].display
        ref_grid = base.get_grid(plane, position)
        grid = ref_grid
        if use_moving_spacing:
            mover = Data.image[self.current_mov].display
            xi, yi, _ = mover.axes[plane]
            grid = base.grid_with_spacing(ref_grid, mover.spacing[xi], mover.spacing[yi])

        return ref_grid, grid

    def _warp_to_reference(self, points, percent=100):
        """Moving native world -> reference world (forward field)."""
        x = self._apply_affine(np.linalg.inv(self.rigid_matrix), points)
        return x + (percent / 100) * self._sample_field_points(self.dvf, x)

    def _warp_to_moving(self, points, percent=100):
        """Reference world -> moving native world (inverted field)."""
        q = np.asarray(points, dtype=float)
        q = q + (percent / 100) * self._sample_field_points(self.get_inverted_dvf(), q)
        return self._apply_affine(self.rigid_matrix, q)

    def add_deformable(self, deformable_name):
        """
        Registers this transformation instance inside the repository storage tracking framework.

        Parameters
        ----------
        deformable_name : str
            Preferred tracking title identity. If None, an identifier is generated.

        Returns
        -------
        str
            The final tracking registry label used to locate this specific deformable context.
        """
        if deformable_name is None:
            if self.reference_name is None and self.moving_name is None:
                deformable_name = 'DVF_Unknown'
            else:
                deformable_name = 'DVF_' + self.reference_name + '_' + self.moving_name

            if deformable_name in Data.deformable_list:
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(deformable_name + '_' + str(n))
                    if new_name not in Data.deformable_list:
                        deformable_name = new_name
                        n = -100

        Data.deformable[deformable_name] = self
        Data.deformable_list += [deformable_name]

        return deformable_name

    def clear_mesh_slice_cache(self, plane=None):
        """Drop cached mesh slices for one plane, or all planes."""
        caches = self._display_cache.get('mesh_slices', {})
        if plane is None:
            caches.clear()
        else:
            caches.pop(plane, None)

    def compute_array(self, plane, position, use_moving_spacing=False, percent=100, background=-3001.0):
        """
        Same call as Rigid.compute_array: plane + the widget's position (top-left corner of the current_ref view).
        Returns current_mov rigidly + deformably mapped onto that view, plus 'scale' and 'pixel_offset'.

        percent scales the displacement (0 = rigid only, 100 = full deformation). For the forward view this scales the
        inverted field, which is the usual approximation for partial-deformation animation.
        """
        if self.dvf is None or self.current_ref is None or self.current_mov is None:
            return None

        ref_grid, grid = self._view_grid(plane, position, use_moving_spacing=use_moving_spacing)
        volume, field, pre, post_index = self._sampling_setup()
        field_origin, field_spacing = self._field_grid()

        x_axis = np.asarray(grid['x_axis'], dtype=np.float32)
        y_axis = np.asarray(grid['y_axis'], dtype=np.float32)
        origin = np.asarray(grid['origin'], dtype=np.float32)
        h, w = grid['shape']

        arr = _deformable_sample_kernel(volume, field, h, w, np.float32(grid['sx']), np.float32(grid['sy']),
                                        origin, x_axis, y_axis, pre, field_origin, field_spacing,
                                        np.float32(percent / 100), post_index, np.float32(background))

        return {'array': arr,
                'dimensions': (w, h),
                'spacing': Data.image[self.current_mov].display.spacing,
                'pixel_spacing': (float(grid['sx']), float(grid['sy'])),
                'x_axis': x_axis, 'y_axis': y_axis, 'normal': grid['normal'], 'origin': origin, 'grid': grid,
                'scale': (grid['sx'] / ref_grid['sx'], grid['sy'] / ref_grid['sy']),
                'pixel_offset': (0.0, 0.0)}

    def compute_grid(self, plane, position, vector='magnitude', use_moving_spacing=False, outside=0.0):
        """
        DVF component ('x', 'y', 'z') or 'magnitude' (mm) on the same view grid as compute_array, for overlaying the
        field on the current_ref view. Always the forward field, evaluated at the rigid-frame point under each pixel.
        """
        if self.dvf is None or self.current_ref is None:
            return None

        ref_grid, grid = self._view_grid(plane, position, use_moving_spacing=use_moving_spacing)
        pre = (self._rigid_inv(self.rigid_matrix) if self.inverse else np.identity(4))[:3].astype(np.float32)
        field_origin, field_spacing = self._field_grid()
        h, w = grid['shape']

        _VECTOR_INDEX = {'x': 0, 'y': 1, 'z': 2, 'magnitude': 3}
        arr = _field_component_kernel(self._get_field('forward'), h, w,
                                      np.float32(grid['sx']), np.float32(grid['sy']),
                                      np.asarray(grid['origin'], dtype=np.float32),
                                      np.asarray(grid['x_axis'], dtype=np.float32),
                                      np.asarray(grid['y_axis'], dtype=np.float32),
                                      pre, field_origin, field_spacing,
                                      _VECTOR_INDEX[vector], np.float32(outside))

        return {'array': arr, 'dimensions': (w, h), 'pixel_spacing': (float(grid['sx']), float(grid['sy'])),
                'grid': grid, 'scale': (grid['sx'] / ref_grid['sx'], grid['sy'] / ref_grid['sy']),
                'pixel_offset': (0.0, 0.0)}

    def compute_aspect(self, plane, use_moving_spacing=False):
        """Aspect matching the (sx, sy) compute_array samples with."""
        _PLANE_AXES = {'Axial': (0, 1, 2), 'Sagittal': (1, 2, 0), 'Coronal': (0, 2, 1)}
        name = self.current_mov if use_moving_spacing else self.current_ref
        spacing = Data.image[name].display.spacing
        xi, yi, _ = _PLANE_AXES[plane]

        return np.round(spacing[xi] / spacing[yi], 2)

    def compute_biomechanical(self):
        """
        Placeholder method for biomechanical simulation algorithms or finite element extensions.
        """
        pass

    def compute_bspline(self, modality_gradient=True, sigma=2, control_spacing=None, mesh_size=None, gradient=1e-5,
                        iterations=100, crop=5):
        """
        Runs a B-Spline deformable registration workflow using metrics managed via SimpleITK components.

        Parameters
        ----------
        modality_gradient : bool, optional
            Evaluates multi-modality cross-corrections if True. Defaults to True.
        sigma : float, optional
            Gaussian blur kernel size applied across structural mask boundaries. Defaults to 2.
        control_spacing : list, optional
            Physical distance mapping parameters defining B-Spline grid densities. Defaults to None.
        mesh_size : list, optional
            Mesh node dimension parameters shaping transform density. Defaults to None.
        gradient : float, optional
            Optimization termination step tolerance. Defaults to 1e-5.
        iterations : int, optional
            Optimization limits capping evaluation routines. Defaults to 100.
        crop : int, optional
            Boundary truncation thickness processing parameters stripping edges. Defaults to 5.
        """
        ref = Data.image[self.reference_name]
        mov = Data.image[self.moving_name]
        ref_image = ref.create_sitk_image()
        mov_image = mov.create_sitk_image()

        euler = sitk.Euler3DTransform()
        euler.SetMatrix(self.rigid_matrix[:3, :3].flatten().tolist())
        euler.SetTranslation(self.rigid_matrix[:3, 3])

        mov_resampled = sitk.Resample(mov_image, ref_image, euler, sitk.sitkLinear, 0.0, mov_image.GetPixelID())

        ref_mask, mov_mask = self._collect_masks()
        deform_itk = DeformableITK(reference_image=ref_image, moving_image=mov_resampled, reference_mask=None,
                                   moving_mask=None)

        if ref.modality != mov.modality and modality_gradient:
            deform_itk.cross_modality_correction()

        if ref_mask is not None and mov_mask is not None:
            deform_itk.create_sitk_image(ref_mask, ref.origin, ref.spacing, ref.matrix, mask=True)
            deform_itk.create_sitk_image(mov_mask, mov.origin, mov.spacing, mov.matrix, reference=False, mask=True)

            if sigma is not None:
                deform_itk.blur_mask(sigma=sigma)

        deform_itk.resample()
        dvf_image = deform_itk.bspline(control_spacing=control_spacing, mesh_size=mesh_size, gradient=gradient,
                                       iterations=iterations, crop=crop)

        self.origin = dvf_image.GetOrigin()
        self.spacing = dvf_image.GetSpacing()
        self.dvf = sitk.GetArrayFromImage(dvf_image)
        self.dimensions = self.dvf.shape[:3]
        self.invalidate_display()

    def compute_demons(self, method=None, modality_gradient=True, sigma=2, smooth=True, std=1, iterations=50,
                       intensity_threshold=0.001, step=2.0, crop=5):
        """
        Executes dense fluid registration variants including standard, fast, or diffeomorphic Demons models.

        Parameters
        ----------
        method : str, optional
            Target alignment flavor flag. Options include 'Demons' or 'Diffeomorphic'. Defaults to None.
        modality_gradient : bool, optional
            Integrates cross-modality intensity matching corrections if True. Defaults to True.
        sigma : float, optional
            Scale settings smoothing binary structural mask surfaces. Defaults to 2.
        smooth : bool, optional
            Smooths displacement entries post-iteration if True. Defaults to True.
        std : float, optional
            Standard deviation governing Gaussian field smoothing processes. Defaults to 1.
        iterations : int, optional
            Iteration runtime loop execution threshold limit bounds. Defaults to 50.
        intensity_threshold : float, optional
            Minimum intensity change required to continue evaluation steps. Defaults to 0.001.
        step : float, optional
            Field translation step scale factor. Defaults to 2.0.
        crop : int, optional
            Boundary exclusion padding pixel counts. Defaults to 5.
        """
        ref = Data.image[self.reference_name]
        mov = Data.image[self.moving_name]

        deform_itk = DeformableITK()
        deform_itk.create_sitk_image(ref.array, ref.origin, ref.spacing, ref.matrix)
        deform_itk.create_sitk_image(mov.array, mov.origin, mov.spacing, mov.matrix, reference=False)

        if ref.modality != mov.modality and modality_gradient:
            deform_itk.cross_modality_correction()

        ref_mask, mov_mask = self._collect_masks()
        if ref_mask is not None and mov_mask is not None:
            deform_itk.create_sitk_image(ref_mask, ref.origin, ref.spacing, ref.matrix, mask=True)
            deform_itk.create_sitk_image(mov_mask, mov.origin, mov.spacing, mov.matrix, reference=False, mask=True)

            if sigma is not None:
                deform_itk.blur_mask(sigma=sigma)

        deform_itk.resample()
        if method in ['Demons', 'demons']:
            dvf_image = deform_itk.demons(smooth=smooth, std=std, iterations=iterations,
                                          intensity_threshold=intensity_threshold, crop=crop)

        elif method in ['Diffeomorphic', 'diffeomorphic']:
            dvf_image = deform_itk.diffeomorphic(smooth=smooth, std=std, iterations=iterations,
                                                 intensity_threshold=intensity_threshold, crop=crop)

        else:
            dvf_image = deform_itk.fast_demons(smooth=smooth, std=std, iterations=iterations,
                                               intensity_threshold=intensity_threshold, step=step, crop=crop)

        self.origin = dvf_image.GetOrigin()
        self.spacing = dvf_image.GetSpacing()
        self.dvf = sitk.GetArrayFromImage(dvf_image)
        self.dimensions = self.dvf.shape[:3]
        self.invalidate_display()

    def compute_components(self, roi_name):
        if self.rois[roi_name] is not None:
            return compute_components(self.rois[roi_name])

        else:
            return None

    def compute_inverted_dvf(self, iterations=50, mean_tolerance=1e-3, max_tolerance=1e-1, set_inverted=True):
        """
        dvf: numpy array, shape (Z, Y, X, 3), displacement in mm, ordered (dx, dy, dz)
        Returns inverted DVF as a numpy array of the same shape/ordering.
        """
        dvf_sitk = sitk.GetImageFromArray(self.dvf, isVector=True)
        dvf_sitk.SetSpacing(self.spacing)
        dvf_sitk.SetOrigin(self.origin)

        inverter = sitk.InvertDisplacementFieldImageFilter()
        inverter.SetMaximumNumberOfIterations(iterations)
        inverter.SetMeanErrorToleranceThreshold(mean_tolerance)
        inverter.SetMaxErrorToleranceThreshold(max_tolerance)
        inverted = sitk.GetArrayFromImage(inverter.Execute(dvf_sitk))

        if set_inverted:
            self.inverted_dvf = inverted
            self._display_cache['inverted_source'] = self.dvf

        return inverted

    def compute_mesh_slice(self, roi_name, plane, position, percent=100, return_pixel=True, use_cache=True):
        """
        Slice current_mov's deformed ROI in the same pixel space as compute_array: (col, row) floats relative to the
        view's top-left corner, pixel centers at integers. Each contour is closed (first point repeated).

        Results are cached per plane, keyed by (roi, percent, position rounded to 1 um, return_pixel). An entry is
        reused only if it was cut from the same deformed mesh object, so a new DVF / rigid / percent / ROI edit (which
        builds a new mesh in get_display_mesh) is picked up automatically. The returned list is the cached object;
        copy it before modifying it in place.
        """
        mesh = self.get_display_mesh(roi_name, percent=percent)
        if mesh is None:
            return [] if return_pixel else None

        grid = Data.image[self.current_ref].display.get_grid(plane, position)

        if use_cache:
            entries = self._slice_cache_for(plane, grid)
            key = (roi_name, float(percent), np.round(np.asarray(position, dtype=float), 3).tobytes(),
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

    @staticmethod
    def correct_dvf_direction(dvf, spacing, origin, matrix):
        """
        Rotates a DVF to a new direction, adjusting vectors and origin so that the DVF
        stays aligned in physical space (rotation about the center of the volume).

        Parameters
        ----------
        dvf : np.ndarray
            Original displacement vector field matrix tensor.
        spacing : list or tuple
            Dimensional voxel sizing array parameters.
        origin : list or tuple
            Starting structural boundary spatial point.
        matrix : np.ndarray
            Target directional rotation transformation matrix mapping properties.

        Returns
        -------
        tuple
            Contains updated (dvf_rotated, spacing, origin_new, dimensions) structures.
        """
        D_new = np.identity(3)
        R = D_new @ np.linalg.inv(matrix)

        center_index = (np.flip(dvf.shape)[1:] - 1) / 2.0
        center_phys = origin + matrix @ (center_index * spacing)

        Z, Y, X, _ = dvf.shape
        dvf_flat = dvf.reshape(-1, 3).T
        dvf_rotated_flat = R @ dvf_flat
        dvf_rotated = dvf_rotated_flat.T.reshape(Z, Y, X, 3)

        center_phys_new = center_phys  # center should not move
        origin_new = center_phys_new - D_new @ (center_index * spacing)

        dimensions = dvf_rotated.shape[0:3]

        return dvf_rotated, spacing, origin_new, dimensions

    def create_image(self, ratio=1):
        """
        Applies combined rigid parameters and displacement fields to resample the moving image volume onto the
        reference grid (full 3D, for export). The display path does NOT use this; see compute_array.

        Parameters
        ----------
        ratio : float, optional
            Displacement scaling factor (0 = rigid only, 1 = full). Defaults to 1.

        Returns
        -------
        SimpleITK.Image
            Resampled and transformed image data volume.
        """
        affine = sitk.AffineTransform(3)
        affine.SetMatrix(self.rigid_matrix[:3, :3].flatten())
        affine.SetTranslation(self.rigid_matrix[:3, 3])

        ref_image = Data.image[self.reference_name].create_sitk_image()
        moving_image = Data.image[self.moving_name].create_sitk_image()

        resampled_image = sitk.Resample(moving_image, ref_image, transform=affine, interpolator=sitk.sitkLinear,
                                        defaultPixelValue=-3001, outputPixelType=moving_image.GetPixelID())

        invert_dvf = sitk.GetImageFromArray((ratio * self.get_inverted_dvf()).astype(np.float64), isVector=True)
        invert_dvf.SetSpacing(self.spacing)
        invert_dvf.SetOrigin(self.origin)
        dis_tx = sitk.DisplacementFieldTransform(invert_dvf)

        return sitk.Resample(resampled_image, dis_tx, sitk.sitkLinear, -3001, resampled_image.GetPixelID())

    def create_roi_image(self, roi_name, sitk_image=True):
        mesh = self.rois[roi_name].triangulate()

        image = vtk.vtkImageData()
        image.SetDimensions(np.flip(Data.image[self.reference_name].dimensions))
        image.SetSpacing(Data.image[self.reference_name].spacing)
        image.SetOrigin(Data.image[self.reference_name].origin)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

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
        mask = array.reshape(Data.image[self.reference_name].dimensions)  # z,y,x

        if sitk_image:
            ref_image = Data.image[self.reference_name].create_sitk_image()
            mask_image = sitk.GetImageFromArray(mask)
            mask_image.CopyInformation(ref_image)

            return mask_image

        else:
            return mask

    def export_image(self, path=None):
        """
        Saves the transformed moving image to a specified file path.

        Parameters
        ----------
        path : str, optional
            Target file system write location destination path. Defaults to None.
        """
        if self.moving_name is not None and path is not None:
            image = self.create_image()

            sitk.WriteImage(image, path)

    def get_display_mesh(self, roi_name, percent=100):
        """
        current_mov's ROI warped into current_ref's world (cached per order / percent / dvf / rigid).
        Forward: moving ROI -> reference world. Inverse: reference ROI -> moving world.
        """
        if self.dvf is None:
            return None

        source = Data.image[self.current_mov].rois.get(roi_name)
        if source is None or source.mesh is None:
            return None

        key = ('mesh', self.inverse, roi_name, float(percent), id(self.dvf), id(source.mesh),
               np.asarray(self.rigid_matrix, dtype=float).tobytes())
        mesh = self._display_cache.get(key)
        if mesh is None:
            mesh = source.mesh.copy()
            if self.inverse:
                mesh.points = self._warp_to_moving(mesh.points, percent=percent)
            else:
                mesh.points = self._warp_to_reference(mesh.points, percent=percent)
            self._display_cache[key] = mesh

        return mesh

    def get_inverted_dvf(self):
        """Inverted DVF, recomputed only when self.dvf has been replaced since the last inversion."""
        if self.inverted_dvf is None or self._display_cache.get('inverted_source') is not self.dvf:
            self.compute_inverted_dvf()

        return self.inverted_dvf

    def get_view_center(self, plane, position):
        """current_ref's slice center in view pixel coords (pixel centers at integers), for crosshair / zoom anchors."""
        base = Data.image[self.current_ref].display
        grid = base.get_grid(plane, position)
        d = base.crosshair - grid['origin']

        return float(d @ grid['x_axis']) / grid['sx'], float(d @ grid['y_axis']) / grid['sy']

    def invalidate_display(self):
        """Drop cached display fields/meshes. Only needed if dvf or rigid_matrix are edited in place."""
        self._display_cache = {}
        self.inverted_dvf = None

    def map_mesh(self, roi_name, percent=100, set_moving=True, return_mesh=False):
        if self.rois[roi_name] is None:
            self.update_rois(roi_name)

        if self.rois[roi_name] is None or self.dvf is None:
            return None

        deformed_points = np.asarray(self.rois[roi_name].points)
        disp = (percent / 100) * self._sample_field_points(self.get_inverted_dvf(), deformed_points)

        self.rigid_rois[roi_name] = copy.deepcopy(self.rois[roi_name])
        self.rigid_rois[roi_name].points = deformed_points + disp

        # undo rigid: forward transform undoes the inv() applied originally
        mapped_mesh = copy.deepcopy(self.rigid_rois[roi_name])
        mapped_mesh.transform(self.rigid_matrix, inplace=True)
        if set_moving:
            Data.image[self.moving_name].rois[roi_name].update_mesh(mapped_mesh)

        if return_mesh:
            return mapped_mesh
        else:
            return None

    def save_deformable(self, path):
        """
        Serializes structural transform attributes and displacement tensor files to disk.

        Parameters
        ----------
        path : str
            File system directory path destination target.
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['rois', 'dvf', 'inverted_dvf', 'rigid_rois',
                                                                        '_display_cache']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'dvf.npy'), self.dvf, allow_pickle=True)

    def set_current_order(self, inverse):
        """
        Flips which image gets deformed onto which, same as Rigid.set_current_order. Does NOT change the registration.
        inverse=False: view built on reference, moving shown deformed onto it (uses the inverted DVF).
        inverse=True:  view built on moving, reference shown deformed onto it (uses the forward DVF directly).
        Callers re-request arrays/slices afterwards.
        """
        self.inverse = bool(inverse)
        if self.inverse:
            self.current_ref, self.current_mov = self.moving_name, self.reference_name
        else:
            self.current_ref, self.current_mov = self.reference_name, self.moving_name

    def update_rois(self, roi_name=None, percent=100):
        """
        Warps visible moving ROI meshes into the reference world: rigid^-1, then + dvf.

        Parameters
        ----------
        roi_name : str, optional
            Target ROI. If None, all instances are updated. Defaults to None.
        percent : float, optional
            Magnitude modifier scaling field vector step sizes. Defaults to 100.
        """
        for name in list(self.rois.keys()):
            if name not in Data.roi_list:
                del self.rois[name]

        for name in Data.roi_list:
            if name not in list(self.rois.keys()):
                self.rois[name] = None

        for name in Data.roi_list:
            if roi_name is None or name == roi_name:
                roi = Data.image[self.moving_name].rois[name]
                if roi.mesh is not None and roi.visible:
                    self.rigid_rois[name] = roi.mesh.transform(np.linalg.inv(self.rigid_matrix), inplace=False)

                    points = np.asarray(self.rigid_rois[name].points)
                    self.rois[name] = copy.deepcopy(self.rigid_rois[name])
                    self.rois[name].points = points + (percent / 100) * self._sample_field_points(self.dvf, points)