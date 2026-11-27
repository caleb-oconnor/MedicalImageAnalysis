"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Image volumes of any modality (CT, MR, PT, US, X-ray, tomo) with their DICOM tags, geometry, ROIs and POIs, plus
    the display helper that reslices them into axis-aligned or oblique 2D views. Images are saved either encrypted
    (pickled metadata + compressed array) or unencrypted (numpy/pickle folders).

    Conventions
    -----------
    * Every image array is (z, y, x), including 2D X-ray (z = 1) and ultrasound frames. Color data is (z, y, x, c).
      ``image.dimensions`` follows the array: (z, y, x).
    * Origin, spacing and world points are (x, y, z). Pixel indices given to ``compute_position`` and returned by
      ``compute_pixel`` are (x, y, z) too.
    * ``image.matrix`` rows are the image axes in world coordinates. VTK and SimpleITK want columns, so they get the
      transpose.
    * ``Display.AXES[plane]`` is (x index, y index, normal index) into (x, y, z): Axial (0, 1, 2), Sagittal (1, 2, 0),
      Coronal (0, 2, 1).
    * A plane's ``position`` is the world point of its slice's top-left pixel, not the slice center.
    * ``Display.matrix`` is the shared view rotation: every plane of an image turns with it. The rows of
      ``image.matrix @ display.matrix.T`` are the view axes.
    * ``rigid_matrix`` (4x4) maps this image's native world to the displayed world, as returned by
      ``Rigid.get_display_matrix``. Identity or None = no registration.
    * ``Image.background`` is the value shown outside the volume (-3001 for images). Subclasses override it, e.g.
      ``Dose.background`` is 0 Gy. Color images always use 0.
    * Images, ROIs and POIs have a permanent integer ``id`` (database key) and a user name that can change. Both look
      them up: ``image.roi[3]`` and ``image.roi['Liver']``. An ROI/POI exists only on an image; a name has the same id
      on every image (``Data.new_roi_id``). Rename through ``Data.rename_image`` / ``rename_roi`` / ``rename_poi``.
    * Markups (``image.moi``) and planes (``image.planeoi``) belong to one image only: unlike ROIs/POIs they are never
      synced to other images. Their ids are unique across every image (``Data.new_moi_id`` / ``new_planeoi_id``),
      their names only on their image. ``image.plane`` is still the acquisition plane; planes of interest live in
      ``image.planeoi``.
    * Models (``image.model``) are meshes derived from this image's ROIs for one computation (clustered surfaces,
      tetrahedral FE meshes). They never replace ``roi.mesh`` and are never synced to other images. Ids are unique
      across every image (``Data.new_model_id``), names only on this image.
    * Groups are an overlay on all of the above: each image has one container per kind (``image.roi_group``,
      ``poi_group``, ``moi_group``, ``planeoi_group``, ``model_group``) of nested groups, looked up by path
      ('Liver/GTV') or id. Like ROIs, every group exists on every image (``Data.match_groups``). An item only carries
      ``group_id`` (None = ungrouped) and stays a normal entry in its registry. Every ``add_*`` method takes an
      optional ``group`` (path, created when missing, or id). ROIs/POIs keep one group on every image: replacing one
      (RTSTRUCT, ``add_roi``) keeps the group it had. A composite ROI group is the parts of one main ROI; making
      and updating the parts is up to the application (``convert_mask`` / ``update_position``, ``Data.set_group``).

Structure:
    - _fused_oblique_sample_kernel: numba kernel that trilinearly samples one oblique slice.
    - Display: slice positions, shared view rotation, crosshair lines and reslicing for one image. Built by Image
      (and by its subclass Dose).
    - Image: array, DICOM tags, geometry, ROIs/POIs and saving/loading.

Examples
--------
>>> import medicalimageanalysis as mia
>>> mia.read_mhd(file='C:/Data/scan.mhd', modality='CT')
>>> image = mia.Data.image[mia.Data.image_list()[0]]
>>> res = image.display.compute_array('Axial', image.display.position['Axial'])
>>> res['array'].shape

Rotate every view of the image 10 degrees about z, then redraw

>>> image.display.update_rotation(r_z=10)
>>> res = image.display.compute_array('Sagittal', image.display.position['Sagittal'])
"""

import os
import copy
import json
import blosc2
import pickle
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk

from numba import njit, prange
from scipy.spatial.transform import Rotation

import vtk
from vtkmodules.util import numpy_support

from ..utils.image.threshold import external
from ..utils.roi.to_contours import mask_to_contours

from .moi import Moi
from .model import Model
from .planeoi import PlaneOI
from .poi import Poi
from .roi import Roi
from ..data import Data
from ..registry import Registry


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _fused_oblique_sample_kernel(volume, h, w, sx, sy,
                                 ox, oy, oz,  # slice top-left origin (world)
                                 xax0, xax1, xax2,  # slice x_axis (unit vector, world)
                                 yax0, yax1, yax2,  # slice y_axis (unit vector, world)
                                 img_ox, img_oy, img_oz,  # image.origin (world), rigid already folded in
                                 Minv,  # 3x3 world->index rotation (rows = index axes in world), rigid folded in
                                 isp0, isp1, isp2,  # image voxel spacing (x, y, z)
                                 background):
    """(h, w) float32 slice of a (z, y, x) volume, sampled trilinearly on an oblique grid.

    One pass per output pixel: builds the world-space sampling point, maps it into the volume's own voxel space, and
    trilinearly samples, all as scalar math inside the compiled loop (no intermediate numpy arrays).

    An axis with a single slice (2D X-ray, one US frame) has no thickness, so a plane would only hit it at exactly
    index 0 and float error from any rigid matrix blanks it. Those axes accept +/- half a voxel and snap onto the slice.
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

            wx = ox + u * xax0 + v * yax0
            wy = oy + u * xax1 + v * yax1
            wz = oz + u * xax2 + v * yax2

            rx = wx - img_ox
            ry = wy - img_oy
            rz = wz - img_oz

            # ii, jj, kk = fractional x, y, z index
            ii = (Minv[0, 0] * rx + Minv[0, 1] * ry + Minv[0, 2] * rz) / isp0
            jj = (Minv[1, 0] * rx + Minv[1, 1] * ry + Minv[1, 2] * rz) / isp1
            kk = (Minv[2, 0] * rx + Minv[2, 1] * ry + Minv[2, 2] * rz) / isp2

            if (kk < -tz or kk > nz - 1 + tz or jj < -ty or jj > ny - 1 + ty or
                    ii < -tx or ii > nx - 1 + tx):
                out[i, j] = background
                continue

            # no-op for normal axes; snaps a single-slice axis onto its slice
            kk = min(max(kk, 0.0), nz - 1.0)
            jj = min(max(jj, 0.0), ny - 1.0)
            ii = min(max(ii, 0.0), nx - 1.0)

            # clamp so the last voxel is sampled (f == n-1 -> t == 1.0) and i0+1 stays in bounds;
            # max(..., 0) and min(i0 + 1, n - 1) only matter for single-slice axes
            z0 = max(min(int(kk), nz - 2), 0)
            y0 = max(min(int(jj), ny - 2), 0)
            x0 = max(min(int(ii), nx - 2), 0)
            z1, y1, x1 = min(z0 + 1, nz - 1), min(y0 + 1, ny - 1), min(x0 + 1, nx - 1)
            fz, fy, fx = kk - z0, jj - y0, ii - x0

            c000 = volume[z0, y0, x0]
            c001 = volume[z0, y0, x1]
            c010 = volume[z0, y1, x0]
            c011 = volume[z0, y1, x1]
            c100 = volume[z1, y0, x0]
            c101 = volume[z1, y0, x1]
            c110 = volume[z1, y1, x0]
            c111 = volume[z1, y1, x1]
            c00 = c000 * (1 - fx) + c001 * fx
            c01 = c010 * (1 - fx) + c011 * fx
            c10 = c100 * (1 - fx) + c101 * fx
            c11 = c110 * (1 - fx) + c111 * fx
            c0 = c00 * (1 - fy) + c01 * fy
            c1 = c10 * (1 - fy) + c11 * fy
            out[i, j] = c0 * (1 - fz) + c1 * fz
    return out


class Display(object):
    """Reslicing and view state for one image: slice positions, the shared view rotation and crosshair lines.

    Built by :class:`Image` (or a subclass such as Dose), not created directly. Each call names its own plane and
    position, so any number of widgets can share one Display.

    Parameters
    ----------
    image : Image
        The image this display reslices.

    Attributes
    ----------
    matrix : (3, 3) ndarray
        Shared view rotation, rebuilt from rotation_totals (Euler 'xyz'). Identity = native orientation.
    rotation_totals : (3,) ndarray
        Total rotation about x, y, z in degrees.
    rotation_center : (3,) ndarray
        World pivot every plane passes through while rotating.
    crosshair : (3,) ndarray
        World point where the crosshair lines cross.
    position : dict
        Plane name -> world position (top-left corner) of that plane's current slice.
    oblique_shape : dict
        Plane name -> (h, w) of the resliced output grid.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> display = mia.Data.image['CT 01'].display
    >>> res = display.compute_array('Axial', display.position['Axial'])
    """
    AXES = {'Axial':    (0, 1, 2),
            'Sagittal': (1, 2, 0),
            'Coronal':  (0, 2, 1)}

    def __init__(self, image):
        self.image = image
        self.matrix = np.eye(3)
        self.spacing = copy.deepcopy(image.spacing)

        self.rotation_totals = np.zeros(3)
        self.rotation_center = np.asarray(image.get_center(), dtype=float)
        self.crosshair = self.rotation_center.copy()

        self.position = {plane: self.image.compute_initial_origin(plane) for plane in self.AXES}
        self.axes = self.AXES     # lowercase alias for callers that read display.axes (e.g. rigid.py)

        # Native direction matrix is orthonormal -> inverse == transpose. Cache it,
        # plus float32 versions of everything the fused kernel needs as scalars, once.
        self._image_matrix_f32 = self.image.matrix.astype(np.float32)
        self._image_origin_f32 = self.image.origin.astype(np.float32)
        self._image_spacing_f32 = self.image.spacing.astype(np.float32)

        # Fixed per-plane oblique output pixel grid (H, W).
        nz, ny, nx = self.image.array.shape[:3]
        self.oblique_shape = {'Axial': (ny, nx), 'Sagittal': (nz, ny), 'Coronal': (nz, nx)}

        self._warm_up_kernel()

    # ---- Compute ----

    @staticmethod
    def _intersection_line(active_plane, crossing_plane):
        """Line where crossing_plane cuts active_plane, in active_plane's pixel coordinates.

        Returns {'pos': (col, row), 'angle': degrees}, or None when the planes are parallel.
        """
        n1, p1 = active_plane['normal'], active_plane['origin']
        n2, p2 = crossing_plane['normal'], crossing_plane['origin']
        u = np.cross(n1, n2)
        u_sq = np.dot(u, u)
        if u_sq < 1e-10:
            return None

        d1, d2 = np.dot(n1, p1), np.dot(n2, p2)
        p0 = (d1 * np.cross(n2, u) + d2 * np.cross(u, n1)) / u_sq
        direction = u / np.linalg.norm(u)

        sx, sy = active_plane['spacing'][0], active_plane['spacing'][1]
        v = p0 - active_plane['origin']
        col0 = np.dot(v, active_plane['x_axis']) / sx
        row0 = np.dot(v, active_plane['y_axis']) / sy
        dcol = np.dot(direction, active_plane['x_axis']) / sx
        drow = np.dot(direction, active_plane['y_axis']) / sy

        return {'pos': (col0, row0), 'angle': np.degrees(np.arctan2(drow, dcol))}

    @staticmethod
    def _to_vtk_scalars(array):
        """2D slice to VTK point scalars; a color slice (h, w, c) keeps its c components."""
        array = np.ascontiguousarray(array)
        flat = array.reshape(-1, array.shape[2]) if array.ndim == 3 else array.ravel(order="C")

        return numpy_support.numpy_to_vtk(flat)

    @staticmethod
    def grid_with_spacing(grid, sx, sy):
        """Same corner, axes and physical extent as grid, with a different pixel size.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> grid = display.get_grid('Axial', display.position['Axial'])
        >>> fine = mia.Display.grid_with_spacing(grid, 0.5, 0.5)
        """
        h, w = grid['shape']
        new = dict(grid)
        new['sx'], new['sy'] = float(sx), float(sy)
        new['shape'] = (int(np.floor((h - 1) * grid['sy'] / sy + 1e-6)) + 1,
                        int(np.floor((w - 1) * grid['sx'] / sx + 1e-6)) + 1)

        return new

    def _get_axis_aligned_array(self, plane, position):
        """Native slice straight from the stored volume (no reslicing), or None when position is outside it."""
        pixel = self.image.compute_pixel(position)
        x_idx, y_idx, z_idx = (int(round(pixel[0])), int(round(pixel[1])), int(round(pixel[2])))
        nz, ny, nx = self.image.array.shape[:3]

        if plane == 'Axial':
            return self.image.array[z_idx, :, :] if 0 <= z_idx < nz else None
        if plane == 'Coronal':
            return self.image.array[:, y_idx, :] if 0 <= y_idx < ny else None
        if plane == 'Sagittal':
            return self.image.array[:, :, x_idx] if 0 <= x_idx < nx else None

        return None

    def compute_array(self, plane, position, matrix_override=None, rigid_matrix=None, grid=None, as_array=True,
                      as_vtk=False, copy_array=False, force_reslice=False, background=None):
        """2D slice of this image for one plane, axis-aligned when possible, otherwise resliced.

        Parameters
        ----------
        plane : str
            'Axial', 'Sagittal' or 'Coronal'.
        position : array_like
            World position (top-left corner) of the slice.
        matrix_override : (3, 3) array_like, optional
            View rotation to use instead of the shared ``self.matrix``.
        rigid_matrix : (4, 4) array_like, optional
            This image's native world -> displayed world (e.g. ``Rigid.get_display_matrix``).
        grid : dict, optional
            Output pixel grid from another Display's :meth:`get_grid` / :meth:`grid_with_spacing`. When given, it fully
            defines the output pixels and this Display only supplies its volume.
        as_array, as_vtk : bool
            Return the slice as a numpy array, a vtkImageData, or both. Color slices keep their channels in VTK.
        copy_array : bool, default False
            Copy the array. The axis-aligned path otherwise returns a view into the volume.
        force_reslice : bool, default False
            Always use the reslice kernel, even when a native slice would do.
        background : float, optional
            Value outside the volume. None = ``image.background`` (-3001 for images, 0 Gy for a dose). Color always
            uses 0.

        Returns
        -------
        dict
            'dimensions', 'spacing', 'pixel_spacing', 'x_axis', 'y_axis', 'normal', 'origin', 'grid', plus 'array'
            and/or 'vtk_image'. A slice outside the volume is filled with background (never None).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> res = display.compute_array('Axial', display.position['Axial'])
        >>> res = display.compute_array('Coronal', display.position['Coronal'], as_vtk=True)
        """
        background = float(self.image.background if background is None else background)
        m = matrix_override if matrix_override is not None else self.matrix
        has_rigid = rigid_matrix is not None and not np.allclose(rigid_matrix, np.eye(4), atol=1e-9)
        rigid_matrix = rigid_matrix if rigid_matrix is not None else np.eye(4)

        xi, yi, ni = self.AXES[plane]
        d = np.asarray(position, dtype=float) - self.image.origin
        on_native_grid = abs(d[xi]) < 1e-3 and abs(d[yi]) < 1e-3
        is_identity = (grid is None and not has_rigid and on_native_grid and
                       np.allclose(self.image.matrix, np.eye(3), atol=1e-6) and
                       np.allclose(m, np.eye(3), atol=1e-6))

        arr = None
        if is_identity and not force_reslice:
            # None past the first/last slice: the kernel below then returns a background slice, so side-by-side and
            # fusion views can keep scrolling beyond this image
            arr = self._get_axis_aligned_array(plane, position)

        if arr is not None:
            h, w = arr.shape[:2]
            grid = self.get_grid(plane, position, matrix_override=m)
            result = {'dimensions': (w, h), 'spacing': self.spacing,
                      'pixel_spacing': (float(self.spacing[xi]), float(self.spacing[yi])),
                      'x_axis': grid['x_axis'].astype(np.float32), 'y_axis': grid['y_axis'].astype(np.float32),
                      'normal': grid['normal'], 'origin': grid['origin'].astype(np.float32), 'grid': grid}

            if as_array:
                result['array'] = arr.copy() if copy_array else arr

            if as_vtk:
                # the native slice keeps its place in the volume, so VTK gets a 3D extent with 1 along the normal
                if plane == 'Axial':
                    dim = [w, h, 1]
                elif plane == 'Sagittal':
                    dim = [1, w, h]
                else:
                    dim = [w, 1, h]
                matrix_reshape = np.linalg.inv(self.image.matrix).reshape(1, 9)[0]

                vtk_out = vtk.vtkImageData()
                vtk_out.SetSpacing(self.image.spacing)
                vtk_out.SetDirectionMatrix(matrix_reshape)
                vtk_out.SetDimensions(dim)
                vtk_out.SetOrigin(position)
                vtk_out.GetPointData().SetScalars(self._to_vtk_scalars(arr))

                result['vtk_image'] = vtk_out

            return result

        if grid is None:
            grid = self.get_grid(plane, position, matrix_override=m)

        x_axis = grid['x_axis'].astype(np.float32)
        y_axis = grid['y_axis'].astype(np.float32)
        normal = grid['normal']
        origin = grid['origin'].astype(np.float32)
        sx, sy = np.float32(grid['sx']), np.float32(grid['sy'])   # keep float32: avoids a numba recompile
        h, w = grid['shape']

        # Fold R^-1 into the native mapping ONCE per call instead of per-pixel:
        # sample this volume at R^-1 (world - t)  ->  index = (M @ R.T) . (world - (R @ img_origin + t))
        Rrot, Rtrans = rigid_matrix[:3, :3], rigid_matrix[:3, 3]
        Mcomb = (self._image_matrix_f32.astype(np.float64) @ Rrot.T).astype(np.float32)
        origin_comb = (Rrot @ self._image_origin_f32.astype(np.float64) + Rtrans).astype(np.float32)

        kernel_args = (h, w, sx, sy,
                       origin[0], origin[1], origin[2],
                       x_axis[0], x_axis[1], x_axis[2],
                       y_axis[0], y_axis[1], y_axis[2],
                       origin_comb[0], origin_comb[1], origin_comb[2],
                       Mcomb,
                       self._image_spacing_f32[0], self._image_spacing_f32[1], self._image_spacing_f32[2])

        volume = self.image.array
        if volume.ndim == 4:
            arr = np.stack([_fused_oblique_sample_kernel(volume[..., c], *kernel_args, 0.0)
                            for c in range(volume.shape[3])], axis=-1)
            if np.issubdtype(volume.dtype, np.integer):
                arr = np.clip(np.rint(arr), 0, np.iinfo(volume.dtype).max).astype(volume.dtype)
        else:
            arr = _fused_oblique_sample_kernel(volume, *kernel_args, background)

        result = {'dimensions': (w, h), 'spacing': self.spacing, 'pixel_spacing': (float(sx), float(sy)),
                  'x_axis': x_axis, 'y_axis': y_axis, 'normal': normal, 'origin': origin, 'grid': grid}
        if as_array:
            result['array'] = arr.copy() if copy_array else arr

        if as_vtk:
            vtk_out = vtk.vtkImageData()
            vtk_out.SetSpacing(sx, sy, 1.0)
            vtk_out.SetDimensions(w, h, 1)
            vtk_out.SetOrigin(*origin)

            direction = np.eye(3)
            direction[:, 0] = x_axis
            direction[:, 1] = y_axis
            direction[:, 2] = normal
            vtk_out.SetDirectionMatrix(direction.reshape(1, 9)[0])

            vtk_out.GetPointData().SetScalars(self._to_vtk_scalars(arr))

            result['vtk_image'] = vtk_out

        return result

    def compute_plane_geometry(self, plane, position, matrix_override=None):
        """Plane origin (= position, the slice's top-left corner), view axes and spacing, used to place the output.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> geometry = display.compute_plane_geometry('Axial', display.position['Axial'])
        >>> geometry['normal']
        """
        m = matrix_override if matrix_override is not None else self.matrix
        effective = self.image.matrix @ m.T
        xi, yi, ni = self.AXES[plane]

        return {'origin': np.asarray(position, dtype=float),
                'x_axis': effective[xi],
                'y_axis': effective[yi],
                'normal': effective[ni],
                'spacing': self.spacing}

    def compute_slice_lines(self, plane, position):
        """Crosshair lines of the other two planes on this plane, keyed by plane name (see _intersection_line).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> lines = display.compute_slice_lines('Axial', display.position['Axial'])
        >>> lines['Sagittal']['pos'], lines['Sagittal']['angle']
        """
        xi, yi, ni = self.AXES[plane]
        effective = self.image.matrix @ self.matrix.T
        active = {'origin': np.asarray(position, dtype=float),
                  'x_axis': effective[xi], 'y_axis': effective[yi], 'normal': effective[ni],
                  'spacing': (self.spacing[xi], self.spacing[yi])}

        lines = {}
        for other in ('Axial', 'Coronal', 'Sagittal'):
            if other == plane:
                continue
            oxi, oyi, oni = self.AXES[other]
            crossing = {'origin': self.crosshair,
                        'x_axis': effective[oxi], 'y_axis': effective[oyi], 'normal': effective[oni],
                        'spacing': (self.spacing[oxi], self.spacing[oyi])}
            line = self._intersection_line(active, crossing)
            if line is not None:
                lines[other] = line

        return lines

    def get_grid(self, plane, position, matrix_override=None):
        """Output pixel grid for this plane: corner, axes, in-plane spacing and shape.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> grid = display.get_grid('Axial', display.position['Axial'])
        >>> grid['shape'], grid['sx'], grid['sy']
        """
        geom = self.compute_plane_geometry(plane, position, matrix_override=matrix_override)
        xi, yi, _ = self.AXES[plane]

        return {'origin': geom['origin'], 'x_axis': geom['x_axis'], 'y_axis': geom['y_axis'],
                'normal': geom['normal'],
                'sx': float(self.spacing[xi]), 'sy': float(self.spacing[yi]),
                'shape': tuple(self.oblique_shape[plane])}

    # ---- Set ----

    def go_to_plane(self, plane, origin, normal, x_axis=None):
        """Rotate the shared view so `plane` lies in a world plane, then scroll every plane through `origin`.

        Used by markups and planes of interest to jump to the view they were drawn on / look straight at a cut. The
        rotation pivot and the crosshair move to origin. Every plane of this image turns with the shared rotation,
        so the other two show the target plane edge-on.

        Parameters
        ----------
        plane : str
            'Axial', 'Sagittal' or 'Coronal': the view that should lie in the world plane.
        origin : (3,) array_like
            World point on the plane; becomes the rotation center and crosshair.
        normal : (3,) array_like
            World normal of the plane.
        x_axis : (3,) array_like, optional
            In-plane direction the view's x axis should follow. None keeps the current x axis as close as possible.

        Returns
        -------
        (3,) ndarray
            The new position (top-left corner) of `plane`.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> position = display.go_to_plane('Axial', [10.0, -40.0, 120.0], [0.0, 0.26, 0.97])
        >>> res = display.compute_array('Axial', position)
        """
        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)

        image_matrix = np.asarray(self.image.matrix, dtype=float)
        xi, yi, ni = self.AXES[plane]

        current = image_matrix @ self.matrix.T
        x_axis = current[xi] if x_axis is None else np.asarray(x_axis, dtype=float)
        x_axis = x_axis - np.dot(x_axis, normal) * normal
        if np.linalg.norm(x_axis) < 1e-6:                    # current x runs along the normal: follow y instead
            x_axis = current[yi] - np.dot(current[yi], normal) * normal
        x_axis = x_axis / np.linalg.norm(x_axis)

        # view axes wanted (rows = view axes, same as image.matrix @ display.matrix.T); keep the image's handedness,
        # since Coronal (x, z, y) is a left-handed set of rows
        effective = np.empty((3, 3))
        effective[xi] = x_axis
        effective[yi] = np.cross(normal, x_axis)
        effective[ni] = normal
        if np.linalg.det(effective) * np.linalg.det(image_matrix) < 0:
            effective[yi] = -effective[yi]

        # effective = image.matrix @ display.matrix.T  ->  display.matrix = effective.T @ image.matrix
        display_matrix = effective.T @ image_matrix
        angles = Rotation.from_matrix(display_matrix).as_euler('xyz', degrees=True)

        self.set_rotation_center(origin)
        self.update_rotation(r_x=angles[0], r_y=angles[1], r_z=angles[2], absolute=True)

        effective = image_matrix @ self.matrix.T
        for name in self.position:
            n = effective[self.AXES[name][2]]
            self.position[name] = self.position[name] + np.dot(origin - self.position[name], n) * n
        self.crosshair = origin.copy()

        return self.position[plane]

    def reset(self):
        """Back to the native orientation: no rotation, pivot and crosshair at the image center, every plane at its
        initial slice.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].display.reset()
        """
        self.matrix = np.eye(3)
        self.rotation_totals = np.zeros(3)
        self.rotation_center = np.asarray(self.image.get_center(), dtype=float)
        self.crosshair = self.rotation_center.copy()
        self.position = {plane: self.image.compute_initial_origin(plane) for plane in self.AXES}

    def set_rotation_center(self, center):
        """Set the rotation pivot (a click, ROI centroid, a Rigid's center, ...) and move the crosshair there.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> image.display.set_rotation_center(image.roi['Liver'].mesh.center)
        """
        self.rotation_center = np.asarray(center, dtype=float)
        self.crosshair = self.rotation_center.copy()

    def update_rotation(self, r_x=0, r_y=0, r_z=0, absolute=True):
        """Rotate the shared view (every plane) and pivot the crosshair and every plane's position to match.

        Parameters
        ----------
        r_x, r_y, r_z : float
            Degrees about the world x, y, z axes.
        absolute : bool, default True
            True: the angles are the total rotation (e.g. read straight off a slider). False: they are added to the
            current rotation_totals.

        Returns
        -------
        (3, 3) ndarray
            The incremental rotation applied this call (old -> new orientation), for :meth:`pivot_position`.

        Notes
        -----
        self.matrix is rebuilt from rotation_totals (Euler 'xyz') each call, not accumulated from deltas. Rotations
        about different axes don't commute, so accumulating would make the orientation path-dependent; rebuilding
        keeps (0, 0, 0) the identity.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> R = display.update_rotation(r_x=0, r_y=0, r_z=15)
        >>> R = display.update_rotation(r_z=1, absolute=False)
        """
        if absolute:
            self.rotation_totals = np.array([r_x, r_y, r_z], dtype=float)
        else:
            self.rotation_totals = self.rotation_totals + np.array([r_x, r_y, r_z], dtype=float)

        old_matrix = self.matrix
        new_matrix = Rotation.from_euler('xyz', self.rotation_totals, degrees=True).as_matrix()
        R = new_matrix @ old_matrix.T  # old is orthonormal -> inverse == transpose

        self.matrix = new_matrix
        self.crosshair = R.dot(self.crosshair - self.rotation_center) + self.rotation_center
        for plane in self.position:
            self.position[plane] = self.pivot_position(self.position[plane], R)

        return R

    def wheel_position(self, plane, position, steps=1, main=True):
        """Move a slice position along its plane's normal by a number of scroll steps (one voxel each).

        main=True also stores it as this plane's position and moves the crosshair with it.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> new_position = display.wheel_position('Axial', display.position['Axial'], steps=-3)
        """
        xi, yi, ni = self.AXES[plane]
        effective = self.image.matrix @ self.matrix.T
        normal = effective[ni]  # row, not effective[:, ni]
        normal_hat = normal / np.linalg.norm(normal)
        delta = normal_hat * self.spacing[ni] * steps
        new_position = np.asarray(position, dtype=float) + delta
        if main:
            self.position[plane] = new_position
            self.crosshair = self.crosshair + delta

        return new_position

    # ---- Other ----

    def _warm_up_kernel(self):
        """Compile the sampling kernel now instead of on the first scroll/rotate.

        Numba compiles a separate specialization per input dtype and layout. Running it once on a throwaway 2x2 array
        of the SAME dtype as the real volume (and a strided channel view for color) moves the ~1-1.5 s one-time cost to
        construction.
        """
        if self.image.array.ndim == 4:
            dummy_vol = np.zeros((2, 2, 2, 3), dtype=self.image.array.dtype)[..., 0]
        else:
            dummy_vol = np.zeros((2, 2, 2), dtype=self.image.array.dtype)
        M = self._image_matrix_f32
        _fused_oblique_sample_kernel(
            dummy_vol, 2, 2, np.float32(1), np.float32(1),
            np.float32(0), np.float32(0), np.float32(0),
            np.float32(1), np.float32(0), np.float32(0),
            np.float32(0), np.float32(1), np.float32(0),
            np.float32(0), np.float32(0), np.float32(0),
            M, np.float32(1), np.float32(1), np.float32(1), 0.0)

    def pivot_position(self, position, R):
        """Rotate one widget's position about rotation_center by R (the value returned by :meth:`update_rotation`).

        update_rotation already moves ``self.position``; use this for positions the caller keeps itself.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> display = mia.Data.image['CT 01'].display
        >>> R = display.update_rotation(r_z=5)
        >>> widget_position = display.pivot_position(widget_position, R)
        """
        position = np.asarray(position, dtype=float)

        return R.dot(position - self.rotation_center) + self.rotation_center


class Image(object):
    """One image volume with its DICOM tags, geometry, ROIs, POIs and display.

    Parameters
    ----------
    image : object
        Reader output (e.g. Read3D, ReadUS, ReadXRay) providing image_set, array, image_name, modality, filepaths,
        sops, plane, spacing, dimensions, orientation, origin, image_matrix, skipped_slice and rgb.

    Attributes
    ----------
    background : float
        Class attribute: value shown outside the volume when it is resliced (-3001). Subclasses override it.
    id : int | None
        Permanent id, written by ``Data.image`` when the image is registered. None until then.
    image_name : str
        User name, mirrored from ``Data.image``. Change it with ``Data.rename_image``.
    roi, poi : Registry
        ROIs / POIs by permanent id or name. Their ids come from ``Data.new_roi_id`` / ``Data.new_poi_id``, so an ROI
        has the same id on every image. Rename them with ``Data.rename_roi`` / ``Data.rename_poi``, not on one image.
    moi, planeoi : Registry
        Markups / planes of interest by permanent id or name. Ids come from ``Data.new_moi_id`` /
        ``Data.new_planeoi_id`` (unique across every image); names are unique on this image only.
    model : Registry
        Models (clustered / tetrahedral meshes derived from this image's ROIs) by permanent id or name. Ids come
        from ``Data.new_model_id`` (unique across every image); names are unique on this image only.
    roi_group, poi_group, moi_group, planeoi_group, model_group : Registry
        Groups of each kind (:class:`~medicalimageanalysis.structure.group.Group`) by permanent id or path. The same
        groups, under the same ids, are on every image (``Data.match_groups``).

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> mia.read_mhd(file='C:/Data/scan.mhd', modality='CT')
    >>> image = mia.Data.image[mia.Data.image_list()[0]]
    >>> image.origin, image.spacing, image.array.shape

    ROIs by name or id

    >>> roi_id = mia.Data.get_roi_id('Liver')
    >>> image.roi['Liver'] is image.roi[roi_id]
    True
    """
    background = -3001.0

    def __init__(self, image):
        self.id = None
        self.roi = Registry(id_source=lambda name, item: Data.new_roi_id(name))
        self.poi = Registry(id_source=lambda name, item: Data.new_poi_id(name))
        self.moi = Registry(id_source=lambda name, item: Data.new_moi_id(name, item))
        self.planeoi = Registry(id_source=lambda name, item: Data.new_planeoi_id(name, item))
        self.model = Registry(id_source=lambda name, item: Data.new_model_id(name, item))
        for kind in Data.GROUP_KINDS:
            setattr(self, kind + '_group', Registry(name_attribute='path',
                                                    id_source=lambda name, item: Data.new_group_id(name, item)))

        self.tags = image.image_set
        self.array = np.ascontiguousarray(image.array)
        self.array_dtype = str(self.array.dtype)

        self.image_name = image.image_name
        self.modality = image.modality

        patient_name = self._first_tag(['PatientName'])
        self.patient_name = 'missing' if patient_name is None else str(patient_name).split('^')[:3]
        self.mrn = str(self._first_tag(['PatientID'], 'missing'))
        self.birthdate = str(self._first_tag(['PatientBirthDate'], ''))
        self.date = self._first_tag(['SeriesDate', 'ContentDate', 'AcquisitionDate', 'StudyDate'], '00000')
        self.time = self._first_tag(['SeriesTime', 'ContentTime', 'AcquisitionTime', 'StudyTime'], '00000')
        self.study_uid = self._first_tag(['StudyInstanceUID'], '00000.00000')
        self.series_uid = self._first_tag(['SeriesInstanceUID'], '00000.00000')
        self.acq_number = self._first_tag(['AcquisitionNumber'], '1')
        self.frame_ref = self._first_tag(['FrameOfReferenceUID'], '00000.00000')
        self.window = self.get_window()

        self.filepaths = image.filepaths
        self.sops = image.sops

        self.plane = image.plane
        self.spacing = image.spacing
        self.dimensions = image.dimensions
        self.orientation = image.orientation
        self.origin = image.origin
        self.matrix = image.image_matrix

        self.skipped_slice = image.skipped_slice
        self.rgb = image.rgb

        self.local_name = None
        self.serialization_version = 1

        self.visual = {'colormap': 'gray', 'bounds': None}
        self.misc = {}

        self.display = Display(self)

    def __getstate__(self):
        """Metadata-only state for pickling (array, ROIs, POIs, markups, planes, models and groups are left out)."""
        state = self.__dict__.copy()
        for key in ('array', 'roi', 'poi', 'moi', 'planeoi', 'model') + tuple(k + '_group' for k in Data.GROUP_KINDS):
            state.pop(key, None)

        return state

    def __setstate__(self, state):
        """Restore metadata-only state with empty array/ROI/POI containers, and point the display back at this image."""
        self.__dict__.update(state)
        self.array = None
        self.roi = Registry(id_source=lambda name, item: Data.new_roi_id(name))
        self.poi = Registry(id_source=lambda name, item: Data.new_poi_id(name))
        self.moi = Registry(id_source=lambda name, item: Data.new_moi_id(name, item))
        self.planeoi = Registry(id_source=lambda name, item: Data.new_planeoi_id(name, item))
        self.model = Registry(id_source=lambda name, item: Data.new_model_id(name, item))
        for kind in Data.GROUP_KINDS:
            setattr(self, kind + '_group', Registry(name_attribute='path',
                                                    id_source=lambda name, item: Data.new_group_id(name, item)))
        if 'id' not in state:
            self.id = None   # metadata saved before ids existed

        # load_metadata copies another instance's state in, so its display would still reslice that instance.
        display = self.__dict__.get('display')
        if display is not None:
            display.image = self

    # ---- Registry ----

    @staticmethod
    def _next_markup_name(registry, prefix):
        """'<prefix> <n>' with the smallest n >= 1 not yet used in registry (e.g. 'Line 3')."""
        n = 1
        while f'{prefix} {n}' in registry:
            n += 1

        return f'{prefix} {n}'

    def _find_annotation(self, kind, name):
        """The ROI (kind='roi') or POI ('poi') with this name on this image, else on another loaded image, else None.
        Used to keep its group when a new object takes its place."""
        registry = getattr(self, kind)
        if name in registry:
            return registry[name]

        for image in Data.image.values():
            if image is not self and name in getattr(image, kind):
                return getattr(image, kind)[name]

        return None

    def _register(self):
        """Store this image in its registry (``Data.image``). Subclasses override it (Dose -> ``Data.dose``).

        A loaded image keeps its saved id unless another image or dose already uses it (then it gets a new one). If
        its name is already registered, it replaces that entry and takes its id. An image that was registered before
        loading (under another name or id) is moved, never held twice.
        """
        for name, item in list(Data.image.items()):
            if item is self and name != self.image_name:
                del Data.image[name]

        if self.image_name in Data.image:
            Data.image[self.image_name] = self
        else:
            taken = self.id is not None and (self.id in Data.image or self.id in Data.dose)
            Data.image.add(self, self.image_name, item_id=None if taken else self.id)

    def _store_annotation(self, kind, name, item):
        """Store a new Roi / Poi under name in self.roi / self.poi, keeping the group of the entry it replaces, or of
        the same ROI/POI on another image, so replacing never ungroups it."""
        previous = self._find_annotation(kind, name)
        if previous is not None:
            item.group_id = getattr(previous, 'group_id', None)

        getattr(self, kind)[name] = item

    def add_model(self, roi_names, kind='tet', name=None, cavities=None, cluster_points=None, edge_length_fac=0.05,
                  edge_length_abs=None, region_edge_lengths=None, epsilon=1e-3, optimize=True, simplify=True,
                  coarsen=False, stop_energy=10.0, num_opt_iter=80, color=None, visible=True, build=False, group=None):
        """Add a model (clustered surface or tetrahedral mesh) built from ROIs of this image. Nothing is synced.

        The recipe is stored on the model; it is only run when ``build=True`` or on ``model.build()``, since
        pytetwild takes seconds.

        Parameters
        ----------
        roi_names : str, int or list of them
            Source ROIs (names or ids), in region order.
        kind : {'tet', 'surface'}, default 'tet'
        name : str, optional
            Name on this image. None = 'Model <n>'.
        cavities, cluster_points, edge_length_fac, edge_length_abs, region_edge_lengths, epsilon, optimize, simplify,
        coarsen, stop_energy, num_opt_iter
            The recipe; see :class:`Model`.
        color : list of int, optional
            RGB, 0-255. Random if None.
        visible : bool, default True
        build : bool, default False
            Run the recipe now.
        group : str or int, optional
            Model group (path or id) to put the model in; a path is created when missing. None = ungrouped.

        Returns
        -------
        Model

        Raises
        ------
        KeyError
            If `name` is already a model on this image.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> fe = image.add_model(['Liver', 'GTV'], name='Liver FE', cluster_points=[2500, 800], edge_length_abs=4.0,
        ...                      region_edge_lengths=[None, 1.5], build=True)
        >>> cpd = image.add_model('Liver', kind='surface', name='Liver CPD', cluster_points=2500, build=True)
        """
        model = Model(self, roi_names, kind=kind, cavities=cavities, cluster_points=cluster_points,
                      edge_length_fac=edge_length_fac, edge_length_abs=edge_length_abs,
                      region_edge_lengths=region_edge_lengths, epsilon=epsilon, optimize=optimize,
                      simplify=simplify, coarsen=coarsen, stop_energy=stop_energy, num_opt_iter=num_opt_iter,
                      color=color, visible=visible, group=group)
        name = self._next_markup_name(self.model, 'Model') if name is None else name
        self.model.add(model, name)

        if build:
            model.build()

        return model

    def add_moi(self, kind, points, view_plane='Axial', position=None, name=None, color=None, visible=True,
                text=None, group=None):
        """Add a markup of interest (line, arrow, angle, polyline, polygon, ellipse or text) drawn on one view.

        The markup keeps that view's plane (current rotation included) and is shown only on views lying in it.
        Nothing is synced to other images.

        Parameters
        ----------
        kind : str
            'line', 'arrow', 'angle', 'polyline', 'polygon', 'ellipse' or 'text' (point counts: ``Moi.POINTS``).
        points : (N, 3) array_like
            World points (mm). From clicks: ``Moi.convert_pixel_to_position(pixels, grid)``.
        view_plane : str, default 'Axial'
            Plane of the view it is drawn on.
        position : (3,) array_like, optional
            That view's position. None = ``self.display.position[view_plane]``.
        name : str, optional
            Name on this image. None = '<Kind> <n>' (e.g. 'Line 3').
        color : list of int, optional
            RGB, 0-255. Defaults to yellow.
        visible : bool, default True
            Show in the views.
        text : str, optional
            Label (the content for kind 'text').
        group : str or int, optional
            Markup group (path or id) to put the markup in; a path is created when missing. None = ungrouped.

        Returns
        -------
        Moi

        Raises
        ------
        KeyError
            If `name` is already a markup on this image.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> position = image.display.position['Axial']
        >>> grid = image.display.get_grid('Axial', position)
        >>> points = mia.Moi.convert_pixel_to_position([[200, 180], [260, 230]], grid)
        >>> ruler = image.add_moi('line', points, view_plane='Axial', position=position)
        >>> ruler.name, ruler.compute_measurement()
        ('Line 1', ('Length', 52.4, 'mm'))
        >>> image.add_moi('text', points[:1], position=position, text='Recurrence?', color=[255, 0, 0])
        """
        position = self.display.position[view_plane] if position is None else position
        grid = self.display.get_grid(view_plane, position)

        moi = Moi(self, kind, points, view_plane=view_plane, plane_origin=grid['origin'],
                  plane_x_axis=grid['x_axis'], plane_y_axis=grid['y_axis'], plane_normal=grid['normal'],
                  color=color, visible=visible, text=text, group=group)
        name = self._next_markup_name(self.moi, kind.capitalize()) if name is None else name
        self.moi.add(moi, name)

        return moi

    def add_planeoi(self, origin, normal, x_axis=None, name=None, extent=None, keep_side=None, group=None,
                     color=None, visible=True):
        """Add a plane of interest (cut plane, reference plane, ...). Nothing is synced to other images.

        Parameters
        ----------
        origin : (3,) array_like
            World point on the plane (mm); the center of its extent.
        normal : (3,) array_like
            Plane normal.
        x_axis : (3,) array_like, optional
            In-plane x direction. None picks one.
        name : str, optional
            Name on this image. None = 'Plane <n>'.
        extent : (2,) array_like, optional
            (width, height) in mm. None = infinite.
        keep_side : {1, -1, None}, optional
            Side kept when cutting (+1 = the side the normal points to).
        group : str or int, optional
            Plane group (path or id) tying planes together (e.g. one resection); a path is created when missing.
            None = ungrouped.
        color : list of int, optional
            RGB, 0-255. Random if None.
        visible : bool, default True
            Show in the views.

        Returns
        -------
        PlaneOI

        Raises
        ------
        KeyError
            If `name` is already a plane on this image.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> cut = image.add_planeoi(origin=[12.0, -80.0, 410.0], normal=[0.0, 0.3, 0.95], extent=(120, 90),
        ...                          keep_side=-1, group='Resection 1', name='Superior cut')

        The plane of the view currently shown, through the crosshair:

        >>> grid = image.display.get_grid('Sagittal', image.display.position['Sagittal'])
        >>> image.add_planeoi(origin=image.display.crosshair, normal=grid['normal'], x_axis=grid['x_axis'])
        """
        plane = PlaneOI(self, origin, normal, x_axis=x_axis, extent=extent, keep_side=keep_side, group=group,
                        color=color, visible=visible)
        name = self._next_markup_name(self.planeoi, 'Plane') if name is None else name
        self.planeoi.add(plane, name)

        return plane

    def add_poi(self, poi_name=None, color=None, visible=False, path=None, point=None, match=True, group=None):
        """Add a point of interest at a world position.

        The POI gets the id poi_name already has on another image, or a new one (``Data.new_poi_id``).

        Parameters
        ----------
        poi_name : str, optional
            Name in ``self.poi``.
        color : list of int, optional
            RGB, 0-255.
        visible : bool, default False
            Show in the views.
        path : str, optional
            Source file.
        point : (3,) array_like, optional
            World position (x, y, z).
        match : bool, default True
            Run ``Data.match_pois`` afterwards. ``match_pois`` itself passes False.
        group : str or int, optional
            POI group (path or id) to put the POI in, on every image; a path is created when missing. None keeps the
            group the POI already has (ungrouped for a new name).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].add_poi('Iso', color=[255, 0, 0], point=[0.0, -20.0, 15.5])
        >>> mia.Data.image['CT 01'].poi['Iso'].id
        """
        self._store_annotation('poi', poi_name, Poi(self, position=point, name=poi_name, color=color, visible=visible,
                                                    filepaths=path))
        if group is not None:
            Data.set_group('poi', poi_name, group)
        if match:
            Data.match_pois()

    def add_roi(self, roi_name=None, color=None, visible=False, path=None, contour=None, plane='Axial', match=True,
                group=None):
        """Add an ROI from world-space contours.

        The ROI gets the id roi_name already has on another image, or a new one (``Data.new_roi_id``).

        Parameters
        ----------
        roi_name : str, optional
            Name in ``self.roi``.
        color : list of int, optional
            RGB, 0-255.
        visible : bool, default False
            Show in the views.
        path : str, optional
            Source file.
        contour : list of (N, 3) array_like, optional
            World-space contour points.
        plane : str, default 'Axial'
            Plane the contours were drawn on.
        match : bool, default True
            Run ``Data.match_rois`` afterwards. ``match_rois`` itself passes False.
        group : str or int, optional
            ROI group (path or id) to put the ROI in, on every image; a path is created when missing. None keeps the
            group the ROI already has (ungrouped for a new name).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].add_roi('Liver', color=[0, 255, 0], contour=contours)
        >>> mia.Data.image['CT 01'].roi['Liver'].id == mia.Data.image['MR 01'].roi['Liver'].id
        True
        """
        self._store_annotation('roi', roi_name, Roi(self, position=contour, name=roi_name, color=color, visible=visible,
                                                    filepaths=path, plane=plane))
        if group is not None:
            Data.set_group('roi', roi_name, group)
        if match:
            Data.match_rois()

    def add_saved_model(self, model, group_map=None):
        """Add a model read with ``Model.load`` / ``Model.from_bytes``, keeping its saved id unless another model on
        any loaded image already uses it; a name already on this image gets the next free '<name> <n>'. Its saved
        group goes through group_map (from :meth:`load_groups` / :meth:`groups_from_json`).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> image.add_saved_model(mia.Model.load(image, 'C:/Data/liver_fe.npz'))
        """
        used = set(Data.model_ids()) | set(self.model.ids())
        item_id = None if model.id is None or model.id in used else model.id
        name = model.name if model.name not in self.model else self._next_markup_name(self.model, model.name)
        model.group_id = self._live_group_id(self.model_group, group_map, getattr(model, 'group_id', None))
        self.model.add(model, name, item_id=item_id)

        return model

    # ---- Group ----

    @staticmethod
    def _live_group_id(registry, group_map, saved_id):
        """Group id in this session of a saved group id: through group_map ({saved: live}, from
        :meth:`groups_from_json`) when given, else kept when registry (this image's container of that kind) has it,
        else None."""
        if saved_id is None:
            return None
        if group_map is not None:
            return group_map.get(int(saved_id))

        return int(saved_id) if int(saved_id) in registry else None

    def groups_from_json(self, text):
        """Add the groups in :meth:`groups_to_json` output to this image and return {saved id: id in this session},
        for the loaders of this image's ROIs, POIs, markups, planes and models.

        A saved group is a group already in the session (same kind) when, in this order:

        1. it has the same id and ``Data.group_id_provider`` is set (ids then come from the database, so they are
           global; without a provider two unrelated files can both hold id 5),
        2. it is a composite of the same main ROI (one composite per main ROI), or
        3. it has the same path.

        That covers a group renamed in the session since the file was saved. The saved group then takes the session
        group's id, path and identity (name, parent, main, method) and keeps this image's saved color, visibility and
        match record. Otherwise it is new and keeps its saved id unless another group uses it. The groups are then
        added to every other image (``Data.match_groups``). Main ROI ids are kept as saved.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> with open('C:/Data/CT 01/groups.json') as f:
        ...     group_map = mia.Data.image['CT 01'].groups_from_json(f.read())
        """
        from .group import Group

        record = json.loads(text)
        used = set()
        for kind in Data.GROUP_KINDS:
            used |= set(Data._group_templates(kind)) | set(getattr(self, kind + '_group').ids())

        mapping = {}
        for kind in Data.GROUP_KINDS:
            registry = getattr(self, kind + '_group')
            groups = [Group.from_json(item_text) for item_text in record.get(kind, [])]
            for group in sorted(groups, key=lambda g: g.path.count('/')):          # parents before children
                saved_id, path = group.id, group.path
                group.parent_id = None if group.parent_id is None else mapping.get(group.parent_id)

                template = None
                if Data.group_id_provider is not None and saved_id is not None and Data.has_group(kind, saved_id):
                    template = Data._group_template(kind, saved_id)
                if template is None and group.main_id is not None:
                    for other in Data._group_templates(kind).values():
                        if other.main_id == group.main_id:
                            template = other
                if template is None and Data.has_group(kind, path):
                    template = Data._group_template(kind, path)
                if template is None and path in registry:
                    template = registry[path]

                if template is not None:
                    mapping[saved_id] = template.id
                    if template.id in registry:                   # already on this image (e.g. made by matching)
                        continue
                    group.copy_identity(template)
                    registry.add(group, template.path, item_id=template.id)
                    continue

                item_id = saved_id if saved_id is not None and saved_id not in used else None
                mapping[saved_id] = registry.add(group, path, item_id=item_id)
                used.add(mapping[saved_id])

        Data.match_groups()

        return mapping

    def groups_to_json(self):
        """Every group of this image, of every kind, as one JSON string: {'schema', 'roi': [...], 'poi': [...], ...},
        parents before children, each entry that group's own ``to_json`` string (including this image's match
        record). Empty groups are saved too: like an ROI, a group exists on every image.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> text = mia.Data.image['CT 01'].groups_to_json()
        """
        record = {'schema': 1}
        for kind in Data.GROUP_KINDS:
            groups = sorted(getattr(self, kind + '_group').values(), key=lambda g: g.path.count('/'))
            record[kind] = [group.to_json() for group in groups]

        return json.dumps(record)

    def load_groups(self, groups_path, key):
        """Decrypt and add the groups written by :meth:`save` (groups.enc) and return {saved id: id in this session}
        for the loaders that follow (see :meth:`groups_from_json`). Call after :meth:`load_metadata` and before
        loading this image's ROIs, POIs, markups and models.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image.load_metadata('C:/Data/CT 01/metadata.enc', key)
        >>> group_map = image.load_groups('C:/Data/CT 01/groups.enc', key)
        >>> image.load_markups('C:/Data/CT 01/markups.enc', key, group_map=group_map)
        >>> image.load_models('C:/Data/CT 01', key, group_map=group_map)
        """
        with open(groups_path, "rb") as f:
            data = f.read()

        nonce, encrypted = data[:12], data[12:]

        return self.groups_from_json(AESGCM(key).decrypt(nonce, encrypted, None).decode('utf-8'))

    # ---- Compute ----

    def _first_tag(self, keywords, default=None):
        """Value of the first DICOM keyword present in the first file's tags, else default."""
        for keyword in keywords:
            if keyword in self.tags[0]:
                return getattr(self.tags[0], keyword)

        return default

    def compute_corner_positions(self):
        """(8, 3) world positions of the corner voxel centers.

        Corners 0-3 go around the first slice: index (0, 0), (x, 0), (x, y), (0, y). Corners 4-7 are the same on the
        last slice, so corner i + 4 sits above corner i.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> corners = mia.Data.image['CT 01'].compute_corner_positions()
        """
        nz, ny, nx = [int(d) - 1 for d in self.dimensions[:3]]
        index = np.array([[0, 0, 0], [nx, 0, 0], [nx, ny, 0], [0, ny, 0],
                          [0, 0, nz], [nx, 0, nz], [nx, ny, nz], [0, ny, nz]], dtype=float)

        pixel_to_position = self.compute_matrix_pixel_to_position()
        return index @ pixel_to_position[:3, :3].T + pixel_to_position[:3, 3]

    def compute_initial_origin(self, plane):
        """World position (top-left corner) of the plane's slice through the middle of the volume.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].compute_initial_origin('Sagittal')
        """
        _, _, ni = Display.AXES[plane]
        dims = self.dimensions  # (z, y, x): xyz index i is dims[2 - i]

        idx_xyz = np.zeros(3)
        idx_xyz[ni] = int(np.round(dims[2 - ni] / 2))  # center index along the through-plane axis
        # in-plane indices stay 0 -> top-left corner

        return self.compute_position(idx_xyz)

    def compute_matrix_pixel_to_position(self):
        """(4, 4) matrix mapping a pixel index (x, y, z, 1) to a world position.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> pixel_to_position = mia.Data.image['CT 01'].compute_matrix_pixel_to_position()
        """
        pixel_to_position = np.identity(4)
        pixel_to_position[:3, :3] = np.asarray(self.matrix, dtype=float).T * np.asarray(self.spacing, dtype=float)
        pixel_to_position[:3, 3] = self.origin

        return pixel_to_position

    def compute_matrix_position_to_pixel(self):
        """(4, 4) matrix mapping a world position (x, y, z, 1) to a fractional pixel index.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> position_to_pixel = mia.Data.image['CT 01'].compute_matrix_position_to_pixel()
        """
        position_to_pixel = np.identity(4)
        position_to_pixel[:3, :3] = np.asarray(self.matrix, dtype=float) / np.asarray(self.spacing, dtype=float)[:, None]
        position_to_pixel[:3, 3] = -position_to_pixel[:3, :3] @ np.asarray(self.origin, dtype=float)

        return position_to_pixel

    def compute_pixel(self, position):
        """Fractional pixel index (x, y, z) of a world position. Round it for a voxel.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].compute_pixel([10.0, -42.5, 103.0])
        """
        return (self.compute_matrix_position_to_pixel() @ [*position, 1])[:3]

    def compute_position(self, xyz):
        """World position of a pixel index (x, y, z), fractional allowed.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].compute_position([256, 256, 40])
        """
        return (self.compute_matrix_pixel_to_position() @ [*xyz, 1])[:3]

    def create_external(self, name='External', color=None, visible=False, filepaths=None, threshold=-250):
        """Body outline ROI from an intensity threshold (contours + mesh).

        Parameters
        ----------
        name : str, default 'External'
            ROI name; created when missing.
        color : list of int, optional
            RGB, 0-255. Defaults to green.
        visible : bool, default False
            Show in the views.
        filepaths : str, optional
            Source file.
        threshold : float, default -250
            Lowest value inside the body (HU for CT).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].create_external(threshold=-300)
        """
        if color is None:
            color = [0, 255, 0]

        if name not in list(self.roi.keys()):
            self._store_annotation('roi', name, Roi(self, name=name, color=color, visible=visible, filepaths=filepaths))

        mask = external(self.array, threshold=threshold, only_mask=True)
        positions = mask_to_contours(mask, self.origin, self.spacing, self.matrix)
        contours = self.roi[name].convert_position_to_pixel(points=positions)

        self.roi[name].contour_pixel = contours
        self.roi[name].contour_position = positions
        self.roi[name].create_mesh(reduction=0.50)

    def create_roi(self, name=None, color=None, visible=False, filepath=None, group=None):
        """Add an empty ROI (to be drawn or filled later). group (path or id) puts it in an ROI group on every image;
        None keeps the group the ROI already has.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].create_roi('GTV', color=[255, 0, 0])
        >>> mia.Data.image['CT 01'].create_roi('Hepatic artery', group='Liver/Vessels')
        """
        self._store_annotation('roi', name, Roi(self, name=name, color=color, visible=visible, filepaths=filepath))
        if group is not None:
            Data.set_group('roi', name, group)
        Data.match_rois()

    def create_sitk_image(self, empty=False):
        """SimpleITK image with this image's geometry. Color becomes a vector image.

        Parameters
        ----------
        empty : bool, default False
            True gives a zero UInt8 image of the same size (e.g. for a mask).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> sitk_image = mia.Data.image['CT 01'].create_sitk_image()
        >>> mask = mia.Data.image['CT 01'].create_sitk_image(empty=True)
        """
        if empty:
            sitk_image = sitk.Image([int(d) for d in self.dimensions[2::-1]], sitk.sitkUInt8)   # (z, y, x) -> (x, y, z)
        else:
            sitk_image = sitk.GetImageFromArray(self.array, isVector=self.array.ndim == 4)

        # rows = axes in this convention; flattening in Fortran order gives sitk its column layout
        sitk_image.SetDirection([float(v) for v in np.asarray(self.matrix).flatten(order='F')])
        sitk_image.SetOrigin([float(v) for v in self.origin])
        sitk_image.SetSpacing([float(v) for v in self.spacing])

        return sitk_image

    def get_aspect(self, slice_plane):
        """Pixel aspect (sx / sy) of a native slice, rounded to 2 decimals.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].get_aspect('Coronal')
        """
        if slice_plane == 'Axial':
            aspect = np.round(self.spacing[0] / self.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.spacing[0] / self.spacing[2], 2)
        else:
            aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        return aspect

    def get_bounds(self):
        """World bounding box of the voxel centers: [x_min, x_max, y_min, y_max, z_min, z_max].

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].get_bounds()
        """
        corners = self.compute_corner_positions()
        low, high = corners.min(axis=0), corners.max(axis=0)

        return [float(low[0]), float(high[0]), float(low[1]), float(high[1]), float(low[2]), float(high[2])]

    def get_center(self, position=True, zyx=False):
        """Center of the volume as a world position or a pixel index.

        Parameters
        ----------
        position : bool, default True
            True gives the world position, False the pixel index.
        zyx : bool, default False
            Return in (z, y, x) order instead of (x, y, z).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].get_center()
        >>> mia.Data.image['CT 01'].get_center(position=False, zyx=True)
        """
        pixel_index = [int(self.dimensions[2] / 2),
                       int(self.dimensions[1] / 2),
                       int(self.dimensions[0] / 2)]

        if position:
            center = self.compute_position(pixel_index)
            if zyx:
                return np.flip(center)
            else:
                return center

        else:
            if zyx:
                return [pixel_index[2], pixel_index[1], pixel_index[0]]
            else:
                return pixel_index

    def get_corner_sides(self):
        """Box surface (6 quads) through the corner voxel centers, for showing the image extent in 3D.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> box = mia.Data.image['CT 01'].get_corner_sides()
        """
        corner_points = self.compute_corner_positions()
        points = [corner_points[0], corner_points[4], corner_points[7], corner_points[3],
                  corner_points[1], corner_points[2], corner_points[6], corner_points[5]]
        faces = [4, 0, 1, 2, 3,
                 4, 4, 5, 6, 7,
                 4, 0, 4, 7, 1,
                 4, 3, 2, 6, 5,
                 4, 0, 3, 5, 4,
                 4, 1, 7, 6, 2]

        return pv.PolyData(points, faces)

    def get_specific_tag(self, tag):
        """A DICOM tag from the first file, or None.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].get_specific_tag('SliceThickness')
        >>> mia.Data.image['CT 01'].get_specific_tag((0x0018, 0x0050))
        """
        if tag in self.tags[0]:
            return self.tags[0][tag]
        else:
            return None

    def get_specific_tag_on_all_files(self, tag):
        """A DICOM tag from every file (list), or None when the first file doesn't have it.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].get_specific_tag_on_all_files('ImagePositionPatient')
        """
        if tag in self.tags[0]:
            return [t[tag] for t in self.tags]
        else:
            return None

    def get_window(self):
        """[lower, upper] display window from WindowCenter/WindowWidth, else the array's min and max.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].get_window()
        [-160, 240]
        """
        if (0x0028, 0x1050) in self.tags[0] and (0x0028, 0x1051) in self.tags[0]:
            center = self.tags[0].WindowCenter
            width = self.tags[0].WindowWidth

            if not isinstance(center, float):
                center = center[0]

            if not isinstance(width, float):
                width = width[0]

            return [int(center) - int(np.round(width / 2)), int(center) + int(np.round(width / 2))]

        elif self.array is not None:
            return [np.min(self.array), np.max(self.array)]

        else:
            return [0, 1]

    # ---- Load ----

    def input_mhd(self, filename, roi_names, values, plane='Axial'):
        """Add ROIs from a MetaImage (.mhd/.raw) label map: each value in values becomes the ROI of the same position.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].input_mhd('C:/Data/labels.mhd', ['Liver', 'Spleen'], [1, 2])
        """
        roi_image = sitk.ReadImage(filename)
        roi_array = sitk.GetArrayFromImage(roi_image)
        for ii, roi_name in enumerate(roi_names):
            if roi_name not in list(self.roi.keys()):
                self._store_annotation('roi', roi_name, Roi(self, name=roi_name, visible=True, filepaths=filename,
                                                            plane=plane))

            roi_mask = roi_array == values[ii]
            self.roi[roi_name].convert_mask(roi_mask)

    def input_rtstruct(self, rtstruct):
        """Add the ROIs and POIs of a read RTSTRUCT. Existing ones are kept unless they have no contours. A replaced
        ROI/POI keeps its group.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].input_rtstruct(rtstruct)
        """
        for ii, roi_name in enumerate(rtstruct.roi_names):
            if roi_name not in list(self.roi.keys()) or self.roi[roi_name].contour_position is None:
                self._store_annotation('roi', roi_name,
                                       Roi(self, position=rtstruct.contours[ii], name=roi_name,
                                           color=rtstruct.roi_colors[ii], visible=False, filepaths=rtstruct.filepaths))

        for ii, poi_name in enumerate(rtstruct.poi_names):
            if poi_name not in list(self.poi.keys()) or self.poi[poi_name].point_position is None:
                self._store_annotation('poi', poi_name,
                                       Poi(self, position=rtstruct.points[ii], name=poi_name,
                                           color=rtstruct.poi_colors[ii], visible=False, filepaths=rtstruct.filepaths))

        Data.match_rois()
        Data.match_pois()

    def load_array(self, array_path, key):
        """Decrypt and decompress the array written by :meth:`save`.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image.load_array('C:/Data/CT 01/array.enc', key)
        """
        with open(array_path, "rb") as f:
            data = f.read()

        nonce, encrypted = data[:12], data[12:]
        raw = blosc2.decompress(AESGCM(key).decrypt(nonce, encrypted, None))
        self.array = np.frombuffer(raw, dtype=self.array_dtype).reshape(self.dimensions).copy()

    def load_markups(self, markups_path, key, group_map=None):
        """Decrypt and add the markups and planes written by :meth:`save` (markups.enc). Call after
        :meth:`load_metadata`, so the image is registered and ids are checked against every loaded image. group_map
        (from :meth:`load_groups`) maps their saved groups.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image.load_metadata('C:/Data/CT 01/metadata.enc', key)
        >>> image.load_array('C:/Data/CT 01/array.enc', key)
        >>> group_map = image.load_groups('C:/Data/CT 01/groups.enc', key)
        >>> image.load_markups('C:/Data/CT 01/markups.enc', key, group_map=group_map)
        """
        with open(markups_path, "rb") as f:
            data = f.read()

        nonce, encrypted = data[:12], data[12:]
        self.markups_from_json(AESGCM(key).decrypt(nonce, encrypted, None).decode('utf-8'), group_map=group_map)

    def load_metadata(self, metadata_path, key):
        """Decrypt the metadata written by :meth:`save` into this image and register it (``Data.image``, or
        ``Data.dose`` for a Dose).

        Call :meth:`load_array` afterwards for the pixels.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Image.__new__(mia.Image)
        >>> image.load_metadata('C:/Data/CT 01/metadata.enc', key)
        >>> image.load_array('C:/Data/CT 01/array.enc', key)
        """
        with open(metadata_path, "rb") as f:
            data = f.read()

        nonce, encrypted = data[:12], data[12:]
        loaded = pickle.loads(AESGCM(key).decrypt(nonce, encrypted, None))
        self.__setstate__(loaded.__dict__)

        self._register()

    def load_models(self, folder_path, key, group_map=None):
        """Decrypt and add every model written by :meth:`save` (model_<id>.enc files). Call after
        :meth:`load_metadata`; group_map (from :meth:`load_groups`) maps their saved groups.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image.load_metadata('C:/Data/CT 01/metadata.enc', key)
        >>> image.load_array('C:/Data/CT 01/array.enc', key)
        >>> image.load_models('C:/Data/CT 01', key)
        """
        aes = AESGCM(key)
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.startswith('model_') and file_name.endswith('.enc'):
                with open(os.path.join(folder_path, file_name), "rb") as f:
                    data = f.read()
                nonce, encrypted = data[:12], data[12:]
                self.add_saved_model(Model.from_bytes(self, aes.decrypt(nonce, encrypted, None)), group_map=group_map)

    def load_unencrypted_image(self, image_path, rois=True, pois=True):
        """Load an image folder written by :meth:`save_unencrypted`, optionally with its ROIs and POIs.

        The image is registered in ``Data.image`` straight after its attributes are restored, under its saved id and
        name (a new id when the saved one is taken in this session), the same as :meth:`load_metadata`. It does not
        need to be registered first; if it was, it is moved, not held twice. The groups (groups.json) are added next,
        so every item finds its group, and at the end every image gets its ROIs / POIs (``Data.match_rois`` /
        ``match_pois``).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image.load_unencrypted_image('C:/Data/CT 01')
        """
        self.array = np.load(os.path.join(image_path, 'array.npy'), allow_pickle=True)
        self.tags = np.load(os.path.join(image_path, 'tags.npy'), allow_pickle=True)
        info = pd.read_pickle(os.path.join(image_path, 'info.p'), )
        for column in list(info.columns):
            setattr(self, column, info.at[0, column])

        # registered now (saved id and name, see _register), so every lookup by image.id below finds this image
        self._register()

        # groups first, so every ROI, POI, markup and model below finds its group
        group_map = None
        groups_path = os.path.join(image_path, 'groups.json')
        if os.path.exists(groups_path):
            with open(groups_path) as f:
                group_map = self.groups_from_json(f.read())

        if rois:
            roi_names = os.listdir(os.path.join(image_path, 'ROIs'))
            for name in roi_names:
                self.load_unencrypted_rois(os.path.join(image_path, 'ROIs', name), group_map=group_map)

        if pois:
            roi_names = os.listdir(os.path.join(image_path, 'POIs'))
            for name in roi_names:
                self.load_unencrypted_pois(os.path.join(image_path, 'POIs', name), group_map=group_map)

        markups_path = os.path.join(image_path, 'markups.json')
        if os.path.exists(markups_path):
            with open(markups_path) as f:
                self.markups_from_json(f.read(), group_map=group_map)

        models_path = os.path.join(image_path, 'Models')
        if os.path.exists(models_path):
            for file_name in sorted(os.listdir(models_path)):
                self.add_saved_model(Model.load(self, os.path.join(models_path, file_name)), group_map=group_map)

        Data.match_rois()                   # every image gets this image's ROIs / POIs (and groups), as on any read
        Data.match_pois()

    def load_unencrypted_pois(self, poi_path, group_map=None):
        """Load one POI folder, keeping its saved id when it doesn't clash.

        A name already on this image gets a _1, _2, ... suffix and a new id. A name already on another image takes
        that image's id, so one name never has two ids. The saved id is also dropped when another POI already uses it.
        Folders saved before ids existed get a new id.

        A POI already on another image keeps that image's group; otherwise the saved group (group_id.npy) is used,
        mapped through group_map (from :meth:`groups_from_json`; :meth:`load_unencrypted_image` passes it).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image.load_unencrypted_pois('C:/Data/CT 01/POIs/3')
        """
        name = str(np.load(os.path.join(poi_path, 'name.npy'), allow_pickle=True))

        poi_id = None
        if os.path.exists(os.path.join(poi_path, 'id.npy')):
            poi_id = int(np.load(os.path.join(poi_path, 'id.npy'), allow_pickle=True))

        existing_pois = list(self.poi.keys())
        if name in existing_pois:
            poi_id = None
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_pois:
                    name = new_name
                    n = -1

        if Data.has_poi(name):                                    # on another image: one id per name
            poi_id = Data.get_poi_id(name)
        elif poi_id is not None and Data.has_poi(poi_id):           # saved id now belongs to another POI
            poi_id = None
        inherited = self._find_annotation('poi', name)            # same POI on another image: one group per POI
        self.poi.add(Poi(self), name, item_id=poi_id)        # None -> Data.new_poi_id
        if inherited is not None:
            self.poi[name].group_id = getattr(inherited, 'group_id', None)
        elif os.path.exists(os.path.join(poi_path, 'group_id.npy')):
            saved = int(np.load(os.path.join(poi_path, 'group_id.npy'), allow_pickle=True))
            self.poi[name].group_id = self._live_group_id(self.poi_group, group_map, saved)
        self.poi[name].visible = bool(np.load(os.path.join(poi_path, 'visible.npy'), allow_pickle=True))
        color = np.load(os.path.join(poi_path, 'color.npy'), allow_pickle=True)
        self.poi[name].color = None if color.ndim == 0 else list(color)          # a POI saved with no color
        self.poi[name].filepaths = str(np.load(os.path.join(poi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(poi_path, 'point_position.npy')):
            self.poi[name].point_position = list(np.load(os.path.join(poi_path, 'point_position.npy'),
                                                         allow_pickle=True))

    def load_unencrypted_rois(self, roi_path, group_map=None):
        """Load one ROI folder, keeping its saved id when it doesn't clash.

        A name already on this image gets a _1, _2, ... suffix and a new id. A name already on another image takes
        that image's id, so one name never has two ids. The saved id is also dropped when another ROI already uses it.
        Folders saved before ids existed get a new id.

        An ROI already on another image keeps that image's group; otherwise the saved group (group_id.npy) is used,
        mapped through group_map (from :meth:`groups_from_json`; :meth:`load_unencrypted_image` passes it).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image.load_unencrypted_rois('C:/Data/CT 01/ROIs/1')
        """
        name = str(np.load(os.path.join(roi_path, 'name.npy'), allow_pickle=True))

        roi_id = None
        if os.path.exists(os.path.join(roi_path, 'id.npy')):
            roi_id = int(np.load(os.path.join(roi_path, 'id.npy'), allow_pickle=True))

        existing_rois = list(self.roi.keys())
        if name in existing_rois:
            roi_id = None
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_rois:
                    name = new_name
                    n = -1

        if Data.has_roi(name):                                    # on another image: one id per name
            roi_id = Data.get_roi_id(name)
        elif roi_id is not None and Data.has_roi(roi_id):           # saved id now belongs to another ROI
            roi_id = None
        inherited = self._find_annotation('roi', name)            # same ROI on another image: one group per ROI
        self.roi.add(Roi(self), name, item_id=roi_id)        # None -> Data.new_roi_id
        if inherited is not None:
            self.roi[name].group_id = getattr(inherited, 'group_id', None)
        elif os.path.exists(os.path.join(roi_path, 'group_id.npy')):
            saved = int(np.load(os.path.join(roi_path, 'group_id.npy'), allow_pickle=True))
            self.roi[name].group_id = self._live_group_id(self.roi_group, group_map, saved)
        self.roi[name].visible = bool(np.load(os.path.join(roi_path, 'visible.npy'), allow_pickle=True))
        self.roi[name].color = list(np.load(os.path.join(roi_path, 'color.npy'), allow_pickle=True))
        self.roi[name].filepaths = str(np.load(os.path.join(roi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(roi_path, 'contour_position.npy')):
            self.roi[name].contour_position = list(np.load(os.path.join(roi_path, 'contour_position.npy'),
                                                            allow_pickle=True))

    def markups_from_json(self, text, group_map=None):
        """Add the markups and planes in :meth:`markups_to_json` output to this image.

        A saved id is kept unless another markup (plane) on any loaded image already uses it; a name already on this
        image gets the next free '<name> <n>'. Saved groups go through group_map (from :meth:`groups_from_json`); with
        None a saved group id is kept when this image has that group.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> with open('C:/Data/CT 01/markups.json') as f:
        ...     mia.Data.image['CT 01'].markups_from_json(f.read())
        """
        record = json.loads(text)

        for kind, markup_class in (('moi', Moi), ('planeoi', PlaneOI)):
            registry = getattr(self, kind)
            used = set(getattr(Data, kind + '_ids')()) | set(registry.ids())

            for item_text in record.get(kind, []):
                item = markup_class.from_json(self, item_text)
                item.group_id = self._live_group_id(getattr(self, kind + '_group'), group_map, item.group_id)
                item_id = None if item.id is None or item.id in used else item.id
                name = item.name if item.name not in registry else self._next_markup_name(registry, item.name)
                used.add(registry.add(item, name, item_id=item_id))

    # ---- Save ----

    def markups_to_json(self):
        """Every markup and plane of this image as one JSON string: {'schema', 'moi': [...], 'planeoi': [...]}, where
        each entry is that item's own ``to_json`` string (so it can also be stored one database row per item). Their
        groups are saved by :meth:`groups_to_json`.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> text = mia.Data.image['CT 01'].markups_to_json()
        """
        return json.dumps({'schema': 1,
                           'moi': [moi.to_json() for moi in self.moi.values()],
                           'planeoi': [plane.to_json() for plane in self.planeoi.values()]})

    def save(self, folder_path, key, clevel=5):
        """Write encrypted metadata.enc (pickled, no array/ROIs/POIs), array.enc (blosc2 ZSTD), groups.enc (groups of
        every kind, JSON), markups.enc (markups and planes, JSON) and one model_<id>.enc per model (npz) into
        folder_path. Load the groups back first (:meth:`load_groups`).

        Parameters
        ----------
        folder_path : str
            Existing folder.
        key : bytes
            AES-GCM key (16, 24 or 32 bytes).
        clevel : int, default 5
            Compression level, 0-9.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].save('C:/Data/CT 01', key)
        """
        aes = AESGCM(key)

        metadata = pickle.dumps(self)
        nonce = os.urandom(12)
        metadata = nonce + aes.encrypt(nonce, metadata, None)
        with open(os.path.join(folder_path, 'metadata.enc'), "wb") as f:
            f.write(metadata)

        raw = self.array.tobytes(order="C")
        compressed = blosc2.compress(raw, typesize=self.array.itemsize,
                                     codec=blosc2.Codec.ZSTD, clevel=clevel, filter=blosc2.Filter.BITSHUFFLE)

        nonce = os.urandom(12)
        encrypted = nonce + aes.encrypt(nonce, compressed, None)
        with open(os.path.join(folder_path, 'array.enc'), "wb") as f:
            f.write(encrypted)

        nonce = os.urandom(12)
        encrypted = nonce + aes.encrypt(nonce, self.groups_to_json().encode('utf-8'), None)
        with open(os.path.join(folder_path, 'groups.enc'), "wb") as f:
            f.write(encrypted)

        nonce = os.urandom(12)
        encrypted = nonce + aes.encrypt(nonce, self.markups_to_json().encode('utf-8'), None)
        with open(os.path.join(folder_path, 'markups.enc'), "wb") as f:
            f.write(encrypted)

        for model_id in self.model.ids():
            nonce = os.urandom(12)
            encrypted = nonce + aes.encrypt(nonce, self.model[model_id].to_bytes(), None)
            with open(os.path.join(folder_path, f'model_{model_id}.enc'), "wb") as f:
                f.write(encrypted)

    def save_unencrypted(self, path, rois=True, pois=True):
        """Write the image to a folder: info.p (attributes), tags.npy, array.npy, groups.json, markups.json (markups
        and planes), plus ROIs/, POIs/ and Models/ folders.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].save_unencrypted('C:/Data/CT 01')
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names
                        if name not in ['roi', 'poi', 'moi', 'planeoi', 'model', 'tags', 'array', 'display']
                        and not name.endswith('_group')]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'tags.npy'), self.tags, allow_pickle=True)
        np.save(os.path.join(path, 'array.npy'), self.array, allow_pickle=True)
        with open(os.path.join(path, 'groups.json'), 'w') as f:
            f.write(self.groups_to_json())
        with open(os.path.join(path, 'markups.json'), 'w') as f:
            f.write(self.markups_to_json())

        if len(self.model) > 0:
            os.mkdir(os.path.join(path, 'Models'))
            for model_id in self.model.ids():
                self.model[model_id].save(os.path.join(path, 'Models', f'{model_id}.npz'))

        if rois:
            self.save_unencrypted_rois(path, create_main_folder=True)

        if pois:
            self.save_unencrypted_pois(path, create_main_folder=True)

    def save_unencrypted_pois(self, path, create_main_folder=False):
        """Write one folder per POI (id, name, visible, color, filepaths, point_position, group_id when grouped) under
        path.

        Folders are named by the permanent POI id, so renaming a POI never moves its folder.
        create_main_folder=True puts them in a new 'POIs' subfolder.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].save_unencrypted_pois('C:/Data/CT 01', create_main_folder=True)
        """
        if create_main_folder:
            path = os.path.join(path, 'POIs')
            os.mkdir(path)

        for poi_id in self.poi.ids():
            poi = self.poi[poi_id]
            poi_path = os.path.join(path, str(poi_id))
            os.mkdir(poi_path)

            np.save(os.path.join(poi_path, 'id.npy'), poi_id, allow_pickle=True)
            np.save(os.path.join(poi_path, 'name.npy'), poi.name, allow_pickle=True)
            np.save(os.path.join(poi_path, 'visible.npy'), poi.visible, allow_pickle=True)
            np.save(os.path.join(poi_path, 'color.npy'), poi.color, allow_pickle=True)
            np.save(os.path.join(poi_path, 'filepaths.npy'), poi.filepaths, allow_pickle=True)
            np.save(os.path.join(poi_path, 'point_position.npy'), poi.point_position, allow_pickle=True)
            if getattr(poi, 'group_id', None) is not None:
                np.save(os.path.join(poi_path, 'group_id.npy'), poi.group_id, allow_pickle=True)

    def save_unencrypted_rois(self, path, create_main_folder=False):
        """Write one folder per ROI (id, name, visible, color, filepaths, contour_position, group_id when grouped)
        under path.

        Folders are named by the permanent ROI id, so renaming an ROI never moves its folder.
        create_main_folder=True puts them in a new 'ROIs' subfolder.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].save_unencrypted_rois('C:/Data/CT 01', create_main_folder=True)
        """
        if create_main_folder:
            path = os.path.join(path, 'ROIs')
            os.mkdir(path)

        for roi_id in self.roi.ids():
            roi = self.roi[roi_id]
            roi_path = os.path.join(path, str(roi_id))
            os.mkdir(roi_path)

            np.save(os.path.join(roi_path, 'id.npy'), roi_id, allow_pickle=True)
            np.save(os.path.join(roi_path, 'name.npy'), roi.name, allow_pickle=True)
            np.save(os.path.join(roi_path, 'visible.npy'), roi.visible, allow_pickle=True)
            np.save(os.path.join(roi_path, 'color.npy'), roi.color, allow_pickle=True)
            np.save(os.path.join(roi_path, 'filepaths.npy'), roi.filepaths, allow_pickle=True)
            if getattr(roi, 'group_id', None) is not None:
                np.save(os.path.join(roi_path, 'group_id.npy'), roi.group_id, allow_pickle=True)
            if roi.contour_position is not None:
                np.save(os.path.join(roi_path, 'contour_position.npy'),
                        np.array(roi.contour_position, dtype=object),
                        allow_pickle=True)

    # ---- Other ----

    def reset_display(self):
        """Clear the view rotation and put every plane back at its initial slice (see :meth:`Display.reset`).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].reset_display()
        """
        self.display.reset()

    def retrieve_angles(self, order='ZXY'):
        """Euler angles (degrees) of the display's view rotation.

        The display builds its rotation from Euler 'xyz', so the angles differ from rotation_totals unless order='xyz'.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].retrieve_angles()
        """
        rotation = Rotation.from_matrix(self.display.matrix[:3, :3])

        return rotation.as_euler(order, degrees=True)