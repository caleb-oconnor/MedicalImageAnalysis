"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Medical imaging visualization library handling coordinate transformations,
    off-axis slice reslicing, and ROI/POI annotations for CT/MR datasets.

Structure:
    - Display: Manages slice viewing planes, coordinate transforms, and VTK reslicing.
    - Image: Holds volumetric arrays, metadata tags, and geometric properties.
"""

import os
import copy
import time
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
from ..utils.roi.contour import contours_from_mask

from .poi import Poi
from .roi import Roi
from ..data import Data


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _fused_oblique_sample_kernel(volume, h, w, sx, sy,
                                 ox, oy, oz,  # slice top-left origin (world)
                                 xax0, xax1, xax2,  # slice x_axis (unit vector, world)
                                 yax0, yax1, yax2,  # slice y_axis (unit vector, world)
                                 img_ox, img_oy, img_oz,  # image.origin (world)
                                 Minv,  # 3x3, image.matrix inverse (== transpose)
                                 isp0, isp1, isp2,  # image voxel spacing (i, j, k)
                                 background):
    """
    Does EVERYTHING in one pass per output pixel: builds the world-space sampling
    point, projects it into the volume's own (possibly oblique) voxel space, and
    trilinearly samples -- all as scalar math inside the compiled loop. No
    intermediate numpy arrays (mesh_u, world_x, rel_x, ijk_i, coords_x, ...) are
    ever materialized. Each of those used to be a separate numpy call; on some
    machines each such call carries ~1ms of fixed overhead (temp allocation +
    dispatch) regardless of array size, and ~20 of them per slice was the actual
    bottleneck -- not the interpolation itself.
    """
    out = np.empty((h, w), dtype=np.float32)
    nz, ny, nx = volume.shape
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

            ii = (Minv[0, 0] * rx + Minv[0, 1] * ry + Minv[0, 2] * rz) / isp0
            jj = (Minv[1, 0] * rx + Minv[1, 1] * ry + Minv[1, 2] * rz) / isp1
            kk = (Minv[2, 0] * rx + Minv[2, 1] * ry + Minv[2, 2] * rz) / isp2

            if kk < 0 or kk >= nz - 1 or jj < 0 or jj >= ny - 1 or ii < 0 or ii >= nx - 1:
                out[i, j] = background
                continue

            z0 = int(kk);
            y0 = int(jj);
            x0 = int(ii)
            z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
            fz, fy, fx = kk - z0, jj - y0, ii - x0

            c000 = volume[z0, y0, x0];
            c001 = volume[z0, y0, x1]
            c010 = volume[z0, y1, x0];
            c011 = volume[z0, y1, x1]
            c100 = volume[z1, y0, x0];
            c101 = volume[z1, y0, x1]
            c110 = volume[z1, y1, x0];
            c111 = volume[z1, y1, x1]
            c00 = c000 * (1 - fx) + c001 * fx;
            c01 = c010 * (1 - fx) + c011 * fx
            c10 = c100 * (1 - fx) + c101 * fx;
            c11 = c110 * (1 - fx) + c111 * fx
            c0 = c00 * (1 - fy) + c01 * fy;
            c1 = c10 * (1 - fy) + c11 * fy
            out[i, j] = c0 * (1 - fz) + c1 * fz
    return out


class Display(object):
    def __init__(self, image):
        self.image = image
        self.matrix = np.eye(3)
        self.spacing = copy.deepcopy(image.spacing)

        self.rotation_totals = np.zeros(3)
        self.rotation_center = np.asarray(image.get_center(), dtype=float)
        self.crosshair = self.rotation_center.copy()

        self.position = {'Axial':    self.image.compute_initial_origin('Axial'),
                         'Sagittal': self.image.compute_initial_origin('Sagittal'),
                         'Coronal':  self.image.compute_initial_origin('Coronal')}
        self.axes = {'Axial':    (0, 1, 2),
                     'Sagittal': (1, 2, 0),
                     'Coronal':  (0, 2, 1)}

        # Native direction matrix is orthonormal -> inverse == transpose. Cache it,
        # plus float32 versions of everything the fused kernel needs as scalars, once.
        self._image_matrix_inv_f32 = np.linalg.inv(self.image.matrix).astype(np.float32)
        self._image_origin_f32 = self.image.origin.astype(np.float32)
        self._image_spacing_f32 = self.image.spacing.astype(np.float32)

        # Fixed per-plane oblique output pixel grid (H, W).
        nz, ny, nx = self.image.array.shape
        self.oblique_shape = {'Axial': (ny, nx), 'Sagittal': (nz, ny), 'Coronal': (nz, nx)}

        self._warm_up_kernel()

    @staticmethod
    def _axes_from_normal(normal, up=(0.0, 1.0, 0.0)):
        """Escape hatch: build an orthonormal basis from a bare normal, for reference views that intentionally ignore
        shared rotation."""

        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        up = np.asarray(up, dtype=float)

        if abs(np.dot(normal, up)) > 0.999:
            up = np.array([1.0, 0.0, 0.0])

        x_axis = np.cross(up, normal); x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)

        return np.stack([x_axis, y_axis, normal], axis=1)   # columns = (x, y, normal)

    @staticmethod
    def _build_vtk_slice_image(array_2d, origin_3d, direction_matrix_3x3, spacing_2d):
        vtk_img = vtk.vtkImageData()

        h, w = array_2d.shape
        vtk_img.SetDimensions(w, h, 1)
        vtk_img.SetSpacing(spacing_2d[0], spacing_2d[1], 1.0)
        vtk_img.SetOrigin(origin_3d[0], origin_3d[1], origin_3d[2])
        vtk_img.SetDirectionMatrix(direction_matrix_3x3.ravel())
        vtk_scalars = numpy_support.numpy_to_vtk(array_2d.ravel(order="C"), deep=False, array_type=vtk.VTK_FLOAT)
        vtk_img.GetPointData().SetScalars(vtk_scalars)

        return vtk_img

    def _compute_plane_geometry_for_lines(self, plane):
        """
        Used by compute_slice_lines: origin = rotation_center (the shared pivot every plane passes through), NOT the
        widget's top-left reslice origin. x_axis/y_axis/normal still come live from self.matrix, so lines rotate
        correctly with any applied rotation.
        """

        xi, yi, ni = self.axes[plane]

        return {'origin': self.rotation_center,
                'x_axis': self.matrix[:, xi],
                'y_axis': self.matrix[:, yi],
                'normal': self.matrix[:, ni],
                'spacing': self.spacing }

    def _get_axis_aligned_array(self, plane, position):
        pixel = self.image.compute_pixel(position)
        x_idx, y_idx, z_idx = (int(round(pixel[0])), int(round(pixel[1])), int(round(pixel[2])))

        if plane == 'Axial' and 0 <= z_idx < self.image.array.shape[0]:
            return self.image.array[z_idx, :, :]
        elif plane == 'Coronal' and 0 <= y_idx < self.image.array.shape[1]:
            return self.image.array[:, y_idx, :]
        elif  0 <= x_idx < self.image.array.shape[2]:
            return self.image.array[:, :, x_idx]

        return None

    @staticmethod
    def _intersection_line(active_plane, crossing_plane):
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

    def _warm_up_kernel(self):
        """
        Numba JIT-compiles a separate specialization per input dtype (int16 vs float32 vs whatever). Trigger that
        compile now, on a throwaway 2x2 array of the SAME dtype as the real volume, so the ~1-1.5s one-time cost happens
        at construction instead of on the user's first scroll/rotate.
        """
        dummy_vol = np.zeros((2, 2, 2), dtype=self.image.array.dtype)
        Minv = self._image_matrix_inv_f32
        _fused_oblique_sample_kernel(
            dummy_vol, 2, 2, np.float32(1), np.float32(1),
            np.float32(0), np.float32(0), np.float32(0),
            np.float32(1), np.float32(0), np.float32(0),
            np.float32(0), np.float32(1), np.float32(0),
            np.float32(0), np.float32(0), np.float32(0),
            Minv, np.float32(1), np.float32(1), np.float32(1), 0.0)

    def compute_array(self, plane, position, matrix_override=None, as_array=True, as_vtk=False,
                      copy_array=False, force_reslice=False, background=-3001.0):
        """
        plane, position supplied by the caller each call. Orientation comes from self.matrix
        (shared) unless matrix_override is given.
        """
        m = matrix_override if matrix_override is not None else self.matrix
        is_identity = (np.allclose(self.image.matrix, np.eye(3), atol=1e-6) and
                       np.allclose(m, np.eye(3), atol=1e-6))

        if is_identity and not force_reslice:
            arr = self._get_axis_aligned_array(plane, position)
            arr_shape = arr.shape
            if plane == 'Axial':
                dim = [arr_shape[1], arr_shape[0], 1]
            elif plane == 'Sagittal':
                dim = [1, arr_shape[1], arr_shape[0]]
            else:
                dim = [arr_shape[1], 1, arr_shape[0]]
            result = {'dimensions': dim, 'spacing': self.spacing}

            if as_array:
                result['array'] = arr.copy() if copy_array else arr

            if as_vtk:
                matrix_reshape = np.linalg.inv(self.image.matrix).reshape(1, 9)[0]

                vtk_out = vtk.vtkImageData()
                vtk_out.SetSpacing(self.image.spacing)
                vtk_out.SetDirectionMatrix(matrix_reshape)
                vtk_out.SetDimensions(dim)
                vtk_out.SetOrigin(position)
                vtk_out.GetPointData().SetScalars(numpy_support.numpy_to_vtk(arr.flatten(order="C")))

                result['vtk_image'] = vtk_out

            return result

        xi, yi, ni = self.axes[plane]
        geom = self.compute_plane_geometry(plane, position, matrix_override=m)
        x_axis = geom['x_axis'].astype(np.float32)
        y_axis = geom['y_axis'].astype(np.float32)
        origin = geom['origin'].astype(np.float32)
        sx, sy = np.float32(self.spacing[xi]), np.float32(self.spacing[yi])
        h, w = self.oblique_shape[plane]

        arr = _fused_oblique_sample_kernel(
            self.image.array, h, w, sx, sy,
            origin[0], origin[1], origin[2],
            x_axis[0], x_axis[1], x_axis[2],
            y_axis[0], y_axis[1], y_axis[2],
            self._image_origin_f32[0], self._image_origin_f32[1], self._image_origin_f32[2],
            self._image_matrix_inv_f32,
            self._image_spacing_f32[0], self._image_spacing_f32[1], self._image_spacing_f32[2],
            background)

        result = {'dimensions': (w, h), 'spacing': self.spacing,
                  'x_axis': x_axis, 'y_axis': y_axis, 'normal': geom['normal'], 'origin': origin}
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
            direction[:, 2] = geom['normal']
            vtk_out.SetDirectionMatrix(direction.reshape(1, 9)[0])

            vtk_out.GetPointData().SetScalars(numpy_support.numpy_to_vtk(arr.ravel(order="C")))

            result['vtk_image'] = vtk_out

        return result

    def compute_plane_geometry(self, plane, position, matrix_override=None):
        """
        Used by get_slice: position = this plane's actual reslice origin (top-left corner), needed to place the VTK
        output correctly.
        """

        m = matrix_override if matrix_override is not None else self.matrix
        xi, yi, ni = self.axes[plane]

        return {'origin': np.asarray(position, dtype=float),
                'x_axis': m[:, xi],
                'y_axis': m[:, yi],
                'normal': m[:, ni],
                'spacing': self.spacing,}

    def compute_slice_lines(self, plane, position):
        """
        position : THIS plane's own top-left origin (for correct pixel conversion of the line into this widget's
        coordinates). The crossing planes are defined via the SHARED crosshair + matrix no other widget's position is
        needed.
        """
        xi, yi, ni = self.axes[plane]
        active = {'origin': np.asarray(position, dtype=float),
                  'x_axis': self.matrix[:, xi],
                  'y_axis': self.matrix[:, yi],
                  'normal': self.matrix[:, ni],
                  'spacing': (self.spacing[xi], self.spacing[yi])}

        lines = {}
        for other in ('Axial', 'Coronal', 'Sagittal'):
            if other == plane:
                continue

            oxi, oyi, oni = self.axes[other]
            crossing = {'origin': self.crosshair,
                        'x_axis': self.matrix[:, oxi],
                        'y_axis': self.matrix[:, oyi],
                        'normal': self.matrix[:, oni],
                        'spacing': (self.spacing[oxi], self.spacing[oyi])}

            line = self._intersection_line(active, crossing)
            if line is not None:
                lines[other] = line

        return lines

    def pivot_position(self, position, R):
        """
        Repositions one widget's origin around self.rotation_center using incremental rotation R (from
        update_rotation's return value).
        Caller applies this to every widget's position right after calling update_rotation.
        """
        position = np.asarray(position, dtype=float)

        return R.dot(position - self.rotation_center) + self.rotation_center

    def set_rotation_center(self, center):
        """Call when the user picks a new pivot (click, ROI centroid, a Rigid instance's target center, etc)."""
        self.rotation_center = np.asarray(center, dtype=float)
        self.crosshair = self.rotation_center.copy()

    def update_rotation(self, r_x=0, r_y=0, r_z=0, absolute=True):
        """
        Rotates the SHARED matrix (affects every plane) and pivots the shared crosshair
        and every plane's position around rotation_center to match. Returns the
        incremental R actually applied this call (old orientation -> new orientation).

        absolute=False (default): r_x/r_y/r_z are INCREMENTAL deltas on top of the
        current rotation_totals.
        absolute=True: r_x/r_y/r_z are the TOTAL desired angle for each axis (e.g. read
        straight off a slider that holds its absolute position).

        Either way, self.matrix is recomputed FRESH from self.rotation_totals against
        self._base_matrix every call -- NOT accumulated by repeatedly multiplying a
        delta onto the previous matrix. This matters because 3D rotations about
        different axes don't commute: if you'd instead multiplied deltas on top of each
        other call after call, the resulting orientation for a given (pitch, yaw, roll)
        triple would depend on the exact sequence of prior moves that got you there --
        e.g. rotating roll, yaw, roll again, pitch, then setting all three sliders back
        to (0, 0, 0) would NOT land back on the original orientation, since the "undo"
        deltas don't retrace the forward path in reverse. Recomputing from the absolute
        totals every time makes (0, 0, 0) always mean exactly self._base_matrix,
        regardless of path.
        """
        if absolute:
            self.rotation_totals = np.array([r_x, r_y, r_z], dtype=float)
        else:
            self.rotation_totals = self.rotation_totals + np.array([r_x, r_y, r_z], dtype=float)

        old_matrix = self.matrix
        new_matrix = Rotation.from_euler('xyz', self.rotation_totals, degrees=True).as_matrix()
        # old_matrix is a pure rotation matrix (orthonormal) -> inverse == transpose.
        R = new_matrix @ old_matrix.T

        self.matrix = new_matrix
        self.crosshair = R.dot(self.crosshair - self.rotation_center) + self.rotation_center
        for plane in self.position:
            self.position[plane] = self.pivot_position(self.position[plane], R)

        return R

    def wheel_position(self, plane, position, steps=1, main=True):
        _, _, ni = self.axes[plane]
        normal = self.matrix[:, ni]
        normal_hat = normal / np.linalg.norm(normal)
        delta = normal_hat * self.spacing[ni] * steps
        self.crosshair = self.crosshair + delta
        new_position = np.asarray(position, dtype=float) + delta
        if main:
            self.position[plane] = new_position

        return new_position


class Image(object):
    """
    Main data class containing standard medical image volume blocks, metadata parsing, structures, and geometric transforms.

    Parameters
    ----------
    image : object
        A wrapper object tracking image arrays, properties, file locations, and DICOM field configurations.
    """
    def __init__(self, image):
        self.rois = {}
        self.pois = {}

        self.tags = image.image_set
        self.array = np.ascontiguousarray(image.array)
        self.array_dtype = str(self.array.dtype)

        self.image_name = image.image_name
        self.modality = image.modality

        self.patient_name = self.get_patient_name()
        self.mrn = self.get_mrn()
        self.birthdate = self.get_birthdate()
        self.date = self.get_date()
        self.time = self.get_time()
        self.series_uid = self.get_series_uid()
        self.acq_number = self.get_acq_number()
        self.frame_ref = self.get_frame_ref()
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
        state = self.__dict__.copy()
        state.pop("array", None)
        state.pop("rois", None)
        state.pop("pois", None)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.array = None
        self.rois = {}
        self.pois = {}

    def input_mhd(self, filename, roi_names, values, plane='Axial'):
        """
        Parses an explicit MetaImage (.mhd/.raw) dataset and adds segmented regions as tracking ROIs.

        Parameters
        ----------
        filename : str
            Full system path string linking directly to the MetaImage configuration header file.
        roi_names : list of str
            Label identifiers to map sequentially to each discrete voxel label segment.
        values : list of int
            The absolute voxel pixel configuration value flags used to isolate segmentation volumes.
        plane : str, default 'Axial'
            The principal acquisition structural reference frame.

        Returns
        -------
        None
        """
        roi_image = sitk.ReadImage(filename)
        roi_array = sitk.GetArrayFromImage(roi_image)
        for ii, roi_name in enumerate(roi_names):
            if roi_name not in list(self.rois.keys()):
                self.rois[roi_name] = Roi(self, name=roi_name, visible=True, filepaths=filename,
                                          plane=plane)

            roi_mask = roi_array == values[ii]
            self.rois[roi_name].convert_mask(roi_mask)

    def input_rtstruct(self, rtstruct):
        """
        Imports structured DICOM RT-Struct data elements, populating the local instances with ROI/POI objects.

        Parameters
        ----------
        rtstruct : object
            An imported container struct containing parsed contours, names, points, and colors.

        Returns
        -------
        None
        """
        for ii, roi_name in enumerate(rtstruct.roi_names):
            if roi_name not in list(self.rois.keys()) or self.rois[roi_name].contour_position is None:
                self.rois[roi_name] = Roi(self, position=rtstruct.contours[ii], name=roi_name,
                                          color=rtstruct.roi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

        for ii, poi_name in enumerate(rtstruct.poi_names):
            if poi_name not in list(self.pois.keys()) or self.pois[poi_name].point_position is None:
                self.pois[poi_name] = Poi(self, position=rtstruct.points[ii], name=poi_name,
                                          color=rtstruct.poi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

        Data.match_rois()
        Data.match_pois()

    def add_roi(self, roi_name=None, color=None, visible=False, path=None, contour=None, plane='Axial'):
        """
        Appends an explicitly instantiated Region of Interest object into the image volume structure tracking frame.

        Parameters
        ----------
        roi_name : str, optional
            A distinct tracking string key identifier.
        color : list of int, optional
            An RGB list of integers mapping structural color displays.
        visible : bool, default False
            Determines whether the structure maps to active slice plot view renders automatically.
        path : str, optional
            Direct file system tracking reference source link.
        contour : array_like, optional
            Coordinate positional boundary point datasets.
        plane : str, default 'Axial'
            Standard tracking plane orientation label.

        Returns
        -------
        None
        """
        self.rois[roi_name] = Roi(self, position=contour, name=roi_name, color=color, visible=visible, filepaths=path,
                                  plane=plane)
        Data.match_rois()

    def add_poi(self, poi_name=None, color=None, visible=False, path=None, point=None):
        """
        Appends a Point of Interest landmark element onto the active frame tracking tracking system.

        Parameters
        ----------
        poi_name : str, optional
            Landmark distinct name label configuration string.
        color : list of int, optional
            An RGB collection tracking visual markers.
        visible : bool, default False
            Active viewport display flag constraint state.
        path : str, optional
            System reference origin source path data details.
        point : array_like, optional
            Length-3 coordinate point setting absolute landmark position.

        Returns
        -------
        None
        """
        self.pois[poi_name] = Poi(self, position=point, name=poi_name, color=color, visible=visible, filepaths=path)
        Data.match_pois()

    def create_roi(self, name=None, color=None, visible=False, filepath=None):
        """
        Initializes an empty tracking ROI structural instance mapped onto the localized tracking frame.

        Parameters
        ----------
        name : str, optional
            Distinct label reference key identifier.
        color : list of int, optional
            RGB visual display settings profile tracking array.
        visible : bool, default False
            Active component visibility state configurations.
        filepath : str, optional
            Source processing storage folder context.

        Returns
        -------
        None
        """
        self.rois[name] = Roi(self, name=name, color=color, visible=visible, filepaths=filepath)
        Data.match_rois()

    def create_rtstruct(self, roi_names=None, poi_names=None):
        """
        Placeholder interface configuration method to handle downstream structured export pipeline generations.

        Parameters
        ----------
        roi_names : list of str, optional
            Target tracking segments to collect into structure.
        poi_names : list of str, optional
            Target landmarks tracking points array metrics.

        Returns
        -------
        None
        """
        pass

    def get_patient_name(self):
        """
        Parses PatientName field segments out from the current base DICOM data metadata structures.

        Returns
        -------
        list of str or str
            A parsed string collection containing name fragments, or 'missing'.
        """
        if 'PatientName' in self.tags[0]:
            return str(self.tags[0].PatientName).split('^')[:3]
        else:
            return 'missing'

    def get_mrn(self):
        """
        Extracts PatientID medical record number identifiers from the base header profiles.

        Returns
        -------
        str
            The tracked reference alphanumeric MRN string label context.
        """
        if 'PatientID' in self.tags[0]:
            return str(self.tags[0].PatientID)
        else:
            return 'missing'

    def get_birthdate(self):
        """
        Extracts PatientBirthDate metadata attributes directly out from the primary DICOM attributes profile.

        Returns
        -------
        str
            The text string date code tracking birth entries.
        """
        if 'PatientBirthDate' in self.tags[0]:
            return str(self.tags[0].PatientBirthDate)
        else:
            return ''

    def get_date(self):
        """
        Finds first available series or acquisition validation date indicators across sequential structural headers.

        Returns
        -------
        str
            A structured numeric string sequence identifying configuration transaction dates.
        """
        if 'SeriesDate' in self.tags[0]:
            return self.tags[0].SeriesDate
        elif 'ContentDate' in self.tags[0]:
            return self.tags[0].ContentDate
        elif 'AcquisitionDate' in self.tags[0]:
            return self.tags[0].AcquisitionDate
        elif 'StudyDate' in self.tags[0]:
            return self.tags[0].StudyDate
        else:
            return '00000'

    def get_time(self):
        """
        Finds chronological tracking timestamp variables across fallback structural dataset headers.

        Returns
        -------
        str
            The identified chronological session execution time value sequence.
        """
        if 'SeriesTime' in self.tags[0]:
            return self.tags[0].SeriesTime
        elif 'ContentTime' in self.tags[0]:
            return self.tags[0].ContentTime
        elif 'AcquisitionTime' in self.tags[0]:
            return self.tags[0].AcquisitionTime
        elif 'StudyTime' in self.tags[0]:
            return self.tags[0].StudyTime
        else:
            return '00000'

    def get_study_uid(self):
        """
        Queries tracking dataset files to expose unique master StudyInstanceUID identity profiles.

        Returns
        -------
        str
            The explicit string mapping absolute global study uniqueness values.
        """
        if 'StudyInstanceUID' in self.tags[0]:
            return self.tags[0].StudyInstanceUID
        else:
            return '00000.00000'

    def get_series_uid(self):
        """
        Queries tracking elements to extract explicit target SeriesInstanceUID system profiles.

        Returns
        -------
        str
            The string mapping detailed operational image series paths.
        """
        if 'SeriesInstanceUID' in self.tags[0]:
            return self.tags[0].SeriesInstanceUID
        else:
            return '00000.00000'

    def get_acq_number(self):
        """
        Locates clear acquisition tracking metrics inside individual data frames.

        Returns
        -------
        str
            The structured acquisition catalog identification key indicator string.
        """
        if 'AcquisitionNumber' in self.tags[0]:
            return self.tags[0].AcquisitionNumber
        else:
            return '1'

    def get_frame_ref(self):
        """
        Acquires FrameOfReferenceUID descriptors matching 3D geometric registration alignments.

        Returns
        -------
        str
            Unique alignment string verification system coordinates.
        """
        if 'FrameOfReferenceUID' in self.tags[0]:
            return self.tags[0].FrameOfReferenceUID
        else:
            return '00000.00000'

    def get_window(self):
        """
        Computes active visual window boundaries tracking WindowCenter and WindowWidth DICOM configurations.

        Returns
        -------
        list of int
            A 2-element collection defining absolute [Lower, Upper] grayscale threshold window parameters.
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

    def get_specific_tag(self, tag):
        """
        Polls the initial index tracking dictionary config profile looking for targeted custom DICOM tags.

        Parameters
        ----------
        tag : str or tuple
            The custom explicit DICOM lookup sequence tracker.

        Returns
        -------
        object or None
            The raw localized tag content payload matching data lookups.
        """
        if tag in self.tags[0]:
            return self.tags[0][tag]
        else:
            return None

    def get_specific_tag_on_all_files(self, tag):
        """
        Loops through all sequential discrete file tags tracking uniform instance parameters.

        Parameters
        ----------
        tag : str or tuple
            The designated search identifier variable parameters.

        Returns
        -------
        list of object or None
            A collected list tracking sequential structural tag findings.
        """
        if tag in self.tags[0]:
            return [t[tag] for t in self.tags]
        else:
            return None

    def save(self, folder_path, key, clevel=5):
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

    def load_metadata(self, metadata_path, key):
        with open(metadata_path, "rb") as f:
            data = f.read()

        nonce, encrypted = data[:12], data[12:]
        loaded = pickle.loads(AESGCM(key).decrypt(nonce, encrypted, None))
        self.__setstate__(loaded.__dict__)

        Data.image[self.image_name] = self
        Data.image_list.append(self.image_name)

    def load_array(self, array_path, key):
        with open(array_path, "rb") as f:
            data = f.read()

        nonce, encrypted = data[:12], data[12:]
        raw = blosc2.decompress(AESGCM(key).decrypt(nonce, encrypted, None))
        self.array = np.frombuffer(raw, dtype=self.array_dtype).reshape(self.dimensions).copy()

    def save_unencrypted(self, path, rois=True, pois=True):
        """
        Serializes data matrices, tags, metadata frames, and ROI trackers directly onto storage directories.

        Parameters
        ----------
        path : str
            The absolute system storage target path context directory.
        rois : bool, default True
            Determines whether to trigger standalone object loops archiving structure regions.
        pois : bool, default True
            Determines whether to process and store companion geometric point landmark items.

        Returns
        -------
        None
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['rois', 'pois', 'tags', 'array', 'display']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'tags.npy'), self.tags, allow_pickle=True)
        np.save(os.path.join(path, 'array.npy'), self.array, allow_pickle=True)

        if rois:
            self.save_rois(path, create_main_folder=True)

        if pois:
            self.save_pois(path, create_main_folder=True)

    def save_unencrypted_rois(self, path, create_main_folder=False):
        """
        Iterates over trackable ROI dictionary values to store serialized NumPy matrices configurations.

        Parameters
        ----------
        path : str
            The parent export folder path location context.
        create_main_folder : bool, default False
            When True, explicitly establishes a new nested directory sub-folder labeled 'ROIs'.

        Returns
        -------
        None
        """
        if create_main_folder:
            path = os.path.join(path, 'ROIs')
            os.mkdir(path)

        for name in list(self.rois.keys()):
            roi_path = os.path.join(os.path.join(path, name))
            os.mkdir(roi_path)

            np.save(os.path.join(roi_path, 'name.npy'), self.rois[name].name, allow_pickle=True)
            np.save(os.path.join(roi_path, 'visible.npy'), self.rois[name].visible, allow_pickle=True)
            np.save(os.path.join(roi_path, 'color.npy'), self.rois[name].color, allow_pickle=True)
            np.save(os.path.join(roi_path, 'filepaths.npy'), self.rois[name].filepaths, allow_pickle=True)
            if self.rois[name].contour_position is not None:
                np.save(os.path.join(roi_path, 'contour_position.npy'),
                        np.array(self.rois[name].contour_position, dtype=object),
                        allow_pickle=True)

    def save_unencrypted_pois(self, path, create_main_folder=False):
        """
        Saves individual Point of Interest coordinate datasets to disk.

        Parameters
        ----------
        path : str
            The targeted backup output destination directory path.
        create_main_folder : bool, default False
            When true, generates a structured container folder named 'POIs'.

        Returns
        -------
        None
        """
        if create_main_folder:
            path = os.path.join(path, 'POIs')
            os.mkdir(path)

        for name in list(self.pois.keys()):
            poi_path = os.path.join(os.path.join(path, name))
            os.mkdir(poi_path)

            np.save(os.path.join(poi_path, 'name.npy'), self.pois[name].name, allow_pickle=True)
            np.save(os.path.join(poi_path, 'visible.npy'), self.pois[name].visible, allow_pickle=True)
            np.save(os.path.join(poi_path, 'color.npy'), self.pois[name].color, allow_pickle=True)
            np.save(os.path.join(poi_path, 'filepaths.npy'), self.pois[name].filepaths, allow_pickle=True)
            np.save(os.path.join(poi_path, 'point_position.npy'), self.pois[name].point_position, allow_pickle=True)

    def load_unencrypted_image(self, image_path, rois=True, pois=True):
        """
        Loads and populates volumetric imaging elements out from saved system processing sub-directories.

        Parameters
        ----------
        image_path : str
            The system path linking directly to the archived target dataset root directory.
        rois : bool, default True
            Enables structural lookups parsing accompanying Region of Interest segments.
        pois : bool, default True
            Enables targeted reconstruction reading Point of Interest configuration arrays.

        Returns
        -------
        None
        """
        self.array = np.load(os.path.join(image_path, 'array.npy'), allow_pickle=True)
        self.tags = np.load(os.path.join(image_path, 'tags.npy'), allow_pickle=True)
        info = pd.read_pickle(os.path.join(image_path, 'info.p'), )
        for column in list(info.columns):
            setattr(self, column, info.at[0, column])

        if rois:
            roi_names = os.listdir(os.path.join(image_path, 'ROIs'))
            for name in roi_names:
                self.load_rois(os.path.join(image_path, 'ROIs', name))

        if pois:
            roi_names = os.listdir(os.path.join(image_path, 'POIs'))
            for name in roi_names:
                self.load_pois(os.path.join(image_path, 'POIs', name))

    def load_unencrypted_rois(self, roi_path):
        """
        Parses archived individual target ROI binary properties, formatting components to avoid index namespace conflicts.

        Parameters
        ----------
        roi_path : str
            Direct source storage context tracking individual mask files.

        Returns
        -------
        None
        """
        name = str(np.load(os.path.join(roi_path, 'name.npy'), allow_pickle=True))

        existing_rois = list(self.rois.keys())
        if name in existing_rois:
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_rois:
                    name = new_name
                    n = -1

        self.rois[name] = Roi(self)
        self.rois[name].name = name
        self.rois[name].visible = bool(np.load(os.path.join(roi_path, 'visible.npy'), allow_pickle=True))
        self.rois[name].color = list(np.load(os.path.join(roi_path, 'color.npy'), allow_pickle=True))
        self.rois[name].filepaths = str(np.load(os.path.join(roi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(roi_path, 'contour_position.npy')):
            self.rois[name].contour_position = list(np.load(os.path.join(roi_path, 'contour_position.npy'),
                                                            allow_pickle=True))

    def load_unencrypted_pois(self, poi_path):
        """
        Parses archived single target landmark configurations from disk.

        Parameters
        ----------
        poi_path : str
            Direct folder path to read specified landmark point geometries.

        Returns
        -------
        None
        """
        name = str(np.load(os.path.join(poi_path, 'name.npy'), allow_pickle=True))

        existing_pois = list(self.pois.keys())
        if name in existing_pois:
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_pois:
                    name = new_name
                    n = -1

        self.pois[name] = Poi(self)
        self.pois[name].name = name
        self.pois[name].visible = bool(np.load(os.path.join(poi_path, 'visible.npy'), allow_pickle=True))
        self.pois[name].color = list(np.load(os.path.join(poi_path, 'color.npy'), allow_pickle=True))
        self.pois[name].filepaths = str(np.load(os.path.join(poi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(poi_path, 'point_position.npy')):
            self.rois[name].contour_position = list(np.load(os.path.join(poi_path, 'point_position.npy'),
                                                            allow_pickle=True))

    def create_sitk_image(self, empty=False):
        """
        Converts the active internal image data block matrix into a native SimpleITK image container.

        Parameters
        ----------
        empty : bool, default False
            When True, drops data allocation arrays to return a zero-initialized UInt8 volume container.

        Returns
        -------
        SimpleITK.Image
            The generated ITK core volumetric structure block tracking spacing and directions.
        """
        if empty:
            sitk_image = sitk.Image([int(dim) for dim in self.dimensions], sitk.sitkUInt8)
        else:
            sitk_image = sitk.GetImageFromArray(self.array)

        matrix_flat = self.matrix.flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        return sitk_image

    def create_rotated_sitk_image(self):
        """
        Applies a custom sample 3D Euler transformation rotation tracking localized anatomical structures.

        Returns
        -------
        numpy.ndarray
            The newly resampled volumetric data matrix block.
        """
        sitk_image = sitk.GetImageFromArray(self.array)
        matrix_flat = self.matrix.flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        transform = sitk.Euler3DTransform()
        transform.SetRotation(0, 0, 10 * np.pi / 180)
        transform.SetCenter(self.rois['Liver'].mesh.center)
        transform.SetComputeZYX(True)

        resample_image = sitk.ResampleImageFilter()
        resample_image.SetOutputDirection(sitk_image.GetDirection())
        resample_image.SetOutputOrigin(sitk_image.GetOrigin())
        resample_image.SetTransform(transform)
        resample_image.SetInterpolator(sitk.sitkLinear)
        resample_image.Execute(sitk_image)

        return sitk.GetArrayFromImage(resample_image)

    def create_external(self, name='External', color=None, visible=False, filepaths=None, threshold=-250):
        """
        Generates a continuous exterior bounding structural contour mask using an intensity threshold setting.

        Parameters
        ----------
        name : str, default 'External'
            Tracking key string mapped onto the generated ROI structure block.
        color : list of int, optional
            Custom visual color definition profile mapping. Defaults to bright green.
        visible : bool, default False
            Standard viewport visualization status tracker constraint.
        filepaths : str, optional
            Source processing storage location links.
        threshold : int, default -250
            The low Hounsfield Unit value or raw scalar intensity configuration cap.

        Returns
        -------
        None
        """
        if color is None:
            color = [0, 255, 0]

        if name not in list(self.rois.keys()):
            self.rois[name] = Roi(self, name=name, color=color, visible=visible, filepaths=filepaths)

        mask = external(self.array, threshold=threshold, only_mask=True)
        contours = contours_from_mask(mask.astype(np.uint8))
        positions = self.rois[name].convert_pixel_to_position(pixel=contours)

        self.rois[name].contour_pixel = contours
        self.rois[name].contour_position = positions
        self.rois[name].create_discrete_mesh()

    def compute_initial_origin(self, plane):
        """
        Top-left corner of this plane's slice, at the volume's center index
        along the plane's normal.

        self.display.image.dimensions is (z, y, x).
        compute_position expects pixel index as [x, y, z].
        axes gives (xi, yi, ni) in xyz terms, so the matching
        dimensions slot for a given xyz index i is dimensions[2 - i].
        """

        axes = {'Axial':    (0, 1, 2),
                'Sagittal': (1, 2, 0),
                'Coronal':  (0, 2, 1)}

        xi, yi, ni = axes[plane]
        dims = self.dimensions  # (z, y, x)

        idx_xyz = np.zeros(3)
        idx_xyz[ni] = dims[2 - ni] / 2  # center index along the through-plane axis
        # xi/yi stay 0 -- top-left corner in-plane

        return self.compute_position(idx_xyz)

    def compute_pixel(self, position):
        """
        Transforms a physical coordinate location string back to fractional/integer index pixel coordinates.

        Parameters
        ----------
        position : array_like
            A 3-element physical point position configuration matrix tracking space.

        Returns
        -------
        numpy.ndarray
            An int32 array tracking specific structural pixel matrix indices.
        """
        matrix = copy.deepcopy(self.matrix)

        hold = np.eye(3)
        hold[0, :] = matrix[0, :] / self.spacing[0]
        hold[1, :] = matrix[1, :] / self.spacing[1]
        hold[2, :] = matrix[2, :] / self.spacing[2]
        pos2pix = np.eye(4)
        pos2pix[:3, :3] = hold
        pos2pix[:3, 3] = np.asarray(self.origin).dot(-hold.T)

        return pos2pix.dot([*position, 1])[:3]

    def compute_position(self, xyz):
        """
        Transforms local pixel grid coordinates to physical 3D space locations.

        Parameters
        ----------
        xyz : array_like
            A 3-element index profile tracking data grid points.

        Returns
        -------
        numpy.ndarray
            A float32 coordinate array defining physical space coordinates.
        """
        matrix = copy.deepcopy(self.matrix)

        p2p = np.eye(4)
        p2p[:3, 0] = matrix[0, :] * self.spacing[0]
        p2p[:3, 1] = matrix[1, :] * self.spacing[1]
        p2p[:3, 2] = matrix[2, :] * self.spacing[2]
        p2p[:3, 3] = self.origin

        return p2p.dot([*xyz, 1])[:3]

    def compute_matrix_pixel_to_position(self):
        """
        Helper method computing homogeneous index transformation systems.

        Returns
        -------
        numpy.ndarray
            A 4x4 coordinate tracking matrix.
        """
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
        """
        Helper method generating transformation matrices targeting internal structural transformations.

        Returns
        -------
        None
        """
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / spacing[0]
        hold_matrix[1, :] = matrix[1, :] / spacing[1]
        hold_matrix[2, :] = matrix[2, :] / spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        return position_to_pixel_matrix

    def get_aspect(self, slice_plane):
        """
        Calculates viewport pixel aspect ratios required to prevent image skewing during display stretching.

        Parameters
        ----------
        slice_plane : str
            The viewport display target orientation frame. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        float
            The proportion scalar value rounded strictly to 2 decimal points.
        """
        if slice_plane == 'Axial':
            aspect = np.round(self.spacing[0] / self.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.spacing[0] / self.spacing[2], 2)
        else:
            aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        return aspect

    def get_bounds(self):
        """
        Calculates absolute spatial bounding box ranges using VTK internal volume tracking logic.

        Returns
        -------
        list of float
            A 6-element list tracking spatial limits: [x_min, x_max, y_min, y_max, z_min, z_max].
        """
        shape = self.array.shape
        matrix_reshape = self.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions([shape[1], shape[2], shape[0]])
        vtk_image.SetOrigin(self.origin)

        x_min, x_max, y_min, y_max, z_min, z_max = vtk_image.GetBounds()

        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def get_center(self, position=True, zyx=False):
        """
        Identifies mid-volume coordinate points in either pixel space grids or absolute physical systems.

        Parameters
        ----------
        position : bool, default True
            When True, transforms center coordinates into millimeter space. Otherwise keeps pixel indexes.
        zyx : bool, default False
            Flips positional vectors to sequence coordinates along inverted index trajectories.

        Returns
        -------
        list of int or numpy.ndarray
            The length-3 mid-volume structural position elements.
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

    def get_corner_positions(self):
        """
        Calculates explicit 3D physical location tracking positions for the eight bounding volume corners.

        Returns
        -------
        list of tuple
            A list containing eight distinct length-3 coordinate measurement tuples.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.display.vtk_image.GetBounds()

        corner_points = [(x_min, y_min, z_min),
                         (x_max, y_min, z_min),
                         (x_max, y_max, z_min),
                         (x_min, y_max, z_min),
                         (x_min, y_min, z_max),
                         (x_max, y_min, z_max),
                         (x_max, y_max, z_max),
                         (x_min, y_max, z_max)]

        return corner_points

    def get_corner_sides(self):
        """
        Generates a PyVista visual wireframe bounding box tracking the extreme dimensional corners.

        Returns
        -------
        pyvista.PolyData
            The generated surface data object ready for rendering pipelines.
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

    def reset_display(self):
        """
        Clears out off-axis reslice transformations to re-establish standard viewing orientations.

        Returns
        -------
        None
        """
        self.display.matrix = np.eye(3)
        self.display.rotation_center = np.asarray(self.get_center(), dtype=float)

    def retrieve_angles(self, order='ZXY'):
        """
        Converts the active viewing matrix configuration into standard Euler angles.

        Parameters
        ----------
        order : str, default 'ZXY'
            The specific axis processing rotation sequence layout constraint.

        Returns
        -------
        numpy.ndarray
            Calculated rotation angle vectors returned explicitly in degrees format.
        """
        rotation = Rotation.from_matrix(self.display.matrix[:3, :3])

        return rotation.as_euler(order, degrees=True)
