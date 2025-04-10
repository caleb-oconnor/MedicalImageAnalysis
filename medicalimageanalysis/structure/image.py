"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import os
import copy
import math
import time

import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk

import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..utils.image.transform import euler_transform

from .poi import Poi
from .roi import Roi


class Display(object):
    def __init__(self, image):
        self.image = image

        self.vtk_image = None
        self.vtk_array = None

        self.matrix = copy.deepcopy(self.image.matrix)
        self.spacing = copy.deepcopy(self.image.spacing)
        self.origin = copy.deepcopy(self.image.origin)

        if self.image.dimensions[2] > 0:
            self.slice_location = [int(self.image.dimensions[0] / 2),
                                   int(self.image.dimensions[1] / 2),
                                   int(self.image.dimensions[2] / 2)]
        else:
            self.slice_location = [int(self.image.dimensions[0] / 2), int(self.image.dimensions[1] / 2), 0]

        self.scroll_max = [self.image.dimensions[0] - 1,
                           self.image.dimensions[1] - 1,
                           self.image.dimensions[2] - 1]

    def compute_matrix_pixel_to_position(self, base=True):
        if base:
            matrix = copy.deepcopy(self.image.matrix)
            spacing = self.image.spacing
            origin = self.image.origin
        else:
            matrix = copy.deepcopy(self.matrix)
            spacing = self.spacing
            origin = self.origin

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * spacing[2]
        pixel_to_position_matrix[:3, 3] = origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self, base=True):
        if base:
            matrix = copy.deepcopy(self.image.matrix)
            spacing = self.image.spacing
            origin = self.image.origin
        else:
            matrix = copy.deepcopy(self.matrix)
            spacing = self.spacing
            origin = self.origin

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / spacing[0]
        hold_matrix[1, :] = matrix[1, :] / spacing[1]
        hold_matrix[2, :] = matrix[2, :] / spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(origin).dot(-hold_matrix.T)

        return position_to_pixel_matrix

    def compute_array(self, slice_plane):
        if np.array_equal(self.matrix, self.image.matrix):
            if slice_plane == 'Axial':
                array = np.flip(self.image.array[:, :, self.slice_location[2]].T, 0)
            elif slice_plane == 'Coronal':
                array = self.image.array[:, self.slice_location[1], :].T
            else:
                array = self.image.array[self.slice_location[0], :, :].T

        else:
            if slice_plane == 'Axial':
                array = np.flip(self.vtk_array[:, :, self.slice_location[2]].T, 0)
            elif slice_plane == 'Coronal':
                array = self.vtk_array[:, self.slice_location[1], :].T
            else:
                array = self.vtk_array[self.slice_location[0], :, :].T

        return array.astype(np.float32)

    def compute_scroll_max(self):
        if np.array_equal(self.matrix, self.image.matrix):
            self.scroll_max = [self.image.dimensions[0] - 1,
                               self.image.dimensions[1] - 1,
                               self.image.dimensions[2] - 1]

        else:
            shape = self.vtk_array.shape
            self.scroll_max[2] = int(shape[2] - 1)
            self.scroll_max[1] = int(shape[1] - 1)
            self.scroll_max[0] = int(shape[0] - 1)

    def compute_slice_line(self, slice_plane):
        pass

    def compute_off_axis_slice(self, slice_plane):
        square_rotated_voxels = self.square_help(slice_plane)

        if slice_plane == 'Axial':
            first_row = np.linspace(square_rotated_voxels[0], square_rotated_voxels[1], 2 * self.image.dimensions[0])
            last_row = np.linspace(square_rotated_voxels[2], square_rotated_voxels[3], 2 * self.image.dimensions[0])
            base_grid = np.linspace(first_row, last_row, 2 * self.image.dimensions[1])
            slice_array = np.full((2 * self.image.dimensions[0], 2 * self.image.dimensions[1]), np.nan)

        elif slice_plane == 'Coronal':
            first_row = np.linspace(square_rotated_voxels[0], square_rotated_voxels[1], 2 * self.image.dimensions[2])
            last_row = np.linspace(square_rotated_voxels[2], square_rotated_voxels[3], 2 * self.image.dimensions[2])
            base_grid = np.linspace(first_row, last_row, 2 * self.image.dimensions[0])
            slice_array = np.full((2 * self.image.dimensions[0], 2 * self.image.dimensions[2]), np.nan)

        else:
            first_row = np.linspace(square_rotated_voxels[0], square_rotated_voxels[1], 2 * self.image.dimensions[2])
            last_row = np.linspace(square_rotated_voxels[2], square_rotated_voxels[3], 2 * self.image.dimensions[2])
            base_grid = np.linspace(first_row, last_row, 2 * self.image.dimensions[1])
            slice_array = np.full((2 * self.image.dimensions[1], 2 * self.image.dimensions[2]), np.nan)

        grid_x = (base_grid[:, :, 0] >= -0.001) & (base_grid[:, :, 0] <= self.image.dimensions[0] - .51)
        grid_y = (base_grid[:, :, 1] >= -0.001) & (base_grid[:, :, 1] <= self.image.dimensions[1] - .51)
        grid_z = (base_grid[:, :, 2] >= -0.001) & (base_grid[:, :, 2] <= self.image.dimensions[2] - .51)
        combine_grid = 1 * ((1 * grid_x + 1 * grid_y + 1 * grid_z) == 3)
        idx = np.transpose(np.where(combine_grid > 0))
        grid_keep = np.round(base_grid[idx[:, 0], idx[:, 1], :]).astype(np.int64)
        grid_keep_transpose = np.asarray([grid_keep[:, 0], grid_keep[:, 1], grid_keep[:, 2]]).T
        keep_values = self.image.array[grid_keep_transpose[:, 0], grid_keep_transpose[:, 1], grid_keep_transpose[:, 2]]

        slice_array[idx[:, 0], idx[:, 1]] = keep_values

        return slice_array

    def compute_vtk_slice(self, slice_plane):
        matrix_reshape = self.image.matrix.reshape(1, 9)[0]
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        if np.array_equal(self.matrix, self.image.matrix):
            if slice_plane == 'Axial':
                location = np.asarray([0, 0, self.slice_location[2], 1])
                array_slice = self.image.array[:, :, self.slice_location[2]]
                array_shape = array_slice.shape
                dim = [array_shape[0], array_shape[1], 1]
            elif slice_plane == 'Coronal':
                location = np.asarray([0, self.slice_location[1], 0, 1])
                array_slice = self.image.array[:, self.slice_location[1], :]
                array_shape = array_slice.shape
                dim = [array_shape[0], 1, array_shape[1]]
            else:
                location = np.asarray([self.slice_location[0], 0, 0, 1])
                array_slice = self.image.array[self.slice_location[0], :, :]
                array_shape = array_slice.shape
                dim = [1, array_shape[0], array_shape[1]]

            slice_origin = location.dot(pixel_to_position_matrix.T)[:3]

            img = vtk.vtkImageData()
            img.SetSpacing(self.image.spacing)
            img.SetDirectionMatrix(matrix_reshape)
            img.SetDimensions(dim)
            img.SetOrigin(slice_origin)
            img.GetPointData().SetScalars(numpy_to_vtk(array_slice.flatten(order="F")))

        else:
            location = np.asarray([self.slice_location[0], self.slice_location[1], self.slice_location[2], 1])
            rotation_center = location.dot(pixel_to_position_matrix.T)[:3]

            if slice_plane == 'Axial':
                normal = self.matrix[:, 2]
            elif slice_plane == 'Coronal':
                normal = self.matrix[:, 1]
            else:
                normal = self.matrix[:, 0]

            plane = vtk.vtkPlane()
            plane.SetOrigin(rotation_center)
            plane.SetNormal(normal)

            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(self.vtk_image)
            cutter.Update()

            img = pv.PolyData(cutter.GetOutput())

        return img

    def compute_vtk_volume(self, slice_plane):
        pass

    def get_scroll_max(self, slice_plane):
        if slice_plane == 'Axial':
            return self.scroll_max[2]

        elif slice_plane == 'Coronal':
            return self.scroll_max[1]

        else:
            return self.scroll_max[0]

    def square_help(self, slice_plane):
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()

        location = np.asarray([self.slice_location[0], self.slice_location[1], self.slice_location[2], 1])
        rotation_center = location.dot(pixel_to_position_matrix.T)[:3]
        transform = euler_transform(matrix=self.matrix, rotation_center=rotation_center)

        center = np.round(np.asarray(self.image.dimensions) / 2).astype(np.int64)
        if slice_plane == 'Axial':
            square = [np.asarray([-center[0], -center[1], location[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([-center[0], 3 * center[1], location[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([3 * center[0], -center[1], location[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([3 * center[0], 3 * center[1], location[2], 1]).dot(pixel_to_position_matrix.T)[:3]]

        elif slice_plane == 'Coronal':
            square = [np.asarray([-center[0], location[1], -center[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([-center[0], location[1], 3 * center[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([3 * center[0], location[1], -center[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([3 * center[0], location[1], 3 * center[2], 1]).dot(pixel_to_position_matrix.T)[:3]]

        else:
            square = [np.asarray([location[0], -center[1], -center[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([location[0], -center[1], 3 * center[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([location[0], 3 * center[1], -center[2], 1]).dot(pixel_to_position_matrix.T)[:3],
                      np.asarray([location[0], 3 * center[1], 3 * center[2], 1]).dot(pixel_to_position_matrix.T)[:3]]

        position_to_pixel_matrix = self.compute_matrix_position_to_pixel()
        square_rotated_position = [transform.TransformPoint(s) for s in square]
        square_rotated_concat = np.concatenate((square_rotated_position,
                                                np.ones((len(square_rotated_position), 1))), axis=1)
        square_rotated_voxels = square_rotated_concat.dot(position_to_pixel_matrix.T)[:, :3]

        return square_rotated_voxels

    def update_rotation(self, rotation_matrix=None):
        if rotation_matrix is None:
            self.matrix = copy.deepcopy(self.image.matrix)
            self.vtk_array = None
            self.vtk_image = None
        else:
            self.matrix = rotation_matrix @ self.image.matrix
            # if self.vtk_image is None:
            #     matrix_reshape = self.image.matrix.reshape(1, 9)[0]
            #     self.vtk_image = vtk.vtkImageData()
            #     self.vtk_image.SetSpacing(self.image.spacing)
            #     self.vtk_image.SetDirectionMatrix(matrix_reshape)
            #     self.vtk_image.SetDimensions(self.image.array.shape)
            #     self.vtk_image.SetOrigin(self.image.origin)
            #     self.vtk_image.GetPointData().SetScalars(numpy_to_vtk(self.image.array.flatten(order="F")))
            #
            # pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
            # location = np.asarray([self.slice_location[0], self.slice_location[1], self.slice_location[2], 1])
            # rotation_center = location.dot(pixel_to_position_matrix.T)[:3]
            #
            # set_matrix = np.identity(4)
            # set_matrix[:3, :3] = self.matrix
            # # set_matrix[:3, 3] = -rotation_center
            #
            # resliceAxes = vtk.vtkMatrix4x4()
            # resliceAxes.DeepCopy(set_matrix.flatten())
            #
            # vtk_reslice = vtk.vtkImageReslice()
            # vtk_reslice.SetInputData(self.vtk_image)
            # vtk_reslice.SetResliceAxes(resliceAxes)
            # vtk_reslice.SetInterpolationModeToLinear()
            # vtk_reslice.AutoCropOutputOff()
            # vtk_reslice.SetBackgroundLevel(-3001)
            # vtk_reslice.Update()
            #
            # reslice_data = vtk_reslice.GetOutput()
            # self.origin = reslice_data.GetOrigin()
            # self.spacing = reslice_data.GetSpacing()
            # dimensions = reslice_data.GetDimensions()
            #
            # scalars = reslice_data.GetPointData().GetScalars()
            # self.vtk_array = np.transpose(vtk_to_numpy(scalars).reshape(dimensions[2],
            #                                                             dimensions[1],
            #                                                             dimensions[0]), (2, 1, 0))

    def update_slice_location(self, scroll, slice_plane):
        if slice_plane == 'Axial':
            self.slice_location[2] = scroll
        elif slice_plane == 'Coronal':
            self.slice_location[1] = scroll
        else:
            self.slice_location[0] = scroll


class Image(object):
    def __init__(self, image):
        self.rois = {}
        self.pois = {}

        self.tags = image.image_set
        self.array = np.transpose(image.array, (2, 1, 0))

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
        self.dimensions = np.asarray(self.array.shape)
        self.orientation = image.orientation
        self.origin = image.origin
        self.matrix = image.image_matrix

        self.unverified = image.unverified
        self.skipped_slice = image.skipped_slice
        self.sections = image.sections
        self.rgb = image.rgb

        self.camera_position = None

        self.display = Display(self)

    def input_rtstruct(self, rtstruct):
        for ii, roi_name in enumerate(rtstruct.roi_names):
            if roi_name not in list(self.rois.keys()):
                self.rois[roi_name] = Roi(self, position=rtstruct.contours[ii], name=roi_name,
                                          color=rtstruct.roi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

        for ii, poi_name in enumerate(rtstruct.poi_names):
            if poi_name not in list(self.pois.keys()):
                self.pois[poi_name] = Poi(self, position=rtstruct.points[ii], name=poi_name,
                                          color=rtstruct.poi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

    def add_roi(self, roi_name=None, color=None, visible=False, path=None, contour=None):
        self.rois[roi_name] = Roi(self, position=contour, name=roi_name, color=color, visible=visible, filepaths=path)

    def add_poi(self, poi_name=None, color=None, visible=False, path=None, point=None):
        self.pois[poi_name] = Poi(self, position=point, name=poi_name, color=color, visible=visible, filepaths=path)

    def get_patient_name(self):
        if 'PatientName' in self.tags[0]:
            return str(self.tags[0].PatientName).split('^')[:3]
        else:
            return 'missing'

    def get_mrn(self):
        if 'PatientID' in self.tags[0]:
            return str(self.tags[0].PatientID)
        else:
            return 'missing'

    def get_birthdate(self):
        if 'PatientBirthDate' in self.tags[0]:
            return str(self.tags[0].PatientBirthDate)
        else:
            return 'missing'

    def get_date(self):
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
        if 'StudyInstanceUID' in self.tags[0]:
            return self.tags[0].StudyInstanceUID
        else:
            return '00000.00000'

    def get_series_uid(self):
        if 'SeriesInstanceUID' in self.tags[0]:
            return self.tags[0].SeriesInstanceUID
        else:
            return '00000.00000'

    def get_acq_number(self):
        if 'AcquisitionNumber' in self.tags[0]:
            return self.tags[0].AcquisitionNumber
        else:
            return '1'

    def get_frame_ref(self):
        if 'FrameOfReferenceUID' in self.tags[0]:
            return self.tags[0].FrameOfReferenceUID
        else:
            return '00000.00000'

    def get_window(self):
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
        if tag in self.tags[0]:
            return self.tags[0][tag]
        else:
            return None

    def get_specific_tag_on_all_files(self, tag):
        if tag in self.tags[0]:
            return [t[tag] for t in self.tags]
        else:
            return None

    def save_image(self, path, rois=True, pois=True):
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['rois', 'pois', 'tags', 'array']]

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

    def save_rois(self, path, create_main_folder=False):
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

    def save_pois(self, path, create_main_folder=False):
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

    def load_image(self, image_path, rois=True, pois=True):

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

    def load_rois(self, roi_path):
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

    def load_pois(self, poi_path):
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

        self.pois[name] = poi(self)
        self.pois[name].name = name
        self.pois[name].visible = bool(np.load(os.path.join(poi_path, 'visible.npy'), allow_pickle=True))
        self.pois[name].color = list(np.load(os.path.join(poi_path, 'color.npy'), allow_pickle=True))
        self.pois[name].filepaths = str(np.load(os.path.join(poi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(poi_path, 'point_position.npy')):
            self.rois[name].contour_position = list(np.load(os.path.join(poi_path, 'point_position.npy'),
                                                            allow_pickle=True))

    def create_sitk_image(self, empty=False):
        if empty:
            sitk_image = sitk.Image([int(dim) for dim in reversed(self.dimensions)], sitk.sitkUInt8)
        else:
            sitk_image = sitk.GetImageFromArray(self.array.T)

        matrix_flat = self.matrix.flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        return sitk_image

    def create_rotated_sitk_image(self, empty=False):
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

        # resample_image = sitk.Resample(sitk_image, transform, sitk.sitkLinear, 0.0, sitk_image.GetPixelID())
        return sitk.GetArrayFromImage(resample_image)

    def compute_aspect(self, slice_plane):
        if slice_plane == 'Axial':
            aspect = np.round(self.spacing[0] / self.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.spacing[0] / self.spacing[2], 2)
        else:
            aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        return aspect

    def compute_matrix_position_to_pixel(self):
        matrix = np.identity(3, dtype=np.float32)
        matrix[0, :] = self.matrix[0, :] / self.spacing[0]
        matrix[1, :] = self.matrix[1, :] / self.spacing[1]
        matrix[2, :] = self.matrix[2, :] / self.spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-matrix.T)

        return position_to_pixel_matrix

    def compute_matrix_pixel_to_position(self):
        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = self.matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = self.matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = self.matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def retrieve_array_plane(self, slice_plane):
        return self.display.compute_array(slice_plane=slice_plane)

    def retrieve_slice_line(self, slice_plane):
        return self.display.compute_slice_line(slice_plane)

    def retrieve_slice_location(self, slice_plane):
        if slice_plane == 'Axial':
            return self.display.slice_location[2]

        elif slice_plane == 'Coronal':
            return self.display.slice_location[1]

        else:
            return self.display.slice_location[0]

    def retrieve_slice_position(self, slice_plane=None):
        pixel_to_position_matrix = self.display.compute_matrix_pixel_to_position()

        if slice_plane is None:
            location = np.asarray([self.display.slice_location[0],
                                   self.display.slice_location[1],
                                   self.display.slice_location[2], 1])
        else:
            if slice_plane == 'Axial':
                location = np.asarray([0, 0, self.display.slice_location[2], 1])
            elif slice_plane == 'Coronal':
                location = np.asarray([0, self.display.slice_location[1], 0, 1])
            else:
                location = np.asarray([self.display.slice_location[0], 0, 0, 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def retrieve_scroll_max(self, slice_plane):
        return self.display.get_scroll_max(slice_plane)

    def retrieve_vtk_slice(self, slice_plane):
        return self.display.compute_vtk_slice(slice_plane)

    def retrieve_vtk_volume(self, slice_plane):
        return self.display.compute_vtk_volume(slice_plane)

    def update_display_rotation(self, rotation_matrix=None, angles=None):
        if rotation_matrix is None:
            sitk_transform = euler_transform(angles=angles)
            rotation_matrix = np.asarray(sitk_transform.GetMatrix()).reshape(3, 3)

        self.display.update_rotation(rotation_matrix=rotation_matrix)
        self.display.compute_scroll_max()
