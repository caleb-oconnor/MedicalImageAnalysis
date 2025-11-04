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

import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk

from scipy.spatial.transform import Rotation

import vtk
from vtkmodules.util import numpy_support

from ..utils.image.transform import euler_transform
from ..utils.image.threshold import external
from ..utils.roi.contour import contours_from_mask

from .poi import Poi
from .roi import Roi


class Display(object):
    def __init__(self, image):
        self.image = image

        self.matrix = copy.deepcopy(self.image.matrix)
        self.spacing = copy.deepcopy(self.image.spacing)
        self.origin = copy.deepcopy(self.image.origin)

        self.slice_location = self.image.compute_center(position=False, zyx=True)
        self.scroll_max = [self.image.dimensions[0] - 1,
                           self.image.dimensions[1] - 1,
                           self.image.dimensions[2] - 1]
        self.secondary_array = None
        self.misc = {}

    def compute_matrix_pixel_to_position(self):
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
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

    def compute_array(self, slice_plane):
        if self.secondary_array is None:
            if slice_plane == 'Axial':
                array = self.image.array[self.slice_location[0], :, :]
            elif slice_plane == 'Coronal':
                array = self.image.array[:, self.slice_location[1], :]
            else:
                array = self.image.array[:, :, self.slice_location[2]]
        else:
            if slice_plane == 'Axial':
                array = self.secondary_array[self.slice_location[0], :, :]
            elif slice_plane == 'Coronal':
                array = self.secondary_array[:, self.slice_location[1], :]
            else:
                array = self.secondary_array[:, :, self.slice_location[2]]

        return array.astype(np.float32)

    def compute_index_positions(self, xyz):
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        location = np.asarray([xyz[0], xyz[1], xyz[2], 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def compute_offaxis_array(self):
        loc = np.flip(self.slice_location)
        base_position_matrix = self.compute_matrix_pixel_to_position()
        slice_position = np.asarray([loc[0], loc[1], loc[2], 1]).dot(base_position_matrix.T)[:3]

        matrix_reshape = self.image.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.image.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions(np.flip(self.image.array.shape))
        vtk_image.SetOrigin(self.image.origin)
        vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(self.image.array.flatten(order="C")))

        matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, self.matrix[i, j])

        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transform.Inverse()

        vtk_reslice = vtk.vtkImageReslice()
        vtk_reslice.SetInputData(vtk_image)
        vtk_reslice.SetResliceTransform(transform)
        vtk_reslice.SetInterpolationModeToLinear()
        vtk_reslice.SetOutputSpacing(self.image.spacing)
        vtk_reslice.AutoCropOutputOn()
        vtk_reslice.SetBackgroundLevel(-3001)
        vtk_reslice.Update()

        reslice_data = vtk_reslice.GetOutput()
        new_origin = reslice_data.GetOrigin()
        self.origin = transform.TransformPoint(new_origin)
        dimensions = reslice_data.GetDimensions()

        position_to_pixel_matrix = self.compute_matrix_position_to_pixel()
        location = np.asarray([slice_position[0], slice_position[1], slice_position[2], 1])
        self.slice_location = list(np.flip(np.round(location.dot(position_to_pixel_matrix.T)[:3])).astype(np.int32))
        self.scroll_max = [dimensions[2] - 1, dimensions[1] - 1, dimensions[0] - 1]
        if self.slice_location[0] > dimensions[2] - 1:
            self.slice_location[0] = dimensions[2] - 1
        if self.slice_location[1] > dimensions[1] - 1:
            self.slice_location[1] = dimensions[1] - 1
        if self.slice_location[2] > dimensions[0] - 1:
            self.slice_location[2] = dimensions[0] - 1

        scalars = reslice_data.GetPointData().GetScalars()
        self.secondary_array = numpy_support.vtk_to_numpy(scalars).reshape(dimensions[2], dimensions[1], dimensions[0])

    def compute_scroll_max(self):
        if self.secondary_array is not None:
            self.scroll_max = [self.secondary_array.shape[0] - 1,
                               self.secondary_array.shape[1] - 1,
                               self.secondary_array.shape[2] - 1]
        else:
            self.scroll_max = [self.image.dimensions[0] - 1,
                               self.image.dimensions[1] - 1,
                               self.image.dimensions[2] - 1]

    def compute_vtk_slice(self, slice_plane):
        matrix_reshape = np.linalg.inv(self.matrix).reshape(1, 9)[0]
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        if slice_plane == 'Axial':
            location = np.asarray([0, 0, self.slice_location[0], 1])
            if self.secondary_array is None:
                array_slice = self.image.array[self.slice_location[0], :, :]
            else:
                array_slice = self.secondary_array[self.slice_location[0], :, :]
            array_shape = array_slice.shape
            dim = [array_shape[1], array_shape[0], 1]
        elif slice_plane == 'Coronal':
            location = np.asarray([0, self.slice_location[1], 0, 1])
            if self.secondary_array is None:
                array_slice = self.image.array[:, self.slice_location[1], :]
            else:
                array_slice = self.secondary_array[:, self.slice_location[1], :]
            array_shape = array_slice.shape
            dim = [array_shape[1], 1, array_shape[0]]
        else:
            location = np.asarray([self.slice_location[2], 0, 0, 1])
            if self.secondary_array is None:
                array_slice = self.image.array[:, :, self.slice_location[2]]
            else:
                array_slice = self.secondary_array[:, :, self.slice_location[2]]
            array_shape = array_slice.shape
            dim = [1, array_shape[1], array_shape[0]]

        slice_origin = location.dot(pixel_to_position_matrix.T)[:3]

        vtk_test = vtk.vtkImageData()
        vtk_test.SetSpacing(self.image.spacing)
        vtk_test.SetDirectionMatrix(matrix_reshape)
        vtk_test.SetDimensions(dim)
        vtk_test.SetOrigin(slice_origin)
        vtk_test.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array_slice.flatten(order="C")))

        return vtk_test

    def update_slice_location(self, scroll, slice_plane):
        if slice_plane == 'Axial':
            self.slice_location[0] = scroll
        elif slice_plane == 'Coronal':
            self.slice_location[1] = scroll
        else:
            self.slice_location[2] = scroll


class Image(object):
    def __init__(self, image):
        self.rois = {}
        self.pois = {}

        self.tags = image.image_set
        self.array = image.array

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

        self.unverified = image.unverified
        self.skipped_slice = image.skipped_slice
        self.sections = image.sections
        self.rgb = image.rgb

        self.camera_position = None
        self.misc = {}

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

    def add_roi(self, roi_name=None, color=None, visible=False, path=None, contour=None, plane='Axial'):
        self.rois[roi_name] = Roi(self, position=contour, name=roi_name, color=color, visible=visible, filepaths=path,
                                  plane=plane)

    def add_poi(self, poi_name=None, color=None, visible=False, path=None, point=None):
        self.pois[poi_name] = Poi(self, position=point, name=poi_name, color=color, visible=visible, filepaths=path)

    def create_roi(self, name=None, color=None, visible=False, filepath=None):
        self.rois[name] = Roi(self, name=name, color=color, visible=visible, filepaths=filepath)

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
            return ''

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
            sitk_image = sitk.Image([int(dim) for dim in self.dimensions], sitk.sitkUInt8)
        else:
            sitk_image = sitk.GetImageFromArray(self.array)

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

    def create_external(self, name='External', color=None, visible=False, filepaths=None, threshold=-250):
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

    def compute_aspect(self, slice_plane):
        if slice_plane == 'Axial':
            aspect = np.round(self.spacing[0] / self.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.spacing[0] / self.spacing[2], 2)
        else:
            aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        return aspect

    def compute_bounds(self):
        shape = self.array.shape
        matrix_reshape = self.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions([shape[1], shape[2], shape[0]])
        vtk_image.SetOrigin(self.origin)

        x_min, x_max, y_min, y_max, z_min, z_max = vtk_image.GetBounds()

        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def compute_center(self, position=True, zyx=False):
        pixel_index = [int(self.dimensions[2] / 2),
                       int(self.dimensions[1] / 2),
                       int(self.dimensions[0] / 2)]

        if position:
            pixel_to_position_matrix = self.display.compute_matrix_pixel_to_position()
            location = np.asarray([pixel_index[0], pixel_index[1], pixel_index[2], 1])

            center = location.dot(pixel_to_position_matrix.T)[:3]
            if zyx:
                return np.flip(center)
            else:
                return center

        else:
            if zyx:
                return [pixel_index[2], pixel_index[1], pixel_index[0]]
            else:
                return pixel_index

    def compute_corner_positions(self):
        shape = self.array.shape
        matrix_reshape = self.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions([shape[1], shape[2], shape[0]])
        vtk_image.SetOrigin(self.origin)

        x_min, x_max, y_min, y_max, z_min, z_max = vtk_image.GetBounds()

        corner_points = [(x_min, y_min, z_min),
                         (x_max, y_min, z_min),
                         (x_max, y_max, z_min),
                         (x_min, y_max, z_min),
                         (x_min, y_min, z_max),
                         (x_max, y_min, z_max),
                         (x_max, y_max, z_max),
                         (x_min, y_max, z_max)]

        return corner_points

    def compute_corner_sides(self):
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

    def compute_pixel(self, position):
        matrix = copy.deepcopy(self.matrix)

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / self.spacing[0]
        hold_matrix[1, :] = matrix[1, :] / self.spacing[1]
        hold_matrix[2, :] = matrix[2, :] / self.spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        location = np.asarray([position[0], position[1], position[2], 1])

        return (np.round(location.dot(position_to_pixel_matrix.T)[:3])).astype(np.int32)

    def compute_position(self, xyz):
        matrix = copy.deepcopy(self.matrix)

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        location = np.asarray([xyz[0], xyz[1], xyz[2], 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def compute_matrix_pixel_to_position(self):
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def reset_array(self):
        self.display.secondary_array = None
        self.display.matrix = copy.deepcopy(self.matrix)
        self.display.origin = copy.deepcopy(self.origin)
        self.display.slice_location = self.compute_center(position=False, zyx=True)

    def retrieve_angles(self, order='ZXY'):
        rotation = Rotation.from_matrix(self.display.matrix[:3, :3])

        return rotation.as_euler(order, degrees=True)

    def retrieve_array_plane(self, slice_plane):
        return self.display.compute_array(slice_plane=slice_plane)

    def retrieve_slice_location(self, slice_plane):
        if slice_plane == 'Axial':
            return self.display.slice_location[0]

        elif slice_plane == 'Coronal':
            return self.display.slice_location[1]

        else:
            return self.display.slice_location[2]

    def retrieve_slice_position(self, slice_plane=None):
        pixel_to_position_matrix = self.display.compute_matrix_pixel_to_position()

        if slice_plane is None:
            location = np.asarray([self.display.slice_location[2],
                                   self.display.slice_location[1],
                                   self.display.slice_location[0], 1])
        else:
            if slice_plane == 'Axial':
                location = np.asarray([0, 0, self.display.slice_location[0], 1])
            elif slice_plane == 'Coronal':
                location = np.asarray([0, self.display.slice_location[1], 0, 1])
                print(location)
            else:
                location = np.asarray([self.display.slice_location[2], 0, 0, 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def retrieve_scroll_max(self, slice_plane):
        if slice_plane == 'Axial':
            return self.display.scroll_max[0]

        elif slice_plane == 'Coronal':
            return self.display.scroll_max[1]

        else:
            return self.display.scroll_max[2]

    def retrieve_vtk_slice(self, slice_plane):
        return self.display.compute_vtk_slice(slice_plane)

    def retrieve_vtk_volume(self, slice_plane):
        return self.display.compute_vtk_volume(slice_plane)

    def update_rotation(self, r_x=0, r_y=0, r_z=0, base=True):
        if r_x != 0 or r_y != 0 or r_z != 0:
            r = Rotation.from_euler('xyz', [r_x, r_y, r_z], degrees=True)
            new_matrix = r.as_matrix()

            if base:
                base_matrix = copy.deepcopy(self.matrix)
                self.display.matrix = new_matrix @ base_matrix
            else:
                self.display.matrix = new_matrix @ self.display.matrix

            self.display.compute_offaxis_array()
            self.display.compute_scroll_max()
        else:
            self.display.compute_scroll_max()
            self.reset_array()
