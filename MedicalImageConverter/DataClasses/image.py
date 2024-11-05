"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import copy
import numpy as np

import vtk
from vtkmodules.util import numpy_support

# from src.DataType.roi import Roi


class Image:
    def __init__(self):
        self.array = None
        self.rois = {}

        self.mrn = None
        self.patient_name = None
        self.description = None
        self.series_uid = None
        self.acquisition_number = None
        self.frame_uid = None
        self.rgb = None

        self.date = None
        self.time = None
        self.plane = None

        self.rows = None
        self.columns = None
        self.slices = None
        self.origin = None
        self.orientation = None

        self.dimension = None
        self.spacing = None
        self.matrix = None
        self.slice_location = None
        self.window = None
        self.camera_position = None
        self.skipped_slice = None
        self.unverified = None

    def get_current_slice(self, plane):
        if plane == 'Axial':
            current_slice = self.slice_location[2]

        elif plane == 'Sagittal':
            current_slice = self.slice_location[0]

        else:
            current_slice = self.slice_location[1]

        return current_slice

    def get_slice_max(self, plane):
        if plane == 'Axial':
            scroll_max = self.slices - 1

        elif plane == 'Sagittal':
            scroll_max = self.rows - 1

        else:
            scroll_max = self.columns - 1

        return scroll_max

    def get_xyz_spacing(self):
        """
        Returns spacing in xyz list format
        :return:
        """
        return [float(self.PixelSpacing[0]), float(self.PixelSpacing[1]), float(self.SliceThickness)]

    def get_xyz_dimensions(self):
        return [self.Rows, self.Columns, self.Slices]

    def create_roi(self, name=None, colors=None, visible=False, filepaths=None):
        self.rois[name] = Roi(self, name, colors, visible, filepaths)

    def create_vtk_slice(self, plane='Axial'):
        matrix_reshape = self.matrix[0:3, 0:3].reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)

        origin_slice = copy.deepcopy(self.origin)
        if plane == 'Axial':
            array_slice = self.array[:, :, self.slice_location[2]]
            origin_slice[2] = origin_slice[2] + (self.spacing[2] * self.slice_location[2])
            array_shape = array_slice.shape
            dim = [array_shape[0], array_shape[1], 1]

        elif plane == 'Sagittal':
            array_slice = self.array[self.slice_location[0], :, :]
            origin_slice[0] = origin_slice[0] + (self.spacing[0] * self.slice_location[0])
            array_shape = array_slice.shape
            dim = [1, array_shape[0], array_shape[1]]

        else:
            array_slice = self.array[:, self.slice_location[1], :]
            origin_slice[1] = origin_slice[1] + (self.spacing[1] * self.slice_location[1])
            array_shape = array_slice.shape
            dim = [array_shape[0], 1, array_shape[1]]

        vtk_image.SetDimensions(dim)
        vtk_image.SetOrigin(origin_slice)
        vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array_slice.flatten(order="F")))

        return vtk_image

    def create_vtk_volume(self):
        matrix_reshape = self.matrix[0:3, 0:3].reshape(1, 9)[0]
        vtk_volume = vtk.vtkImageData()
        vtk_volume.SetSpacing(self.spacing)
        vtk_volume.SetDirectionMatrix(matrix_reshape)
        vtk_volume.SetDimensions(self.array.shape)
        vtk_volume.SetOrigin(self.origin)
        vtk_volume.GetPointData().SetScalars(numpy_support.numpy_to_vtk(self.array.flatten(order="F")))

        return vtk_volume

    def create_vtk_volume_sliced(self, plane='Axial'):
        matrix_reshape = self.matrix[0:3, 0:3].reshape(1, 9)[0]
        vtk_volume = vtk.vtkImageData()
        vtk_volume.SetSpacing(self.spacing)
        vtk_volume.SetDirectionMatrix(matrix_reshape)

        origin_slice = copy.deepcopy(self.origin)
        if plane == 'Axial':
            array_slices = self.array[:, :, self.slice_location[2]:]
            origin_slice[2] = origin_slice[2] + (self.spacing[2] * self.slice_location[2])

        elif plane == 'Sagittal':
            array_slices = self.array[self.slice_location[0]:, :, :]
            origin_slice[0] = origin_slice[0] + (self.spacing[0] * self.slice_location[0])

        else:
            array_slices = self.array[:, self.slice_location[1]:, :]
            origin_slice[1] = origin_slice[1] + (self.spacing[1] * self.slice_location[1])

        vtk_volume.SetDimensions(dim)
        vtk_volume.SetOrigin(array_slices.shape)
        vtk_volume.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array_slices.flatten(order="F")))

        return vtk_volume
