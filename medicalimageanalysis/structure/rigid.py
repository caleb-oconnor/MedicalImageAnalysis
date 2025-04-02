"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import numpy as np

import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from scipy.spatial.transform import Rotation

from ..utils.rigid.icp import ICP
from ..data import Data


class Rigid(object):
    def __init__(self, source_name, target_name, rigid_name=None, roi_names=None, matrix=None, combo_matrix=None,
                 combo_name=None):
        self.source_name = source_name
        self.target_name = target_name
        self.combo_name = combo_name

        if rigid_name is None:
            self.rigid_name = self.source_name + '_' + self.target_name
        else:
            self.rigid_name = rigid_name

        if roi_names is None:
            self.roi_names = ['Unknown']
        else:
            self.roi_names = roi_names

        if matrix is None:
            self.matrix = np.identity(4)
        else:
            self.matrix = matrix

        if combo_matrix is None:
            self.combo_matrix = np.identity(4)
        else:
            self.combo_matrix = combo_matrix

        self.angles = np.asarray([0, 0, 0])
        self.translation = np.asarray([0, 0, 0])
        self.update_angles_translation()

        self.vtk_array = None
        self.origin = None
        self.spacing = None

    def compute_icp_vtk(self, source_mesh, target_mesh, distance=1e-5, iterations=1000, landmarks=None):
        icp = ICP(source_mesh, target_mesh)
        if self.combo_name:
            icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=False)
        else:
            icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=True)

        self.matrix = icp.get_matrix()
        self.update_angles_translation()

    def compute_image(self, spacing=None):
        if spacing is None:
            spacing = Data.images[self.target_name].spacing

        matrix_reshape = Data.images[self.target_name].matrix[:3, :3].reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions(Data.images[self.target_name].array.shape)
        vtk_image.SetOrigin(Data.images[self.target_name].origin)
        vtk_image.GetPointData().SetScalars(numpy_to_vtk(Data.images[self.target_name].array.flatten(order="F")))

        set_matrix = np.identity(4)
        set_matrix[:3, :3] = self.matrix[:3, :3]
        set_matrix = set_matrix.T

        transform = vtk.vtkTransform()
        matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                matrix.SetElement(i, j, set_matrix[i, j])
        transform.SetMatrix(matrix)

        vtk_reslice = vtk.vtkImageReslice()
        vtk_reslice.SetInputData(vtk_image)
        vtk_reslice.SetResliceTransform(transform)
        vtk_reslice.SetInterpolationModeToLinear()
        vtk_reslice.SetOutputSpacing(spacing)
        vtk_reslice.AutoCropOutputOn()
        vtk_reslice.Update()

        reslice_data = vtk_reslice.GetOutput()
        self.origin = reslice_data.GetOrigin() + self.translation
        self.spacing = reslice_data.GetSpacing()
        dimensions = reslice_data.GetDimensions()

        scalars = reslice_data.GetPointData().GetScalars()
        self.vtk_array = np.transpose(vtk_to_numpy(scalars).reshape(dimensions[2],
                                                                    dimensions[1],
                                                                    dimensions[0]), (2, 1, 0))

    def retrieve_array(self, slice_plane):
        position = Data.images[self.source_name].retrieve_slice_position(slice_plane)

        source_shape = np.asarray(Data.images[self.source_name].array.shape)
        image_pixel_to_position = Data.images[self.source_name].display.compute_matrix_pixel_to_position()
        source_edge_pixel = np.concatenate((source_shape - 1, [1]))
        source_edge_position = source_edge_pixel.dot(image_pixel_to_position.T)
        source_bounds = (source_edge_position[:3] - position)
        source_spacing = source_bounds / (source_shape - 1)

        pixel_location = (position - self.origin) / self.spacing
        pixel_space_correction = (self.spacing / source_spacing) * pixel_location

        shape = np.asarray(self.vtk_array.shape)
        bounds = shape * self.spacing

        array = None
        rect = [0, 0, 0, 0]
        if slice_plane == 'Axial':
            slice_location = int(np.round(pixel_location[2]))
            if 0 <= slice_location < shape[2]:
                array = np.flip(self.vtk_array[:, :, slice_location].T, 0)
                # pixel_shape_alteration = pixel_location[:2] + (shape - bounds)[:2]
                # offset = pixel_shape_alteration * (self.spacing / source_spacing)[:2]

                bounds_shift = (bounds[:2] / source_bounds[:2]) * (source_shape[:2] - 1)
                offset = [pixel_space_correction[0], (bounds_shift[1] - source_shape[1]) - pixel_space_correction[1]]
                rect = [-offset[0], -offset[1], bounds_shift[0], bounds_shift[1]]
                # rect = [58, -3, bounds_shift[0], bounds_shift[1]]

                # Original for same spacing between images
                # offset = [pixel_location[0], ((shape[1] - source_shape[1]) - pixel_location[1])]
                # rect = [-offset[0], -offset[1], shape[0], shape[1]]

        elif slice_plane == 'Coronal':
            array = self.vtk_array[:, Data.images[self.target_name].slice_location[1], :].T

        else:
            array = self.vtk_array[Data.images[self.target_name].slice_location[0], :, :].T

        return array, rect

    def update_rotation(self, matrix):
        self.matrix = matrix
        self.update_angles_translation()

    def update_angles_translation(self):
        rotation = Rotation.from_matrix(self.matrix[:3, :3])
        self.angles = rotation.as_euler("ZXY", degrees=True)
        self.translation = self.matrix[:3, 3]

    def add_rigid(self):
        if np.array_equal(self.combo_matrix, np.identity(4)):
            name = self.source_name + '_' + self.target_name
        else:
            name = self.source_name + '_' + self.target_name + '_combo'

        if name in Data.rigid_list:
            n = 0
            while n > -1:
                n += 1
                name = name + '_' + str(n)
                if name not in Data.rigid_list:
                    n = -100

        Data.rigid[name] = self
        Data.rigid_list += [name]
