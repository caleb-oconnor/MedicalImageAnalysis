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

    def compute_icp_vtk(self, source_mesh, target_mesh, distance=10, iterations=1000, landmarks=None):
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

        matrix_reshape = self.matrix[:3, :3].reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions(Data.images[self.target_name].array.shape)
        vtk_image.SetOrigin(Data.images[self.target_name].origin)
        vtk_image.GetPointData().SetScalars(numpy_to_vtk(Data.images[self.target_name].array.flatten(order="F")))

        transform = vtk.vtkTransform()
        transform.RotateZ(self.angles[0])
        transform.RotateX(self.angles[1])
        transform.RotateY(self.angles[2])

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

    def compute_matrix_pixel_to_position(self):
        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = self.matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = self.matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = self.matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = self.matrix[0, :] / self.spacing[0]
        hold_matrix[1, :] = self.matrix[1, :] / self.spacing[1]
        hold_matrix[2, :] = self.matrix[2, :] / self.spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        return position_to_pixel_matrix

    def retrieve_array(self, slice_plane):
        position = Data.images[self.source_name].retrieve_slice_position(slice_plane)
        if slice_plane == 'Axial':
            array = np.flip(self.vtk_array[:, :, Data.images[self.target_name].slice_location[2]].T, 0)
        elif slice_plane == 'Coronal':
            array = self.vtk_array[:, Data.images[self.target_name].slice_location[1], :].T
        else:
            array = self.vtk_array[Data.images[self.target_name].slice_location[0], :, :].T

        return array

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
