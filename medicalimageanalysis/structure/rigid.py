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

from scipy.spatial.transform import Rotation

from ..utils.rigid.icp import ICP
from ..data import Data


class Display(object):
    def __init__(self, rigid):
        self.rigid = rigid

        self.origin = None
        self.spacing = None
        self.bounds = None
        self.array = None

        self.slice_location = [0, 0, 0]
        self.scroll_max = None
        self.offset = {'Axial': [0, 0], 'Coronal': [0, 0], 'Sagittal': [0, 0]}

    def compute_array_slice(self, slice_plane):
        array_slice = None
        if slice_plane == 'Axial':
            if 0 <= self.slice_location[0] < self.array.shape[0]:
                array_slice = self.array[self.slice_location[0], :, :].astype(np.double)

        elif slice_plane == 'Coronal':
            if 0 <= self.slice_location[1] < self.array.shape[1]:
                array_slice = self.array[:, self.slice_location[1], :].astype(np.double)

        else:
            if 0 <= self.slice_location[2] < self.array.shape[2]:
                array_slice = self.array[:, :, self.slice_location[2]].astype(np.double)

        return array_slice

    def compute_offset(self):
        pos = Data.images[self.rigid.source_name].origin

        self.offset['Axial'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
        self.offset['Axial'][1] = (self.origin[1] - pos[1]) / self.spacing[1]
        self.offset['Coronal'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
        self.offset['Coronal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]
        self.offset['Sagittal'][0] = (self.origin[1] - pos[1]) / self.spacing[1]
        self.offset['Sagittal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]

    def compute_matrix_pixel_to_position(self):
        matrix = copy.deepcopy(Data.images[self.rigid.target_name].matrix)

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
        matrix = copy.deepcopy(Data.images[self.rigid.target_name].matrix)

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / self.spacing[0]
        hold_matrix[1, :] = matrix[1, :] / self.spacing[1]
        hold_matrix[2, :] = matrix[2, :] / self.spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        return position_to_pixel_matrix

    def compute_mesh_slice(self, roi_name=None, location=None, slice_plane=None, return_pixel=False):
        matrix = np.identity(4)
        if slice_plane == 'Axial':
            normal = matrix[:3, 2]
        elif slice_plane == 'Coronal':
            normal = matrix[:3, 1]
        else:
            normal = matrix[:3, 0]

        roi_slice = self.rigid.rois[roi_name].slice(normal=normal, origin=location)

        if return_pixel:
            if roi_slice.number_of_points > 0:
                roi_strip = roi_slice.strip(max_length=10000000)

                position = [np.asarray(c.points) for c in roi_strip.cell]
                pixels = self.convert_position_to_pixel(position=position)
                pixel_corrected = []
                for pixel in pixels:

                    if slice_plane in 'Axial':
                        pixel_reshape = pixel[:, :2]
                        pixel_corrected += [np.asarray([pixel_reshape[:, 0], pixel_reshape[:, 1]]).T]

                    elif slice_plane == 'Coronal':
                        pixel_reshape = np.column_stack((pixel[:, 0], pixel[:, 2]))
                        pixel_corrected += [pixel_reshape]

                    else:
                        pixel_reshape = pixel[:, 1:]
                        pixel_corrected += [pixel_reshape]

                return pixel_corrected

            else:
                return []

        else:
            return roi_slice

    def compute_slice_location(self, position=None):
        bounds = np.asarray([self.bounds[0], self.bounds[2], self.bounds[4]])
        if position is None:
            source_location = np.flip(Data.images[self.rigid.source_name].display.slice_location)
            position = Data.images[self.rigid.source_name].display.compute_index_positions(source_location)
        self.slice_location = np.flip(np.round((position - bounds) / self.spacing).astype(np.int32))

    def compute_slice_origin(self, slice_plane):
        slice_origin = None
        if slice_plane == 'Axial' and 0 <= self.slice_location[0] <= self.scroll_max:
            location = np.asarray([0, 0, self.slice_location[0]])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Coronal' and 0 <= self.slice_location[1] <= self.scroll_max:
            location = np.asarray([0, self.slice_location[1], 0])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Sagittal' and 0 <= self.slice_location[2] <= self.scroll_max:
            location = np.asarray([self.slice_location[2], 0, 0])
            slice_origin = self.origin + (location * self.spacing)

        return slice_origin

    def compute_scroll_max(self):
        self.scroll_max = [self.array.shape[0] - 1,
                           self.array.shape[1] - 1,
                           self.array.shape[2] - 1]

    def compute_vtk_slice(self, slice_plane):
        if self.array is None:
            self.compute_reslice()
            self.compute_slice_location()
            self.scroll_max = [self.array.dimensions[0] - 1,
                               self.array.dimensions[1] - 1,
                               self.array.dimensions[2] - 1]

        slice_array = None
        slice_origin = None
        array_shape = self.array.shape
        if slice_plane == 'Axial' and 0 <= self.slice_location[0] <= self.scroll_max:
            location = np.asarray([0, 0, self.slice_location[0]])
            slice_origin = self.origin + (location * self.spacing)

            slice_array = np.zeros((1, array_shape[1], array_shape[2]))
            slice_array[0, :, :] = self.array[self.slice_location[0], :, :]

        elif slice_plane == 'Coronal' and 0 <= self.slice_location[1] <= self.scroll_max:
            location = np.asarray([0, self.slice_location[1], 0])
            slice_origin = self.origin + (location * self.spacing)

            slice_array = np.zeros((array_shape[0], 1, array_shape[2]))
            slice_array[:, 0, :] = self.array[:, self.slice_location[1], :]

        elif slice_plane == 'Sagittal' and 0 <= self.slice_location[2] <= self.scroll_max:
            location = np.asarray([self.slice_location[2], 0, 0])
            slice_origin = self.origin + (location * self.spacing)

            slice_array = np.zeros((array_shape[0], array_shape[1], 1))
            slice_array[:, :, 0] = self.array[:, :, self.slice_location[2]]

        vtk_image = None
        if slice_array is not None:
            vtk_image = vtk.vtkImageData()
            vtk_image.SetSpacing(self.spacing)
            vtk_image.SetDirectionMatrix(1, 0, 0, 0, 1, 0, 0, 0, 1)
            vtk_image.SetDimensions(np.flip(slice_array.shape))
            vtk_image.SetOrigin(slice_origin)
            vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(self.array.flatten(order="C")))

        return vtk_image

    def compute_reslice(self):
        name = self.rigid.target_name
        matrix_reshape = Data.images[name].matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(Data.images[name].spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions(np.flip(Data.images[name].array.shape))
        vtk_image.SetOrigin(Data.images[name].origin)
        vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(Data.images[name].array.flatten(order="C")))

        matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                matrix.SetElement(i, j, self.rigid.matrix[i, j])

        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        # transform.Inverse()

        vtk_reslice = vtk.vtkImageReslice()
        vtk_reslice.SetInputData(vtk_image)
        vtk_reslice.SetResliceTransform(transform)
        vtk_reslice.SetInterpolationModeToLinear()
        vtk_reslice.SetOutputSpacing(Data.images[self.rigid.source_name].spacing)
        vtk_reslice.SetOutputDirection(1, 0, 0, 0, 1, 0, 0, 0, 1)
        vtk_reslice.AutoCropOutputOn()
        vtk_reslice.SetBackgroundLevel(-3001)
        vtk_reslice.Update()

        reslice_data = vtk_reslice.GetOutput()
        self.origin = np.asarray(reslice_data.GetOrigin())
        self.spacing = reslice_data.GetSpacing()
        self.bounds = reslice_data.GetBounds()
        dimensions = reslice_data.GetDimensions()
        self.compute_offset()

        scalars = reslice_data.GetPointData().GetScalars()
        self.array = numpy_support.vtk_to_numpy(scalars).reshape(dimensions[2], dimensions[1], dimensions[0])

        if self.rigid.combo_name is not None:
            vtk_image = vtk.vtkImageData()
            vtk_image.SetSpacing(self.spacing)
            vtk_image.SetDirectionMatrix(matrix_reshape)
            vtk_image.SetDimensions(np.flip(self.array.shape))
            vtk_image.SetOrigin(self.origin)
            vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(self.array.T.flatten(order="F")))

            rotation = Rotation.from_matrix(self.rigid.combo_matrix[:3, :3])
            euler_angles = rotation.as_euler("ZXY", degrees=True)

            transform = vtk.vtkTransform()
            transform.RotateZ(-euler_angles[0])
            transform.RotateX(euler_angles[1])
            transform.RotateY(euler_angles[2])

            vtk_reslice = vtk.vtkImageReslice()
            vtk_reslice.SetInputData(vtk_image)
            vtk_reslice.SetResliceTransform(transform)
            vtk_reslice.SetInterpolationModeToLinear()
            vtk_reslice.AutoCropOutputOn()
            vtk_reslice.Update()

            reslice_data = vtk_reslice.GetOutput()
            self.origin = reslice_data.GetOrigin()
            self.spacing = reslice_data.GetSpacing()
            dimensions = reslice_data.GetDimensions()

            scalars = reslice_data.GetPointData().GetScalars()
            self.array = numpy_support.vtk_to_numpy(scalars).reshape(dimensions[2], dimensions[1], dimensions[0])

    def convert_position_to_pixel(self, position=None):
        position_to_pixel_matrix = Data.images[self.rigid.source_name].display.compute_matrix_position_to_pixel()

        pixel = []
        for ii, pos in enumerate(position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            pixel_3_axis = p_concat.dot(position_to_pixel_matrix.T)[:, :3]
            pixel += [np.vstack((pixel_3_axis, pixel_3_axis[0, :]))]

        return pixel


class Rigid(object):
    def __init__(self, source_name, target_name, rigid_name=None, roi_names=None, matrix=None, combo_matrix=None,
                 combo_name=None):
        self.source_name = source_name
        self.target_name = target_name
        self.combo_name = combo_name
        self.rois = dict.fromkeys(Data.roi_list)

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

        self.rigid_name = self.add_rigid(rigid_name)
        self.rotation_center = np.asarray([0, 0, 0])
        self.display = Display(self)

    def add_rigid(self, rigid_name):
        if rigid_name is None:
            if np.array_equal(self.combo_matrix, np.identity(4)):
                rigid_name = self.source_name + '_' + self.target_name
            else:
                rigid_name = self.source_name + '_' + self.target_name + '_combo'

            if rigid_name in Data.rigid_list:
                n = 0
                while n > -1:
                    n += 1
                    rigid_name = rigid_name + '_' + str(n)
                    if rigid_name not in Data.rigid_list:
                        n = -100

        Data.rigid[rigid_name] = self
        Data.rigid_list += [rigid_name]

        return rigid_name

    def compute_icp_vtk(self, source_mesh, target_mesh, distance=1e-5, iterations=1000, landmarks=None,
                        com_matching=True, inverse=False):
        icp = ICP(source_mesh, target_mesh)
        icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=com_matching,
                        inverse=inverse)
        self.matrix = icp.get_matrix()
        self.update_mesh(all_rois=True)

    def pre_alignment(self, superior=False, center=False, origin=False):
        if superior:
            pass
        elif center:
            self.matrix[:3, 3] = Data.images[self.source_name].origin - Data.images[self.target_name].origin
        elif origin:
            pass

    def retrieve_array_plane(self, slice_plane, solo=None, position=None):
        if self.display.array is None:
            self.display.compute_reslice()
            self.display.compute_scroll_max()

        if solo is None:
            self.display.compute_slice_location(position=position)

        return self.display.compute_array_slice(slice_plane=slice_plane)

    def retrieve_offset(self, slice_plane):
        return self.display.offset[slice_plane]

    def update_rotation(self, r_x=0, r_y=0, r_z=0):
        new_matrix = np.identity(4)
        if r_x:
            radians = np.deg2rad(r_x)
            new_matrix[:3, :3] = Rotation.from_euler('x', radians).as_matrix()
        if r_y:
            radians = np.deg2rad(r_y)
            new_matrix[:3, :3] = Rotation.from_euler('y', radians).as_matrix()

        if r_z:
            radians = np.deg2rad(r_z)
            new_matrix[:3, :3] = Rotation.from_euler('z', radians).as_matrix()

        self.matrix = new_matrix @ self.matrix
        self.display.compute_reslice()
        self.display.compute_scroll_max()
        self.update_mesh(all_rois=True)

    def update_translation(self, t_x=0, t_y=0, t_z=0):
        self.matrix[0, 3] = self.matrix[0, 3] + t_x
        self.matrix[1, 3] = self.matrix[1, 3] + t_y
        self.matrix[2, 3] = self.matrix[2, 3] + t_z

        self.display.origin[0] = self.display.origin[0] + t_x
        self.display.origin[1] = self.display.origin[1] + t_y
        self.display.origin[2] = self.display.origin[2] + t_z

        self.display.compute_offset()
        self.display.compute_scroll_max()
        self.update_mesh(all_rois=True)

    def retrieve_angles(self, order='ZXY'):
        rotation = Rotation.from_matrix(self.matrix[:3, :3])
        return rotation.as_euler(order, degrees=True)

    def retrieve_translation(self):
        return self.matrix[:3, 3]

    def update_mesh(self, roi_name=None, all_rois=False):
        if all_rois:
            for roi_name in Data.roi_list:
                roi = Data.images[self.target_name].rois[roi_name]
                if roi.mesh is not None:
                    self.rois[roi_name] = roi.mesh.translate(-self.rotation_center, inplace=False)
                    self.rois[roi_name].transform(np.linalg.inv(self.matrix), inplace=True)
                    self.rois[roi_name].translate(self.rotation_center, inplace=True)

        else:
            roi = Data.images[self.target_name].rois[roi_name]
            if roi.mesh is not None and roi.visible:
                self.rois[roi_name] = roi.mesh.translate(-self.rotation_center, inplace=False)
                self.rois[roi_name].transform(np.linalg.inv(self.matrix), inplace=True)
                self.rois[roi_name].translate(self.rotation_center, inplace=True)

