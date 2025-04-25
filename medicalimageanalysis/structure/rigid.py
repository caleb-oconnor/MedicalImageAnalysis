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
        self.rotation_center = np.asarray([0, 0, 0])
        self.update_angles_translation()

        self.vtk_array = None
        self.origin = None
        self.spacing = None

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

    def compute_icp_vtk(self, source_mesh, target_mesh, distance=1e-5, iterations=1000, landmarks=None):
        icp = ICP(source_mesh, target_mesh)
        if self.combo_name:
            icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=False)
        else:
            icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=True)

        self.matrix = icp.get_matrix()
        self.update_angles_translation()

    def pre_alignment(self, superior=False, center=False, origin=False):
        if superior:
            pass
        elif center:
            self.matrix[:3, 3] = Data.images[self.source_name].origin - Data.images[self.target_name].origin
            self.rotation_center = np.asarray(Data.images[self.target_name].origin)
        elif origin:
            pass

    def update_rotation(self, t_x=0, t_y=0, t_z=0, r_x=0, r_y=0, r_z=0):
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

        if t_x:
            self.matrix[0, 3] = self.matrix[0, 3] + t_x

        if t_y:
            self.matrix[1, 3] = self.matrix[1, 3] + t_y

        if t_z:
            self.matrix[2, 3] = self.matrix[2, 3] + t_z

        self.matrix = new_matrix @ self.matrix

    def update_angles_translation(self):
        rotation = Rotation.from_matrix(self.matrix[:3, :3])
        self.angles = rotation.as_euler("ZXY", degrees=True)
        self.translation = self.matrix[:3, 3]

    def update_mesh(self, roi_name, base=True):
        if self.combo_name is None:
            roi = Data.images[self.target_name].rois[roi_name]
            if roi.mesh is not None and roi.visible:
                mesh = roi.mesh.translate(-self.rotation_center, inplace=False)
                mesh.transform(self.matrix, inplace=True)
                mesh.translate(self.rotation_center, inplace=True)

                return mesh

            else:
                return None
