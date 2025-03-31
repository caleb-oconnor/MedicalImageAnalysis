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

        self.angles = None
        self.translation = None
        self.update_angles_translation()

    def compute_icp_vtk(self, source_mesh, target_mesh, distance=10, iterations=1000, landmarks=None):
        icp = ICP(source_mesh, target_mesh)
        if self.combo_name:
            icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=False)
        else:
            icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=True)

        self.matrix = icp.get_matrix()
        self.update_angles_translation()

    def update_matrix(self, matrix):
        self.matrix = matrix
        self.update_angles_translation()

    def update_angles_translation(self):
        rotation = Rotation.from_matrix(self.matrix[:3, :3])
        self.angles = rotation.as_euler("ZXY", degrees=True)
        self.translation = self.matrix[:3, 3]

    def add_rigid(self):
        Data.rigid = [self]
