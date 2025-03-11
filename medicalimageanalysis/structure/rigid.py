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

from ..utils.rigid.icp import IcpVtk, IcpOpen3d
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

    def compute_icp_vtk(self, source_mesh, target_mesh, landmarks=None, distance=None, iterations=None):
        icp_vtk = IcpVtk(source_mesh, target_mesh)
        icp_vtk.update_parameters(landmarks=landmarks, distance=distance, iterations=iterations)
        if self.combo_name:
            self.matrix = icp_vtk.compute_icp(com_matching=False)
        else:
            self.matrix = icp_vtk.compute_icp(com_matching=True)

    def add_rigid(self):
        Data.rigid = [self]
