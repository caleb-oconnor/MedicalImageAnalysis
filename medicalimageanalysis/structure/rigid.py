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

from ..utils.rigid.icp import IcpOpen3d
from ..data import Data


class Rigid(Data):
    def __init__(self, source_name=None, target_name=None, roi_names=None):

        self.source_name = None
        self.target_name = None
        self.roi_names = roi_names

        self.matrix = np.identity(4)
        self.combo_matrix = np.identity(4)
        self.combo_name = None

    def loaded_rigid(self):
        self.roi_names = ['Unknown']

    def compute_icp(self, algorithm='vtk'):
        if algorithm == 'vtk':
            icp = IcpOpen3d()
