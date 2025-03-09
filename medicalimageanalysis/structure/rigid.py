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


class Rigid(object):
    def __init__(self, source_name=None, target_name=None, roi_names=None):
        self.source_name = source_name
        self.target_name = target_name
        self.roi_names = roi_names

        self.matrix = np.identity(4)
        self.combo_matrix = np.identity(4)
        self.combo_name = None

    def loaded_rigid(self):
        self.roi_names = ['Unknown']

    def set_matrix(self, matrix, combo_matrix=None, combo_name=None):
        self.matrix = matrix
        self.combo_matrix = combo_matrix
        self.combo_name = combo_name

    def add_rigid(self):
        Data.rigid += [self]
