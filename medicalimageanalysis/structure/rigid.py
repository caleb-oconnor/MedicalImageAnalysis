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


class Rigid(object):
    def __init__(self, source_name, target_name, roi_names):

        self.reference_name = None
        self.target_name = None
        self.roi_names = roi_names

        self.matrix = np.identity(4)
        self.combo_matrix = np.identity(4)
        self.combo_name = None

    def compute_icp(self, algorithm='vtk'):
        if algorithm == 'vtk':
            icp = IcpOpen3d()
