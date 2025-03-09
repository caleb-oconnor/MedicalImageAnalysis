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


class Rigid(object):
    def __init__(self, images):
        self.images = images

        self.reference_name = None
        self.target_name = None
        self.rotation_center = (0, 0, 0)
        self.matrix = np.identity(4)
        self.inverse_matrix = None

        self.angles = np.zeros(3)
        self.translation = np.zeros(3)

        self.combo_name = None
        self.combo_matrix = None
