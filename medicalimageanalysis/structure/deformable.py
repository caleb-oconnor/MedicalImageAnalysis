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


class Deformable(object):
    def __init__(self, images):
        self.images = images

        self.reference_name = None
        self.target_name = None
        self.matrix = np.identity(4)
