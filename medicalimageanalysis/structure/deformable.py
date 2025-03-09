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
    def __init__(self, source_name=None, target_name=None, roi_names=None):

        self.source_name = None
        self.target_name = None

        self.matrix = np.identity(4)
        self.dvf = None

    def add_deformable(self):
        Data.deformable += [self]
