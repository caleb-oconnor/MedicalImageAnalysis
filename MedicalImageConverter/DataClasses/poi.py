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


class Poi(object):
    def __init__(self, image, name=None, color=None, visible=None, filepaths=None):
        self.image = image

        self.name = name
        self.visible = visible
        self.color = color
        self.filepaths = filepaths

        self.point_position = None
        self.point_pixel = None
