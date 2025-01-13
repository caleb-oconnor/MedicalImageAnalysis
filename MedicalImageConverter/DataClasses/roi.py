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


class Roi(object):
    def __init__(self, image, name, color, visible, filepaths):
        self.image = image

        self.name = name
        self.visible = visible
        self.color = color
        self.filepaths = filepaths

        self.contour_position = None
        self.contour_pixel = None
        self.mesh = None
        self.display_mesh = None
        self.decimate_mesh = None
        self.mesh_volume = None
        self.mesh_com = None
        self.bounds = None
