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

import SimpleITK as sitk


class Roi(object):
    def __init__(self, image, position=None, name=None, color=None, visible=None, filepaths=None):
        self.image = image

        self.name = name
        self.visible = visible
        self.color = color
        self.filepaths = filepaths

        if position is not None:
            self.contour_position = position
            self.contour_pixel = self.convert_position_to_pixel()
        else:
            self.contour_position = None
            self.contour_pixel = None
            
        self.mesh = None
        self.display_mesh = None
        self.decimate_mesh = None
        self.mesh_volume = None
        self.mesh_com = None
        self.bounds = None

    def convert_position_to_pixel(self):
        sitk_image = self.image.create_sitk_image(empty=True)

        pixel = [[]] * len(self.contour_position)
        for ii, contours in enumerate(self.contour_position):
            pixel[ii] = [sitk_image.TransformPhysicalPointToContinuousIndex(contour) for contour in contours]

        return pixel
