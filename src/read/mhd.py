"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:

Functions:

"""

import os

import numpy as np
import SimpleITK as sitk


class MhdReader(object):
    """

    """
    def __init__(self, reader):
        self.reader = reader

    def load(self):
        for file_path in self.reader.files['Mhd']:
            self.read(file_path)

    def read(self, path):
        sitk_image = sitk.ReadImage(path)


