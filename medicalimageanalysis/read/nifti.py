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
import nibabel as nib


class NiftiReader(object):
    """

    """
    def __init__(self, reader):
        self.reader = reader

    def load(self):
        for file_path in self.reader.files['Nifti']:
            self.read(file_path)

    def read(self, path):
        nifti_image = nib.load(path)
        array = nifti_image.get_fdata()
        header = nifti_image.header
