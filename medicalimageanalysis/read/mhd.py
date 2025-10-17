"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import os

import numpy as np

import SimpleITK as sitk

from ..data import Data
from ..utils.creation import CreateImageFromMask


class MhdReader(object):
    def __init__(self, file, modality=None, reference_name=None, roi_name=None, dose=None, dvf=None):
        self.file = file
        self.modality = modality
        self.reference_name = reference_name
        self.roi_name = roi_name
        self.dose = dose
        self.dvf = dvf

        self.mhd = None

    def load(self):
        self.mhd = sitk.ReadImage(self.file)

        if self.reference_name is not None:
            if self.dvf is not None:
                pass

            elif self.dose is not None:
                pass

            elif self.roi_name is not None:
                pass

            else:
                self.create_image()

        else:
            self.create_image()

    def create_image(self):
        if self.modality is None:
            filename = os.path.basename(self.file)
            image_name = os.path.splitext(filename)[0]
            self.modality = 'CT'
        else:
            idx = len(Data.image_list)
            if idx < 9:
                image_name = self.modality + ' 0' + str(1 + idx)
            else:
                image_name = self.modality + ' ' + str(1 + idx)

        dimensions = np.flip(np.asarray(self.mhd.GetSize()))
        orientation = np.asarray(self.mhd.GetDirection())
        origin = np.asarray(self.mhd.GetOrigin())
        spacing = np.asarray(self.mhd.GetSpacing())

        array = sitk.GetArrayFromImage(self.mhd)

        create = CreateImageFromMask(array, origin, spacing, image_name, dimensions=dimensions, orientation=orientation,
                                     plane='Axial', description='Mhd to Image', modality=self.modality)
        create.add_image()

        print(1)

    def create_roi(self):
        pass

    def create_dose(self):
        pass

    def create_dvf(self):
        pass
