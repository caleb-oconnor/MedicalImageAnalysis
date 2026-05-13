"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

MetaImage (MHD) Reading and Object Dispatcher
=============================================

Description:
    This module provides the `MhdReader` class, designed to ingest .mhd/.raw
    file pairs. Unlike a standard file reader, this class acts as a
    dispatcher; based on the initialization flags, it can transform raw
    volumetric data into standard imaging volumes or complex 4D Deformable
    Vector Fields (DVFs).

Structure:
    * MhdReader: The primary class handling SimpleITK integration.
        - load(): Main entry point that routes data to specific creators.
        - create_image(): Generates an axial Image object and registers
          it to the global Data state.
        - create_dvf(): Constructs a Deformable object for registration
          tracking between a reference and moving image.
        - Stubs: Reserved methods for create_roi and create_dose for
          future MetaImage extensions.

Dependencies:
    - SimpleITK: For robust header parsing and pixel array extraction.
    - numpy: For coordinate system transformations (matrix reshaping/flipping).
    - ..data.Data: Global state registry for application-wide data access.

Usage:
    >>> reader = MhdReader(file="path/to/data.mhd", modality="CT")
    >>> reader.load()  # This populates Data.image and Data.image_list

"""

import os

import numpy as np

import SimpleITK as sitk

from ..data import Data
from ..structure.deformable import Deformable
from ..utils.creation import CreateImageFromMask



class MhdReader(object):
    """
    Reader for `.mhd` (MetaImage) files.

    This class loads volumetric medical images and optionally creates:
    - Images (default behavior)
    - Deformable Vector Fields (DVFs)
    - ROIs (stub)
    - Dose objects (stub)

    The behavior depends on which optional parameters are provided during initialization.

    Parameters
    ----------
    file : str
        Path to the `.mhd` file.
    modality : str, optional
        Imaging modality (e.g., 'CT', 'MR'). If not provided, inferred as 'CT'.
    reference_name : str, optional
        Reference image name used for DVF registration.
    moving_name : str, optional
        Moving image name used for DVF registration.
    roi_name : str, optional
        Name of ROI to create (not yet implemented).
    dose : object, optional
        Dose object (not yet implemented).
    dvf : object, optional
        Deformable vector field flag/object.

    Examples
    --------
    Basic image loading::

        reader = MhdReader("scan.mhd")
        reader.load()

    DVF creation::

        reader = MhdReader(
            file="dvf.mhd",
            reference_name="CT_01",
            moving_name="CT_02",
            dvf=True
        )
        reader.load()
    """

    def __init__(self, file, modality=None, reference_name=None,
                 moving_name=None, roi_name=None, dose=None, dvf=None):
        """
        Initialize the MHD reader.

        Parameters
        ----------
        file : str
            Path to `.mhd` file.
        modality : str, optional
            Imaging modality (CT, MR, etc.).
        reference_name : str, optional
            Reference image name for DVF creation.
        moving_name : str, optional
            Moving image name for DVF creation.
        roi_name : str, optional
            ROI name (not implemented).
        dose : object, optional
            Dose object (not implemented).
        dvf : object, optional
            DVF flag/object.
        """
        self.file = file
        self.modality = modality
        self.reference_name = reference_name
        self.moving_name = moving_name
        self.roi_name = roi_name
        self.dose = dose
        self.dvf = dvf

        self.mhd = None

    def load(self):
        """
        Load the `.mhd` file and dispatch processing based on configuration.

        Logic:
        - If `reference_name` is provided:
            - Create DVF if `dvf` and `moving_name` are set
            - Dose/ROI hooks are reserved
        - Otherwise:
            - Create image object
        """
        self.mhd = sitk.ReadImage(self.file)

        if self.reference_name is not None:

            if self.dvf is not None and self.moving_name is not None:
                self.create_dvf()

            elif self.dose is not None:
                pass  # TODO: implement dose

            elif self.roi_name is not None:
                pass  # TODO: implement ROI

        else:
            self.create_image()

    def create_image(self):
        """
        Convert `.mhd` image into internal Image object and register it.

        The image is converted from a SimpleITK volume into:
        - NumPy array
        - Proper spacing/origin/orientation metadata
        - Registered via `CreateImageFromMask`
        """
        if self.modality is None:
            filename = os.path.basename(self.file)
            image_name = os.path.splitext(filename)[0]
            self.modality = 'CT'
        else:
            idx = len(Data.image_list)
            image_name = (
                f"{self.modality} {idx + 1:02d}"
                if idx < 9 else
                f"{self.modality} {idx + 1}"
            )

        dimensions = np.flip(np.asarray(self.mhd.GetSize()))
        orientation = np.asarray(self.mhd.GetDirection())
        origin = np.asarray(self.mhd.GetOrigin())
        spacing = np.asarray(self.mhd.GetSpacing())

        array = sitk.GetArrayFromImage(self.mhd)

        creator = CreateImageFromMask(
            array,
            origin,
            spacing,
            image_name,
            dimensions=dimensions,
            orientation=orientation,
            plane='Axial',
            description='Mhd to Image',
            modality=self.modality
        )
        creator.add_image()

    def create_roi(self):
        """
        Placeholder for ROI creation from `.mhd`.

        Not yet implemented.
        """
        pass

    def create_dose(self):
        """
        Placeholder for dose creation from `.mhd`.

        Not yet implemented.
        """
        pass

    def create_dvf(self):
        """
        Create a deformable vector field (DVF) from `.mhd`.

        Handles:
        - Name collision resolution in `Data.deformable_list`
        - Conversion of voxel field into DVF object
        - Registration of DVF into global `Data` structure
        """
        registration_name = (
            f"DVF_{self.reference_name}_{self.moving_name}"
        )

        if registration_name in Data.deformable_list:
            n = 0
            while True:
                n += 1
                new_name = f"{registration_name}_{n}"
                if new_name not in Data.deformable_list:
                    registration_name = new_name
                    break

        dimensions = np.flip(np.asarray(self.mhd.GetSize()))
        dvf_matrix = np.asarray(self.mhd.GetDirection()).reshape(3, 3)
        origin = np.asarray(self.mhd.GetOrigin())
        spacing = np.asarray(self.mhd.GetSpacing())

        array = sitk.GetArrayFromImage(self.mhd)

        Deformable(
            array,
            origin,
            spacing,
            dimensions,
            dvf_matrix=dvf_matrix,
            registration_name=registration_name,
            reference_name=self.reference_name,
            moving_name=self.moving_name
        )
