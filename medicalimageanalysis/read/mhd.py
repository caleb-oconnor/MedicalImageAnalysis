"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

MetaImage (MHD) Reading and Object Dispatcher
=============================================

Description:
    This module provides the `MhdReader` class, which ingests .mhd/.raw
    file pairs through SimpleITK. It acts as a dispatcher: depending on the
    arguments it was built with, the same file is registered either as an
    image volume or as a deformable vector field (DVF).

Dispatch Rules:
    * no `reference_name`                       -> image in `Data.image`
    * `reference_name` + `moving_name` + `dvf`  -> DVF in `Data.deformable`
    * `reference_name` + `dose` / `roi_name`    -> reserved (no-op for now)

Key Functionalities:
    1. **Image Creation**: `create_image` converts the SimpleITK volume to
       a z-y-x array with its origin, spacing and direction, and registers
       it as an axial image.
    2. **DVF Creation**: `create_dvf` builds a `Deformable` object named
       'DVF_<reference>_<moving>', adding a numeric suffix on collision.
    3. **Stubs**: `create_roi` and `create_dose` are placeholders for
       future MetaImage label maps and dose grids.

Dependencies:
    - SimpleITK: header parsing and pixel array extraction.
    - numpy: geometry conversion (size flip, direction reshape).
    - ..data.Data: global state registry.

Architecture Note:
    SimpleITK returns arrays in z-y-x order while `GetSize`/`GetSpacing`/
    `GetOrigin` are x-y-z. Dimensions are flipped to z-y-x to match the
    array; spacing and origin stay x-y-z, as everywhere else in the package.

Usage:
    >>> from medicalimageanalysis.read.mhd import MhdReader
    >>> from medicalimageanalysis import Data
    >>> reader = MhdReader(file='C:/Data/cbct.mhd', modality='CT')
    >>> reader.load()
    >>> Data.image_list()

Examples
--------
>>> import medicalimageanalysis as mia
>>> from medicalimageanalysis import Data
>>> Data.clear()
>>> mia.read_mhd(file='C:/Data/Patient01/ct.mhd', modality='CT')
>>> mia.read_mhd(file='C:/Data/Patient01/mr.mhd', modality='MR')
>>> Data.image_list()
['CT 01', 'MR 02']
>>> mia.read_mhd(
...     file='C:/Data/Patient01/dvf_ct_mr.mhd',
...     reference_name='CT 01',
...     moving_name='MR 02',
...     dvf=True,
... )
>>> Data.deformable_list()
['DVF_CT 01_MR 02']
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

    Loads the file with SimpleITK and, depending on the constructor
    arguments, registers it as an image or as a deformable vector field.

    Parameters
    ----------
    file : str
        Path to the `.mhd` file (the `.raw` must sit next to it).

    modality : str, optional
        Imaging modality (e.g. 'CT', 'MR'). When given, the image is named
        '<modality> NN'. When None, the file stem is the name and the
        modality is recorded as 'CT'.

    reference_name : str, optional
        Loaded image the DVF is defined on. Its presence switches the
        reader from image mode to registration mode.

    moving_name : str, optional
        Loaded image the DVF maps to.

    roi_name : str, optional
        Reserved for label maps (not implemented).

    dose : object, optional
        Reserved for dose grids (not implemented).

    dvf : object, optional
        Any non-None value (normally True) marks the file as a vector field.

    Attributes
    ----------
    mhd : SimpleITK.Image | None
        The loaded SimpleITK image, set by `load`.

    Examples
    --------
    Load a volume as an image:

    >>> from medicalimageanalysis.read.mhd import MhdReader
    >>> from medicalimageanalysis import Data
    >>> Data.clear()
    >>> MhdReader(file='C:/Data/cbct.mhd', modality='CT').load()
    >>> Data.image_list()
    ['CT 01']

    Load a vector field between two images already in the session:

    >>> Data.image_list()
    ['CT 01', 'MR 02']
    >>> reader = MhdReader(
    ...     file='C:/Data/dvf_ct_mr.mhd',
    ...     reference_name='CT 01',
    ...     moving_name='MR 02',
    ...     dvf=True,
    ... )
    >>> reader.load()
    >>> Data.deformable_list()
    ['DVF_CT 01_MR 02']
    """

    def __init__(self, file, modality=None, reference_name=None,
                 moving_name=None, roi_name=None, dose=None, dvf=None):
        """
        Store the reader settings. Nothing is read until `load`.

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
            DVF flag.

        Examples
        --------
        The file is not opened at construction time:

        >>> from medicalimageanalysis.read.mhd import MhdReader
        >>> reader = MhdReader(file='C:/Data/cbct.mhd', modality='CT')
        >>> reader.modality, reader.mhd
        ('CT', None)
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
        Read the `.mhd` file and dispatch it to the matching creator.

        Logic:
            - If `reference_name` is set:
                - `dvf` and `moving_name` set -> `create_dvf`
                - otherwise the dose / ROI hooks are reserved (no-op)
            - Otherwise -> `create_image`

        Returns
        -------
        None

        Examples
        --------
        Image mode (no reference image given):

        >>> from medicalimageanalysis.read.mhd import MhdReader
        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> reader = MhdReader(file='C:/Data/cbct.mhd', modality='CT')
        >>> reader.load()
        >>> reader.mhd.GetSize()
        (512, 512, 88)
        >>> Data.image_list()
        ['CT 01']

        Registration mode without `dvf` currently loads the file but
        creates nothing:

        >>> MhdReader(file='C:/Data/dose.mhd', reference_name='CT 01', dose=True).load()
        >>> Data.dose_list()
        []
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
        Convert the loaded volume into an Image and register it.

        Builds a z-y-x array with `sitk.GetArrayFromImage`, flips the size
        to z-y-x dimensions, keeps origin/spacing in x-y-z, passes the
        direction as orientation, and registers the result through
        `CreateImageFromMask.add_image` with plane 'Axial'.

        Naming:
            - `modality` given -> '<modality> NN' (next index in session)
            - `modality` None  -> file stem, modality recorded as 'CT'

        Returns
        -------
        None

        Examples
        --------
        Called by `load`; `self.mhd` must already be set:

        >>> import SimpleITK as sitk
        >>> from medicalimageanalysis.read.mhd import MhdReader
        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> reader = MhdReader(file='C:/Data/cbct.mhd', modality='CT')
        >>> reader.mhd = sitk.ReadImage(reader.file)
        >>> reader.create_image()
        >>> Data.image_list()
        ['CT 01']

        Without a modality the file stem is used:

        >>> reader = MhdReader(file='C:/Data/liver_phase2.mhd')
        >>> reader.load()
        >>> Data.image_list(), reader.modality
        (['CT 01', 'liver_phase2'], 'CT')
        """
        if self.modality is None:
            filename = os.path.basename(self.file)
            image_name = os.path.splitext(filename)[0]
            self.modality = 'CT'
        else:
            idx = len(Data.image_list())
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
        Placeholder for creating an ROI from an `.mhd` label map.

        Not yet implemented; `load` never calls it.

        Returns
        -------
        None

        Examples
        --------
        Currently a no-op:

        >>> from medicalimageanalysis.read.mhd import MhdReader
        >>> MhdReader(file='C:/Data/liver_mask.mhd', roi_name='Liver').create_roi()
        """
        pass

    def create_dose(self):
        """
        Placeholder for creating a dose object from an `.mhd` dose grid.

        Not yet implemented; `load` never calls it.

        Returns
        -------
        None

        Examples
        --------
        Currently a no-op:

        >>> from medicalimageanalysis.read.mhd import MhdReader
        >>> MhdReader(file='C:/Data/dose.mhd', dose=True).create_dose()
        """
        pass

    def create_dvf(self):
        """
        Build a `Deformable` registration from the loaded vector field.

        The registration is named 'DVF_<reference>_<moving>'. If that name
        already exists, '_1', '_2', ... is appended until it is unique.
        The vector array is passed as returned by SimpleITK, i.e.
        (z, y, x, 3); the direction becomes a 3x3 `dvf_matrix`.

        Returns
        -------
        None

        Examples
        --------
        Called by `load` when `reference_name`, `moving_name` and `dvf`
        are set:

        >>> from medicalimageanalysis.read.mhd import MhdReader
        >>> from medicalimageanalysis import Data
        >>> reader = MhdReader(
        ...     file='C:/Data/dvf_ct_mr.mhd',
        ...     reference_name='CT 01',
        ...     moving_name='MR 02',
        ...     dvf=True,
        ... )
        >>> reader.load()
        >>> Data.deformable_list()
        ['DVF_CT 01_MR 02']

        Loading a second field for the same pair gets a suffix:

        >>> reader.load()
        >>> Data.deformable_list()
        ['DVF_CT 01_MR 02', 'DVF_CT 01_MR 02_1']
        """
        registration_name = (
            f"DVF_{self.reference_name}_{self.moving_name}"
        )

        if registration_name in Data.deformable_list():
            n = 0
            while True:
                n += 1
                new_name = f"{registration_name}_{n}"
                if new_name not in Data.deformable_list():
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