"""Core structure classes used by the package.

These classes represent loaded images and their associated rigid or
deformable registrations.

Examples
--------
>>> import medicalimageanalysis as mia
>>> mia.read_mhd(file='C:/Data/scan.mhd', modality='CT')
>>> image = mia.Data.image[mia.Data.image_list()[0]]
"""

from .image import Image
from .rigid import Rigid
from .deformable import Deformable
