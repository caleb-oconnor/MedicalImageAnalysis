"""Top-level package exports for :mod:`medicalimageanalysis`.

This package exposes the primary file-reading entry points, reader classes,
registration structures, and utility helpers used throughout the project.

Notes
-----
Importing this package instantiates :class:`medicalimageanalysis.data.Data`
once at module import time so the global registry is available immediately.

Examples
--------
>>> import medicalimageanalysis as mia
>>> files = mia.file_parser(folder_path='C:/Data/Patient01')
>>> mia.read_dicoms(file_list=files['Dicom'])
"""

from .reader import file_parser, read_3mf, read_dicoms, read_mhd

from .read import DicomReader, MhdReader, StlReader, ThreeMfReader, VtkReader
from .structure import Rigid, Deformable
from .utils import *

from .data import Data

Data()
