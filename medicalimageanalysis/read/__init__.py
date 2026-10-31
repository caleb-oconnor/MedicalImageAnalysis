"""Reader subpackage exports.

This subpackage groups format-specific readers for DICOM, MetaImage, 3MF,
STL, and VTK inputs so Sphinx can document them from a single namespace.

Examples
--------
>>> import medicalimageanalysis as mia
>>> files = mia.file_parser(folder_path='C:/Data/Patient01')
>>> mia.read_dicoms(file_list=files['Dicom'])
"""

from .dicom import DicomReader
from .mhd import MhdReader
from .mf3 import ThreeMfReader
from .stl import StlReader
from .vtk import VtkReader
