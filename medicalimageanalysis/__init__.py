
from .reader import file_parsar, read_3mf, read_dicoms, read_mhd

from .read import DicomReader, MhdReader, StlReader, ThreeMfReader, VtkReader
from .structure import Rigid, Deformable
from .utils import *

from .data import Data

Data()
