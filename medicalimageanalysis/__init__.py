
from .reader import Reader

from .read import DicomReader, MhdReader, NiftiReader, StlReader, VtkReader, ThreeMfReader
from .structure import Rigid, Deformable

from .data import Data
Data()
