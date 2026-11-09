"""Utility exports for ROI conversion, meshing, registration, and statistics.

The functions and classes re-exported here form the higher-level utility API
used throughout the package and are intended to be easy for Sphinx autodoc to
discover.

Examples
--------
>>> import numpy as np
>>> import medicalimageanalysis as mia
>>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
>>> contours = mia.mask_to_contours(mask, (0, 0, 0), (1, 1, 1), np.eye(3))
"""

from .roi.discrete_converters import ContourToDiscreteMesh, ContourToMask, MaskToContour, ModelToMask
from .roi.to_contours import mask_to_contours, mesh_to_contours
from .roi.to_mask import contours_to_mask, mesh_to_mask
from .roi.to_mesh import contours_to_mesh, contours_to_mask_and_mesh

from .mesh.volume import Volume
from .mesh.surface import Refinement
from .mesh.slicer import MeshSlicer

from .deformable.simpleitk import DeformableITK
from .roi.statistics import contour_longest_distance, mesh_longest_distance
