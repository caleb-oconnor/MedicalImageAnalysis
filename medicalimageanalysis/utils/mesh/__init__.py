"""Mesh-processing utility namespace.

Examples
--------
>>> import numpy as np
>>> import medicalimageanalysis as mia
>>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
"""
from .volume import Volume
from .surface import Refinement, expansion, surface_boundary
