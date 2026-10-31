"""Point-of-interest data model.

This module defines a lightweight container for named landmark points that are
associated with an :class:`medicalimageanalysis.structure.image.Image`.

Examples
--------
>>> import medicalimageanalysis as mia
>>> mia.read_mhd(file='C:/Data/scan.mhd', modality='CT')
>>> image = mia.Data.image[mia.Data.image_list[0]]
"""

import copy
import numpy as np


class Poi(object):
    """Represent a named point of interest attached to an image.
    
    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image[mia.Data.image_list[0]]
    >>> poi = image.pois['Isocenter']
    """

    def __init__(self, image, position=None, name=None, color=None, visible=None, filepaths=None):
        """Create a POI using world coordinates and optional display metadata.
        
        Examples
        --------
        >>> obj = Poi(...)
        >>> obj.__init__(...)
        """
        self.image = image

        self.name = name
        self.visible = visible
        self.color = color
        self.filepaths = filepaths

        self.point_position = position
        self.point_pixel = None
