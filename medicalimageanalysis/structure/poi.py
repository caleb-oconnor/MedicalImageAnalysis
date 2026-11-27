"""Point-of-interest data model.

This module defines a lightweight container for named landmark points that are
associated with an :class:`medicalimageanalysis.structure.image.Image`.

Examples
--------
>>> import medicalimageanalysis as mia
>>> mia.read_mhd(file='C:/Data/scan.mhd', modality='CT')
>>> image = mia.Data.image[mia.Data.image_list()[0]]
"""

import copy
import numpy as np


class Poi(object):
    """Represent a named point of interest attached to an image.

    Attributes
    ----------
    id : int | None
        Permanent POI id, written by the image's POI registry. The same POI has the same id on every image.

    group_id : int | None
        POI group the POI is in (``image.poi_group``), the same on every image. None = ungrouped. Set it with
        ``Data.set_group('poi', ...)``.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image[mia.Data.image_list()[0]]
    >>> poi = image.poi['Isocenter']
    """

    def __init__(self, image, position=None, name=None, color=None, visible=None, filepaths=None):
        """Create a POI using world coordinates and optional display metadata.

        Examples
        --------
        >>> obj = Poi(...)
        >>> obj.__init__(...)
        """
        self.image = image

        self.id = None
        self.name = name
        self.visible = visible
        self.color = color
        self.filepaths = filepaths

        self.group_id = None

        self.point_position = position
        self.point_pixel = None