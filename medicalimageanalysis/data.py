"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Data Management Module for Medical Imaging and Transformations
==============================================================

This module provides a centralized, global state container for medical imaging
data, including volumetric images, dose distributions, and spatial
transformations (rigid and deformable).

Architecture:
    The module utilizes a Singleton-style pattern via class-level attributes
    in the `Data` class. This ensures that data loaded in one part of the
    application (e.g., a loader module) is immediately available to other
    components (e.g., a visualization or registration module) without
    explicit object passing.

Key Components:
    * Image Storage: Dict-based lookup for volumetric data.
    * Transformations: Storage for Rigid and Deformable registration results.
    * Annotations: Synchronization logic for ROIs (Regions of Interest) and
      POIs (Points of Interest) to ensure consistency across multiple datasets.

Usage:
    Import the class and access attributes directly:
    >>> from medicalimageanalysis import Data
    >>> Data.image_list

Warning:
    Because this class uses global state, calling `Data.clear()` will affect
    all modules currently importing this class.

"""


class Data(object):
    """
    A global orchestrator for medical imaging data and transformations.

    This class acts as a centralized, static repository (Singleton pattern)
    for managing volumetric data, spatial transformations, and clinical
    annotations. Because data is stored at the class level, all modifications
    are reflected globally across the application.

    Attributes:
        image (dict): Mapping of image names to image objects.
        rigid (dict): Mapping of transformation IDs to rigid transform objects.
        deformable (dict): Mapping of transformation IDs to deformable transform objects.
        dose (dict): Mapping of dose IDs to dose distribution objects.
        image_list (list): Ordered keys for registered images.
        roi_list (list): Master list of unique Region of Interest names.
        poi_list (list): Master list of unique Point of Interest names.
    """

    image = {}
    rigid = {}
    deformable = {}
    dose = {}

    image_list = []
    deformable_list = []
    dose_list = []
    poi_list = []
    rigid_list = []
    roi_list = []

    @classmethod
    def clear(cls):
        """
        Wipe all data from the global registry.

        This method performs a full reset of the class-level storage,
        effectively clearing the current session's memory. Use with caution
        as this action is non-reversible.

        Example:
            >>> Data.clear()
            >>> len(Data.image_list)
            0
        """
        cls.image = {}
        cls.rigid = {}
        cls.deformable = {}
        cls.dose = {}

        cls.image_list = []
        cls.poi_list = []
        cls.roi_list = []
        cls.rigid_list = []
        cls.deformable_list = []
        cls.dose_list = []

    @classmethod
    def delete_image(cls, image_name):
        """
        Remove a specific image and its registry entry.

        Args:
            image_name (str): The unique identifier/key of the image to delete.

        Raises:
            KeyError: If the image_name does not exist in the registry.
        """
        del cls.image[image_name]
        cls.image_list.remove(image_name)

    @classmethod
    def match_rois(cls):
        """
        Synchronize ROI definitions across all loaded images.

        This method performs a 'union' operation on all Region of Interest
        names found across the dataset. It ensures every image has a
        corresponding ROI object for every name in the master list.

        Logic:
            1. Aggregates all unique ROI names from all images.
            2. For each ROI, identifies its 'authoritative' color and visibility.
            3. Injects missing ROI definitions into images that lack them,
               preserving visual consistency across the session.
        """
        image_rois = [list(cls.image[image_name].rois.keys()) for image_name in list(cls.image.keys())]
        roi_names = list({x for r in image_rois for x in r})
        Data.roi_list = roi_names

        color = [[128, 128, 128]] * len(roi_names)
        visible = [False] * len(roi_names)
        for ii, roi_name in enumerate(roi_names):
            for image_name in list(cls.image.keys()):
                rois_on_image = list(cls.image[image_name].rois.keys())
                if roi_name in rois_on_image:
                    if cls.image[image_name].rois[roi_name].color is not None:
                        color[ii] = cls.image[image_name].rois[roi_name].color
                        visible[ii] = cls.image[image_name].rois[roi_name].visible

        for ii, roi_name in enumerate(roi_names):
            for image_name in list(cls.image.keys()):
                rois_on_image = list(cls.image[image_name].rois.keys())
                if roi_name not in rois_on_image:
                    cls.image[image_name].add_roi(roi_name=roi_name, color=color[ii], visible=visible[ii])

    @classmethod
    def match_pois(cls):
        """
        Synchronize POI definitions across all loaded images.

        Similar to `match_rois`, this ensures Point of Interest consistency.
        It is particularly useful when comparing landmarks across different
        modalities (e.g., CT vs MR) where a point identified in one should
        exist (even if unplaced) in the other.

        Note:
            Initializes missing POIs with a default grey color [128, 128, 128]
            and sets visibility to False.
        """
        image_pois = [list(cls.image[image_name].pois.keys()) for image_name in list(cls.image.keys())]
        poi_names = list({x for r in image_pois for x in r})
        Data.poi_list = poi_names

        color = [[128, 128, 128]] * len(poi_names)
        visible = [False] * len(poi_names)
        for ii, poi_name in enumerate(poi_names):
            for image_name in list(cls.image.keys()):
                pois_on_image = list(cls.image[image_name].pois.keys())
                if poi_name in pois_on_image:
                    if cls.image[image_name].pois[poi_name].color is not None:
                        color[ii] = cls.image[image_name].pois[poi_name].color
                        visible[ii] = cls.image[image_name].pois[poi_name].visible

        for ii, poi_name in enumerate(poi_names):
            for image_name in list(cls.image.keys()):
                pois_on_image = list(cls.image[image_name].pois.keys())
                if poi_name not in pois_on_image:
                    cls.image[image_name].add_poi(poi_name=poi_name, color=color[ii], visible=visible[ii])
