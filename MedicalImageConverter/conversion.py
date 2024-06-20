"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Converts a 3D model/s into a mask. The mask can either be an empty array (default) or a binary mask of the model/s.
    The mask is given a spacing buffer of 5 indexes on each side x, y, z.
"""

import os

import cv2
import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk


class ModelToMask:
    """
    Converts a 3D model/s into a mask. The mask can either be an empty array (default) or a binary mask of the model/s.
    The mask is given a spacing buffer of 5 indexes on each side x, y, z.
    """
    def __init__(self, models, empty_array=True, convert=True):
        """

        Parameters
        ----------
        models - List of all models
        empty_array -
        convert
        """
        self.models = models
        self.empty_array = empty_array

        self.bounds = None
        self.spacing = None
        self.dims = None
        self.slice_locations = None

        self.mask = None
        self.origin = None

        if convert:
            self.compute_bounds()
            self.compute_mask()

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_spacing(self, spacing):
        self.spacing = spacing

    def compute_bounds(self):
        """
        Computes the boundary for the mask using the model/s bounds. The boundary is the min/max x, y, z combination of
        all the model/s bounds. Default spacing options are [1, 1, 3] or [1, 1, 5] depending on the z axis bound. If
        the bounds are too large then nothing is computed with the assumption that the models are not from the same
        image and the models should have their own independent mask.
        Returns
        -------

        """
        model_bounds = [model.GetBounds() for model in self.models]
        model_min = np.min(model_bounds, axis=0)
        model_max = np.max(model_bounds, axis=0)

        model_min_max = [model_min[0], model_max[1], model_min[2], model_max[3], model_min[4], model_max[5]]

        if model_min_max[1] - model_min_max[0] < 512 and model_min_max[3] - model_min_max[2] < 512:
            if model_min_max[5] - model_min_max[4] < 450:
                self.spacing = [1, 1, 3]

            elif model_min_max[5] - model_min_max[4] < 750:
                self.spacing = [1, 1, 5]

        if self.spacing is not None:
            self.bounds = [int(model_min_max[0] - 5 * self.spacing[0]), int(model_min_max[1] + 5 * self.spacing[0]),
                           int(model_min_max[2] - 5 * self.spacing[1]), int(model_min_max[3] + 5 * self.spacing[1]),
                           int(model_min_max[4] - 5 * self.spacing[2]), int(model_min_max[5] + 5 * self.spacing[2])]
            self.origin = [self.bounds[0], self.bounds[2], self.bounds[4]]

            self.slice_locations = [i for i in range(self.bounds[4], self.bounds[5], self.spacing[2])]
            self.dims = [len(self.slice_locations), self.bounds[1] - self.bounds[0] + 1, self.bounds[3] - self.bounds[2] + 1]

    def compute_mask(self):
        """
        Default is a empty array. If a binary mask is wanted then the contours are computed for each model, then the
        contours are used to fill the binary mask.
        Returns
        -------

        """
        self.mask = np.zeros((self.dims[0], self.dims[2], self.dims[1]))
        if not self.empty_array:
            for model in self.models:
                com = model.center
                org_bounds = model.GetBounds()

                contours = []
                for s in self.slice_locations:
                    if org_bounds[4] < s < org_bounds[5]:
                        hold_contour = model.slice(normal='z', origin=[com[0], com[1], s])
                        contours.append((np.asarray(hold_contour.points)[:, 0:2] -
                                         (self.bounds[0], self.bounds[2])) / (self.spacing[0:2]))
                    else:
                        contours.append([])

                for ii, s in enumerate(self.slice_locations):
                    if len(contours[ii]) > 0:
                        frame = np.zeros((self.dims[2], self.dims[1]))
                        # noinspection PyTypeChecker
                        cv2.fillPoly(frame, np.array([contours[ii]], dtype=np.int32), 1)
                        self.mask[ii, :, :] = self.mask[ii, :, :] + frame

        self.mask = self.mask.astype(np.int8)

    def save_image(self, path):
        """
        Uses SimpleITK to write out the mask.

        Returns
        -------

        """
        image = sitk.GetImageFromArray(self.mask)
        image.SetSpacing(self.spacing)
        image.SetOrigin([self.bounds[0], self.bounds[2], self.bounds[4]])
        sitk.WriteImage(image, path)
