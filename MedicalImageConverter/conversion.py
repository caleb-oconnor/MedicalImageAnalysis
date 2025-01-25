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

import vtk
from vtk.util import numpy_support


class ContourToDiscreteMesh(object):
    def __init__(self, contour_position=None, contour_pixel=None, spacing=None, origin=None, dimensions=None, matrix=None):
        self.contour_position = contour_position
        self.contour_pixel = contour_pixel
        self.spacing = spacing
        self.origin = origin
        self.dimensions = dimensions
        self.matrix = matrix

        self.mask = None
        self.mesh = None

    def create_mesh(self):
        if self.contour_pixel is not None:
            self.convert_to_pixel_spacing()

        if self.mask is not None:
            self.compute_mask()

        self.compute_mesh()

    def convert_to_pixel_spacing(self):
        sitk_image = sitk.Image([int(dim) for dim in self.dimensions], sitk.sitkUInt8)
        matrix_flat = self.image_matrix[0:3, 0:3].flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        self.contour_pixel = [[]] * len(self.contour_position)
        for ii, contours in enumerate(self.contour_position):
            self.contour_pixel[ii] = [sitk_image.TransformPhysicalPointToContinuousIndex(contour) for contour in contours]

    def compute_mask(self):
        slice_check = np.zeros(self.dimensions[2])
        hold_mask = np.zeros([self.dimensions[2], self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
        for c in self.contour_pixel:
            contour_stacked = np.vstack((c[:, 0:2], c[0, 0:2]))
            new_contour = np.array([contour_stacked], dtype=np.int32)
            image = np.zeros([self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
            cv2.fillPoly(image, new_contour, 1)

            slice_num = int(c[0, 2])
            if slice_check[slice_num] == 0:
                hold_mask[slice_num, :, :] = image
                slice_check[slice_num] = 1
            else:
                hold_mask[slice_num, :, :] = hold_mask[slice_num, :, :] + image
        self.mask = (hold_mask > 0).astype(np.uint8)

    def compute_mesh(self):
        label = numpy_support.numpy_to_vtk(num_array=np.asarray(self.mask).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        img_vtk = vtk.vtkImageData()
        img_vtk.SetDimensions([self.dimensions[1], self.dimensions[0], self.dimensions[2]])
        img_vtk.SetSpacing([self.spacing[1], self.spacing[0], self.spacing[2]])
        img_vtk.SetOrigin(self.origin)
        img_vtk.SetDirectionMatrix(self.matrix[0:3, 0:3].reshape(9))
        img_vtk.GetPointData().SetScalars(label)

        vtk_mesh = vtk.vtkDiscreteMarchingCubes()
        vtk_mesh.SetInputData(img_vtk)
        vtk_mesh.GenerateValues(1, 1, 1)
        vtk_mesh.Update()

        self.mesh = pv.PolyData(vtk_mesh.GetOutput())


class ContourToMask(object):
    def __init__(self, contour_position=None, contour_pixel=None, spacing=None, origin=None, dimensions=None, matrix=None):
        self.contour_position = contour_position
        self.contour_pixel = contour_pixel
        self.spacing = spacing
        self.origin = origin
        self.dimensions = dimensions
        self.matrix = matrix

        self.mask = None

    def create_mask(self):
        if self.contour_pixel is not None:
            self.convert_to_pixel_spacing()

        self.compute_mask()

    def convert_to_pixel_spacing(self):
        sitk_image = sitk.Image([int(dim) for dim in self.dimensions], sitk.sitkUInt8)
        matrix_flat = self.image_matrix[0:3, 0:3].flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        self.contour_pixel = [[]] * len(self.contour_position)
        for ii, contours in enumerate(self.contour_position):
            self.contour_pixel[ii] = [sitk_image.TransformPhysicalPointToContinuousIndex(contour) for contour in contours]

    def compute_mask(self):
        slice_check = np.zeros(self.dimensions[2])
        hold_mask = np.zeros([self.dimensions[2], self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
        for c in self.contour_pixel:
            contour_stacked = np.vstack((c[:, 0:2], c[0, 0:2]))
            new_contour = np.array([contour_stacked], dtype=np.int32)
            image = np.zeros([self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
            cv2.fillPoly(image, new_contour, 1)

            slice_num = int(c[0, 2])
            if slice_check[slice_num] == 0:
                hold_mask[slice_num, :, :] = image
                slice_check[slice_num] = 1
            else:
                hold_mask[slice_num, :, :] = hold_mask[slice_num, :, :] + image
        self.mask = (hold_mask > 0).astype(np.uint8)


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

        self.contours = []
        self.mask = None
        self.origin = None

        if convert:
            self.compute_bounds()
            self.compute_contours()
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

    def compute_contours(self):
        """
        Compute the contours along the z-axis for all models.
        Returns
        -------

        """
        for model in self.models:
            com = model.center
            org_bounds = model.GetBounds()

            model_contours = []
            for s in self.slice_locations:
                if org_bounds[4] < s < org_bounds[5]:
                    hold_contour = model.slice(normal='z', origin=[com[0], com[1], s])
                    model_contours.append((np.asarray(hold_contour.points)[:, 0:2] -
                                     (self.bounds[0], self.bounds[2])) / (self.spacing[0:2]))
                else:
                    model_contours.append([])

            self.contours.append(model_contours)

    def compute_mask(self):
        """
        Default is an empty array. Use the computed contours to fill the mask, not needed if the user wants a empty
        array.
        Returns
        -------

        """
        self.mask = np.zeros((self.dims[0], self.dims[2], self.dims[1]))
        if not self.empty_array:
            for ii, model in enumerate(self.models):

                model_contours = self.contours[ii]
                for jj, s in enumerate(self.slice_locations):
                    if len(model_contours[jj]) > 0:
                        frame = np.zeros((self.dims[2], self.dims[1]))
                        # noinspection PyTypeChecker
                        cv2.fillPoly(frame, np.array([model_contours[jj]], dtype=np.int32), 1)
                        self.mask[jj, :, :] = self.mask[jj, :, :] + frame

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
