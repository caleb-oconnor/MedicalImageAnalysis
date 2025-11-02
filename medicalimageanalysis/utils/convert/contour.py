"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:

Functions:

"""

import os
import sys
import copy
import time

import cv2
import pyacvd
import numpy as np
import pyvista as pv
import SimpleITK as sitk

import vtk
from vtk.util import numpy_support

from scipy.spatial.distance import cdist
from matplotlib import path


class ContourToDiscreteMesh(object):
    """
    Converts 2D contour points into a 3D discrete or smoothed surface mesh, optionally generating a volumetric mask
    along the way. It supports multiple anatomical planes (Axial, Coronal, Sagittal).
    """
    def __init__(self, contour_position=None, contour_pixel=None, spacing=None, origin=None, dimensions=None,
                 matrix=None, plane='Axial', mask=None):
        """
        contour_position: Original contour points in physical coordinates.
        contour_pixel: Contours converted to voxel coordinates.
        spacing, origin, dimensions: Image geometry information.
        matrix: Transformation matrix for voxel conversion.
        plane: Plane along which slices are defined (Axial, Coronal, Sagittal).
        """
        self.contour_position = contour_position
        self.contour_pixel = contour_pixel
        self.spacing = spacing
        self.origin = origin
        self.dimensions = dimensions
        self.plane = plane

        self.mask = mask

        if matrix is None:
            self.matrix = np.identity(3)
        else:
            self.matrix = matrix

        if self.contour_pixel is None and self.mask is None:
            self.convert_to_pixel_spacing()

        if self.mask is None:
            self.compute_mask()

    def convert_to_pixel_spacing(self):
        """
        Converts physical contour points to voxel coordinates using a 4x4 conversion matrix.
        """
        matrix = np.identity(3, dtype=np.float32)
        matrix[0, :] = self.matrix[0, :] / self.spacing[0]
        matrix[1, :] = self.matrix[1, :] / self.spacing[1]
        matrix[2, :] = self.matrix[2, :] / self.spacing[2]

        conversion_matrix = np.identity(4, dtype=np.float32)
        conversion_matrix[:3, :3] = matrix
        conversion_matrix[:3, 3] = np.asarray(self.origin).dot(-matrix.T)

        self.contour_pixel = []
        for ii, pos in enumerate(self.contour_position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            self.contour_pixel += [p_concat.dot(conversion_matrix.T)[:, :3]]

    def compute_mask(self):
        """
        Generates a 3D binary mask slice-by-slice.
        """
        hold_mask = np.zeros([self.dimensions[0], self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
        if self.plane == 'Axial':
            slice_check = np.zeros(self.dimensions[0])
            for c in self.contour_pixel:
                contour_stacked = np.vstack((c[:, 0:2], c[0, 0:2]))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image = np.zeros([self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 2]))
                if slice_check[slice_num] == 0:
                    hold_mask[slice_num, :, :] = image
                    slice_check[slice_num] = 1
                else:
                    hold_mask[slice_num, :, :] = hold_mask[slice_num, :, :] + image
            self.mask = (hold_mask > 0).astype(np.uint8)

        elif self.plane == 'Coronal':
            slice_check = np.zeros(self.dimensions[1])
            for c in self.contour_pixel:
                stack_1 = np.vstack((c[:, 0], c[:, 2])).T
                stack_2 = np.asarray([c[0, 0], c[0, 2]])
                contour_stacked = np.vstack((stack_1, stack_2))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image = np.zeros([self.dimensions[0], self.dimensions[2]], dtype=np.uint8)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 1]))
                if slice_check[slice_num] == 0:
                    hold_mask[:, slice_num, :] = image
                    slice_check[slice_num] = 1
                else:
                    hold_mask[:, slice_num, :] = hold_mask[:, slice_num, :] + image
            self.mask = (hold_mask > 0).astype(np.uint8)

        else:
            slice_check = np.zeros(self.dimensions[2])
            for c in self.contour_pixel:
                contour_stacked = np.vstack((c[:, 1:], c[0, 1:]))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image = np.zeros([self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 0]))
                if slice_check[slice_num] == 0:
                    hold_mask[:, :, slice_num] = image
                    slice_check[slice_num] = 1
                else:
                    hold_mask[:, :, slice_num] = hold_mask[:, :, slice_num] + image
            self.mask = (hold_mask > 0).astype(np.uint8)

    def compute_mesh(self, discrete=False, smoothing_iterations=20, smoothing_relaxation=.5, smoothing_distance=1):
        """
        Converts the mask into a 3D mesh.
        """
        label = numpy_support.numpy_to_vtk(num_array=self.mask.ravel(),
                                           deep=True,
                                           array_type=vtk.VTK_FLOAT)

        img_vtk = vtk.vtkImageData()
        img_vtk.SetDimensions(self.dimensions[2], self.dimensions[1], self.dimensions[0])
        img_vtk.SetSpacing(self.spacing)
        img_vtk.SetOrigin(self.origin)
        img_vtk.SetDirectionMatrix(self.matrix.flatten(order='F'))
        img_vtk.GetPointData().SetScalars(label)

        pad_filter = vtk.vtkImageConstantPad()
        pad_filter.SetInputData(img_vtk)
        pad_filter.SetConstant(0)

        old_extent = img_vtk.GetExtent()
        new_extent = [old_extent[0] - 1, old_extent[1] + 1,
                      old_extent[2] - 1, old_extent[3] + 1,
                      old_extent[4] - 1, old_extent[5] + 1]
        pad_filter.SetOutputWholeExtent(new_extent)
        pad_filter.Update()
        pad_image = pad_filter.GetOutput()

        if discrete:
            vtk_mesh = vtk.vtkDiscreteMarchingCubes()
            vtk_mesh.SetInputData(pad_image)
            vtk_mesh.GenerateValues(1, 1, 1)
            vtk_mesh.Update()

            return pv.PolyData(vtk_mesh.GetOutput())

        else:
            # Uses VTK surface nets 3d
            img = pv.ImageData(pad_image)
            mesh = img.contour_labels(smoothing=True,
                                      output_mesh_type='triangles',
                                      smoothing_iterations=smoothing_iterations,
                                      smoothing_relaxation=smoothing_relaxation,
                                      smoothing_distance=smoothing_distance)

            return mesh


class ContourToMask(object):
    """
    Converts 2D contour points (from medical images or 3D scans) into a 3D volumetric mask, supporting multiple planes
    (Axial, Coronal, Sagittal).
    """
    def __init__(self, contour_position=None, contour_pixel=None, spacing=None, origin=None, dimensions=None,
                 matrix=None, plane='Axial'):
        """
        contour_position: Original contour points in physical coordinates.
        contour_pixel: Contours converted to pixel indices (voxel coordinates).
        spacing, origin, dimensions: Image geometry information.
        matrix: Transformation matrix to map physical coordinates to pixel space.
        plane: Axis along which slices are interpreted (Axial/Coronal/Sagittal).
        """
        self.contour_position = contour_position
        self.contour_pixel = contour_pixel
        self.spacing = spacing
        self.origin = origin
        self.dimensions = dimensions
        self.matrix = matrix
        self.plane = plane

        self.mask = None

    def create_mask(self):
        """
        Converts contours to pixel space if necessary, then computes the 3D mask.
        """
        if self.contour_pixel is None:
            self.convert_to_pixel_spacing()

        self.compute_mask()

    def convert_to_pixel_spacing(self):
        """
        Converts physical contour coordinates to voxel indices using SimpleITK.
        """
        sitk_image = sitk.Image([int(dim) for dim in self.dimensions], sitk.sitkUInt8)
        matrix_flat = self.matrix[0:3, 0:3].flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        self.contour_pixel = [[]] * len(self.contour_position)
        for ii, contours in enumerate(self.contour_position):
            self.contour_pixel[ii] = [sitk_image.TransformPhysicalPointToContinuousIndex(contour) for contour in contours]

    def compute_mask(self):
        """
        Generates a 3D mask array from the pixel contours.
        """
        hold_mask = np.zeros([self.dimensions[0], self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
        if self.plane == 'Axial':
            slice_check = np.zeros(self.dimensions[0])
            for c in self.contour_pixel:
                contour_stacked = np.vstack((c[:, 1:], c[0, 1:]))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image = np.zeros([self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 0]))
                if slice_check[slice_num] == 0:
                    hold_mask[slice_num, :, :] = image
                    slice_check[slice_num] = 1
                else:
                    hold_mask[slice_num, :, :] = hold_mask[slice_num, :, :] + image
            self.mask = (hold_mask > 0).astype(np.uint8)

        elif self.plane == 'Coronal':
            slice_check = np.zeros(self.dimensions[1])
            for c in self.contour_pixel:
                stack_1 = np.vstack((c[:, 0], c[:, 2])).T
                stack_2 = np.asarray([c[0, 0], c[0, 2]])
                contour_stacked = np.vstack((stack_1, stack_2))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image = np.zeros([self.dimensions[0], self.dimensions[2]], dtype=np.uint8)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 1]))
                if slice_check[slice_num] == 0:
                    hold_mask[:, slice_num, :] = image
                    slice_check[slice_num] = 1
                else:
                    hold_mask[:, slice_num, :] = hold_mask[:, slice_num, :] + image
            self.mask = (hold_mask > 0).astype(np.uint8)

        else:
            slice_check = np.zeros(self.dimensions[2])
            for c in self.contour_pixel:
                contour_stacked = np.vstack((c[:, 1:], c[0, 1:]))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image = np.zeros([self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 0]))
                if slice_check[slice_num] == 0:
                    hold_mask[:, :, slice_num] = image
                    slice_check[slice_num] = 1
                else:
                    hold_mask[:, :, slice_num] = hold_mask[:, :, slice_num] + image
            self.mask = (hold_mask > 0).astype(np.uint8)


class ModelToMask(object):
    """
    Converts 3D model(s) into a volumetric mask. The mask can either be:
        - an empty array (default)
        - a binary mask representing the models.

    The mask is padded with a spacing buffer of 5 units along each axis (x, y, z) to ensure the model fits comfortably.
    """
    def __init__(self, models, origin=None, spacing=None, dims=None, slice_locations=None, matrix=None,
                 empty_array=True, convert=True):
        """
        Parameters
        ----------
        models : list
            List of 3D models to convert into a mask.
        origin : list or None
            The origin of the mask in x, y, z coordinates.
        spacing : list or None
            Voxel spacing along x, y, z axes.
        dims : list or None
            Dimensions of the mask array.
        slice_locations : list or None
            Z-axis slice locations at which contours are computed.
        matrix : np.array or None
            Transformation matrix for the models. Defaults to identity if None.
        empty_array : bool
            If True, only an empty mask is created. If False, the mask is filled with model contours.
        convert : bool
            If True, automatically computes bounds, contours, and mask during initialization.
        """
        self.models = models
        self.empty_array = empty_array

        self.spacing = spacing
        self.origin = origin
        self.dims = dims
        self.slice_locations = slice_locations

        if matrix is None:
            self.matrix = np.identity(4)
        else:
            self.matrix = matrix

        self.bounds = None

        self.contours = []
        self.mask = None

        if convert:
            self.compute_bounds()
            self.compute_contours()
            self.compute_mask()

    def compute_bounds(self):
        """
        Computes the bounding box for all models combined. Adds a buffer of 5 units multiplied by the spacing on each
        axis. Sets default spacing based on model size if spacing is not provided.
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
            self.dims = [len(self.slice_locations), self.bounds[3] - self.bounds[2] + 1, self.bounds[1] - self.bounds[0] + 1]

    def compute_contours(self):
        """
        Computes 2D contours along the z-axis for each model at the pre-defined slice locations. These contours are
        later used to fill the mask.
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
        Creates a 3D mask. Default is an empty array. If empty_array=False, fills mask using computed contours.
        """
        self.mask = np.zeros((self.dims[0], self.dims[1], self.dims[2]))
        if not self.empty_array:
            for ii, model in enumerate(self.models):

                model_contours = self.contours[ii]
                for jj, s in enumerate(self.slice_locations):
                    if len(model_contours[jj]) > 0:
                        frame = np.zeros((self.dims[1], self.dims[2]))
                        cv2.fillPoly(frame, np.array([model_contours[jj]], dtype=np.int32), 1)
                        self.mask[jj, :, :] = self.mask[jj, :, :] + frame

        self.mask = self.mask.astype(np.int8)

    def save_image(self, export_path):
        """
        Saves the mask as an image file using SimpleITK.

        Parameters
        ----------
        export_path : str
            Output file path to save the mask.

        """
        image = sitk.GetImageFromArray(self.mask)
        image.SetSpacing(self.spacing)
        image.SetOrigin([self.bounds[0], self.bounds[2], self.bounds[4]])
        sitk.WriteImage(image, export_path)
