"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import vtk
import numpy as np

import SimpleITK as sitk

from ..utils.mesh.surface import Refinement
from ..utils.conversion import ContourToDiscreteMesh


class Roi(object):
    def __init__(self, image, position=None, name=None, color=None, visible=False, filepaths=None):
        self.image = image

        self.name = name
        self.visible = visible
        self.color = color
        self.filepaths = filepaths

        if position is not None:
            self.contour_position = position
            self.contour_pixel = self.convert_position_to_pixel(position)
        else:
            self.contour_position = None
            self.contour_pixel = None

        self.mesh = None
        self.display_mesh = None

        self.volume = None
        self.com = None
        self.bounds = None

        self.rotated_mesh = None
        self.multi_color = None

    def convert_position_to_pixel(self, position=None):
        position_to_pixel_matrix = self.image.compute_matrix_position_to_pixel()

        pixel = []
        for ii, pos in enumerate(position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            pixel_3_axis = p_concat.dot(position_to_pixel_matrix.T)[:, :3]
            pixel += [np.vstack((pixel_3_axis, pixel_3_axis[0, :]))]

        return pixel

    def convert_pixel_to_position(self, pixel=None):
        pixel_to_position_matrix = self.image.compute_matrix_pixel_to_position()

        position = []
        for ii, pix in enumerate(pixel):
            p_concat = np.concatenate((pix, np.ones((pix.shape[0], 1))), axis=1)
            position += [p_concat.dot(pixel_to_position_matrix.T)[:, :3]]

        return position

    def create_discrete_mesh(self):
        meshing = ContourToDiscreteMesh(contour_pixel=self.contour_pixel,
                                        spacing=self.image.spacing,
                                        origin=self.image.origin,
                                        dimensions=self.image.dimensions,
                                        matrix=self.image.matrix)
        self.mesh = meshing.compute_mesh()
        self.volume = self.mesh.volume
        self.com = self.mesh.center
        self.bounds = self.mesh.bounds

    def create_display_mesh(self, iterations=20, angle=60, passband=0.001):
        refine = Refinement(self.mesh)
        self.display_mesh = refine.smooth(iterations=iterations, angle=angle, passband=passband)

    def create_decimate_mesh(self, percent=None, display=True):
        if display:
            refine = Refinement(self.display_mesh)
        else:
            refine = Refinement(self.mesh)
            
        return refine.decimate(percent=percent)

    def create_cluster_mesh(self, points=None, display=True):
        if display:
            refine = Refinement(self.display_mesh)
        else:
            refine = Refinement(self.mesh)

        return refine.cluster(points=points)

    def compute_contour(self, slice_location):
        contour_list = []
        if self.contour_pixel is not None:
            roi_z = [np.round(c[0, 2]).astype(int) for c in self.contour_pixel]
            keep_idx = np.argwhere(np.asarray(roi_z) == slice_location)

            if len(keep_idx) > 0:
                for ii, idx in enumerate(keep_idx):
                    contour_corrected = np.vstack((self.contour_pixel[idx[0]][:, 0:2], self.contour_pixel[idx[0]][0, 0:2]))
                    contour_corrected[:, 1] = self.image.dimensions[1] - contour_corrected[:, 1]
                    contour_list.append(contour_corrected)

        return contour_list

    def compute_mesh_slice(self, display=True, location=None, plane=None, normal=None, return_pixel=False):
        if normal is None:
            matrix = self.image.display_matrix.T
            if plane == 'Axial':
                normal = matrix[:3, 2]
            elif plane == 'Coronal':
                normal = matrix[:3, 1]
            else:
                normal = matrix[:3, 0]

        if display:
            roi_slice = self.display_mesh.slice(normal=normal, origin=location)
        else:
            roi_slice = self.mesh.slice(normal=normal, origin=location)

        if return_pixel:
            if roi_slice.number_of_points > 0:
                roi_strip = roi_slice.strip()
                position = [np.asarray(c.points) for c in roi_strip.cell]

                pixel = self.convert_position_to_pixel(position=position)
                pixel_correct = self.pixel_slice_correction(pixel, plane)

                return pixel_correct

            else:
                return []

        else:
            return roi_slice

    def pixel_slice_correction(self, pixels, plane):
        pixel_corrected = []
        for pixel in pixels:

            if plane in 'Axial':
                pixel_reshape = pixel[:, :2]
                pixel_corrected += [np.asarray([pixel_reshape[:, 0],
                                                self.image.dimensions[1] - pixel_reshape[:, 1]]).T]

            elif plane == 'Coronal':
                pixel_reshape = np.column_stack((pixel[:, 0], pixel[:, 2]))
                pixel_corrected += [pixel_reshape]

            else:
                pixel_reshape = pixel[:, 1:]
                pixel_corrected += [pixel_reshape]

        return pixel_corrected
