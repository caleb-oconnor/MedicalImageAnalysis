"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import copy
import numpy as np

import SimpleITK as sitk

from ..conversion import ContourToDiscreteMesh


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

    def convert_position_to_pixel(self, position=None, exclude_column=None):
        matrix = np.identity(3, dtype=np.float32)
        matrix[0, :] = self.image.image_matrix[0, :] / self.image.spacing[0]
        matrix[1, :] = self.image.image_matrix[1, :] / self.image.spacing[1]
        matrix[2, :] = self.image.image_matrix[2, :] / self.image.spacing[2]

        conversion_matrix = np.identity(4, dtype=np.float32)
        conversion_matrix[:3, :3] = matrix
        conversion_matrix[:3, 3] = np.asarray(self.image.origin).dot(-matrix.T)

        pixel = []
        for ii, pos in enumerate(position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            pixel_3_axis = p_concat.dot(conversion_matrix.T)[:, :3]
            pixel_corrected = np.vstack((pixel_3_axis, pixel_3_axis[0, :]))

            if exclude_column == 2:
                pixel += [pixel_corrected[:, :2]]

            elif exclude_column == 1:
                pixel += [np.column_stack((pixel_corrected[:, 0], pixel_corrected[:, 2]))]

            elif exclude_column == 0:
                pixel += [pixel_corrected[:, 1:]]

            else:
                pixel += [pixel_3_axis]

        return pixel

    def convert_pixel_to_position(self, pixel=None):
        conversion_matrix = np.identity(4, dtype=np.float32)
        conversion_matrix[0, :3] = self.image.image_matrix[0, :] * self.image.spacing[0]
        conversion_matrix[1, :3] = self.image.image_matrix[1, :] * self.image.spacing[1]
        conversion_matrix[2, :3] = self.image.image_matrix[2, :] * self.image.spacing[2]
        conversion_matrix[:3, 3] = self.image.origin

        position = []
        for ii, pix in enumerate(pixel):
            p_concat = np.concatenate((pix, np.ones((pix.shape[0], 1))), axis=1)
            position += [p_concat.dot(conversion_matrix.T)[:, :3]]

        return position

    def create_discrete_mesh(self):
        meshing = ContourToDiscreteMesh(contour_pixel=self.contour_pixel,
                                        spacing=self.image.spacing,
                                        origin=self.image.origin,
                                        dimensions=self.image.dimensions,
                                        matrix=self.image.image_matrix)
        meshing.create_mesh()
        self.mesh = meshing.mesh
        self.display_mesh = copy.deepcopy(meshing.mesh)

    def slice_mesh(self, location=None, plane=None, normal=None, column_idx=2, return_pixel=False):
        if normal is None:
            matrix = self.image.display_matrix.T
            if self.image.plane == 'Axial':
                if plane == 'Axial':
                    column_idx = 2
                elif plane == 'Coronal':
                    column_idx = 1
                else:
                    column_idx = 0

            elif self.image.plane == 'Coronal':
                if plane == 'Axial':
                    column_idx = 1
                elif plane == 'Coronal':
                    column_idx = 2
                else:
                    column_idx = 0

            else:
                if plane == 'Axial':
                    column_idx = 1
                elif plane == 'Coronal':
                    column_idx = 0
                else:
                    column_idx = 2

            normal = matrix[:3, column_idx]

        roi_slice = self.mesh.slice(normal=normal, origin=location)

        if return_pixel:
            if roi_slice.number_of_points > 0:
                roi_strip = roi_slice.strip()
                position = [np.asarray(c.points) for c in roi_strip.cell]

                pixel = self.convert_position_to_pixel(position=position, exclude_column=column_idx)

                return pixel

            else:
                return []

        else:
            return roi_slice
