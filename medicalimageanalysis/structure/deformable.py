"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import numpy as np

from ..data import Data


class Display(object):
    def __init__(self, deformable):
        self.deformable = deformable

        self.origin = None
        self.spacing = None
        self.array = []
        self.array_splits = {'Current': 0, 'Max': 1}

        self.slice_location = [0, 0, 0]
        self.scroll_max = None
        self.offset = {'Axial': [0, 0], 'Coronal': [0, 0], 'Sagittal': [0, 0]}
        self.misc = {}

    def compute_deformation(self):
        pass

    def compute_offset(self):
        if self.deformable.reference_name is not None:
            pos = Data.image[self.deformable.reference_name].origin

            self.offset['Axial'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
            self.offset['Axial'][1] = (self.origin[1] - pos[1]) / self.spacing[1]
            self.offset['Coronal'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
            self.offset['Coronal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]
            self.offset['Sagittal'][0] = (self.origin[1] - pos[1]) / self.spacing[1]
            self.offset['Sagittal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]

    def compute_slice_location(self, position=None):
        if position is None:
            source_location = np.flip(Data.image[self.deformable.reference_name].display.slice_location)
            position = Data.image[self.deformable.reference_name].display.compute_index_positions(source_location)
        self.slice_location = np.flip(np.round((position - self.origin) / self.spacing).astype(np.int32))

    def compute_slice_origin(self, slice_plane):
        slice_origin = None
        if slice_plane == 'Axial' and 0 <= self.slice_location[0] <= self.scroll_max[0]:
            location = np.asarray([0, 0, self.slice_location[0]])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Coronal' and 0 <= self.slice_location[1] <= self.scroll_max[1]:
            location = np.asarray([0, self.slice_location[1], 0])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Sagittal' and 0 <= self.slice_location[2] <= self.scroll_max[2]:
            location = np.asarray([self.slice_location[2], 0, 0])
            slice_origin = self.origin + (location * self.spacing)

        return slice_origin

    def compute_scroll_max(self):
        self.scroll_max = [self.array.shape[0] - 1,
                           self.array.shape[1] - 1,
                           self.array.shape[2] - 1]

    def convert_position_to_pixel(self, position=None):
        position_to_pixel_matrix = Data.image[self.rigid.reference_name].display.compute_matrix_position_to_pixel()

        pixel = []
        for ii, pos in enumerate(position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            pixel_3_axis = p_concat.dot(position_to_pixel_matrix.T)[:, :3]
            pixel += [np.vstack((pixel_3_axis, pixel_3_axis[0, :]))]

        return pixel


class Deformable(object):
    def __init__(self, dvf, origin, spacing, dimensions, roi_names=None, rigid_matrix=None, dvf_matrix=None,
                 registration_name=None, reference_name=None, moving_name=None, reference_sops=None, moving_sops=None):
        self.reference_name = reference_name
        self.reference_sops = reference_sops
        self.moving_name = moving_name
        self.moving_sops = moving_sops
        self.roi_names = roi_names
        self.rois = dict.fromkeys(Data.roi_list)

        self.dvf = dvf
        self.origin = origin
        self.spacing = spacing
        self.dimensions = dimensions

        if rigid_matrix is None:
            self.rigid_matrix = np.identity(4)
        else:
            self.rigid_matrix = rigid_matrix

        if dvf_matrix is None:
            self.dvf_matrix = np.identity(4)
        else:
            self.dvf_matrix = dvf_matrix

        self.deformable_name = self.add_deformable(registration_name)

        # self.display = Display(self)

    def add_deformable(self, deformable_name):
        if deformable_name is None:
            if self.reference_name is None and self.moving_name is None:
                deformable_name = 'DVF_Unknown'
            else:
                deformable_name = 'DVF_' + self.reference_name + '_' + self.moving_name

            if deformable_name in Data.deformable_list:
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(deformable_name + '_' + str(n))
                    if new_name not in Data.deformable_list:
                        n = -100

        Data.deformable[deformable_name] = self
        Data.deformable_list += [deformable_name]

        return deformable_name
