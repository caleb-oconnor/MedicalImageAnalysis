"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import os
import copy
import numpy as np
import pandas as pd

import vtk
from vtkmodules.util import numpy_support

import SimpleITK as sitk

from .poi import Poi
from .roi import Roi


class Image(object):
    def __init__(self):
        self.rois = {}
        self.pois = {}

        self.tags = None
        self.array = None

        self.image_name = None
        self.patient_name = None
        self.mrn = None
        self.date = None
        self.time = None
        self.series_uid = None
        self.frame_ref = None
        self.modality = None

        self.filepaths = None
        self.sops = None

        self.plane = None
        self.spacing = None
        self.dimensions = None
        self.orientation = None
        self.origin = None
        self.image_matrix = None
        self.display_matrix = np.identity(4, dtype=np.float32)
        self.window = None
        self.camera_position = None

        self.unverified = None
        self.base_position = None
        self.skipped_slice = None
        self.sections = None
        self.rgb = False

        self.slice_location = [0, 0, 0]

    def input(self, image):
        self.tags = image.image_set
        self.array = image.array

        self.patient_name = self.get_patient_name()
        self.mrn = self.get_mrn()
        self.date = self.get_date()
        self.time = self.get_time()
        self.series_uid = self.get_series_uid()
        self.frame_ref = self.get_frame_ref()
        self.window = self.get_window()

        self.filepaths = image.filepaths
        self.sops = image.sops

        self.plane = image.plane
        self.spacing = image.spacing
        self.dimensions = list(self.array.shape)
        self.orientation = image.orientation
        self.origin = image.origin
        self.image_matrix = image.image_matrix
        self.display_matrix[:3, :3] = copy.deepcopy(self.image_matrix)

        self.unverified = image.unverified
        self.base_position = image.base_position
        self.skipped_slice = image.skipped_slice
        self.sections = image.sections
        self.rgb = image.rgb

        self.modality = image.modality

        self.slice_location = [int(self.dimensions[0] / 2), int(self.dimensions[1] / 2), int(self.dimensions[2] / 2)]

    def input_rtstruct(self, rtstruct):
        for ii, roi_name in enumerate(rtstruct.roi_names):
            if roi_name not in list(self.rois.keys()):
                self.rois[roi_name] = Roi(self, position=rtstruct.contours[ii], name=roi_name,
                                          color=rtstruct.roi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

        for ii, poi_name in enumerate(rtstruct.poi_names):
            if poi_name not in list(self.pois.keys()):
                self.pois[poi_name] = Poi(self, position=rtstruct.points[ii], name=poi_name,
                                          color=rtstruct.poi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

    def add_roi(self, roi_name=None, color=None, visible=False, path=None, contour=None):
        self.rois[roi_name] = Roi(self, position=contour, name=roi_name, color=color, visible=visible, filepaths=path)

    def add_poi(self, poi_name=None, color=None, visible=False, path=None, point=None):
        self.pois[poi_name] = Poi(self, position=point, name=poi_name, color=color, visible=visible, filepaths=path)

    def get_patient_name(self):
        if 'PatientName' in self.tags[0]:
            return self.tags[0].PatientName
        else:
            return 'Name tag missing'

    def get_mrn(self):
        if 'PatientID' in self.tags[0]:
            return self.tags[0].PatientID
        else:
            return 'MRN tag missing'

    def get_date(self):
        if 'SeriesDate' in self.tags[0]:
            return self.tags[0].SeriesDate
        elif 'ContentDate' in self.tags[0]:
            return self.tags[0].ContentDate
        elif 'AcquisitionDate' in self.tags[0]:
            return self.tags[0].AcquisitionDate
        elif 'StudyDate' in self.tags[0]:
            return self.tags[0].StudyDate
        else:
            return '00000'

    def get_time(self):
        if 'SeriesTime' in self.tags[0]:
            return self.tags[0].SeriesTime
        elif 'ContentTime' in self.tags[0]:
            return self.tags[0].ContentTime
        elif 'AcquisitionTime' in self.tags[0]:
            return self.tags[0].AcquisitionTime
        elif 'StudyTime' in self.tags[0]:
            return self.tags[0].StudyTime
        else:
            return '00000'

    def get_study_uid(self):
        if 'StudyInstanceUID' in self.tags[0]:
            return self.tags[0].StudyInstanceUID
        else:
            return '00000.00000'

    def get_series_uid(self):
        if 'SeriesInstanceUID' in self.tags[0]:
            return self.tags[0].SeriesInstanceUID
        else:
            return '00000.00000'

    def get_frame_ref(self):
        if 'FrameOfReferenceUID' in self.tags[0]:
            return self.tags[0].FrameOfReferenceUID
        else:
            return '00000.00000'

    def get_window(self):
        if (0x0028, 0x1050) in self.tags[0] and (0x0028, 0x1051) in self.tags[0]:
            center = self.tags[0].WindowCenter
            width = self.tags[0].WindowWidth

            if not isinstance(center, float):
                center = center[0]

            if not isinstance(width, float):
                width = width[0]

            return [int(center) - int(np.round(width / 2)), int(center) + int(np.round(width / 2))]

        elif self.array is not None:
            return [np.min(self.array), np.max(self.array)]

        else:
            return [0, 1]

    def get_specific_tag(self, tag):
        if tag in self.tags[0]:
            return self.tags[0][tag]
        else:
            return None

    def get_specific_tag_on_all_files(self, tag):
        if tag in self.tags[0]:
            return [t[tag] for t in self.tags]
        else:
            return None

    def get_slice_location(self, plane):
        if self.plane == 'Axial':
            if plane == self.plane:
                location = self.slice_location[0]
            elif plane == 'Coronal':
                location = self.slice_location[1]
            else:
                location = self.slice_location[2]

        elif self.plane == 'Coronal':
            if plane == self.plane:
                location = self.slice_location[0]
            elif plane == 'Axial':
                location = self.slice_location[1]
            else:
                location = self.slice_location[2]

        else:
            if plane == self.plane:
                location = self.slice_location[0]
            elif plane == 'Axial':
                location = self.slice_location[1]
            else:
                location = self.slice_location[2]

        return location

    def save_image(self, path, rois=True, pois=True):
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['rois', 'pois', 'tags', 'array']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'tags.npy'), self.tags, allow_pickle=True)
        np.save(os.path.join(path, 'array.npy'), self.array, allow_pickle=True)

        if rois:
            self.save_rois(path, create_main_folder=True)

        if pois:
            self.save_pois(path, create_main_folder=True)

    def save_rois(self, path, create_main_folder=False):
        if create_main_folder:
            path = os.path.join(path, 'ROIs')
            os.mkdir(path)

        for name in list(self.rois.keys()):
            roi_path = os.path.join(os.path.join(path, name))
            os.mkdir(roi_path)

            np.save(os.path.join(roi_path, 'name.npy'), self.rois[name].name, allow_pickle=True)
            np.save(os.path.join(roi_path, 'visible.npy'), self.rois[name].visible, allow_pickle=True)
            np.save(os.path.join(roi_path, 'color.npy'), self.rois[name].color, allow_pickle=True)
            np.save(os.path.join(roi_path, 'filepaths.npy'), self.rois[name].filepaths, allow_pickle=True)
            if self.rois[name].contour_position is not None:
                np.save(os.path.join(roi_path, 'contour_position.npy'),
                        np.array(self.rois[name].contour_position, dtype=object),
                        allow_pickle=True)

    def save_pois(self, path, create_main_folder=False):
        if create_main_folder:
            path = os.path.join(path, 'POIs')
            os.mkdir(path)

        for name in list(self.pois.keys()):
            poi_path = os.path.join(os.path.join(path, name))
            os.mkdir(poi_path)

            np.save(os.path.join(poi_path, 'name.npy'), self.pois[name].name, allow_pickle=True)
            np.save(os.path.join(poi_path, 'visible.npy'), self.pois[name].visible, allow_pickle=True)
            np.save(os.path.join(poi_path, 'color.npy'), self.pois[name].color, allow_pickle=True)
            np.save(os.path.join(poi_path, 'filepaths.npy'), self.pois[name].filepaths, allow_pickle=True)
            np.save(os.path.join(poi_path, 'point_position.npy'), self.pois[name].point_position, allow_pickle=True)

    def load_image(self, image_path, rois=True, pois=True):

        self.array = np.load(os.path.join(image_path, 'array.npy'), allow_pickle=True)
        self.tags = np.load(os.path.join(image_path, 'tags.npy'), allow_pickle=True)
        info = pd.read_pickle(os.path.join(image_path, 'info.p'),)
        for column in list(info.columns):
            setattr(self, column, info.at[0, column])

        if rois:
            roi_names = os.listdir(os.path.join(image_path, 'ROIs'))
            for name in roi_names:
                self.load_rois(os.path.join(image_path, 'ROIs', name))

        if pois:
            roi_names = os.listdir(os.path.join(image_path, 'POIs'))
            for name in roi_names:
                self.load_pois(os.path.join(image_path, 'POIs', name))

    def load_rois(self, roi_path):
        name = str(np.load(os.path.join(roi_path, 'name.npy'), allow_pickle=True))

        existing_rois = list(self.rois.keys())
        if name in existing_rois:
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_rois:
                    name = new_name
                    n = -1

        self.rois[name] = Roi(self)
        self.rois[name].name = name
        self.rois[name].visible = bool(np.load(os.path.join(roi_path, 'visible.npy'), allow_pickle=True))
        self.rois[name].color = list(np.load(os.path.join(roi_path, 'color.npy'), allow_pickle=True))
        self.rois[name].filepaths = str(np.load(os.path.join(roi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(roi_path, 'contour_position.npy')):
            self.rois[name].contour_position = list(np.load(os.path.join(roi_path, 'contour_position.npy'),
                                                            allow_pickle=True))

    def load_pois(self, poi_path):
        name = str(np.load(os.path.join(poi_path, 'name.npy'), allow_pickle=True))

        existing_pois = list(self.pois.keys())
        if name in existing_pois:
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_pois:
                    name = new_name
                    n = -1

        self.pois[name] = poi(self)
        self.pois[name].name = name
        self.pois[name].visible = bool(np.load(os.path.join(poi_path, 'visible.npy'), allow_pickle=True))
        self.pois[name].color = list(np.load(os.path.join(poi_path, 'color.npy'), allow_pickle=True))
        self.pois[name].filepaths = str(np.load(os.path.join(poi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(poi_path, 'point_position.npy')):
            self.rois[name].contour_position = list(np.load(os.path.join(poi_path, 'point_position.npy'),
                                                            allow_pickle=True))

    def create_sitk_image(self, empty=False):
        if empty:
            sitk_image = sitk.Image([int(dim) for dim in reversed(self.dimensions)], sitk.sitkUInt8)
        else:
            sitk_image = sitk.GetImageFromArray(self.array)

        matrix_flat = self.image_matrix.flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        return sitk_image

    def array_slice_plane(self, slice_plane='Axial'):
        """
        Axial plane:
        -	Array order [a, c, s]
        -	Axial flipped on x-axis, Inferior to Superior
        -	Coronal, Anterior to Posterior
        -	Sagittal, Left to Right
        
        Coronal plane: 
        -	Array order [c, a, s]
        -	Coronal flipped on x-axis, Anterior to Posterior
        -	Axial flipped on x-axis, Inferior to Superior
        -	Sagittal transposed flipped on y-axis, Left to Right
        
        Sagittal plane:
        -	Array order [s, a, c]
        -	Sagittal flipped on x-axis, Left to Right
        -	Axial transpose, flipped on x-axis, flipped on y-axis, Superior to Inferior
        -	Coronal transpose, flipped on x-axis, flipped on y-axis, Anterior to Posterior

        """
        if self.plane == 'Axial':
            if slice_plane == 'Axial':
                array = np.flip(self.array[self.slice_location[0], :, ], 0)
            elif slice_plane == 'Coronal':
                array = self.array[:, self.slice_location[1], :]
            else:
                array = self.array[:, :, self.slice_location[2]]

        elif self.plane == 'Coronal':
            if slice_plane == 'Coronal':
                array = np.flip(self.array[self.slice_location[0], :, ], 0)
            elif slice_plane == 'Axial':
                array = np.flip(self.array[:, self.slice_location[1], :], 0)
            else:
                array = np.flip(self.array[:, :, self.slice_location[2]].T, 0)

        else:
            if slice_plane == 'Sagittal':
                array = np.flip(self.array[self.slice_location[0], :, ], 0)
            elif slice_plane == 'Axial':
                array = np.flip(np.flip(self.array[:, self.slice_location[1], :].T, 0), 1)
            else:
                array = np.flip(np.flip(self.array[:, :, self.slice_location[2]].T, 0), 1)

        return array

    def compute_aspect(self, plane):
        if self.plane == 'Axial':
            if plane == self.plane:
                aspect = np.round(self.spacing[0] / self.spacing[1], 2)
            elif plane == 'Coronal':
                aspect = np.round(self.spacing[0] / self.spacing[2], 2)
            else:
                aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        elif self.plane == 'Coronal':
            if plane == self.plane:
                aspect = np.round(self.spacing[0] / self.spacing[1], 2)
            elif plane == 'Axial':
                aspect = np.round(self.spacing[0] / self.spacing[2], 2)
            else:
                aspect = np.round(self.spacing[2] / self.spacing[1], 2)

        else:
            if plane == self.plane:
                aspect = np.round(self.spacing[0] / self.spacing[1], 2)
            elif plane == 'Axial':
                aspect = np.round(self.spacing[2] / self.spacing[0], 2)
            else:
                aspect = np.round(self.spacing[2] / self.spacing[1], 2)

        return aspect

    def compute_scroll_max(self, plane):
        if self.plane == 'Axial':
            if plane == self.plane:
                scroll_max = self.dimensions[0] - 1
            elif plane == 'Coronal':
                scroll_max = self.dimensions[1] - 1
            else:
                scroll_max = self.dimensions[2] - 1

        elif self.plane == 'Coronal':
            if plane == self.plane:
                scroll_max = self.dimensions[0] - 1
            elif plane == 'Axial':
                scroll_max = self.dimensions[1] - 1
            else:
                scroll_max = self.dimensions[2] - 1

        else:
            if plane == self.plane:
                scroll_max = self.dimensions[0] - 1
            elif plane == 'Axial':
                scroll_max = self.dimensions[1] - 1
            else:
                scroll_max = self.dimensions[2] - 1

        return scroll_max

    def update_slice_location(self, location, plane):
        if self.plane == 'Axial':
            if plane == self.plane:
                self.slice_location[0] = location
            elif plane == 'Coronal':
                self.slice_location[1] = location
            else:
                self.slice_location[2] = location

        elif self.plane == 'Coronal':
            if plane == self.plane:
                self.slice_location[0] = location
            elif plane == 'Axial':
                self.slice_location[1] = location
            else:
                self.slice_location[2] = location

        else:
            if plane == self.plane:
                self.slice_location[0] = location
            elif plane == 'Axial':
                self.slice_location[1] = location
            else:
                self.slice_location[2] = location

        return location
