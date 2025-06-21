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
import copy
import time
import gdcm
import threading

import vtk
import numpy as np
import pandas as pd
import pydicom as dicom
from pydicom.uid import generate_uid

from ..structure.image import Image

from ..data import Data


def sort_images_by_datetime():
    date_time = [str(Data.images[name].date) + str(Data.images[name].time) for name in Data.image_list]
    new_key_order = [Data.image_list[idx] for idx in np.argsort(date_time)]

    Data.images = {key: Data.images[key] for key in new_key_order}
    Data.image_list = list(Data.images.keys())


def thread_process_dicom(path, stop_before_pixels=False):
    try:
        datasets = dicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)
    except:
        datasets = []

    return datasets


class DicomReader(object):
    def __init__(self, reader):
        """
        Takes in reader parent, which will be used to add to image list variable.

        :param reader:
        :type reader: object
        """
        self.reader = reader

        self.ds = []
        self.ds_modality = {key: [] for key in ['CT', 'MR', 'PT', 'US', 'DX', 'RF', 'CR', 'RTSTRUCT']}

    def load(self, display_time=False):
        """
        Reads in the dicom files, separates the images by modality, lasty adds each image to the reader image list
        variable.

        :param display_time: prints the total read in time in seconds
        :type display_time: bool
        :return:
        :rtype:
        """
        t1 = time.time()
        self.read()
        self.separate_modalities_and_images()
        self.image_creation()
        sort_images_by_datetime()
        t2 = time.time()

        if display_time:
            print('Dicom Read Time: ', t2 - t1)

    def read(self):
        """
        Reads in the dicom files using a threading process, and the user input "only_tags" determines if only the tags
        are loaded or the tags and array.

        """
        threads = []

        def read_file_thread(file_path):
            self.ds.append(thread_process_dicom(file_path, stop_before_pixels=self.reader.only_tags))

        for file_path in self.reader.files['Dicom']:
            thread = threading.Thread(target=read_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def separate_modalities_and_images(self):
        """
        The files are first sorted by Modality with these options:
            CT, MR, PT, US, DX, MG, NM, XA, CR, RTSTRUCT, REG, RTDose

        Then the files are separated into images using the SeriesInstanceUID and AcquisitionNumber. The image
        orientation and image position is used to determine how the slices are sorted incase they are read in out of
        order. However, for 2d images or non image files (US, DX, MG, XA, CR, RTSTRUCT, REG, RTDose), sorting is not
        required.
        Returns
        -------

        """
        for modality in list(self.ds_modality.keys()):
            images_in_modality = [d for d in self.ds if (0x0008, 0x0060) in d and d['Modality'].value == modality]
            if len(images_in_modality) > 0 and modality in self.reader.only_modality:
                if modality in ['US', 'DX', 'RF', 'CR', 'RTSTRUCT']:
                    for image in images_in_modality:
                        self.ds_modality[modality] += [image]

                elif modality == 'RTSTRUCT':
                    for image in images_in_modality:
                        if 'StructureSetROISequence' in image and 'ROIContourSequence' in image:
                            self.ds_modality[modality] += [image]

                else:
                    sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['AcquisitionNumber'].value] if
                                               'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None
                                               else [img['SeriesInstanceUID'].value, 1] for img in images_in_modality])

                    unique_tags = np.unique(sorting_tags, axis=0)
                    for tag in unique_tags:
                        sorted_idx = np.where((sorting_tags[:, 0] == tag[0]) & (sorting_tags[:, 1] == tag[1]))
                        image_tags = [images_in_modality[idx] for idx in sorted_idx[0] if 'ImageOrientationPatient' in images_in_modality[idx] and 'ImagePositionPatient' in images_in_modality[idx]]

                        orientations = np.asarray([img['ImageOrientationPatient'].value for img in image_tags])
                        _, indices = np.unique(np.round(orientations, 3), axis=0, return_index=True)
                        unique_orientations = [orientations[ind] for ind in indices]
                        for orient in unique_orientations:
                            orient_idx = np.where((np.round(orientations[:, 0], 3) == np.round(orient[0], 3)) &
                                                  (np.round(orientations[:, 1], 3) == np.round(orient[1], 3)) &
                                                  (np.round(orientations[:, 2], 3) == np.round(orient[2], 3)) &
                                                  (np.round(orientations[:, 3], 3) == np.round(orient[3], 3)) &
                                                  (np.round(orientations[:, 4], 3) == np.round(orient[4], 3)) &
                                                  (np.round(orientations[:, 5], 3) == np.round(orient[5], 3)))

                            orient_tags = [image_tags[idx] for idx in orient_idx[0]]
                            correct_orientation = orient_tags[0]['ImageOrientationPatient'].value

                            position_tags = np.asarray([t['ImagePositionPatient'].value for t in orient_tags])

                            x = np.abs(correct_orientation[0]) + np.abs(correct_orientation[3])
                            y = np.abs(correct_orientation[1]) + np.abs(correct_orientation[4])
                            z = np.abs(correct_orientation[2]) + np.abs(correct_orientation[5])

                            row_direction = correct_orientation[:3]
                            column_direction = correct_orientation[3:]
                            slice_direction = np.cross(row_direction, column_direction)
                            if x < y and x < z:
                                if slice_direction[0] > 0:
                                    slice_idx = np.argsort(position_tags[:, 0])
                                else:
                                    slice_idx = np.argsort(position_tags[:, 0])[::-1]
                            elif y < x and y < z:
                                if slice_direction[1] > 0:
                                    slice_idx = np.argsort(position_tags[:, 1])
                                else:
                                    slice_idx = np.argsort(position_tags[:, 1])[::-1]
                            else:
                                if slice_direction[2] > 0:
                                    slice_idx = np.argsort(position_tags[:, 2])
                                else:
                                    slice_idx = np.argsort(position_tags[:, 2])[::-1]

                            self.ds_modality[modality] += [[orient_tags[idx] for idx in slice_idx]]

    def image_creation(self):
        """
        Currently only reading in 5 modalities (CT, MR, DX, US, RTSTRUCT) and using specific modality class readers.
        First the image volume modalities are created, then RTSTRUCT is added to the image that it associates with.

        :return:
        :rtype:
        """
        for modality in ['CT', 'MR', 'PT', 'DX', 'RF', 'CR', 'US']:
            for image_set in self.ds_modality[modality]:
                if modality in ['CT', 'MR', 'PT']:
                    Read3D(image_set, self.reader.only_tags)

                elif modality in ['DX', 'CR']:
                    ReadXRay(image_set, self.reader.only_tags)

                elif modality == 'RF':
                    ReadRF(image_set, self.reader.only_tags)

                elif modality == 'US':
                    ReadUS(image_set, self.reader.only_tags)

        for modality in ['RTSTRUCT']:
            for image_set in self.ds_modality[modality]:
                if modality == 'RTSTRUCT':
                    read_rtstruct = ReadRTStruct(image_set, self.reader.only_tags)
                    if read_rtstruct.match_image_name is not None:
                        Data.images[read_rtstruct.match_image_name].input_rtstruct(read_rtstruct)
                    else:
                        print('dicom: rtstruct has no matching image')


class Read3D(object):
    """
    This is currently for CT and MR modalities.
    """
    def __init__(self, image_set, only_tags):
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.unverified = None
        self.base_position = None
        self.skipped_slice = None
        self.sections = None
        self.rgb = False

        self.modality = self.image_set[0].Modality
        self.filepaths = [image.filename for image in self.image_set]
        self.sops = [image.SOPInstanceUID for image in self.image_set]

        self.array = None
        if not self.only_tags:
            self._compute_array()

        self.orientation = self._compute_orientation()
        self.plane = self._compute_plane()
        self.spacing = self._compute_spacing()
        self.dimensions = self._compute_dimensions()
        self._verify_axial_orientation()

        self.image_matrix = self._compute_image_matrix()
        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.images[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_array(self):
        """
        Combines all the slice arrays into a 3D array.
        :return:
        :rtype:
        """

        image_slices = []
        for _slice in self.image_set:
            if (0x0028, 0x1052) in _slice:
                intercept = _slice.RescaleIntercept
            else:
                intercept = 0

            if (0x0028, 0x1053) in _slice:
                slope = _slice.RescaleSlope
            else:
                slope = 1

            image_slices.append(((_slice.pixel_array*slope)+intercept).astype('int16'))

            del _slice.PixelData

        self.array = np.asarray(image_slices)

    def _compute_orientation(self):
        """
        Looks in the tags for image orientation, typically exist in ImageOrientationPatient.
        :return:
        :rtype:
        """
        orientation = np.asarray([1, 0, 0, 0, 1, 0])
        if 'ImageOrientationPatient' in self.image_set[0]:
            orientation = np.asarray(self.image_set[0]['ImageOrientationPatient'].value)

        else:
            if 'SharedFunctionalGroupsSequence' in self.image_set[0]:
                seq_str = 'SharedFunctionalGroupsSequence'
                if 'PlaneOrientationSequence' in self.image_set[0][0][seq_str][0]:
                    plane_str = 'PlaneOrientationSequence'
                    image_str = 'ImageOrientationPatient'
                    orientation = np.asarray(self.image_set[0][0][seq_str][0][plane_str][0][image_str].value)

                else:
                    self.unverified = 'Orientation'

            else:
                self.unverified = 'Orientation'

        return orientation

    def _compute_plane(self):
        """
        Computes the image plane for the slices
        :return:
        :rtype:
        """
        x = np.abs(self.orientation[0]) + np.abs(self.orientation[3])
        y = np.abs(self.orientation[1]) + np.abs(self.orientation[4])
        z = np.abs(self.orientation[2]) + np.abs(self.orientation[5])

        if x < y and x < z:
            return 'Sagittal'
        elif y < x and y < z:
            return 'Coronal'
        else:
            return 'Axial'

    def _compute_spacing(self):
        """
        Creates 3 axis spacing by inplane pixel spacing the slice thickness
        :return:
        :rtype:
        """
        inplane_spacing = [1, 1]
        slice_thickness = np.double(self.image_set[0].SliceThickness)

        if 'PixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].PixelSpacing

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set[0]:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        if len(self.image_set) > 1:
            row_direction = self.orientation[:3]
            column_direction = self.orientation[3:]
            slice_direction = np.cross(row_direction, column_direction)

            first = np.dot(slice_direction, self.image_set[0].ImagePositionPatient)
            second = np.dot(slice_direction, self.image_set[1].ImagePositionPatient)
            last = np.dot(slice_direction, self.image_set[-1].ImagePositionPatient)
            first_last_spacing = np.asarray((last - first) / (len(self.image_set) - 1))
            if np.abs((second - first) - first_last_spacing) > 0.01:
                if not self.only_tags:
                    self._find_skipped_slices(slice_direction)
                slice_thickness = second - first
            else:
                slice_thickness = np.asarray((last - first) / (len(self.image_set) - 1))

        if self.plane == 'Axial':
            return np.asarray([inplane_spacing[1], inplane_spacing[0], slice_thickness])

        elif self.plane == 'Coronal':
            return np.asarray([inplane_spacing[1], slice_thickness, inplane_spacing[0]])

        else:
            return np.asarray([slice_thickness, inplane_spacing[1], inplane_spacing[0]])

    def _compute_dimensions(self):
        """
        Creates dimensions by columns, rows, and number of slices.
        :return:
        :rtype:
        """
        shape = self.array.shape
        if self.plane == 'Axial':
            return np.asarray([shape[0], shape[1], shape[2]])

        elif self.plane == 'Coronal':
            return np.asarray([shape[1], shape[0], shape[2]])

        else:
            return np.asarray([shape[1], shape[2], shape[0]])

    def _compute_image_matrix(self):
        """
        Computes the image matrix using the image orientation.
        :return:
        :rtype:
        """
        row_direction = self.orientation[:3]
        column_direction = self.orientation[3:]
        slice_direction = np.cross(row_direction, column_direction)

        mat = np.identity(3, dtype=np.float32)
        mat[0, :3] = row_direction
        mat[1, :3] = column_direction
        mat[2, :3] = slice_direction

        return mat

    def _verify_axial_orientation(self):
        shape = self.array.shape
        if self.plane == 'Axial':
            spacing = self.spacing
        elif self.plane == 'Coronal':
            spacing = [self.spacing[0], self.spacing[2], self.spacing[1]]
        else:
            spacing = [self.spacing[1], self.spacing[2], self.spacing[0]]

        slices = shape[0] - 1
        y = shape[1] - 1
        x = shape[2] - 1

        origin = np.asarray(self.image_set[0]['ImagePositionPatient'].value)

        row_direction = self.orientation[:3]
        column_direction = self.orientation[3:]
        slice_direction = np.cross(row_direction, column_direction)

        corners = np.zeros((8, 3))
        corners[0] = origin
        corners[1] = origin + (x * spacing[0] * row_direction)
        corners[2] = origin + (y * spacing[1] * column_direction)
        corners[3] = (origin + (x * spacing[0] * row_direction) + (y * spacing[1] * column_direction))

        corners[4] = origin + (slices * spacing[2] * slice_direction)
        corners[5] = (origin + (slices * spacing[2] * slice_direction) + (x * spacing[0] * row_direction))
        corners[6] = (origin + (slices * spacing[2] * slice_direction) + (y * spacing[1] * column_direction))
        corners[7] = (origin + (slices * spacing[2] * slice_direction) + (x * spacing[0] * row_direction) +
                      (y * spacing[1] * column_direction))

        corner_idx = np.argmin(np.sum(corners, axis=1))
        if corner_idx != 0:
            self.origin = corners[corner_idx]
            if self.plane == "Axial":
                if corner_idx == 1:
                    self.array = np.rot90(self.array, 1, (1, 2))
                elif corner_idx == 2:
                    self.array = np.rot90(self.array, 3, (1, 2))
                else:
                    self.array = np.rot90(self.array, 2, (1, 2))

                if corner_idx < 4:
                    square = corners[:4, :]
                else:
                    square = corners[4:, :]

            elif self.plane == 'Coronal':
                self.array = np.rot90(self.array, 1, (0, 1))

                s1 = np.argsort(corners[:4, 2])
                s2 = np.argsort(corners[4:, 2]) + 4

                square = [corners[s1[0]], corners[s1[1]], corners[s2[0]], corners[s2[1]]]

            else:
                self.array = np.flip(np.rot90(self.array, 1, (0, 1)).transpose(0, 2, 1), axis=2)

                s1 = np.argsort(corners[:4, 2])
                s2 = np.argsort(corners[4:, 2]) + 4

                square = [corners[s1[0]], corners[s1[1]], corners[s2[0]], corners[s2[1]]]

            distances = np.asarray([np.linalg.norm(corners[corner_idx, :] - s) for s in square])
            sorted_args = np.argsort(distances)

            c1 = square[sorted_args[1]] - corners[corner_idx]
            c2 = square[sorted_args[2]] - corners[corner_idx]

            if np.abs(c1[0]) > np.abs(c2[0]):
                # self.orientation[:3] = c1 / self.spacing / np.flip(self.dimensions - 1)
                # self.orientation[3:] = c2 / self.spacing / np.flip(self.dimensions - 1)
                # self.orientation[:3] = c1 * self.spacing / np.linalg.norm(c1 * self.spacing)
                # self.orientation[3:] = c2 * self.spacing / np.linalg.norm(c2 * self.spacing)
                self.orientation[:3] = c1 / (self.spacing[0] * self.dimensions[2])
                self.orientation[3:] = c2 / (self.spacing[1] * self.dimensions[1])
            else:
                # self.orientation[:3] = c2 / self.spacing / np.flip(self.dimensions - 1)
                # self.orientation[3:] = c1 / self.spacing / np.flip(self.dimensions - 1)
                # self.orientation[:3] = c2 * self.spacing / np.linalg.norm(c2 * self.spacing)
                # self.orientation[3:] = c1 * self.spacing / np.linalg.norm(c1 * self.spacing)
                self.orientation[:3] = c2 / (self.spacing[0] * self.dimensions[2])
                self.orientation[3:] = c1 / (self.spacing[1] * self.dimensions[1])

        else:
            self.origin = origin

    def _find_skipped_slices(self, slice_direction):
        base_spacing = None
        for ii in range(len(self.image_set) - 1):
            position_1 = np.dot(slice_direction, self.image_set[ii].ImagePositionPatient)
            position_2 = np.dot(slice_direction, self.image_set[ii + 1].ImagePositionPatient)
            if ii == 0:
                base_spacing = position_2 - position_1
            if ii > 0 and np.abs(base_spacing - (position_2 - position_1)) > 0.01:
                self.unverified = 'Skipped'
                self.skipped_slice = ii + 1


class ReadXRay(object):
    """
    This is X-ray images, modalities are DX or MG (mammograms). Mammograms can also be tomosynthesis which are not read
    in this class.
    """
    def __init__(self, image_set, only_tags):
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.unverified = 'Modality'
        self.skipped_slice = None
        self.sections = None
        self.rgb = False
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.asarray([0, 0, 0])
        self.image_matrix = np.identity(3, dtype=np.float32)

        self.modality = self.image_set[0].Modality

        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID

        self.plane = self._compute_plane()
        self.dimensions = self._compute_dimensions()
        self.spacing = self._compute_spacing()

        self.array = None
        if not self.only_tags:
            self._compute_array()

        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.images[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_plane(self):
        if 'PatientOrientation' in self.image_set[0]:
            orient = self.image_set[0].PatientOrientation
            if 'L' in orient or 'R' in orient:
                return 'Coronal'

            elif 'A' in orient or 'P' in orient:
                return 'Sagittal'

            else:
                return 'Axial'

        else:
            return 'Axial'

    def _compute_dimensions(self):
        if self.plane == 'Axial':
            return np.asarray([self.image_set[0]['Columns'].value, self.image_set[0]['Rows'].value, 1])

        elif self.plane == 'Coronal':
            return np.asarray([self.image_set[0]['Columns'].value, 1, self.image_set[0]['Rows'].value])

        else:
            return np.asarray([1, self.image_set[0]['Columns'].value, self.image_set[0]['Rows'].value])

    def _compute_spacing(self):
        """
        Creates 3 axis spacing by inplane pixel spacing the 1 mm being the slice thickness even though 2D images don't
        have thickness.

        :return:
        :rtype:
        """
        inplane_spacing = [1, 1]
        slice_thickness = 1

        if 'PixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].PixelSpacing

        elif 'ImagerPixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].ImagerPixelSpacing

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set[0]:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        if self.plane == 'Axial':
            return np.asarray([inplane_spacing[1], inplane_spacing[0], slice_thickness])

        elif self.plane == 'Coronal':
            return np.asarray([inplane_spacing[1], slice_thickness, inplane_spacing[0]])

        else:
            return np.asarray([slice_thickness, inplane_spacing[1], inplane_spacing[0]])

    def _compute_array(self):
        """
        Creates the image array.
        :return:
        :rtype:
        """
        self.array = self.image_set[0].pixel_array.astype('int16')
        del self.image_set[0].PixelData

        if 'PresentationLUTShape' in self.image_set[0] and self.image_set[0]['PresentationLUTShape'] == 'Inverse':
            self.array = 16383 - self.array

        if self.plane == 'Axial':
            self.array = self.array.reshape((1, self.array.shape[0], self.array.shape[1]))
        elif self.plane == 'Coronal':
            self.array = self.array.reshape((self.array.shape[0], 1, self.array.shape[1]))
        else:
            self.array = self.array.reshape((self.array.shape[0], self.array.shape[1], 1))


class ReadRF(object):
    def __init__(self, image_set, only_tags):
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.unverified = 'Modality'
        self.skipped_slice = None
        self.sections = None
        self.rgb = False
        self.dimensions = None

        self.modality = self.image_set[0].Modality

        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.asarray([0, 0, 0])
        self.image_matrix = np.identity(3, dtype=np.float32)

        self.plane = self._compute_plane()

        self.array = None
        if not self.only_tags:
            self._compute_array()
        self.spacing = self._compute_spacing()

        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.images[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_plane(self):
        if 'PatientOrientation' in self.image_set[0]:
            orient = self.image_set[0].PatientOrientation
            if 'L' in orient or 'R' in orient:
                return 'Coronal'

            elif 'A' in orient or 'P' in orient:
                return 'Sagittal'

            else:
                return 'Axial'

        else:
            return 'Axial'

    def _compute_array(self):
        """
        Creates the image array.
        :return:
        :rtype:
        """
        self.array = self.image_set[0].pixel_array.astype('int16')
        del self.image_set[0].PixelData

        if len(self.array.shape) < 3:
            if self.plane == 'Axial':
                self.array = self.array.reshape((self.array.shape[2], self.array.shape[0], self.array.shape[1]))
            elif self.plane == 'Coronal':
                self.array = self.array.reshape((self.array.shape[0], self.array.shape[2], self.array.shape[1]))
            else:
                self.array = self.array.reshape((self.array.shape[0], self.array.shape[1], self.array.shape[2]))

        self.dimensions = self.array.shape

    def _compute_spacing(self):
        """
        Creates 3 axis spacing by inplane pixel spacing the 1 mm being the slice thickness even though 2D images don't
        have thickness.

        :return:
        :rtype:
        """
        inplane_spacing = [1, 1]
        slice_thickness = 1

        if 'PixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].PixelSpacing

        elif 'ImagerPixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].ImagerPixelSpacing

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set[0]:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        if self.plane == 'Axial':
            return np.asarray([inplane_spacing[1], inplane_spacing[0], slice_thickness])

        elif self.plane == 'Coronal':
            return np.asarray([inplane_spacing[1], slice_thickness, inplane_spacing[0]])

        else:
            return np.asarray([slice_thickness, inplane_spacing[1], inplane_spacing[0]])


class ReadUS(object):
    """
    This is Ultrasound images, modality is US. Similar to DX modality, except US can have stacks of "slices". Not slices
    in the traditional because they don't correlate to one another.
    """
    def __init__(self, image_set, only_tags):
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.unverified = 'Modality'
        self.base_position = None
        self.skipped_slice = None
        self.sections = None
        self.rgb = False

        self.modality = self.image_set[0].Modality

        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID
        self.plane = 'Axial'
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.asarray([0, 0, 0])
        self.image_matrix = np.identity(3, dtype=np.float32)
        self.dimensions = np.asarray([self.image_set[0]['Columns'].value, self.image_set[0]['Rows'].value, 1])

        self.array = None
        if not self.only_tags:
            self._compute_array()
        self.spacing = self._compute_spacing()

        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.images[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_array(self):
        us_data = np.asarray(self.image_set[0].pixel_array)
        del self.image_set[0].PixelData

        if len(us_data.shape) == 2:
            us_data = us_data.reshape((1, us_data.shape[0], us_data.shape[1]))

        if len(us_data.shape) == 3:
            us_binary = (1 * (np.std(us_data, axis=2) == 0) == 1)
            self.array = (us_binary * us_data[:, :, 0]).astype('uint8')

        else:
            us_binary = (1 * (np.std(us_data, axis=3) == 0) == 1)
            self.array = (us_binary * us_data[:, :, :, 0]).astype('uint8')

        if len(self.array.shape) > 3:
            self.dimensions[2] = self.array.shape[0]

    def _compute_spacing(self):
        inplane_spacing = [1, 1]
        slice_thickness = 1

        if 'PixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].PixelSpacing

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set[0]:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        elif 'SequenceOfUltrasoundRegions' in self.image_set[0]:
            if 'PhysicalDeltaX' in self.image_set[0].SequenceOfUltrasoundRegions[0]:
                inplane_spacing = [10 * np.round(self.image_set[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaY, 4),
                                   10 * np.round(self.image_set[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaX, 4)]

        return np.asarray([inplane_spacing[1], inplane_spacing[0], slice_thickness])


class ReadRTStruct(object):
    def __init__(self, image_set, only_tags):
        self.image_set = image_set
        self.only_tags = only_tags

        self.series_uid = self._get_series_uid()
        self.filepaths = self.image_set.filename

        self._properties = self._get_properties()
        self.roi_names = [prop[1] for prop in self._properties if prop[3].lower() == 'closed_planar']
        self.roi_colors = [prop[2] for prop in self._properties if prop[3].lower() == 'closed_planar']
        self.poi_names = [prop[1] for prop in self._properties if prop[3].lower() == 'point']
        self.poi_colors = [prop[2] for prop in self._properties if prop[3].lower() == 'point']

        if len(self.roi_names) > 0 or len(self.poi_names) > 0:
            self.match_image_name = self._match_with_image()

            self.contours = []
            self.points = []
            if not self.only_tags:
                self._structure_positions()
        else:
            self.match_image_name = None

    def _get_series_uid(self):
        study = 'RTReferencedStudySequence'
        series = 'RTReferencedSeriesSequence'
        ref = self.image_set.ReferencedFrameOfReferenceSequence

        return ref[0][study][0][series][0]['SeriesInstanceUID'].value

    def _get_properties(self):
        names = [s.ROIName for s in self.image_set.StructureSetROISequence]
        colors = [s.ROIDisplayColor for s in self.image_set.ROIContourSequence]
        geometric = [s['ContourSequence'][0]['ContourGeometricType'].value for s in self.image_set.ROIContourSequence]

        sop = []
        for ii, s in enumerate(self.image_set.ROIContourSequence):
            slice_sop = []
            if geometric[ii].lower() == 'closed_planar':
                for seq in s['ContourSequence']:
                    slice_sop += [seq['ContourImageSequence'][0]['ReferencedSOPInstanceUID'].value]
            sop += [slice_sop]

        properties = []
        for ii in range(len(names)):
            properties += [[ii, names[ii], colors[ii], geometric[ii], sop[ii]]]

        return properties

    def _match_with_image(self):
        match_image_name = None

        for image_name in Data.images:
            if self.series_uid == Data.images[image_name].series_uid:
                if self._properties[0][4][0] in Data.images[image_name].sops:
                    match_image_name = image_name

        return match_image_name

    def _structure_positions(self):
        sequences = self.image_set.ROIContourSequence
        for prop in self._properties:
            if prop[3].lower() == 'closed_planar':
                seq = sequences[prop[0]]

                contour_list = []
                for c in seq.ContourSequence:
                    contour_hold = np.round(np.array(c['ContourData'].value), 3)
                    contour = contour_hold.reshape(int(len(contour_hold) / 3), 3)
                    contour_list.append(contour)

                self.contours += [contour_list]

            else:
                seq = sequences[prop[0]]

                contour_list = []
                for c in seq.ContourSequence:
                    contour_hold = np.round(np.array(c['ContourData'].value), 3)
                    contour = contour_hold.reshape(int(len(contour_hold) / 3), 3)
                    contour_list.append(contour)

                self.points += contour_list


def create_image_name(modality):
    idx = len(Data.image_list)
    if idx < 9:
        image_name = modality + ' 0' + str(1 + idx)
    else:
        image_name = modality + ' ' + str(1 + idx)

    return image_name
