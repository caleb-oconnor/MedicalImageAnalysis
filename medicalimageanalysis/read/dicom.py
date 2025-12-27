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
from struct import unpack

import numpy as np
import pandas as pd
import pydicom as dicom
from openpyxl.descriptors import NoneSet
from pydicom.uid import generate_uid

from ..structure.deformable import Deformable
from ..structure.dose import Dose
from ..structure.image import Image
from ..structure.rigid import Rigid

from ..data import Data


def sort_images_by_datetime():
    """
    Reorders the global Data.image dictionary and Data.image_list based on the DICOM image acquisition date and time.
    """
    date_time = [str(Data.image[name].date) + str(Data.image[name].time) for name in Data.image_list]
    new_key_order = [Data.image_list[idx] for idx in np.argsort(date_time)]

    Data.image = {key: Data.image[key] for key in new_key_order}
    Data.image_list = list(Data.image.keys())


def thread_process_dicom(path, stop_before_pixels=False):
    """
    Reads a DICOM file from the given path, optionally stopping before loading pixel data.
    """
    try:
        datasets = dicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)
    except:
        datasets = []

    return datasets


class DicomReader(object):
    """
    Main class to read, organize, and process a collection of DICOM files.
    - Multithreaded reading of DICOMs
    - Sorting by modality and orientation
    - Image creation and registration
    - Associating RTSTRUCT/RTDOSE data with corresponding images
    """
    def __init__(self, files, only_tags, only_modality, only_load_roi_names, clear):
        """
        Initializes DICOM reading parameters.

        Arguments:
            files: dictionary containing DICOM file paths.
            only_tags: if True, reads only metadata (no pixel arrays).
            only_modality: list of modalities to process (CT, MR, etc.).
            only_load_roi_names: if True, loads only ROI names (not contours).
            clear: if True, clears the global Data structure before loading.
        """
        self.files = files
        self.only_tags = only_tags
        self.only_load_roi_names = only_load_roi_names

        if only_modality is not None:
            self.only_modality = only_modality
        else:
            self.only_modality = ['CT', 'MR', 'PT', 'US', 'DX', 'RF', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE']

        if clear:
            Data.clear()

        self.ds = []
        self.ds_modality = {key: [] for key in ['CT', 'MR', 'PT', 'US', 'DX', 'RF', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE']}

    def load(self, display_time=False):
        """
        Main entry point to read and organize all DICOM data.
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
        Reads all DICOM files in parallel threads.
        """
        threads = []

        def read_file_thread(file_path):
            self.ds.append(thread_process_dicom(file_path, stop_before_pixels=self.only_tags))

        for file_path in self.files['Dicom']:
            thread = threading.Thread(target=read_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def separate_modalities_and_images(self):
        """
        Purpose:
        Sorts all loaded DICOM datasets (self.ds) into groups by modality, series, orientation, and slice position.

        Divides DICOMs by Modality (e.g., CT, MR, RTSTRUCT).
            For 3D modalities (CT, MR, PT):
                Groups by SeriesInstanceUID and AcquisitionNumber.
                Sorts slices by spatial position and image orientation.
                Determines the acquisition plane (Axial, Coronal, Sagittal).
            For 2D or non-image modalities (US, CR, RTSTRUCT, etc.):
                Simply groups without sorting.

        Populates self.ds_modality[modality] with properly ordered image sets.
        """
        for modality in list(self.ds_modality.keys()):
            images_in_modality = [d for d in self.ds if (0x0008, 0x0060) in d and d['Modality'].value == modality]
            if len(images_in_modality) > 0 and modality in self.only_modality:
                if modality in ['US', 'DX', 'RF', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE']:
                    for image in images_in_modality:
                        self.ds_modality[modality] += [image]

                else:
                    sorting_tags = []
                    for img in images_in_modality:
                        if 'ImageOrientationPatient' not in img or 'ImagePositionPatient' not in img:
                            continue

                        orient = np.asarray(img['ImageOrientationPatient'].value)
                        pos = np.asarray(img['ImagePositionPatient'].value)
                        if 'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None:
                            acq = np.int64(img['AcquisitionNumber'].value)
                        else:
                            acq = 1

                        sorting_tags += [[img['SeriesInstanceUID'].value, acq, orient[0], orient[1], orient[2],
                                          orient[3], orient[4], orient[5], pos[0], pos[1], pos[2]]]

                    sorting_tags = np.asarray(sorting_tags)
                    unique_series = np.unique(np.asarray(sorting_tags[:, 0]), axis=0)
                    for series in unique_series:
                        idx = np.where(sorting_tags[:, 0] == series)
                        series_tags = sorting_tags[idx[0], :]
                        series_image = [images_in_modality[ii] for ii in idx[0]]

                        orientations = series_tags[:, 2:8].astype(np.float64)
                        _, indices = np.unique(np.round(orientations, 3), axis=0, return_index=True)
                        unique_orientations = [orientations[ind].astype(np.float64) for ind in indices]
                        for orient in unique_orientations:
                            orient_idx = np.where((np.round(orientations[:, 0], 3) == np.round(orient[0], 3)) &
                                                  (np.round(orientations[:, 1], 3) == np.round(orient[1], 3)) &
                                                  (np.round(orientations[:, 2], 3) == np.round(orient[2], 3)) &
                                                  (np.round(orientations[:, 3], 3) == np.round(orient[3], 3)) &
                                                  (np.round(orientations[:, 4], 3) == np.round(orient[4], 3)) &
                                                  (np.round(orientations[:, 5], 3) == np.round(orient[5], 3)))

                            orient_tags = np.asarray([series_tags[orient] for orient in orient_idx[0]])
                            orient_image = [series_image[orient] for orient in orient_idx[0]]
                            correct_orientation = orient_tags[0, 2:8].astype(np.float64)

                            x = np.abs(correct_orientation[0]) + np.abs(correct_orientation[3])
                            y = np.abs(correct_orientation[1]) + np.abs(correct_orientation[4])
                            z = np.abs(correct_orientation[2]) + np.abs(correct_orientation[5])

                            row_direction = correct_orientation[:3]
                            column_direction = correct_orientation[3:]
                            slice_direction = np.cross(row_direction, column_direction)

                            unique_acq = np.unique(orient_tags[:, 1])

                            acq_plane = []
                            acq_images = []
                            acq_positions = []
                            for acq in unique_acq:
                                orient_idx = np.where(orient_tags == acq)[0]
                                acq_tags = orient_tags[orient_idx]
                                acq_image = [orient_image[ii] for ii in orient_idx]
                                position_tags = np.asarray([np.asarray(t[8:]).astype(np.double) for t in acq_tags])

                                if x < y and x < z:
                                    acq_plane += ['Sagittal']
                                    if slice_direction[0] > 0:
                                        slice_idx = np.argsort(position_tags[:, 0])
                                    else:
                                        slice_idx = np.argsort(position_tags[:, 0])[::-1]
                                elif y < x and y < z:
                                    acq_plane += ['Coronal']
                                    if slice_direction[1] > 0:
                                        slice_idx = np.argsort(position_tags[:, 1])
                                    else:
                                        slice_idx = np.argsort(position_tags[:, 1])[::-1]
                                else:
                                    acq_plane += ['Axial']
                                    if slice_direction[2] > 0:
                                        slice_idx = np.argsort(position_tags[:, 2])
                                    else:
                                        slice_idx = np.argsort(position_tags[:, 2])[::-1]

                                acq_images += [np.asarray([acq_image[idx] for idx in slice_idx])]
                                acq_positions += [np.asarray([acq_tags[idx] for idx in slice_idx])]

                            if len(acq_positions) > 1:
                                exclude_images = np.zeros((len(acq_positions), 1))
                                for ii in range(len(acq_positions)):
                                    for jj in range(len(acq_positions)):
                                        if ii != jj:
                                            if acq_plane[0] == 'Sagittal':
                                                base_first = acq_positions[ii][0, 8]
                                                base_last = acq_positions[ii][-1, 8]
                                                check_first = acq_positions[jj][0, 8]
                                                check_last = acq_positions[jj][-1, 8]
                                            elif acq_plane[0] == 'Coronal':
                                                base_first = acq_positions[ii][0, 9]
                                                base_last = acq_positions[ii][-1, 9]
                                                check_first = acq_positions[jj][0, 9]
                                                check_last = acq_positions[jj][-1, 9]
                                            else:
                                                base_first = acq_positions[ii][0, 10]
                                                base_last = acq_positions[ii][-1, 10]
                                                check_first = acq_positions[jj][0, 10]
                                                check_last = acq_positions[jj][-1, 10]

                                            base_first = np.float64(base_first)
                                            base_last = np.float64(base_last)
                                            check_first = np.float64(check_first)
                                            check_last = np.float64(check_last)

                                            if base_first > check_first and base_first > check_last:
                                                pass

                                            elif base_last < check_first and base_last < check_last:
                                                pass

                                            else:
                                                exclude_images[ii] = 1

                                if np.sum(exclude_images) == 0:
                                    if acq_plane[0] == 'Sagittal':
                                        pos = np.asarray([[p[0, 8], p[-1, 8]] for p in acq_positions])
                                    elif acq_plane[0] == 'Coronal':
                                        pos = np.asarray([[p[0, 9], p[-1, 9]] for p in acq_positions])
                                    else:
                                        pos = np.asarray([[p[0, 10], p[-1, 10]] for p in acq_positions]).astype(
                                            np.float64)

                                    pos_idx = np.argsort(pos[:, 0])
                                    pos_sort = pos[pos_idx]
                                    pos_diff = [pos_sort[ii + 1, 0] - pos_sort[ii, 1] for ii in range(len(pos) - 1)]
                                    if len(np.unique(np.round(pos_diff, 2))) == 1:
                                        img = []
                                        for ii in pos_idx:
                                            for acq in acq_images[ii]:
                                                img += [acq]
                                        self.ds_modality[modality] += [img]

                                    else:
                                        for img in acq_images:
                                            self.ds_modality[modality] += [img.tolist()]

                                else:
                                    for img in acq_images:
                                        self.ds_modality[modality] += [img.tolist()]

                            else:
                                for img in acq_images:
                                    self.ds_modality[modality] += [img.tolist()]

    def image_creation(self):
        """
        Converts grouped DICOM datasets into actual image/structure objects and integrates them into the global Data structure.

        Image Modalities:
            Uses specialized readers (Read3D, ReadXRay, ReadRF, ReadUS) for CT/MR/PT, DX/CR, RF, US images.

        RTSTRUCT (contours):
            Reads using ReadRTStruct, associates to matching image via Data.image[...].
            Calls Data.match_rois() and Data.match_pois().

        Registration and Dose:
            Reads REG and RTDOSE modalities using their specific readers (ReadREG, ReadRTDose).
        """
        for modality in ['CT', 'MR', 'PT', 'DX', 'RF', 'CR', 'US']:
            for image_set in self.ds_modality[modality]:
                if modality in ['CT', 'MR', 'PT']:
                    Read3D(image_set, self.only_tags)

                elif modality in ['DX', 'CR']:
                    ReadXRay(image_set, self.only_tags)

                elif modality == 'RF':
                    ReadRF(image_set, self.only_tags)

                elif modality == 'US':
                    ReadUS(image_set, self.only_tags)

        for modality in ['RTSTRUCT']:
            for image_set in self.ds_modality[modality]:
                read_rtstruct = ReadRTStruct(image_set, self.only_tags)
                if read_rtstruct.match_image_name is not None:
                    Data.image[read_rtstruct.match_image_name].input_rtstruct(read_rtstruct)
                else:
                    print('dicom: rtstruct has no matching image')

            Data.match_rois()
            Data.match_pois()

        for modality in ['REG']:
            for image_set in self.ds_modality[modality]:
                ReadREG(image_set, self.only_tags)

        for modality in ['RTDOSE']:
            for image_set in self.ds_modality[modality]:
                ReadRTDose(image_set, self.only_tags)


class Read3D(object):
    """
    Handles reading and constructing 3D medical image volumes (specifically CT/MR/PT modalities) from a list of DICOM
    slices. It extracts geometry, orientation, and pixel data to create a volumetric representation of the image and
    registers it in the global Data structure.
    """

    def __init__(self, image_set, only_tags):
        """
        image_set: list of DICOM slices (or a single slice).
        only_tags: if True, loads only DICOM header data (no pixel array).

        Creates an Image object and adds it to Data.image and Data.image_list.
        """
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
        Data.image[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_array(self):
        """
        Applies rescale slope/intercept for intensity normalization, then stacks slices. Deletes PixelData from each
        DICOM object to free memory.
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

            image_slices.append(((_slice.pixel_array * slope) + intercept).astype('int16'))

            del _slice.PixelData

        self.array = np.asarray(image_slices)

    def _compute_orientation(self):
        """
        Extracts the image’s directional cosines (row/column orientation).
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
        Determines the anatomical plane (Axial, Coronal, or Sagittal) based on the orientation vectors.
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
        """
        Ensures that the 3D volume is oriented correctly in physical space.
            - Computes all 8 corner coordinates of the image volume using orientation and spacing.
            - Identifies the physical origin and corrects rotations or flips via np.rot90 and transpositions.
            - Adjusts orientation vectors based on corrected axes.
        """
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
        """
        Detects missing slices by checking irregular gaps in slice positions.
        """
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
    Reads and constructs 2D X-ray images (modalities DX and MG) from DICOM files. Tomosynthesis mammograms (3D MG) are
    not handled by this class.
    """

    def __init__(self, image_set, only_tags):
        """
        image_set: single DICOM image or list of DICOM datasets.
        only_tags: if True, loads only metadata without pixel data.


        """
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
        Data.image[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_plane(self):
        """
        Determines the anatomical plane (Axial, Coronal, Sagittal) from the PatientOrientation DICOM tag. Defaults to
        Axial if not specified.
        """
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
        """
        Defines image dimensions as [depth, height, width]. For 2D X-rays, one axis is set to 1 depending on the
        detected plane.
        """
        if self.plane == 'Axial':
            return np.asarray([1, self.image_set[0]['Rows'].value, self.image_set[0]['Columns'].value])

        elif self.plane == 'Coronal':
            return np.asarray([self.image_set[0]['Rows'].value, 1, self.image_set[0]['Columns'].value])

        else:
            return np.asarray([self.image_set[0]['Rows'].value, self.image_set[0]['Columns'].value, 1])

    def _compute_spacing(self):
        """
        Calculates voxel spacing [x, y, z]. Uses available DICOM tags (e.g. PixelSpacing, ImagerPixelSpacing).
        Sets slice thickness to 1 mm since X-rays are 2D. Adjusts spacing order based on the anatomical plane.
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
        Reads the image pixel array (pixel_array) and converts it to int16. Deletes raw pixel data from memory after
        reading. If the PresentationLUTShape tag is 'Inverse', inverts grayscale values (16383 - array). Reshapes and
        flips the array according to the detected plane for consistent orientation.
        """
        self.array = self.image_set[0].pixel_array.astype('int16')
        del self.image_set[0].PixelData

        if 'PresentationLUTShape' in self.image_set[0] and self.image_set[0]['PresentationLUTShape'] == 'Inverse':
            self.array = 16383 - self.array

        if self.plane == 'Axial':
            self.array = self.array.reshape((1, self.array.shape[0], self.array.shape[1]))
        elif self.plane == 'Coronal':
            self.array = np.flip(np.flip(self.array.reshape((self.array.shape[0], 1, self.array.shape[1])), axis=0),
                                 axis=1)
        else:
            self.array = np.flip(self.array.reshape((self.array.shape[0], self.array.shape[1], 1)), axis=0)


class ReadRF(object):
    """
    Handles Radio Fluoroscopy (RF) image data from DICOM files. Builds a standardized representation containing
    metadata, orientation, and pixel data for integration into the global Data structure.
    """
    def __init__(self, image_set, only_tags):
        """
        image_set: a DICOM object or list of DICOM datasets.
        only_tags: if True, loads metadata only (no pixel data).
        """
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
        Data.image[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_plane(self):
        """
        Determines the anatomical viewing plane (Axial, Coronal, Sagittal) based on the DICOM PatientOrientation tag.
        Defaults to Axial if the tag is missing.
        """
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
        Reads image pixel data into a NumPy array (int16). Deletes raw pixel data from memory to save space.
        Reshapes the array into a consistent 3D form, even for single 2D frames, depending on the detected plane.
        Saves the final shape as self.dimensions.
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
        Determines voxel spacing using standard DICOM tags (PixelSpacing, ImagerPixelSpacing, etc.). If no tags are
        found, defaults to [1, 1, 1] (1 mm in each direction). Adjusts order of spacing values depending on the
        anatomical plane.
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
    Handles Ultrasound (US) images from DICOM files. Similar in structure to ReadXRay, but US images can include stacks
    of frames that do not represent true anatomical slices.
    """

    def __init__(self, image_set, only_tags):
        """
        image_set: a single DICOM US image or a list of images.
        only_tags: if True, loads only metadata without pixel data.
        """
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
        self.dimensions = np.asarray([1, self.image_set[0]['Rows'].value, self.image_set[0]['Columns'].value])

        self.array = None
        if not self.only_tags:
            self._compute_array()
        self.spacing = self._compute_spacing()

        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list += [self.image_name]

    def _compute_array(self):
        """
        Reads pixel data and converts to NumPy array (uint8). Reshapes 2D images to 3D arrays with a single frame.
        For multi-frame images:
            Checks for uniform frames and extracts the first channel.
            Updates self.dimensions based on the number of frames.
        """
        us_data = np.asarray(self.image_set[0].pixel_array)
        del self.image_set[0].PixelData

        if len(us_data.shape) == 2:
            us_data = us_data.reshape((1, us_data.shape[0], us_data.shape[1]))

        if len(us_data.shape) == 3:
            us_binary = (1 * (np.std(us_data, axis=2) == 0) == 1)
            self.array = (us_binary * us_data[:, :, 0]).astype('uint8')
            if len(self.array.shape) == 2:
                self.array = np.expand_dims(self.array, axis=0)

        else:
            us_binary = (1 * (np.std(us_data, axis=3) == 0) == 1)
            self.array = (us_binary * us_data[:, :, :, 0]).astype('uint8')

        if len(self.array.shape) == 3:
            self.dimensions[0] = self.array.shape[0]

    def _compute_spacing(self):
        """
        Determines pixel spacing using standard DICOM tags (PixelSpacing, DetectorElementSpacing, etc.) or
        ultrasound-specific sequences. Defaults to [1, 1, 1] if tags are missing. Returns spacing in [x, y, z] order
        with a default slice thickness of 1 mm.
        """
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
    """
    Handles Radiotherapy Structure Set (RTSTRUCT) DICOM files. Extracts ROI (Region of Interest) contours and POI
    (Points of Interest), matches them to a corresponding image series, and prepares them for integration with the
    global Data structure.
    """
    def __init__(self, image_set, only_tags):
        """
        image_set: a DICOM RTSTRUCT object.
        only_tags: if True, only metadata is loaded (no contour/point coordinates).
        """
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
        """
        Reads the SeriesInstanceUID of the image series referenced by the RTSTRUCT.
        """
        study = 'RTReferencedStudySequence'
        series = 'RTReferencedSeriesSequence'
        ref = self.image_set.ReferencedFrameOfReferenceSequence

        return ref[0][study][0][series][0]['SeriesInstanceUID'].value

    def _get_properties(self):
        """
        Iterates through ROIContourSequence. Collects ROI/POI names, colors, geometric type, and linked SOPInstanceUIDs.
        Assigns random colors if none are provided.
        """
        tracker = []
        sop = []
        names = []
        colors = []
        geometric = []
        if ROIContourSequence in self.image_set:
            for ii, s in enumerate(self.image_set.ROIContourSequence):
                if hasattr(self.image_set.StructureSetROISequence[ii], 'ROIName'):
                    if hasattr(s, 'ContourSequence'):
                        tracker += [ii]
                        names += [self.image_set.StructureSetROISequence[ii].ROIName]
                        geometric += [s['ContourSequence'][0]['ContourGeometricType'].value]

                        slice_sop = []
                        if geometric[-1].lower() == 'closed_planar':
                            for seq in s['ContourSequence']:
                                slice_sop += [seq['ContourImageSequence'][0]['ReferencedSOPInstanceUID'].value]
                        else:
                            if 'ContourImageSequence' in s['ContourSequence'][0]:
                                slice_sop = [s['ContourSequence'][0]['ContourImageSequence'][0]['ReferencedSOPInstanceUID'].value]
                            else:
                                slice_sop = []
                        sop += [slice_sop]

                        if hasattr(s, 'ROIDisplayColor'):
                            colors += [s.ROIDisplayColor]
                        else:
                            colors += [[np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]]

        properties = []
        for ii in range(len(names)):
            properties += [[tracker[ii], names[ii], colors[ii], geometric[ii], sop[ii]]]

        return properties

    def _match_with_image(self):
        """
        Searches Data.image for an image with a matching series UID and SOPInstanceUID. Returns the matched image name
        or None if no match exists.
        """
        match_image_name = None

        for image_name in Data.image:
            if self.series_uid == Data.image[image_name].series_uid:
                if self._properties[0][4][0] in Data.image[image_name].sops:
                    match_image_name = image_name

        return match_image_name

    def _structure_positions(self):
        """
        Extracts the 3D coordinates of contours and points. Stores planar ROIs in self.contours and points in
        self.points.
        """
        sequences = self.image_set.ROIContourSequence
        for prop in self._properties:
            seq = sequences[prop[0]]

            contour_list = []
            for c in seq.ContourSequence:
                contour_hold = np.round(np.array(c['ContourData'].value), 3)
                contour = contour_hold.reshape(int(len(contour_hold) / 3), 3)
                contour_list.append(contour)

            if prop[3].lower() == 'closed_planar':
                self.contours += [contour_list]
            else:
                self.points += contour_list


class ReadREG(object):
    def __init__(self, image_set, only_tags):
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.reference_name = None
        self.reference_series = self.image_set[0].ReferencedSeriesSequence[0].SeriesInstanceUID
        self.reference_sops = [sop.ReferencedSOPInstanceUID for sop in
                               self.image_set[0].ReferencedSeriesSequence[0].ReferencedInstanceSequence]

        self.moving_name = None
        if len(self.image_set[0].ReferencedSeriesSequence) == 2:
            self.moving_series = self.image_set[0].ReferencedSeriesSequence[1].SeriesInstanceUID
            self.moving_sops = [sop.ReferencedSOPInstanceUID for sop in
                                self.image_set[0].ReferencedSeriesSequence[1].ReferencedInstanceSequence]
        else:
            sequence = self.image_set[0].StudiesContainingOtherReferencedInstancesSequence[0].ReferencedSeriesSequence[0]
            self.moving_series = sequence.SeriesInstanceUID
            self.moving_sops = [sop.ReferencedSOPInstanceUID for sop in sequence.ReferencedInstanceSequence]

        self.spacing = None
        self.dimensions = None
        self.origin = None

        self.reference_matrix = None
        self.moving_matrix = None
        self.dvf_matrix = None
        self.dvf = None

        self.registration_name = None
        if 'DeformableRegistrationSequence' in self.image_set[0]:
            self._compute_rigid(deformable=True)
            self._compute_dvf()
            self._create_name(deformable=True)
            self._create_registration(deformable=True)
        else:
            self._compute_rigid()
            self._create_name()
            self._create_registration()

    def _compute_sops(self):
        pass

    def _compute_rigid(self, deformable=False):
        if deformable:
            matrix = self.image_set[0].DeformableRegistrationSequence[0].PreDeformationMatrixRegistrationSequence[0][
                0x3006, 0x00C6].value
            orientation = self.image_set[0].DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].ImageOrientationPatient
            row_direction = orientation[:3]
            column_direction = orientation[3:]
            slice_direction = np.cross(row_direction, column_direction)

            self.dvf_matrix = np.identity(3, dtype=np.float32)
            self.dvf_matrix[0, :3] = row_direction
            self.dvf_matrix[1, :3] = column_direction
            self.dvf_matrix[2, :3] = slice_direction
            self.moving_matrix = np.linalg.inv(np.asarray(matrix).reshape(4, 4))
        else:
            self.reference_matrix = self.image_set[0].RegistrationSequence[1]['MatrixRegistrationSequence'][0]['MatrixSequence'][0][0x3006, 0x00C6].value
            matrix = self.image_set[0].RegistrationSequence[1]['MatrixRegistrationSequence'][0]['MatrixSequence'][0][0x3006, 0x00C6].value
            self.moving_matrix = np.linalg.inv(np.asarray(matrix).reshape(4, 4))

    def _compute_dvf(self):
        self.origin = self.image_set[0].DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[
            0].ImagePositionPatient
        self.dimensions = np.flip(
            self.image_set[0].DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridDimensions)
        self.spacing = self.image_set[0].DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].GridResolution

        dvf_raw = self.image_set[0].DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[
            0].VectorGridData
        dvf_unstructured = unpack(f"<{len(dvf_raw) // 4}f", dvf_raw)
        self.dvf = np.reshape(dvf_unstructured, list(self.dimensions) + [3, ])

        del self.image_set[0].DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[0].VectorGridData

    def _create_name(self, deformable=False):
        for image_name in Data.image_list:
            if self.reference_sops[0] in Data.image[image_name].sops:
                self.reference_name = image_name

            elif self.moving_sops[0] in Data.image[image_name].sops:
                self.moving_name = image_name

        if deformable:
            registration = 'DVF_'
        else:
            registration = ''

        if self.reference_name is None and self.moving_name is None:
            registration_name = registration + '_Unknown'
        else:
            registration_name = registration + self.reference_name + '_' + self.moving_name

        if deformable:
            if registration_name in Data.deformable_list:
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(registration_name + '_' + str(n))
                    if new_name not in Data.deformable_list:
                        self.registration_name = new_name
                        n = -100
            else:
                self.registration_name = registration_name
        else:
            if registration_name in Data.rigid_list:
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(registration_name + '_' + str(n))
                    if new_name not in Data.rigid_list:
                        self.registration_name = new_name
                        n = -100
            else:
                self.registration_name = registration_name

    def _create_registration(self, deformable=False):
        if deformable:
            Deformable(self.dvf, self.origin, self.spacing, self.dimensions, rigid_matrix=self.moving_matrix,
                       dvf_matrix=self.dvf_matrix, registration_name=self.registration_name,
                       reference_name=self.reference_name, moving_name=self.moving_name,
                       reference_sops=self.reference_sops, moving_sops=self.moving_sops)

        elif self.reference_name is not None and self.moving_name is not None:
            Rigid(self.reference_name, self.moving_name, rigid_name=self.registration_name,
                  reference_sops=self.reference_sops, moving_sops=self.moving_sops,
                  reference_matrix=self.reference_matrix, matrix=self.moving_matrix)


class ReadRTDose(object):
    """
    Handles Radiotherapy Dose (RTDOSE) DICOM files. Reads dose arrays, determines spacing, orientation, and dimensions,
    and integrates the dose into the global Data structure.
    """
    def __init__(self, image_set, only_tags):
        """
        image_set: one or more DICOM RTDOSE objects.
        only_tags: if True, only metadata is loaded (no dose array computation).
        """
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.unverified = None
        self.base_position = None
        self.skipped_slice = None
        self.sections = None
        self.modality = 'RTDOSE'

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
        self.dose_name = create_dose_name(self.modality)

        dose = Dose(self)
        Data.dose[self.dose_name] = dose
        Data.dose_list += [self.dose_name]

    def _compute_array(self):
        """
        Combines slices into a 3D dose array. Applies DoseGridScaling if present. Deletes original pixel data.
        """

        if (0x3004, 0x000E) in self.image_set[0]:
            slope = self.image_set[0].DoseGridScaling
        else:
            slope = 1

        dose_array = (self.image_set[0].pixel_array * slope)

        self.array = np.asarray(dose_array)
        if len(dose_array.shape) == 2:
            self.array.reshape((1, self.array.shape[0], self.array.shape[1]))

        del self.image_set[0].PixelData

    def _compute_orientation(self):
        """
        Reads orientation from ImageOrientationPatient or functional group sequences. Flags unverified if orientation
        cannot be determined.
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
        Determines the anatomical plane (Axial, Coronal, Sagittal) based on orientation vectors.
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
        Computes 3-axis voxel spacing from pixel spacing and slice thickness. Detects skipped slices and adjusts spacing
        if slices are irregularly spaced (_find_skipped_slices).
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
        Sets dimensions based on array shape and plane orientation.
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
        Constructs a 3×3 image matrix using row, column, and slice directions.
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
        """
        Checks and corrects the origin and orientation of the dose array to ensure proper alignment. Rotates or flips
        the array as needed.
        """
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
        """
        Detects skipped or irregular slices along the slice direction and flags them.
        """
        base_spacing = None
        for ii in range(len(self.image_set) - 1):
            position_1 = np.dot(slice_direction, self.image_set[ii].ImagePositionPatient)
            position_2 = np.dot(slice_direction, self.image_set[ii + 1].ImagePositionPatient)
            if ii == 0:
                base_spacing = position_2 - position_1
            if ii > 0 and np.abs(base_spacing - (position_2 - position_1)) > 0.01:
                self.unverified = 'Skipped'
                self.skipped_slice = ii + 1


def create_image_name(modality):
    """
    Generate a unique, sequential name for an image based on its modality.
    """
    idx = len(Data.image_list)
    if idx < 9:
        image_name = modality + ' 0' + str(1 + idx)
    else:
        image_name = modality + ' ' + str(1 + idx)

    return image_name


def create_dose_name(modality):
    """
    Generate a unique, sequential name for a dose based on its modality.
    """
    idx = len(Data.dose_list)
    if idx < 9:
        image_name = modality + ' 0' + str(1 + idx)
    else:
        image_name = modality + ' ' + str(1 + idx)

    return image_name
