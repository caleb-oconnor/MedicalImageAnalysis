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

import numpy as np
import pandas as pd
import pydicom as dicom
from pydicom.uid import generate_uid

from ..Data import Image


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
        self.ds_modality = {key: [] for key in ['CT', 'MR', 'PT', 'US', 'DX', 'MG', 'NM', 'XA', 'CR',
                                                'RTSTRUCT', 'REG', 'RTDose']}

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
            images_in_modality = [d for d in self.ds if d['Modality'].value == modality]
            if len(images_in_modality) > 0 and modality in self.reader.only_modality:
                if modality not in ['US', 'DX', 'MG', 'XA', 'CR', 'RTSTRUCT', 'REG', 'RTDose']:
                    sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['AcquisitionNumber'].value] if
                                               'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None
                                               else [img['SeriesInstanceUID'].value, 1] for img in images_in_modality])

                    unique_tags = np.unique(sorting_tags, axis=0)
                    for tag in unique_tags:
                        sorted_idx = np.where((sorting_tags[:, 0] == tag[0]) & (sorting_tags[:, 1] == tag[1]))
                        image_tags = [images_in_modality[idx] for idx in sorted_idx[0]]

                        if 'ImageOrientationPatient' in image_tags[0] and 'ImagePositionPatient' in image_tags[0]:
                            orientations = np.asarray([img['ImageOrientationPatient'].value for img in image_tags])
                            _, indices = np.unique(np.round(orientations, 4), axis=0, return_index=True)
                            unique_orientations = [orientations[ind] for ind in indices]
                            for orient in unique_orientations:
                                orient_idx = np.where((np.round(orientations[:, 0]) == np.round(orient[0])) &
                                                      (np.round(orientations[:, 1]) == np.round(orient[1])) &
                                                      (np.round(orientations[:, 2]) == np.round(orient[2])) &
                                                      (np.round(orientations[:, 3]) == np.round(orient[3])) &
                                                      (np.round(orientations[:, 4]) == np.round(orient[4])) &
                                                      (np.round(orientations[:, 5]) == np.round(orient[5])))

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

                elif modality == 'RTSTRUCT':
                    for image in images_in_modality:
                        if 'StructureSetROISequence' in image and 'ROIContourSequence' in image:
                            self.ds_modality[modality] += [image]

                elif modality in ['US', 'DX', 'MG', 'XA', 'CR', 'RTSTRUCT', 'REG', 'RTDose']:
                    for image in images_in_modality:
                        self.ds_modality[modality] += [image]

    def image_creation(self):
        """
        Currently only reading in 5 modalities (CT, MR, DX, US, RTSTRUCT) and using specific modality class readers.
        First the image volume modalities are created, then RTSTRUCT is added to the image that it associates with.

        :return:
        :rtype:
        """
        for modality in ['CT', 'MR', 'DX', 'MG', 'US']:
            read_image = None
            for image_set in self.ds_modality[modality]:
                load = False

                if modality in ['CT', 'MR']:
                    load = True
                    read_image = Read3D(image_set, self.reader.only_tags)

                elif modality == 'DX':
                    load = True
                    read_image = ReadDX(image_set, self.reader.only_tags)

                elif modality == 'MG':
                    if 'ImageType' in image_set:
                        if 'VOLUME' in image_set['ImageType'].value or 'TOMOSYNTHESIS' in image_set['ImageType'].value:
                            pass
                            # images += [ReadMG(image_set, self.reader.only_tags)]

                        else:
                            load = True
                            read_image = ReadDX(image_set, self.reader.only_tags)

                elif modality == 'US':
                    load = True
                    read_image = ReadUS(image_set, self.reader.only_tags)

                if load:
                    image = Image()
                    image.input(read_image)
                    self.reader.images += [image]

        for modality in ['RTSTRUCT']:
            for image_set in self.ds_modality[modality]:
                if modality == 'RTSTRUCT':
                    read_rtstruct = ReadRTStruct(image_set, self.reader.images, self.reader.only_tags)
                    if read_rtstruct.match_image_idx is not None:
                        self.reader.images[read_rtstruct.match_image_idx].input_rtstruct(read_rtstruct)
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

        self.array = None
        if not self.only_tags:
            self._compute_array()

        self.filepaths = [image.filename for image in self.image_set]
        self.sops = [image.SOPInstanceUID for image in self.image_set]
        self.plane = self._compute_plane()
        self.spacing = self._compute_spacing()
        self.dimensions = self._compute_dimensions()
        self.orientation = self._compute_orientation()
        self.origin = self._compute_origin()
        self.image_matrix = self._compute_image_matrix()

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

    def _compute_plane(self):
        """
        Computes the image plane for the slices
        :return:
        :rtype:
        """
        orientation = self.image_set[0]['ImageOrientationPatient'].value
        x = np.abs(orientation[0]) + np.abs(orientation[3])
        y = np.abs(orientation[1]) + np.abs(orientation[4])
        z = np.abs(orientation[2]) + np.abs(orientation[5])

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

        return np.asarray([inplane_spacing[0], inplane_spacing[1], slice_thickness])

    def _compute_dimensions(self):
        """
        Creates dimensions by columns, rows, and number of slices.
        :return:
        :rtype:
        """
        return np.asarray([self.image_set[0]['Columns'].value, self.image_set[0]['Rows'].value, len(self.image_set)])

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

    def _compute_origin(self):
        """
        Patient can exist on stomach, back, or side laying on the bench. This is used to rotate the image to always be
        feet first supine. It creates the array, the orientation, and the position.
        :return:
        :rtype:
        """
        origin = np.asarray(self.image_set[0]['ImagePositionPatient'].value)
        if 'PatientPosition' in self.image_set[0]:
            self.base_position = self.image_set[0]['PatientPosition'].value

            if self.base_position in ['HFDR', 'FFDR']:
                if self.only_tags:
                    self.array = np.rot90(self.array, 3, (1, 2))

                origin[0] = np.double(origin[0]) - self.spacing[0] * (self.dimensions[0] - 1)
                self.orientation = [-self.orientation[3], -self.orientation[4], -self.orientation[5],
                                    self.orientation[0], self.orientation[1], self.orientation[2]]

            elif self.base_position in ['HFP', 'FFP']:
                if self.only_tags:
                    self.array = np.rot90(self.array, 2, (1, 2))

                origin[0] = np.double(origin[0]) - self.spacing[0] * (self.dimensions[0] - 1)
                origin[1] = np.double(origin[1]) - self.spacing[1] * (self.dimensions[1] - 1)
                self.orientation = -np.asarray(self.orientation)

            elif self.base_position in ['HFDL', 'FFDL']:
                if self.only_tags:
                    self.array = np.rot90(self.array, 1, (1, 2))

                origin[1] = np.double(origin[1]) - self.spacing[1] * (self.dimensions[1] - 1)
                self.orientation = [self.orientation[3], self.orientation[4], self.orientation[5],
                                    -self.orientation[0], -self.orientation[1], -self.orientation[2]]

        return origin

    def _compute_image_matrix(self):
        """
        Computes the image matrix using the image orientation.

        Sometimes SliceThickness tag isn't correct this looks for position changes over the slices to recalculate the
        slice thickness as need. Also, rarely when scans transition from abdomen to pelvis protocol there is a skipped
        slice, this signals when that happens (it could happen in other instances this is just one I am familiar with).
        :return:
        :rtype:
        """
        row_direction = self.orientation[:3]
        column_direction = self.orientation[3:]

        slice_direction = np.cross(row_direction, column_direction)
        if len(self.image_set) > 1:
            first = np.dot(slice_direction, self.image_set[0].ImagePositionPatient)
            second = np.dot(slice_direction, self.image_set[1].ImagePositionPatient)
            last = np.dot(slice_direction, self.image_set[-1].ImagePositionPatient)
            first_last_spacing = np.asarray((last - first) / (len(self.image_set) - 1))
            if np.abs((second - first) - first_last_spacing) > 0.01:
                if not self.only_tags:
                    self._find_skipped_slices(slice_direction)
                slice_spacing = second - first
            else:
                slice_spacing = np.asarray((last - first) / (len(self.image_set) - 1))

            self.spacing[2] = slice_spacing
            
        mat = np.identity(3, dtype=np.float32)
        mat[0, :3] = row_direction
        mat[1, :3] = column_direction
        mat[2, :3] = slice_direction

        return mat

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


class ReadDX(object):
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
        self.base_position = self.image_set[0].PatientOrientation
        self.skipped_slice = None
        self.sections = None
        self.rgb = False

        self.modality = self.image_set[0].Modality

        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID
        self.plane = self.image_set[0].ViewPosition
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.asarray([0, 0, 0])
        self.image_matrix = np.identity(4, dtype=np.float32)
        self.dimensions = np.asarray([self.image_set[0]['Columns'].value, self.image_set[0]['Rows'].value, 1])

        self.array = None
        if not self.only_tags:
            self._compute_array()
        self.spacing = self._compute_spacing()

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

        self.array = self.array.reshape((1, self.array.shape[0], self.array.shape[1]))

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

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set[0]:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        return np.asarray([inplane_spacing[0], inplane_spacing[1], slice_thickness])


class ReadMG(object):
    def __init__(self, image_set, only_tags):
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.unverified = 'Modality'
        self.base_position = self.image_set[0].PatientOrientation
        self.skipped_slice = None
        self.sections = None
        self.rgb = False

        self.modality = self.image_set[0].Modality

        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID
        self.origin = np.asarray([0, 0, 0])

        self.array = None
        if not self.only_tags:
            self._compute_array()
        self.spacing = self._compute_spacing()
        self.dimensions = self._compute_dimensions()
        self.orientation = self._compute_orientation()
        self.plane = self._compute_plane
        self.image_matrix = None
        # self.image_matrix = self._compute_image_matrix()

    def _compute_array(self):
        if (0x0028, 0x1052) in self.image_set[0]:
            intercept = self.image_set[0].RescaleIntercept
        else:
            intercept = 0

        if (0x0028, 0x1053) in self.image_set[0]:
            slope = self.image_set[0].RescaleSlope
        else:
            slope = 1

        self.array = ((self.image_set[0].pixel_array*slope)+intercept).astype('int16')

        del self.image_set[0].PixelData

    def _compute_plane(self):
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
        inplane_spacing = [1, 1]
        slice_thickness = 1

        if 'PixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].PixelSpacing

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        return np.asarray([inplane_spacing[0], inplane_spacing[1], slice_thickness])

    def _compute_dimensions(self):
        if self.array is not None:
            slices = self.array.shape[0]
        else:
            slices = 1
        return np.asarray([self.image_set[0]['Columns'].value, self.image_set[0]['Rows'].value, slices])

    def _compute_orientation(self):
        orientation = np.asarray([1, 0, 0, 0, 1, 0])
        if 'ImageOrientationPatient' in self.image_set[0]:
            orientation = np.asarray(self.image_set[0]['ImageOrientationPatient'].value)

        else:
            if 'SharedFunctionalGroupsSequence' in self.image_set[0]:
                seq_str = 'SharedFunctionalGroupsSequence'
                if 'PlaneOrientationSequence' in self.image_set[0][seq_str][0]:
                    plane_str = 'PlaneOrientationSequence'
                    image_str = 'ImageOrientationPatient'
                    orientation = np.asarray(self.image_set[0][seq_str][0][plane_str][0][image_str].value)

                else:
                    self.unverified = 'Orientation'

            else:
                self.unverified = 'Orientation'

        return orientation

    def _compute_image_matrix(self):
        row_direction = self.orientation[:3]
        column_direction = self.orientation[3:]

        slice_direction = np.cross(row_direction, column_direction)
        if len(self.image_set) > 1:
            first = np.dot(slice_direction, self.image_set[0].ImagePositionPatient)
            last = np.dot(slice_direction, self.image_set[-1].ImagePositionPatient)

            self.spacing[2] = np.asarray((last - first) / (len(self.image_set) - 1))

        mat = np.identity(3, dtype=np.float32)
        mat[0, :3] = row_direction
        mat[1, :3] = column_direction
        mat[2, :3] = slice_direction

        return mat


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
        self.image_matrix = np.identity(4, dtype=np.float32)
        self.dimensions = np.asarray([self.image_set[0]['Columns'].value, self.image_set[0]['Rows'].value, 1])

        self.array = None
        if not self.only_tags:
            self._compute_array()
        self.spacing = self._compute_spacing()

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
                inplane_spacing = [10 * np.round(self.image_set[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaX, 4),
                                   10 * np.round(self.image_set[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaY, 4)]

        return np.asarray([inplane_spacing[0], inplane_spacing[1], slice_thickness])


class ReadRTStruct(object):
    def __init__(self, image_set, reference_images, only_tags):
        self.image_set = image_set
        self.reference_images = reference_images
        self.only_tags = only_tags

        self.series_uid = self._get_series_uid()
        self.filepaths = self.image_set.filename

        self._properties = self._get_properties()
        self.roi_names = [prop[1] for prop in self._properties if prop[3].lower() == 'closed_planar']
        self.roi_colors = [prop[2] for prop in self._properties if prop[3].lower() == 'closed_planar']
        self.poi_names = [prop[1] for prop in self._properties if prop[3].lower() == 'point']
        self.poi_colors = [prop[2] for prop in self._properties if prop[3].lower() == 'point']

        self.match_image_idx = self._match_with_image()

        self.contours = []
        self.points = []
        if not self.only_tags:
            self._structure_positions()

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
        match_image_idx = None
        for ii, reference in enumerate(self.reference_images):
            if self.series_uid == reference.series_uid:
                if self._properties[0][4][0] in reference.sops:
                    match_image_idx = ii

        return match_image_idx

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
