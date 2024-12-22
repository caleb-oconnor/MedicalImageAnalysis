
import os
import copy
import time
import gdcm
import threading

import numpy as np
import pandas as pd
import pydicom as dicom
from pydicom.uid import generate_uid

from ..DataClasses import Image


def thread_process_dicom(path, stop_before_pixels=False):
    try:
        datasets = dicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)
    except:
        datasets = []

    return datasets


class DicomReader:
    def __init__(self, dicom_files, only_tags=False, only_load_roi_names=None):
        self.dicom_files = dicom_files
        self.only_tags = only_tags
        self.only_load_roi_names = only_load_roi_names

        self.ds = []
        self.ds_modality = {key: [] for key in ['CT', 'MR', 'PT', 'US', 'DX', 'MG', 'NM', 'XA', 'CR', 'RTSTRUCT', 'REG',
                                          'RTDose']}

        self.images = None

    def add_dicom_extension(self):
        for ii, name in enumerate(self.dicom_files):
            a, b = os.path.splitext(name)
            if not b:
                self.dicom_files[ii] = name + '.dcm'

    def load_dicoms(self, display_time=True):
        t1 = time.time()
        self.read()
        self.separate_modalities()
        self.image_creation()

        # if not only_tags:
        #     self.convert_images()
        #     self.fix_orientation()
        #     self.separate_contours()
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
            self.ds.append(thread_process_dicom(file_path, stop_before_pixels=self.only_tags))

        for file_path in self.dicom_files:
            thread = threading.Thread(target=read_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def separate_modalities(self):
        """
        Separate read in files by their modality and image based on SeriesInstanceUID and AcquisitionNumber.
        US and DX (X-ray) are considered 2d images, therefore they don't require image separation, because each file
        is considered to be a unique image, even though US can have multiple "slices" per file each slice will be
        considered a 2d image.

        ds_modality - dictionary of different modalities
        Returns
        -------

        """
        for modality in list(self.ds_modality.keys()):
            modalities = [d for d in self.ds if d['Modality'].value == modality]
            if len(modalities) > 0:
                if modality not in ['US', 'DX', 'RTSTRUCT', 'REG', 'RTDose']:
                    sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['AcquisitionNumber'].value] if
                                               'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None
                                               else [img['SeriesInstanceUID'].value, 1] for img in modalities])

                    unique_tags = np.unique(sorting_tags, axis=0)
                    for tag in unique_tags:
                        sorted_idx = np.where((sorting_tags[:, 0] == tag[0]) & (sorting_tags[:, 1] == tag[1]))
                        image_tags = [modalities[idx] for idx in sorted_idx[0]]

                        if 'ImageOrientationPatient' in image_tags[0] and 'ImagePositionPatient' in image_tags[0]:
                            orientation = image_tags[0]['ImageOrientationPatient'].value
                            position_tags = np.asarray([t['ImagePositionPatient'].value for t in image_tags])

                            x = np.abs(orientation[0]) + np.abs(orientation[3])
                            y = np.abs(orientation[1]) + np.abs(orientation[4])
                            z = np.abs(orientation[2]) + np.abs(orientation[5])

                            if x < y and x < z:
                                slice_idx = np.argsort(position_tags[:, 0])
                            elif y < x and y < z:
                                slice_idx = np.argsort(position_tags[:, 1])
                            else:
                                slice_idx = np.argsort(position_tags[:, 2])

                            self.ds_modality[modality] += [[image_tags[idx] for idx in slice_idx]]

                        else:
                            self.ds_modality[modality] += [image_tags]

                elif modality in ['US', 'DX', 'RTSTRUCT', 'REG', 'RTDose']:
                    for image in modalities:
                        self.ds_modality[modality] += [image]

    def image_creation(self):
        for modality in list(self.ds_modality.keys()):
            image_3d = []
            for image_set in self.ds_modality[modality]:
                if modality not in ['US', 'DX', 'RTSTRUCT', 'REG', 'RTDose']:
                    image_3d += [Image3d(image_set, self.only_tags)]
            print(1)

    def separate_types(self):
        for modality in list(self.ds_modality.keys()):
            if len(self.ds_modality[modality]) > 0:
                if modality not in ['US', 'DX', 'RTSTRUCT', 'REG', 'RTDose']:
                    sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['AcquisitionNumber'].value] if
                                               'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None
                                               else [img['SeriesInstanceUID'].value, 1] for img in
                                               self.ds_modality[modality]])

                    unique_tags = np.unique(sorting_tags, axis=0)
                    for tag in unique_tags:
                        sorted_idx = np.where((sorting_tags[:, 0] == tag[0]) & (sorting_tags[:, 1] == tag[1]))
                        image_tags = [self.ds_modality[modality][idx] for idx in sorted_idx[0]]

                        if 'ImageOrientationPatient' in image_tags[0] and 'ImagePositionPatient' in image_tags[0]:
                            orientation = image_tags[0]['ImageOrientationPatient'].value
                            position_tags = np.asarray([t['ImagePositionPatient'].value for t in image_tags])

                            x = np.abs(orientation[0]) + np.abs(orientation[3])
                            y = np.abs(orientation[1]) + np.abs(orientation[4])
                            z = np.abs(orientation[2]) + np.abs(orientation[5])

                            if x < y and x < z:
                                slice_idx = np.argsort(position_tags[:, 0])
                            elif y < x and y < z:
                                slice_idx = np.argsort(position_tags[:, 1])
                            else:
                                slice_idx = np.argsort(position_tags[:, 2])

                            self.ds_type['Images'] += [[image_tags[idx] for idx in slice_idx]]

                        else:
                            self.ds_type['Images'] += [image_tags]

                elif modality in ['US', 'DX']:
                    for image in self.ds_modality[modality]:
                        self.ds_type['Images'] += [image]

                elif modality == 'RTSTRUCT':
                    for image in self.ds_modality[modality]:
                        self.ds_type['Rt'] += [image]

                elif modality == 'REG':
                    for image in self.ds_modality[modality]:
                        self.ds_type['Reg'] += [image]

                elif modality == 'RTDose':
                    for image in self.ds_modality[modality]:
                        self.ds_type['Dose'] += [image]

    def separate_rts(self):
        """
        Loops through all RTSTRUCT files found. Some required information that will be used later in making the contours
        is pulled:
            SeriesInstanceUID
            RoiNames
            RoiSOP - this will be used to determine what slice the contour exist on
        Returns
        -------

        """
        for ii, rt_ds in enumerate(self.ds_modality['RTSTRUCT']):
            ref = rt_ds.ReferencedFrameOfReferenceSequence
            series_uid = ref[0]['RTReferencedStudySequence'][0]['RTReferencedSeriesSequence'][0][
                'SeriesInstanceUID'].value

            roi_sop = []
            for contour_list in rt_ds.ROIContourSequence:
                points = [c.NumberOfContourPoints for c in contour_list['ContourSequence']]
                if np.sum(np.asarray(points)) > 3:
                    roi_sop.append(contour_list['ContourSequence'][0]
                                   ['ContourImageSequence'][0]['ReferencedSOPInstanceUID'].value)

            self.rt_df.at[ii, 'FilePath'] = rt_ds.filename
            self.rt_df.at[ii, 'SeriesInstanceUID'] = series_uid
            self.rt_df.at[ii, 'RoiSOP'] = roi_sop
            self.rt_df.at[ii, 'RoiNames'] = [s.ROIName for s in rt_ds.StructureSetROISequence]

    def standard_useful_tags(self):
        """
        Important tags for each image that I use in DRAGON:
        Returns
        -------

        """
        for ii, image in enumerate(self.ds_images):
            for t in list(self.image_info.keys()):
                if t == 'FilePath':
                    self.image_info.at[ii, t] = [img.filename for img in image]

                elif t == 'SOPInstanceUID':
                    self.image_info.at[ii, t] = [img.SOPInstanceUID for img in image]

                elif t == 'PixelSpacing':
                    self.find_pixel_spacing(image[0], ii)

                elif t == 'ImagePositionPatient':
                    if image[0].Modality in ['US', 'CR', 'DX', 'MG', 'NM', 'XA']:
                        self.image_info.at[ii, t] = [0, 0, 0]
                    else:
                        self.image_info.at[ii, t] = image[0].ImagePositionPatient

                elif t == 'SliceThickness':
                    if len(image) > 1:
                        thickness = (np.asarray(image[1]['ImagePositionPatient'].value[2]).astype(float) -
                                     np.asarray(image[0]['ImagePositionPatient'].value[2]).astype(float))
                    elif t in image[0]:
                        thickness = np.asarray(image[0]['SliceThickness'].value).astype(float)
                    else:
                        thickness = 1

                    self.image_info.at[ii, t] = thickness

                elif t == 'Slices':
                    self.image_info.at[ii, t] = len(image)

                elif t == 'DefaultWindow':
                    if (0x0028, 0x1050) in image[0] and (0x0028, 0x1051) in image[0]:
                        center = image[0].WindowCenter
                        width = image[0].WindowWidth
                        if not isinstance(center, float):
                            center = center[0]

                        if not isinstance(width, float):
                            width = width[0]

                        self.image_info.at[ii, t] = [int(center), int(np.round(width/2))]

                    elif image[0].Modality == 'US':
                        self.image_info.at[ii, t] = [128, 128]

                    else:
                        self.image_info.at[ii, t] = None

                elif t == 'FullWindow':
                    self.image_info.at[ii, t] = None

                elif t == 'ImageMatrix':
                    pass

                elif t == 'SkippedSlice':
                    pass

                elif t == 'ImagePlane':
                    if image[0].Modality in ['US', 'CR', 'DX', 'MG', 'NM', 'XA']:
                        self.image_info.at[ii, t] = 'Axial'
                    else:
                        orientation = image[0]['ImageOrientationPatient'].value
                        x = np.abs(orientation[0]) + np.abs(orientation[3])
                        y = np.abs(orientation[1]) + np.abs(orientation[4])
                        z = np.abs(orientation[2]) + np.abs(orientation[5])

                        if x < y and x < z:
                            self.image_info.at[ii, t] = 'Sagittal'
                        elif y < x and y < z:
                            self.image_info.at[ii, t] = 'Coronal'
                        else:
                            self.image_info.at[ii, t] = 'Axial'

                elif t == 'Unverified':
                    pass

                else:
                    if t in image[0]:
                        self.image_info.at[ii, t] = image[0][t].value

                    else:
                        if t == 'SeriesDate':
                            if 'StudyDate' in image[0]:
                                self.image_info.at[ii, t] = image[0]['StudyDate'].value
                            else:
                                self.image_info.at[ii, t] = '0'

                        elif t == 'SeriesTime':
                            if 'StudyTime' in image[0]:
                                self.image_info.at[ii, t] = image[0]['StudyTime'].value
                            else:
                                self.image_info.at[ii, t] = '00000'

                        elif t == 'SeriesDescription':
                            self.image_info.at[ii, t] = 'None'

                        elif t == 'FrameOfReferenceUID':
                            if 'FrameOfReferenceUID' in image[0]:
                                self.image_info.at[ii, t] = image[0]['FrameOfReferenceUID'].value
                            else:
                                self.image_info.at[ii, t] = generate_uid()

    def find_pixel_spacing(self, image, ii):
        spacing = 'PixelSpacing'
        if image.Modality == 'US':
            if 'SequenceOfUltrasoundRegions' in image:
                if 'PhysicalDeltaX' in image.SequenceOfUltrasoundRegions[0]:
                    self.image_info.at[ii, spacing] = [
                        10 * np.round(image.SequenceOfUltrasoundRegions[0].PhysicalDeltaX, 4),
                        10 * np.round(image.SequenceOfUltrasoundRegions[0].PhysicalDeltaY, 4)]
                else:
                    self.image_info.at[ii, spacing] = [1, 1]
            else:
                self.image_info.at[ii, spacing] = [1, 1]

        elif image.Modality in ['DX', 'XA']:
            self.image_info.at[ii, spacing] = image.ImagerPixelSpacing

        elif 'PixelSpacing' in image:
            self.image_info.at[ii, spacing] = image.PixelSpacing

        elif 'ContributingSourcesSequence' in image:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in image[sequence][0]:
                self.image_info.at[ii, spacing] = image[sequence][0]['DetectorElementSpacing'].value

        elif 'PerFrameFunctionalGroupsSequence' in image:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in image[sequence][0]:
                self.image_info.at[ii, spacing] = image[sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing'].value

        else:
            self.image_info.at[ii, spacing] = [1, 1]

    def convert_images(self):
        """
        Gets the 2D slice for each image and combines them into a 3D array per each image. Uses the RescaleIntercept
        and RescaleSlope to adjust the HU.

        The US is a different story. The image was saved as an RGB value, which also contained like metadata and
        patient information embedded in the image itself. Luckily there was a simple way to get the actual US out, and
        that was using the fact that when all three RGB values are the same thing it corresponds to the image (this
        pulls some additional none image stuff but not nearly as bad). The quickest way I thought would be to find the
        standard deviation of all three values and if it is zero then it is a keeper.

        Sometimes the images are in a shape [1, 10, 512, 512] meaning 10 "slices" by 512x512 array. Not sure why the 1
        is there, so it checks if the shape is 4 and if so it only saves the image as a [10, 512, 512]
        Returns
        -------

        """
        for ii, image in enumerate(self.ds_images):
            image_slices = []
            if self.image_info.at[ii, 'Modality'] in ['CT', 'MR', 'PT', 'MG', 'NM', 'XA', 'CR']:
                for slice_ in image:
                    if (0x0028, 0x1052) in slice_:
                        intercept = slice_.RescaleIntercept
                    else:
                        intercept = 0

                    if (0x0028, 0x1053) in slice_:
                        slope = slice_.RescaleSlope
                    else:
                        slope = 1

                    image_slices.append(((slice_.pixel_array*slope)+intercept).astype('int16'))

            elif self.image_info.at[ii, 'Modality'] == 'DX':
                if (0x2050, 0x0020) in image[0]:
                    if image[0].PresentationLUTShape == 'INVERSE':
                        hold_array = image[0].pixel_array.astype('int16')
                        image_slices.append(16383 - hold_array)
                        self.image_info.at[ii, 'DefaultWindow'][0] = 16383 - self.image_info.at[ii, 'DefaultWindow'][0]
                    else:
                        image_slices.append(image[0].pixel_array.astype('int16'))
                else:
                    image_slices.append(image[0].pixel_array.astype('int16'))

            elif self.image_info.at[ii, 'Modality'] == 'US':
                if len(image) == 1:
                    us_data = image[0].pixel_array
                    if len(us_data.shape) == 3:
                        us_binary = (1 * (np.std(us_data, axis=2) == 0) == 1)
                        image_slices = (us_binary * us_data[:, :, 0]).astype('uint8')

                    else:
                        us_binary = (1 * (np.std(us_data, axis=3) == 0) == 1)
                        image_slices = (us_binary * us_data[:, :, :, 0]).astype('uint8')
                else:
                    print('Need to finish')

            image_hold = np.asarray(image_slices)
            if len(image_hold.shape) > 3:
                self.image_data.append(image_hold[0])
                self.image_info.at[ii, 'Slices'] = image_hold[0].shape[0]
            else:
                self.image_data.append(image_hold)

            image_min = np.min(self.image_data[-1])
            image_max = np.max(self.image_data[-1])
            self.image_info.at[ii, 'FullWindow'] = [image_min, image_max]

    def fix_orientation(self, convert_axial=True):
        """
        Corrects position for orientation fix. I force everything to be FFS so for non-FFS images the corner position
        is incorrect, below corrects for the position using the Pixel Spacing and Orientation Matrix
        Returns
        -------

        """
        for ii, image in enumerate(self.image_data):
            if self.image_info.at[ii, 'Modality'] in ['US', 'CR', 'DX', 'XA']:
                self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
                self.image_info.at[ii, 'Unverified'] = True

            elif self.image_info.at[ii, 'Modality'] == 'NM':
                self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
                self.image_info.at[ii, 'Unverified'] = True

            elif self.image_info.at[ii, 'Modality'] == 'MG':
                self.image_info.at[ii, 'Unverified'] = True
                if self.image_info.at[ii, 'Slices'] == 1:
                    self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                    self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]

                else:
                    if 'SharedFunctionalGroupsSequence' in self.ds_images[ii][0]:
                        sequence = 'SharedFunctionalGroupsSequence'
                        if 'PlaneOrientationSequence' in self.ds_images[ii][0][sequence][0]:
                            self.image_info.at[ii, 'ImageOrientationPatient'] = self.ds_images[ii][0][sequence][0]['PlaneOrientationSequence'][0]['ImageOrientationPatient'].value
                            self.compute_image_matrix(ii)
                        else:
                            self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                            self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
                    else:
                        self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                        self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]

            else:
                if self.image_info.at[ii, 'PatientPosition']:
                    position = self.image_info.at[ii, 'PatientPosition']
                    rows = self.image_info.at[ii, 'Rows']
                    columns = self.image_info.at[ii, 'Columns']
                    spacing = self.image_info.at[ii, 'PixelSpacing']
                    coordinates = self.image_info.at[ii, 'ImagePositionPatient']
                    orientation = np.asarray(self.image_info.at[ii, 'ImageOrientationPatient'])

                    if self.image_info.at[ii, 'ImagePlane'] == 'Axial':
                        self.fix_axial(position, rows, columns, spacing, coordinates, orientation)

                    elif self.image_info.at[ii, 'ImagePlane'] == 'Sagittal':
                        self.fix_sagittal(position, rows, columns, spacing, coordinates, orientation)

                    self.compute_image_matrix(ii)

    def fix_axial(self, position, rows, columns, spacing, coordinates, orientation):
        if position in ['HFDR', 'FFDR']:
            self.image_data[ii] = np.rot90(image, 3, (1, 2))

            new_coordinates = np.double(coordinates[0]) - spacing[0] * (columns - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][0] = new_coordinates
            self.image_info.at[ii, 'ImageOrientationPatient'] = [-orientation[3],
                                                                 -orientation[4],
                                                                 -orientation[5],
                                                                 orientation[0],
                                                                 orientation[1],
                                                                 orientation[2]]
        elif position in ['HFP', 'FFP']:
            self.image_data[ii] = np.rot90(image, 2, (1, 2))

            new_coordinates = np.double(coordinates[0]) - spacing[0] * (columns - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][0] = new_coordinates

            new_coordinates = np.double(coordinates[1]) - spacing[1] * (rows - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][1] = new_coordinates
            self.image_info.at[ii, 'ImageOrientationPatient'] = [-orientation[0],
                                                                 -orientation[1],
                                                                 -orientation[2],
                                                                 -orientation[3],
                                                                 -orientation[4],
                                                                 -orientation[5]]
        elif position in ['HFDL', 'FFDL']:
            self.image_data[ii] = np.rot90(image, 1, (1, 2))

            new_coordinates = np.double(coordinates[1]) - spacing[1] * (rows - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][1] = new_coordinates
            self.image_info.at[ii, 'ImageOrientationPatient'] = [orientation[3],
                                                                 orientation[4],
                                                                 orientation[5],
                                                                 -orientation[0],
                                                                 -orientation[1],
                                                                 -orientation[2]]

        if self.image_info.at[ii, 'ImageOrientationPatient'][0] < 0 or self.image_info.at[ii, 'ImageOrientationPatient'][5] < 0:
            self.image_info.at[ii, 'Unverified'] = True
        self.compute_image_matrix(ii)

    def fix_sagittal(self, position, rows, columns, spacing, coordinates, orientation):
        b = copy.deepcopy(self.image_data[ii])
        b = np.swapaxes(b, 0, 1)
        b = np.swapaxes(b, 1, 2)
        self.image_data[ii] = np.flip(b, axis=0)

    def compute_image_matrix(self, ii):
        """
        Computes the image rotation matrix, often seen in MR images where the image is tilted.


        Returns
        -------

        """
        row_direction = np.array(self.image_info.at[ii, 'ImageOrientationPatient'][:3])
        column_direction = np.array(self.image_info.at[ii, 'ImageOrientationPatient'][3:])
        translation_offset = np.asarray(self.image_info.at[ii, 'ImagePositionPatient'])

        # noinspection PyUnreachableCode
        slice_direction = np.cross(row_direction, column_direction)
        if len(self.ds_images[ii]) > 1:
            first = np.dot(slice_direction, self.ds_images[ii][0].ImagePositionPatient)
            second = np.dot(slice_direction, self.ds_images[ii][1].ImagePositionPatient)
            last = np.dot(slice_direction, self.ds_images[ii][-1].ImagePositionPatient)
            first_last_spacing = np.asarray((last - first) / (self.image_info.at[ii, 'Slices'] - 1))
            if np.abs((second - first) - first_last_spacing) > 0.01:
                self.find_skipped_slices(slice_direction, ii)
                slice_spacing = second - first
            else:
                slice_spacing = np.asarray((last - first) / (self.image_info.at[ii, 'Slices'] - 1))

            self.image_info.at[ii, 'SliceThickness'] = slice_spacing
            if slice_spacing < 0 and self.image_info.at[ii, 'ImagePlane']:
                self.image_info.at[ii, 'ImagePositionPatient'] = self.ds_images[ii][-1].ImagePositionPatient

        mat = np.identity(4, dtype=np.float32)
        mat[0, :3] = row_direction
        mat[1, :3] = column_direction
        mat[2, :3] = slice_direction
        mat[0:3, 3] = -translation_offset

        self.image_info.at[ii, 'ImageMatrix'] = mat

    def separate_contours(self):
        """
        existing_image_info is required if the users only loads a RTSTRUCT file, this is needed to match contours with
        the image they correspond to.

        It is pretty gross after that. For a given ROI each contour is read-in, matched with their image, then combined
        all the slices of each contour into their own numpy array.

        Returns
        -------

        """
        info = self.image_info
        if self.existing_image_info is not None:
            if len(list(info.index)) > 0:
                print('fix')
            else:
                info = self.existing_image_info

        index_list = list(info.index)
        for ii in range(len(info.index)):
            img_sop = info.at[index_list[ii], 'SOPInstanceUID']
            img_series = info.at[index_list[ii], 'SeriesInstanceUID']

            image_contour_list = []
            roi_names = []
            roi_filepaths = []
            for jj in range(len(self.rt_df.index)):
                if img_series == self.rt_df.at[jj, 'SeriesInstanceUID'] and self.rt_df.at[jj, 'RoiSOP'][0] in img_sop:
                    roi_sequence = self.ds_modality['RTSTRUCT'][jj].ROIContourSequence
                    for kk, sequence in enumerate(roi_sequence):
                        contour_list = []
                        if not self.only_load_roi_names or self.rt_df.RoiNames[jj][kk] in self.only_load_roi_names:
                            for c in sequence.ContourSequence:
                                if int(c.NumberOfContourPoints) > 1:
                                    contour_hold = np.round(np.array(c['ContourData'].value), 3)
                                    contour = contour_hold.reshape(int(len(contour_hold) / 3), 3)
                                    contour_list.append(contour)

                            if len(contour_list) > 0:
                                image_contour_list.append(contour_list)
                                roi_filepaths.append(self.rt_df.at[jj, 'FilePath'])
                                roi_names.append(self.rt_df.RoiNames[jj][kk])

            if len(roi_names) > 0:
                if not np.isnan(self.image_info.at[ii, 'SkippedSlice']):
                    image_contour_list = self.calculate_skipped_contours(ii, image_contour_list)
                self.roi_contour.append(image_contour_list)
                self.roi_info.at[ii, 'FilePath'] = roi_filepaths
                self.roi_info.at[ii, 'RoiNames'] = roi_names
            else:
                self.roi_contour.append([])
                self.roi_info.at[ii, 'FilePath'] = None
                self.roi_info.at[ii, 'RoiNames'] = None

    def find_skipped_slices(self, slice_direction, ii):
        image = self.ds_images[ii]
        base_spacing = None
        for jj in range(len(image) - 1):
            position_1 = np.dot(slice_direction, image[jj].ImagePositionPatient)
            position_2 = np.dot(slice_direction, image[jj + 1].ImagePositionPatient)
            if jj == 0:
                base_spacing = position_2 - position_1
            if jj > 0 and np.abs(base_spacing - (position_2 - position_1)) > 0.01:
                self.image_info.at[ii, 'SkippedSlice'] = jj + 1
                self.image_info.at[ii, 'Slices'] = len(image) + 1

                hold_data = copy.deepcopy(self.image_data[ii])
                interpolate_slice = np.mean(self.image_data[ii][jj:jj + 2, :, :], axis=0).astype(np.int16)
                self.image_data[ii] = np.insert(hold_data,
                                                self.image_info.at[ii, 'SkippedSlice'],
                                                interpolate_slice,
                                                axis=0)

    def calculate_skipped_contours(self, ii, image_contours):
        thickness = self.image_info.at[ii, 'SliceThickness']
        skipped = self.image_info.at[ii, 'SkippedSlice']
        z_positions = np.asarray([self.ds_images[ii][skipped - 1].ImagePositionPatient[2],
                                  self.ds_images[ii][skipped].ImagePositionPatient[2]])

        for jj, roi_contour in enumerate(image_contours):
            if len(roi_contour) > 1:
                for kk in range(len(roi_contour) - 1):
                    position_1 = roi_contour[kk][0][2]
                    position_2 = roi_contour[kk + 1][0][2]
                    if position_1 == z_positions[0] and position_2 == z_positions[1]:
                        roi_copy = copy.deepcopy(roi_contour)
                        roi_copy.insert(jj + 1, copy.deepcopy(roi_copy[kk]))
                        roi_copy[jj + 1][:, 2] = position_1 + thickness
                        self.image_info.at[ii, 'SOPInstanceUID'].insert(jj + 1, generate_uid())
                        image_contours[jj] = roi_copy

        return image_contours

    def convert_to_axial(self):
        for ii in range(len(self.ds_images)):
            if self.image_info.at[ii, 'ImagePlane'] == 'Coronal':
                b = copy.deepcopy(self.image_data[ii])
                self.image_data[ii] = np.flip(b, axis=0)

            elif self.image_info.at[ii, 'ImagePlane'] == 'Sagittal':
                b = copy.deepcopy(self.image_data[ii])
                b = np.swapaxes(b, 0, 1)
                b = np.swapaxes(b, 1, 2)
                self.image_data[ii] = np.flip(b, axis=0)

        print(1)


class Image3d(object):
    """
    All images and their tags are oriented in the x, y, z orientation (Sagittal, Coronal, Axial). Only variable
    "plane" is left in the original image orientation to illustrate the original main view.
    """
    def __init__(self, image_set, only_tags):
        self.image_set = image_set
        self.only_tags = only_tags

        self.unverified = None
        self.base_position = None
        self.skipped_slice = None

        if not self.only_tags:
            self.array = self.compute_array()
        self.plane = self.compute_plane()
        self.spacing = self.compute_spacing()
        self.orientation = self.compute_orientation()
        self.origin = self.compute_origin()
        self.image_matrix = self.compute_image_matrix()

        # if self.plane != 'Axial':
        #     self.axial_correction()

    def compute_array(self):
        image_slices = []
        for slice_ in self.image_set:
            if (0x0028, 0x1052) in slice_:
                intercept = slice_.RescaleIntercept
            else:
                intercept = 0

            if (0x0028, 0x1053) in slice_:
                slope = slice_.RescaleSlope
            else:
                slope = 1

            image_slices.append(((slice_.pixel_array*slope)+intercept).astype('int16'))

        image_hold = np.asarray(image_slices)
        if len(image_hold.shape) > 3:
            return image_hold[0]
        else:
            return image_hold

    def compute_plane(self):
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

    def compute_spacing(self):
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

    def compute_orientation(self):
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

    def compute_origin(self):
        origin = np.asarray(self.image_set[0]['ImagePositionPatient'].value)
        if 'PatientPosition' in self.image_set[0]:
            self.base_position = self.image_set[0]['PatientPosition'].value
            if self.base_position in ['HFDR', 'FFDR']:
                if self.only_tags:
                    self.array = np.rot90(self.array, 3, (1, 2))

                origin[0] = np.double(origin[0]) - self.spacing[0] * (self.image_set[0]['Columns'] - 1)
                self.orientation = [-self.orientation[3], -self.orientation[4], -self.orientation[5],
                                    self.orientation[0], self.orientation[1], self.orientation[2]]

            elif self.base_position in ['HFP', 'FFP']:
                if self.only_tags:
                    self.array = np.rot90(self.array, 2, (1, 2))

                origin[0] = np.double(origin[0]) - self.spacing[0] * (self.image_set[0]['Columns'] - 1)
                origin[1] = np.double(origin[1]) - self.spacing[1] * (self.image_set[0]['Rows'] - 1)
                self.orientation = -np.asarray(self.orientation)

            elif self.base_position in ['HFDL', 'FFDL']:
                if self.only_tags:
                    self.array = np.rot90(self.array, 1, (1, 2))

                origin[1] = np.double(origin[1]) - self.spacing[1] * (self.image_set[0]['Rows'] - 1)
                self.orientation = [self.orientation[3], self.orientation[4], self.orientation[5],
                                    -self.orientation[0], -self.orientation[1], -self.orientation[2]]

        return origin

    def compute_image_matrix(self):
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
                    self.find_skipped_slices(slice_direction)
                slice_spacing = second - first
            else:
                slice_spacing = np.asarray((last - first) / (len(self.image_set) - 1))

            self.spacing[2] = slice_spacing

        mat = np.identity(4, dtype=np.float32)
        mat[0, :3] = row_direction
        mat[1, :3] = column_direction
        mat[2, :3] = slice_direction
        mat[0:3, 3] = -self.origin

        return mat

    def find_skipped_slices(self, slice_direction):
        base_spacing = None
        for ii in range(len(self.image_set) - 1):
            position_1 = np.dot(slice_direction, self.image_set[ii].ImagePositionPatient)
            position_2 = np.dot(slice_direction, self.image_set[ii + 1].ImagePositionPatient)
            if ii == 0:
                base_spacing = position_2 - position_1
            if ii > 0 and np.abs(base_spacing - (position_2 - position_1)) > 0.01:
                self.skipped_slice = ii + 1

                hold_data = copy.deepcopy(self.array)
                interpolate_slice = np.mean(self.array[ii:ii + 2, :, :], axis=0).astype(np.int16)
                self.array = np.insert(hold_data, self.skipped_slice, interpolate_slice, axis=0)

    def axial_correction(self):
        if self.plane == 'Sagittal':
            array_hold = copy.deepcopy(self.array)
            array_hold = np.swapaxes(array_hold, 0, 1)
            array_hold = np.swapaxes(array_hold, 1, 2)
            self.array = np.flip(array_hold, axis=0)

            self.orientation[0:2] = 1 - self.orientation[0:2]
            self.orientation[4] = 1 - self.orientation[4]
            self.orientation[5] = 1 + self.orientation[5]

        elif self.plane == 'Coronal':
            array_hold = copy.deepcopy(self.array)
            self.array = np.flip(array_hold, axis=0)

            self.orientation[4] = 1 - self.orientation[4]
            self.orientation[5] = 1 + self.orientation[5]
