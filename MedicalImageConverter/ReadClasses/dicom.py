
import os
import copy
import time
import gdcm
import threading

import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import generate_uid


def add_dicom_extension(self, dicom_files):
    for ii, name in enumerate(dicom_files):
        a, b = os.path.splitext(name)
        if not b:
            dicom_files[ii] = name + '.dcm'

    return dicom_files


def thread_process_dicom(path):
    try:
        datasets = pydicom.dcmread(str(path))
    except:
        datasets = []

    return datasets


class DicomReader:
    def __init__(self, dicom_files, existing_image_info=None, only_load_roi_names=None):
        """

        Parameters
        ----------
        dicom_files - list of all the dicom paths
        existing_image_info - dataframe of image_info same as structure below (for when only loading RTSTRUCTS)
        only_load_roi_names - list of Roi names that will only be uploaded (so total segementator won't load all 100
                              rois)
        """
        self.dicom_files = dicom_files
        self.existing_image_info = existing_image_info
        self.only_load_roi_names = only_load_roi_names

        self.ds = []
        self.ds_images = []
        self.ds_dictionary = dict.fromkeys(['CT', 'MR', 'PT', 'US', 'DX', 'MG', 'NM', 'XA', 'CR', 'RTSTRUCT'])
        self.rt_df = pd.DataFrame(columns=['FilePath', 'SeriesInstanceUID', 'RoiSOP', 'RoiNames'])

        self.image_info = pd.DataFrame(columns=['FilePath', 'SOPInstanceUID', 'PatientPosition'])
        self.image_data = []

        self.roi_info = pd.DataFrame(columns=['FilePath', 'RoiNames'])
        self.roi_contour = []
        self.roi_pixel_position = []

        self.contours = []

        self.image = []
        self.roi = []
        self.poi = []
        self.rigid = []
        self.deformable = []
        self.dose = []

    def load_dicom(self, display_time=True):
        t1 = time.time()
        self.read()
        self.separate_modalities()
        self.separate_images()
        self.separate_rt_images()
        self.standard_useful_tags()
        self.convert_images()
        self.fix_orientation()
        self.separate_contours()
        t2 = time.time()
        if display_time:
            print('Dicom Read Time: ', t2 - t1)

    def read(self):
        threads = []

        def read_file_thread(file_path):
            self.ds.append(thread_process_dicom(file_path))

        for file_path in self.dicom_files:
            thread = threading.Thread(target=read_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for modality in list(self.ds_dictionary.keys()):
            ds_modality = [d for d in self.ds if d['Modality'].value == modality]
            self.ds_dictionary[modality] = [ds_mod for ds_mod in ds_modality]

    def separate_images(self):
        for modality in list(self.ds_dictionary.keys()):
            if len(self.ds_dictionary[modality]) > 0 and modality not in ['RTSTRUCT', 'US', 'DX']:
                sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['AcquisitionNumber'].value]
                                           if 'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None
                                           else [img['SeriesInstanceUID'].value, 1]
                                           for img in self.ds_dictionary[modality]])

                unique_tags = np.unique(sorting_tags, axis=0)
                for tag in unique_tags:
                    sorted_idx = np.where((sorting_tags[:, 0] == tag[0]) & (sorting_tags[:, 1] == tag[1]))
                    image_tags = [self.ds_dictionary[modality][idx] for idx in sorted_idx[0]]

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

                        self.ds_images.append([image_tags[idx] for idx in slice_idx])

                    else:
                        self.ds_images.append(image_tags)

            elif len(self.ds_dictionary[modality]) > 0 and modality in ['US', 'DX']:
                for image in self.ds_dictionary[modality]:
                    self.ds_images.append([image])

    def separate_rt_images(self):
        for ii, rt_ds in enumerate(self.ds_dictionary['RTSTRUCT']):
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
                    roi_sequence = self.ds_dictionary['RTSTRUCT'][jj].ROIContourSequence
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
        #
        # b = np.swapaxes(self.image_data[2], 0, 1)
        # b = np.swapaxes(b, 1, 2)
        # c = np.flip(b, axis=0)
        print(1)

    def get_image_info(self):
        return self.image_info

    def get_image_data(self):
        return self.image_data

    def get_roi_contour(self):
        return self.roi_contour

    def get_roi_info(self):
        return self.roi_info

    def get_ds_images(self):
        return self.ds_images
