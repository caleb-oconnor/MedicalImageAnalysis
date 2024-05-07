"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:
    This is a "supposed" to be a multi data medical imagery reader. Currently, only works for dicom file types. Reads in
    imagery data and converts them to separate 3D numpy arrays. Also, will read in any associated ROI contours.

    Using the "DicomReader" class the user can input a folder directory and output the images in numpy arrays along with
    their respective rois (if any). The data does not need to be organized inside folder directory, the reader will
    sort the images appropriately. It does not separate different patients if they exist in the same folder.

    The reader currently imports 9 different modalites and RTSTRUCT files. The accepted modalites are:
        CT
        MR
        US
        PT
        MG
        DX
        NM
        XA
        CR

    The CT and MR modalities have been tested extensively, along with their respective ROIs. The other 7 modalities
    have been tested but only on a few datasets a piece. For RTSTRUCTS, only those referencing CT and MR have been
    tested.
"""

import os
import time
import threading

import numpy as np
import pandas as pd
import pydicom as dicom


def thread_process_dicom(path):
    try:
        datasets = dicom.dcmread(str(path))
    except:
        datasets = []

    return datasets


def thread_process_contour(c):
    contour_hold = np.round(np.array(c['ContourData'].value), 3)
    contour = contour_hold.reshape(int(len(contour_hold) / 3), 3)
    return contour


class DicomReader:
    def __init__(self, dicom_files, existing_image_info=None, only_load_roi_names=None):
        """
        This reader splits dicom images/rtstructs using the SeriesInstanceUID/AcquisitionNumber. It can read any of
        these modalities:
            CT
            MR
            US
            PT
            MG
            DX
            NM
            XA
            CR
            RTSTRUCT

        The images are combined into 3D numpy arrays. CT/MR/PT images are corrected to be Head-First-Supine (HFS) if not
        already. The ROIs are combined into a numpy array list of list with each inner list being the points on a given
        slice.


        Parameters
        ----------
        dicom_files - list of dicom files created using file_parsar function
        existing_image_info - either None or a dataframe containing image information, this would be required when only
                loading an RTSTRUCT it needs to reference to original image. The format of the dataframe is the same as
                the variable image_info below.
        """
        self.dicom_files = dicom_files
        self.existing_image_info = existing_image_info
        self.only_load_roi_names = only_load_roi_names

        self.ds = []
        self.ds_images = []
        self.ds_dictionary = dict.fromkeys(['CT', 'MR', 'PT', 'US', 'DX', 'MG', 'NM', 'XA', 'CR', 'RTSTRUCT'])
        self.rt_df = pd.DataFrame(columns=['FilePath', 'SeriesInstanceUID', 'RoiSOP', 'RoiNames'])

        keep_tags = ['FilePath', 'SOPInstanceUID', 'PatientID', 'PatientName', 'Modality',
                     'SeriesDescription', 'SeriesDate', 'SeriesTime', 'SeriesInstanceUID', 'SeriesNumber',
                     'AcquisitionNumber', 'SliceThickness', 'PixelSpacing', 'Rows', 'Columns', 'PatientPosition',
                     'ImagePositionPatient', 'ImageOrientationPatient', 'Slices', 'DefaultWindow', 'FullWindow']
        self.image_info = pd.DataFrame(columns=keep_tags)
        self.image_data = []

        self.roi_info = pd.DataFrame(columns=['FilePath', 'RoiNames'])
        self.roi_contour = []

        self.contours = []

    def add_dicom_extension(self):
        """
        Will add .dcm extension to any file inside self.dicom_files that doesn't have an extension
        Returns
        -------

        """
        for ii, name in enumerate(self.dicom_files):
            a, b = os.path.splitext(name)
            if not b:
                self.dicom_files[ii] = name + '.dcm'

    def load_dicom(self, display_time=True):
        """
        Runs through all the base functions required to load in images/rois.

        Parameters
        ----------
        display_time - True if user wants to print load time

        Returns
        -------

        """
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
        """
        Uses the threading to read in the data.

        self.ds -> contains tag/data from pydicom read-in

        Returns
        -------

        """
        threads = []

        def read_file_thread(file_path):
            self.ds.append(thread_process_dicom(file_path))

        for file_path in self.dicom_files:
            thread = threading.Thread(target=read_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def separate_modalities(self):
        """
        Currently, separates the files into 10 different modalities. Files with a different modality are not kept.
        Certain tags are required depending on the modality, if those tags don't exist for its respective modality
        then it is not kept.

        Returns
        -------

        """
        req = {'CT': ['SeriesInstanceUID', 'AcquisitionNumber', 'ImagePositionPatient', 'SliceThickness', 'PixelData',
                      'FrameOfReferenceUID'],
               'MR': ['SeriesInstanceUID', 'AcquisitionNumber', 'ImagePositionPatient', 'SliceThickness', 'PixelData',
                      'FrameOfReferenceUID'],
               'PT': ['SeriesInstanceUID', 'ImagePositionPatient', 'SliceThickness', 'PixelData',
                      'FrameOfReferenceUID'],
               'US': ['SeriesInstanceUID', 'PixelData'],
               'DX': ['SeriesInstanceUID', 'PixelData'],
               'MG': ['SeriesInstanceUID', 'PixelData'],
               'NM': ['SeriesInstanceUID', 'PixelData'],
               'XA': ['SeriesInstanceUID', 'NumberOfFrames', 'PixelData'],
               'CR': ['SeriesInstanceUID', 'PixelData'],
               'RTSTRUCT': ['SeriesInstanceUID']}

        for modality in list(self.ds_dictionary.keys()):
            ds_modality = [d for d in self.ds if d['Modality'].value == modality]
            if modality == 'CT' or modality == 'MR' or modality == 'PT':
                self.ds_dictionary[modality] = [ds_mod for ds_mod in ds_modality if
                                                len([r for r in req[modality] if r in ds_mod]) == len(req[modality]) and
                                                ds_mod['SliceThickness'].value]
            else:
                self.ds_dictionary[modality] = [ds_mod for ds_mod in ds_modality if
                                                len([r for r in req[modality] if r in ds_mod]) == len(req[modality])]

    def separate_images(self):
        """
        This is used to separate the different modalities into images. Each modality has specific requirements to exist
        in the tags:
            CT/MR = SeriesInstanceUID, SliceThickness, AcquisitionNumber, ImagePositionPatient
            US = SeriesInstanceUID

        CT - the 4 main tags are pulled into a numpy array. There is a quick fix because sometimes AcquisitionNumber is
             empty, if so then '1001' is inserted in its place. All the unique combinations are found using series
             instance uid, slice thickness, acquisition number. The unique combinations are used in a for loop and
             all slices that match that criteria are selected. Those slices are then sorted by image position patient.
        MR/PT - same as CT
        US/DX/MG/NM/XA/CR - The images are just sorted using series instance uid.


        Note: slices thickness is needed because some scans are saved with the first slice being a single slice of
              coronal plane, then the rest of the slices are in the axial plane. I think it is called a scout scan,
              basically they have a single coronal slice to show the area that will be covered, then it is followed
              by your standard axial plane view. Anyway, I separate out that slice using the slice thickness because
              it will be different from the axial.

        Returns
        -------

        """
        standard_modalities = ['CT', 'MR', 'PT']
        for mod in standard_modalities:
            if len(self.ds_dictionary[mod]) > 0:
                if mod in ['CT', 'MR']:
                    sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['SliceThickness'].value,
                                                img['AcquisitionNumber'].value, img['ImagePositionPatient'].value[0],
                                                img['ImagePositionPatient'].value[1],
                                                img['ImagePositionPatient'].value[2], ii]
                                               for ii, img in enumerate(self.ds_dictionary[mod])])
                else:
                    sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['SliceThickness'].value,
                                                None, img['ImagePositionPatient'].value[0],
                                                img['ImagePositionPatient'].value[1],
                                                img['ImagePositionPatient'].value[2], ii]
                                               for ii, img in enumerate(self.ds_dictionary[mod])])

                sorting_tags_fix = np.asarray([[s[0], s[1], '1001', s[3], s[4], s[5], s[6]] if s[2] is None else s
                                               for s in sorting_tags])
                unique_tags = np.unique(sorting_tags_fix[:, 0:3].astype(str), axis=0)

                for tags in unique_tags:
                    unsorted_values = sorting_tags_fix[np.where((sorting_tags_fix[:, 0] == tags[0]) &
                                                                (sorting_tags_fix[:, 1] == tags[1]) &
                                                                (sorting_tags_fix[:, 2] == tags[2]))]

                    # noinspection PyBroadException
                    try:
                        orientation = np.round(self.ds_dictionary[mod][0].ImageOrientationPatient)
                        if orientation[0] == 1 and orientation[4] == 1:
                            orient_idx = 5
                        elif orientation[0] == 0 and orientation[3] == 0:
                            orient_idx = 3
                        elif orientation[1] == 0 and orientation[4] == 0:
                            orient_idx = 4
                        else:
                            orient_idx = 5
                    except:
                        orient_idx = 5

                    sorted_values = unsorted_values[np.argsort(unsorted_values[:, orient_idx].astype('float'))[::-1]]
                    self.ds_images.append([self.ds_dictionary[mod][int(idx[6])] for idx in sorted_values])

        nonstandard_modalities = ['US', 'DX', 'MG', 'NM', 'XA', 'CR']
        for mod in nonstandard_modalities:
            if len(self.ds_dictionary[mod]) > 0:
                sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, ii]
                                           for ii, img in enumerate(self.ds_dictionary[mod])])
                unique_tags = np.unique(sorting_tags[:, 0], axis=0)
                for tags in unique_tags:
                    unsorted_values = sorting_tags[np.where(sorting_tags[:, 0] == tags)]
                    sorted_values = unsorted_values[np.argsort(unsorted_values[:, 0])]
                    self.ds_images.append([self.ds_dictionary[mod][int(idx[1])] for idx in sorted_values])

    def separate_rt_images(self):
        """
        Loops through all RTSTRUCT files found. Some required information that will be used later in making the contours
        is pulled:
            SeriesInstanceUID
            RoiNames
            RoiSOP - this will be used to determine what slice the contour exist on
        Returns
        -------

        """
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
        """
        Important tags for each image that I use in DRAGON:
            ['FilePath', 'SOPInstanceUID', 'PatientID', 'PatientName', 'Modality',
             'SeriesDescription', 'SeriesDate', 'SeriesTime', 'SeriesInstanceUID', 'SeriesNumber',
             'AcquisitionNumber', 'SliceThickness', 'PixelSpacing', 'Rows', 'Columns', 'ImagePositionPatient',
             'Slices', 'DefaultWindow']

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
                    if image[0].Modality == 'US':
                        self.image_info.at[ii, t] = [
                            10 * np.round(image[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaX, 4),
                            10 * np.round(image[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaY, 4)]
                    elif image[0].Modality in ['DX', 'XA']:
                        self.image_info.at[ii, t] = image[0].ImagerPixelSpacing
                    else:
                        self.image_info.at[ii, t] = image[0].PixelSpacing

                elif t == 'ImagePositionPatient':
                    if image[0].Modality in ['US', 'CR', 'DX', 'MG', 'NM', 'XA']:
                        self.image_info.at[ii, t] = [0, 0, 0]
                    else:
                        self.image_info.at[ii, t] = image[-1].ImagePositionPatient

                elif t == 'SliceThickness':
                    if len(image) > 1:
                        thickness = (np.asarray(image[0]['ImagePositionPatient'].value[2]).astype(float) -
                                     np.asarray(image[1]['ImagePositionPatient'].value[2]).astype(float))
                    else:
                        thickness = 1

                    self.image_info.at[ii, t] = thickness

                elif t == 'Slices':
                    self.image_info.at[ii, t] = len(image)

                elif t == 'DefaultWindow':
                    if (0x0028, 0x1050) in image[0] and (0x0028, 0x1051) in image[0]:
                        if image[0].Modality == 'DX':
                            self.image_info.at[ii, t] = [int(image[0].WindowCenter),
                                                         int(np.round(image[0].WindowWidth/2))]
                        else:
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
                else:
                    if t in image[0]:
                        self.image_info.at[ii, t] = image[0][t].value
                    else:
                        self.image_info.at[ii, t] = None

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
                for slice_ in reversed(image):
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
                    else:
                        image_slices.append(image[0].pixel_array.astype('int16'))
                else:
                    image_slices.append(image[0].pixel_array.astype('int16'))

            elif self.image_info.at[ii, 'Modality'] == 'US':
                if len(image) == 1:
                    us_data = image[0].pixel_array
                    us_binary = (1 * (np.std(us_data, axis=3) == 0) == 1)
                    image_slices = (us_binary * us_data[:, :, :, 0]).astype('uint8')
                else:
                    print('Need to finish')
                self.image_info.at[ii, 'Slices'] = len(image_slices)

            image_hold = np.asarray(image_slices)
            if len(image_hold.shape) > 3:
                self.image_data.append(image_hold[0])
            else:
                self.image_data.append(image_hold)

            image_min = np.min(self.image_data[-1])
            image_max = np.max(self.image_data[-1])
            self.image_info.at[ii, 'FullWindow'] = [image_min, image_max]

    def fix_orientation(self):
        for ii, image in enumerate(self.image_data):
            if self.image_info.at[ii, 'PatientPosition']:
                position = self.image_info.at[ii, 'PatientPosition']
                rows = self.image_info.at[ii, 'Rows']
                columns = self.image_info.at[ii, 'Columns']
                spacing = self.image_info.at[ii, 'PixelSpacing']
                coordinates = self.image_info.at[ii, 'ImagePositionPatient']

                if position in ['HFDR', 'FFDR']:
                    self.image_data[ii] = np.rot90(image, 3, (1, 2))

                    new_coordinates = np.double(coordinates[0]) - spacing[0] * (columns - 1)
                    self.image_info.at[ii, 'ImagePositionPatient'][0] = new_coordinates
                elif position in ['HFP', 'FFP']:
                    self.image_data[ii] = np.rot90(image, 2, (1, 2))

                    new_coordinates = np.double(coordinates[0]) - spacing[0] * (columns - 1)
                    self.image_info.at[ii, 'ImagePositionPatient'][0] = new_coordinates

                    new_coordinates = np.double(coordinates[1]) - spacing[1] * (rows - 1)
                    self.image_info.at[ii, 'ImagePositionPatient'][1] = new_coordinates
                elif position in ['HFDL', 'FFDL']:
                    self.image_data[ii] = np.rot90(image, 1, (1, 2))

                    new_coordinates = np.double(coordinates[1]) - spacing[1] * (rows - 1)
                    self.image_info.at[ii, 'ImagePositionPatient'][1] = new_coordinates

    def separate_contours(self):
        """
        existing_image_info is required if the users only loads a RTSTRUCT file, this is needed to match contours with
        the image they correspond to.

        It is pretty gross after that. For a given ROI each contour is read-in, matched with their image, then combined
        all the slices of each contour into there own numpy array.

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
                self.roi_contour.append(image_contour_list)
                self.roi_info.at[ii, 'FilePath'] = roi_filepaths
                self.roi_info.at[ii, 'RoiNames'] = roi_names
            else:
                self.roi_contour.append([])
                self.roi_info.at[ii, 'FilePath'] = None
                self.roi_info.at[ii, 'RoiNames'] = None

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


if __name__ == '__main__':
    from parsar import file_parsar

    path = r'C:\Users\csoconnor\OneDrive - Inside MD Anderson\Documents\Breast_Pathology\759335'
    exclude_files = []
    existing_image_info = None
    file_dictionary = file_parsar(path, exclude_files)
    dicom_reader = DicomReader(file_dictionary['Dicom'], existing_image_info)
    dicom_reader.load_dicom()

    # pass
