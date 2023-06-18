"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:
    This is a "supposed" to be a multi data type medical imagery reader. Currently, it just reads dicom images of CT,
    MR, US, and RTSTRUCTs. The secondary requirement is that the images are in orientation of [1, 0, 0, 0, 1 ,0].
    This is also the reader that is used for DRAGON.

    Using the "DicomReader" class the user can input a folder directory and output the images in numpy arrays along with
    their respective rois (if any). The data does not need to be organized inside folder directory, the reader will
    sort the images appropriately.

    Using the "RayStationCorrection" class follows after the "DicomReader" class. This is used to correct the dicom tags
    in a way that is readable into RayStation. It will also move the dicoms into new image folders and if duplicate
    SeriesInstanceUIDS are present it will assign new UIDs.

Code Overview:
    -

Requirements:
    -
"""


import os
import time
import psutil

import gdcm  # python-gdcm
import numpy as np
import pandas as pd
import pydicom as dicom
import SimpleITK as sitk

from multiprocessing import Pool
from pydicom.uid import generate_uid


def multi_process_dicom(path):
    try:
        datasets = dicom.dcmread(str(path))
    except:
        datasets = []

    return datasets, path


class RayStationCorrection:
    def __init__(self, ds_images, file_info, export_path):
        self.ds_images = ds_images
        self.file_info = file_info
        self.export_path = export_path

        self.original_series = [ds[0]['SeriesInstanceUID'].value for ds in self.ds_images]
        self.series_update = []
        self.frame_reference_update = []

        self.ptid = None
        self.name = None
        self.ds_date = None
        self.ds_time = None

    def get_series_update(self):
        unique_series = np.unique(self.original_series)

        for s in unique_series:
            series_idx = [ii for ii, value in enumerate(self.original_series) if value == s]
            if len(series_idx) == 1:
                self.series_update.append([0, self.original_series[series_idx[0]]])
            else:
                for jj, idx in enumerate(series_idx):
                    if jj == 0:
                        self.series_update.append([0, self.original_series[idx]])
                    else:
                        self.series_update.append([1, generate_uid()])

    def get_frame_of_reference_update(self):
        frame = [ds[0]['FrameOfReferenceUID'].value for ds in self.ds_images]

        for for_img in frame:
            if len(for_img) == 1:
                if for_img[0] != '':
                    self.frame_reference_update.append([0, for_img])
                else:
                    self.frame_reference_update.append([1, generate_uid()])
            else:
                self.frame_reference_update.append([1, for_img])

    def update_ptid_and_name(self, ptid, name):
        self.ptid = ptid
        self.name = name

    def update_date(self, ds_date):
        self.ds_date = ds_date

    def update_time(self, ds_time):
        self.ds_time = ds_time

    def apply_correction(self):
        for ii in range(len(self.ds_images)):

            if self.series_update[ii][0] == 1:
                for jj in range(len(self.ds_images[ii])):
                    self.ds_images[ii][jj]['SeriesInstanceUID'].value = self.series_update[ii][1]

            if self.frame_reference_update[ii][0] == 1:
                for jj in range(len(self.ds_images[ii])):
                    self.ds_images[ii][jj]['FrameOfReferenceUID'].value = self.frame_reference_update[ii][1]

            if self.ptid:
                for jj in range(len(self.ds_images[ii])):
                    self.ds_images[ii][jj]['PatientID'].value = self.ptid

            if self.name:
                for jj in range(len(self.ds_images[ii])):
                    self.ds_images[ii][jj]['PatientName'].value = self.name

            if self.ds_date:
                for jj in range(len(self.ds_images[ii])):
                    self.ds_images[ii][jj]['StudyDate'].value = self.ds_date

            if self.ds_time:
                for jj in range(len(self.ds_images[ii])):
                    self.ds_images[ii][jj]['StudyTime'].value = self.ds_time

    def save_data(self):
        ptid_path, ptid_count = self.get_ptid_path()
        img_path = self.get_img_path(ptid_path, ptid_count)
        change_idx = [x[0] for x in self.series_update]
        if 1 in change_idx:
            self.save_new_uid(ptid_path, img_path)

        for ii, ds_img in enumerate(self.ds_images):

            try:
                unique_padding = np.unique([ds['PixelPaddingValue'].value for ds in ds_img])
                padding = unique_padding[0]
            except:
                padding = None

            for jj, ds_file in enumerate(ds_img):
                if (0x0008, 0x1032) in ds_file:
                    del ds_file[0x0008, 0x1032]
                if (0x0012, 0x0063) in ds_file:
                    del ds_file[0x0012, 0x0063]
                if (0x0012, 0x0064) in ds_file:
                    del ds_file[0x0012, 0x0064]
                if (0x0028, 0x0120) in ds_file and padding:
                    ds_file[0x0028, 0x0120].value = int(padding)

                ds_file.save_as(os.path.join(img_path[ii], self.file_info[ii][jj][0].split('\\')[-1]))

    def get_ptid_path(self):
        save_path = os.path.join(self.export_path, 'Corrections')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        ptid = self.ds_images[0][0]['PatientID'].value
        ptid_path = os.path.join(save_path, ptid)
        if not os.path.exists(ptid_path):
            os.mkdir(ptid_path)
            ptid_count = 1
        else:
            ptid_count = len([x[0] for x in os.walk(ptid_path)])

        return ptid_path, ptid_count

    def get_img_path(self, ptid_path, ptid_count):
        img_path = []
        for ii, ds_img in enumerate(self.ds_images):
            modality = ds_img[0]['Modality'].value

            if ii + ptid_count < 10:
                ct_number = '0' + str(ii+ptid_count)
            else:
                ct_number = str(ii+ptid_count)

            try:
                thickness = np.round(ds_img[0]['SliceThickness'].value,3)
            except:
                thickness = 'None'

            try:
                orient = ds_img[0]['ImageOrientationPatient'].value
                if orient == [1, 0, 0, 0, 1, 0]:
                    view = 'axial'
                else:
                    view = 'nonaxial'
            except:
                view = 'nonaxial'

            img_path.append(os.path.join(ptid_path,
                            modality + '_' + ct_number + '_' + str(thickness) + 'mm' + '_' + view))
            os.mkdir(img_path[-1])

        return img_path

    def save_new_uid(self, ptid_path, img_path):
        file = os.path.join(ptid_path, 'series_uid_changes.txt')
        if os.path.isfile(file):
            read_type = "a"
        else:
            read_type = "w"

        with open(file, read_type) as f:
            for ii, new_uid in enumerate(self.series_update):
                if new_uid[0] == 1:
                    f.write(img_path[ii].split('\\')[-1] + '\t' +
                            self.original_series[ii] + '\t' +
                            new_uid[1] + '\n')
        f.close()


class DicomReader:
    def __init__(self, dicom_files, existing_image_info, existing_file_info):
        self.dicom_files = dicom_files
        self.existing_image_info = existing_image_info
        self.existing_file_info = existing_file_info

        self.ds = []
        self.ds_images = []
        self.ds_dictionary = dict.fromkeys(['CT', 'MR', 'US', 'RTSTRUCT'])
        self.rt_df = pd.DataFrame(columns=['FilePath', 'SeriesInstanceUID', 'RoiNames', 'RoiSOP', 'ContourData'])

        self.image_data = []
        self.file_info = []

        keep_tags = ['PatientID', 'PatientName', 'Modality', 'SeriesDescription', 'SeriesDate', 'SeriesTime',
                     'SeriesInstanceUID', 'SeriesNumber', 'AcquisitionNumber', 'SliceThickness', 'PixelSpacing',
                     'Rows', 'Columns', 'ImagePositionPatient', 'Slices']
        self.image_info = pd.DataFrame(columns=keep_tags)

        self.roi_info = []
        self.roi_data = []

    def load_dicom(self):
        t1 = time.time()
        self.read()
        self.separate_modalities()
        self.combine_images()
        self.separate_rt_images()
        self.standard_useful_tags()
        self.convert_images()
        self.collect_file_info()
        self.rt_matching_new()
        self.roi_contour_fix()
        t2 = time.time()
        print('Dicom Read Time: ', t2-t1)

        print(1)

    def read(self):
        """
        Uses the multiprocessing module to read in the data. The dicom files are sent to "multi_process_dicom"
        function outside this class, which returns the read-in dicom tags/data. The tags/data are only kept if there
        is a Modality tag.

        self.ds -> contains tag/data from pydicom read-in

        Returns
        -------

        """
        p = Pool()
        for x, y in p.imap_unordered(multi_process_dicom, self.dicom_files):
            if x and 'Modality' in x:
                self.ds.append(x)
        p.close()

    def separate_modalities(self):
        """
        Currently, separates the files into 4 different modalities (CT, MR, US, RTSTRUCT). Files with a different
        modality are not used. Certain tags are required depending on the modality, if those tags don't exist for its
        respective modality then it is not used.

        Returns
        -------

        """
        req = {'CT': ['SeriesInstanceUID', 'AcquisitionNumber', 'ImagePositionPatient', 'SliceThickness', 'PixelData', 'FrameOfReferenceUID'],
               'MR': ['SeriesInstanceUID', 'AcquisitionNumber', 'ImagePositionPatient', 'SliceThickness', 'PixelData', 'FrameOfReferenceUID'],
               'US': ['SeriesInstanceUID', 'PixelData'],
               'RTSTRUCT': ['SeriesInstanceUID', 'FrameOfReferenceUID']}

        for modality in list(self.ds_dictionary.keys()):
            ds_modality = [d for d in self.ds if d['Modality'].value == modality]
            if modality != 'RTSTRUCT' and modality != 'US':
                self.ds_dictionary[modality] = [ds_mod for ds_mod in ds_modality if
                                                len([r for r in req[modality] if r in ds_mod]) == len(req[modality]) and
                                                ds_mod['SliceThickness'].value]
            else:
                self.ds_dictionary[modality] = [ds_mod for ds_mod in ds_modality if
                                                len([r for r in req[modality] if r in ds_mod]) == len(req[modality])]

    def combine_images(self):
        standard_modalities = ['CT', 'MR']
        for mod in standard_modalities:
            if len(self.ds_dictionary[mod]) > 0:
                sorting_tags = np.asarray([[img['SeriesInstanceUID'].value,  img['SliceThickness'].value,
                                            img['AcquisitionNumber'].value, img['ImagePositionPatient'].value[2], ii]
                                           for ii, img in enumerate(self.ds_dictionary[mod])])
                sorting_tags_fix = np.asarray([[s[0], s[1], 1001, s[3], s[4]]
                                               if s[2] is None else s for s in sorting_tags])

                unique_tags = np.unique(sorting_tags_fix[:, 0:3].astype(str), axis=0)
                for tags in unique_tags:
                    unsorted_values = sorting_tags_fix[np.where((sorting_tags_fix[:, 0] == tags[0]) &
                                                                (sorting_tags_fix[:, 1] == tags[1]) &
                                                                (sorting_tags_fix[:, 2] == tags[2]))]

                    sorted_values = unsorted_values[np.argsort(unsorted_values[:, 3].astype('float'))[::-1]]
                    self.ds_images.append([self.ds_dictionary[mod][int(idx[4])] for idx in sorted_values])

        if len(self.ds_dictionary['US']) > 0:
            sorting_tags = np.asarray([[img['SeriesInstanceUID'].value,  ii]
                                       for ii, img in enumerate(self.ds_dictionary['US'])])
            unique_tags = np.unique(sorting_tags[:, 0], axis=0)
            for tags in unique_tags:
                unsorted_values = sorting_tags[np.where(sorting_tags[:, 0] == tags)]
                sorted_values = unsorted_values[np.argsort(unsorted_values[:, 0])]
                self.ds_images.append([self.ds_dictionary['US'][int(idx[1])] for idx in sorted_values])

    def separate_rt_images(self):
        for ii, rt_ds in enumerate(self.ds_dictionary['RTSTRUCT']):
            ref = rt_ds.ReferencedFrameOfReferenceSequence
            series_uid = ref[0]['RTReferencedStudySequence'][0]['RTReferencedSeriesSequence'][0][
                'SeriesInstanceUID'].value

            contour, roi_sop = [], []
            for contour_list in rt_ds.ROIContourSequence:
                contour_slice, roi_sop_slice = [], []
                for c in contour_list['ContourSequence']:
                    contour_slice_hold = np.round(np.array(c['ContourData'].value), 3)
                    if len(contour_slice_hold) > 8:
                        contour_slice.append(contour_slice_hold.reshape(int(len(contour_slice_hold) / 3), 3))
                        roi_sop_slice.append(str(c['ContourImageSequence'][0]['ReferencedSOPInstanceUID'].value))

                contour.append(contour_slice)
                roi_sop.append(roi_sop_slice)

            self.rt_df.at[ii, 'FilePath'] = rt_ds.filename
            self.rt_df.at[ii, 'SeriesInstanceUID'] = series_uid
            self.rt_df.at[ii, 'RoiNames'] = [s.ROIName for s in rt_ds.StructureSetROISequence]
            self.rt_df.at[ii, 'RoiSOP'] = roi_sop
            self.rt_df.at[ii, 'ContourData'] = contour

    def standard_useful_tags(self):
        for ii, image in enumerate(self.ds_images):
            for t in list(self.image_info.keys()):
                if t == 'PixelSpacing':
                    if image[0].Modality == 'US':
                        self.image_info.at[ii, t] = [np.round(image[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaX, 4),
                                                     np.round(image[0].SequenceOfUltrasoundRegions[0].PhysicalDeltaY, 4)]
                    else:
                        self.image_info.at[ii, t] = image[0].PixelSpacing
                elif t == 'ImagePositionPatient':
                    if image[0].Modality == 'US':
                        self.image_info.at[ii, t] = [0, 0, 0]
                    else:
                        self.image_info.at[ii, t] = image[0].ImagePositionPatient
                elif t == 'Slices':
                    self.image_info.at[ii, t] = len(image)
                else:
                    if t in image[0]:
                        self.image_info.at[ii, t] = image[0][t].value
                    else:
                        self.image_info.at[ii, t] = None

    def convert_images(self):
        for ii, image in enumerate(self.ds_images):
            image_slices = []
            if self.image_info.at[ii, 'Modality'] == 'CT':
                for slice_ in image:
                    if isinstance(slice_.pixel_array[0][0], np.int16):
                        image_slices.append(slice_.pixel_array - 1024)
                    else:
                        image_slices.append(slice_.pixel_array.astype('int16') - 1024)

            elif self.image_info.at[ii, 'Modality'] == 'MR':
                for slice_ in image:
                    image_slices.append(slice_.pixel_array.astype('int16'))

            elif self.image_info.at[ii, 'Modality'] == 'US':
                if len(image) == 1:
                    us_data = image[0].pixel_array
                    us_std = np.std(us_data, axis=3)
                    us_find_image = us_std == 0
                    us_binary = 1 * (us_find_image == 1)
                    image_slices = us_binary * us_data[:, :, :, 0]
                else:
                    print('Need to finish')

            self.image_data.append(np.asarray(image_slices))

    def collect_file_info(self):
        for image in self.ds_images:
            hold_info = []
            for img in image:
                file = img.filename
                sop = img.SOPInstanceUID
                hold_info.append([file, sop])

            self.file_info.append(hold_info)

    def rt_matching_new(self):
        info = self.image_info
        sop_info = self.file_info
        if self.existing_image_info and self.existing_file_info:
            info.append(self.existing_image_info)
            sop_info.append(self.existing_file_info)

        for ii in range(len(info.index)):
            img_sop = [sop[1] for sop in self.file_info[ii]]
            img_series = info.at[ii, 'SeriesInstanceUID']
            spacing_array = [info.at[ii, 'PixelSpacing'][0],
                             info.at[ii, 'PixelSpacing'][1],
                             info.at[ii, 'SliceThickness']]

            corner_array = [float(info.at[ii, 'ImagePositionPatient'][0])-float(spacing_array[0]/2),
                            float(info.at[ii, 'ImagePositionPatient'][1])-float(spacing_array[0]/2),
                            float(info.at[ii, 'ImagePositionPatient'][2])]

            rows = float(info.at[ii, 'Rows'])
            slices = len(self.ds_images[ii])

            rt_hold_name = []
            rt_hold_contour = []
            for jj in range(len(self.rt_df.index)):
                if img_series == self.rt_df.at[jj, 'SeriesInstanceUID'] and self.rt_df.at[jj, 'RoiSOP'][0][0] in img_sop:
                    for kk in range(len(self.rt_df.RoiNames[jj])):
                        roi_contour = [[]] * slices
                        contour_count = 0
                        for mm, rt_sop in enumerate(self.rt_df.RoiSOP[jj][kk]):
                            contour_count += 1
                            contour = self.rt_df.ContourData[jj][kk][mm]
                            hold_contour = np.round(np.abs((contour - corner_array) / spacing_array))
                            hold_contour[:, 1] = rows - hold_contour[:, 1]
                            slice_num = int(hold_contour[0][2])
                            if len(roi_contour[slice_num]) > 0:
                                roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                         hold_contour[0, 0:2])))
                            else:
                                roi_contour[slice_num] = []
                                roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                         hold_contour[0, 0:2])))

                        if contour_count > 0:
                            rt_hold_name.append(self.rt_df.RoiNames[jj][kk])
                            rt_hold_contour.append(roi_contour)

                    self.roi_info.append([self.rt_df.FilePath[jj], rt_hold_name])
                    self.roi_data.append(rt_hold_contour)
            if len(rt_hold_contour) == 0:
                self.roi_info.append([[], []])
                self.roi_data.append([])

    def roi_contour_fix(self):
        for j1, img in enumerate(self.roi_data):
            for j2, roi in enumerate(img):
                for j3, r in enumerate(roi):
                    for j4, slic in enumerate(r):
                        previous_contour = [-1, -1]
                        roi_fix = []
                        for point in slic:
                            if previous_contour[0] != point[0] or previous_contour[1] != point[1]:
                                roi_fix.append(point)
                            previous_contour = point

                        self.roi_data[j1][j2][j3][j4] = np.asarray(roi_fix)

    def get_file_info(self):
        return self.file_info

    def get_image_info(self):
        return self.image_info

    def get_image_data(self):
        return self.image_data

    def get_roi_data(self):
        return self.roi_data

    def get_roi_info(self):
        return self.roi_info

    def get_ds_images(self):
        return self.ds_images


def file_parsar(path, exclude_files):
    no_file_extension = []
    dicom_files = []
    mhd_files = []
    raw_files = []
    stl_files = []

    n = 0
    for root, dirs, files in os.walk(path):
        if files:
            n += 1
            for name in files:
                filepath = os.path.join(root, name)
                if filepath not in exclude_files:
                    filename, file_extension = os.path.splitext(filepath)
                    if file_extension == '.dcm':
                        dicom_files.append(filepath)
                    elif file_extension == '.mhd':
                        mhd_files.append([filepath, n])
                    elif file_extension == '.raw':
                        raw_files.append([filepath, n])
                    elif file_extension == '.stl':
                        stl_files.append([filepath, n])
                    elif file_extension == '':
                        no_file_extension.append([filepath, n])

    file_dictionary = {'Dicom': dicom_files,
                       'MHD': mhd_files,
                       'Raw': raw_files,
                       'Stl': stl_files,
                       'NoExtension': no_file_extension}

    return file_dictionary


def add_dicom_extension(files):
    new_files = []
    for name in files:
        a, b = os.path.splitext(name[0])
        if not b:
            newfile = name[0] + '.dcm'
            os.rename(name[0], newfile)
            new_files.append(newfile)
        else:
            new_files.append(name[0])

    return new_files


def check_memory(file_dictionary):
    dicom_size = 0
    for file in file_dictionary['Dicom']:
        dicom_size = dicom_size + os.path.getsize(file)

    raw_size = 0
    for file in file_dictionary['Raw']:
        raw_size = raw_size + os.path.getsize(file[0])

    stl_size = 0
    for file in file_dictionary['Stl']:
        stl_size = stl_size + os.path.getsize(file[0])

    total_size = dicom_size + raw_size + stl_size
    available_memory = psutil.virtual_memory()[1]
    memory_left = (available_memory - total_size)/1000000000

    return memory_left


def read_main():
    dir_path = r'C:\Users\csoconnor\Desktop\mic_test_data'

    exclude_files = []
    file_dictionary = file_parsar(dir_path, exclude_files)
    dicom_reader = DicomReader(file_dictionary['Dicom'], None, None)
    dicom_reader.load_dicom()


# def rs_main():
#     dir_path = r'C:\Users\csoconnor\Desktop\UPMC\Left'
#     export_path = r'C:\Users\csoconnor\Desktop\UPMC\Left'
#
#     subfolders = [x[0] for x in os.walk(dir_path)]
#     for folder in subfolders:
#         files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
#         if len(files) > 1:
#             print(folder)
#             exclude_files = []
#             file_dictionary = file_parsar(folder, exclude_files)
#             dicom_reader = DicomReader(file_dictionary['Dicom'], None, None)
#             dicom_reader.load_dicom()
#
#             if dicom_reader.get_file_info():
#                 rs_correction = RayStationCorrection(dicom_reader.ds_images, dicom_reader.file_info, export_path)
#                 rs_correction.get_series_update()
#                 rs_correction.get_frame_of_reference_update()
#                 rs_correction.apply_correction()
#                 rs_correction.save_data()


if __name__ == '__main__':
    # rs_main()
    read_main()
