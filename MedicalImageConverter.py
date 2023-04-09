
import os
import time
import psutil

import gdcm  # python-gdcm
import numpy as np
import pandas as pd
import pydicom as dicom
import SimpleITK as sitk

# from stl import mesh
from multiprocessing import Pool


def multi_process_dicom(path):
    try:
        datasets = dicom.dcmread(str(path))
    except ImportError:
        datasets = []

    return datasets


def add_dicom_extension(files):
    new_files = []
    for name in files:
        a, b = os.path.splitext(name[0])
        if not b:
            newfile = name[0] + '.dcm'
            os.rename(name[0], newfile)
            new_files.append([newfile, name[1]])
        else:
            new_files.append(name)


class RayStationCorrection:
    def __init__(self, ds, img_info, img_tag_info, file_info, data_path):
        self.ds = ds
        self.img_info = img_info
        self.img_tag_info = img_tag_info
        self.file_info = file_info
        self.data_path = data_path

        self.ds_split = []

        self.series_update = []
        self.frame_reference_update = []
        self.ptid = None
        self.name = None

    def get_series_update(self):
        series = self.img_info['SeriesInstanceUID'].to_list()
        unique_series = self.img_info['SeriesInstanceUID'].unique().tolist()

        for s in unique_series:
            series_idx = [ii for ii, value in enumerate(series) if value == s]
            if len(series_idx) == 1:
                self.series_update.append([0, series[series_idx[0]]])
            else:
                for jj, idx in enumerate(series_idx):
                    if jj == 0:
                        self.series_update.append([0, series[idx]])
                    else:
                        self.series_update.append([1, series[idx] + '.' + str(jj)])
                        # last_element = series[idx].split('.')[-1]
                        # if len(last_element) == 1:
                        #     last_digit = int(last_element)
                        # else:
                        #     last_digit = int(last_element[-1])
                        #
                        # if last_digit + jj < 10:
                        #     self.series_update.append([1, series[idx][:-1] + str(last_digit + jj)])
                        # else:
                        #     self.series_update.append([1, series[idx][:-1] + str(last_digit + jj)])

    def get_frame_of_reference_update(self):
        for_unique = [tag['FrameOfReferenceUID'].unique().tolist() for tag in self.img_tag_info]

        for for_img in for_unique:
            if len(for_img) == 1:
                self.frame_reference_update.append([0, for_img[0]])
            else:
                self.frame_reference_update.append([1, for_img[0]])

    def update_ds_order(self):
        for ii, info in enumerate(self.img_tag_info):
            ds_idx = self.img_tag_info[ii]['DatasetIndex'].tolist()

            hold_ds = []
            for idx in ds_idx:
                hold_ds.append(self.ds[idx])

            self.ds_split.append(hold_ds)

    def update_ptid_and_name(self, ptid, name):
        self.ptid = ptid
        self.name = name

    def apply_correction(self):
        for ii, info in enumerate(self.img_tag_info):
            ds_idx = self.img_tag_info[ii]['DatasetIndex'].tolist()

            if self.series_update[ii][0] == 1:
                for jj in range(len(ds_idx)):
                    self.ds_split[ii][jj]['SeriesInstanceUID'].value = self.series_update[ii][1]

            if self.frame_reference_update[ii][0] == 1:
                for jj in range(len(ds_idx)):
                    self.ds_split[ii][jj]['FrameOfReferenceUID'].value = self.frame_reference_update[ii][1]

            if self.ptid:
                for jj in range(len(ds_idx)):
                    self.ds_split[ii][jj]['PatientID'].value = self.ptid

            if self.name:
                for jj in range(len(ds_idx)):
                    self.ds_split[ii][jj]['PatientName'].value = self.name

    def save_data(self):
        save_path = os.path.join(self.data_path, 'Corrections')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        ptid = self.ds_split[0][0]['PatientID'].value
        ptid_path = os.path.join(save_path, ptid)
        if not os.path.exists(ptid_path):
            os.mkdir(ptid_path)

            for ii, ds_img in enumerate(self.ds_split):
                modality = self.ds_split[ii][0]['Modality'].value
                thickness = np.round(self.img_info.at[ii, 'CalculatedSliceThickness'], 3)
                img_path = os.path.join(ptid_path, modality + '_' + str(ii) + '_' + str(thickness) + 'mm')
                os.mkdir(img_path)

                for jj, ds_file in enumerate(ds_img):
                    ds_file.save_as(os.path.join(img_path, self.file_info[ii][0][jj].split('\\')[-1]))


class DicomReader:
    def __init__(self, dicom_files):
        self.dicom_files = dicom_files

        self.ds = []
        self.file_info = []
        self.img_tag_df = []
        self.img_info = None
        self.img_data = []
        self.tag_info = None
        self.tags = ['DatasetIndex',
                     'FilePath',
                     'SOPInstanceUID',
                     'SpecificCharacterSet',
                     'InstanceCreationDate', 'StudyDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate',
                     'InstanceCreationTime', 'StudyTime', 'SeriesTime', 'AcquisitionTime', 'ContentTime',
                     'Modality',
                     'SeriesDescription',
                     'PatientName', 'PatientID', 'PatientBirthDate', 'PatientPosition', 'PatientSex',
                     'SliceThickness', 'CalculatedSliceThickness',
                     'StudyInstanceUID',
                     'SeriesInstanceUID',
                     'SeriesNumber', 'AcquisitionNumber', 'InstanceNumber',
                     'ImagePositionPatient', 'ImageOrientationPatient',
                     'FrameOfReferenceUID',
                     'Rows', 'Columns',
                     'PixelSpacing', 'PixelPaddingValue',
                     'RescaleIntercept', 'RescaleSlope', 'RescaleType']

        rt_column_tags = ['FilePath', 'SeriesUID', 'RoiNames', 'RoiSOP', 'ContourData']
        self.rt_df = pd.DataFrame(columns=rt_column_tags)
        self.roi_info = []
        self.roi_data = []
        self.roi_name = []

    def read_dicom(self):
        if len(self.dicom_files) > 500:
            p = Pool()
            for x in p.imap_unordered(multi_process_dicom, self.dicom_files):
                if x:
                    self.ds.append(x)
            p.close()
        else:
            for path in self.dicom_files:
                self.ds.append(dicom.dcmread(path))

    def get_tags(self):
        tag_list = []
        for ii, info in enumerate(self.ds):
            tag_hold = []
            if info:
                for t in self.tags:
                    if t == 'DatasetIndex':
                        tag_hold.append(ii)
                    elif t == 'FilePath':
                        tag_hold.append(self.dicom_files[ii])
                    elif t == 'CalculatedSliceThickness':
                        tag_hold.append('')
                    elif t in info:
                        tag_hold.append(info[t].value)
                    else:
                        tag_hold.append('')
            else:
                for t in self.tags:
                    tag_hold.append('')

            tag_list.append(tag_hold)
        self.tag_info = pd.DataFrame(tag_list, columns=self.tags)

    def separate_dicom_images(self):
        imgs_idx = self.tag_info.index[(self.tag_info['Modality'] == 'CT') |
                                       (self.tag_info['Modality'] == 'MR')].tolist()
        imgs_df = self.tag_info.loc[imgs_idx]

        if imgs_idx:
            series_unique = imgs_df.SeriesInstanceUID.unique()
            for series in series_unique:
                series_idx = imgs_df.index[imgs_df['SeriesInstanceUID'] == series].tolist()
                series_df = imgs_df.loc[series_idx]

                acquisition_unique = series_df.AcquisitionNumber.unique()
                if len(acquisition_unique) == 1:
                    thick_unique = series_df.SliceThickness.unique()
                    for thick in thick_unique:
                        thick_idx = series_df.index[series_df['SliceThickness'] == thick].tolist()
                        self.img_tag_df.append(series_df.loc[thick_idx])
                else:
                    for acquisition in acquisition_unique:
                        acquisition_idx = series_df.index[series_df['AcquisitionNumber'] == acquisition].tolist()
                        acquisition_df = series_df.loc[acquisition_idx]

                        thick_unique = acquisition_df.SliceThickness.unique()
                        for thick in thick_unique:
                            thick_idx = acquisition_df.index[acquisition_df['SliceThickness'] == thick].tolist()
                            self.img_tag_df.append(acquisition_df.loc[thick_idx])

    def separate_rt_images(self):
        rts_idx = self.tag_info.index[self.tag_info['Modality'] == 'RTSTRUCT'].tolist()

        for ii, idx in enumerate(rts_idx):
            rt_ds = self.ds[idx]
            ref = rt_ds.ReferencedFrameOfReferenceSequence
            structure = rt_ds.StructureSetROISequence
            contours_combined = rt_ds.ROIContourSequence

            series_uid = ref[0]['RTReferencedStudySequence'][0]['RTReferencedSeriesSequence'][0][
                'SeriesInstanceUID'].value

            contour, roi_sop = [], []
            for contour_list in contours_combined:
                contour_slice, roi_sop_slice = [], []
                for c in contour_list['ContourSequence']:
                    contour_slice_hold = np.round(np.array(c['ContourData'].value), 3)
                    if len(contour_slice_hold) > 8:
                        contour_slice.append(contour_slice_hold.reshape(int(len(contour_slice_hold) / 3), 3))
                        roi_sop_slice.append(str(c['ContourImageSequence'][0]['ReferencedSOPInstanceUID'].value))

                contour.append(contour_slice)
                roi_sop.append(roi_sop_slice)

            self.rt_df.at[ii, 'FilePath'] = self.tag_info.at[idx, 'FilePath']
            self.rt_df.at[ii, 'SeriesUID'] = series_uid
            self.rt_df.at[ii, 'RoiNames'] = [s.ROIName for s in structure]
            self.rt_df.at[ii, 'RoiSOP'] = roi_sop
            self.rt_df.at[ii, 'ContourData'] = contour

    def get_calculated_thickness(self):
        for ii in range(len(self.img_tag_df)):
            if len(self.img_tag_df[ii]) > 1:
                calculated_slice_thickness = abs(float(self.img_tag_df[ii]['ImagePositionPatient'].iloc[1][2]) -
                                                 float(self.img_tag_df[ii]['ImagePositionPatient'].iloc[0][2]))
            else:
                idx = self.img_tag_df[ii].index.tolist()
                calculated_slice_thickness = self.img_tag_df[ii].at[idx[0], 'SliceThickness']

            self.img_tag_df[ii]['CalculatedSliceThickness'] = calculated_slice_thickness

    def sort_instances(self):
        for ii in range(len(self.img_tag_df)):
            self.img_tag_df[ii] = self.img_tag_df[ii].sort_values(by=['InstanceNumber'], ascending=True)

    def dicom_info(self):
        info_tags = ['PatientID', 'PatientName', 'Modality', 'SeriesDescription', 'Date', 'Time',
                     'SeriesInstanceUID', 'SeriesNumber', 'AcquisitionNumber',
                     'Slices', 'SliceThickness', 'CalculatedSliceThickness', 'SliceRange',
                     'PixelSpacing', 'Rows', 'Columns',
                     'ImageOrientationPatient', 'PatientPosition', 'ImagePositionPatient']
        self.img_info = pd.DataFrame(columns=info_tags)

        date_list = ['InstanceCreationDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate']
        time_list = ['InstanceCreationTime', 'SeriesTime', 'AcquisitionTime', 'ContentTime']
        for ii, img in enumerate(self.img_tag_df):
            date_hold = 'Unknown'
            for date_idx in date_list:
                if not img[date_idx].iloc[0] == '':
                    date_hold = img[date_idx].iloc[0]

            time_hold = 'Unknown'
            for time_idx in time_list:
                if not img[time_idx].iloc[0] == '':
                    time_hold = str(int(np.round(float(img[time_idx].iloc[0]))))

            for t in info_tags:
                if t == 'Date':
                    self.img_info.at[ii, 'Date'] = date_hold
                elif t == 'Time':
                    self.img_info.at[ii, 'Time'] = time_hold
                elif t == 'SliceRange':
                    self.img_info.at[ii, 'SliceRange'] = [img['ImagePositionPatient'].iloc[0][2],
                                                          img['ImagePositionPatient'].iloc[-1][2]]
                elif t == 'Slices':
                    self.img_info.at[ii, 'Slices'] = len(img['FilePath'])
                else:
                    self.img_info.at[ii, t] = img[t].iloc[0]

    def dicom_images(self):
        for ii, img in enumerate(self.img_tag_df):
            img_slice = []
            for idx in img['DatasetIndex']:
                if img['Modality'][idx] == 'CT':
                    if isinstance(self.ds[idx].pixel_array[0][0], np.int16):
                        img_slice.append(self.ds[idx].pixel_array - 1024)
                    else:
                        img_slice.append(self.ds[idx].pixel_array.astype('int16') - 1024)

                elif img['Modality'][idx] == 'MR':
                    img_slice.append(self.ds[idx].pixel_array.astype('int16'))

            if self.img_info.at[ii, 'PatientPosition'] == 'FFS':
                if float(self.img_info.at[ii, 'SliceRange'][0]) < float(self.img_info.at[ii, 'SliceRange'][1]):
                    self.img_data.append(np.flip(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 0), 2))
                else:
                    self.img_data.append(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 2))
            else:
                if float(self.img_info.at[ii, 'SliceRange'][0]) < float(self.img_info.at[ii, 'SliceRange'][1]):
                    self.img_data.append(np.flip(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 0), 2))
                else:
                    self.img_data.append(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 2))

    def get_file_info(self):
        for ii, img in enumerate(self.img_tag_df):
            files = self.img_tag_df[ii]['FilePath'].tolist()
            sop = self.img_tag_df[ii]['SOPInstanceUID'].tolist()

            self.file_info.append([files, sop])

    def rt_matching(self, image_info, file_info):
        if not image_info and not file_info:
            image_info = self.img_info
            file_info = self.file_info

        for nn in range(len(image_info.index)):
            img_series = image_info.at[nn, 'SeriesInstanceUID']
            img_sop = file_info[nn][1]
            spacing_array = [image_info.at[nn, 'PixelSpacing'][0],
                             image_info.at[nn, 'PixelSpacing'][1],
                             image_info.at[nn, 'CalculatedSliceThickness']]

            corner_array = [float(image_info.at[nn, 'ImagePositionPatient'][0]),
                            float(image_info.at[nn, 'ImagePositionPatient'][1]),
                            float(image_info.at[nn, 'ImagePositionPatient'][2])]

            img_direction = image_info.at[nn, 'PatientPosition']
            img_z_direction = [image_info.at[nn, 'SliceRange'][0],
                               image_info.at[nn, 'SliceRange'][1]]
            slices = image_info.at[nn, 'Slices']

            rt_hold = []
            rt_hold_name = []
            rt_hold_contour = []
            for ii in range(len(self.rt_df)):
                rt_series = self.rt_df.SeriesUID.iloc[ii]
                if rt_series == img_series:

                    for jj in range(len(self.rt_df.RoiNames[ii])):

                        roi_sop = []
                        roi_contour = [[]]*slices
                        contour_count = 0
                        for kk, rt_sop in enumerate(self.rt_df.RoiSOP[ii][jj]):
                            if rt_sop in img_sop:
                                contour_count += 1
                                roi_sop.append(rt_sop)
                                contour = self.rt_df.ContourData[ii][jj][kk]
                                hold_contour = np.round(np.abs((contour - corner_array) / spacing_array))
                                hold_contour[:, 1] = 511 - hold_contour[:, 1]
                                slice_num = int(hold_contour[0][2])
                                if img_direction == 'FFS':
                                    if float(img_z_direction[1]) < float(img_z_direction[0]):
                                        if len(roi_contour[slice_num]) > 0:
                                            roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                                     hold_contour[0, 0:2])))
                                        else:
                                            roi_contour[slice_num] = []
                                            roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                                     hold_contour[0, 0:2])))
                                    else:
                                        if len(roi_contour[int(slices) - slice_num - 1]) > 0:
                                            roi_contour[int(slices) - slice_num - 1]. \
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))
                                        else:
                                            roi_contour[int(slices) - slice_num - 1] = []
                                            roi_contour[int(slices) - slice_num - 1]. \
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))
                                else:
                                    if float(img_z_direction[0]) < float(img_z_direction[1]):
                                        if len(roi_contour[int(slices) - slice_num - 1]) > 0:
                                            roi_contour[int(slices) - slice_num - 1]. \
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))
                                        else:
                                            roi_contour[int(slices) - slice_num - 1] = []
                                            roi_contour[int(slices) - slice_num - 1]. \
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))
                                    else:
                                        if len(roi_contour[slice_num]) > 0:
                                            roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                                     hold_contour[0, 0:2])))
                                        else:
                                            roi_contour[slice_num] = []
                                            roi_contour[slice_num]. \
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))

                        if contour_count > 0:
                            rt_hold_name.append(self.rt_df.RoiNames[ii][jj])
                            rt_hold.append([self.rt_df.FilePath[ii], self.rt_df.RoiNames[ii][jj], roi_sop])
                            rt_hold_contour.append(roi_contour)

            self.roi_name.append(rt_hold_name)
            self.roi_info.append(rt_hold)
            self.roi_data.append(rt_hold_contour)

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


class MedicalImageConverter:
    def __init__(self, main_path, exclude_files, multi_folder, existing_img_info, existing_file_info):
        self.main_path = main_path
        self.exclude_files = exclude_files
        self.multi_folder = multi_folder
        self.existing_img_info = existing_img_info
        self.existing_file_info = existing_file_info

        self.total_size = None
        self.available_memory = None
        self.memory_left = None

        self.dicom_reader = None

        self.no_file_extenstion = []
        self.dicom_files = []
        self.mhd_files = []
        self.raw_files = []
        self.stl_files = []

        self.ds = []

        self.mhd_type = []
        self.mhd_img = []
        self.mhd_info = []
        self.stl_img = []

    def file_parsar(self):
        n = 0
        for root, dirs, files in os.walk(self.main_path):
            if files:
                n += 1
                for name in files:
                    filepath = os.path.join(root, name)
                    if filepath not in self.exclude_files:
                        filename, file_extension = os.path.splitext(filepath)
                        if file_extension == '.dcm':
                            self.dicom_files.append(filepath)
                        elif file_extension == '.mhd':
                            self.mhd_files.append([filepath, n])
                        elif file_extension == '.raw':
                            self.raw_files.append([filepath, n])
                        elif file_extension == '.stl':
                            self.stl_files.append([filepath, n])
                        elif file_extension == '':
                            self.no_file_extenstion.append([filepath, n])

    def check_memory(self):
        dicom_size = 0
        for file in self.dicom_files:
            dicom_size = dicom_size + os.path.getsize(file)

        raw_size = 0
        for file in self.raw_files:
            raw_size = raw_size + os.path.getsize(file[0])

        stl_size = 0
        for file in self.stl_files:
            stl_size = stl_size + os.path.getsize(file[0])

        self.total_size = dicom_size + raw_size + stl_size
        self.available_memory = psutil.virtual_memory()[1]
        self.memory_left = (self.available_memory - self.total_size)/1000000000

    def read_dicom(self):
        t1 = time.time()
        self.dicom_reader = DicomReader(self.dicom_files)
        self.dicom_reader.read_dicom()
        self.dicom_reader.get_tags()
        self.dicom_reader.separate_dicom_images()
        self.dicom_reader.separate_rt_images()
        self.dicom_reader.get_calculated_thickness()
        self.dicom_reader.sort_instances()
        self.dicom_reader.dicom_info()
        self.dicom_reader.dicom_images()
        self.dicom_reader.get_file_info()
        self.dicom_reader.rt_matching(self.existing_img_info, self.existing_file_info)
        self.dicom_reader.roi_contour_fix()
        t2 = time.time()
        print('Dicom Read Time: ', t2-t1)

    def get_file_info(self):
        return self.dicom_reader.file_info

    def get_img_info(self):
        return self.dicom_reader.img_info

    def get_img_data(self):
        return self.dicom_reader.img_data

    def get_roi_info(self):
        return self.dicom_reader.roi_info

    def get_roi_data(self):
        return self.dicom_reader.roi_data

    def get_roi_name(self):
        return self.dicom_reader.roi_name

    def get_tag_info(self):
        return self.dicom_reader.img_tag_df

    def get_ds(self):
        return self.dicom_reader.ds

    def sort_meta(self):
        mhd_paths_only = [item[0] for item in self.mhd_files]
        raw_paths_only = [item[0] for item in self.raw_files]

        for ii, mhd in enumerate(mhd_paths_only):
            for jj, raw in enumerate(raw_paths_only):
                mhd_split = mhd.split('\\')[-1].split('.')[0].lower()
                raw_split = raw.split('\\')[-1].split('.')[0].lower()
                if mhd_split == raw_split and self.mhd_files[ii][1] == self.mhd_files[jj][1]:
                    if 'ct' in mhd_split:
                        self.mhd_type.append([mhd, self.mhd_files[ii][1], 'CT'])
                    elif 'mr' in mhd_split:
                        self.mhd_type.append([mhd, self.mhd_files[ii][1], 'MR'])
                    elif 'roi' in mhd_split:
                        self.mhd_type.append([mhd, self.mhd_files[ii][1], 'ROI'])
                    else:
                        self.mhd_type.append([mhd, self.mhd_files[ii][1], 'Failed'])

    def read_meta(self):
        t1 = time.time()
        self.mhd_img = []
        self.mhd_info = []
        for mhd in self.mhd_type:
            if mhd[2] != 'Failed':
                sitk_data = sitk.ReadImage(mhd[0])
                self.mhd_img.append(sitk.GetArrayFromImage(sitk_data))
                self.mhd_info.append([sitk_data.GetOrigin(),
                                      sitk_data.GetSpacing(),
                                      sitk_data.GetSize(),
                                      sitk_data.GetDirection(),
                                      mhd[1],
                                      mhd[2]])
        t2 = time.time()
        print('Meta Read Time: ', t2-t1)

    # def read_stl(self):
    #     t1 = time.time()
    #     for file in self.stl_files:
    #         self.stl_img.append(mesh.Mesh.from_file(file[0]))
    #     t2 = time.time()
    #     print('STL Read (numpy) Time: ', t2 - t1)


def main():
    data_path = r'C:\Users\csoconnor\Desktop\coh_test\COH_0001\COH_00010001'

    mic = MedicalImageConverter(data_path,
                                exclude_files=[],
                                multi_folder=False,
                                existing_img_info=None,
                                existing_file_info=None)
    mic.file_parsar()
    if mic.no_file_extenstion:
        add_dicom_extension(mic.no_file_extenstion)
    mic.check_memory()
    mic.read_dicom()

    rs_correction = RayStationCorrection(mic.get_ds(), mic.get_img_info(), mic.get_tag_info(), mic.get_file_info(),
                                         data_path)
    rs_correction.get_series_update()
    rs_correction.get_frame_of_reference_update()
    rs_correction.update_ds_order()
    rs_correction.update_ptid_and_name('COH_0001', 'COH_0001')
    rs_correction.apply_correction()
    rs_correction.save_data()


if __name__ == '__main__':
    main()
