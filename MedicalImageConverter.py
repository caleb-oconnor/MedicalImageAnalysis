
import os
import time
import psutil

import numpy as np
import pandas as pd
import pydicom as dicom
import SimpleITK as sitk

from stl import mesh
from multiprocessing import Pool


def multi_process_dicom(path):
    try:
        datasets = dicom.dcmread(str(path), stop_before_pixels=True)
    except ImportError:
        datasets = []

    return datasets


class DicomReader:
    def __init__(self, dicom_files):
        self.dicom_files = dicom_files

        self.ds = []
        self.img_df = []
        self.img_info = []
        self.img_data = []
        self.img_filepath = []
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
                        self.img_df.append(series_df.loc[thick_idx])
                else:
                    for acquisition in acquisition_unique:
                        acquisition_idx = series_df.index[series_df['AcquisitionNumber'] == acquisition].tolist()
                        acquisition_df = series_df.loc[acquisition_idx]

                        thick_unique = acquisition_df.SliceThickness.unique()
                        for thick in thick_unique:
                            thick_idx = acquisition_df.index[acquisition_df['SliceThickness'] == thick].tolist()
                            self.img_df.append(acquisition_df.loc[thick_idx])

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

    def dicom_info(self):
        for ii in range(len(self.img_df)):
            if len(self.img_df[ii]) > 1:
                calculated_slice_thickness = abs(float(self.img_df[ii]['ImagePositionPatient'].iloc[1][2]) -
                                                 float(self.img_df[ii]['ImagePositionPatient'].iloc[0][2]))
            else:
                calculated_slice_thickness = self.img_df[ii]['SliceThickness'].iloc[0][2]

            self.img_df[ii]['CalculatedSliceThickness'] = calculated_slice_thickness

        for ii in range(len(self.img_df)):
            self.img_df[ii] = self.img_df[ii].sort_values(by=['InstanceNumber'], ascending=True)

        date_list = ['InstanceCreationDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate']
        time_list = ['InstanceCreationTime', 'SeriesTime', 'AcquisitionTime', 'ContentTime']

        for ii, img in enumerate(self.img_df):
            img_list = [img['Modality'].iloc[0],
                        len(img['FilePath']),
                        img['PatientID'].iloc[0],
                        img['SeriesDescription'].iloc[0],
                        img['SeriesNumber'].iloc[0]]

            date_hold = ''
            for date_idx in date_list:
                if not img[date_idx].iloc[0] == '':
                    date_hold = img[date_idx].iloc[0]
            if date_hold == '':
                img_list.append('Unknown')
            else:
                img_list.append(date_hold)

            time_hold = ''
            for time_idx in time_list:
                if not img[time_idx].iloc[0] == '':
                    time_hold = str(int(np.round(float(img[time_idx].iloc[0]))))
            if time_hold == '':
                img_list.append('Unknown')
            else:
                img_list.append(time_hold)

            img_list.append(img['SliceThickness'].iloc[0])
            img_list.append(img['PixelSpacing'].iloc[0])
            img_list.append([img['ImagePositionPatient'].iloc[0][2], img['ImagePositionPatient'].iloc[-1][2]])
            img_list.append(img['ImageOrientationPatient'].iloc[0])
            img_list.append(img['PatientPosition'].iloc[0])
            img_list.append(img['ImagePositionPatient'].iloc[0])
            img_list.append(img['Rows'].iloc[0])
            img_list.append(img['Columns'].iloc[0])
            img_list.append(img['PatientName'].iloc[0])
            img_list.append(img['CalculatedSliceThickness'].iloc[0])
            self.img_info.append(img_list)

    def dicom_images(self):
        for ii, img in enumerate(self.img_df):
            self.img_filepath.append(img['FilePath'])

            img_slice = []
            for idx in img['DatasetIndex']:
                if img['Modality'][idx] == 'CT':
                    if isinstance(self.ds[idx].pixel_array[0][0], np.int16):
                        img_slice.append(self.ds[idx].pixel_array - 1024)
                    else:
                        img_slice.append(self.ds[idx].pixel_array.astype('int16') - 1024)
                elif img['Modality'][idx] == 'MR':
                    img_slice.append(self.ds[idx].pixel_array.astype('int16'))

            if self.img_info[ii][11] == 'FFS':
                if float(self.img_info[ii][9][0]) < float(self.img_info[ii][9][1]):
                    self.img_data.append(np.flip(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 0), 2))
                else:
                    self.img_data.append(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 2))
            else:
                if float(self.img_info[ii][9][0]) < float(self.img_info[ii][9][1]):
                    self.img_data.append(np.flip(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 0), 2))
                else:
                    self.img_data.append(np.flip(np.transpose(np.asarray(img_slice), (0, 2, 1)), 2))

    def rt_matching(self, image_dataframe):
        if image_dataframe:
            image_df = image_dataframe
        else:
            image_df = self.img_df

        for img in image_df:
            img_series = img.SeriesInstanceUID.iloc[0]
            img_sop = img.SOPInstanceUID.tolist()
            spacing_array = [float(img.PixelSpacing.iloc[0][0]),
                             float(img.PixelSpacing.iloc[0][1]),
                             float(img.CalculatedSliceThickness.iloc[0])]

            corner_array = [float(img['ImagePositionPatient'].iloc[0][0]),
                            float(img['ImagePositionPatient'].iloc[0][1]),
                            float(img['ImagePositionPatient'].iloc[0][2])]

            img_direction = img['PatientPosition'].iloc[0]
            img_z_direction = [img['ImagePositionPatient'].iloc[0][2],
                               img['ImagePositionPatient'].iloc[-1][2]]

            rt_hold = []
            rt_hold_contour = []
            for ii in range(len(self.rt_df)):
                rt_series = self.rt_df.SeriesUID.iloc[ii]
                if rt_series == img_series:

                    for jj in range(len(self.rt_df.RoiNames[ii])):

                        roi_sop = []
                        roi_contour = [[]]*img.shape[0]
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
                                    if len(roi_contour[slice_num]) > 0:
                                        roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                                 hold_contour[0, 0:2])))
                                    else:
                                        roi_contour[slice_num] = []
                                        roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                                 hold_contour[0, 0:2])))
                                else:
                                    if float(img_z_direction[0]) < float(img_z_direction[1]):
                                        if len(roi_contour[int(img.shape[0])-slice_num-1]) > 0:
                                            roi_contour[int(img.shape[0])-slice_num-1].\
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))
                                        else:
                                            roi_contour[int(img.shape[0])-slice_num-1] = []
                                            roi_contour[int(img.shape[0])-slice_num-1].\
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))
                                    else:
                                        if len(roi_contour[slice_num]) > 0:
                                            roi_contour[slice_num].append(np.vstack((hold_contour[:, 0:2],
                                                                                     hold_contour[0, 0:2])))
                                        else:
                                            roi_contour[slice_num] = []
                                            roi_contour[slice_num].\
                                                append(np.vstack((hold_contour[:, 0:2],
                                                                  hold_contour[0, 0:2])))

                        if contour_count > 0:
                            rt_hold.append([self.rt_df.FilePath[ii], self.rt_df.RoiNames[ii][jj], roi_sop])
                            rt_hold_contour.append(roi_contour)

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
    def __init__(self, main_path, exclude_files, multi_folder, existing_ct_dataframe):
        self.main_path = main_path
        self.exclude_files = exclude_files
        self.multi_folder = multi_folder
        self.existing_ct_dataframe = existing_ct_dataframe

        self.total_size = None
        self.available_memory = None
        self.memory_left = None

        self.dicom_reader = None

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
                        if name.endswith('.dcm'):
                            self.dicom_files.append(filepath)
                        elif name.endswith('.mhd'):
                            self.mhd_files.append([filepath, n])
                        elif name.endswith('.raw'):
                            self.raw_files.append([filepath, n])
                        elif name.endswith('.stl'):
                            self.stl_files.append([filepath, n])

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
        self.dicom_reader.dicom_info()
        self.dicom_reader.dicom_images()
        self.dicom_reader.rt_matching(self.existing_ct_dataframe)
        self.dicom_reader.roi_contour_fix()
        t2 = time.time()
        print('Dicom Read Time: ', t2-t1)

    def get_dicom_images(self):
        return self.dicom_reader.img_data, self.dicom_reader.img_info, self.dicom_reader.img_filepath

    def get_dicom_roi(self):
        return self.dicom_reader.roi_data, self.dicom_reader.roi_info

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

    def read_stl(self):
        t1 = time.time()
        for file in self.stl_files:
            self.stl_img.append(mesh.Mesh.from_file(file[0]))
        t2 = time.time()
        print('STL Read (numpy) Time: ', t2 - t1)


# def main():
#     path = r'C:\Users\csoconnor\Desktop\read_test_3'
#
#     mic = MedicalImageConverter(path, exclude_files=[], multi_folder=False, existing_ct_dataframe=None)
#     mic.file_parsar()
#     mic.check_memory()
#     mic.read_dicom()
#     # mic.read_meta()
#     # mic.read_stl()
#
#
# if __name__ == '__main__':
#     main()
