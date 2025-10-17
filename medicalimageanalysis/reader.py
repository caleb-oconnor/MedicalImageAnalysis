"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:
    Currently, reads in dicom files for modalities: CT, MR, DX, MG, US, RTSTRUCTS.

    The user inputs either a given folder path (can contain multiple images and subfolders). The files are sorted into
    separate images.

    Other user input options:
        file_list - if the user already has the files wanted to read in, must be in type list
        exclude_files - if the user wants to not read certain files
        only_tags - does not read in the pixel array just the tags
        only_modality - specify which modalities to read in, if not then all modalities will be read
        only_load_roi_names - will only load rois with input name, list format

Functions:
    read_dicoms - Reads in all dicom files and separates them into the image list variable

"""

import os
import psutil

from .read import DicomReader, MhdReader, StlReader, VtkReader, ThreeMfReader


def check_memory(files):
    dicom_size = 0
    for file in files['Dicom']:
        dicom_size = dicom_size + os.path.getsize(file)

    nifti_size = 0
    for file in files['Nifti']:
        nifti_size = nifti_size + os.path.getsize(file)

    raw_size = 0
    for file in files['Raw']:
        raw_size = raw_size + os.path.getsize(file)

    stl_size = 0
    for file in files['Stl']:
        stl_size = stl_size + os.path.getsize(file)

    vtk_size = 0
    for file in files['Vtk']:
        vtk_size = vtk_size + os.path.getsize(file)

    mf3_size = 0
    for file in files['3mf']:
        mf3_size = mf3_size + os.path.getsize(file)

    total_size = dicom_size + raw_size + nifti_size + stl_size + vtk_size + mf3_size
    available_memory = psutil.virtual_memory()[1]
    return (available_memory - total_size) / 1000000000


def file_parsar(folder_path=None, file_list=None, exclude_files=None):
    """
    Walks through all the subfolders and checks each file extensions. Sorts them into 6 different options:
        Dicom
        MHD
        Raw
        Nifti
        VTK
        STL
        3mf
        No extension

    :param folder_path: single folder path
    :type folder_path: string path
    :param file_list: list of filepaths
    :type file_list: list
    :param exclude_files: list of filepaths
    :type exclude_files: list
    :return: dictionary of into files sorted by extensions
    :rtype: dictionary
    """

    no_file_extension = []
    dicom_files = []
    mhd_files = []
    raw_files = []
    nifti_files = []
    stl_files = []
    vtk_files = []
    mf3_files = []

    if not exclude_files:
        exclude_files = []

    if file_list is None:
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            if files:
                for name in files:
                    file_list += [os.path.join(root, name)]

    for filepath in file_list:
        if filepath not in exclude_files:
            filename, file_extension = os.path.splitext(filepath)

            if file_extension == '.dcm':
                dicom_files.append(filepath)

            elif file_extension == '.mhd':
                mhd_files.append(filepath)

            elif file_extension == '.raw':
                raw_files.append(filepath)

            elif file_extension == '.gz':
                if filepath[-6:] == 'nii.gz':
                    nifti_files.append(filepath)

            elif file_extension == '.stl':
                stl_files.append(filepath)

            elif file_extension == '.vtk':
                vtk_files.append(filepath)

            elif file_extension == '.3mf':
                mf3_files.append(filepath)

            elif file_extension == '':
                no_file_extension.append(filepath)

    files = {'Dicom': dicom_files,
             'MHD': mhd_files,
             'Raw': raw_files,
             'Nifti': nifti_files,
             'Stl': stl_files,
             'Vtk': vtk_files,
             '3mf': mf3_files,
             'NoExtension': no_file_extension}

    return files


def read_dicoms(folder_path=None, file_list=None, exclude_files=None, only_tags=False, only_modality=None,
                only_load_roi_names=None, clear=True):
    if only_modality is not None:
        only_modality = only_modality
    else:
        only_modality = ['CT', 'MR', 'PT', 'US', 'DX', 'RF', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE']

    files = None
    if folder_path is not None or file_list is not None:
        files = file_parsar(folder_path=folder_path, file_list=file_list, exclude_files=exclude_files)

    dicom_reader = DicomReader(files, only_tags, only_modality, only_load_roi_names, clear)
    dicom_reader.load()


def read_3mf(file=None, roi_name=None):
    mf3_reader = ThreeMfReader(file, roi_name=roi_name)
    mf3_reader.load()


def read_mhd(file=None, modality=None, reference_name=None, roi_name=None, dose=None, dvf=None):
    if file is not None:
        mhd_reader = MhdReader(file=file, modality=modality, reference_name=reference_name, roi_name=roi_name,
                               dose=dose, dvf=dvf)
        mhd_reader.load()


# def read_stl(self, files=None, create_image=False, match_image=None):
#     stl_reader = StlReader(self)
#     if files is not None:
#         stl_reader.input_files(files)
#     stl_reader.load()
#
#
# def read_vtk(self, files=None, create_image=False, match_image=None):
#         vtk_reader = VtkReader(self)
#         if files is not None:
#             vtk_reader.input_files(files)
#         vtk_reader.load()


if __name__ == '_main__':
    pass
