"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org
"""

import os
import psutil

from .read import DicomReader, MhdReader, StlReader, VtkReader, ThreeMfReader


def check_memory(files):
    """
    Estimate available memory after accounting for file sizes in different formats.

    Parameters
    ----------
    files : dict
        Dictionary of file lists grouped by type. Keys: 'Dicom', 'Nifti', 'Raw', 'Stl', 'Vtk', '3mf'

    Returns
    -------
    float
        Available memory in GB after subtracting the total file sizes.
    """

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
    Walk through folders or file list and sort files into categories based on extension.

    Supports:
        DICOM (.dcm)
        MHD (.mhd)
        RAW (.raw)
        Nifti (.nii.gz)
        STL (.stl)
        VTK (.vtk)
        3MF (.3mf)
        No extension

    Parameters
    ----------
    folder_path : str, optional
        Path to a folder containing files.
    file_list : list, optional
        Explicit list of file paths to process.
    exclude_files : list, optional
        List of file paths to exclude.

    Returns
    -------
    dict
        Dictionary of sorted file lists keyed by type.
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
    """
    Load DICOM files from a folder or file list using DicomReader.

    Parameters
    ----------
    folder_path : str, optional
        Path to folder containing DICOM files.
    file_list : list, optional
        Explicit list of file paths to read.
    exclude_files : list, optional
        List of file paths to exclude.
    only_tags : bool, optional
        If True, only read DICOM metadata tags.
    only_modality : list, optional
        List of DICOM modalities to include (e.g., CT, MR, PT, RTSTRUCT).
    only_load_roi_names : list, optional
        List of ROI names to load.
    clear : bool, optional
        Whether to clear existing DICOM data before loading.
    """
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
    """
    Load 3MF file using ThreeMfReader.

    Parameters
    ----------
    file : str
        Path to the 3MF file.
    roi_name : str, optional
        Name of the ROI to associate with the 3MF file.
    """
    mf3_reader = ThreeMfReader(file, roi_name=roi_name)
    mf3_reader.load()


def read_mhd(file=None, modality=None, reference_name=None, moving_name=None, roi_name=None, dose=None, dvf=None):
    """
    Load MHD (MetaImage) file using MhdReader.

    Parameters
    ----------
    file : str
        Path to the MHD file.
    modality : str, optional
        Imaging modality (e.g., CT, MR).
    reference_name : str, optional
        Name of the reference image for registration.
    moving_name : str, optional
        Name of the moving image for registration.
    roi_name : str, optional
        ROI name to associate with this image.
    dose : optional
        Dose object to associate.
    dvf : optional
        Deformation vector field to associate.
    """
    if file is not None:
        mhd_reader = MhdReader(file=file, modality=modality, reference_name=reference_name, moving_name=moving_name,
                               roi_name=roi_name, dose=dose, dvf=dvf)
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
