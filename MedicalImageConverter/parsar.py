"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:
    Primary use is for the Reader class to sort files depending on the file extension.

"""

import os


def file_parsar(path, exclude_files=None):
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

    :param path: single folder path
    :type path: string path
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

    for root, dirs, files in os.walk(path):
        if files:
            for name in files:
                filepath = os.path.join(root, name)

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

    file_dictionary = {'Dicom': dicom_files,
                       'MHD': mhd_files,
                       'Raw': raw_files,
                       'Nifti': nifti_files,
                       'Stl': stl_files,
                       'Vtk': vtk_files,
                       '3mf': mf3_files,
                       'NoExtension': no_file_extension}

    return file_dictionary
