"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Gets total memory size of all the files that can be loaded. Also, returns how much memory would be left if all the
    files were loaded in.

    Note: The memory usage will actually be more than the total file size of all files loaded.

"""

import os
import psutil


def check_memory(file_dictionary):
    dicom_size = 0
    for file in file_dictionary['Dicom']:
        dicom_size = dicom_size + os.path.getsize(file)

    nifti_size = 0
    for file in file_dictionary['Nifti']:
        nifti_size = nifti_size + os.path.getsize(file)
        
    raw_size = 0
    for file in file_dictionary['Raw']:
        raw_size = raw_size + os.path.getsize(file)

    stl_size = 0
    for file in file_dictionary['Stl']:
        stl_size = stl_size + os.path.getsize(file)

    vtk_size = 0
    for file in file_dictionary['Vtk']:
        vtk_size = vtk_size + os.path.getsize(file)

    mf3_size = 0
    for file in file_dictionary['3mf']:
        mf3_size = mf3_size + os.path.getsize(file)

    total_size = dicom_size + raw_size + nifti_size + stl_size + vtk_size + mf3_size
    available_memory = psutil.virtual_memory()[1]
    memory_left = (available_memory - total_size) / 1000000000

    return total_size, available_memory, memory_left
