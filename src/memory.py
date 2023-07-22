
import os
import psutil


def check_memory(file_dictionary):
    dicom_size = 0
    for file in file_dictionary['Dicom']:
        dicom_size = dicom_size + os.path.getsize(file)

    raw_size = 0
    for file in file_dictionary['Raw']:
        raw_size = raw_size + os.path.getsize(file)

    stl_size = 0
    for file in file_dictionary['Stl']:
        stl_size = stl_size + os.path.getsize(file)

    total_size = dicom_size + raw_size + stl_size
    available_memory = psutil.virtual_memory()[1]
    memory_left = (available_memory - total_size) / 1000000000

    return total_size, available_memory, memory_left
