
import os


def file_parsar(path, exclude_files=None):
    if not exclude_files:
        exclude_files = []
    no_file_extension = []
    dicom_files = []
    mhd_files = []
    raw_files = []
    stl_files = []

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

                    elif file_extension == '.stl':
                        stl_files.append(filepath)

                    elif file_extension == '':
                        no_file_extension.append(filepath)

    file_dictionary = {'Dicom': dicom_files,
                       'MHD': mhd_files,
                       'Raw': raw_files,
                       'Stl': stl_files,
                       'NoExtension': no_file_extension}

    return file_dictionary


if __name__ == '__main__':
    pass
