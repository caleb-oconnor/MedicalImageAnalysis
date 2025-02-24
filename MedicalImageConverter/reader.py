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

from .ReadClasses import DicomReader, MhdReader, NiftiReader, StlReader, VtkReader, ThreeMfReader


class Reader:
    """
    Currently, reads in dicom files for modalities: CT, MR, DX, MG, US, RTSTRUCTS.

    The user inputs either a given folder path (can contain multiple images and subfolders). The files are sorted into
    separate images.

    Other user input options:
        file_list - if the user already has the files wanted to read in, must be in type list
        existing_images - if the user wants to not read in certain images that may already have been read in
        only_tags - does not read in the pixel array just the tags
        only_modality - specify which modalities to read in, if not then all modalities will be read
        only_load_roi_names - will only load rois with input name, list format
    """
    def __init__(self, folder_path=None, file_list=None, exclude_files=None, only_tags=False, only_modality=None,
                 only_load_roi_names=None):
        """
        User must input either folder_path or file_list, this will be used to determine which files to read in.
        If folder_path is input the "file_parsar" function runs to sort all the files into a dictionary of:
        Dicom, MHD, Raw, Stl, 3mf

        :param folder_path: single folder path
        :type folder_path: string path
        :param file_list: instead of folder_path a file_list can be given if the user already has the files to read in
        :type file_list: list
        :param exclude_files: if the user wants to not read certain files
        :type exclude_files: list of file
        :param only_tags: does not read in the pixel array just the tags
        :type only_tags: bool
        :param only_modality: specify which modalities to read in, if not then all modalities will be read
        :type only_modality: list
        :param only_load_roi_names: will only load rois with input name, list format
        :type only_load_roi_names: list
        """
        self.exclude_files = exclude_files
        self.only_tags = only_tags
        self.only_load_roi_names = only_load_roi_names
        if only_modality is not None:
            self.only_modality = only_modality
        else:
            self.only_modality = ['CT', 'MR', 'PT', 'US', 'DX', 'MG', 'NM', 'XA', 'CR', 'RTSTRUCT', 'REG', 'RTDose']

        self.files = None
        self.file_parsar(folder_path=folder_path, file_list=file_list, exclude_files=self.exclude_files)

        self.images = []
        self.rigid = []
        self.deformable = []
        self.dose = []
        self.meshes = []

    def file_parsar(self, folder_path=None, file_list=None, exclude_files=None):
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

        self.files = {'Dicom': dicom_files,
                      'MHD': mhd_files,
                      'Raw': raw_files,
                      'Nifti': nifti_files,
                      'Stl': stl_files,
                      'Vtk': vtk_files,
                      '3mf': mf3_files,
                      'NoExtension': no_file_extension}
        
    def read_dicoms(self):
        """
        Reads in all dicom files and separates them into the image list variable.
        :return:
        :rtype:
        """
        dicom_reader = DicomReader(self)
        dicom_reader.load()

    def read_rtstruct_only(self, base_image=None):
        print('reader')

    def read_mhd(self, match_image=None, create_contours=False):
        mf3_reader = MhdReader(self)
        mf3_reader.load()

    def read_nifti(self, match_image=None, create_contours=False):
        nifti_reader = NiftiReader(self)
        nifti_reader.load()

    def read_stl(self, create_image=False, match_image=None):
        stl_reader = StlReader(self)
        stl_reader.load()

    def read_vtk(self, create_image=False, match_image=None):
        vtk_reader = VtkReader(self)
        vtk_reader.load()

    def read_3mf(self, create_image=False, match_image=None):
        mf3_reader = ThreeMfReader(self)
        mf3_reader.load()


if __name__ == '_main__':
    pass
