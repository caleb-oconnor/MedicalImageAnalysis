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

from .parsar import file_parsar
from .ReadClasses import DicomReader


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

        if folder_path is not None:
            self.files = file_parsar(folder_path, exclude_files=self.exclude_files)
        else:
            self.files = file_list

        self.images = []
        self.rigid = []
        self.deformable = []
        self.dose = []

    def read_all(self, dcm=False, mhd=False, stl=False, mf3=False):
        if dcm:
            dicom_reader = DicomReader(self)
            dicom_reader.load_dicom()

    def read_dicoms(self):
        """
        Reads in all dicom files and separates them into the image list variable.
        :return:
        :rtype:
        """
        dicom_reader = DicomReader(self)
        dicom_reader.load_dicoms()

    def read_rtstruct_only(self, base_image=None):
        print('reader')

    def read_mhd(self):
        print('reader')

    def read_nifti(self):
        print('reader')

    def read_stl(self, create_image=True):
        print('reader')

    def read_3mf(self, create_image=True):
        print('reader')


if __name__ == '_main__':
    pass
