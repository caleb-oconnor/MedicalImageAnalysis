
from parsar import file_parsar
from ReadClasses.dicom import DicomReader


class Reader:
    def __init__(self, folder_path=None, file_list=None, exclude_files=None):
        if folder_path is not None:
            self.files = file_parsar(folder_path, exclude_files=exclude_files)
        else:
            self.files = file_list

        self.images = None
        self.rigid = None
        self.deformable = None
        self.pois = None
        self.dose = None

    def read_all(self, dcm=None, mhd=None, stl=None, mf3=None):
        if dcm is not None:
            dicom_reader = DicomReader(self.files)

    def read_dicoms(self, only_tags=False, only_load_roi_names=None):
        if only_tags:
            print('reader')
        else:
            dicom_reader = DicomReader(self.files, only_load_roi_names=only_load_roi_names)

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


if __name__ == '__main__':
    pass
