"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import os
import datetime

import numpy as np
import pydicom as dicom
from pydicom.uid import generate_uid, UID, ExplicitVRLittleEndian

from ..data import Data


class CreateDicomImage(object):
    def __init__(self, output_dir, data, study=None, series=None, frame=None, origin=None, spacing=None,
                 thickness=None):
        self.output_dir = output_dir
        self.data = data
        self.study = study
        self.series = series
        self.frame = frame
        self.origin = origin
        self.spacing = spacing
        self.thickness = thickness

        self.orientation = [1, 0, 0, 0, 1, 0]

    def set_study(self, study):
        self.study = study

    def set_series(self, series):
        self.series = series

    def set_frame(self, frame):
        self.frame = frame

    def set_origin(self, origin):
        self.origin = origin

    def set_spacing(self, spacing):
        self.spacing = spacing

    def set_thickness(self, thickness):
        self.thickness = thickness

    def run(self):
        if self.study is None:
            self.study = generate_uid()
        if self.series is None:
            self.series = generate_uid()
        if self.frame is None:
            self.frame = generate_uid()
        if self.origin is None:
            self.origin = [0, 0, 0]
        if self.spacing is None:
            self.spacing = [1, 1]
        if self.thickness is None:
            self.thickness = 1

        for ii in range(self.data.shape[0]):
            array = self.data[ii, :, :]

            ds = dicom.Dataset()
            ds.file_meta = dicom.Dataset()
            ds.file_meta.ImplementationClassUID = "1.2.3.4"
            ds.file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
            ds.file_meta.MediaStorageSOPInstanceUID = str(10000 + ii)
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.PatientName = 'ForAI'
            ds.PatientSex = 'M'
            ds.SeriesDescription = 'Fake for AI'
            ds.PatientID = '12345'
            ds.Modality = 'CT'
            ds.StudyDate = str(datetime.date.today()).replace('-', '')
            ds.ContentDate = str(datetime.date.today()).replace('-', '')
            ds.StudyTime = str(10)
            ds.ContentTime = str(10)
            ds.StudyInstanceUID = self.study
            ds.SeriesInstanceUID = self.series
            ds.SOPInstanceUID = UID(str(10000 + ii))
            ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
            ds.StudyID = '100'

            ds.FrameOfReferenceUID = self.frame
            ds.AcquisitionNumber = '1'
            ds.SeriesNumber = '2'
            ds.InstanceNumber = str(ii + 1)
            ds.ImageOrientationPatient = self.orientation
            ds.PixelSpacing = self.spacing
            ds.SliceThickness = self.thickness
            ds.ImagePositionPatient = [self.origin[0], self.origin[1], (self.origin[2] + (ii * self.thickness))]

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 1
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.Columns = array.shape[0]
            ds.Rows = array.shape[1]
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1
            ds.PixelData = array.tobytes()

            export_file = os.path.join(self.output_dir, str(ii) + '.dcm')
            ds.save_as(export_file, write_like_original=False)


class CreateImageFromMask(object):
    def __init__(self, array, origin, spacing, image_name):
        self.rois = {}
        self.pois = {}

        self.array = array
        self.spacing = spacing
        self.origin = origin

        self.image_name = image_name

        now = datetime.datetime.now()
        self.date = str(now.year) + str(now.month) + str(now.day)
        if len(str(now.second)) == 1:
            self.time = str(now.hour) + '0' + str(now.second) + '00'
        else:
            self.time = str(now.hour) + str(now.second) + '00'
        self.birthdate = self.date

        self.filepaths = None

        self.plane = 'Axial'
        self.dimensions = array.shape
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.image_matrix = np.identity(3, dtype=np.float32)
        self.display_matrix = np.identity(4, dtype=np.float32)

        self.camera_position = None
        self.unverified = None
        self.skipped_slice = None
        self.sections = None
        self.rgb = False

        self.sops = self.sops = [generate_uid() for ii in range(self.dimensions[0])]
        self.slice_location = [int(self.dimensions[0] / 2), int(self.dimensions[1] / 2), int(self.dimensions[2] / 2)]

        self.rotated_array = None

        ds = dicom.Dataset()
        ds.file_meta = dicom.Dataset()
        ds.file_meta.ImplementationClassUID = "1.2.3.4"
        ds.file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
        ds.file_meta.MediaStorageSOPInstanceUID = str(10000)
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        self.study_uid = generate_uid()
        self.series_uid = generate_uid()
        self.series_uid = generate_uid()
        self.frame_ref = generate_uid()
        self.acq_number = 1
        self.window = [0, 1]
        self.modality = 'CT'

        ds.PatientName = 'User^Created^ ^'
        ds.PatientSex = 'M'
        ds.SeriesDescription = '3mf to Image'
        ds.PatientID = 'User^Created^ ^'
        ds.Modality = 'CT'
        ds.StudyDate = self.date
        ds.ContentDate = self.date
        ds.StudyTime = self.time
        ds.ContentTime = self.time
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = UID(str(10000))
        ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
        ds.StudyID = '100'

        ds.FrameOfReferenceUID = generate_uid()
        ds.AcquisitionNumber = '1'
        ds.SeriesNumber = '2'
        ds.InstanceNumber = str(1)
        ds.ImageOrientationPatient = self.orientation
        ds.PixelSpacing = spacing[:2]
        ds.SliceThickness = spacing[2]
        ds.ImagePositionPatient = [self.origin[0], self.origin[1], (self.origin[2] + spacing[2])]

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 1
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.Columns = array.shape[0]
        ds.Rows = array.shape[1]
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1

        self.image_set = [ds]

    def add_mesh_roi(self, mesh, roi_name):
        Data.image[self.image_name].create_roi(self, name=roi_name, color=[0, 0, 255], visible=False, filepath=None)
        self.rois[roi_name].mesh = mesh
        
        self.rois[roi_name].volume = mesh.volume
        self.rois[roi_name].com = mesh.center
        self.rois[roi_name].bounds = mesh.bounds


        