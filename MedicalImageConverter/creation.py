
import os
import datetime

import numpy as np
import pydicom as dicom
from pydicom.uid import generate_uid, UID, ExplicitVRLittleEndian


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
            self.origin = [0 ,0, 0]
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
