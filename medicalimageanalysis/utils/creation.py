"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Tools for generating DICOM image datasets and converting internal masks/arrays
    into standard DICOM image structures compatible with internal framework models.

Structure:
    - CreateDicomImage: Class to generate and write physical .dcm slice files from an array.
    - CreateImageFromMask: Class to instantiate frame-of-reference compliant dataset models.
"""

import os
import copy
import datetime

import numpy as np
import pydicom as dicom
from pydicom.uid import (generate_uid, UID, ExplicitVRLittleEndian, ImplicitVRLittleEndian, RTStructureSetStorage,
                         CTImageStorage, MRImageStorage, PositronEmissionTomographyImageStorage)

from ..data import Data
from ..structure import Image


class CreateDicomImage(object):
    """
    Handles the generation and physical serialization of standard DICOM images from voxel data.

    Parameters
    ----------
    output_dir : str
        The target directory path where generated DICOM (.dcm) files will be saved.
    data : numpy.ndarray
        A 3D numpy array representing image voxel intensities across slices.
    study : str, optional
        Unique identifier for the study UID. Generates automatically if None.
    series : str, optional
        Unique identifier for the series UID. Generates automatically if None.
    frame : str, optional
        Unique identifier for the frame of reference UID. Generates automatically if None.
    origin : list of float, optional
        The patient coordinates (X, Y, Z) of the first transmitted voxel. Defaults to [0, 0, 0].
    spacing : list of float, optional
        The physical distance (X, Y) between adjacent pixels. Defaults to [1, 1].
    thickness : float, optional
        The nominal thickness of the image slice along the Z-axis. Defaults to 1.
    """
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
        """
        Sets the Study Instance UID.

        Parameters
        ----------
        study : str
            The new Study Instance UID.
        """
        self.study = study

    def set_series(self, series):
        """
        Sets the Series Instance UID.

        Parameters
        ----------
        series : str
            The new Series Instance UID.
        """
        self.series = series

    def set_frame(self, frame):
        """
        Sets the Frame of Reference UID.

        Parameters
        ----------
        frame : str
            The new Frame of Reference UID.
        """
        self.frame = frame

    def set_origin(self, origin):
        """
        Sets the physical image origin coordinates.

        Parameters
        ----------
        origin : list of float
            The [X, Y, Z] components of the new origin.
        """
        self.origin = origin

    def set_spacing(self, spacing):
        """
        Sets the internal pixel grid resolution spacing.

        Parameters
        ----------
        spacing : list of float
            The [X, Y] spacing components.
        """
        self.spacing = spacing

    def set_thickness(self, thickness):
        """
        Sets the thickness configuration for each slice.

        Parameters
        ----------
        thickness : float
            The thickness value.
        """
        self.thickness = thickness

    def run(self, patient_name='Test', patient_id='Test', modality='CT', description='', sex='M'):
        """
        Executes the DICOM generation routine, writing each slice as an individual file to disk.

        Parameters
        ----------
        patient_name : str, optional
            The Patient Name attribute value. Defaults to 'Test'.
        patient_id : str, optional
            The Patient ID identifier string. Defaults to 'Test'.
        modality : str, optional
            The target modality code (e.g., 'CT', 'MR'). Defaults to 'CT'.
        description : str, optional
            A descriptive label for the generated series. Defaults to ''.
        sex : str, optional
            The Patient Sex demographic value ('M', 'F', 'O'). Defaults to 'M'.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> voxels = np.zeros((10, 512, 512), dtype=np.int16)
        >>> generator = CreateDicomImage('/path/to/output', voxels)
        >>> generator.run(patient_name='Doe^John', modality='MR')
        """
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
            ds.file_meta.ImplementationClassUID = generate_uid()
            if modality == 'CT':
                ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
            elif modality == 'MR':
                ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
            else:
                ds.file_meta.MediaStorageSOPClassUID = PositronEmissionTomographyImageStorage

            ds.file_meta.MediaStorageSOPInstanceUID = str(10000 + ii)
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.PatientName = patient_name
            ds.PatientSex = sex
            ds.SeriesDescription = description
            ds.PatientID = patient_id
            ds.Modality = modality
            ds.StudyDate = str(datetime.date.today()).replace('-', '')
            ds.ContentDate = str(datetime.date.today()).replace('-', '')
            ds.StudyTime = str(10)
            ds.ContentTime = str(10)
            ds.StudyInstanceUID = self.study
            ds.SeriesInstanceUID = self.series
            ds.SOPInstanceUID = UID(str(10000 + ii))
            ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
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
            ds.Columns = array.shape[1]
            ds.Rows = array.shape[0]
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1
            ds.PixelData = array.tobytes()

            export_file = os.path.join(self.output_dir, str(ii) + '.dcm')
            ds.save_as(export_file, write_like_original=False)


class CreateImageFromMask(object):
    """
    Constructs an internal image coordinate frame context from segmentation masks.

    Parameters
    ----------
    array : numpy.ndarray
        The input matrix data containing volume or binary masks.
    origin : list or numpy.ndarray
        The spatial starting point (X, Y, Z) coordinates of the matrix field.
    spacing : list or numpy.ndarray
        Voxel resolution spacing components [X_spacing, Y_spacing, Z_thickness].
    image_name : str
        The dictionary identifier key to assign this volume state inside `Data`.
    dimensions : tuple of int, optional
        Explicit spatial dimensions. Defaults to the shape layout of `array` if None.
    orientation : list of float, optional
        Six floating value components for orientation vectors. Defaults to identity mapping.
    plane : str, optional
        Anatomical viewing plane classification description string. Defaults to 'Axial'.
    description : str, optional
        Textual details stored in the Series Description metadata field. Defaults to 'Mask to Image'.
    modality : str, optional
        Target medical system scan string abbreviation. Defaults to 'CT'.
    """
    def __init__(self, array, origin, spacing, image_name, dimensions=None, orientation=None, plane='Axial',
                 description='Mask to Image', modality='CT'):
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

        self.plane = plane
        if dimensions is None:
            self.dimensions = array.shape
        else:
            self.dimensions = dimensions

        if orientation is None:
            self.orientation = [1, 0, 0, 0, 1, 0]
        else:
            self.orientation = orientation

        row_direction = self.orientation[:3]
        column_direction = self.orientation[3:6]
        slice_direction = np.cross(row_direction, column_direction)
        self.image_matrix = np.identity(3, dtype=np.float32)
        self.image_matrix[0, :3] = row_direction
        self.image_matrix[1, :3] = column_direction
        self.image_matrix[2, :3] = slice_direction

        self.camera_position = None
        self.unverified = None
        self.skipped_slice = None
        self.sections = None
        self.rgb = False

        self.sops = [generate_uid() for ii in range(self.dimensions[0])]
        self.slice_location = [int(self.dimensions[0] / 2), int(self.dimensions[1] / 2), int(self.dimensions[2] / 2)]

        self.study_uid = generate_uid()
        self.series_uid = generate_uid()
        self.frame_ref = generate_uid()
        self.acq_number = '1'
        self.window = [0, 1]
        self.modality = modality
        sop_class = generate_uid()

        dicoms = []
        for ii in range(self.dimensions[0]):
            ds = dicom.Dataset()
            ds.file_meta = dicom.Dataset()
            ds.file_meta.ImplementationClassUID = "1.2.3.4"
            ds.file_meta.MediaStorageSOPClassUID = UID(sop_class)
            ds.file_meta.MediaStorageSOPInstanceUID = UID(str(self.sops[ii]))
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.PatientName = 'User^Created^ ^'
            ds.PatientSex = 'M'
            ds.SeriesDescription = description
            ds.PatientID = 'User^Created^ ^'
            ds.Modality = modality
            ds.StudyDate = self.date
            ds.ContentDate = self.date
            ds.StudyTime = self.time
            ds.ContentTime = self.time
            ds.StudyInstanceUID = self.study_uid
            ds.SeriesInstanceUID = self.series_uid
            ds.SOPInstanceUID = UID(str(self.sops[ii]))
            ds.SOPClassUID = UID(str(sop_class))
            ds.StudyID = '1'

            ds.FrameOfReferenceUID = self.frame_ref
            ds.AcquisitionNumber = self.acq_number
            ds.SeriesNumber = '1'
            ds.InstanceNumber = str(ii)
            ds.ImageOrientationPatient = list(self.orientation[:6])
            ds.PixelSpacing = list(spacing[:2])
            ds.SliceThickness = spacing[2]

            position = self.compute_position(ii)
            ds.ImagePositionPatient = [position[0], position[1], position[2]]

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 1
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.Columns = array.shape[1]
            ds.Rows = array.shape[2]
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1

            dicoms += [ds]

        self.image_set = dicoms

    def add_image(self):
        """
        Registers the configured mask instance directly to global tracking states.

        Returns
        -------
        None
        """
        Data.image[self.image_name] = Image(self)
        Data.image_list += [self.image_name]

    def add_mesh_roi(self, mesh, roi_name):
        """
        Appends complex physical 3D mesh elements inside sub-region tracking fields.

        Parameters
        ----------
        mesh : object
            A mesh structure instance exposing metric fields (.volume, .center, .bounds).
        roi_name : str
            The lookup dictionary sub-key label name assigned to this ROI.

        Returns
        -------
        None
        """
        Data.image[self.image_name].create_roi(self, name=roi_name, color=[0, 0, 255], visible=False, filepath=None)
        self.rois[roi_name].mesh = mesh

        self.rois[roi_name].volume = mesh.volume
        self.rois[roi_name].com = mesh.center
        self.rois[roi_name].bounds = mesh.bounds

    def compute_position(self, z):
        """
        Calculates the real-world spatial position context of a slice index.

        Parameters
        ----------
        z : int or float
            The slice layer index stepping component along the direction axis.

        Returns
        -------
        numpy.ndarray
            A 3-element vector array denoting coordinates [X, Y, Z].
        """
        matrix = copy.deepcopy(self.image_matrix)

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        location = np.asarray([0, 0, z, 1])

        return location.dot(pixel_to_position_matrix.T)[:3]