"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Create DICOM image data from NumPy arrays, either as .dcm files written to disk or as an in-memory image
    registered in ``Data``.

    Methods
    -------
    * Files: :class:`CreateDicomImage` writes one .dcm file per slice of a 3D array.
    * In memory: :class:`CreateImageFromMask` builds DICOM headers for an array (typically a mask) and registers it
      as an :class:`Image` in ``Data.image``, so it behaves like an image read from DICOM.

    Conventions
    -----------
    * Arrays are (z, y, x): slice, row, column.
    * ``origin`` is the patient position (mm) of the first voxel of the first slice.
    * ``spacing`` is given in (x, y[, z]) order. DICOM PixelSpacing is (row, column) = (y, x), so the order is
      swapped when the tag is written.
    * ``orientation`` is the six ImageOrientationPatient values (row direction, column direction). Slices are
      stacked along their cross product, starting at ``origin``.
    * New Study, Series, Frame of Reference and SOP Instance UIDs are generated unless given.

Structure:
    _sop_class_uid               modality -> SOP Class UID
    _ds                          numbers -> DICOM decimal strings (max 16 characters)
    CreateDicomImage
    ├── __init__                 store array and geometry
    ├── set_*                    update UIDs / geometry before run
    └── run                      write one .dcm file per slice
    CreateImageFromMask
    ├── __init__                 build the per-slice DICOM headers
    ├── add_image                register as Image in Data.image
    ├── add_mesh_roi             attach a mesh as an ROI of the registered image
    └── compute_position         slice index -> patient position

Examples
--------
>>> import numpy as np
>>> import medicalimageanalysis as mia
>>> from medicalimageanalysis.utils.creation import CreateImageFromMask
>>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
>>> creator = CreateImageFromMask(mask, origin=[0, 0, 0], spacing=[1, 1, 2], image_name='Mask')
>>> creator.add_image()
>>> mia.Data.image['Mask'].array.shape
(8, 32, 32)
"""

import os
import datetime

import numpy as np
import pydicom as dicom
from pydicom.dataset import FileMetaDataset
from pydicom.valuerep import DSfloat
from pydicom.uid import (generate_uid, ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID, CTImageStorage,
                         MRImageStorage, PositronEmissionTomographyImageStorage)

from ..data import Data
from ..structure import Image


def _ds(values):
    """Converts numbers to DICOM decimal strings, rounded so each fits the 16-character DS limit.

    Parameters
    ----------
    values : array_like
        Numbers to convert.

    Returns
    -------
    list of pydicom.valuerep.DSfloat
        One decimal string per value.

    Examples
    --------
    >>> [str(v) for v in _ds([1, 2.5, -249.51171875])]
    ['1.0', '2.5', '-249.51171875']
    """

    return [DSfloat(float(v), auto_format=True) for v in np.ravel(values)]


def _sop_class_uid(modality):
    """Returns the image storage SOP Class UID for a modality.

    Parameters
    ----------
    modality : {'CT', 'MR', 'PT'}
        DICOM modality code. 'PET' is accepted as an alias for 'PT'.

    Returns
    -------
    pydicom.uid.UID
        The matching image storage SOP Class UID.

    Raises
    ------
    ValueError
        If the modality is not supported.

    Examples
    --------
    >>> _sop_class_uid('CT') == CTImageStorage
    True
    """

    if modality == 'CT':
        return CTImageStorage

    if modality == 'MR':
        return MRImageStorage

    if modality in ('PT', 'PET'):
        return PositronEmissionTomographyImageStorage

    raise ValueError(f"Unsupported modality: {modality}")


class CreateDicomImage(object):
    """Writes a 3D array to disk as a DICOM series, one .dcm file per slice.

    Parameters
    ----------
    output_dir : str
        Folder the .dcm files are written to. Created if it does not exist.
    data : (Z, Y, X) ndarray
        Voxel values. Written as signed 16-bit integers (floats are rounded), so values must fit in int16.
    study : str, optional
        Study Instance UID. Generated in :meth:`run` if None.
    series : str, optional
        Series Instance UID. Generated in :meth:`run` if None.
    frame : str, optional
        Frame of Reference UID. Generated in :meth:`run` if None.
    origin : (3,) array_like, optional
        Patient position (mm) of the first voxel. Defaults to [0, 0, 0].
    spacing : (2,) array_like, optional
        In-plane pixel spacing (x, y) in mm. Defaults to [1, 1].
    thickness : float, optional
        Slice thickness in mm, also used as the distance between slices. Defaults to 1.
    orientation : (6,) array_like, optional
        ImageOrientationPatient (row direction, column direction). Defaults to [1, 0, 0, 0, 1, 0].

    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> creator = CreateDicomImage(tempfile.mkdtemp(), np.zeros((4, 16, 16), dtype=np.int16), thickness=2)
    >>> len(creator.run())
    4
    """

    def __init__(self, output_dir, data, study=None, series=None, frame=None, origin=None, spacing=None,
                 thickness=None, orientation=None):
        """Stores the array, UIDs and geometry used by :meth:`run`.

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateDicomImage('C:/Data/output', np.zeros((4, 16, 16), dtype=np.int16))
        >>> creator.orientation
        [1, 0, 0, 0, 1, 0]
        """

        self.output_dir = output_dir
        self.data = data

        self.study = study
        self.series = series
        self.frame = frame

        self.origin = origin
        self.spacing = spacing
        self.thickness = thickness
        self.orientation = orientation if orientation is not None else [1, 0, 0, 0, 1, 0]

    def set_study(self, study):
        """Sets the Study Instance UID.

        Parameters
        ----------
        study : str
            Study Instance UID.

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateDicomImage('C:/Data/output', np.zeros((4, 16, 16), dtype=np.int16))
        >>> creator.set_study('1.2.3')
        """

        self.study = study

    def set_series(self, series):
        """Sets the Series Instance UID.

        Parameters
        ----------
        series : str
            Series Instance UID.

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateDicomImage('C:/Data/output', np.zeros((4, 16, 16), dtype=np.int16))
        >>> creator.set_series('1.2.3.4')
        """

        self.series = series

    def set_frame(self, frame):
        """Sets the Frame of Reference UID.

        Parameters
        ----------
        frame : str
            Frame of Reference UID.

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateDicomImage('C:/Data/output', np.zeros((4, 16, 16), dtype=np.int16))
        >>> creator.set_frame('1.2.3.4.5')
        """

        self.frame = frame

    def set_origin(self, origin):
        """Sets the patient position (mm) of the first voxel.

        Parameters
        ----------
        origin : (3,) array_like
            Origin (x, y, z).

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateDicomImage('C:/Data/output', np.zeros((4, 16, 16), dtype=np.int16))
        >>> creator.set_origin([-250, -250, 100])
        """

        self.origin = origin

    def set_spacing(self, spacing):
        """Sets the in-plane pixel spacing (mm).

        Parameters
        ----------
        spacing : (2,) array_like
            Spacing (x, y).

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateDicomImage('C:/Data/output', np.zeros((4, 16, 16), dtype=np.int16))
        >>> creator.set_spacing([0.98, 0.98])
        """

        self.spacing = spacing

    def set_thickness(self, thickness):
        """Sets the slice thickness (mm), also used as the distance between slices.

        Parameters
        ----------
        thickness : float
            Slice thickness.

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateDicomImage('C:/Data/output', np.zeros((4, 16, 16), dtype=np.int16))
        >>> creator.set_thickness(2.5)
        """

        self.thickness = thickness

    def run(self, patient_name='Test', patient_id='Test', modality='CT', description='', sex='M'):
        """Writes one .dcm file per slice to ``output_dir``.

        Missing UIDs and geometry are filled with their defaults and stored on the instance, so the generated UIDs
        can be read back afterwards.

        Parameters
        ----------
        patient_name : str
            Patient Name.
        patient_id : str
            Patient ID.
        modality : {'CT', 'MR', 'PT'}
            Modality, which also sets the SOP Class UID.
        description : str
            Series Description.
        sex : {'M', 'F', 'O'}
            Patient Sex.

        Returns
        -------
        list of str
            Paths of the written files, in slice order.

        Raises
        ------
        ValueError
            If the modality is not supported.

        Examples
        --------
        >>> import tempfile
        >>> import numpy as np
        >>> import pydicom
        >>> voxels = np.zeros((10, 64, 32), dtype=np.int16)
        >>> creator = CreateDicomImage(tempfile.mkdtemp(), voxels, spacing=[0.5, 0.8], thickness=2)
        >>> paths = creator.run(patient_name='Doe^John', modality='MR')
        >>> ds = pydicom.dcmread(paths[-1])
        >>> ds.Rows, ds.Columns, [float(v) for v in ds.PixelSpacing], float(ds.ImagePositionPatient[2])
        (64, 32, [0.8, 0.5], 18.0)
        """

        sop_class = _sop_class_uid(modality)

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

        data = np.asarray(self.data)
        if data.dtype != np.int16:
            data = np.rint(data).astype(np.int16)

        origin = np.asarray(self.origin, dtype=float)
        normal = np.cross(self.orientation[:3], self.orientation[3:6])

        now = datetime.datetime.now()
        date = now.strftime('%Y%m%d')
        time = now.strftime('%H%M%S')

        os.makedirs(self.output_dir, exist_ok=True)

        paths = []
        for ii in range(data.shape[0]):
            array = np.ascontiguousarray(data[ii])
            sop = generate_uid()

            ds = dicom.Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.MediaStorageSOPClassUID = sop_class
            ds.file_meta.MediaStorageSOPInstanceUID = sop
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.PatientName = patient_name
            ds.PatientID = patient_id
            ds.PatientSex = sex
            ds.Modality = modality
            ds.SeriesDescription = description
            ds.StudyDate = date
            ds.ContentDate = date
            ds.StudyTime = time
            ds.ContentTime = time

            ds.SOPClassUID = sop_class
            ds.SOPInstanceUID = sop
            ds.StudyInstanceUID = self.study
            ds.SeriesInstanceUID = self.series
            ds.FrameOfReferenceUID = self.frame
            ds.StudyID = '100'
            ds.SeriesNumber = '2'
            ds.AcquisitionNumber = '1'
            ds.InstanceNumber = str(ii + 1)

            ds.ImageOrientationPatient = _ds(self.orientation[:6])
            ds.ImagePositionPatient = _ds(origin + ii * self.thickness * normal)
            ds.PixelSpacing = _ds([self.spacing[1], self.spacing[0]])
            ds.SliceThickness = _ds([self.thickness])[0]

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.Rows = array.shape[0]
            ds.Columns = array.shape[1]
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1
            ds.PixelData = array.tobytes()

            path = os.path.join(self.output_dir, str(ii) + '.dcm')
            ds.save_as(path, write_like_original=False)
            paths.append(path)

        return paths


class CreateImageFromMask(object):
    """Builds DICOM headers for an array and registers it in ``Data.image`` as an :class:`Image`.

    The instance exposes the same attributes a DICOM reader does (``image_set``, ``array``, ``image_matrix``, ...),
    so :class:`Image` can be constructed from it directly. Nothing is written to disk.

    Parameters
    ----------
    array : (Z, Y, X) ndarray
        Image or mask array.
    origin : (3,) array_like
        Patient position (mm) of the first voxel.
    spacing : (3,) array_like
        Voxel spacing (x, y, z) in mm.
    image_name : str
        Key used in ``Data.image``.
    dimensions : tuple of int, optional
        Array dimensions (z, y, x). Defaults to ``array.shape``.
    orientation : (6,) array_like, optional
        ImageOrientationPatient (row direction, column direction). Defaults to [1, 0, 0, 0, 1, 0].
    plane : str
        Acquisition plane.
    description : str
        Series Description.
    modality : {'CT', 'MR', 'PT'}
        Modality, which also sets the SOP Class UID.

    Examples
    --------
    >>> import numpy as np
    >>> creator = CreateImageFromMask(np.zeros((8, 32, 16), dtype=np.uint8), [0, 0, 0], [1, 1, 2], 'Mask')
    >>> len(creator.image_set), creator.image_set[0].Rows, creator.image_set[0].Columns
    (8, 32, 16)
    """

    def __init__(self, array, origin, spacing, image_name, dimensions=None, orientation=None, plane='Axial',
                 description='Mask to Image', modality='CT'):
        """Stores the array and geometry and builds one DICOM header per slice in ``self.image_set``.

        Raises
        ------
        ValueError
            If the modality is not supported.

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateImageFromMask(np.zeros((8, 32, 32), dtype=np.uint8), [0, 0, 0], [1, 1, 2], 'Mask')
        >>> creator.image_matrix.tolist()
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """

        sop_class = _sop_class_uid(modality)

        self.array = array
        self.image_name = image_name
        self.modality = modality

        self.origin = np.asarray(origin, dtype=float)
        self.spacing = np.asarray(spacing, dtype=float)
        self.dimensions = tuple(dimensions) if dimensions is not None else array.shape
        self.orientation = list(orientation) if orientation is not None else [1, 0, 0, 0, 1, 0]
        self.plane = plane

        # Rows are the row, column and slice directions, matching the reader classes.
        row_direction = np.asarray(self.orientation[:3], dtype=float)
        column_direction = np.asarray(self.orientation[3:6], dtype=float)
        self.image_matrix = np.vstack((row_direction, column_direction, np.cross(row_direction, column_direction)))

        # Reader attributes that Image expects but do not apply to a created image.
        self.filepaths = None
        self.skipped_slice = None
        self.rgb = False

        self.sops = [generate_uid() for _ in range(self.dimensions[0])]

        study = generate_uid()
        series = generate_uid()
        frame = generate_uid()

        now = datetime.datetime.now()
        date = now.strftime('%Y%m%d')
        time = now.strftime('%H%M%S')

        self.image_set = []
        for ii in range(self.dimensions[0]):
            ds = dicom.Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.MediaStorageSOPClassUID = sop_class
            ds.file_meta.MediaStorageSOPInstanceUID = self.sops[ii]
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.PatientName = 'User^Created^ ^'
            ds.PatientID = 'User^Created^ ^'
            ds.PatientSex = 'M'
            ds.Modality = modality
            ds.SeriesDescription = description
            ds.StudyDate = date
            ds.ContentDate = date
            ds.StudyTime = time
            ds.ContentTime = time

            ds.SOPClassUID = sop_class
            ds.SOPInstanceUID = self.sops[ii]
            ds.StudyInstanceUID = study
            ds.SeriesInstanceUID = series
            ds.FrameOfReferenceUID = frame
            ds.StudyID = '1'
            ds.SeriesNumber = '1'
            ds.AcquisitionNumber = '1'
            ds.InstanceNumber = str(ii + 1)

            ds.ImageOrientationPatient = _ds(self.orientation[:6])
            ds.ImagePositionPatient = _ds(self.compute_position(ii))
            ds.PixelSpacing = _ds([self.spacing[1], self.spacing[0]])
            ds.SliceThickness = _ds([self.spacing[2]])[0]

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.Rows = self.dimensions[1]
            ds.Columns = self.dimensions[2]
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1

            self.image_set.append(ds)

    def add_image(self):
        """Creates an :class:`Image` from this instance and stores it as ``Data.image[image_name]``.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> import medicalimageanalysis as mia
        >>> creator = CreateImageFromMask(np.zeros((8, 32, 32), dtype=np.uint8), [0, 0, 0], [1, 1, 2], 'Mask')
        >>> creator.add_image()
        >>> 'Mask' in mia.Data.image
        True
        """

        Data.image[self.image_name] = Image(self)

    def add_mesh_roi(self, mesh, roi_name, color=None, visible=False):
        """Adds a mesh as an ROI of the registered image, calling :meth:`add_image` first if needed.

        Parameters
        ----------
        mesh : pyvista.PolyData
            Closed surface mesh in patient coordinates.
        roi_name : str
            ROI name.
        color : list of int, optional
            RGB color. Defaults to blue [0, 0, 255].
        visible : bool
            Initial ROI visibility.

        Returns
        -------
        None
            Sets the ROI's mesh, volume, com and bounds.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import medicalimageanalysis as mia
        >>> creator = CreateImageFromMask(np.zeros((8, 32, 32), dtype=np.uint8), [0, 0, 0], [1, 1, 2], 'Mask')
        >>> creator.add_mesh_roi(pv.Sphere(radius=5, center=(16, 16, 8)), 'Sphere')
        >>> 'Sphere' in mia.Data.image['Mask'].roi
        True
        """

        if self.image_name not in Data.image:
            self.add_image()

        image = Data.image[self.image_name]
        image.create_roi(name=roi_name, color=color if color is not None else [0, 0, 255], visible=visible)

        roi = image.roi[roi_name]
        roi.mesh = mesh
        roi.volume = mesh.volume
        roi.com = mesh.center
        roi.bounds = mesh.bounds

        Data.match_rois()

    def compute_position(self, z):
        """Returns the patient position (mm) of the first voxel of a slice.

        Parameters
        ----------
        z : int or float
            Slice index.

        Returns
        -------
        (3,) ndarray
            Position (x, y, z).

        Examples
        --------
        >>> import numpy as np
        >>> creator = CreateImageFromMask(np.zeros((8, 32, 32), dtype=np.uint8), [0, 0, 0], [1, 1, 2], 'Mask')
        >>> creator.compute_position(3).tolist()
        [0.0, 0.0, 6.0]
        """

        return self.origin + z * self.spacing[2] * self.image_matrix[2]