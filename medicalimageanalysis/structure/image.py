"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Medical imaging visualization library handling coordinate transformations,
    off-axis slice reslicing, and ROI/POI annotations for CT/MR datasets.

Structure:
    - Display: Manages slice viewing planes, coordinate transforms, and VTK reslicing.
    - Image: Holds volumetric arrays, metadata tags, and geometric properties.
"""

import os
import copy

import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk

from pydicom.uid import generate_uid
from scipy.spatial.transform import Rotation

import vtk
from vtkmodules.util import numpy_support

from ..utils.image.threshold import external
from ..utils.roi.contour import contours_from_mask

from .poi import Poi
from .roi import Roi
from ..data import Data


class Display(object):
    """
    Handles slice viewing states, image coordinate spaces, and off-axis reslicing updates.

    Parameters
              ----------
    image : Image
        The parent Image instance containing the underlying data matrix and metadata.
    """
    def __init__(self, image):
        self.image = image

        self.matrix = copy.deepcopy(self.image.matrix)
        self.spacing = copy.deepcopy(self.image.spacing)
        self.origin = copy.deepcopy(self.image.origin)

        self.slice_location = self.image.compute_center(position=False, zyx=True)
        self.scroll_max = [self.image.dimensions[0] - 1,
                           self.image.dimensions[1] - 1,
                           self.image.dimensions[2] - 1]
        self.secondary_array = None
        self.misc = {}

    def compute_matrix_pixel_to_position(self):
        """
        Computes the $4 \times 4$ homogeneous matrix converting pixel indices to physical coordinates.

        Returns
        -------
        numpy.ndarray
            A 4x4 matrix mapping [x, y, z, 1] index spaces to physical space.

        Examples
        --------
        >>> disp = Display(img)
        >>> tf_matrix = disp.compute_matrix_pixel_to_position()
        """
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
        """
        Computes the $4 \times 4$ homogeneous transformation matrix from physical coordinates to pixel space.

        Returns
        -------
        numpy.ndarray
            A 4x4 inverse coordinate transformation matrix.
        """
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / spacing[0]
        hold_matrix[1, :] = matrix[1, :] / spacing[1]
        hold_matrix[2, :] = matrix[2, :] / spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        return position_to_pixel_matrix

    def compute_array(self, slice_plane):
        """
        Extracts a single 2D slice from the volume on the specified anatomical standard plane.

        Parameters
        ----------
        slice_plane : str
            The desired orientation view plane. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        numpy.ndarray
            A float32 2D image matrix at the current active tracking slice location.
        """
        if self.secondary_array is None:
            if slice_plane == 'Axial':
                array = self.image.array[self.slice_location[0], :, :]
            elif slice_plane == 'Coronal':
                array = self.image.array[:, self.slice_location[1], :]
            else:
                array = self.image.array[:, :, self.slice_location[2]]
        else:
            if slice_plane == 'Axial':
                array = self.secondary_array[self.slice_location[0], :, :]
            elif slice_plane == 'Coronal':
                array = self.secondary_array[:, self.slice_location[1], :]
            else:
                array = self.secondary_array[:, :, self.slice_location[2]]

        return array.astype(np.float32)

    def compute_index_positions(self, xyz):
        """
        Converts pixel coordinates to an absolute physical 3D coordinate vector.

        Parameters
        ----------
        xyz : array_like
            A 3-element pixel coordinate vector representing [x, y, z] indexes.

        Returns
        -------
        numpy.ndarray
            A length-3 array containing the physical position values.
        """
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        location = np.asarray([xyz[0], xyz[1], xyz[2], 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def compute_offaxis_array(self):
        """
        Applies a VTK Reslice pipeline to interpolate the image volume data across non-orthogonal viewing axes.

        Returns
        -------
        None
        """
        loc = np.flip(self.slice_location)
        base_position_matrix = self.compute_matrix_pixel_to_position()
        slice_position = np.asarray([loc[0], loc[1], loc[2], 1]).dot(base_position_matrix.T)[:3]

        matrix_reshape = self.image.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.image.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions(np.flip(self.image.array.shape))
        vtk_image.SetOrigin(self.image.origin)
        vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(self.image.array.flatten(order="C")))

        matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, self.matrix[i, j])

        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transform.Inverse()

        vtk_reslice = vtk.vtkImageReslice()
        vtk_reslice.SetInputData(vtk_image)
        vtk_reslice.SetResliceTransform(transform)
        vtk_reslice.SetInterpolationModeToLinear()
        vtk_reslice.SetOutputSpacing(self.image.spacing)
        vtk_reslice.AutoCropOutputOn()
        vtk_reslice.SetBackgroundLevel(-3001)
        vtk_reslice.Update()

        reslice_data = vtk_reslice.GetOutput()
        new_origin = reslice_data.GetOrigin()
        self.origin = transform.TransformPoint(new_origin)
        dimensions = reslice_data.GetDimensions()

        position_to_pixel_matrix = self.compute_matrix_position_to_pixel()
        location = np.asarray([slice_position[0], slice_position[1], slice_position[2], 1])
        self.slice_location = list(np.flip(np.round(location.dot(position_to_pixel_matrix.T)[:3])).astype(np.int32))
        self.scroll_max = [dimensions[2] - 1, dimensions[1] - 1, dimensions[0] - 1]
        if self.slice_location[0] > dimensions[2] - 1:
            self.slice_location[0] = dimensions[2] - 1
        if self.slice_location[1] > dimensions[1] - 1:
            self.slice_location[1] = dimensions[1] - 1
        if self.slice_location[2] > dimensions[0] - 1:
            self.slice_location[2] = dimensions[0] - 1

        scalars = reslice_data.GetPointData().GetScalars()
        self.secondary_array = numpy_support.vtk_to_numpy(scalars).reshape(dimensions[2], dimensions[1], dimensions[0])

    def compute_scroll_max(self):
        """
        Recalculates maximum layout frame index limits for structural scroll interfaces.

        Returns
        -------
        None
        """
        if self.secondary_array is not None:
            self.scroll_max = [self.secondary_array.shape[0] - 1,
                               self.secondary_array.shape[1] - 1,
                               self.secondary_array.shape[2] - 1]
        else:
            self.scroll_max = [self.image.dimensions[0] - 1,
                               self.image.dimensions[1] - 1,
                               self.image.dimensions[2] - 1]

    def compute_vtk_slice(self, slice_plane):
        """
        Builds an independent 2D `vtkImageData` instance representing an individual orientation viewport plane.

        Parameters
        ----------
        slice_plane : str
            Target view plane orientation slice context. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        vtk.vtkImageData
            The structural 2D imaging data formatted explicitly for VTK pipeline visualization elements.
        """
        matrix_reshape = np.linalg.inv(self.matrix).reshape(1, 9)[0]
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        if slice_plane == 'Axial':
            location = np.asarray([0, 0, self.slice_location[0], 1])
            if self.secondary_array is None:
                array_slice = self.image.array[self.slice_location[0], :, :]
            else:
                array_slice = self.secondary_array[self.slice_location[0], :, :]
            array_shape = array_slice.shape
            dim = [array_shape[1], array_shape[0], 1]
        elif slice_plane == 'Coronal':
            location = np.asarray([0, self.slice_location[1], 0, 1])
            if self.secondary_array is None:
                array_slice = self.image.array[:, self.slice_location[1], :]
            else:
                array_slice = self.secondary_array[:, self.slice_location[1], :]
            array_shape = array_slice.shape
            dim = [array_shape[1], 1, array_shape[0]]
        else:
            location = np.asarray([self.slice_location[2], 0, 0, 1])
            if self.secondary_array is None:
                array_slice = self.image.array[:, :, self.slice_location[2]]
            else:
                array_slice = self.secondary_array[:, :, self.slice_location[2]]
            array_shape = array_slice.shape
            dim = [1, array_shape[1], array_shape[0]]

        slice_origin = location.dot(pixel_to_position_matrix.T)[:3]

        vtk_test = vtk.vtkImageData()
        vtk_test.SetSpacing(self.image.spacing)
        vtk_test.SetDirectionMatrix(matrix_reshape)
        vtk_test.SetDimensions(dim)
        vtk_test.SetOrigin(slice_origin)
        vtk_test.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array_slice.flatten(order="C")))

        return vtk_test

    def update_slice_location(self, scroll, slice_plane):
        """
        Updates the internal frame index for a targeted display plane view interface.

        Parameters
        ----------
        scroll : int
            The new coordinate plane view array frame index value.
        slice_plane : str
            Anatomical view tracking target label. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        None
        """
        if slice_plane == 'Axial':
            self.slice_location[0] = scroll
        elif slice_plane == 'Coronal':
            self.slice_location[1] = scroll
        else:
            self.slice_location[2] = scroll


class Image(object):
    """
    Main data class containing standard medical image volume blocks, metadata parsing, structures, and geometric transforms.

    Parameters
    ----------
    image : object
        A wrapper object tracking image arrays, properties, file locations, and DICOM field configurations.
    """
    def __init__(self, image):
        self.rois = {}
        self.pois = {}

        self.tags = image.image_set
        self.array = image.array

        self.image_name = image.image_name
        self.modality = image.modality

        self.patient_name = self.get_patient_name()
        self.mrn = self.get_mrn()
        self.birthdate = self.get_birthdate()
        self.date = self.get_date()
        self.time = self.get_time()
        self.local_uid = generate_uid()
        self.series_uid = self.get_series_uid()
        self.acq_number = self.get_acq_number()
        self.frame_ref = self.get_frame_ref()
        self.window = self.get_window()

        self.filepaths = image.filepaths
        self.sops = image.sops

        self.plane = image.plane
        self.spacing = image.spacing
        self.dimensions = image.dimensions
        self.orientation = image.orientation
        self.origin = image.origin
        self.matrix = image.image_matrix

        self.unverified = image.unverified
        self.skipped_slice = image.skipped_slice
        self.rgb = image.rgb

        self.camera_position = None

        self.visual = {'colormap': 'gray', 'bounds': None}
        self.misc = {}

        self.display = Display(self)

    def input_mhd(self, filename, roi_names, values, plane='Axial'):
        """
        Parses an explicit MetaImage (.mhd/.raw) dataset and adds segmented regions as tracking ROIs.

        Parameters
        ----------
        filename : str
            Full system path string linking directly to the MetaImage configuration header file.
        roi_names : list of str
            Label identifiers to map sequentially to each discrete voxel label segment.
        values : list of int
            The absolute voxel pixel configuration value flags used to isolate segmentation volumes.
        plane : str, default 'Axial'
            The principal acquisition structural reference frame.

        Returns
        -------
        None
        """
        roi_image = sitk.ReadImage(filename)
        roi_array = sitk.GetArrayFromImage(roi_image)
        for ii, roi_name in enumerate(roi_names):
            if roi_name not in list(self.rois.keys()):
                self.rois[roi_name] = Roi(self, name=roi_name, visible=True, filepaths=filename,
                                          plane=plane)

            roi_mask = roi_array == values[ii]
            self.rois[roi_name].convert_mask(roi_mask)

    def input_rtstruct(self, rtstruct):
        """
        Imports structured DICOM RT-Struct data elements, populating the local instances with ROI/POI objects.

        Parameters
        ----------
        rtstruct : object
            An imported container struct containing parsed contours, names, points, and colors.

        Returns
        -------
        None
        """
        for ii, roi_name in enumerate(rtstruct.roi_names):
            if roi_name not in list(self.rois.keys()):
                self.rois[roi_name] = Roi(self, position=rtstruct.contours[ii], name=roi_name,
                                          color=rtstruct.roi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

        for ii, poi_name in enumerate(rtstruct.poi_names):
            if poi_name not in list(self.pois.keys()):
                self.pois[poi_name] = Poi(self, position=rtstruct.points[ii], name=poi_name,
                                          color=rtstruct.poi_colors[ii], visible=False, filepaths=rtstruct.filepaths)

        Data.match_rois()
        Data.match_pois()

    def add_roi(self, roi_name=None, color=None, visible=False, path=None, contour=None, plane='Axial'):
        """
        Appends an explicitly instantiated Region of Interest object into the image volume structure tracking frame.

        Parameters
        ----------
        roi_name : str, optional
            A distinct tracking string key identifier.
        color : list of int, optional
            An RGB list of integers mapping structural color displays.
        visible : bool, default False
            Determines whether the structure maps to active slice plot view renders automatically.
        path : str, optional
            Direct file system tracking reference source link.
        contour : array_like, optional
            Coordinate positional boundary point datasets.
        plane : str, default 'Axial'
            Standard tracking plane orientation label.

        Returns
        -------
        None
        """
        self.rois[roi_name] = Roi(self, position=contour, name=roi_name, color=color, visible=visible, filepaths=path,
                                  plane=plane)
        Data.match_rois()

    def add_poi(self, poi_name=None, color=None, visible=False, path=None, point=None):
        """
        Appends a Point of Interest landmark element onto the active frame tracking tracking system.

        Parameters
        ----------
        poi_name : str, optional
            Landmark distinct name label configuration string.
        color : list of int, optional
            An RGB collection tracking visual markers.
        visible : bool, default False
            Active viewport display flag constraint state.
        path : str, optional
            System reference origin source path data details.
        point : array_like, optional
            Length-3 coordinate point setting absolute landmark position.

        Returns
        -------
        None
        """
        self.pois[poi_name] = Poi(self, position=point, name=poi_name, color=color, visible=visible, filepaths=path)
        Data.match_pois()

    def create_roi(self, name=None, color=None, visible=False, filepath=None):
        """
        Initializes an empty tracking ROI structural instance mapped onto the localized tracking frame.

        Parameters
        ----------
        name : str, optional
            Distinct label reference key identifier.
        color : list of int, optional
            RGB visual display settings profile tracking array.
        visible : bool, default False
            Active component visibility state configurations.
        filepath : str, optional
            Source processing storage folder context.

        Returns
        -------
        None
        """
        self.rois[name] = Roi(self, name=name, color=color, visible=visible, filepaths=filepath)
        Data.match_rois()

    def create_rtstruct(self, roi_names=None, poi_names=None):
        """
        Placeholder interface configuration method to handle downstream structured export pipeline generations.

        Parameters
        ----------
        roi_names : list of str, optional
            Target tracking segments to collect into structure.
        poi_names : list of str, optional
            Target landmarks tracking points array metrics.

        Returns
        -------
        None
        """
        pass

    def get_patient_name(self):
        """
        Parses PatientName field segments out from the current base DICOM data metadata structures.

        Returns
        -------
        list of str or str
            A parsed string collection containing name fragments, or 'missing'.
        """
        if 'PatientName' in self.tags[0]:
            return str(self.tags[0].PatientName).split('^')[:3]
        else:
            return 'missing'

    def get_mrn(self):
        """
        Extracts PatientID medical record number identifiers from the base header profiles.

        Returns
        -------
        str
            The tracked reference alphanumeric MRN string label context.
        """
        if 'PatientID' in self.tags[0]:
            return str(self.tags[0].PatientID)
        else:
            return 'missing'

    def get_birthdate(self):
        """
        Extracts PatientBirthDate metadata attributes directly out from the primary DICOM attributes profile.

        Returns
        -------
        str
            The text string date code tracking birth entries.
        """
        if 'PatientBirthDate' in self.tags[0]:
            return str(self.tags[0].PatientBirthDate)
        else:
            return ''

    def get_date(self):
        """
        Finds first available series or acquisition validation date indicators across sequential structural headers.

        Returns
        -------
        str
            A structured numeric string sequence identifying configuration transaction dates.
        """
        if 'SeriesDate' in self.tags[0]:
            return self.tags[0].SeriesDate
        elif 'ContentDate' in self.tags[0]:
            return self.tags[0].ContentDate
        elif 'AcquisitionDate' in self.tags[0]:
            return self.tags[0].AcquisitionDate
        elif 'StudyDate' in self.tags[0]:
            return self.tags[0].StudyDate
        else:
            return '00000'

    def get_time(self):
        """
        Finds chronological tracking timestamp variables across fallback structural dataset headers.

        Returns
        -------
        str
            The identified chronological session execution time value sequence.
        """
        if 'SeriesTime' in self.tags[0]:
            return self.tags[0].SeriesTime
        elif 'ContentTime' in self.tags[0]:
            return self.tags[0].ContentTime
        elif 'AcquisitionTime' in self.tags[0]:
            return self.tags[0].AcquisitionTime
        elif 'StudyTime' in self.tags[0]:
            return self.tags[0].StudyTime
        else:
            return '00000'

    def get_study_uid(self):
        """
        Queries tracking dataset files to expose unique master StudyInstanceUID identity profiles.

        Returns
        -------
        str
            The explicit string mapping absolute global study uniqueness values.
        """
        if 'StudyInstanceUID' in self.tags[0]:
            return self.tags[0].StudyInstanceUID
        else:
            return '00000.00000'

    def get_series_uid(self):
        """
        Queries tracking elements to extract explicit target SeriesInstanceUID system profiles.

        Returns
        -------
        str
            The string mapping detailed operational image series paths.
        """
        if 'SeriesInstanceUID' in self.tags[0]:
            return self.tags[0].SeriesInstanceUID
        else:
            return '00000.00000'

    def get_acq_number(self):
        """
        Locates clear acquisition tracking metrics inside individual data frames.

        Returns
        -------
        str
            The structured acquisition catalog identification key indicator string.
        """
        if 'AcquisitionNumber' in self.tags[0]:
            return self.tags[0].AcquisitionNumber
        else:
            return '1'

    def get_frame_ref(self):
        """
        Acquires FrameOfReferenceUID descriptors matching 3D geometric registration alignments.

        Returns
        -------
        str
            Unique alignment string verification system coordinates.
        """
        if 'FrameOfReferenceUID' in self.tags[0]:
            return self.tags[0].FrameOfReferenceUID
        else:
            return '00000.00000'

    def get_window(self):
        """
        Computes active visual window boundaries tracking WindowCenter and WindowWidth DICOM configurations.

        Returns
        -------
        list of int
            A 2-element collection defining absolute [Lower, Upper] grayscale threshold window parameters.
        """
        if (0x0028, 0x1050) in self.tags[0] and (0x0028, 0x1051) in self.tags[0]:
            center = self.tags[0].WindowCenter
            width = self.tags[0].WindowWidth

            if not isinstance(center, float):
                center = center[0]

            if not isinstance(width, float):
                width = width[0]

            return [int(center) - int(np.round(width / 2)), int(center) + int(np.round(width / 2))]

        elif self.array is not None:
            return [np.min(self.array), np.max(self.array)]

        else:
            return [0, 1]

    def get_specific_tag(self, tag):
        """
        Polls the initial index tracking dictionary config profile looking for targeted custom DICOM tags.

        Parameters
        ----------
        tag : str or tuple
            The custom explicit DICOM lookup sequence tracker.

        Returns
        -------
        object or None
            The raw localized tag content payload matching data lookups.
        """
        if tag in self.tags[0]:
            return self.tags[0][tag]
        else:
            return None

    def get_specific_tag_on_all_files(self, tag):
        """
        Loops through all sequential discrete file tags tracking uniform instance parameters.

        Parameters
        ----------
        tag : str or tuple
            The designated search identifier variable parameters.

        Returns
        -------
        list of object or None
            A collected list tracking sequential structural tag findings.
        """
        if tag in self.tags[0]:
            return [t[tag] for t in self.tags]
        else:
            return None

    def save_image(self, path, rois=True, pois=True):
        """
        Serializes data matrices, tags, metadata frames, and ROI trackers directly onto storage directories.

        Parameters
        ----------
        path : str
            The absolute system storage target path context directory.
        rois : bool, default True
            Determines whether to trigger standalone object loops archiving structure regions.
        pois : bool, default True
            Determines whether to process and store companion geometric point landmark items.

        Returns
        -------
        None
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['rois', 'pois', 'tags', 'array', 'display']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'tags.npy'), self.tags, allow_pickle=True)
        np.save(os.path.join(path, 'array.npy'), self.array, allow_pickle=True)

        if rois:
            self.save_rois(path, create_main_folder=True)

        if pois:
            self.save_pois(path, create_main_folder=True)

    def save_rois(self, path, create_main_folder=False):
        """
        Iterates over trackable ROI dictionary values to store serialized NumPy matrices configurations.

        Parameters
        ----------
        path : str
            The parent export folder path location context.
        create_main_folder : bool, default False
            When True, explicitly establishes a new nested directory sub-folder labeled 'ROIs'.

        Returns
        -------
        None
        """
        if create_main_folder:
            path = os.path.join(path, 'ROIs')
            os.mkdir(path)

        for name in list(self.rois.keys()):
            roi_path = os.path.join(os.path.join(path, name))
            os.mkdir(roi_path)

            np.save(os.path.join(roi_path, 'name.npy'), self.rois[name].name, allow_pickle=True)
            np.save(os.path.join(roi_path, 'visible.npy'), self.rois[name].visible, allow_pickle=True)
            np.save(os.path.join(roi_path, 'color.npy'), self.rois[name].color, allow_pickle=True)
            np.save(os.path.join(roi_path, 'filepaths.npy'), self.rois[name].filepaths, allow_pickle=True)
            if self.rois[name].contour_position is not None:
                np.save(os.path.join(roi_path, 'contour_position.npy'),
                        np.array(self.rois[name].contour_position, dtype=object),
                        allow_pickle=True)

    def save_pois(self, path, create_main_folder=False):
        """
        Saves individual Point of Interest coordinate datasets to disk.

        Parameters
        ----------
        path : str
            The targeted backup output destination directory path.
        create_main_folder : bool, default False
            When true, generates a structured container folder named 'POIs'.

        Returns
        -------
        None
        """
        if create_main_folder:
            path = os.path.join(path, 'POIs')
            os.mkdir(path)

        for name in list(self.pois.keys()):
            poi_path = os.path.join(os.path.join(path, name))
            os.mkdir(poi_path)

            np.save(os.path.join(poi_path, 'name.npy'), self.pois[name].name, allow_pickle=True)
            np.save(os.path.join(poi_path, 'visible.npy'), self.pois[name].visible, allow_pickle=True)
            np.save(os.path.join(poi_path, 'color.npy'), self.pois[name].color, allow_pickle=True)
            np.save(os.path.join(poi_path, 'filepaths.npy'), self.pois[name].filepaths, allow_pickle=True)
            np.save(os.path.join(poi_path, 'point_position.npy'), self.pois[name].point_position, allow_pickle=True)

    def load_image(self, image_path, rois=True, pois=True):
        """
        Loads and populates volumetric imaging elements out from saved system processing sub-directories.

        Parameters
        ----------
        image_path : str
            The system path linking directly to the archived target dataset root directory.
        rois : bool, default True
            Enables structural lookups parsing accompanying Region of Interest segments.
        pois : bool, default True
            Enables targeted reconstruction reading Point of Interest configuration arrays.

        Returns
        -------
        None
        """
        self.array = np.load(os.path.join(image_path, 'array.npy'), allow_pickle=True)
        self.tags = np.load(os.path.join(image_path, 'tags.npy'), allow_pickle=True)
        info = pd.read_pickle(os.path.join(image_path, 'info.p'), )
        for column in list(info.columns):
            setattr(self, column, info.at[0, column])

        if rois:
            roi_names = os.listdir(os.path.join(image_path, 'ROIs'))
            for name in roi_names:
                self.load_rois(os.path.join(image_path, 'ROIs', name))

        if pois:
            roi_names = os.listdir(os.path.join(image_path, 'POIs'))
            for name in roi_names:
                self.load_pois(os.path.join(image_path, 'POIs', name))

    def load_rois(self, roi_path):
        """
        Parses archived individual target ROI binary properties, formatting components to avoid index namespace conflicts.

        Parameters
        ----------
        roi_path : str
            Direct source storage context tracking individual mask files.

        Returns
        -------
        None
        """
        name = str(np.load(os.path.join(roi_path, 'name.npy'), allow_pickle=True))

        existing_rois = list(self.rois.keys())
        if name in existing_rois:
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_rois:
                    name = new_name
                    n = -1

        self.rois[name] = Roi(self)
        self.rois[name].name = name
        self.rois[name].visible = bool(np.load(os.path.join(roi_path, 'visible.npy'), allow_pickle=True))
        self.rois[name].color = list(np.load(os.path.join(roi_path, 'color.npy'), allow_pickle=True))
        self.rois[name].filepaths = str(np.load(os.path.join(roi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(roi_path, 'contour_position.npy')):
            self.rois[name].contour_position = list(np.load(os.path.join(roi_path, 'contour_position.npy'),
                                                            allow_pickle=True))

    def load_pois(self, poi_path):
        """
        Parses archived single target landmark configurations from disk.

        Parameters
        ----------
        poi_path : str
            Direct folder path to read specified landmark point geometries.

        Returns
        -------
        None
        """
        name = str(np.load(os.path.join(poi_path, 'name.npy'), allow_pickle=True))

        existing_pois = list(self.pois.keys())
        if name in existing_pois:
            n = 0
            while n >= 0:
                n += 1
                new_name = name + '_' + str(n)
                if new_name not in existing_pois:
                    name = new_name
                    n = -1

        self.pois[name] = poi(self)
        self.pois[name].name = name
        self.pois[name].visible = bool(np.load(os.path.join(poi_path, 'visible.npy'), allow_pickle=True))
        self.pois[name].color = list(np.load(os.path.join(poi_path, 'color.npy'), allow_pickle=True))
        self.pois[name].filepaths = str(np.load(os.path.join(poi_path, 'filepaths.npy'), allow_pickle=True))

        if os.path.exists(os.path.join(poi_path, 'point_position.npy')):
            self.rois[name].contour_position = list(np.load(os.path.join(poi_path, 'point_position.npy'),
                                                            allow_pickle=True))

    def create_sitk_image(self, empty=False):
        """
        Converts the active internal image data block matrix into a native SimpleITK image container.

        Parameters
        ----------
        empty : bool, default False
            When True, drops data allocation arrays to return a zero-initialized UInt8 volume container.

        Returns
        -------
        SimpleITK.Image
            The generated ITK core volumetric structure block tracking spacing and directions.
        """
        if empty:
            sitk_image = sitk.Image([int(dim) for dim in self.dimensions], sitk.sitkUInt8)
        else:
            sitk_image = sitk.GetImageFromArray(self.array)

        matrix_flat = self.matrix.flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        return sitk_image

    def create_rotated_sitk_image(self):
        """
        Applies a custom sample 3D Euler transformation rotation tracking localized anatomical structures.

        Returns
        -------
        numpy.ndarray
            The newly resampled volumetric data matrix block.
        """
        sitk_image = sitk.GetImageFromArray(self.array)
        matrix_flat = self.matrix.flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        transform = sitk.Euler3DTransform()
        transform.SetRotation(0, 0, 10 * np.pi / 180)
        transform.SetCenter(self.rois['Liver'].mesh.center)
        transform.SetComputeZYX(True)

        resample_image = sitk.ResampleImageFilter()
        resample_image.SetOutputDirection(sitk_image.GetDirection())
        resample_image.SetOutputOrigin(sitk_image.GetOrigin())
        resample_image.SetTransform(transform)
        resample_image.SetInterpolator(sitk.sitkLinear)
        resample_image.Execute(sitk_image)

        return sitk.GetArrayFromImage(resample_image)

    def create_external(self, name='External', color=None, visible=False, filepaths=None, threshold=-250):
        """
        Generates a continuous exterior bounding structural contour mask using an intensity threshold setting.

        Parameters
        ----------
        name : str, default 'External'
            Tracking key string mapped onto the generated ROI structure block.
        color : list of int, optional
            Custom visual color definition profile mapping. Defaults to bright green.
        visible : bool, default False
            Standard viewport visualization status tracker constraint.
        filepaths : str, optional
            Source processing storage location links.
        threshold : int, default -250
            The low Hounsfield Unit value or raw scalar intensity configuration cap.

        Returns
        -------
        None
        """
        if color is None:
            color = [0, 255, 0]

        if name not in list(self.rois.keys()):
            self.rois[name] = Roi(self, name=name, color=color, visible=visible, filepaths=filepaths)

        mask = external(self.array, threshold=threshold, only_mask=True)
        contours = contours_from_mask(mask.astype(np.uint8))
        positions = self.rois[name].convert_pixel_to_position(pixel=contours)

        self.rois[name].contour_pixel = contours
        self.rois[name].contour_position = positions
        self.rois[name].create_discrete_mesh()

    def compute_aspect(self, slice_plane):
        """
        Calculates viewport pixel aspect ratios required to prevent image skewing during display stretching.

        Parameters
        ----------
        slice_plane : str
            The viewport display target orientation frame. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        float
            The proportion scalar value rounded strictly to 2 decimal points.
        """
        if slice_plane == 'Axial':
            aspect = np.round(self.spacing[0] / self.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.spacing[0] / self.spacing[2], 2)
        else:
            aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        return aspect

    def compute_bounds(self):
        """
        Calculates absolute spatial bounding box ranges using VTK internal volume tracking logic.

        Returns
        -------
        list of float
            A 6-element list tracking spatial limits: [x_min, x_max, y_min, y_max, z_min, z_max].
        """
        shape = self.array.shape
        matrix_reshape = self.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions([shape[1], shape[2], shape[0]])
        vtk_image.SetOrigin(self.origin)

        x_min, x_max, y_min, y_max, z_min, z_max = vtk_image.GetBounds()

        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def compute_center(self, position=True, zyx=False):
        """
        Identifies mid-volume coordinate points in either pixel space grids or absolute physical systems.

        Parameters
        ----------
        position : bool, default True
            When True, transforms center coordinates into millimeter space. Otherwise keeps pixel indexes.
        zyx : bool, default False
            Flips positional vectors to sequence coordinates along inverted index trajectories.

        Returns
        -------
        list of int or numpy.ndarray
            The length-3 mid-volume structural position elements.
        """
        pixel_index = [int(self.dimensions[2] / 2),
                       int(self.dimensions[1] / 2),
                       int(self.dimensions[0] / 2)]

        if position:
            pixel_to_position_matrix = self.display.compute_matrix_pixel_to_position()
            location = np.asarray([pixel_index[0], pixel_index[1], pixel_index[2], 1])

            center = location.dot(pixel_to_position_matrix.T)[:3]
            if zyx:
                return np.flip(center)
            else:
                return center

        else:
            if zyx:
                return [pixel_index[2], pixel_index[1], pixel_index[0]]
            else:
                return pixel_index

    def compute_corner_positions(self):
        """
        Calculates explicit 3D physical location tracking positions for the eight bounding volume corners.

        Returns
        -------
        list of tuple
            A list containing eight distinct length-3 coordinate measurement tuples.
        """
        shape = self.array.shape
        matrix_reshape = self.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions([shape[1], shape[2], shape[0]])
        vtk_image.SetOrigin(self.origin)

        x_min, x_max, y_min, y_max, z_min, z_max = vtk_image.GetBounds()

        corner_points = [(x_min, y_min, z_min),
                         (x_max, y_min, z_min),
                         (x_max, y_max, z_min),
                         (x_min, y_max, z_min),
                         (x_min, y_min, z_max),
                         (x_max, y_min, z_max),
                         (x_max, y_max, z_max),
                         (x_min, y_max, z_max)]

        return corner_points

    def compute_corner_sides(self):
        """
        Generates a PyVista visual wireframe bounding box tracking the extreme dimensional corners.

        Returns
        -------
        pyvista.PolyData
            The generated surface data object ready for rendering pipelines.
        """
        corner_points = self.compute_corner_positions()
        points = [corner_points[0], corner_points[4], corner_points[7], corner_points[3],
                  corner_points[1], corner_points[2], corner_points[6], corner_points[5]]
        faces = [4, 0, 1, 2, 3,
                 4, 4, 5, 6, 7,
                 4, 0, 4, 7, 1,
                 4, 3, 2, 6, 5,
                 4, 0, 3, 5, 4,
                 4, 1, 7, 6, 2]

        return pv.PolyData(points, faces)

    def compute_pixel(self, position):
        """
        Transforms a physical coordinate location string back to fractional/integer index pixel coordinates.

        Parameters
        ----------
        position : array_like
            A 3-element physical point position configuration matrix tracking space.

        Returns
        -------
        numpy.ndarray
            An int32 array tracking specific structural pixel matrix indices.
        """
        matrix = copy.deepcopy(self.matrix)

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / self.spacing[0]
        hold_matrix[1, :] = matrix[1, :] / self.spacing[1]
        hold_matrix[2, :] = matrix[2, :] / self.spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        location = np.asarray([position[0], position[1], position[2], 1])

        return (np.round(location.dot(position_to_pixel_matrix.T)[:3])).astype(np.int32)

    def compute_position(self, xyz):
        """
        Transforms local pixel grid coordinates to physical 3D space locations.

        Parameters
        ----------
        xyz : array_like
            A 3-element index profile tracking data grid points.

        Returns
        -------
        numpy.ndarray
            A float32 coordinate array defining physical space coordinates.
        """
        matrix = copy.deepcopy(self.matrix)

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        location = np.asarray([xyz[0], xyz[1], xyz[2], 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def compute_matrix_pixel_to_position(self):
        """
        Helper method computing homogeneous index transformation systems.

        Returns
        -------
        numpy.ndarray
            A 4x4 coordinate tracking matrix.
        """
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
        """
        Helper method generating transformation matrices targeting internal structural transformations.

        Returns
        -------
        None
        """
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / spacing[0]
        hold_matrix[1, :] = matrix[1, :] / spacing[1]
        hold_matrix[2, :] = matrix[2, :] / spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

    def reset_array(self):
        """
        Clears out off-axis reslice transformations to re-establish standard viewing orientations.

        Returns
        -------
        None
        """
        self.display.secondary_array = None
        self.display.matrix = copy.deepcopy(self.matrix)
        self.display.origin = copy.deepcopy(self.origin)
        self.display.slice_location = self.compute_center(position=False, zyx=True)

    def retrieve_angles(self, order='ZXY'):
        """
        Converts the active viewing matrix configuration into standard Euler angles.

        Parameters
        ----------
        order : str, default 'ZXY'
            The specific axis processing rotation sequence layout constraint.

        Returns
        -------
        numpy.ndarray
            Calculated rotation angle vectors returned explicitly in degrees format.
        """
        rotation = Rotation.from_matrix(self.display.matrix[:3, :3])

        return rotation.as_euler(order, degrees=True)

    def retrieve_array_plane(self, slice_plane):
        """
        Retrieves the 2D image pixel tracking block matching specified active orientations.

        Parameters
        ----------
        slice_plane : str
            Target viewing matrix tracking context. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        numpy.ndarray
            The localized matrix tracking slice views.
        """
        return self.display.compute_array(slice_plane=slice_plane)

    def retrieve_slice_location(self, slice_plane):
        """
        Queries active coordinate locations for specific viewport projection elements.

        Parameters
        ----------
        slice_plane : str
            Desired standard anatomic plane descriptor profile label.

        Returns
        -------
        int
            The currently indexed matrix location tracking number.
        """
        if slice_plane == 'Axial':
            return self.display.slice_location[0]

        elif slice_plane == 'Coronal':
            return self.display.slice_location[1]

        else:
            return self.display.slice_location[2]

    def retrieve_slice_position(self, slice_plane=None):
        """
        Determines localized 3D physical position coordinates for tracking items across active planes.

        Parameters
        ----------
        slice_plane : str, optional
            Target structural monitoring views. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        numpy.ndarray
            The length-3 physical coordinate position vector tracking items.
        """
        pixel_to_position_matrix = self.display.compute_matrix_pixel_to_position()

        if slice_plane is None:
            location = np.asarray([self.display.slice_location[2],
                                   self.display.slice_location[1],
                                   self.display.slice_location[0], 1])
        else:
            if slice_plane == 'Axial':
                location = np.asarray([0, 0, self.display.slice_location[0], 1])
            elif slice_plane == 'Coronal':
                location = np.asarray([0, self.display.slice_location[1], 0, 1])
                print(location)
            else:
                location = np.asarray([self.display.slice_location[2], 0, 0, 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def retrieve_scroll_max(self, slice_plane):
        """
        Exposes extreme valid coordinate positions tracking specific frame scroll bounds.

        Parameters
        ----------
        slice_plane : str
            Target system mapping plane tracking labels.

        Returns
        -------
        int
            The maximum available structural index value.
        """
        if slice_plane == 'Axial':
            return self.display.scroll_max[0]

        elif slice_plane == 'Coronal':
            return self.display.scroll_max[1]

        else:
            return self.display.scroll_max[2]

    def retrieve_vtk_slice(self, slice_plane):
        """
        Direct wrapper passing calls to compile structural VTK image slices.

        Parameters
        ----------
        slice_plane : str
            The targeted anatomical plane tracking descriptor.

        Returns
        -------
        vtk.vtkImageData
            The structural VTK slice dataset object.
        """
        return self.display.compute_vtk_slice(slice_plane)

    def retrieve_vtk_volume(self, slice_plane):
        """
        Placeholder configuration mapping intended to provide automated support for full VTK volume blocks.

        Parameters
        ----------
        slice_plane : str
            Target orientation view labels tracker context.

        Returns
        -------
        object
            Downstream pipeline structural container components output.
        """
        return self.display.compute_vtk_volume(slice_plane)

    def update_rotation(self, r_x=0, r_y=0, r_z=0, base=True):
        """
        Applies incremental rotation updates across viewing matrices, triggering data interpolation pipelines.

        Parameters
        ----------
        r_x : float, default 0
            Rotation component applied explicitly around X-axis tracking paths in degrees.
        r_y : float, default 0
            Rotation component tracking Y-axis movements in degrees.
        r_z : float, default 0
            Rotation component tracking Z-axis movements in degrees.
        base : bool, default True
            When True, evaluates incremental modifications directly from original root geometry transforms.

        Returns
        -------
        None
        """
        if r_x != 0 or r_y != 0 or r_z != 0:
            r = Rotation.from_euler('xyz', [r_x, r_y, r_z], degrees=True)
            new_matrix = r.as_matrix()

            if base:
                base_matrix = copy.deepcopy(self.matrix)
                self.display.matrix = new_matrix @ base_matrix
            else:
                self.display.matrix = new_matrix @ self.display.matrix

            self.display.compute_offaxis_array()
            self.display.compute_scroll_max()
        else:
            self.display.compute_scroll_max()
            self.reset_array()