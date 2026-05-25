"""
Morfeus Lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Radiation dose volume management, off-axis VTK reslicing, and multi-planar
    display handling. This module manages coordinate translations between pixel indexes
    and physical space for 3D dose matrices, alongside Dose-Volume Histogram (DVH) computation.

Structure:
    * Display: Controls orthogonal and off-axis reslice extraction using VTK pipelines.
    * Dose: Holds 3D dose arrays, handles pydicom metadata harvesting, and calculates ROI-specific metrics.
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

from ..data import Data


class Display(object):
    """
    Handles coordinate mapping, orthogonal slice extractions, and off-axis image reslicing.

    This class decouples volumetric properties from active viewports, providing support
    for linear coordinate matrices and real-time VTK dataset transformations.

    Attributes
    ----------
    dose : Dose
        Parent Dose instance owning raw data matrices and patient parameters.
    matrix : np.ndarray
        Active 3x3 orientation direction matrix mapping voxel axes to world coordinates.
    spacing : list or tuple
        Voxel grid sizes in millimeters across X, Y, and Z axes.
    origin : list or tuple
        Physical coordinates matching the current spatial index origin foundation.
    slice_location : list of int
        Active viewing plane indices [Z, Y, X] across orthogonal matrices.
    scroll_max : list of int
        Maximum index offsets allowed for bounded dimension updates.
    secondary_array : np.ndarray
        Cached off-axis array volume processed through VTK reslice filters.
    misc : dict
        Arbitrary configuration store tracking UI window configurations.
    """

    def __init__(self, dose):
        """
        Initializes a Display viewport tracking manager for an active Dose volume.

        Parameters
        ----------
        dose : Dose
            Parent dose dataset container object.
        """
        self.dose = dose

        self.matrix = copy.deepcopy(self.dose.matrix)
        self.spacing = copy.deepcopy(self.dose.spacing)
        self.origin = copy.deepcopy(self.dose.origin)

        self.slice_location = self.dose.compute_center(position=False, zyx=True)
        self.scroll_max = [self.dose.dimensions[0] - 1,
                           self.dose.dimensions[1] - 1,
                           self.dose.dimensions[2] - 1]
        self.secondary_array = None
        self.misc = {}

    def compute_matrix_pixel_to_position(self):
        """
        Calculates a 4x4 homogeneous matrix mapping discrete pixel indexes to continuous world coordinates.

        Returns
        -------
        np.ndarray
            Homogeneous 4x4 coordinate transformation array (float32).
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
        Calculates a 4x4 homogeneous transform matrix mapping continuous physical spaces to matrix voxel indexes.

        Returns
        -------
        np.ndarray
            Homogeneous 4x4 mapping array (float32) handling spatial index inversions.
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
        Extracts a 2D intensity cross-section matching the specified orientation path.

        Parameters
        ----------
        slice_plane : str
            The slicing orientation targeting dataset profiles. Must be 'Axial', 'Coronal', or 'Sagittal'.

        Returns
        -------
        np.ndarray
            Floating-point 2D matrix representing dose or image values across the selected viewport plane.
        """
        if self.secondary_array is None:
            if slice_plane == 'Axial':
                array = self.dose.array[self.slice_location[0], :, :]
            elif slice_plane == 'Coronal':
                array = self.dose.array[:, self.slice_location[1], :]
            else:
                array = self.dose.array[:, :, self.slice_location[2]]
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
        Translates explicit discrete matrix indexes into continuous spatial points.

        Parameters
        ----------
        xyz : list or np.ndarray
            Voxel grid pixel indexes along the primary dimensions.

        Returns
        -------
        np.ndarray
            Continuous 3D spatial world coordinate point (float32).
        """
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        location = np.asarray([xyz[0], xyz[1], xyz[2], 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def compute_offaxis_array(self):
        """
        Executes an off-axis volumetric reslicing routine using an interactive VTK image reslice pipeline.

        Extracts transformed slice arrays, mutates active internal orientation attributes,
        and dynamically fits tracking steps inside bounding constraints.
        """
        loc = np.flip(self.slice_location)
        base_position_matrix = self.compute_matrix_pixel_to_position()
        slice_position = np.asarray([loc[0], loc[1], loc[2], 1]).dot(base_position_matrix.T)[:3]

        matrix_reshape = self.dose.matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(self.dose.spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions(np.flip(self.dose.array.shape))
        vtk_image.SetOrigin(self.dose.origin)
        vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(self.dose.array.flatten(order="C")))

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
        vtk_reslice.SetOutputSpacing(self.dose.spacing)
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
        Recalculates discrete coordinate layout limiters to bound viewport sliders appropriately.
        """
        if self.secondary_array is not None:
            self.scroll_max = [self.secondary_array.shape[0] - 1,
                               self.secondary_array.shape[1] - 1,
                               self.secondary_array.shape[2] - 1]
        else:
            self.scroll_max = [self.dose.dimensions[0] - 1,
                               self.dose.dimensions[1] - 1,
                               self.dose.dimensions[2] - 1]

    def compute_vtk_slice(self, slice_plane):
        """
        Constructs a standalone vtkImageData slice wrapper instance matching target viewports.

        Parameters
        ----------
        slice_plane : str
            Target viewing matrix. Must be 'Axial', 'Coronal', or 'Sagittal'.

        Returns
        -------
        vtk.vtkImageData
            Independent single-slice VTK structure formatted with appropriate geometric properties.
        """
        matrix_reshape = np.linalg.inv(self.matrix).reshape(1, 9)[0]
        pixel_to_position_matrix = self.compute_matrix_pixel_to_position()
        if slice_plane == 'Axial':
            location = np.asarray([0, 0, self.slice_location[0], 1])
            if self.secondary_array is None:
                array_slice = self.dose.array[self.slice_location[0], :, :]
            else:
                array_slice = self.secondary_array[self.slice_location[0], :, :]
            array_shape = array_slice.shape
            dim = [array_shape[1], array_shape[0], 1]
        elif slice_plane == 'Coronal':
            location = np.asarray([0, self.slice_location[1], 0, 1])
            if self.secondary_array is None:
                array_slice = self.dose.array[:, self.slice_location[1], :]
            else:
                array_slice = self.secondary_array[:, self.slice_location[1], :]
            array_shape = array_slice.shape
            dim = [array_shape[1], 1, array_shape[0]]
        else:
            location = np.asarray([self.slice_location[2], 0, 0, 1])
            if self.secondary_array is None:
                array_slice = self.dose.array[:, :, self.slice_location[2]]
            else:
                array_slice = self.secondary_array[:, :, self.slice_location[2]]
            array_shape = array_slice.shape
            dim = [1, array_shape[1], array_shape[0]]

        slice_origin = location.dot(pixel_to_position_matrix.T)[:3]

        vtk_test = vtk.vtkImageData()
        vtk_test.SetSpacing(self.dose.spacing)
        vtk_test.SetDirectionMatrix(matrix_reshape)
        vtk_test.SetDimensions(dim)
        vtk_test.SetOrigin(slice_origin)
        vtk_test.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array_slice.flatten(order="C")))

        return vtk_test

    def update_slice_location(self, scroll, slice_plane):
        """
        Manually forces a new tracking row or column viewing index step for internal displays.

        Parameters
        ----------
        scroll : int
            The target index step level parameter to pass.
        slice_plane : str
            Orientation target key identifier. Must be 'Axial', 'Coronal', or 'Sagittal'.
        """
        if slice_plane == 'Axial':
            self.slice_location[0] = scroll
        elif slice_plane == 'Coronal':
            self.slice_location[1] = scroll
        else:
            self.slice_location[2] = scroll


class Dose(object):
    """
    Manages 3D medical radiation dose distribution grids and maps spatial DICOM attributes.

    Provides core tracking indicators for patient identification, image grid geometries,
    and implements parsing utilities to process regions of interest (ROI) alongside dose arrays.

    Attributes
    ----------
    tags : list
        Collection of raw pydicom dataset instances harvesting DICOM block elements.
    array : np.ndarray
        3D tensor holding floating-point radiation dose entry indicators.
    dose_name : str
        Registry name ID assigned to identify the active record.
    modality : str
        DICOM medical scanning metadata modality (e.g., 'RTDOSE').
    patient_name : str or list
        Extracted patient naming metrics parsed from metadata tags.
    mrn : str
        Patient Medical Record Number identification string.
    birthdate : str
        Patient birthdate attribute registry token string.
    date : str
        DICOM series capturing chronological date token entries.
    time : str
        DICOM collection timestamp registry tracker.
    local_uid : str
        Internal execution token key generated for identification pipelines.
    series_uid : str
        Unique identifier tagging the source image series context.
    acq_number : str or int
        DICOM scan sequence acquisition cycle tracking number.
    frame_ref : str
        Frame of Reference identification identifier linking datasets spatially.
    window : list
        Contrast presentation levels containing standard window minimum and maximum pairs.
    filepaths : list of str
        Source directory paths providing raw files.
    sops : list of str
        Service-Object Pair Instance UIDs identifying individual items.
    plane : str
        Primary image orientation definition name.
    spacing : list or tuple
        Physical metric distance between voxel nodes in millimeters.
    dimensions : tuple or list
        Matrix voxel limits representing tensor boundaries.
    orientation : str
        Spatial position layout definitions.
    origin : list or tuple
        Continuous world coordinate point representing index foundations.
    matrix : np.ndarray
        Direction mapping orientation configurations relating matrices to real space.
    camera_position : list or None
        Cached perspective camera spatial coordinates for visualization scenes.
    misc : dict
        Arbitrary configuration store tracking analytical parameter sets.
    rois : dict
        Collection tracking regions of interest linked to the dataset module.
    display : Display
        Active view reconstruction component parsing orthogonal mapping matrices.
    """

    def __init__(self, dose):
        """
        Initializes a Dose distribution data record tracking structural arrays and attributes.
        """
        self.tags = dose.image_set
        self.array = dose.array

        self.dose_name = dose.dose_name
        self.modality = dose.modality

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

        self.filepaths = dose.filepaths
        self.sops = dose.sops

        self.plane = dose.plane
        self.spacing = dose.spacing
        self.dimensions = dose.dimensions
        self.orientation = dose.orientation
        self.origin = dose.origin
        self.matrix = dose.image_matrix

        self.camera_position = None
        self.misc = {}

        self.rois = {}
        self.display = Display(self)

    def get_patient_name(self):
        """
        Extracts structured patient identity strings from parsed DICOM metadata.

        Returns
        -------
        list of str or str
            Splits text strings into component components, returns 'missing' on failure.
        """
        if 'PatientName' in self.tags[0]:
            return str(self.tags[0].PatientName).split('^')[:3]
        else:
            return 'missing'

    def get_mrn(self):
        """
        Queries patient Medical Record Numbers from loaded header definitions.

        Returns
        -------
        str
            Parsed identification index string, or 'missing'.
        """
        if 'PatientID' in self.tags[0]:
            return str(self.tags[0].PatientID)
        else:
            return 'missing'

    def get_birthdate(self):
        """
        Queries patient birthday sequences from metadata fields.

        Returns
        -------
        str
            Chronological date metadata context, or an empty string.
        """
        if 'PatientBirthDate' in self.tags[0]:
            return str(self.tags[0].PatientBirthDate)
        else:
            return ''

    def get_date(self):
        """
        Inspects progressive acquisition dates to establish creation timelines.

        Returns
        -------
        str
            Chronological timestamp data string, defaults to '00000'.
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
        Inspects dataset headers to extract logging time flags.

        Returns
        -------
        str
            Temporal timestamp registry identifier context string.
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
        Queries global unique Study Instance UIDs.

        Returns
        -------
        str
            DICOM study identification context string token.
        """
        if 'StudyInstanceUID' in self.tags[0]:
            return self.tags[0].StudyInstanceUID
        else:
            return '00000.00000'

    def get_series_uid(self):
        """
        Queries global unique Series Instance UIDs.

        Returns
        -------
        str
            DICOM series identification context string token.
        """
        if 'SeriesInstanceUID' in self.tags[0]:
            return self.tags[0].SeriesInstanceUID
        else:
            return '00000.00000'

    def get_acq_number(self):
        """
        Queries metadata blocks to establish dataset acquisition indexes.

        Returns
        -------
        str
            Sequence location descriptor, or '1'.
        """
        if 'AcquisitionNumber' in self.tags[0]:
            return self.tags[0].AcquisitionNumber
        else:
            return '1'

    def get_frame_ref(self):
        """
        Queries unique spatial coordinate Frame of Reference tokens.

        Returns
        -------
        str
            Geometric framework indexing mapping string.
        """
        if 'FrameOfReferenceUID' in self.tags[0]:
            return self.tags[0].FrameOfReferenceUID
        else:
            return '00000.00000'

    def get_window(self):
        """
        Calculates optimal presentation window widths for window/level rendering systems.

        Returns
        -------
        list of int
            Contains the calculated intensity boundary pair [WindowMin, WindowMax].
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
        Queries arbitrary target metadata entries directly from DICOM frames.

        Parameters
        ----------
        tag : str or tuple
            Target DICOM keyword string or hex identifier tag to evaluate.

        Returns
        -------
        pydicom.dataelem.DataElement or None
            Raw header attribute data if verified inside dataset records.
        """
        if tag in self.tags[0]:
            return self.tags[0][tag]
        else:
            return None

    def compute_aspect(self, slice_plane):
        """
        Extracts directional pixel aspect scaling multipliers for 2D visual projection pipelines.

        Parameters
        ----------
        slice_plane : str
            Target viewing matrix axis. Must be 'Axial', 'Coronal', or 'Sagittal'.

        Returns
        -------
        float
            Aspect scaling multiplier ratio.
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
        Calculates absolute spatial bounding extents enclosing the image configuration.

        Returns
        -------
        list of float
            Physical space limits organized as [Xmin, Xmax, Ymin, Ymax, Zmin, Zmax].
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
        Calculates the exact volumetric center voxel or its matching physical point.

        Parameters
        ----------
        position : bool, optional
            Converts target indexes into continuous physical coordinates if True. Defaults to True.
        zyx : bool, optional
            Flips array dimensional tracking layouts to ZYX order if True. Defaults to False.

        Returns
        -------
        list or np.ndarray
            Calculated 3D center spatial point vector or matrix indices.
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
        Extracts coordinates for all eight bounding corners of the continuous image cube.

        Returns
        -------
        list of tuple
            Collection containing physical 3D spatial points mapping volume apex boundaries.
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
        Constructs a 3D structural mesh block modeling outer surface bounding walls.

        Returns
        -------
        pyvista.PolyData
            Unstructured PyVista surface dataset enclosing the volume block profile.
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

    def compute_dose_statistics(self):
        """
        Placeholder method for global radiation dose volumetric data evaluations.
        """
        pass

    def compute_roi_dose_array(self, image_name, roi_name):
        """
        Isolates structural dose entries bounded inside a regional segmentation mask.

        Resamples dose matrices onto reference dimensions to align structural points.

        Parameters
        ----------
        image_name : str
            Label tracking reference structural anatomical volumes.
        roi_name : str
            Target structure region identifier key.

        Returns
        -------
        np.ndarray
            Flattened 1D array collecting floating dose value intersections localized inside masks.
        """
        dose_image = self.create_sitk_image()
        image = Data.image[image_name].create_sitk_image()
        roi_image = Data.image[image_name].rois[roi_name].create_sitk_mask()

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        dose_resampled = resampler.Execute(dose_image)

        dose_arr = sitk.GetArrayFromImage(dose_resampled)
        mask_arr = sitk.GetArrayFromImage(roi_image)

        mask_indices = np.where(mask_arr > 0)
        dose_in_roi = dose_arr[mask_indices]

        return dose_in_roi

    def compute_roi_dose_statistics(self, image_name, roi_name, max_dose=150, increment=5):
        """
        Generates full analytical indicators and cumulative data for Dose-Volume Histograms (DVH).

        Parameters
        ----------
        image_name : str
            Label tracking references pointing to primary imaging volumes.
        roi_name : str
            Target anatomical outline region descriptor name.
        max_dose : int, optional
            Upper limit bounding dose tracking arrays (Gy). Defaults to 150.
        increment : int, optional
            Step intervals computing structural coverage states. Defaults to 5.

        Returns
        -------
        dict
            Dose volume metrics tracking volume (cc), min, max, mean, D-percents, and VS parameters.
        """
        spacing = Data.image[image_name].spacing
        dose_in_roi = self.compute_roi_dose_array(image_name, roi_name)

        voxel_vol_cc = np.prod(spacing) / 1000.0
        roi_volume_cc = dose_in_roi.size * voxel_vol_cc

        dvh = {"ROI": roi_name,
               "Volume (cc)": roi_volume_cc,
               "Dmin": np.min(dose_in_roi),
               "Dmax": np.max(dose_in_roi),
               "Dmean": np.mean(dose_in_roi),
               "Dmedian": np.median(dose_in_roi),
               "Dstd": np.std(dose_in_roi)}

        d_values = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98, 99]
        dvh.update({f"D{d}": np.percentile(dose_in_roi, 100 - d) for d in d_values})
        voxel_vol_cc = np.prod(spacing) / 1000.0
        for d in range(0, max_dose + increment, 5):
            mask = dose_in_roi < d
            dvh[f"VS{d}Gy_percent"] = np.mean(mask) * 100
            dvh[f"VS{d}Gy_cc"] = np.sum(mask) * voxel_vol_cc

        return dvh

    def compute_pixel(self, position):
        """
        Maps continuous 3D physical positions to absolute discrete voxel indices.

        Parameters
        ----------
        position : list or np.ndarray
            Continuous world coordinate entries mapping spatial points.

        Returns
        -------
        np.ndarray
            Discrete 3D pixel array coordinates (int32).
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
        Maps discrete voxel indexes directly back to continuous physical locations.

        Parameters
        ----------
        xyz : list or np.ndarray
            Discrete matrix coordinates mapping matrix pixel centers.

        Returns
        -------
        np.ndarray
            Continuous 3D spatial coordinate vector (float32).
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
        Calculates an intrinsic 4x4 coordinate transform matrix mapping voxel nodes to spatial spaces.

        Returns
        -------
        np.ndarray
            Homogeneous translation coordinate mapping tensor matrix.
        """
        matrix = copy.deepcopy(self.matrix)
        spacing = self.spacing

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def create_sitk_image(self, empty=False):
        """
        Generates an independent SimpleITK image container wrapped with current orientation profiles.

        Parameters
        ----------
        empty : bool, optional
            Builds a blank 8-bit unsigned integer placeholder image array if True. Defaults to False.

        Returns
        -------
        SimpleITK.Image
            Configured ITK data object holding metadata alignments.
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

    def reset_array(self):
        """
        Clears secondary off-axis caches, restoring tracking properties back to baseline states.
        """
        self.display.secondary_array = None
        self.display.matrix = copy.deepcopy(self.matrix)
        self.display.origin = copy.deepcopy(self.origin)
        self.display.slice_location = self.compute_center(position=False, zyx=True)

    def retrieve_angles(self, order='ZXY'):
        """
        Extracts directional Euler angles representing current viewport rotations.

        Parameters
        ----------
        order : str, optional
            Target axis sequence constraints formatting outputs. Defaults to 'ZXY'.

        Returns
        -------
        np.ndarray
            Rotation degrees calculated across coordinate planes.
        """
        rotation = Rotation.from_matrix(self.display.matrix[:3, :3])

        return rotation.as_euler(order, degrees=True)

    def retrieve_array_plane(self, slice_plane):
        """
        Queries 2D view arrays along chosen projection slicing cuts.

        Parameters
        ----------
        slice_plane : str
            Target viewing orientation axis.

        Returns
        -------
        np.ndarray
            2D scalar distribution matrix data slice.
        """
        return self.display.compute_array(slice_plane=slice_plane)

    def retrieve_slice_location(self, slice_plane):
        """
        Queries discrete index tracking values currently intersecting active canvases.

        Parameters
        ----------
        slice_plane : str
            Target orthogonal workspace plane key name.

        Returns
        -------
        int
            Voxel matrix index tracking step level.
        """
        if slice_plane == 'Axial':
            return self.display.slice_location[0]

        elif slice_plane == 'Coronal':
            return self.display.slice_location[1]

        else:
            return self.display.slice_location[2]

    def retrieve_slice_position(self, slice_plane=None):
        """
        Maps current slice visualization indexes back into continuous world 3D position spaces.

        Parameters
        ----------
        slice_plane : str, optional
            Target viewport section plane profile tracker. Defaults to None.

        Returns
        -------
        np.ndarray
            Continuous 3D world coordinate vector.
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
        Queries upper bounding index ranges for scroll components.

        Parameters
        ----------
        slice_plane : str
            Target viewport tracker orientation name.

        Returns
        -------
        int
            Maximum allowed index position tracking step.
        """
        if slice_plane == 'Axial':
            return self.display.scroll_max[0]

        elif slice_plane == 'Coronal':
            return self.display.scroll_max[1]

        else:
            return self.display.scroll_max[2]

    def retrieve_vtk_slice(self, slice_plane):
        """
        Queries independent VTK slice representations matching active view planes.

        Parameters
        ----------
        slice_plane : str
            Target plane tracking workspace identifier.

        Returns
        -------
        vtk.vtkImageData
            Structured single-slice collection array.
        """
        return self.display.compute_vtk_slice(slice_plane)

    def retrieve_vtk_volume(self, slice_plane):
        """
        Queries independent full VTK volume blocks configured inside displays.

        Parameters
        ----------
        slice_plane : str
            Target processing plane context wrapper name.

        Returns
        -------
        vtk.vtkImageData
            Structured multi-channel data.
        """
        return self.display.compute_vtk_volume(slice_plane)

    def save_image(self, path):
        """
        Serializes structural metadata tables and tensor data files to disk targets.

        Parameters
        ----------
        path : str
            File system folder destination targeted to write files.
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['tags', 'array', 'display', 'rois']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'tags.npy'), self.tags, allow_pickle=True)
        np.save(os.path.join(path, 'array.npy'), self.array, allow_pickle=True)

    def update_rotation(self, r_x=0, r_y=0, r_z=0, base=True):
        """
        Applies extrinsic directional Euler rotations to recalculate off-axis visualization spaces.

        Forces structural arrays through oblique transformation filters using underlying
        VTK matrix interpolation models.

        Parameters
        ----------
        r_x : float, optional
            Rotation angle applied around the spatial X axis. Defaults to 0.
        r_y : float, optional
            Rotation angle applied around the spatial Y axis. Defaults to 0.
        r_z : float, optional
            Rotation angle applied around the spatial Z axis. Defaults to 0.
        base : bool, optional
            Resets properties back to native image matrices before execution if True. Defaults to True.
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
