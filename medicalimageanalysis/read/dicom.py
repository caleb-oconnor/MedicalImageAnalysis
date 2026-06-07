"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

DICOM Pipeline Orchestrator and Volumetric Reconstructor
========================================================

Description:
    This module serves as the primary ingestion engine for DICOM data. It
    facilitates the transition from raw pydicom datasets to structured,
    physically-accurate 3D volumes and 2D images within the global `Data` state.

Core Components:
    1. **DicomReader**: The high-level orchestrator. It manages multithreaded
       file reading, filters by modality, and groups individual slices into
       logical series based on `SeriesInstanceUID` and spatial orientation.
    2. **Read3D**: The volumetric engine for CT, MR, and PT. It performs the
       heavy lifting of slice stacking, verifying physical spacing, detecting
       missing slices, and computing the Patient-to-Voxel coordinate matrix.
    3. **ReadXRay / ReadRF / ReadUS**: Specialized 2D and pseudo-3D readers
       that normalize non-volumetric modalities into a consistent format
       compatible with the application's internal Image class.
    4. **Sorting Logic**: Ensures that disparate imaging series are ordered
       chronologically using acquisition date and time metadata for
       longitudinal analysis.

Geometric Processing:
    The module implements rigorous DICOM coordinate system logic, utilizing
    `ImageOrientationPatient` and `ImagePositionPatient` to:
    - Determine anatomical planes (Axial, Coronal, Sagittal).
    - Construct 3x3 orientation matrices.
    - Validate slice continuity and handle irregular acquisitions.

Thread Safety & Memory Management:
    - Utilizes `threading` for I/O bound DICOM parsing.
    - Explicitly deletes `PixelData` buffers after NumPy array conversion to
      minimize RAM overhead—critical for processing large medical datasets.

Usage:
    >>> files = {"Dicom": ["/path/to/slice1.dcm", "/path/to/slice2.dcm"]}
    >>> reader = DicomReader(files, only_tags=False, clear=True)
    >>> reader.load()
    >>> # Volumes are now available in Data.image

"""

import copy
import time
import gdcm
import threading
from struct import unpack

import numpy as np
import pydicom as dicom
from openpyxl.descriptors import NoneSet
from pydicom.uid import generate_uid

from ..structure.deformable import Deformable
from ..structure.dose import Dose
from ..structure.image import Image
from ..structure.rigid import Rigid

from ..data import Data


def sort_images_by_datetime():
    """
    Reorder the global `Data.image` dictionary and `Data.image_list`
    based on DICOM acquisition date and time.

    Sorting is performed lexicographically on:
    `str(date) + str(time)`
    """
    date_time = [
        str(Data.image[name].date) + str(Data.image[name].time)
        for name in Data.image_list
    ]

    new_key_order = [
        Data.image_list[idx] for idx in np.argsort(date_time)
    ]

    Data.image = {key: Data.image[key] for key in new_key_order}
    Data.image_list = list(Data.image.keys())


def thread_process_dicom(path, stop_before_pixels=False):
    """
    Read a DICOM file using pydicom in a thread-safe manner.

    Parameters
    ----------
    path : str
        Path to the DICOM file.
    stop_before_pixels : bool, optional
        If True, only metadata is loaded (no pixel data).

    Returns
    -------
    pydicom.dataset.FileDataset or list
        Parsed DICOM dataset or empty list on failure.
    """
    try:
        datasets = dicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)
    except Exception:
        datasets = []

    return datasets


class DicomReader(object):
    """
    Main DICOM pipeline for reading, organizing, and converting datasets.

    Features
    --------
    - Multithreaded DICOM reading
    - Modality-based grouping
    - Series and slice sorting
    - Image/structure creation
    - RTSTRUCT / RTDOSE association
    - Global Data integration

    Parameters
    ----------
    files : dict
        Dictionary containing file lists (expects key ``'Dicom'``).
    only_tags : bool
        If True, loads only metadata (no pixel arrays).
    only_modality : list of str or None
        Modalities to process. If None, defaults to all supported modalities.
    only_load_roi_names : bool
        If True, loads only ROI names (not full contours).
    clear : bool
        If True, clears global `Data` before loading.

    Examples
    --------
    Basic usage::

        reader = DicomReader(
            files={"Dicom": dicom_paths},
            only_tags=True,
            only_modality=None,
            only_load_roi_names=False,
            clear=True
        )
        reader.load()
    """

    def __init__(self, files, only_tags, only_modality, only_load_roi_names, clear):
        """
        Initialize DICOM reader.
        """
        self.files = files
        self.only_tags = only_tags
        self.only_load_roi_names = only_load_roi_names

        self.only_modality = (
            only_modality
            if only_modality is not None
            else ['CT', 'MR', 'PT', 'US', 'DX', 'RF', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE']
        )

        if clear:
            Data.clear()

        self.ds = []
        self.ds_modality = {key: [] for key in self.only_modality}

    def load(self, display_time=False):
        """
        Execute full DICOM pipeline.

        Steps
        -----
        1. Read files (multithreaded)
        2. Separate modalities
        3. Create images/structures
        4. Sort by acquisition time

        Parameters
        ----------
        display_time : bool
            If True, prints total runtime.
        """
        t1 = time.time()

        self.read()
        self.separate_modalities_and_images()
        self.image_creation()
        sort_images_by_datetime()

        t2 = time.time()

        if display_time:
            print("Dicom Read Time:", t2 - t1)

    def read(self):
        """
        Read all DICOM files using multithreading.
        """
        threads = []
        def read_file_thread(file_path):
            self.ds.append(thread_process_dicom(file_path, stop_before_pixels=self.only_tags))

        for file_path in self.files['Dicom']:
            thread = threading.Thread(target=read_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def separate_modalities_and_images(self):
        """
        Group DICOM datasets by modality, series, orientation, and slice order.

        This function:
        - Separates modalities (CT, MR, RTSTRUCT, etc.)
        - Groups by SeriesInstanceUID
        - Sorts slices by spatial orientation and position
        - Determines acquisition plane (Axial / Coronal / Sagittal)
        - Stores results in `self.ds_modality`
        """
        for modality in list(self.ds_modality.keys()):
            images_in_modality = [d for d in self.ds if (0x0008, 0x0060) in d and d['Modality'].value == modality]
            if len(images_in_modality) > 0 and modality in self.only_modality:
                if modality in ['US', 'DX', 'RF', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE']:
                    for image in images_in_modality:
                        self.ds_modality[modality] += [image]

                else:
                    sorting_tags = []
                    for img in images_in_modality:
                        if 'ImageOrientationPatient' not in img or 'ImagePositionPatient' not in img:
                            continue

                        orient = np.asarray(img['ImageOrientationPatient'].value)
                        pos = np.asarray(img['ImagePositionPatient'].value)
                        if 'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None:
                            acq = np.int64(img['AcquisitionNumber'].value)
                        else:
                            acq = 1

                        sorting_tags += [[img['SeriesInstanceUID'].value, acq, orient[0], orient[1], orient[2],
                                          orient[3], orient[4], orient[5], pos[0], pos[1], pos[2]]]

                    if len(sorting_tags) == 0:
                        continue

                    sorting_tags = np.asarray(sorting_tags)
                    unique_series = np.unique(np.asarray(sorting_tags[:, 0]), axis=0)
                    for series in unique_series:
                        idx = np.where(sorting_tags[:, 0] == series)
                        series_tags = sorting_tags[idx[0], :]
                        series_image = [images_in_modality[ii] for ii in idx[0]]

                        orientations = series_tags[:, 2:8].astype(np.float64)
                        _, indices = np.unique(np.round(orientations, 3), axis=0, return_index=True)
                        unique_orientations = [orientations[ind].astype(np.float64) for ind in indices]
                        for orient in unique_orientations:
                            orient_idx = np.where((np.round(orientations[:, 0], 3) == np.round(orient[0], 3)) &
                                                  (np.round(orientations[:, 1], 3) == np.round(orient[1], 3)) &
                                                  (np.round(orientations[:, 2], 3) == np.round(orient[2], 3)) &
                                                  (np.round(orientations[:, 3], 3) == np.round(orient[3], 3)) &
                                                  (np.round(orientations[:, 4], 3) == np.round(orient[4], 3)) &
                                                  (np.round(orientations[:, 5], 3) == np.round(orient[5], 3)))

                            orient_tags = np.asarray([series_tags[orient] for orient in orient_idx[0]])
                            orient_image = [series_image[orient] for orient in orient_idx[0]]
                            correct_orientation = orient_tags[0, 2:8].astype(np.float64)

                            x = np.abs(correct_orientation[0]) + np.abs(correct_orientation[3])
                            y = np.abs(correct_orientation[1]) + np.abs(correct_orientation[4])
                            z = np.abs(correct_orientation[2]) + np.abs(correct_orientation[5])

                            row_direction = correct_orientation[:3]
                            column_direction = correct_orientation[3:]
                            slice_direction = np.cross(row_direction, column_direction)

                            unique_acq = np.unique(orient_tags[:, 1])

                            acq_plane = []
                            acq_images = []
                            acq_positions = []
                            for acq in unique_acq:
                                orient_idx = np.where(orient_tags == acq)[0]
                                acq_tags = orient_tags[orient_idx]
                                acq_image = [orient_image[ii] for ii in orient_idx]
                                position_tags = np.asarray([np.asarray(t[8:]).astype(np.double) for t in acq_tags])

                                if x < y and x < z:
                                    acq_plane += ['Sagittal']
                                    if slice_direction[0] > 0:
                                        slice_idx = np.argsort(position_tags[:, 0])
                                    else:
                                        slice_idx = np.argsort(position_tags[:, 0])[::-1]
                                elif y < x and y < z:
                                    acq_plane += ['Coronal']
                                    if slice_direction[1] > 0:
                                        slice_idx = np.argsort(position_tags[:, 1])
                                    else:
                                        slice_idx = np.argsort(position_tags[:, 1])[::-1]
                                else:
                                    acq_plane += ['Axial']
                                    if slice_direction[2] > 0:
                                        slice_idx = np.argsort(position_tags[:, 2])
                                    else:
                                        slice_idx = np.argsort(position_tags[:, 2])[::-1]

                                acq_images += [np.asarray([acq_image[idx] for idx in slice_idx])]
                                acq_positions += [np.asarray([acq_tags[idx] for idx in slice_idx])]

                            if len(acq_positions) > 1:
                                exclude_images = np.zeros((len(acq_positions), 1))
                                for ii in range(len(acq_positions)):
                                    for jj in range(len(acq_positions)):
                                        if ii != jj:
                                            if acq_plane[0] == 'Sagittal':
                                                base_first = acq_positions[ii][0, 8]
                                                base_last = acq_positions[ii][-1, 8]
                                                check_first = acq_positions[jj][0, 8]
                                                check_last = acq_positions[jj][-1, 8]
                                            elif acq_plane[0] == 'Coronal':
                                                base_first = acq_positions[ii][0, 9]
                                                base_last = acq_positions[ii][-1, 9]
                                                check_first = acq_positions[jj][0, 9]
                                                check_last = acq_positions[jj][-1, 9]
                                            else:
                                                base_first = acq_positions[ii][0, 10]
                                                base_last = acq_positions[ii][-1, 10]
                                                check_first = acq_positions[jj][0, 10]
                                                check_last = acq_positions[jj][-1, 10]

                                            base_first = np.float64(base_first)
                                            base_last = np.float64(base_last)
                                            check_first = np.float64(check_first)
                                            check_last = np.float64(check_last)

                                            if base_first > check_first and base_first > check_last:
                                                pass

                                            elif base_last < check_first and base_last < check_last:
                                                pass

                                            else:
                                                exclude_images[ii] = 1

                                if np.sum(exclude_images) == 0:
                                    if acq_plane[0] == 'Sagittal':
                                        pos = np.asarray([[p[0, 8], p[-1, 8]] for p in acq_positions])
                                    elif acq_plane[0] == 'Coronal':
                                        pos = np.asarray([[p[0, 9], p[-1, 9]] for p in acq_positions])
                                    else:
                                        pos = np.asarray([[p[0, 10], p[-1, 10]] for p in acq_positions]).astype(
                                            np.float64)

                                    pos_idx = np.argsort(pos[:, 0])
                                    pos_sort = pos[pos_idx]
                                    pos_diff = [pos_sort[ii + 1, 0] - pos_sort[ii, 1] for ii in range(len(pos) - 1)]
                                    if len(np.unique(np.round(pos_diff, 2))) == 1:
                                        img = []
                                        for ii in pos_idx:
                                            for acq in acq_images[ii]:
                                                img += [acq]
                                        self.ds_modality[modality] += [img]

                                    else:
                                        for img in acq_images:
                                            self.ds_modality[modality] += [img.tolist()]

                                else:
                                    for img in acq_images:
                                        self.ds_modality[modality] += [img.tolist()]

                            else:
                                for img in acq_images:
                                    self.ds_modality[modality] += [img.tolist()]

    def image_creation(self):
        """
        Convert grouped DICOM datasets into internal image structures.

        Handles:
        - CT/MR/PT → 3D image reader
        - DX/CR → X-ray reader
        - RF → fluoroscopy reader
        - US → ultrasound reader
        - RTSTRUCT → ROI association
        - REG / RTDOSE → specialized readers
        """

        for modality in ['CT', 'MR', 'PT', 'DX', 'RF', 'CR', 'US']:
            for image_set in self.ds_modality[modality]:
                if modality in ['CT', 'MR', 'PT']:
                    Read3D(image_set, self.only_tags)

                elif modality in ['DX', 'CR']:
                    ReadXRay(image_set, self.only_tags)

                elif modality == 'RF':
                    ReadRF(image_set, self.only_tags)

                elif modality == 'US':
                    ReadUS(image_set, self.only_tags)

        for modality in ['RTSTRUCT']:
            for image_set in self.ds_modality[modality]:
                read_rtstruct = ReadRTStruct(image_set, self.only_tags)
                if read_rtstruct.match_image_name is not None:
                    Data.image[read_rtstruct.match_image_name].input_rtstruct(read_rtstruct)
                else:
                    print('dicom: rtstruct has no matching image')

        for modality in ['REG']:
            for image_set in self.ds_modality[modality]:
                ReadREG(image_set, self.only_tags)

        for modality in ['RTDOSE']:
            for image_set in self.ds_modality[modality]:
                ReadRTDose(image_set, self.only_tags)


class Read3D(object):
    """
    Reads and constructs 3D medical image volumes from DICOM slices.

    This class processes CT/MR/PT series into a volumetric representation by:
    - Stacking DICOM slices into a 3D array
    - Computing orientation, spacing, and geometry
    - Verifying physical consistency of the volume
    - Registering the result into the global `Data` structure

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        List of DICOM slices representing a volume.
    only_tags : bool
        If True, only metadata is loaded (pixel data is skipped).

    Attributes
    ----------
    array : np.ndarray
        3D image volume (slice stack).
    orientation : np.ndarray
        DICOM direction cosines (row/column vectors).
    spacing : np.ndarray
        Voxel spacing in physical space.
    dimensions : np.ndarray
        Volume dimensions in voxel space.
    plane : str
        Anatomical plane (Axial, Coronal, Sagittal).
    image_name : str
        Generated internal image identifier.

    Examples
    --------
    Basic usage::

        reader = Read3D(dicom_series, only_tags=False)
        print(reader.image_name)
    """

    def __init__(self, image_set, only_tags):
        """
        Initialize and immediately construct a 3D volume.
        """
        self.image_set = (
            image_set if isinstance(image_set, list)
            else [image_set]
        )

        self.only_tags = only_tags

        # --- internal state ---
        self.unverified = None
        self.base_position = None
        self.skipped_slice = []
        self.rgb = False

        # --- metadata ---
        self.modality = self.image_set[0].Modality
        self.filepaths = [img.filename for img in self.image_set]
        self.sops = [img.SOPInstanceUID for img in self.image_set]

        self.orientation = self._compute_orientation()
        self.plane = self._compute_plane()
        self.spacing = self._compute_spacing()

        # --- volume data ---
        self.array = None
        if not self.only_tags:
            self._compute_array()
            self.dimensions = self._compute_dimensions()
            self._verify_axial_orientation()

        self.image_matrix = self._compute_image_matrix()
        self.image_name = create_image_name(self.modality)

        # --- register into global system ---
        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)

    def _compute_array(self):
        """
        Stack DICOM slices into a 3D NumPy array.

        Applies:
        - RescaleSlope
        - RescaleIntercept
        - PixelData cleanup for memory efficiency
        """
        image_slices = []
        for _slice in self.image_set:
            if (0x0028, 0x1052) in _slice:
                intercept = _slice.RescaleIntercept
            else:
                intercept = 0

            if (0x0028, 0x1053) in _slice:
                slope = _slice.RescaleSlope
            else:
                slope = 1

            image_slices.append(((_slice.pixel_array * slope) + intercept).astype('int16'))

            del _slice.PixelData

        self.array = np.asarray(image_slices)

    def _compute_orientation(self):
        """
        Extract DICOM orientation (ImageOrientationPatient).
        """
        orientation = np.asarray([1, 0, 0, 0, 1, 0])
        if 'ImageOrientationPatient' in self.image_set[0]:
            orientation = np.asarray(self.image_set[0]['ImageOrientationPatient'].value)

        else:
            if 'SharedFunctionalGroupsSequence' in self.image_set[0]:
                seq_str = 'SharedFunctionalGroupsSequence'
                if 'PlaneOrientationSequence' in self.image_set[0][0][seq_str][0]:
                    plane_str = 'PlaneOrientationSequence'
                    image_str = 'ImageOrientationPatient'
                    orientation = np.asarray(self.image_set[0][0][seq_str][0][plane_str][0][image_str].value)

                else:
                    self.unverified = 'Orientation'

            else:
                self.unverified = 'Orientation'

        return orientation

    def _compute_plane(self):
        """
        Determine anatomical acquisition plane.
        """
        x = np.abs(self.orientation[0]) + np.abs(self.orientation[3])
        y = np.abs(self.orientation[1]) + np.abs(self.orientation[4])
        z = np.abs(self.orientation[2]) + np.abs(self.orientation[5])

        if x < y and x < z:
            return "Sagittal"
        elif y < x and y < z:
            return "Coronal"
        else:
            return "Axial"

    def _compute_spacing(self):
        """
        Compute voxel spacing in 3D (x, y, z).

        Includes:
        - PixelSpacing (fallbacks for enhanced DICOM)
        - Slice spacing from ImagePositionPatient
        - Detection of irregular slice spacing
        """
        inplane_spacing = [1, 1]
        slice_thickness = np.double(self.image_set[0].SliceThickness)

        if 'PixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].PixelSpacing

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set[0]:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        if len(self.image_set) > 1:
            row_direction = self.orientation[:3]
            column_direction = self.orientation[3:]
            slice_direction = np.cross(row_direction, column_direction)

            first = np.dot(slice_direction, self.image_set[0].ImagePositionPatient)
            second = np.dot(slice_direction, self.image_set[1].ImagePositionPatient)
            last = np.dot(slice_direction, self.image_set[-1].ImagePositionPatient)
            first_last_spacing = np.asarray((last - first) / (len(self.image_set) - 1))
            if np.abs((second - first) - first_last_spacing) > 0.01:
                if not self.only_tags:
                    self._find_skipped_slices(slice_direction)
                slice_thickness = second - first
            else:
                slice_thickness = np.asarray((last - first) / (len(self.image_set) - 1))

        if self.plane == 'Axial':
            return np.asarray([inplane_spacing[1], inplane_spacing[0], slice_thickness])

        elif self.plane == 'Coronal':
            return np.asarray([inplane_spacing[1], slice_thickness, inplane_spacing[0]])

        else:
            return np.asarray([slice_thickness, inplane_spacing[1], inplane_spacing[0]])

    def _compute_dimensions(self):
        """
        Compute voxel dimensions based on volume shape.
        """
        shape = self.array.shape

        if self.plane == "Axial":
            return np.array([shape[0], shape[1], shape[2]])

        elif self.plane == "Coronal":
            return np.array([shape[1], shape[0], shape[2]])

        else:
            return np.array([shape[1], shape[2], shape[0]])

    def _compute_image_matrix(self):
        """
        Construct 3×3 orientation matrix.
        """
        row = self.orientation[:3]
        col = self.orientation[3:]
        slc = np.cross(row, col)

        mat = np.eye(3, dtype=np.float32)
        mat[0] = row
        mat[1] = col
        mat[2] = slc

        return mat

    def _verify_axial_orientation(self):
        """
        Ensures that the 3D volume is oriented correctly in physical space.
            - Computes all 8 corner coordinates of the image volume using orientation and spacing.
            - Identifies the physical origin and corrects rotations or flips via np.rot90 and transpositions.
            - Adjusts orientation vectors based on corrected axes.
        """
        shape = self.array.shape
        if self.plane == 'Axial':
            spacing = self.spacing
        elif self.plane == 'Coronal':
            spacing = [self.spacing[0], self.spacing[2], self.spacing[1]]
        else:
            spacing = [self.spacing[1], self.spacing[2], self.spacing[0]]

        slices = shape[0] - 1
        y = shape[1] - 1
        x = shape[2] - 1

        origin = np.asarray(self.image_set[0]['ImagePositionPatient'].value)

        row_direction = self.orientation[:3]
        column_direction = self.orientation[3:]
        slice_direction = np.cross(row_direction, column_direction)

        corners = np.zeros((8, 3))
        corners[0] = origin
        corners[1] = origin + (x * spacing[0] * row_direction)
        corners[2] = origin + (y * spacing[1] * column_direction)
        corners[3] = (origin + (x * spacing[0] * row_direction) + (y * spacing[1] * column_direction))

        corners[4] = origin + (slices * spacing[2] * slice_direction)
        corners[5] = (origin + (slices * spacing[2] * slice_direction) + (x * spacing[0] * row_direction))
        corners[6] = (origin + (slices * spacing[2] * slice_direction) + (y * spacing[1] * column_direction))
        corners[7] = (origin + (slices * spacing[2] * slice_direction) + (x * spacing[0] * row_direction) +
                      (y * spacing[1] * column_direction))

        corner_idx = np.argmin(np.sum(corners, axis=1))
        if corner_idx != 0:
            self.origin = corners[corner_idx]
            if self.plane == "Axial":
                if corner_idx == 1:
                    self.array = np.rot90(self.array, 1, (1, 2))
                elif corner_idx == 2:
                    self.array = np.rot90(self.array, 3, (1, 2))
                else:
                    self.array = np.rot90(self.array, 2, (1, 2))

                if corner_idx < 4:
                    square = corners[:4, :]
                else:
                    square = corners[4:, :]

            elif self.plane == 'Coronal':
                self.array = np.rot90(self.array, 1, (0, 1))

                s1 = np.argsort(corners[:4, 2])
                s2 = np.argsort(corners[4:, 2]) + 4

                square = [corners[s1[0]], corners[s1[1]], corners[s2[0]], corners[s2[1]]]

            else:
                self.array = np.flip(np.rot90(self.array, 1, (0, 1)).transpose(0, 2, 1), axis=2)

                s1 = np.argsort(corners[:4, 2])
                s2 = np.argsort(corners[4:, 2]) + 4

                square = [corners[s1[0]], corners[s1[1]], corners[s2[0]], corners[s2[1]]]

            distances = np.asarray([np.linalg.norm(corners[corner_idx, :] - s) for s in square])
            sorted_args = np.argsort(distances)

            c1 = square[sorted_args[1]] - corners[corner_idx]
            c2 = square[sorted_args[2]] - corners[corner_idx]

            if np.abs(c1[0]) > np.abs(c2[0]):
                self.orientation[:3] = c1 / (self.spacing[0] * self.dimensions[2])
                self.orientation[3:] = c2 / (self.spacing[1] * self.dimensions[1])
            else:
                self.orientation[:3] = c2 / (self.spacing[0] * self.dimensions[2])
                self.orientation[3:] = c1 / (self.spacing[1] * self.dimensions[1])

            self.image_matrix = self._compute_image_matrix()

        else:
            self.origin = origin

    def _find_skipped_slices(self):
        """
        Detect and interpolate missing slices.

        Inserts synthetic DICOM slices directly into self.image_set.
        """

        if len(self.image_set) < 2:
            return

        row = self.orientation[:3]
        col = self.orientation[3:]
        slice_dir = np.cross(row, col)

        positions = np.array([np.dot(slice_dir, ds.ImagePositionPatient) for ds in self.image_set])
        order = np.argsort(positions)
        self.image_set = [self.image_set[i] for i in order]
        positions = positions[order]

        diffs = np.diff(positions)
        expected_spacing = np.median(diffs)
        rebuilt = [self.image_set[0]]

        self.missing_slices = []
        for i in range(len(self.image_set) - 1):

            ds1 = self.image_set[i]
            ds2 = self.image_set[i + 1]

            p1 = positions[i]
            p2 = positions[i + 1]

            gap = p2 - p1
            n_expected = int(round(gap / expected_spacing))
            rebuilt.append(ds1)
            if n_expected <= 1:
                continue

            n_missing = n_expected - 1
            self.unverified = "Skipped"
            self.skipped_slice += [i + 1]

            self.missing_slices.append({
                "insert_index": len(rebuilt),
                "num_missing": n_missing,
                "between": (
                    ds1.SOPInstanceUID,
                    ds2.SOPInstanceUID
                )
            })

            img1 = ds1.pixel_array.astype(np.float32)
            img2 = ds2.pixel_array.astype(np.float32)

            pos1 = np.asarray(ds1.ImagePositionPatient, dtype=np.float64)
            pos2 = np.asarray(ds2.ImagePositionPatient, dtype=np.float64)

            for m in range(n_missing):

                alpha = (m + 1) / (n_missing + 1)
                interp = (1.0 - alpha) * img1 + alpha * img2
                interp = np.round(interp).astype(ds1.pixel_array.dtype)

                new_ds = copy.deepcopy(ds1)
                new_pos = pos1 + alpha * (pos2 - pos1)

                new_ds.ImagePositionPatient = [
                    float(v) for v in new_pos
                ]

                new_ds.PixelData = interp.tobytes()
                new_ds.SOPInstanceUID = generate_uid()

                if "InstanceNumber" in new_ds:
                    new_ds.InstanceNumber = (
                            ds1.InstanceNumber + m + 1
                    )

                if hasattr(new_ds, "file_meta"):
                    new_ds.file_meta.MediaStorageSOPInstanceUID = (
                        new_ds.SOPInstanceUID
                    )

                rebuilt.append(new_ds)
        rebuilt.append(self.image_set[-1])
        self.image_set = rebuilt


class ReadXRay(object):
    """
    Reads and constructs 2D X-ray images (DX, CR, MG modalities) from DICOM files.

    This class converts a single DICOM image into an internal representation,
    handling:
    - Orientation inference (Axial / Coronal / Sagittal)
    - Pixel spacing extraction
    - 2D image reshaping into pseudo-3D format
    - Optional intensity inversion (PresentationLUTShape)
    - Registration into global `Data` structure

    Notes
    -----
    - Tomosynthesis (3D MG) is not supported.
    - Output is stored as a 3D-like array with one singleton dimension.

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        Single or list of DICOM datasets representing an X-ray image.
    only_tags : bool
        If True, only metadata is loaded (pixel data is skipped).

    Attributes
    ----------
    array : np.ndarray
        Image pixel array (reshaped to pseudo-3D format).
    orientation : list
        Default orientation vector (identity unless extended later).
    spacing : np.ndarray
        Physical pixel spacing in mm.
    dimensions : np.ndarray
        Image dimensions (one axis is 1 due to 2D nature).
    image_name : str
        Internal identifier used for registration.

    Examples
    --------
    Basic usage::

        reader = ReadXRay(dicom_dataset, only_tags=False)
        print(reader.image_name)
    """

    def __init__(self, image_set, only_tags):
        """
        Initialize X-ray reader and immediately construct image object.
        """

        self.image_set = (
            image_set if isinstance(image_set, list)
            else [image_set]
        )

        self.only_tags = only_tags

        self.unverified = 'Modality'
        self.skipped_slice = None
        self.rgb = False

        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.array([0, 0, 0])
        self.image_matrix = np.eye(3, dtype=np.float32)

        self.modality = self.image_set[0].Modality
        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID

        self.plane = self._compute_plane()
        self.dimensions = self._compute_dimensions()
        self.spacing = self._compute_spacing()

        self.array = None
        if not self.only_tags:
            self._compute_array()

        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)


    def _compute_plane(self):
        """
        Determine anatomical plane from PatientOrientation tag.

        Returns
        -------
        str
            One of: 'Axial', 'Coronal', 'Sagittal'
        """
        img = self.image_set[0]

        if 'PatientOrientation' in img:
            orient = img.PatientOrientation

            if 'L' in orient or 'R' in orient:
                return 'Coronal'
            elif 'A' in orient or 'P' in orient:
                return 'Sagittal'
            else:
                return 'Axial'

        return 'Axial'

    def _compute_dimensions(self):
        """
        Compute 3D-like dimensions for a 2D image.

        One axis is set to 1 depending on orientation.
        """
        rows = int(self.image_set[0]['Rows'].value)
        cols = int(self.image_set[0]['Columns'].value)

        if self.plane == 'Axial':
            return np.array([1, rows, cols])

        elif self.plane == 'Coronal':
            return np.array([rows, 1, cols])

        else:
            return np.array([rows, cols, 1])

    def _compute_spacing(self):
        """
        Compute pixel spacing (x, y, z) for X-ray images.

        Notes
        -----
        - Slice thickness is always 1 (2D image)
        - Supports multiple DICOM spacing tags
        """
        img = self.image_set[0]

        inplane = [1, 1]
        slice_thickness = 1

        if 'PixelSpacing' in img:
            inplane = img.PixelSpacing

        elif 'ImagerPixelSpacing' in img:
            inplane = img.ImagerPixelSpacing

        elif 'ContributingSourcesSequence' in img:
            seq = img.ContributingSourcesSequence[0]
            if 'DetectorElementSpacing' in seq:
                inplane = seq.DetectorElementSpacing

        elif 'PerFrameFunctionalGroupsSequence' in img:
            seq = img.PerFrameFunctionalGroupsSequence[0]
            if 'PixelMeasuresSequence' in seq:
                inplane = seq.PixelMeasuresSequence[0].PixelSpacing

        # --- reorder depending on plane ---
        if self.plane == 'Axial':
            return np.array([inplane[1], inplane[0], slice_thickness])

        elif self.plane == 'Coronal':
            return np.array([inplane[1], slice_thickness, inplane[0]])

        else:
            return np.array([slice_thickness, inplane[1], inplane[0]])

    def _compute_array(self):
        """
        Load and normalize pixel data.

        Steps
        -----
        - Convert to int16
        - Apply LUT inversion if needed
        - Reshape into pseudo-3D array
        - Flip for consistent orientation
        """
        img = self.image_set[0]

        self.array = img.pixel_array.astype('int16')
        del img.PixelData  # free memory

        # --- intensity inversion ---
        if ('PresentationLUTShape' in img and
                img.PresentationLUTShape == 'Inverse'):
            self.array = 16383 - self.array

        # --- reshape into pseudo-3D ---
        if self.plane == 'Axial':
            self.array = self.array.reshape((1, *self.array.shape))

        elif self.plane == 'Coronal':
            self.array = np.flip(
                np.flip(
                    self.array.reshape((self.array.shape[0], 1, self.array.shape[1])),
                    axis=0
                ),
                axis=1
            )

        else:
            self.array = np.flip(
                self.array.reshape((self.array.shape[0], self.array.shape[1], 1)),
                axis=0
            )


class ReadRF(object):
    """
    Reads and constructs Radio Fluoroscopy (RF) DICOM images.

    This class converts RF DICOM data into a standardized internal representation,
    including:
    - Pixel data extraction
    - Orientation inference
    - Spatial spacing computation
    - Integration into the global `Data` structure

    Notes
    -----
    - RF data is typically dynamic 2D/2.5D imaging.
    - Slice thickness is assumed to be 1 mm unless otherwise specified.
    - No full volumetric reconstruction is performed.

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        RF DICOM dataset(s).
    only_tags : bool
        If True, only metadata is loaded (no pixel data).

    Attributes
    ----------
    array : np.ndarray
        Pixel array representation of RF image.
    spacing : np.ndarray
        Physical spacing (x, y, z).
    dimensions : tuple
        Shape of the RF image array.
    orientation : list
        Default orientation vector.
    image_name : str
        Internal identifier for global registration.

    Examples
    --------
    Basic usage::

        reader = ReadRF(dicom_rf, only_tags=False)
        print(reader.image_name)
    """

    def __init__(self, image_set, only_tags):
        """
        Initialize RF reader and construct image representation.
        """

        self.image_set = (
            image_set if isinstance(image_set, list)
            else [image_set]
        )

        self.only_tags = only_tags

        self.unverified = 'Modality'
        self.skipped_slice = None
        self.rgb = False
        self.dimensions = None

        self.modality = self.image_set[0].Modality
        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID

        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.array([0, 0, 0])
        self.image_matrix = np.eye(3, dtype=np.float32)
        self.plane = self._compute_plane()

        self.array = None
        if not self.only_tags:
            self._compute_array()

        self.spacing = self._compute_spacing()
        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)

    def _compute_plane(self):
        """
        Infer anatomical plane from PatientOrientation.

        Returns
        -------
        str
            'Axial', 'Coronal', or 'Sagittal'
        """
        img = self.image_set[0]

        if 'PatientOrientation' in img:
            orient = img.PatientOrientation

            if 'L' in orient or 'R' in orient:
                return 'Coronal'
            elif 'A' in orient or 'P' in orient:
                return 'Sagittal'
            else:
                return 'Axial'

        return 'Axial'

    def _compute_array(self):
        """
        Load RF pixel data into a NumPy array.

        Steps
        -----
        - Convert to int16
        - Remove raw PixelData for memory efficiency
        - Reshape into consistent 3D-like structure
        - Store final shape in `self.dimensions`
        """
        self.array = self.image_set[0].pixel_array.astype('int16')
        del self.image_set[0].PixelData

        # --- ensure 3D shape consistency ---
        if len(self.array.shape) < 3:

            if self.plane == 'Axial':
                self.array = self.array.reshape(
                    (self.array.shape[2],
                     self.array.shape[0],
                     self.array.shape[1])
                )

            elif self.plane == 'Coronal':
                self.array = self.array.reshape(
                    (self.array.shape[0],
                     self.array.shape[2],
                     self.array.shape[1])
                )

            else:
                self.array = self.array.reshape(
                    (self.array.shape[0],
                     self.array.shape[1],
                     self.array.shape[2])
                )

        self.dimensions = self.array.shape

    def _compute_spacing(self):
        """
        Compute voxel spacing for RF imaging.

        Uses:
        - PixelSpacing
        - ImagerPixelSpacing
        - DetectorElementSpacing (fallbacks)

        Returns
        -------
        np.ndarray
            Spacing in (x, y, z) order depending on plane.
        """
        img = self.image_set[0]

        inplane = [1, 1]
        slice_thickness = 1

        if 'PixelSpacing' in img:
            inplane = img.PixelSpacing

        elif 'ImagerPixelSpacing' in img:
            inplane = img.ImagerPixelSpacing

        elif 'ContributingSourcesSequence' in img:
            seq = img.ContributingSourcesSequence[0]
            if 'DetectorElementSpacing' in seq:
                inplane = seq.DetectorElementSpacing

        elif 'PerFrameFunctionalGroupsSequence' in img:
            seq = img.PerFrameFunctionalGroupsSequence[0]
            if 'PixelMeasuresSequence' in seq:
                inplane = seq.PixelMeasuresSequence[0].PixelSpacing

        # --- reorder by plane ---
        if self.plane == 'Axial':
            return np.array([inplane[1], inplane[0], slice_thickness])

        elif self.plane == 'Coronal':
            return np.array([inplane[1], slice_thickness, inplane[0]])

        else:
            return np.array([slice_thickness, inplane[1], inplane[0]])


class ReadUS(object):
    """
    Reads and constructs Ultrasound (US) DICOM images.

    Ultrasound imaging differs from other modalities in that:
    - Frames may not represent true anatomical slices
    - Multi-frame images are common
    - Pixel values often require filtering or channel extraction

    This class standardizes US data into a consistent internal format and
    registers it in the global `Data` structure.

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        One or more ultrasound DICOM datasets.
    only_tags : bool
        If True, only metadata is loaded (no pixel data).

    Attributes
    ----------
    array : np.ndarray
        Processed ultrasound image array.
    spacing : np.ndarray
        Pixel spacing (x, y, z).
    dimensions : np.ndarray
        Image dimensions (frames × rows × cols).
    plane : str
        Fixed to 'Axial' for consistency with internal framework.
    image_name : str
        Internal identifier used for registration.

    Examples
    --------
    Basic usage::

        reader = ReadUS(dicom_us, only_tags=False)
        print(reader.image_name)
    """

    def __init__(self, image_set, only_tags):
        """
        Initialize ultrasound reader and construct image representation.
        """

        self.image_set = (
            image_set if isinstance(image_set, list)
            else [image_set]
        )

        self.only_tags = only_tags

        self.unverified = 'Modality'
        self.base_position = None
        self.skipped_slice = None
        self.rgb = False

        self.modality = self.image_set[0].Modality
        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID

        self.plane = 'Axial'
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.array([0, 0, 0])
        self.image_matrix = np.eye(3, dtype=np.float32)

        self.dimensions = np.array([
            1,
            self.image_set[0]['Rows'].value,
            self.image_set[0]['Columns'].value
        ])

        self.array = None
        if not self.only_tags:
            self._compute_array()

        self.spacing = self._compute_spacing()
        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)

    def _compute_array(self):
        """
        Convert ultrasound pixel data into a standardized NumPy array.

        Handles:
        - 2D single-frame images
        - Multi-frame ultrasound sequences
        - Channel filtering for uniform frames
        - Conversion to uint8
        """
        us_data = np.asarray(self.image_set[0].pixel_array)
        del self.image_set[0].PixelData

        # --- single frame ---
        if len(us_data.shape) == 2:
            us_data = us_data.reshape((1, *us_data.shape))

        # --- multi-frame (3D) ---
        if len(us_data.shape) == 3:
            uniform_mask = (np.std(us_data, axis=2) == 0)
            self.array = (uniform_mask * us_data[:, :, 0]).astype(np.uint8)

            if len(self.array.shape) == 2:
                self.array = np.expand_dims(self.array, axis=0)

        # --- multi-channel / higher-dimensional ---
        else:
            uniform_mask = (np.std(us_data, axis=3) == 0)
            self.array = (uniform_mask * us_data[:, :, :, 0]).astype(np.uint8)

        # --- ensure frame dimension exists ---
        if len(self.array.shape) == 3:
            self.dimensions[0] = self.array.shape[0]

    def _compute_spacing(self):
        """
        Compute ultrasound pixel spacing.

        Uses (in priority order):
        - PixelSpacing
        - DetectorElementSpacing
        - PixelMeasuresSequence
        - SequenceOfUltrasoundRegions (US-specific metadata)

        Returns
        -------
        np.ndarray
            Spacing in (x, y, z) format.
        """
        img = self.image_set[0]

        inplane = [1, 1]
        slice_thickness = 1

        if 'PixelSpacing' in img:
            inplane = img.PixelSpacing

        elif 'ContributingSourcesSequence' in img:
            seq = img.ContributingSourcesSequence[0]
            if 'DetectorElementSpacing' in seq:
                inplane = seq.DetectorElementSpacing

        elif 'PerFrameFunctionalGroupsSequence' in img:
            seq = img.PerFrameFunctionalGroupsSequence[0]
            if 'PixelMeasuresSequence' in seq:
                inplane = seq.PixelMeasuresSequence[0].PixelSpacing

        elif 'SequenceOfUltrasoundRegions' in img:
            region = img.SequenceOfUltrasoundRegions[0]

            if 'PhysicalDeltaX' in region:
                inplane = [
                    10 * np.round(region.PhysicalDeltaY, 4),
                    10 * np.round(region.PhysicalDeltaX, 4)
                ]

        return np.array([inplane[1], inplane[0], slice_thickness])


class ReadRTStruct(object):
    """
    Reads and parses Radiotherapy Structure Set (RTSTRUCT) DICOM files.

    This class extracts:
    - ROI (Region of Interest) contour data
    - POI (Point of Interest) coordinates
    - Associated geometric metadata
    - Mapping to a corresponding image series in `Data`

    It organizes structures for downstream visualization, segmentation,
    and analysis workflows.

    Notes
    -----
    - Only supports standard RTSTRUCT contour formats.
    - Closed planar contours are treated as ROIs.
    - Point-based structures are treated as POIs.
    - Color is taken from DICOM or randomly generated if missing.

    Parameters
    ----------
    image_set : pydicom.Dataset
        RTSTRUCT DICOM dataset.
    only_tags : bool
        If True, only metadata is parsed (no contour extraction).

    Attributes
    ----------
    roi_names : list of str
        Names of ROI structures.
    roi_colors : list
        RGB colors for ROIs.
    poi_names : list of str
        Names of POI structures.
    poi_colors : list
        RGB colors for POIs.
    contours : list
        Extracted ROI contour coordinate arrays.
    points : list
        Extracted POI coordinate sets.
    match_image_name : str or None
        Name of matched image in global `Data`.

    Examples
    --------
    Basic usage::

        rt = ReadRTStruct(rtstruct_dataset, only_tags=False)
        print(rt.roi_names)
    """

    def __init__(self, image_set, only_tags):
        """
        Initialize RTSTRUCT parser and extract structure metadata.
        """

        self.image_set = image_set
        self.only_tags = only_tags

        self.series_uid = self._get_series_uid()
        self.filepaths = self.image_set.filename

        self._properties = self._get_properties()
        self.roi_names = [prop[1] for prop in self._properties if prop[3].lower() == 'closed_planar']
        self.roi_colors = [prop[2] for prop in self._properties if prop[3].lower() == 'closed_planar']
        self.poi_names = [prop[1] for prop in self._properties if prop[3].lower() == 'point']
        self.poi_colors = [prop[2] for prop in self._properties if prop[3].lower() == 'point']

        if len(self.roi_names) > 0 or len(self.poi_names) > 0:
            self.match_image_name = self._match_with_image()

            self.contours = []
            self.points = []
            if not self.only_tags:
                self._structure_positions()
        else:
            self.match_image_name = None

    def _get_series_uid(self):
        """
        Extract SeriesInstanceUID from referenced image series.

        Returns
        -------
        str
            SeriesInstanceUID of referenced image.
        """
        ref = self.image_set.ReferencedFrameOfReferenceSequence

        return ref[0][
            'RTReferencedStudySequence'
        ][0][
            'RTReferencedSeriesSequence'
        ][0]['SeriesInstanceUID'].value

    def _get_properties(self):
        """
        Extract ROI and POI metadata from RTSTRUCT.

        Collects:
        - ROI index
        - ROI name
        - Color (or random fallback)
        - Geometric type
        - Referenced SOPInstanceUIDs

        Returns
        -------
        list
            Structured ROI/POI metadata entries.
        """
        tracker = []
        sop = []
        names = []
        colors = []
        geometric = []

        if 'ROIContourSequence' in self.image_set:

            for ii, s in enumerate(self.image_set.ROIContourSequence):

                if hasattr(
                    self.image_set.StructureSetROISequence[ii],
                    'ROIName'
                ):

                    if hasattr(s, 'ContourSequence') and len(s.ContourSequence) > 0:

                        tracker.append(ii)
                        names.append(
                            self.image_set.StructureSetROISequence[ii].ROIName
                        )

                        geometric.append(
                            s.ContourSequence[0].ContourGeometricType
                        )

                        slice_sop = []

                        if geometric[-1].lower() == 'closed_planar':

                            for seq in s.ContourSequence:
                                slice_sop.append(
                                    seq.ContourImageSequence[0]
                                    .ReferencedSOPInstanceUID
                                )

                        else:
                            if hasattr(s.ContourSequence[0], 'ContourImageSequence'):
                                slice_sop = [
                                    s.ContourSequence[0]
                                    .ContourImageSequence[0]
                                    .ReferencedSOPInstanceUID
                                ]

                        sop.append(slice_sop)

                        if hasattr(s, 'ROIDisplayColor'):
                            colors.append(s.ROIDisplayColor)
                        else:
                            colors.append([
                                np.random.randint(0, 256),
                                np.random.randint(0, 256),
                                np.random.randint(0, 256)
                            ])

        return [
            [tracker[i], names[i], colors[i], geometric[i], sop[i]]
            for i in range(len(names))
        ]

    def _match_with_image(self):
        """
        Match RTSTRUCT to a corresponding image in global Data.

        Returns
        -------
        str or None
            Matched image name if found.
        """
        for image_name in Data.image:

            if self.series_uid == Data.image[image_name].series_uid:

                if self._properties[0][4][0] in Data.image[image_name].sops:
                    return image_name

        return None

    def _structure_positions(self):
        """
        Extract contour and point coordinates from RTSTRUCT.

        - Closed planar contours → stored in `self.contours`
        - Point structures → stored in `self.points`
        """
        sequences = self.image_set.ROIContourSequence

        for prop in self._properties:
            seq = sequences[prop[0]]

            contour_list = []

            for c in seq.ContourSequence:
                contour_data = np.round(
                    np.array(c.ContourData),
                    3
                )

                contour = contour_data.reshape(-1, 3)
                contour_list.append(contour)

            if prop[3].lower() == 'closed_planar':
                self.contours.append(contour_list)
            else:
                self.points.extend(contour_list)


class ReadREG(object):
    """
    Reads and processes DICOM Spatial Registration (REG) objects.

    This class handles both:
    - Rigid registrations (matrix-based transformations)
    - Deformable registrations (DVF-based transformations)

    It extracts:
    - Reference and moving image series
    - Transformation matrices
    - Deformable vector fields (if present)
    - Registration metadata

    The resulting registration is stored in the global `Data` structure.

    Notes
    -----
    - Supports both rigid and deformable REG objects.
    - Deformable registrations require DVF grid data.
    - Handles naming conflicts by appending indices.

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        REG DICOM dataset(s).
    only_tags : bool
        If True, only metadata is parsed (no DVF or matrix computation).

    Attributes
    ----------
    reference_name : str
        Matched reference image name in `Data`.
    moving_name : str
        Matched moving image name in `Data`.
    reference_series : str
        SeriesInstanceUID of reference series.
    moving_series : str
        SeriesInstanceUID of moving series.
    reference_sops : list
        SOPInstanceUIDs for reference images.
    moving_sops : list
        SOPInstanceUIDs for moving images.
    registration_name : str
        Unique registration identifier.
    dvf : np.ndarray or None
        Deformable vector field (if applicable).
    reference_matrix : np.ndarray
        Reference transformation matrix (rigid case).
    moving_matrix : np.ndarray
        Moving transformation matrix.

    Examples
    --------
    Basic usage::

        reg = ReadREG(reg_dataset, only_tags=False)
        print(reg.registration_name)
    """

    def __init__(self, image_set, only_tags):
        """
        Initialize REG reader and compute registration data.
        """

        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]
        self.only_tags = only_tags

        self.reference_name = None
        self.reference_series = self.image_set[0].ReferencedSeriesSequence[0].SeriesInstanceUID
        self.reference_sops = [sop.ReferencedSOPInstanceUID for sop in
                               self.image_set[0].ReferencedSeriesSequence[0].ReferencedInstanceSequence]

        self.moving_name = None
        if len(self.image_set[0].ReferencedSeriesSequence) == 2:
            self.moving_series = self.image_set[0].ReferencedSeriesSequence[1].SeriesInstanceUID
            self.moving_sops = [sop.ReferencedSOPInstanceUID for sop in
                                self.image_set[0].ReferencedSeriesSequence[1].ReferencedInstanceSequence]
        else:
            sequence = self.image_set[0].StudiesContainingOtherReferencedInstancesSequence[0].ReferencedSeriesSequence[0]
            self.moving_series = sequence.SeriesInstanceUID
            self.moving_sops = [sop.ReferencedSOPInstanceUID for sop in sequence.ReferencedInstanceSequence]

        self.spacing = None
        self.dimensions = None
        self.origin = None

        self.reference_matrix = None
        self.moving_matrix = None
        self.dvf_matrix = None
        self.dvf = None

        self.registration_name = None
        if 'DeformableRegistrationSequence' in self.image_set[0]:
            self._compute_rigid(deformable=True)
            self._compute_dvf()
            self._create_name(deformable=True)
            self._create_registration(deformable=True)
        else:
            self._compute_rigid()
            self._create_name()
            self._create_registration()

    def _compute_rigid(self, deformable=False):
        """
        Compute rigid transformation matrices.

        Parameters
        ----------
        deformable : bool
            If True, extracts matrices from deformable registration
            metadata as well.
        """

        if deformable:
            matrix = (
                self.image_set[0]
                .DeformableRegistrationSequence[0]
                .PreDeformationMatrixRegistrationSequence[0][0x3006, 0x00C6]
                .value
            )

            orientation = (
                self.image_set[0]
                .DeformableRegistrationSequence[0]
                .DeformableRegistrationGridSequence[0]
                .ImageOrientationPatient
            )

            row = orientation[:3]
            col = orientation[3:]
            slc = np.cross(row, col)

            self.dvf_matrix = np.eye(3, dtype=np.float32)
            self.dvf_matrix[0, :3] = row
            self.dvf_matrix[1, :3] = col
            self.dvf_matrix[2, :3] = slc

            self.moving_matrix = np.linalg.inv(np.asarray(matrix).reshape(4, 4))

        else:
            matrix = (
                self.image_set[0]
                .RegistrationSequence[1]
                .MatrixRegistrationSequence[0]
                .MatrixSequence[0][0x3006, 0x00C6]
                .value
            )

            self.reference_matrix = matrix

            self.moving_matrix = np.linalg.inv(
                np.asarray(matrix).reshape(4, 4)
            )

    def _compute_dvf(self):
        """
        Extract deformable vector field (DVF) from DICOM grid.
        """

        grid = (
            self.image_set[0]
            .DeformableRegistrationSequence[0]
            .DeformableRegistrationGridSequence[0]
        )

        self.origin = grid.ImagePositionPatient
        self.dimensions = np.flip(grid.GridDimensions)
        self.spacing = grid.GridResolution

        raw = grid.VectorGridData
        values = unpack(f"<{len(raw) // 4}f", raw)

        self.dvf = np.reshape(values, list(self.dimensions) + [3])

        del grid.VectorGridData

    def _create_name(self, deformable=False):
        """
        Create unique registration name and resolve conflicts.
        """

        for image_name in Data.image_list:

            if self.reference_sops[0] in Data.image[image_name].sops:
                self.reference_name = image_name

            elif self.moving_sops[0] in Data.image[image_name].sops:
                self.moving_name = image_name

        prefix = 'DVF_' if deformable else ''

        if self.reference_name is None and self.moving_name is None:
            base = prefix + '_Unknown'
        else:
            base = prefix + f'{self.reference_name}_{self.moving_name}'

        registry = (
            Data.deformable_list if deformable
            else Data.rigid_list
        )

        if base in registry:
            i = 1
            while True:
                candidate = f'{base}_{i}'
                if candidate not in registry:
                    self.registration_name = candidate
                    break
                i += 1
        else:
            self.registration_name = base

    def _create_registration(self, deformable=False):
        """
        Instantiate Rigid or Deformable registration objects.
        """

        if deformable:
            Deformable(
                self.dvf,
                self.origin,
                self.spacing,
                self.dimensions,
                rigid_matrix=self.moving_matrix,
                dvf_matrix=self.dvf_matrix,
                registration_name=self.registration_name,
                reference_name=self.reference_name,
                moving_name=self.moving_name,
                reference_sops=self.reference_sops,
                moving_sops=self.moving_sops
            )

        elif self.reference_name and self.moving_name:
            Rigid(
                self.reference_name,
                self.moving_name,
                rigid_name=self.registration_name,
                reference_sops=self.reference_sops,
                moving_sops=self.moving_sops,
                reference_matrix=self.reference_matrix,
                matrix=self.moving_matrix
            )


class ReadRTDose(object):
    """
    Reads and processes Radiotherapy Dose (RTDOSE) DICOM objects.

    Responsibilities:
        - Loads dose grid data (optionally skipping pixel data if only_tags=True)
        - Applies DoseGridScaling
        - Extracts orientation, spacing, and geometry
        - Ensures consistent axial alignment
        - Registers dose into the global Data structure
    """

    def __init__(self, image_set, only_tags):
        if isinstance(image_set, list):
            self.image_set = image_set
        else:
            self.image_set = [image_set]

        self.only_tags = only_tags
        self.unverified = None
        self.base_position = None
        self.skipped_slice = None

        self.modality = "RTDOSE"

        self.filepaths = [img.filename for img in self.image_set]
        self.sops = [img.SOPInstanceUID for img in self.image_set]

        self.array = None
        if not self.only_tags:
            self._compute_array()

        self.orientation = self._compute_orientation()
        self.plane = self._compute_plane()
        self.spacing = self._compute_spacing()
        self.dimensions = self._compute_dimensions()

        self._verify_axial_orientation()
        self.image_matrix = self._compute_image_matrix()

        self.dose_name = create_dose_name(self.modality)

        dose = Dose(self)
        Data.dose[self.dose_name] = dose
        Data.dose_list += [self.dose_name]

    def _compute_array(self):
        """
        Builds 3D dose array and applies DoseGridScaling if present.
        """
        if (0x3004, 0x000E) in self.image_set[0]:
            slope = self.image_set[0].DoseGridScaling
        else:
            slope = 1

        dose = self.image_set[0].pixel_array * slope
        self.array = np.asarray(dose)

        if len(self.array.shape) == 2:
            self.array = self.array.reshape((1, self.array.shape[0], self.array.shape[1]))

        del self.image_set[0].PixelData

    def _compute_orientation(self):
        """
        Extracts image orientation from DICOM tags or functional groups.
        """
        orientation = np.asarray([1, 0, 0, 0, 1, 0])

        if "ImageOrientationPatient" in self.image_set[0]:
            orientation = np.asarray(self.image_set[0]["ImageOrientationPatient"].value)

        elif "SharedFunctionalGroupsSequence" in self.image_set[0]:
            try:
                seq = self.image_set[0][0]["SharedFunctionalGroupsSequence"][0]
                orientation = np.asarray(
                    seq["PlaneOrientationSequence"][0]["ImageOrientationPatient"].value
                )
            except Exception:
                self.unverified = "Orientation"

        else:
            self.unverified = "Orientation"

        return orientation

    def _compute_plane(self):
        """
        Determines anatomical plane from orientation vectors.
        """
        x = np.abs(self.orientation[0]) + np.abs(self.orientation[3])
        y = np.abs(self.orientation[1]) + np.abs(self.orientation[4])
        z = np.abs(self.orientation[2]) + np.abs(self.orientation[5])

        if x < y and x < z:
            return "Sagittal"
        elif y < x and y < z:
            return "Coronal"
        else:
            return "Axial"

    def _compute_spacing(self):
        """
        Computes voxel spacing and handles irregular slice spacing if present.
        """
        inplane_spacing = [1, 1]
        slice_thickness = np.double(self.image_set[0].SliceThickness)
        if np.isnan(slice_thickness):
            if 'GridFrameOffsetVector' in self.image_set[0]:
                grid_vector = self.image_set[0].GridFrameOffsetVector
                if len(grid_vector) > 1:
                    slice_thickness = grid_vector[1] - grid_vector[0]

        if 'PixelSpacing' in self.image_set[0]:
            inplane_spacing = self.image_set[0].PixelSpacing

        elif 'ContributingSourcesSequence' in self.image_set[0]:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['DetectorElementSpacing']

        elif 'PerFrameFunctionalGroupsSequence' in self.image_set[0]:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in self.image_set[0][sequence][0]:
                inplane_spacing = self.image_set[0][sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing']

        if len(self.image_set) > 1:
            row_direction = self.orientation[:3]
            column_direction = self.orientation[3:]
            slice_direction = np.cross(row_direction, column_direction)

            first = np.dot(slice_direction, self.image_set[0].ImagePositionPatient)
            last = np.dot(slice_direction, self.image_set[-1].ImagePositionPatient)
            slice_thickness = np.asarray((last - first) / (len(self.image_set) - 1))

        if self.plane == 'Axial':
            return np.asarray([inplane_spacing[1], inplane_spacing[0], slice_thickness])

        elif self.plane == 'Coronal':
            return np.asarray([inplane_spacing[1], slice_thickness, inplane_spacing[0]])

        else:
            return np.asarray([slice_thickness, inplane_spacing[1], inplane_spacing[0]])

    def _compute_dimensions(self):
        """
        Computes volume dimensions from array shape and orientation.
        """
        shape = self.array.shape

        if self.plane == "Axial":
            return np.asarray([shape[0], shape[1], shape[2]])
        elif self.plane == "Coronal":
            return np.asarray([shape[1], shape[0], shape[2]])
        else:
            return np.asarray([shape[1], shape[2], shape[0]])

    def _compute_image_matrix(self):
        """
        Constructs 3x3 orientation matrix (row, column, slice directions).
        """
        row = self.orientation[:3]
        col = self.orientation[3:]
        slc = np.cross(row, col)

        mat = np.identity(3, dtype=np.float32)
        mat[0, :3] = row
        mat[1, :3] = col
        mat[2, :3] = slc

        return mat

    def _verify_axial_orientation(self):
        """
        Ensures correct physical orientation of the dose grid and fixes flips/rotations if needed.
        """
        shape = self.array.shape
        if self.plane == 'Axial':
            spacing = self.spacing
        elif self.plane == 'Coronal':
            spacing = [self.spacing[0], self.spacing[2], self.spacing[1]]
        else:
            spacing = [self.spacing[1], self.spacing[2], self.spacing[0]]

        slices = shape[0] - 1
        y = shape[1] - 1
        x = shape[2] - 1

        origin = np.asarray(self.image_set[0]['ImagePositionPatient'].value)

        row_direction = self.orientation[:3]
        column_direction = self.orientation[3:]
        slice_direction = np.cross(row_direction, column_direction)

        corners = np.zeros((8, 3))
        corners[0] = origin
        corners[1] = origin + (x * spacing[0] * row_direction)
        corners[2] = origin + (y * spacing[1] * column_direction)
        corners[3] = (origin + (x * spacing[0] * row_direction) + (y * spacing[1] * column_direction))

        corners[4] = origin + (slices * spacing[2] * slice_direction)
        corners[5] = (origin + (slices * spacing[2] * slice_direction) + (x * spacing[0] * row_direction))
        corners[6] = (origin + (slices * spacing[2] * slice_direction) + (y * spacing[1] * column_direction))
        corners[7] = (origin + (slices * spacing[2] * slice_direction) + (x * spacing[0] * row_direction) +
                      (y * spacing[1] * column_direction))

        corner_idx = np.argmin(np.sum(corners, axis=1))
        if corner_idx != 0:
            self.origin = corners[corner_idx]
            if self.plane == "Axial":
                if corner_idx == 1:
                    self.array = np.rot90(self.array, 1, (1, 2))
                elif corner_idx == 2:
                    self.array = np.rot90(self.array, 3, (1, 2))
                else:
                    self.array = np.rot90(self.array, 2, (1, 2))

                if corner_idx < 4:
                    square = corners[:4, :]
                else:
                    square = corners[4:, :]

            elif self.plane == 'Coronal':
                self.array = np.rot90(self.array, 1, (0, 1))

                s1 = np.argsort(corners[:4, 2])
                s2 = np.argsort(corners[4:, 2]) + 4

                square = [corners[s1[0]], corners[s1[1]], corners[s2[0]], corners[s2[1]]]

            else:
                self.array = np.flip(np.rot90(self.array, 1, (0, 1)).transpose(0, 2, 1), axis=2)

                s1 = np.argsort(corners[:4, 2])
                s2 = np.argsort(corners[4:, 2]) + 4

                square = [corners[s1[0]], corners[s1[1]], corners[s2[0]], corners[s2[1]]]

            distances = np.asarray([np.linalg.norm(corners[corner_idx, :] - s) for s in square])
            sorted_args = np.argsort(distances)

            c1 = square[sorted_args[1]] - corners[corner_idx]
            c2 = square[sorted_args[2]] - corners[corner_idx]

            if np.abs(c1[0]) > np.abs(c2[0]):
                self.orientation[:3] = c1 / (self.spacing[0] * self.dimensions[2])
                self.orientation[3:] = c2 / (self.spacing[1] * self.dimensions[1])
            else:
                self.orientation[:3] = c2 / (self.spacing[0] * self.dimensions[2])
                self.orientation[3:] = c1 / (self.spacing[1] * self.dimensions[1])

            self.image_matrix = self._compute_image_matrix()

        else:
            self.origin = origin


def create_image_name(modality):
    """
    Generate a unique, sequential name for an image based on its modality.

    This function checks the current number of images in the global data list
    and appends a zero-padded index to the modality string to create a
    standardized identifier.

    Parameters
    ----------
    modality : str
        The imaging modality (e.g., 'CT', 'MR', 'PET').

    Returns
    -------
    str
        A formatted string containing the modality and a two-digit index
        (e.g., 'CT 01').

    Examples
    --------
    >>> # Assuming Data.image_list is empty
    >>> create_image_name('CT')
    'CT 01'
    >>> # Assuming Data.image_list has 9 items
    >>> create_image_name('MR')
    'MR 10'
    """
    idx = len(Data.image_list)
    if idx < 9:
        image_name = modality + ' 0' + str(1 + idx)
    else:
        image_name = modality + ' ' + str(1 + idx)

    return image_name


def create_dose_name(modality):
    """
    Generate a unique, sequential name for a dose based on its modality.

    This function calculates the next available index for a dose object
    and returns a formatted string identifier.

    Parameters
    ----------
    modality : str
        The type of dose or modality associated with it (e.g., 'RTDOSE').

    Returns
    -------
    str
        A formatted string containing the modality and a two-digit index
        (e.g., 'RTDOSE 01').

    Examples
    --------
    >>> # Assuming Data.dose_list has 2 items
    >>> create_dose_name('Dose')
    'Dose 03'
    """
    idx = len(Data.dose_list)
    if idx < 9:
        image_name = modality + ' 0' + str(1 + idx)
    else:
        image_name = modality + ' ' + str(1 + idx)

    return image_name
