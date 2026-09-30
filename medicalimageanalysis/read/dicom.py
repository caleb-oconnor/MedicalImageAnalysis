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
import hashlib
import threading
import itertools

import numpy as np
import pydicom as dicom
from pydicom.uid import generate_uid
from pydicom.pixels import apply_color_lut

from ..structure.deformable import Deformable
from ..structure.dose import Dose
from ..structure.image import Image
from ..structure.rigid import Rigid

from ..data import Data


def get_modality(series_sop_class):
    # --- Computed/Digital Radiography ---
    cr_sop_class = ["1.2.840.10008.5.1.4.1.1.1"]

    dx_sop_class = ["1.2.840.10008.5.1.4.1.1.1.1", "1.2.840.10008.5.1.4.1.1.1.1.1"]

    mammo_sop_class = ["1.2.840.10008.5.1.4.1.1.1.2", "1.2.840.10008.5.1.4.1.1.1.2.1"]

    intraoral_sop_class = ["1.2.840.10008.5.1.4.1.1.1.3", "1.2.840.10008.5.1.4.1.1.1.3.1"]

    # --- CT ---
    ct_sop_class = ["1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.3"]
    ct_sop_class_enhanced = ["1.2.840.10008.5.1.4.1.1.2.1", "1.2.840.10008.5.1.4.1.1.2.2",
                             "1.2.840.10008.5.1.4.1.1.2.4", "1.2.840.10008.5.1.4.1.1.2.5"]

    # --- Ultrasound ---
    us_sop_class = ["1.2.840.10008.5.1.4.1.1.6.1", "1.2.840.10008.5.1.4.1.1.3.1"]
    us_sop_class_enhanced = ["1.2.840.10008.5.1.4.1.1.6.2", "1.2.840.10008.5.1.4.1.1.6.3"]

    # --- MR ---
    mr_sop_class = ["1.2.840.10008.5.1.4.1.1.4", "1.2.840.10008.5.1.4.1.1.4.2"]
    mr_sop_class_enhanced = ["1.2.840.10008.5.1.4.1.1.4.1", "1.2.840.10008.5.1.4.1.1.4.3",
                             "1.2.840.10008.5.1.4.1.1.4.4"]

    # --- Secondary Capture ---
    sc_sop_class = ["1.2.840.10008.5.1.4.1.1.7", "1.2.840.10008.5.1.4.1.1.7.1",
                    "1.2.840.10008.5.1.4.1.1.7.2", "1.2.840.10008.5.1.4.1.1.7.3",
                    "1.2.840.10008.5.1.4.1.1.7.4"]

    # --- Waveforms ---
    waveform_sop_class = ["1.2.840.10008.5.1.4.1.1.9.1.1", "1.2.840.10008.5.1.4.1.1.9.1.2",
                          "1.2.840.10008.5.1.4.1.1.9.1.3", "1.2.840.10008.5.1.4.1.1.9.1.4",
                          "1.2.840.10008.5.1.4.1.1.9.2.1", "1.2.840.10008.5.1.4.1.1.9.3.1",
                          "1.2.840.10008.5.1.4.1.1.9.4.1", "1.2.840.10008.5.1.4.1.1.9.4.2",
                          "1.2.840.10008.5.1.4.1.1.9.5.1", "1.2.840.10008.5.1.4.1.1.9.6.1",
                          "1.2.840.10008.5.1.4.1.1.9.6.2", "1.2.840.10008.5.1.4.1.1.9.7.1",
                          "1.2.840.10008.5.1.4.1.1.9.7.2", "1.2.840.10008.5.1.4.1.1.9.7.3",
                          "1.2.840.10008.5.1.4.1.1.9.7.4", "1.2.840.10008.5.1.4.1.1.9.8.1"]

    # --- Presentation States ---
    presentation_state_sop_class = ["1.2.840.10008.5.1.4.1.1.9.100.1", "1.2.840.10008.5.1.4.1.1.9.100.2",
                                    "1.2.840.10008.5.1.4.1.1.11.1", "1.2.840.10008.5.1.4.1.1.11.2",
                                    "1.2.840.10008.5.1.4.1.1.11.3", "1.2.840.10008.5.1.4.1.1.11.4",
                                    "1.2.840.10008.5.1.4.1.1.11.5", "1.2.840.10008.5.1.4.1.1.11.6",
                                    "1.2.840.10008.5.1.4.1.1.11.7", "1.2.840.10008.5.1.4.1.1.11.8",
                                    "1.2.840.10008.5.1.4.1.1.11.9", "1.2.840.10008.5.1.4.1.1.11.10",
                                    "1.2.840.10008.5.1.4.1.1.11.11", "1.2.840.10008.5.1.4.1.1.11.12"]

    # --- Angiography / Fluoro ---
    xa_sop_class = ["1.2.840.10008.5.1.4.1.1.12.1"]
    xa_sop_class_enhanced = ["1.2.840.10008.5.1.4.1.1.12.1.1"]

    xrf_sop_class = ["1.2.840.10008.5.1.4.1.1.12.2"]
    xrf_sop_class_enhanced = ["1.2.840.10008.5.1.4.1.1.12.2.1"]

    # --- X-Ray 3D / Breast Tomosynthesis ---
    x3d_sop_class = ["1.2.840.10008.5.1.4.1.1.13.1.1", "1.2.840.10008.5.1.4.1.1.13.1.2"]

    tomosynthesis_sop_class = ["1.2.840.10008.5.1.4.1.1.13.1.3"]

    breast_projection_sop_class = ["1.2.840.10008.5.1.4.1.1.13.1.4", "1.2.840.10008.5.1.4.1.1.13.1.5"]

    # --- Intravascular OCT ---
    ivoct_sop_class = ["1.2.840.10008.5.1.4.1.1.14.1", "1.2.840.10008.5.1.4.1.1.14.2"]

    # --- Nuclear Medicine ---
    nm_sop_class = ["1.2.840.10008.5.1.4.1.1.20"]

    # --- Parametric Map / Raw / Registration / Segmentation ---
    parametric_map_sop_class = ["1.2.840.10008.5.1.4.1.1.30"]

    raw_data_sop_class = ["1.2.840.10008.5.1.4.1.1.66"]

    registration_sop_class = ["1.2.840.10008.5.1.4.1.1.66.1", "1.2.840.10008.5.1.4.1.1.66.2",
                              "1.2.840.10008.5.1.4.1.1.66.3"]

    segmentation_sop_class = ["1.2.840.10008.5.1.4.1.1.66.4", "1.2.840.10008.5.1.4.1.1.66.5",
                              "1.2.840.10008.5.1.4.1.1.66.6", "1.2.840.10008.5.1.4.1.1.66.7",
                              "1.2.840.10008.5.1.4.1.1.66.8"]

    real_world_value_sop_class = ["1.2.840.10008.5.1.4.1.1.67"]

    surface_scan_sop_class = ["1.2.840.10008.5.1.4.1.1.68.1", "1.2.840.10008.5.1.4.1.1.68.2"]

    # --- Visible Light ---
    vl_endoscopic_sop_class = ["1.2.840.10008.5.1.4.1.1.77.1.1", "1.2.840.10008.5.1.4.1.1.77.1.1.1"]

    vl_microscopic_sop_class = ["1.2.840.10008.5.1.4.1.1.77.1.2", "1.2.840.10008.5.1.4.1.1.77.1.2.1",
                                "1.2.840.10008.5.1.4.1.1.77.1.3"]

    vl_photographic_sop_class = ["1.2.840.10008.5.1.4.1.1.77.1.4", "1.2.840.10008.5.1.4.1.1.77.1.4.1"]

    whole_slide_sop_class = ["1.2.840.10008.5.1.4.1.1.77.1.6"]

    dermoscopic_sop_class = ["1.2.840.10008.5.1.4.1.1.77.1.7"]

    confocal_microscopy_sop_class = ["1.2.840.10008.5.1.4.1.1.77.1.8", "1.2.840.10008.5.1.4.1.1.77.1.9"]

    # --- Ophthalmic Photography / Imaging ---
    ophthalmic_photo_sop_class = ["1.2.840.10008.5.1.4.1.1.77.1.5.1", "1.2.840.10008.5.1.4.1.1.77.1.5.2",
                                  "1.2.840.10008.5.1.4.1.1.77.1.5.3", "1.2.840.10008.5.1.4.1.1.77.1.5.4",
                                  "1.2.840.10008.5.1.4.1.1.77.1.5.5", "1.2.840.10008.5.1.4.1.1.77.1.5.6",
                                  "1.2.840.10008.5.1.4.1.1.77.1.5.7", "1.2.840.10008.5.1.4.1.1.77.1.5.8"]

    # --- Ophthalmic Measurements ---
    ophthalmic_measurement_sop_class = ["1.2.840.10008.5.1.4.1.1.78.1", "1.2.840.10008.5.1.4.1.1.78.2",
                                        "1.2.840.10008.5.1.4.1.1.78.3", "1.2.840.10008.5.1.4.1.1.78.4",
                                        "1.2.840.10008.5.1.4.1.1.78.5", "1.2.840.10008.5.1.4.1.1.78.6",
                                        "1.2.840.10008.5.1.4.1.1.78.7", "1.2.840.10008.5.1.4.1.1.78.8",
                                        "1.2.840.10008.5.1.4.1.1.79.1", "1.2.840.10008.5.1.4.1.1.80.1",
                                        "1.2.840.10008.5.1.4.1.1.81.1", "1.2.840.10008.5.1.4.1.1.82.1"]

    # --- Structured Reports ---
    sr_sop_class = ["1.2.840.10008.5.1.4.1.1.88.11", "1.2.840.10008.5.1.4.1.1.88.22",
                    "1.2.840.10008.5.1.4.1.1.88.33", "1.2.840.10008.5.1.4.1.1.88.34",
                    "1.2.840.10008.5.1.4.1.1.88.35", "1.2.840.10008.5.1.4.1.1.88.40",
                    "1.2.840.10008.5.1.4.1.1.88.50", "1.2.840.10008.5.1.4.1.1.88.59",
                    "1.2.840.10008.5.1.4.1.1.88.65", "1.2.840.10008.5.1.4.1.1.88.67",
                    "1.2.840.10008.5.1.4.1.1.88.68", "1.2.840.10008.5.1.4.1.1.88.69",
                    "1.2.840.10008.5.1.4.1.1.88.70", "1.2.840.10008.5.1.4.1.1.88.71",
                    "1.2.840.10008.5.1.4.1.1.88.72", "1.2.840.10008.5.1.4.1.1.88.73",
                    "1.2.840.10008.5.1.4.1.1.88.74", "1.2.840.10008.5.1.4.1.1.88.75",
                    "1.2.840.10008.5.1.4.1.1.88.76", "1.2.840.10008.5.1.4.1.1.88.77"]

    # --- Content Assessment / Microscopy Annotations ---
    content_assessment_sop_class = ["1.2.840.10008.5.1.4.1.1.90.1"]

    microscopy_annotation_sop_class = ["1.2.840.10008.5.1.4.1.1.91.1"]

    # --- Encapsulated Documents ---
    encapsulated_doc_sop_class = ["1.2.840.10008.5.1.4.1.1.104.1", "1.2.840.10008.5.1.4.1.1.104.2",
                                  "1.2.840.10008.5.1.4.1.1.104.3", "1.2.840.10008.5.1.4.1.1.104.4",
                                  "1.2.840.10008.5.1.4.1.1.104.5"]

    # --- PET ---
    pet_sop_class = ["1.2.840.10008.5.1.4.1.1.128"]
    pet_sop_class_enhanced = ["1.2.840.10008.5.1.4.1.1.130", "1.2.840.10008.5.1.4.1.1.128.1"]

    # --- Structured Display ---
    structured_display_sop_class = ["1.2.840.10008.5.1.4.1.1.131"]

    # --- Performed Procedure Protocols ---
    performed_protocol_sop_class = ["1.2.840.10008.5.1.4.1.1.200.2", "1.2.840.10008.5.1.4.1.1.200.8"]

    # --- Radiotherapy (RT) ---
    rt_image_sop_class = ["1.2.840.10008.5.1.4.1.1.481.1"]
    rt_image_sop_class_enhanced = ["1.2.840.10008.5.1.4.1.1.481.23", "1.2.840.10008.5.1.4.1.1.481.24"]

    rt_dose_sop_class = ["1.2.840.10008.5.1.4.1.1.481.2"]

    rt_structure_set_sop_class = ["1.2.840.10008.5.1.4.1.1.481.3"]

    rt_plan_sop_class = ["1.2.840.10008.5.1.4.1.1.481.5"]
    rt_ion_plan_sop_class = ["1.2.840.10008.5.1.4.1.1.481.8"]

    rt_treatment_record_sop_class = ["1.2.840.10008.5.1.4.1.1.481.4", "1.2.840.10008.5.1.4.1.1.481.6",
                                     "1.2.840.10008.5.1.4.1.1.481.7", "1.2.840.10008.5.1.4.1.1.481.9"]

    rt_intent_annotation_sop_class = ["1.2.840.10008.5.1.4.1.1.481.10", "1.2.840.10008.5.1.4.1.1.481.11"]

    rt_radiation_set_sop_class = ["1.2.840.10008.5.1.4.1.1.481.12", "1.2.840.10008.5.1.4.1.1.481.16",
                                  "1.2.840.10008.5.1.4.1.1.481.17", "1.2.840.10008.5.1.4.1.1.481.21"]

    rt_radiation_sop_class = ["1.2.840.10008.5.1.4.1.1.481.13", "1.2.840.10008.5.1.4.1.1.481.14",
                              "1.2.840.10008.5.1.4.1.1.481.15"]

    rt_radiation_record_sop_class = ["1.2.840.10008.5.1.4.1.1.481.18", "1.2.840.10008.5.1.4.1.1.481.19",
                                     "1.2.840.10008.5.1.4.1.1.481.20"]

    rt_treatment_prep_sop_class = ["1.2.840.10008.5.1.4.1.1.481.22", "1.2.840.10008.5.1.4.1.1.481.25"]

    rt_beams_delivery_instruction_sop_class = ["1.2.840.10008.5.1.4.34.7", "1.2.840.10008.5.1.4.34.10"]

    _SOP_CLASS_GROUPS = {
        "CR": (cr_sop_class, None),
        "DX": (dx_sop_class, None),
        "MG": (mammo_sop_class, None),
        "IO": (intraoral_sop_class, None),
        "CT": (ct_sop_class, ct_sop_class_enhanced),
        "US": (us_sop_class, us_sop_class_enhanced),
        "MR": (mr_sop_class, mr_sop_class_enhanced),
        "SC": (sc_sop_class, None),
        "WAVEFORM": (waveform_sop_class, None),
        "PR": (presentation_state_sop_class, None),
        "XA": (xa_sop_class, xa_sop_class_enhanced),
        "RF": (xrf_sop_class, xrf_sop_class_enhanced),
        "X3D": (x3d_sop_class, None),
        "TOMOSYNTHESIS": (tomosynthesis_sop_class, None),
        "BREAST_PROJECTION": (breast_projection_sop_class, None),
        "IVOCT": (ivoct_sop_class, None),
        "NM": (nm_sop_class, None),
        "PARAMETRIC_MAP": (parametric_map_sop_class, None),
        "RAW": (raw_data_sop_class, None),
        "REG": (registration_sop_class, None),
        "SEG": (segmentation_sop_class, None),
        "RWV": (real_world_value_sop_class, None),
        "SURFACE_SCAN": (surface_scan_sop_class, None),
        "VL_ENDOSCOPIC": (vl_endoscopic_sop_class, None),
        "VL_MICROSCOPIC": (vl_microscopic_sop_class, None),
        "VL_PHOTOGRAPHIC": (vl_photographic_sop_class, None),
        "WHOLE_SLIDE": (whole_slide_sop_class, None),
        "DERMOSCOPIC": (dermoscopic_sop_class, None),
        "CONFOCAL": (confocal_microscopy_sop_class, None),
        "OPHTHALMIC_PHOTO": (ophthalmic_photo_sop_class, None),
        "OPHTHALMIC_MEASUREMENT": (ophthalmic_measurement_sop_class, None),
        "SR": (sr_sop_class, None),
        "CONTENT_ASSESSMENT": (content_assessment_sop_class, None),
        "MICROSCOPY_ANNOTATION": (microscopy_annotation_sop_class, None),
        "ENCAPSULATED_DOC": (encapsulated_doc_sop_class, None),
        "PET": (pet_sop_class, pet_sop_class_enhanced),
        "STRUCTURED_DISPLAY": (structured_display_sop_class, None),
        "PERFORMED_PROTOCOL": (performed_protocol_sop_class, None),
        "RTIMAGE": (rt_image_sop_class, rt_image_sop_class_enhanced),
        "RTDOSE": (rt_dose_sop_class, None),
        "RTSTRUCT": (rt_structure_set_sop_class, None),
        "RTPLAN": (rt_plan_sop_class, None),
        "RTIONPLAN": (rt_ion_plan_sop_class, None),
        "RT_TREATMENT_RECORD": (rt_treatment_record_sop_class, None),
        "RT_INTENT_ANNOTATION": (rt_intent_annotation_sop_class, None),
        "RT_RADIATION_SET": (rt_radiation_set_sop_class, None),
        "RT_RADIATION": (rt_radiation_sop_class, None),
        "RT_RADIATION_RECORD": (rt_radiation_record_sop_class, None),
        "RT_TREATMENT_PREP": (rt_treatment_prep_sop_class, None),
        "RT_BEAMS_DELIVERY_INSTRUCTION": (rt_beams_delivery_instruction_sop_class, None),
    }

    # Flatten into a single UID -> (modality, enhanced) lookup dict, built once.
    _SOP_CLASS_LOOKUP = {}
    for _modality, (_plain, _enhanced) in _SOP_CLASS_GROUPS.items():
        for _uid in _plain:
            _SOP_CLASS_LOOKUP[_uid] = (_modality, False)
        if _enhanced:
            for _uid in _enhanced:
                _SOP_CLASS_LOOKUP[_uid] = (_modality, True)

    return _SOP_CLASS_LOOKUP.get(series_sop_class, (None, None))


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
            images_in_modality = [d for d in self.ds if (0x0008, 0x0016) in d
                                  if get_modality(d[0x0008, 0x0016].value)[0] == modality]

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

    Handles CT/MR/PT as either classic single-frame series or enhanced
    (multi-frame) objects. The volume is:
    - Sorted along the slice normal, with duplicate positions removed
    - Rescaled per slice (RescaleSlope/RescaleIntercept)
    - Gap-filled by linear interpolation where slices are missing
    - Reoriented so the array is (z, y, x) with each axis increasing along +z, +y, +x
    - Registered into the global `Data` structure

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        Classic slices of one volume, or one or more enhanced multi-frame datasets.
    only_tags : bool
        If True, only geometry/metadata is computed (pixel data is not decoded).

    Attributes
    ----------
    array : np.ndarray or None
        Volume in (z, y, x) order. int16 when all rescale values are integers and the
        data fits, int32 if it overflows int16, float32 for non-integer rescale (e.g. PT).
    spacing : np.ndarray
        Voxel spacing [x, y, z] in mm.
    dimensions : np.ndarray
        Volume shape [z, y, x] (also valid when only_tags=True).
    origin : np.ndarray
        Patient-space position of voxel array[0, 0, 0].
    image_matrix : np.ndarray
        3x3, rows are the patient-space directions of the array's x, y, z axes.
        Identity for any orthogonal acquisition, residual rotation for oblique ones.
    orientation : np.ndarray
        Direction cosines of the reoriented array (row = x axis, column = y axis).
    acquisition_orientation : np.ndarray
        ImageOrientationPatient as acquired (orthonormalized).
    plane : str
        Acquisition plane (Axial, Coronal, Sagittal).
    enhanced : bool
        True if the input contained enhanced multi-frame DICOM.
    skipped_slice : list[int]
        Index (in the sorted real slices) of the slice following each gap.
    missing_slices : list[dict]
        Per gap: insert_index into the stacked volume, num_missing, and the labels
        of the bounding slices (SOPInstanceUID, or (SOPInstanceUID, frame) for enhanced).
    unverified : str or None
        First geometry problem found; all problems are in `unverified_reasons`.
    frame_datasets, frame_numbers, frame_positions, frame_orientations,
    frame_pixel_spacings, frame_slopes, frame_intercepts, frame_thicknesses : list
        Per-frame values, sorted along the slice normal with duplicates removed.
        frame_numbers is None for classic slices, the 0-based frame for enhanced.
    duplicate_frames : list
        Labels of frames dropped for sharing a position with an earlier frame.
    image_name : str
        Generated internal image identifier.

    Examples
    --------
    Basic usage::

        reader = Read3D(dicom_series, only_tags=False)
        print(reader.image_name)
    """
    def __init__(self, image_set, only_tags):
        self.image_set = image_set if isinstance(image_set, list) else [image_set]
        self.only_tags = only_tags

        # --- internal state ---
        self.unverified = None
        self.unverified_reasons = []
        self.base_position = None
        self.skipped_slice = []
        self.missing_slices = []
        self.duplicate_frames = []
        self.rgb = False

        # --- metadata ---
        self.modality = self.image_set[0].Modality
        self.enhanced = any(self._is_enhanced(ds) for ds in self.image_set)

        # --- geometry (header only, no pixel decoding) ---
        self._collect_frames()
        self.acquisition_orientation = self.compute_orientation()
        self.plane = self.compute_plane()
        self.slice_spacing, self.slice_map = self.compute_slice_layout()
        self.compute_geometry()

        # after sorting/deduplication so they line up with the real slices
        self.filepaths = [getattr(ds, 'filename', None) for ds in self.image_set]
        self.sops = [ds.SOPInstanceUID for ds in self.image_set]

        # --- volume data ---
        self.array = None
        if not self.only_tags:
            self.compute_array()

        self.image_name = create_image_name(self.modality)

        # --- register into global system ---
        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)

    @staticmethod
    def _functional_item(ds, frame_index, keyword):
        """
        First item of a functional-group macro for one frame.
        Per-frame groups take priority over shared groups.
        """
        seq = ds.PerFrameFunctionalGroupsSequence[frame_index].get(keyword)
        if seq:
            return seq[0]
        shared = ds.get('SharedFunctionalGroupsSequence')
        if shared:
            seq = shared[0].get(keyword)
            if seq:
                return seq[0]
        return None

    @staticmethod
    def _is_enhanced(ds):
        return 'PerFrameFunctionalGroupsSequence' in ds

    @staticmethod
    def _number(value, default=None):
        if value is None or isinstance(value, (str, bytes)) and not value:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _vector(value):
        if value is None or isinstance(value, (str, bytes)):
            return None
        arr = np.asarray(value, dtype=np.float64).ravel()
        return arr if arr.size else None

    def _add_classic_frame(self, ds):
        spacing = self._vector(ds.get('PixelSpacing'))
        if spacing is None:
            spacing = self._vector(ds.get('ImagerPixelSpacing'))
        if spacing is None and ds.get('ContributingSourcesSequence'):
            spacing = self._vector(ds.ContributingSourcesSequence[0].get('DetectorElementSpacing'))

        self._add_frame(
            ds=ds,
            frame_number=None,
            position=self._vector(ds.get('ImagePositionPatient')),
            orientation=self._vector(ds.get('ImageOrientationPatient')),
            pixel_spacing=spacing,
            slope=self._number(ds.get('RescaleSlope'), 1.0) or 1.0,
            intercept=self._number(ds.get('RescaleIntercept'), 0.0),
            thickness=self._number(ds.get('SpacingBetweenSlices'),
                                   self._number(ds.get('SliceThickness'))),
        )

    def _add_enhanced_frames(self, ds):
        def get(item, keyword):
            return item.get(keyword) if item is not None else None

        for f in range(len(ds.PerFrameFunctionalGroupsSequence)):
            pos = self._functional_item(ds, f, 'PlanePositionSequence')
            ori = self._functional_item(ds, f, 'PlaneOrientationSequence')
            pix = self._functional_item(ds, f, 'PixelMeasuresSequence')
            val = self._functional_item(ds, f, 'PixelValueTransformationSequence')

            # some legacy-converted objects keep rescale at the top level
            slope = get(val, 'RescaleSlope') if val is not None else ds.get('RescaleSlope')
            intercept = get(val, 'RescaleIntercept') if val is not None else ds.get('RescaleIntercept')

            self._add_frame(
                ds=ds,
                frame_number=f,
                position=self._vector(get(pos, 'ImagePositionPatient')),
                orientation=self._vector(get(ori, 'ImageOrientationPatient')),
                pixel_spacing=self._vector(get(pix, 'PixelSpacing')),
                slope=self._number(slope, 1.0) or 1.0,
                intercept=self._number(intercept, 0.0),
                thickness=self._number(get(pix, 'SpacingBetweenSlices'),
                                       self._number(get(pix, 'SliceThickness'))),
            )

    def _add_frame(self, ds, frame_number, position, orientation, pixel_spacing,
                   slope, intercept, thickness):
        self.frame_datasets.append(ds)
        self.frame_numbers.append(frame_number)
        self.frame_positions.append(position)
        self.frame_orientations.append(orientation)
        self.frame_pixel_spacings.append(pixel_spacing)
        self.frame_slopes.append(slope)
        self.frame_intercepts.append(intercept)
        self.frame_thicknesses.append(thickness)

    def _collect_frames(self):
        """
        Flatten classic slices and enhanced frames into parallel per-frame lists.
        """
        shapes = {(int(ds.Rows), int(ds.Columns)) for ds in self.image_set}
        if len(shapes) > 1:
            raise ValueError(f'Cannot stack slices with different Rows/Columns: {sorted(shapes)}')

        self.frame_datasets = []  # owning pydicom Dataset
        self.frame_numbers = []  # frame in multi-frame PixelData, None for classic
        self.frame_positions = []  # ImagePositionPatient (3,)
        self.frame_orientations = []  # ImageOrientationPatient (6,)
        self.frame_pixel_spacings = []  # [row spacing, column spacing]
        self.frame_slopes = []
        self.frame_intercepts = []
        self.frame_thicknesses = []  # SpacingBetweenSlices, else SliceThickness

        for ds in self.image_set:
            if self._is_enhanced(ds):
                self._add_enhanced_frames(ds)
            else:
                self._add_classic_frame(ds)

    def _flag(self, reason):
        if reason not in self.unverified_reasons:
            self.unverified_reasons.append(reason)
        if self.unverified is None:
            self.unverified = reason

    def _frame_label(self, i):
        uid = self.frame_datasets[i].SOPInstanceUID
        n = self.frame_numbers[i]
        return uid if n is None else (uid, n + 1)

    def _interpolate_missing(self, volume):
        """
        Linearly interpolate missing slices in rescaled space (in place).
        """
        real = [k for k, idx in enumerate(self.slice_map) if idx is not None]
        is_int = np.issubdtype(volume.dtype, np.integer)

        for lo, hi in zip(real[:-1], real[1:]):
            if hi - lo < 2:
                continue
            a = volume[lo].astype(np.float32)
            b = volume[hi].astype(np.float32)
            for k in range(lo + 1, hi):
                t = (k - lo) / (hi - lo)
                plane = (1.0 - t) * a + t * b
                volume[k] = np.rint(plane) if is_int else plane

    def _nominal_thickness(self):
        for thickness in self.frame_thicknesses:
            if thickness:
                return abs(thickness)
        self._flag('Spacing')
        return 1.0

    def _read_plane(self, i, cache):
        """
        Stored pixel values of frame i. PixelData is released after decoding.
        """
        ds = self.frame_datasets[i]
        frame_number = self.frame_numbers[i]
        if frame_number is None:
            plane = ds.pixel_array
            del ds.PixelData

            return plane

        key = id(ds)
        if key not in cache:
            data = ds.pixel_array
            cache[key] = data[np.newaxis] if data.ndim == 2 else data
            del ds.PixelData

        return cache[key][frame_number]

    def _reorder_frames(self, order):
        """
        Reorder (or subset) every per-frame list with the same index list.
        """
        for name in ('frame_datasets', 'frame_numbers', 'frame_positions', 'frame_orientations',
                     'frame_pixel_spacings', 'frame_slopes', 'frame_intercepts', 'frame_thicknesses'):
            values = getattr(self, name)
            setattr(self, name, [values[i] for i in order])

    def _reorient(self, volume):
        """
        Permute/flip the stacked volume into (+z, +y, +x).
        """
        if self._stack_perm == (0, 1, 2) and not any(self._stack_flips):
            return volume

        volume = volume.transpose(self._stack_perm)
        flip_axes = tuple(i for i, f in enumerate(self._stack_flips) if f)
        if flip_axes:
            volume = np.flip(volume, axis=flip_axes)

        return np.ascontiguousarray(volume)

    def compute_array(self):
        """
        Stack rescaled frames, interpolate gaps and reorient to (z, y, x).

        _INT16_MIN, _INT16_MAX = np.iinfo(np.int16).min, np.iinfo(np.int16).max
        """
        ds0 = self.image_set[0]
        rows, cols = int(ds0.Rows), int(ds0.Columns)
        integral = all(float(s).is_integer() and float(b).is_integer()
                       for s, b in zip(self.frame_slopes, self.frame_intercepts))
        volume = np.empty((len(self.slice_map), rows, cols),
                          dtype=np.int16 if integral else np.float32)

        cache = {}
        for k, idx in enumerate(self.slice_map):
            if idx is None:
                continue
            raw = self._read_plane(idx, cache)
            slope, intercept = self.frame_slopes[idx], self.frame_intercepts[idx]

            if integral:
                plane = raw.astype(np.int32) * int(slope) + int(intercept)
                if volume.dtype == np.int16 and (plane.min() < np.iinfo(np.int16).min or plane.max() > np.iinfo(np.int16).max):
                    volume = volume.astype(np.int32)

            else:
                plane = raw.astype(np.float32) * np.float32(slope) + np.float32(intercept)

            volume[k] = plane
        cache.clear()

        self._interpolate_missing(volume)
        self.array = self._reorient(volume)

    def compute_geometry(self):
        """
        Work out how the stacked volume maps to (z, y, x) and the resulting
        spacing, dimensions, origin and image matrix. Header only, so it is
        valid with only_tags=True; the array itself is permuted in _reorient.
        """
        row = self.acquisition_orientation[:3]
        col = self.acquisition_orientation[3:]
        normal = np.cross(row, col)
        pixel_spacing = self._compute_pixel_spacing()

        # stacked axes: 0 = slices (normal), 1 = rows (column dir), 2 = columns (row dir)
        axis_dirs = np.stack([normal, col, row])
        axis_spacing = np.array([self.slice_spacing, pixel_spacing[0], pixel_spacing[1]],
                                dtype=np.float64)
        ds0 = self.image_set[0]
        stacked_shape = np.array([len(self.slice_map), int(ds0.Rows), int(ds0.Columns)])

        # patient axis (0=x, 1=y, 2=z) for each stacked axis; the permutation with the
        # best total alignment, so 45-degree obliques can't map two axes to one
        axis_to_world = max(itertools.permutations(range(3)),
                            key=lambda p: sum(abs(axis_dirs[a, p[a]]) for a in range(3)))
        world_to_axis = np.argsort(axis_to_world)
        perm = tuple(int(world_to_axis[w]) for w in (2, 1, 0))
        flips = tuple(bool(axis_dirs[a, axis_to_world[a]] < 0) for a in perm)

        # array[0, 0, 0] moves to the far end of every flipped axis
        origin = np.asarray(self.frame_positions[0], dtype=np.float64).copy()
        for a in range(3):
            if axis_dirs[a, axis_to_world[a]] < 0:
                origin += (stacked_shape[a] - 1) * axis_spacing[a] * axis_dirs[a]

        z_dir, y_dir, x_dir = (axis_dirs[a] * (-1.0 if f else 1.0) for a, f in zip(perm, flips))

        self._stack_perm = perm
        self._stack_flips = flips
        self.origin = origin
        self.spacing = axis_spacing[[perm[2], perm[1], perm[0]]]
        self.dimensions = stacked_shape[list(perm)]
        self.orientation = np.concatenate([x_dir, y_dir])
        self.image_matrix = np.array([x_dir, y_dir, z_dir], dtype=np.float32)

    def compute_orientation(self):
        """
        Common ImageOrientationPatient of all frames, orthonormalized.

        _ORIENTATION_TOL = 1e-3  # max direction-cosine difference between frames of one volume
        """
        orientations = [o for o in self.frame_orientations if o is not None and o.size == 6]
        if not orientations:
            self._flag('Orientation')
            return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        ref = orientations[0]
        if any(np.max(np.abs(o - ref)) > 1e-3 for o in orientations[1:]):
            self._flag('Orientation')

        row = ref[:3] / np.linalg.norm(ref[:3])
        col = ref[3:] - np.dot(ref[3:], row) * row
        col /= np.linalg.norm(col)
        return np.concatenate([row, col])

    def compute_plane(self):
        """
        Acquisition plane from the dominant axis of the slice normal.
        """
        normal = np.cross(self.acquisition_orientation[:3], self.acquisition_orientation[3:])
        return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(normal)))]

    def compute_pixel_spacing(self):
        """
        _SPACING_TOL = 1e-4  # mm, max PixelSpacing difference between frames
        """
        spacings = [s for s in self.frame_pixel_spacings if s is not None and s.size == 2]
        if not spacings:
            self._flag('Spacing')
            return np.array([1.0, 1.0])
        ref = spacings[0]
        if any(np.max(np.abs(s - ref)) > 1e-4 for s in spacings[1:]):
            self._flag('Spacing')
        return ref

    def compute_slice_layout(self):
        """
        Sort frames along the slice normal, drop duplicates and detect gaps.

        _POSITION_TOL = 0.01  # mm, frames closer than this along the normal are duplicates
        _IRREGULAR_TOL = 0.05  # mm, max deviation of a real slice from the uniform slice grid

        Returns
        -------
        spacing : float
            Slice spacing along the normal.
        slice_map : list
            One entry per slice of the stacked volume: index into the frame lists,
            or None for a missing slice that will be interpolated.
        """
        normal = np.cross(self.acquisition_orientation[:3], self.acquisition_orientation[3:])

        if any(p is None for p in self.frame_positions):
            self._flag('Position')
            if not self.enhanced:
                order = sorted(range(len(self.frame_datasets)),
                               key=lambda i: self._number(self.frame_datasets[i].get('InstanceNumber'), 0.0))
                self._reorder_frames(order)
            step = self._nominal_thickness()
            base = next((p for p in self.frame_positions if p is not None), np.zeros(3))
            self.frame_positions = [base + k * step * normal for k in range(len(self.frame_positions))]

        proj = np.array([np.dot(normal, p) for p in self.frame_positions])
        order = np.argsort(proj, kind='stable')
        proj = proj[order]

        # duplicate positions (multi-stack enhanced MR, repeated slices): keep the first
        keep = [0]
        for i in range(1, len(order)):
            if proj[i] - proj[keep[-1]] < 0.01:
                self.duplicate_frames.append(int(order[i]))
            else:
                keep.append(i)
        if self.duplicate_frames:
            self._flag('Duplicate')
            self.duplicate_frames = [self._frame_label(i) for i in self.duplicate_frames]

        self._reorder_frames([int(order[i]) for i in keep])
        proj = proj[keep]

        if not self.enhanced:
            self.image_set = list(self.frame_datasets)

        if len(proj) == 1:
            return self._nominal_thickness(), [0]

        diffs = np.diff(proj)
        n_steps = np.maximum(np.rint(diffs / np.median(diffs)).astype(int), 1)
        grid = np.concatenate([[0], np.cumsum(n_steps)])
        spacing = float((proj[-1] - proj[0]) / grid[-1])

        if np.max(np.abs(proj - (proj[0] + grid * spacing))) > 0.05:
            self._flag('Irregular')

        slice_map = [None] * (int(grid[-1]) + 1)
        for i, g in enumerate(grid):
            slice_map[int(g)] = i

        for i, n in enumerate(n_steps):
            if n > 1:
                self._flag('Skipped')
                self.skipped_slice.append(i + 1)
                self.missing_slices.append({
                    'insert_index': int(grid[i]) + 1,
                    'num_missing': int(n) - 1,
                    'between': (self._frame_label(i), self._frame_label(i + 1)),
                })

        return spacing, slice_map


class ReadXRay:
    """
    Reads 2D X-ray (DX / CR / MG) and tomosynthesis into a z-y-x array.

    Array axes (0, 1, 2) always run along patient +z, +y, +x (LPS), and
    image_matrix / orientation / origin / spacing describe that array.

    Orientation source, in priority order
    -------------------------------------
    1. ImageOrientationPatient (top level or functional groups; tomo has it)
    2. PatientOrientation letters
    3. ViewPosition (+ laterality for mammography)
    4. identity (row +x, column +y)

    Tomo frames are sorted along the slice normal and the origin is the
    first frame's ImagePositionPatient. 2D images without a position get
    origin (0, 0, 0).

    Parameters
    ----------
    image_set : pydicom.Dataset or list
        X-ray dataset. Only the first is read; tomo is one multi-frame file.
    only_tags : bool
        If True, metadata only. Geometry is still computed from tags.
    """

    def __init__(self, image_set, only_tags):
        self.image_set = image_set if isinstance(image_set, list) else [image_set]
        self.only_tags = only_tags
        self.ds = self.image_set[0]
        ds = self.ds

        self.unverified = 'Modality'
        self.skipped_slice = None
        self.rgb = False

        self.modality = ds.Modality
        self.filepaths = getattr(ds, 'filename', None)
        self.sops = ds.SOPInstanceUID

        self.number_of_frames = int(ds.get('NumberOfFrames', 1) or 1)
        self.is_tomo = False
        self.spacing_known = False
        self._frame_order = None
        self._frame_origin = None
        self._slice_dz = 1.0

        self.origin = [0, 0, 0]
        self.spacing = None
        self.dimensions = None
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.image_matrix = np.identity(3)

        # Stored (pre-reorientation) row / column directions
        self.orientation = np.asarray(self.initial_orientation(), dtype=float)
        self.plane = self.compute_plane()
        self._setup_frames()

        self.array = None
        if not self.only_tags:
            self.compute_array()

        self.compute_geometry()

        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)

    @staticmethod
    def _fg_value(frame, shared, seq_name, attr):
        """
        Functional-group attribute, per-frame first, then shared.
        """
        for group in (frame, shared):
            if group is None:
                continue

            seq = group.get(seq_name)
            if seq and attr in seq[0]:
                return getattr(seq[0], attr)

        return None

    def _frame_positions(self):
        """(F, 3) ImagePositionPatient per frame, or None if incomplete."""
        per = self.ds.get('PerFrameFunctionalGroupsSequence')
        if not per or len(per) != self.number_of_frames:
            return None

        shared = self._shared_group()

        positions = []
        for frame in per:
            ipp = self._fg_value(frame, shared, 'PlanePositionSequence', 'ImagePositionPatient')
            if ipp is None:
                return None

            positions.append([float(v) for v in ipp])

        return np.array(positions)

    @staticmethod
    def _letters_to_dir(value):
        """
        'L', 'FR', ... -> unit LPS vector, or None.
        """

        # Patient-direction letters -> LPS unit vectors
        letter_dir = {
            'L': (1.0, 0.0, 0.0), 'R': (-1.0, 0.0, 0.0),
            'P': (0.0, 1.0, 0.0), 'A': (0.0, -1.0, 0.0),
            'H': (0.0, 0.0, 1.0), 'F': (0.0, 0.0, -1.0),
        }

        vec = np.zeros(3)
        for ch in str(value).strip().upper():
            if ch not in letter_dir:
                return None

            vec += letter_dir[ch]
        norm = np.linalg.norm(vec)

        return vec / norm if norm > 0 else None

    def _inplane_spacing(self):
        """
        [row, col] spacing in mm, or None.
        """
        ds = self.ds
        if 'PixelSpacing' in ds:
            return [float(v) for v in ds.PixelSpacing]

        ps = self._pixel_measures('PixelSpacing')
        if ps is not None:
            return [float(v) for v in ps]

        if 'ImagerPixelSpacing' in ds:
            # Detector-plane spacing; divide by magnification for anatomy spacing
            mag = float(ds.get('EstimatedRadiographicMagnificationFactor', 1.0) or 1.0)
            return [float(v) / mag for v in ds.ImagerPixelSpacing]

        seq = ds.get('ContributingSourcesSequence')
        if seq and 'DetectorElementSpacing' in seq[0]:
            return [float(v) for v in seq[0].DetectorElementSpacing]

        return None

    def _orientation_from_view(self):
        # ViewPosition -> (row direction, column direction), used only when the file
        # has no ImageOrientationPatient and no PatientOrientation. Assumes the image
        # is stored display-ready (patient's right on the viewer's left for AP/PA).
        view_orientation = {
            'AP': ('L', 'F'),
            'PA': ('L', 'F'),
            'LL': ('P', 'F'),
            'LAT': ('P', 'F'),
            'RL': ('A', 'F'),
        }

        # Mammography views depend on laterality (IHE Mammography Image profile)
        mammo_orientation = {
            ('CC', 'R'): ('P', 'L'),
            ('CC', 'L'): ('A', 'R'),
            ('MLO', 'R'): ('P', 'FL'),
            ('MLO', 'L'): ('A', 'FR'),
        }

        ds = self.ds
        view = str(ds.get('ViewPosition', '') or '').strip().upper()
        if not view:
            return None

        laterality = str(ds.get('ImageLaterality') or ds.get('Laterality') or '').strip().upper()
        if (view, laterality) in mammo_orientation:
            return mammo_orientation[(view, laterality)]

        return view_orientation.get(view)

    def _pixel_measures(self, attr):
        per = self.ds.get('PerFrameFunctionalGroupsSequence')
        frame = per[0] if per else None

        return self._fg_value(frame, self._shared_group(), 'PixelMeasuresSequence', attr)

    def _setup_frames(self):
        if self.number_of_frames == 1:
            self._slice_dz = 1.0
            return

        dz = 0.0
        positions = self._frame_positions()
        if positions is not None:
            normal = np.cross(self.orientation[:3], self.orientation[3:])
            proj = positions @ normal
            order = np.argsort(proj)

            self.is_tomo = True
            self._frame_order = order
            self._frame_origin = positions[order[0]]
            if len(order) > 1:
                dz = float(np.median(np.diff(proj[order])))

        if not dz > 0:
            dz = self._slice_spacing() or 1.0

        self._slice_dz = dz

    def _shared_group(self):
        seq = self.ds.get('SharedFunctionalGroupsSequence')

        return seq[0] if seq else None

    def _slice_spacing(self):
        for value in (self._pixel_measures('SpacingBetweenSlices'),
                      self.ds.get('SpacingBetweenSlices'),
                      self._pixel_measures('SliceThickness')):
            if value is not None and float(value) > 0:
                return float(value)

        return None

    def compute_array(self):
        ds = self.ds
        arr = ds.pixel_array
        bits = int(ds.get('BitsStored', 16))
        arr = arr.astype(np.int16 if bits <= 15 else np.int32, copy=False)
        del ds.PixelData

        photometric = str(ds.get('PhotometricInterpretation', 'MONOCHROME2')).upper()
        lut_shape = str(ds.get('PresentationLUTShape', '')).upper()
        if photometric == 'MONOCHROME1' or lut_shape == 'INVERSE':
            if int(ds.get('PixelRepresentation', 0)) == 0:
                arr = ((1 << bits) - 1) - arr

            else:
                arr = (arr.max() + arr.min()) - arr

        if arr.ndim == 2:
            arr = arr[np.newaxis]  # (1, R, C)

        if self._frame_order is not None:
            arr = arr[self._frame_order]

        self.array = arr  # (frames, rows, cols); compute_geometry reorders

    def compute_geometry(self):
        """
        Reorders and flips the grid so array axes (0, 1, 2) run along +z, +y, +x.

        Works from tags alone, so it is valid with only_tags=True; the array
        is transformed alongside when it has been loaded.
        """
        row = np.asarray(self.orientation[:3], dtype=float)
        col = np.asarray(self.orientation[3:], dtype=float)
        dz = self._slice_dz

        inplane = self._inplane_spacing()
        self.spacing_known = inplane is not None
        if inplane is None:
            inplane = [1.0, 1.0]

        if self._frame_origin is not None:
            origin = np.asarray(self._frame_origin, dtype=float)
        elif 'ImagePositionPatient' in self.ds:
            origin = np.asarray(self.ds.ImagePositionPatient, dtype=float)
        else:
            origin = np.zeros(3)

        # patient direction, step size and length of each stored array axis
        dirs = np.stack([np.cross(row, col) * np.sign(dz), col, row])
        steps = np.array([abs(dz), float(inplane[0]), float(inplane[1])])
        shape = np.array([self.number_of_frames, int(self.ds.Rows), int(self.ds.Columns)])

        # permute so array axes map to patient z, y, x
        dominant = np.argmax(np.abs(dirs), axis=1)
        if len(set(dominant)) == 3:
            perm = [int(np.where(dominant == a)[0][0]) for a in (2, 1, 0)]
        else:
            perm = [0, 1, 2]

        dirs, steps, shape = dirs[perm], steps[perm], shape[perm]
        if self.array is not None:
            self.array = self.array.transpose(perm)

        # flip any axis that runs against its patient axis
        for k, a in enumerate((2, 1, 0)):
            if dirs[k, a] < 0:
                origin = origin + (shape[k] - 1) * steps[k] * dirs[k]
                dirs[k] = -dirs[k]
                if self.array is not None:
                    self.array = np.flip(self.array, axis=k)

        if self.array is not None:
            self.array = np.ascontiguousarray(self.array)

        self.origin = origin
        self.spacing = steps[::-1].copy()
        self.dimensions = shape
        self.orientation = np.concatenate([dirs[2], dirs[1]])
        self.image_matrix = np.stack([dirs[2], dirs[1], dirs[0]]).astype(np.float32)

    def compute_plane(self):
        """Acquisition plane from the stored slice normal."""
        normal = np.cross(self.orientation[:3], self.orientation[3:])

        return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(normal)))]

    def initial_orientation(self):
        ds = self.ds

        iop = ds.get('ImageOrientationPatient')
        if iop is None:
            per = ds.get('PerFrameFunctionalGroupsSequence')
            iop = self._fg_value(per[0] if per else None, self._shared_group(),
                                 'PlaneOrientationSequence', 'ImageOrientationPatient')
        if iop is not None and len(iop) == 6:
            return [float(v) for v in iop]

        letters = ds.get('PatientOrientation')
        if not letters or len(letters) < 2:
            letters = self._orientation_from_view()

        if letters:
            row = self._letters_to_dir(letters[0])
            col = self._letters_to_dir(letters[1])
            if row is not None and col is not None:
                col = col - np.dot(col, row) * row  # make orthogonal
                norm = np.linalg.norm(col)
                if norm > 1e-6:
                    return [*row, *(col / norm)]

        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class ReadRF:
    """
    Reads and constructs Radio Fluoroscopy (RF) DICOM images.

    This class converts RF DICOM data into a standardized internal representation,
    including:
    - Pixel data extraction (single-frame, multi-frame cine, grayscale or RGB)
    - Orientation inference
    - Spatial spacing computation
    - Integration into the global `Data` structure

    Notes
    -----
    - Arrays are stored in z-y-x order. For multi-frame (cine) data, the frame
      axis is placed on the through-plane axis for the inferred plane, so
      scrolling through slices steps through frames.
    - Frame spacing is nominal (1 mm). Real frame timing is kept in
      `frame_times` when the DICOM provides it.
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
        Pixel array in z-y-x order (with a trailing channel axis if RGB).
    spacing : np.ndarray
        Physical spacing (x, y, z) in mm.
    dimensions : tuple
        Spatial shape (z, y, x) of the image. Available even when only_tags=True.
    orientation : list
        Default orientation vector.
    frame_times : np.ndarray or None
        Per-frame time offsets in ms, if available.
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
        self.rgb = int(self.image_set[0].get('SamplesPerPixel', 1)) == 3
        self.dimensions = None

        self.modality = self.image_set[0].Modality
        self.filepaths = self.image_set[0].filename
        self.sops = self.image_set[0].SOPInstanceUID

        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.array([0, 0, 0], dtype=float)
        self.image_matrix = np.eye(3, dtype=np.float32)
        self.plane = self.compute_plane()
        self.frame_times = self.compute_frame_times()

        self.array = None
        if not self.only_tags:
            self.compute_array()
        else:
            self.dimensions = self.dimensions_from_tags()

        self.spacing = self.compute_spacing()
        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)

    @staticmethod
    def _functional_group_spacing(img):
        """
        Look for PixelSpacing in enhanced multi-frame functional groups,
        shared group first since that is where it usually lives.
        """
        for group_name in ('SharedFunctionalGroupsSequence',
                           'PerFrameFunctionalGroupsSequence'):
            if group_name not in img:
                continue

            group = img[group_name].value
            if not group:
                continue

            item = group[0]
            if 'PixelMeasuresSequence' in item:
                measures = item.PixelMeasuresSequence[0]
                if 'PixelSpacing' in measures:
                    return measures.PixelSpacing

        return None

    def _orientation_string(self):
        """
        Return PatientOrientation joined into one string, e.g. 'LF' or 'LPFL'.
        """
        orient = self.image_set[0].get('PatientOrientation', None)
        if not orient:
            return ''

        if isinstance(orient, str):
            return orient.replace('\\', '')

        return ''.join(str(o) for o in orient)

    def _rows_toward_feet(self):
        """
        True if the column direction (second PatientOrientation value)
        points toward the feet, i.e. increasing row index = inferior.
        """
        orient = self.image_set[0].get('PatientOrientation', None)
        if not orient or isinstance(orient, str) or len(orient) < 2:
            return False

        return 'F' in str(orient[1])

    def _target_dtype(self):
        """
        int16 unless the data is unsigned with more than 15 bits stored,
        in which case int16 would wrap and int32 is used instead.
        """
        ds = self.image_set[0]
        unsigned = int(ds.get('PixelRepresentation', 0)) == 0
        bits_stored = int(ds.get('BitsStored', 16))
        if unsigned and bits_stored > 15:
            return np.int32

        return np.int16

    def compute_array(self):
        """
        Load RF pixel data into a z-y-x NumPy array.

        Steps
        -----
        - Normalize to (frames, rows, cols[, 3])
        - Cast grayscale data to a safe integer type
        - Move the frame axis onto the through-plane axis for the plane
        - Optionally flip so +z points superior
        - Remove raw PixelData for memory efficiency
        - Store spatial shape in `self.dimensions`
        """
        ds = self.image_set[0]
        arr = ds.pixel_array
        n_frames = int(ds.get('NumberOfFrames', 1) or 1)

        # --- normalize to (frames, rows, cols[, 3]) ---
        if n_frames == 1:
            arr = arr[np.newaxis]

        if not self.rgb:
            arr = arr.astype(self._target_dtype())

        # --- place frame axis on the through-plane axis (z-y-x) ---
        if self.plane == 'Coronal':
            # rows=z, cols=x -> (z, frames, x)
            arr = np.swapaxes(arr, 0, 1)

        elif self.plane == 'Sagittal':
            # rows=z, cols=y -> (z, y, frames)
            arr = np.moveaxis(arr, 0, 2)

        # Axial: (frames, rows, cols) is already (z, y, x)

        # --- make +z superior if rows run toward the feet ---
        if(self.plane in ('Coronal', 'Sagittal')) and self._rows_toward_feet():
            arr = np.flip(arr, axis=0)

        self.array = np.ascontiguousarray(arr)
        self.dimensions = self.array.shape[:3]

        del ds.PixelData

    def compute_frame_times(self):
        """
        Per-frame time offsets in ms from FrameTimeVector or FrameTime.

        Returns
        -------
        np.ndarray or None
        """
        ds = self.image_set[0]
        n_frames = int(ds.get('NumberOfFrames', 1) or 1)
        if n_frames <= 1:
            return None

        if 'FrameTimeVector' in ds:
            return np.cumsum(np.array(ds.FrameTimeVector, dtype=float))

        if 'FrameTime' in ds:
            return np.arange(n_frames, dtype=float) * float(ds.FrameTime)

        return None

    def compute_plane(self):
        """
        Infer anatomical plane from PatientOrientation.

        The H/F letter marks the vertical axis: with L/R it is coronal,
        with A/P it is sagittal. No H/F means axial.

        Returns
        -------
        str
            'Axial', 'Coronal', or 'Sagittal'
        """
        orient = self._orientation_string()

        if not orient:
            # Frontal (AP/PA) projection is the typical fluoro default
            return 'Coronal'

        has_si = any(c in orient for c in 'HF')

        if has_si and any(c in orient for c in 'LR'):
            return 'Coronal'

        if has_si and any(c in orient for c in 'AP'):
            return 'Sagittal'

        return 'Axial'

    def compute_spacing(self):
        """
        Compute voxel spacing for RF imaging.

        Uses, in order:
        - PixelSpacing
        - ImagerPixelSpacing (corrected by EstimatedRadiographicMagnificationFactor)
        - Shared / per-frame functional group PixelMeasuresSequence
        - DetectorElementSpacing

        Returns
        -------
        np.ndarray
            Spacing in (x, y, z) order, matched to the z-y-x array layout.
        """
        img = self.image_set[0]

        inplane = [1.0, 1.0]
        slice_thickness = 1.0

        if 'PixelSpacing' in img:
            inplane = img.PixelSpacing

        elif 'ImagerPixelSpacing' in img:
            inplane = [float(v) for v in img.ImagerPixelSpacing]
            mag = img.get('EstimatedRadiographicMagnificationFactor', None)
            if mag:
                inplane = [v / float(mag) for v in inplane]

        else:
            found = self._functional_group_spacing(img)
            if found is not None:
                inplane = found

            elif 'ContributingSourcesSequence' in img:
                seq = img.ContributingSourcesSequence[0]
                if 'DetectorElementSpacing' in seq:
                    inplane = seq.DetectorElementSpacing

        row_sp, col_sp = float(inplane[0]), float(inplane[1])

        # --- reorder by plane (x, y, z) ---
        if self.plane == 'Axial':
            return np.array([col_sp, row_sp, slice_thickness], dtype=float)

        elif self.plane == 'Coronal':
            return np.array([col_sp, slice_thickness, row_sp], dtype=float)

        else:
            return np.array([slice_thickness, col_sp, row_sp], dtype=float)

    def dimensions_from_tags(self):
        """
        Compute the z-y-x spatial shape from header tags without decoding
        pixel data (used when only_tags=True).
        """
        ds = self.image_set[0]
        rows = int(ds.get('Rows', 0))
        cols = int(ds.get('Columns', 0))
        frames = int(ds.get('NumberOfFrames', 1) or 1)

        if self.plane == 'Coronal':
            return rows, frames, cols

        if self.plane == 'Sagittal':
            return rows, cols, frames

        return frames, rows, cols


class ReadUS:
    """
    Reads ultrasound DICOM into a standardized array and registers it in `Data`.

    Output array layout
    -------------------
    grayscale (default)   : (frames, rows, cols) uint8
    keep_color=True       : (frames, rows, cols, 3) uint8, self.rgb = True

    Parameters
    ----------
    image_set : pydicom.Dataset or list
        Ultrasound dataset(s). Only the first is read.
    only_tags : bool
        If True, metadata only.
    keep_color : bool
        Keep RGB instead of extracting the grayscale B-mode content.
    color_tolerance : int
        Max channel spread (0-255) for a pixel to count as gray. Lossy JPEG
        rarely gives exactly equal channels, so 0 is too strict.
    """

    def __init__(self, image_set, only_tags, keep_color=False, color_tolerance=3):
        self.image_set = image_set if isinstance(image_set, list) else [image_set]
        self.only_tags = only_tags
        self.keep_color = keep_color
        self.color_tolerance = color_tolerance

        ds = self.image_set[0]

        self.unverified = 'Modality'
        self.base_position = None
        self.skipped_slice = None
        self.rgb = False

        self.modality = ds.Modality
        self.filepaths = getattr(ds, 'filename', None)
        self.sops = ds.SOPInstanceUID

        self.plane = 'Axial'
        self.orientation = [1, 0, 0, 0, 1, 0]
        self.origin = np.zeros(3)
        self.image_matrix = np.eye(3, dtype=np.float32)

        self.samples_per_pixel = int(getattr(ds, 'SamplesPerPixel', 1))
        self.number_of_frames = int(getattr(ds, 'NumberOfFrames', 1) or 1)
        self.frame_time = float(ds.FrameTime) if 'FrameTime' in ds else None  # ms

        self.dimensions = np.array([self.number_of_frames, int(ds.Rows), int(ds.Columns)])

        self.array = None
        if not self.only_tags:
            self.compute_array()

        self.spacing_known = False
        self.spacing = self.compute_spacing()
        self.image_name = create_image_name(self.modality)

        image = Image(self)
        Data.image[self.image_name] = image
        Data.image_list.append(self.image_name)

    @staticmethod
    def _detector_spacing(ds):
        seq = getattr(ds, 'ContributingSourcesSequence', None)
        if seq and 'DetectorElementSpacing' in seq[0]:
            return [float(v) for v in seq[0].DetectorElementSpacing]

        return None

    @staticmethod
    def _pixel_spacing(ds):
        if 'PixelSpacing' in ds:
            return [float(v) for v in ds.PixelSpacing]

        for key in ('SharedFunctionalGroupsSequence', 'PerFrameFunctionalGroupsSequence'):
            groups = getattr(ds, key, None)
            if groups:
                pm = getattr(groups[0], 'PixelMeasuresSequence', None)
                if pm and 'PixelSpacing' in pm[0]:
                    return [float(v) for v in pm[0].PixelSpacing]

        return None

    @staticmethod
    def _region_spacing(ds):
        """
        First 2D region with cm units in both directions, converted to mm.

        SequenceOfUltrasoundRegions codes (PS3.3 C.8.5.5.1)
        _REGION_SPATIAL_2D = 1
        _UNITS_CM = 3

        """
        for r in getattr(ds, 'SequenceOfUltrasoundRegions', []):
            if (int(getattr(r, 'RegionSpatialFormat', -1)) == 1
                    and int(getattr(r, 'PhysicalUnitsXDirection', -1)) == 3
                    and int(getattr(r, 'PhysicalUnitsYDirection', -1)) == 3
                    and 'PhysicalDeltaX' in r and 'PhysicalDeltaY' in r):
                return [abs(float(r.PhysicalDeltaY)) * 10.0, abs(float(r.PhysicalDeltaX)) * 10.0]

        return None

    @staticmethod
    def _to_uint8(arr, bits):
        if arr.dtype == np.uint8:
            return arr
        shift = max(int(bits) - 8, 0)
        return np.clip(arr >> shift, 0, 255).astype(np.uint8)

    def compute_array(self):
        ds = self.image_set[0]
        arr = ds.pixel_array
        spp = self.samples_per_pixel
        photometric = str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')).upper()

        if photometric == 'PALETTE COLOR':
            arr = apply_color_lut(arr, ds)
            spp = 3
            bits = 8 * arr.dtype.itemsize  # LUT output depth, not BitsStored
        else:
            bits = int(getattr(ds, 'BitsStored', 8))

        # Decide layout from the tags, not from ndim: grayscale -> (F, R, C), color -> (F, R, C, S)
        frame_ndim = 2 if spp == 1 else 3
        if arr.ndim == frame_ndim:
            arr = arr[np.newaxis]

        if arr.ndim != frame_ndim + 1:
            raise ValueError(
                f"Unexpected US pixel array shape {arr.shape} "
                f"(SamplesPerPixel={spp}, NumberOfFrames={self.number_of_frames})"
            )

        arr = self._to_uint8(arr, bits)
        if spp == 1:
            self.array = arr

        elif self.keep_color:
            self.array = arr[..., :3]
            self.rgb = True

        else:
            rgb = arr[..., :3]
            spread = rgb.max(axis=-1) - rgb.min(axis=-1)  # uint8-safe, max >= min
            self.array = np.where(spread <= self.color_tolerance, rgb[..., 0], 0).astype(np.uint8)

        del ds.PixelData
        self.dimensions = np.array(self.array.shape[:3])

    def compute_spacing(self):
        """
        Returns (x, y, z). Priority: PixelSpacing / functional groups,
        then the 2D tissue region in cm, then DetectorElementSpacing.
        z is 1.0: US frames are time, not depth.
        """
        ds = self.image_set[0]
        inplane = self._pixel_spacing(ds) or self._region_spacing(ds) or self._detector_spacing(ds)

        self.spacing_known = inplane is not None
        if inplane is None:
            inplane = [1.0, 1.0]

        return np.array([inplane[1], inplane[0], 1.0])  # [row, col] -> (x, y, z)


class ReadRTStruct:
    """
    Reads RTSTRUCT datasets: closed-planar ROIs, POINT POIs, and the
    image series they belong to.

    Parameters
    ----------
    image_set : pydicom.Dataset
        RTSTRUCT dataset.
    only_tags : bool
        If True, parse metadata only (no coordinates).
    images : dict, optional
        name -> image object with `series_uid`, `sops`, and optionally
        `frame_of_reference_uid`. Defaults to the global `Data.image`.
    """

    def __init__(self, image_set, only_tags, images=None):
        self.image_set = image_set
        self.only_tags = only_tags
        self.filepaths = getattr(image_set, "filename", None)

        self.frame_of_reference_uid = self.get_frame_of_reference_uid()
        self.series_uid = self.get_referenced_series_uid()

        self._properties = self.get_properties()

        rois = [p for p in self._properties if p["type"] != "POINT"]
        pois = [p for p in self._properties if p["type"] == "POINT"]
        self.roi_names = [p["name"] for p in rois]
        self.roi_colors = [p["color"] for p in rois]

        self.roi_types = [p["type"] for p in rois]
        self.poi_names = [p["name"] for p in pois]
        self.poi_colors = [p["color"] for p in pois]

        # Always defined, even when the file has no usable structures
        self.contours = []
        self.points = []
        self.match_image_name = None

        if self.roi_names or self.poi_names:
            if images is None:
                images = Data.image
            self.match_image_name = self.match_with_image(images)
            if not self.only_tags:
                self.structure_positions()

    @staticmethod
    def _stable_color(name):
        """
        Deterministic fallback color derived from the ROI name.
        """
        h = hashlib.md5(name.encode("utf-8")).digest()

        return [int(h[0]), int(h[1]), int(h[2])]

    def get_frame_of_reference_uid(self):
        ds = self.image_set
        for roi in getattr(ds, "StructureSetROISequence", []):
            uid = getattr(roi, "ReferencedFrameOfReferenceUID", None)
            if uid:
                return str(uid)

        for ref in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
            uid = getattr(ref, "FrameOfReferenceUID", None)
            if uid:
                return str(uid)

        return None

    def get_properties(self):
        ds = self.image_set
        roi_by_number = {
            int(r.ROINumber): r
            for r in getattr(ds, "StructureSetROISequence", [])
            if hasattr(r, "ROINumber")
        }

        props = []
        for idx, rc in enumerate(getattr(ds, "ROIContourSequence", [])):
            if not hasattr(rc, "ReferencedROINumber"):
                continue
            roi = roi_by_number.get(int(rc.ReferencedROINumber))
            if roi is None or not hasattr(roi, "ROIName"):
                continue

            contours = [c for c in getattr(rc, "ContourSequence", [])
                        if hasattr(c, "ContourData") and len(c.ContourData) >= 3]
            if not contours:
                continue

            geom = str(contours[0].ContourGeometricType).upper()
            if geom not in ("CLOSED_PLANAR", "OPEN_PLANAR", "OPEN_NONPLANAR", "CLOSEDPLANAR_XOR", "POINT"):
                continue

            sops = []
            for c in contours:
                for img in getattr(c, "ContourImageSequence", []):
                    uid = getattr(img, "ReferencedSOPInstanceUID", None)
                    if uid:
                        sops.append(str(uid))

            name = str(roi.ROIName)
            if hasattr(rc, "ROIDisplayColor"):
                color = [int(v) for v in rc.ROIDisplayColor]
            else:
                color = self._stable_color(name)

            props.append({
                "index": idx,
                "number": int(rc.ReferencedROINumber),
                "name": name,
                "color": color,
                "type": geom,
                "sops": sops,  # may be empty; ContourImageSequence is optional
            })
        return props

    def get_referenced_series_uid(self):
        """
        Series UID of the referenced image series (not the RTSTRUCT's own).
        """
        for ref in getattr(self.image_set, "ReferencedFrameOfReferenceSequence", []):
            for study in getattr(ref, "RTReferencedStudySequence", []):
                for series in getattr(study, "RTReferencedSeriesSequence", []):
                    uid = getattr(series, "SeriesInstanceUID", None)
                    if uid:
                        return str(uid)

        return None

    def match_with_image(self, images):
        """
        Priority:
          1. referenced series UID (confirmed by SOP overlap when SOPs exist)
          2. any image containing referenced SOPs
          3. same Frame of Reference UID
        """
        all_sops = {s for p in self._properties for s in p["sops"]}

        sop_match = None
        for name, img in images.items():
            img_sops = set(getattr(img, "sops", []))
            overlap = bool(all_sops & img_sops)

            if self.series_uid and self.series_uid == getattr(img, "series_uid", None):
                if not all_sops or overlap:
                    return name
            if sop_match is None and overlap:
                sop_match = name

        if sop_match is not None:
            return sop_match

        if self.frame_of_reference_uid:
            for name, img in images.items():
                if getattr(img, "frame_of_reference_uid", None) == self.frame_of_reference_uid:
                    return name

        return None

    def structure_positions(self):
        """
        contours : one list per ROI (aligned with roi_names / roi_types),
                   each an (N, 3) array per contour item
        points   : one (1, 3) array per POI, aligned with poi_names
        """
        seqs = self.image_set.ROIContourSequence
        for p in self._properties:
            arrays = []
            for c in seqs[p["index"]].ContourSequence:
                if not hasattr(c, "ContourData"):
                    continue
                if str(c.ContourGeometricType).upper() != p["type"]:
                    continue
                arrays.append(np.asarray(c.ContourData, dtype=float).reshape(-1, 3))

            if p["type"] == "POINT":
                self.points.append(arrays[0])
            else:
                self.contours.append(arrays)


class ReadREG:
    """
    Reads a DICOM Spatial Registration (rigid) or Deformable Spatial Registration.

    Reference vs. moving is resolved per registration item:
    1. Distinct FoRs -> item matching the REG FrameOfReferenceUID is the reference.
    2. Shared FoR    -> identity-transform item is the reference.
    3. Ambiguous     -> first item is the reference.

    moving_matrix is reference -> moving (for pull-resampling the moving image).
    dvf is (z, y, x, 3); dimensions are (z, y, x); spacing/origin are (x, y, z).
    """

    def __init__(self, image_set, only_tags=False):
        self.ds = image_set[0] if isinstance(image_set, (list, tuple)) else image_set
        self.only_tags = only_tags
        self.deformable = 'DeformableRegistrationSequence' in self.ds

        self.reference_name, self.moving_name = None, None
        self.reference_series, self.moving_series = None, None
        self.reference_sops, self.moving_sops = [], []
        self._ref_item, self._mov_item = None, None

        self.spacing = None
        self.dimensions = None
        self.origin = None

        self.reference_matrix = None
        self.moving_matrix = None
        self.post_matrix = None
        self.dvf_matrix = None
        self.dvf = None

        self.registration_name = None
        self.registration = None

        self.order_items()
        self.resolve_series()
        self.create_name()

        if only_tags:
            return

        if self.deformable:
            self.compute_deformable()

        else:
            self.compute_rigid()

        self.create_registration()

    @staticmethod
    def _deform_matrix(self, item, keyword):
        if item is None or keyword not in item:
            return np.eye(4)

        return np.asarray(item[keyword][0].FrameOfReferenceTransformationMatrix, dtype=float).reshape(4, 4)

    def _is_identity_item(self, item):
        if self.deformable:
            return ('DeformableRegistrationGridSequence' not in item
                    and np.allclose(self._deform_matrix(item, 'PreDeformationMatrixRegistrationSequence'), np.eye(4))
                    and np.allclose(self._deform_matrix(item, 'PostDeformationMatrixRegistrationSequence'), np.eye(4)))

        return np.allclose(self._rigid_matrix(item), np.eye(4))

    @staticmethod
    def _item_sops(item):
        if item is None:
            return []

        return [ref.ReferencedSOPInstanceUID for ref in item.get('ReferencedImageSequence', [])]

    def _rigid_matrix(self, item):
        if item is None or 'MatrixRegistrationSequence' not in item:
            return np.eye(4)

        m = np.eye(4)
        for mat in item.MatrixRegistrationSequence[0].MatrixSequence:
            m = self._to_4x4(mat.FrameOfReferenceTransformationMatrix) @ m
            m = np.asarray(mat.FrameOfReferenceTransformationMatrix, dtype=float).reshape(4, 4) @ m

        return m

    @staticmethod
    def _to_4x4(values):
        return np.asarray(values, dtype=float).reshape(4, 4)

    def compute_rigid(self):
        self.reference_matrix = self._rigid_matrix(self._ref_item)
        moving_to_ref = np.linalg.inv(self.reference_matrix) @ self._rigid_matrix(self._mov_item)
        self.moving_matrix = np.linalg.inv(moving_to_ref)

    def compute_deformable(self):
        item = self._mov_item
        pre = self._deform_matrix(item, 'PreDeformationMatrixRegistrationSequence')
        self.post_matrix = self._deform_matrix(item, 'PostDeformationMatrixRegistrationSequence')
        self.moving_matrix = np.linalg.inv(pre)

        grid = item.DeformableRegistrationGridSequence[0]

        row = np.asarray(grid.ImageOrientationPatient[:3], dtype=np.float32)
        col = np.asarray(grid.ImageOrientationPatient[3:], dtype=np.float32)
        self.dvf_matrix = np.stack([row, col, np.cross(row, col)])

        self.origin = np.asarray(grid.ImagePositionPatient, dtype=float)
        self.dimensions = np.flip(np.asarray(grid.GridDimensions, dtype=int))
        self.spacing = np.asarray(grid.GridResolution, dtype=float)

        # read-only view of the buffer; add .copy() if Deformable writes into it
        self.dvf = np.frombuffer(grid.VectorGridData, dtype='<f4').reshape(*self.dimensions, 3)

        del grid.VectorGridData

    def create_name(self):
        ref_first = self.reference_sops[0] if self.reference_sops else None
        mov_first = self.moving_sops[0] if self.moving_sops else None

        for image_name in Data.image_list:
            sops = Data.image[image_name].sops
            if self.reference_name is None and ref_first in sops:
                self.reference_name = image_name

            elif self.moving_name is None and mov_first in sops:
                self.moving_name = image_name

        prefix = 'DVF_' if self.deformable else ''
        base = f"{prefix}{self.reference_name or 'Unknown'}_{self.moving_name or 'Unknown'}"

        registry = Data.deformable_list if self.deformable else Data.rigid_list
        name, i = base, 1
        while name in registry:
            name = f'{base}_{i}'
            i += 1

        self.registration_name = name

    def create_registration(self):
        if not (self.reference_name and self.moving_name):
            return

        if self.deformable:
            self.registration = Deformable(
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
                moving_sops=self.moving_sops,
            )
        else:
            self.registration = Rigid(
                self.reference_name,
                self.moving_name,
                rigid_name=self.registration_name,
                reference_sops=self.reference_sops,
                moving_sops=self.moving_sops,
                reference_matrix=self.reference_matrix,
                matrix=self.moving_matrix,
            )

    def order_items(self):
        if self.deformable:
            items, uid_attr = self.ds.DeformableRegistrationSequence, 'SourceFrameOfReferenceUID'
        else:
            items, uid_attr = self.ds.RegistrationSequence, 'FrameOfReferenceUID'

        if len(items) == 1:
            if self._is_identity_item(items[0]):
                self._ref_item, self._mov_item = items[0], None
            else:
                self._ref_item, self._mov_item = None, items[0]

            return

        a, b = items[0], items[1]
        reg_for = self.ds.get('FrameOfReferenceUID')
        a_for, b_for = a.get(uid_attr), b.get(uid_attr)

        if a_for != b_for and reg_for in (a_for, b_for):
            self._ref_item, self._mov_item = (a, b) if a_for == reg_for else (b, a)

        elif self._is_identity_item(b) and not self._is_identity_item(a):
            self._ref_item, self._mov_item = b, a

        else:
            self._ref_item, self._mov_item = a, b

    def resolve_series(self):
        series = {}
        seqs = list(self.ds.get('ReferencedSeriesSequence', []))
        for study in self.ds.get('StudiesContainingOtherReferencedInstancesSequence', []):
            seqs += list(study.get('ReferencedSeriesSequence', []))

        for s in seqs:
            series[s.SeriesInstanceUID] = [i.ReferencedSOPInstanceUID
                                           for i in s.get('ReferencedInstanceSequence', [])]

        def lookup(item):
            sops = self._item_sops(item)
            if sops:
                for uid, series_sops in series.items():
                    if sops[0] in series_sops:
                        return uid
            return None

        ref_uid, mov_uid = lookup(self._ref_item), lookup(self._mov_item)
        leftover = [u for u in series if u not in (ref_uid, mov_uid)]

        if ref_uid is None and mov_uid is None and len(leftover) == 2:
            ref_uid, mov_uid = leftover

        elif ref_uid is None and len(leftover) == 1:
            ref_uid = leftover[0]

        elif mov_uid is None and len(leftover) == 1:
            mov_uid = leftover[0]

        self.reference_series, self.moving_series = ref_uid, mov_uid
        self.reference_sops = series.get(ref_uid) or self._item_sops(self._ref_item)
        self.moving_sops = series.get(mov_uid) or self._item_sops(self._mov_item)


class ReadRTDose:
    """
    Reads an RTDOSE object into a canonical grid.

    After loading, the array is indexed [z, y, x] with every axis increasing in
    patient coordinates, and origin is the patient position of voxel [0, 0, 0].
    spacing and origin are (x, y, z); dimensions match array.shape (z, y, x).
    plane describes the grid as stored in the file, before reorientation.
    """

    def __init__(self, image_set, only_tags=False):
        self.ds = image_set[0] if isinstance(image_set, (list, tuple)) else image_set
        self.image_set = [self.ds]
        self.only_tags = only_tags
        self.modality = 'RTDOSE'

        self.unverified = None
        self.base_position = None
        self.skipped_slice = None

        self.filepaths = [self.ds.filename]
        self.sops = [self.ds.SOPInstanceUID]

        self.array = None
        self.origin = None
        self.spacing = None
        self.dimensions = None
        self.image_matrix = None

        if not only_tags:
            self.compute_array()

        self.orientation = self.compute_orientation()
        self.plane = self.compute_plane()
        self.compute_geometry()

        self.dose_name = create_dose_name(self.modality)
        Data.dose[self.dose_name] = Dose(self)
        Data.dose_list.append(self.dose_name)

    def _num_frames(self):
        return int(self.ds.get('NumberOfFrames') or 1)

    def _slice_step(self):
        """Signed distance between frames along the plane normal."""
        offsets = self.ds.get('GridFrameOffsetVector')
        if offsets is not None and not isinstance(offsets, (str, float)) and len(offsets) > 1:
            step = float(offsets[1]) - float(offsets[0])
            if step != 0:
                return step

        return float(self.ds.get('SliceThickness') or 1.0)

    def compute_array(self):
        scaling = float(self.ds.get('DoseGridScaling') or 1.0)
        shape = (self._num_frames(), int(self.ds.Rows), int(self.ds.Columns))
        self.array = (self.ds.pixel_array.astype(np.float32) * scaling).reshape(shape)

        del self.ds.PixelData

    def compute_geometry(self):
        """
        Reorders and flips the grid so array axes (0, 1, 2) run along +z, +y, +x.

        Works from tags alone, so it is valid with only_tags=True; the array
        is transformed alongside when it has been loaded.
        """
        row = self.orientation[:3]
        col = self.orientation[3:]
        dz = self._slice_step()

        pixel_spacing = self.ds.get('PixelSpacing') or [1.0, 1.0]
        origin = np.asarray(self.ds.ImagePositionPatient, dtype=float)

        # patient direction, step size and length of each stored array axis
        dirs = np.stack([np.cross(row, col) * np.sign(dz), col, row])
        steps = np.array([abs(dz), float(pixel_spacing[0]), float(pixel_spacing[1])])
        shape = np.array([self._num_frames(), int(self.ds.Rows), int(self.ds.Columns)])

        # permute so array axes map to patient z, y, x
        dominant = np.argmax(np.abs(dirs), axis=1)
        if len(set(dominant)) == 3:
            perm = [int(np.where(dominant == a)[0][0]) for a in (2, 1, 0)]
        else:
            perm = [0, 1, 2]

        dirs, steps, shape = dirs[perm], steps[perm], shape[perm]
        if self.array is not None:
            self.array = self.array.transpose(perm)

        # flip any axis that runs against its patient axis
        for k, a in enumerate((2, 1, 0)):
            if dirs[k, a] < 0:
                origin = origin + (shape[k] - 1) * steps[k] * dirs[k]
                dirs[k] = -dirs[k]
                if self.array is not None:
                    self.array = np.flip(self.array, axis=k)

        if self.array is not None:
            self.array = np.ascontiguousarray(self.array)

        self.origin = origin
        self.spacing = steps[::-1].copy()
        self.dimensions = shape
        self.orientation = np.concatenate([dirs[2], dirs[1]])
        self.image_matrix = np.stack([dirs[2], dirs[1], dirs[0]]).astype(np.float32)

    def compute_orientation(self):
        if 'ImageOrientationPatient' in self.ds:
            return np.asarray(self.ds.ImageOrientationPatient, dtype=float)
        self.unverified = 'Orientation'
        return np.array([1, 0, 0, 0, 1, 0], dtype=float)

    def compute_plane(self):
        normal = np.cross(self.orientation[:3], self.orientation[3:])
        return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(normal)))]


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
