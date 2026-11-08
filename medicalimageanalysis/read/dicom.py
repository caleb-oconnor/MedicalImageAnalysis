"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

DICOM Pipeline Orchestrator and Volumetric Reconstructor
========================================================

Description:
    This module is the ingestion engine for DICOM data. It takes raw
    pydicom datasets and turns them into physically consistent images,
    structures, registrations and dose grids inside the global `Data`
    registry.

Supported Objects:
    * CT / MR / PT          -> `Read3D` (classic and enhanced multi-frame)
    * DX / CR / Tomo        -> `ReadXRay`
    * RF (fluoroscopy)      -> `ReadRF`
    * US                    -> `ReadUS`
    * RTSTRUCT              -> `ReadRTStruct` (attached to its image)
    * REG (rigid/deformable)-> `ReadREG` (`Data.rigid` / `Data.deformable`)
    * RTDOSE                -> `ReadRTDose` (`Data.dose`)

Key Functionalities:
    1. **Orchestration**: `DicomReader` reads files on a thread pool, filters by
       SOP Class, and groups slices into volumes by SeriesInstanceUID,
       orientation and acquisition number.
    2. **Volume Reconstruction**: `Read3D` sorts slices along the normal,
       drops duplicates, rescales, fills missing slices by interpolation,
       and reorients to a canonical (z, y, x) array.
    3. **Projection / 2D Modalities**: `ReadXRay`, `ReadRF` and `ReadUS`
       normalize non-volumetric data into the same (z, y, x) layout.
    4. **Chronological Ordering**: `sort_images_by_datetime` orders the
       loaded images by their best available acquisition timestamp.

Geometric Conventions:
    - Arrays are (z, y, x), each axis increasing along patient +z, +y, +x
      (LPS).
    - `origin` is the patient position of `array[0, 0, 0]`.
    - `spacing` and `origin` are (x, y, z); `dimensions` is (z, y, x).
    - `image_matrix` rows are the patient directions of the array's x, y
      and z axes (identity for any orthogonal acquisition).

Memory Management:
    Pixel buffers (`PixelData`) are deleted from each dataset right after
    decoding, so only the NumPy volume stays in RAM.

Architecture Note:
    Reader classes do their work in `__init__` and register the result in
    `Data` themselves; callers normally only use `DicomReader` (or the
    `read_dicoms` wrapper) and then read from `Data`.

Usage:
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import DicomReader
    >>> files = {'Dicom': ['C:/Data/ct_001.dcm', 'C:/Data/ct_002.dcm']}
    >>> reader = DicomReader(files, only_tags=False, only_modality=None,
    ...                      only_load_roi_names=None, clear=True)
    >>> reader.load()
    >>> Data.image_list()

Examples
--------
>>> import medicalimageanalysis as mia
>>> from medicalimageanalysis import Data
>>> mia.read_dicoms(folder_path='C:/Data/Patient01')
>>> Data.image_list()
['CT 01', 'MR 02']
>>> ct = Data.image['CT 01']
>>> ct.array.shape, ct.spacing
((142, 512, 512), array([0.977, 0.977, 2.5  ]))
>>> Data.rigid_list(), Data.dose_list()
(['CT 01_MR 02'], ['RTDOSE 01'])
"""

import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
import itertools
from datetime import datetime

import numpy as np
import pydicom as dicom
from pydicom.pixels import apply_color_lut

from ..structure.deformable import Deformable
from ..structure.dose import Dose
from ..structure.image import Image
from ..structure.rigid import Rigid

from ..data import Data


def get_modality(series_sop_class):
    """
    Map a DICOM SOP Class UID to an internal modality label.

    Covers the image, RT, registration, segmentation, report and document
    SOP classes. Enhanced (multi-frame) variants of CT, MR, US, XA, RF, PET
    and RT Image are flagged so callers can treat them differently.

    Note that PET maps to 'PET' (not the DICOM Modality value 'PT').

    Parameters
    ----------
    series_sop_class : str
        SOP Class UID from DICOM tag (0008,0016).

    Returns
    -------
    tuple[str | None, bool | None]
        Internal modality name and whether it is an enhanced variant.
        (None, None) for unsupported UIDs.

    Examples
    --------
    Classic and enhanced CT:

    >>> from medicalimageanalysis.read.dicom import get_modality
    >>> get_modality('1.2.840.10008.5.1.4.1.1.2')
    ('CT', False)
    >>> get_modality('1.2.840.10008.5.1.4.1.1.2.1')
    ('CT', True)

    RT objects:

    >>> get_modality('1.2.840.10008.5.1.4.1.1.481.3')
    ('RTSTRUCT', False)
    >>> get_modality('1.2.840.10008.5.1.4.1.1.481.2')
    ('RTDOSE', False)

    Read the UID straight from a dataset:

    >>> import pydicom
    >>> ds = pydicom.dcmread('C:/Data/ct_001.dcm', stop_before_pixels=True)
    >>> get_modality(ds[0x0008, 0x0016].value)
    ('CT', False)

    Unknown UIDs return a pair of Nones:

    >>> get_modality('1.2.3.4')
    (None, None)
    """
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


def _parse_dicom_date(value):
    """
    Return a `datetime.date` parsed from a DICOM DA-like value.

    Non-digit characters are ignored, so both '20240315' and the legacy
    '2024.03.15' form parse. Values with fewer than 8 digits, or invalid
    dates, return None.

    Examples
    --------
    >>> from medicalimageanalysis.read.dicom import _parse_dicom_date
    >>> _parse_dicom_date('20240315')
    datetime.date(2024, 3, 15)
    >>> _parse_dicom_date('2024.03.15')
    datetime.date(2024, 3, 15)
    >>> _parse_dicom_date('2024') is None, _parse_dicom_date('20241345') is None
    (True, True)
    """
    if value is None:
        return None

    digits = ''.join(ch for ch in str(value).strip() if ch.isdigit())
    if len(digits) < 8:
        return None

    try:
        return datetime.strptime(digits[:8], '%Y%m%d').date()
    except ValueError:
        return None


def _parse_dicom_time(value):
    """
    Return (hour, minute, second, microsecond) from a DICOM TM-like value.

    Minutes and seconds may be omitted (DICOM allows 'HH' and 'HHMM'). The
    fraction is padded/truncated to microseconds. Out-of-range values return
    None.

    Examples
    --------
    >>> from medicalimageanalysis.read.dicom import _parse_dicom_time
    >>> _parse_dicom_time('143015.25')
    (14, 30, 15, 250000)
    >>> _parse_dicom_time('1430')
    (14, 30, 0, 0)
    >>> _parse_dicom_time('25') is None, _parse_dicom_time('') is None
    (True, True)
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    main, _, frac = raw.partition('.')
    main = ''.join(ch for ch in main if ch.isdigit())
    frac = ''.join(ch for ch in frac if ch.isdigit())[:6]

    if len(main) < 2:
        return None

    hour = int(main[:2])
    minute = int(main[2:4]) if len(main) >= 4 else 0
    second = int(main[4:6]) if len(main) >= 6 else 0
    microsecond = int(frac.ljust(6, '0')) if frac else 0

    if hour > 23 or minute > 59 or second > 59:
        return None

    return hour, minute, second, microsecond


def _parse_dicom_datetime(date_value=None, time_value=None, datetime_value=None):
    """
    Return a Python `datetime` parsed from DICOM date/time metadata.

    A DT value (`datetime_value`) takes priority over separate DA/TM values.
    A valid date with a missing or invalid time gives midnight.

    Examples
    --------
    >>> from medicalimageanalysis.read.dicom import _parse_dicom_datetime
    >>> _parse_dicom_datetime('20240315', '143015')
    datetime.datetime(2024, 3, 15, 14, 30, 15)
    >>> _parse_dicom_datetime(datetime_value='20240315143015.5')
    datetime.datetime(2024, 3, 15, 14, 30, 15, 500000)
    >>> _parse_dicom_datetime('20240315')
    datetime.datetime(2024, 3, 15, 0, 0)
    >>> _parse_dicom_datetime(None, '143015') is None
    True
    """
    if datetime_value is not None:
        raw = ''.join(ch for ch in str(datetime_value).strip() if ch.isdigit() or ch == '.')
        if len(raw) >= 8:
            date_value = raw[:8]
            time_value = raw[8:]

    date_part = _parse_dicom_date(date_value)
    if date_part is None:
        return None

    time_part = _parse_dicom_time(time_value)
    if time_part is None:
        return datetime.combine(date_part, datetime.min.time())

    return datetime(
        date_part.year,
        date_part.month,
        date_part.day,
        time_part[0],
        time_part[1],
        time_part[2],
        time_part[3],
    )


def _image_datetime(image):
    """
    Best-available timestamp of a loaded image for chronological ordering.

    Tries AcquisitionDateTime, then acquisition, series, content, study and
    instance-creation date/time from the image's first tag set. Returns None
    when nothing parses.

    Examples
    --------
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import _image_datetime
    >>> _image_datetime(Data.image['CT 01'])
    datetime.datetime(2024, 3, 15, 14, 30, 15)
    """
    tags = image.tags[0] if getattr(image, 'tags', None) else None
    if tags is None:
        return None

    candidates = [
        (None, None, tags.get('AcquisitionDateTime')),
        (tags.get('AcquisitionDate'), tags.get('AcquisitionTime'), None),
        (tags.get('SeriesDate'), tags.get('SeriesTime'), None),
        (tags.get('ContentDate'), tags.get('ContentTime'), None),
        (tags.get('StudyDate'), tags.get('StudyTime'), None),
        (tags.get('InstanceCreationDate'), tags.get('InstanceCreationTime'), None),
    ]

    for date_value, time_value, datetime_value in candidates:
        parsed = _parse_dicom_datetime(date_value, time_value, datetime_value)
        if parsed is not None:
            return parsed

    return None


def sort_images_by_datetime():
    """
    Reorder `Data.image` chronologically by DICOM date/time metadata.

    All loaded images are reordered by their best available timestamp,
    regardless of import order, using this priority: AcquisitionDateTime ->
    acquisition -> series -> content -> study -> instance creation. Images
    without a usable timestamp go last. Image names are not changed, only the
    order of `Data.image`.

    Returns
    -------
    None

    Examples
    --------
    A follow-up loaded before the baseline is moved behind it:

    >>> import medicalimageanalysis as mia
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import sort_images_by_datetime
    >>> mia.read_dicoms(folder_path='C:/Data/Patient01_followup')
    >>> mia.read_dicoms(folder_path='C:/Data/Patient01_baseline', clear=False)
    >>> sort_images_by_datetime()
    >>> Data.image_list()
    ['CT 02', 'CT 01']

    `DicomReader.load` already calls this, so it is only needed after
    adding images by other means (e.g. MHD):

    >>> mia.read_mhd(file='C:/Data/cbct.mhd', modality='CT')
    >>> sort_images_by_datetime()
    """
    ordered_names = sorted(
        Data.image_list(),
        key=lambda name: (
            _image_datetime(Data.image[name]) is None,
            _image_datetime(Data.image[name]) or datetime.min,
        ),
    )

    Data.image = {key: Data.image[key] for key in ordered_names}


def thread_process_dicom(path, stop_before_pixels=False):
    """
    Read one DICOM file, returning an empty list instead of raising.

    Used as the worker for the threaded read in `DicomReader.read`, where a
    single unreadable file should not stop the whole load.

    Parameters
    ----------
    path : str
        Path to the DICOM file.

    stop_before_pixels : bool, optional
        If True, only metadata is loaded (no pixel data).

    Returns
    -------
    pydicom.dataset.FileDataset or list
        Parsed dataset, or [] if the file could not be read.

    Examples
    --------
    Header-only read:

    >>> from medicalimageanalysis.read.dicom import thread_process_dicom
    >>> ds = thread_process_dicom('C:/Data/ct_001.dcm', stop_before_pixels=True)
    >>> ds.Modality, 'PixelData' in ds
    ('CT', False)

    A non-DICOM file gives an empty list:

    >>> thread_process_dicom('C:/Data/notes.txt')
    []
    """
    try:
        datasets = dicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)
    except Exception:
        datasets = []

    return datasets


class DicomReader(object):
    """
    Main DICOM pipeline: read, group, reconstruct and register.

    Reads files on a bounded thread pool, keeps the SOP classes listed in
    `only_modality`, groups slices into volumes and hands each group to the
    matching reader class. Images are created first so RTSTRUCT, REG and
    RTDOSE can be matched to them, then everything is sorted by date.

    Parameters
    ----------
    files : dict
        Categorized file paths with a 'Dicom' key (output of `file_parser`).

    only_tags : bool
        If True, load metadata only (no pixel arrays). Geometry is still
        computed.

    only_modality : list of str or None
        Modalities to process. None means ['CT', 'MR', 'PT', 'US', 'DX',
        'RF', 'TOMOSYNTHESIS', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE'].

    only_load_roi_names : list of str or None
        ROI names to keep from RTSTRUCT files.

    clear : bool
        If True, clears the global `Data` registry before loading.

    Attributes
    ----------
    ds : list
        Every dataset read (failed reads appear as empty lists).

    ds_modality : dict
        Modality -> list of datasets (2D/RT) or list of slice groups (3D).

    Examples
    --------
    Load a patient through the reader directly:

    >>> import medicalimageanalysis as mia
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import DicomReader
    >>> files = mia.file_parser(folder_path='C:/Data/Patient01')
    >>> reader = DicomReader(files, only_tags=False, only_modality=None,
    ...                      only_load_roi_names=None, clear=True)
    >>> reader.load()
    >>> Data.image_list()
    ['CT 01', 'MR 02']

    Inspect the grouping after a load:

    >>> {key: len(groups) for key, groups in reader.ds_modality.items() if groups}
    {'CT': 1, 'MR': 1, 'RTSTRUCT': 1, 'REG': 1}

    The `read_dicoms` wrapper builds the file dictionary for you:

    >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
    """

    def __init__(self, files, only_tags, only_modality, only_load_roi_names, clear):
        """
        Store settings, optionally clear `Data`, and prepare modality buckets.

        Parameters
        ----------
        files : dict
            Categorized file paths with a 'Dicom' key.

        only_tags : bool
            Metadata-only flag.

        only_modality : list of str or None
            Modalities to keep (None for the default set).

        only_load_roi_names : list of str or None
            ROI names to keep from RTSTRUCT files.

        clear : bool
            Clear the `Data` registry now.

        Examples
        --------
        One empty bucket is created per requested modality:

        >>> from medicalimageanalysis.read.dicom import DicomReader
        >>> reader = DicomReader({'Dicom': []}, only_tags=True,
        ...                      only_modality=['CT', 'RTSTRUCT'],
        ...                      only_load_roi_names=None, clear=False)
        >>> reader.ds_modality
        {'CT': [], 'RTSTRUCT': []}
        """
        self.files = files
        self.only_tags = only_tags
        self.only_load_roi_names = only_load_roi_names

        self.only_modality = (
            only_modality
            if only_modality is not None
            else ['CT', 'MR', 'PT', 'US', 'DX', 'RF', 'TOMOSYNTHESIS', 'CR', 'RTSTRUCT', 'REG', 'RTDOSE']
        )

        if clear:
            Data.clear()

        self.ds = []
        self.ds_modality = {key: [] for key in self.only_modality}

    def load(self, display_time=False):
        """
        Run the full pipeline: read -> group -> create -> sort by date.

        Parameters
        ----------
        display_time : bool
            If True, print the total runtime.

        Returns
        -------
        None

        Examples
        --------
        Load a folder and time it:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> from medicalimageanalysis.read.dicom import DicomReader
        >>> files = mia.file_parser(folder_path='C:/Data/Patient01')
        >>> reader = DicomReader(files, False, None, None, True)
        >>> reader.load(display_time=True)
        Dicom Read Time: 3.41
        >>> Data.image_list()
        ['CT 01', 'MR 02']
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
        Read every file in `files['Dicom']` into `self.ds` with a thread pool.

        Uses a bounded `ThreadPoolExecutor` (Python's default worker count,
        min(32, CPU count + 4)), so a study with thousands of files does not
        start thousands of threads. Unreadable files are stored as empty lists
        and skipped later. `self.ds` keeps the input file order.

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import DicomReader
        >>> reader = DicomReader({'Dicom': ['C:/Data/ct_001.dcm', 'C:/Data/ct_002.dcm']},
        ...                      True, ['CT'], None, False)
        >>> reader.read()
        >>> len(reader.ds), reader.ds[0].Modality
        (2, 'CT')

        Results line up with the input paths:

        >>> reader.ds[1].filename
        'C:/Data/ct_002.dcm'
        """
        with ThreadPoolExecutor() as executor:
            self.ds.extend(executor.map(
                lambda file_path: thread_process_dicom(file_path, stop_before_pixels=self.only_tags),
                self.files['Dicom'],
            ))

    def separate_modalities_and_images(self):
        """
        Group the read datasets into `self.ds_modality`.

        - US, DX, RF, CR, TOMOSYNTHESIS, RTSTRUCT, REG and RTDOSE: one entry per
          dataset.
        - CT, MR, PT: one entry per volume. Slices are split by
          SeriesInstanceUID, then orientation (rounded to 3 decimals), then
          AcquisitionNumber, and sorted along the acquisition plane. Acquisitions
          that tile without overlapping and with equal gaps are merged back into
          one volume; otherwise each stays its own entry.

        Slices missing ImageOrientationPatient or ImagePositionPatient are
        dropped.

        Returns
        -------
        None

        Examples
        --------
        A CT series and a two-orientation MR series become three volumes:

        >>> from medicalimageanalysis.read.dicom import DicomReader
        >>> reader = DicomReader(files, False, ['CT', 'MR'], None, True)
        >>> reader.read()
        >>> reader.separate_modalities_and_images()
        >>> len(reader.ds_modality['CT']), len(reader.ds_modality['MR'])
        (1, 2)
        >>> len(reader.ds_modality['CT'][0])
        142
        """
        for modality in list(self.ds_modality.keys()):
            images_in_modality = [d for d in self.ds if (0x0008, 0x0016) in d
                                  if get_modality(d[0x0008, 0x0016].value)[0] == modality]

            if len(images_in_modality) > 0 and modality in self.only_modality:
                if modality in ['US', 'DX', 'RF', 'CR', 'TOMOSYNTHESIS', 'RTSTRUCT', 'REG', 'RTDOSE']:
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
        Turn the grouped datasets into registered objects.

        Order matters so the RT objects can find their images:

        1. CT/MR/PT -> `Read3D`; DX/CR/TOMOSYNTHESIS -> `ReadXRay`;
           RF -> `ReadRF`; US -> `ReadUS`
        2. RTSTRUCT -> `ReadRTStruct`, attached via `Image.input_rtstruct`
           (a message is printed when no matching image is found)
        3. REG -> `ReadREG`
        4. RTDOSE -> `ReadRTDose`

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> from medicalimageanalysis.read.dicom import DicomReader
        >>> reader = DicomReader(files, False, None, None, True)
        >>> reader.read()
        >>> reader.separate_modalities_and_images()
        >>> reader.image_creation()
        >>> Data.image_list(), Data.dose_list()
        (['CT 01', 'MR 02'], ['RTDOSE 01'])
        """

        for modality in self.only_modality:
            for image_set in self.ds_modality[modality]:
                if modality in ['CT', 'MR', 'PT']:
                    Read3D(image_set, self.only_tags)

                elif modality in ['DX', 'CR', 'TOMOSYNTHESIS']:
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
    Reconstruct a 3D CT / MR / PT volume from DICOM and register it.

    Handles classic single-frame series and enhanced multi-frame objects.
    The volume is:

    - sorted along the slice normal, with duplicate positions removed
    - rescaled per slice (RescaleSlope / RescaleIntercept)
    - gap-filled by linear interpolation where slices are missing
    - reoriented so the array is (z, y, x) with each axis increasing along
      patient +z, +y, +x
    - registered in `Data.image` under a generated name

    Geometry problems never stop the load; they are recorded in
    `unverified` / `unverified_reasons` ('Orientation', 'Spacing',
    'Position', 'Duplicate', 'Irregular', 'Skipped').

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        Classic slices of one volume, or one or more enhanced datasets.

    only_tags : bool
        If True, only geometry/metadata is computed (pixels not decoded).

    Attributes
    ----------
    array : np.ndarray or None
        Volume in (z, y, x). int16 when every rescale value is an integer and
        the data fits, int32 if it overflows, float32 for non-integer rescale
        (e.g. PT in Bq/ml).

    spacing : np.ndarray
        Voxel spacing [x, y, z] in mm.

    dimensions : np.ndarray
        Volume shape [z, y, x] (valid with only_tags=True).

    origin : np.ndarray
        Patient-space position of array[0, 0, 0].

    image_matrix : np.ndarray
        3x3, rows are the patient directions of the array's x, y, z axes.
        Identity for orthogonal acquisitions, residual rotation for obliques.

    orientation : np.ndarray
        Direction cosines of the reoriented array (row = x, column = y).

    acquisition_orientation : np.ndarray
        ImageOrientationPatient as acquired (orthonormalized).

    plane : str
        Acquisition plane ('Axial', 'Coronal', 'Sagittal').

    enhanced : bool
        True if the input contained enhanced multi-frame DICOM.

    skipped_slice : list[int]
        Index (in the sorted real slices) of the slice following each gap.

    missing_slices : list[dict]
        Per gap: insert_index into the stacked volume, num_missing, and the
        labels of the bounding slices.

    unverified : str or None
        First geometry problem found; all are in `unverified_reasons`.

    frame_datasets, frame_numbers, frame_positions, frame_orientations,
    frame_pixel_spacings, frame_slopes, frame_intercepts, frame_thicknesses : list
        Per-frame values, sorted along the normal with duplicates removed.
        frame_numbers is None for classic slices, the 0-based frame for
        enhanced.

    duplicate_frames : list
        Labels of frames dropped for sharing a position with an earlier one.

    image_name : str
        Name the image was registered under.

    Examples
    --------
    Reconstruct a sorted CT series (normally done by `DicomReader`):

    >>> import pydicom
    >>> from glob import glob
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import Read3D
    >>> slices = [pydicom.dcmread(p) for p in glob('C:/Data/Patient01/CT/*.dcm')]
    >>> reader = Read3D(slices, only_tags=False)
    >>> reader.image_name, reader.plane, reader.array.dtype
    ('CT 01', 'Axial', dtype('int16'))
    >>> reader.array.shape, reader.spacing
    ((142, 512, 512), array([0.977, 0.977, 2.5  ]))
    >>> Data.image['CT 01'] is not None
    True

    A series with one missing slice is still loaded, and the gap is
    reported:

    >>> reader.unverified, reader.missing_slices
    ('Skipped', [{'insert_index': 57, 'num_missing': 1, 'between': ('1.2.3.57', '1.2.3.58')}])

    Headers only, e.g. for a series browser:

    >>> reader = Read3D(slices, only_tags=True)
    >>> reader.array is None, tuple(reader.dimensions)
    (True, (142, 512, 512))
    """
    def __init__(self, image_set, only_tags):
        """
        Compute geometry, optionally decode pixels, and register the image.

        Steps: collect per-frame tags -> common orientation -> plane -> slice
        layout (sort, dedupe, gaps) -> geometry -> pixels (unless only_tags) ->
        name -> `Data.image[image_name] = Image(self)`.

        Parameters
        ----------
        image_set : list or pydicom.Dataset
            Slices of one volume, or enhanced dataset(s).

        only_tags : bool
            Skip pixel decoding.

        Raises
        ------
        ValueError
            If the slices do not all share the same Rows/Columns.

        Examples
        --------
        A coronal MR stack ends up (z, y, x) like every other volume:

        >>> from medicalimageanalysis.read.dicom import Read3D
        >>> reader = Read3D(coronal_mr_slices, only_tags=False)
        >>> reader.plane, reader.array.shape
        ('Coronal', (256, 40, 256))
        """
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

        self._stack_perm = None
        self._stack_flips = None
        self.origin = None
        self.spacing = None
        self.dimensions = None
        self.orientation = None
        self.image_matrix = None

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

    @staticmethod
    def _functional_item(ds, frame_index, keyword):
        """
        First item of a functional-group macro for one frame.

        Per-frame groups take priority over shared groups; None if neither has
        the macro.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import Read3D
        >>> item = Read3D._functional_item(enhanced_mr, 0, 'PixelMeasuresSequence')
        >>> list(item.PixelSpacing)
        [0.9375, 0.9375]
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
        """
        Return True when a dataset stores per-frame functional groups.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import Read3D
        >>> Read3D._is_enhanced(enhanced_mr), Read3D._is_enhanced(ct_slice)
        (True, False)
        """
        return 'PerFrameFunctionalGroupsSequence' in ds

    @staticmethod
    def _number(value, default=None):
        """
        Convert a DICOM value to float, with a fallback for empty/invalid.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import Read3D
        >>> Read3D._number('2.5')
        2.5
        >>> Read3D._number('', 1.0), Read3D._number('abc', 0.0), Read3D._number(None)
        (1.0, 0.0, None)
        """
        if value is None or isinstance(value, (str, bytes)) and not value:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _vector(value):
        """
        Convert a DICOM multi-value to a flat float array, or None if empty.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import Read3D
        >>> Read3D._vector([0.5, 0.5])
        array([0.5, 0.5])
        >>> Read3D._vector('') is None, Read3D._vector([]) is None
        (True, True)
        """
        if value is None or isinstance(value, (str, bytes)):
            return None
        arr = np.asarray(value, dtype=np.float64).ravel()
        return arr if arr.size else None

    def _add_classic_frame(self, ds):
        """
        Collect geometry and rescale tags from one classic slice.

        Pixel spacing falls back from PixelSpacing to ImagerPixelSpacing to
        DetectorElementSpacing; thickness from SpacingBetweenSlices to
        SliceThickness; slope defaults to 1 and intercept to 0.

        Examples
        --------
        >>> reader._add_classic_frame(ct_slice)
        >>> reader.frame_slopes[-1], reader.frame_intercepts[-1]
        (1.0, -1024.0)
        """
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
        """
        Collect geometry and rescale tags for every frame of an enhanced object.

        Reads PlanePosition, PlaneOrientation, PixelMeasures and
        PixelValueTransformation per frame (falling back to shared groups, and
        to top-level rescale tags for legacy-converted objects).

        Examples
        --------
        >>> reader._add_enhanced_frames(enhanced_mr)
        >>> len(reader.frame_positions), reader.frame_numbers[:3]
        (176, [0, 1, 2])
        """
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
        """
        Append one normalized frame record to the parallel per-frame lists.

        Examples
        --------
        >>> reader._add_frame(ds=ct_slice, frame_number=None,
        ...                   position=np.array([-250.0, -250.0, 12.5]),
        ...                   orientation=np.array([1.0, 0, 0, 0, 1.0, 0]),
        ...                   pixel_spacing=np.array([0.977, 0.977]),
        ...                   slope=1.0, intercept=-1024.0, thickness=2.5)
        >>> reader.frame_positions[-1]
        array([-250. , -250. ,   12.5])
        """
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

        Raises
        ------
        ValueError
            If the inputs have different Rows/Columns.

        Examples
        --------
        >>> reader._collect_frames()
        >>> len(reader.frame_datasets) == len(reader.frame_positions)
        True
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
        """
        Record a geometry issue without interrupting loading.

        The first reason becomes `unverified`; every distinct reason is kept in
        `unverified_reasons`.

        Examples
        --------
        >>> reader._flag('Spacing')
        >>> reader._flag('Skipped')
        >>> reader.unverified, reader.unverified_reasons
        ('Spacing', ['Spacing', 'Skipped'])
        """
        if reason not in self.unverified_reasons:
            self.unverified_reasons.append(reason)
        if self.unverified is None:
            self.unverified = reason

    def _frame_label(self, i):
        """
        Identifier for frame i: SOPInstanceUID, or (SOPInstanceUID, frame) for enhanced.

        Enhanced frame numbers are reported 1-based.

        Examples
        --------
        >>> reader._frame_label(0)
        '1.2.840.113619.2.55.3.1'
        >>> enhanced_reader._frame_label(4)
        ('1.2.840.113619.2.55.3.9', 5)
        """
        uid = self.frame_datasets[i].SOPInstanceUID
        n = self.frame_numbers[i]
        return uid if n is None else (uid, n + 1)

    def _interpolate_missing(self, volume):
        """
        Linearly interpolate missing slices in rescaled space (in place).

        Slices between two real slices `lo` and `hi` get
        (1 - t) * volume[lo] + t * volume[hi]; integer volumes are rounded.

        Examples
        --------
        >>> reader.slice_map
        [0, 1, None, 2]
        >>> volume[:, 0, 0]
        array([  0, 100,   0, 300], dtype=int16)
        >>> reader._interpolate_missing(volume)
        >>> volume[:, 0, 0]
        array([  0, 100, 200, 300], dtype=int16)
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
        """
        First available slice thickness; flags 'Spacing' and returns 1.0 if none.

        Examples
        --------
        >>> reader.frame_thicknesses
        [None, 2.5, 2.5]
        >>> reader._nominal_thickness()
        2.5
        """
        for thickness in self.frame_thicknesses:
            if thickness:
                return abs(thickness)
        self._flag('Spacing')
        return 1.0

    def _read_plane(self, i, cache):
        """
        Stored (un-rescaled) pixel values of frame i.

        Classic slices free their PixelData immediately. Enhanced datasets are
        decoded once into `cache` (keyed by dataset id) and then indexed by
        frame.

        Examples
        --------
        >>> cache = {}
        >>> plane = reader._read_plane(0, cache)
        >>> plane.shape, plane.dtype
        ((512, 512), dtype('uint16'))
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

        Examples
        --------
        >>> [p[2] for p in reader.frame_positions]
        [-7.5, -5.0, -10.0]
        >>> reader._reorder_frames([2, 0, 1])
        >>> [p[2] for p in reader.frame_positions]
        [-10.0, -7.5, -5.0]
        """
        for name in ('frame_datasets', 'frame_numbers', 'frame_positions', 'frame_orientations',
                     'frame_pixel_spacings', 'frame_slopes', 'frame_intercepts', 'frame_thicknesses'):
            values = getattr(self, name)
            setattr(self, name, [values[i] for i in order])

    def _reorient(self, volume):
        """
        Permute/flip the stacked volume into (+z, +y, +x).

        Uses the permutation and flips worked out by `compute_geometry`; returns
        the input unchanged for a standard axial stack.

        Examples
        --------
        >>> reader._stack_perm, reader._stack_flips
        ((1, 0, 2), (True, False, False))
        >>> reader._reorient(np.zeros((40, 256, 256))).shape
        (256, 40, 256)
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

        The volume starts as int16 when every slope and intercept is an integer
        and is promoted to int32 if a slice overflows int16; otherwise it is
        float32. Missing slices are interpolated before reorientation. Sets
        `self.array`.

        Returns
        -------
        None

        Examples
        --------
        CT stays integer HU:

        >>> reader = Read3D(ct_slices, only_tags=True)
        >>> reader.compute_array()
        >>> reader.array.dtype, int(reader.array.min())
        (dtype('int16'), -1024)

        PET with a fractional slope is float:

        >>> reader = Read3D(pet_slices, only_tags=True)
        >>> reader.compute_array()
        >>> reader.array.dtype
        dtype('float32')
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
        Work out how the stacked volume maps to (z, y, x).

        From header tags only (valid with only_tags=True), finds the axis
        permutation with the best alignment to patient x, y, z (so 45-degree
        obliques cannot map two axes to one), which axes need flipping, and the
        resulting spacing, dimensions, origin, orientation and image matrix. The
        array itself is permuted later in `_reorient`.

        Returns
        -------
        None

        Examples
        --------
        Standard axial CT: no permutation, identity matrix, origin = lowest slice:

        >>> reader = Read3D(ct_slices, only_tags=True)
        >>> reader._stack_perm, reader._stack_flips
        ((0, 1, 2), (False, False, False))
        >>> reader.image_matrix
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

        Coronal MR (IOP [1, 0, 0, 0, 0, -1]), 40 slices of 256 x 256:

        >>> reader = Read3D(coronal_mr_slices, only_tags=True)
        >>> reader._stack_perm, reader._stack_flips
        ((1, 0, 2), (True, False, False))
        >>> tuple(reader.dimensions)
        (256, 40, 256)
        >>> reader.spacing   # x = column, y = slice, z = row spacing
        array([0.9375, 4.    , 0.9375])
        """
        row = self.acquisition_orientation[:3]
        col = self.acquisition_orientation[3:]
        normal = np.cross(row, col)
        pixel_spacing = self.compute_pixel_spacing()

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

        Flags 'Orientation' when frames differ by more than 1e-3 in any
        direction cosine, or when no frame has an orientation (identity is then
        assumed).

        Returns
        -------
        np.ndarray
            Six direction cosines (row, column).

        Examples
        --------
        >>> reader.compute_orientation()
        array([1., 0., 0., 0., 1., 0.])

        A slightly non-orthogonal header is cleaned up:

        >>> reader.frame_orientations = [np.array([1.0, 0, 0, 0.001, 1.0, 0])]
        >>> np.round(reader.compute_orientation(), 6)
        array([1., 0., 0., 0., 1., 0.])
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

        Returns
        -------
        str
            'Axial', 'Coronal' or 'Sagittal'.

        Examples
        --------
        >>> reader.acquisition_orientation = np.array([1.0, 0, 0, 0, 1.0, 0])
        >>> reader.compute_plane()
        'Axial'
        >>> reader.acquisition_orientation = np.array([0, 1.0, 0, 0, 0, -1.0])
        >>> reader.compute_plane()
        'Sagittal'
        """
        normal = np.cross(self.acquisition_orientation[:3], self.acquisition_orientation[3:])
        return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(normal)))]

    def compute_pixel_spacing(self):
        """
        In-plane [row, column] spacing shared by all frames.

        Flags 'Spacing' when frames differ by more than 1e-4 mm, or when no
        frame has spacing (1.0 mm is then assumed).

        Returns
        -------
        np.ndarray
            [row spacing, column spacing] in mm.

        Examples
        --------
        >>> reader.compute_pixel_spacing()
        array([0.977, 0.977])

        Missing spacing falls back to 1 mm and is flagged:

        >>> reader.frame_pixel_spacings = [None]
        >>> reader.compute_pixel_spacing(), reader.unverified
        (array([1., 1.]), 'Spacing')
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

        - Frames closer than 0.01 mm along the normal are duplicates; the first
          is kept and the rest go to `duplicate_frames` ('Duplicate' flag).
        - Gaps are found as steps that are multiples of the median step; each
          missing slice gets a None entry in the slice map ('Skipped' flag).
        - A real slice more than 0.05 mm off the uniform grid flags 'Irregular'.
        - Without positions, slices are ordered by InstanceNumber and spaced by
          the nominal thickness ('Position' flag).

        Returns
        -------
        spacing : float
            Slice spacing along the normal.

        slice_map : list
            One entry per slice of the stacked volume: index into the frame
            lists, or None for a missing slice to interpolate.

        Examples
        --------
        Slices at z = 0, 2, 4, 8, 10 mm (6 mm missing):

        >>> spacing, slice_map = reader.compute_slice_layout()
        >>> spacing, slice_map
        (2.0, [0, 1, 2, None, 3, 4])
        >>> reader.skipped_slice, reader.missing_slices[0]['num_missing']
        ([3], 1)
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
    Read 2D X-ray (DX / CR / MG) and tomosynthesis into a (z, y, x) array.

    Array axes (0, 1, 2) always run along patient +z, +y, +x (LPS), and
    `image_matrix`, `orientation`, `origin` and `spacing` describe that
    array. A frontal projection therefore becomes a one-voxel-thick coronal
    "slab".

    Orientation source, in priority order:
        1. ImageOrientationPatient (top level or functional groups; tomo has it)
        2. PatientOrientation letters
        3. ViewPosition (+ laterality for mammography)
        4. identity (row +x, column +y)

    Tomo frames are sorted along the slice normal and the origin is the
    first frame's ImagePositionPatient. 2D images without a position get
    origin (0, 0, 0). In-plane spacing from ImagerPixelSpacing is divided by
    EstimatedRadiographicMagnificationFactor.

    Parameters
    ----------
    image_set : pydicom.Dataset or list
        X-ray dataset. Only the first is read; tomo is one multi-frame file.

    only_tags : bool
        If True, metadata only. Geometry is still computed from tags.

    Attributes
    ----------
    array : np.ndarray or None
        (z, y, x) array, int16 (<= 15 bits stored) or int32. MONOCHROME1 is
        inverted so bone is always bright.

    is_tomo : bool
        True when per-frame positions were found.

    spacing_known : bool
        False when no spacing tag was present and 1 mm was assumed.

    plane : str
        Plane of the stored image ('Coronal' for AP/PA, 'Sagittal' for
        laterals).

    image_name : str
        Name the image was registered under.

    Examples
    --------
    An AP chest radiograph (2500 rows x 2048 columns):

    >>> import pydicom
    >>> from medicalimageanalysis.read.dicom import ReadXRay
    >>> ds = pydicom.dcmread('C:/Data/chest_ap.dcm')
    >>> reader = ReadXRay(ds, only_tags=False)
    >>> reader.image_name, reader.plane
    ('DX 01', 'Coronal')
    >>> reader.array.shape
    (2500, 1, 2048)

    Spacing corrected for magnification (ImagerPixelSpacing 0.15 mm,
    factor 1.25):

    >>> reader.spacing
    array([0.12, 1.  , 0.12])

    A breast tomosynthesis volume keeps one slice per frame:

    >>> reader = ReadXRay(pydicom.dcmread('C:/Data/tomo_rcc.dcm'), only_tags=True)
    >>> reader.is_tomo, reader.number_of_frames
    (True, 61)
    """

    def __init__(self, image_set, only_tags):
        """
        Resolve orientation and frames, optionally decode pixels, then register.

        Parameters
        ----------
        image_set : pydicom.Dataset or list
            X-ray dataset (first one used).

        only_tags : bool
            Skip pixel decoding.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadXRay
        >>> reader = ReadXRay(lateral_ds, only_tags=True)
        >>> reader.plane, reader.array
        ('Sagittal', None)
        """
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

    @staticmethod
    def _fg_value(frame, shared, seq_name, attr):
        """
        Functional-group attribute, per-frame group first, then shared group.

        Examples
        --------
        >>> per = tomo_ds.PerFrameFunctionalGroupsSequence[0]
        >>> shared = tomo_ds.SharedFunctionalGroupsSequence[0]
        >>> ReadXRay._fg_value(per, shared, 'PixelMeasuresSequence', 'PixelSpacing')
        [0.085, 0.085]
        """
        for group in (frame, shared):
            if group is None:
                continue

            seq = group.get(seq_name)
            if seq and attr in seq[0]:
                return getattr(seq[0], attr)

        return None

    def _frame_positions(self):
        """
        (F, 3) ImagePositionPatient per frame, or None if any frame lacks one.

        Examples
        --------
        >>> reader._frame_positions().shape
        (61, 3)
        >>> ReadXRay(chest_ds, only_tags=True)._frame_positions() is None
        True
        """
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
        Patient-direction letters ('L', 'FR', ...) to a unit LPS vector, or None.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadXRay
        >>> ReadXRay._letters_to_dir('L')
        array([1., 0., 0.])
        >>> ReadXRay._letters_to_dir('FR')
        array([-0.70710678,  0.        , -0.70710678])
        >>> ReadXRay._letters_to_dir('X') is None
        True
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
        [row, column] spacing in mm, or None.

        Priority: PixelSpacing, functional-group PixelSpacing, ImagerPixelSpacing
        divided by EstimatedRadiographicMagnificationFactor, then
        DetectorElementSpacing.

        Examples
        --------
        >>> ds.ImagerPixelSpacing, ds.EstimatedRadiographicMagnificationFactor
        ([0.15, 0.15], 1.25)
        >>> reader._inplane_spacing()
        [0.12, 0.12]
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
        """
        (row, column) letters inferred from ViewPosition, or None.

        Used only when neither ImageOrientationPatient nor PatientOrientation is
        present. Assumes the image is stored display-ready. Mammography views
        also use ImageLaterality / Laterality.

        Examples
        --------
        >>> ds.ViewPosition = 'AP'
        >>> reader._orientation_from_view()
        ('L', 'F')
        >>> ds.ViewPosition, ds.ImageLaterality = 'CC', 'R'
        >>> reader._orientation_from_view()
        ('P', 'L')
        """
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
        """
        Value from the first frame's (or shared) PixelMeasuresSequence.

        Examples
        --------
        >>> reader._pixel_measures('SliceThickness')
        1.0
        """
        per = self.ds.get('PerFrameFunctionalGroupsSequence')
        frame = per[0] if per else None

        return self._fg_value(frame, self._shared_group(), 'PixelMeasuresSequence', attr)

    def _setup_frames(self):
        """
        Work out frame order, slice step and first-frame origin for multi-frame data.

        Single-frame images get a 1 mm step. When every frame has a position the
        data is treated as tomo: frames are sorted along the normal and the step
        is the median position difference; otherwise the slice-spacing tags
        (or 1 mm) are used.

        Examples
        --------
        >>> reader._setup_frames()
        >>> reader.is_tomo, reader._slice_dz
        (True, 1.0)
        """
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
        """
        Shared functional-group item, or None.

        Examples
        --------
        >>> reader._shared_group() is None   # plain 2D DX
        True
        """
        seq = self.ds.get('SharedFunctionalGroupsSequence')

        return seq[0] if seq else None

    def _slice_spacing(self):
        """
        Best through-plane spacing from the tags, or None.

        Tries SpacingBetweenSlices (functional groups, then top level), then
        SliceThickness.

        Examples
        --------
        >>> reader._slice_spacing()
        1.0
        """
        for value in (self._pixel_measures('SpacingBetweenSlices'),
                      self.ds.get('SpacingBetweenSlices'),
                      self._pixel_measures('SliceThickness')):
            if value is not None and float(value) > 0:
                return float(value)

        return None

    def compute_array(self):
        """
        Decode pixels, invert MONOCHROME1, and stack as (frames, rows, cols).

        The final (z, y, x) reordering happens in `compute_geometry`.

        Returns
        -------
        None

        Examples
        --------
        A MONOCHROME1 CR is inverted so bone is bright:

        >>> reader = ReadXRay(cr_ds, only_tags=True)
        >>> reader.compute_array()
        >>> reader.array.shape, reader.array.dtype
        ((1, 2140, 1760), dtype('int16'))
        """
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
        Reorder and flip the grid so array axes (0, 1, 2) run along +z, +y, +x.

        Works from tags alone, so it is valid with only_tags=True; the array is
        transformed alongside when it has been loaded.

        Returns
        -------
        None

        Examples
        --------
        AP chest (row = L, column = F): rows become z and are flipped so +z
        points to the head, the single frame becomes y:

        >>> reader.orientation = np.array([1.0, 0, 0, 0, 0, -1.0])
        >>> reader.compute_geometry()
        >>> tuple(reader.dimensions), reader.image_matrix
        ((2500, 1, 2048), array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32))
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
        """
        Plane of the stored image from the row x column normal.

        Returns
        -------
        str
            'Axial', 'Coronal' or 'Sagittal'.

        Examples
        --------
        >>> reader.orientation = np.array([1.0, 0, 0, 0, 0, -1.0])   # AP
        >>> reader.compute_plane()
        'Coronal'
        >>> reader.orientation = np.array([0, 1.0, 0, 0, 0, -1.0])   # lateral
        >>> reader.compute_plane()
        'Sagittal'
        """
        normal = np.cross(self.orientation[:3], self.orientation[3:])

        return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(normal)))]

    def initial_orientation(self):
        """
        Best available (row, column) direction cosines of the stored image.

        Uses ImageOrientationPatient, else PatientOrientation letters, else
        ViewPosition, else identity. Letter-derived columns are made orthogonal
        to the row.

        Returns
        -------
        list[float]
            Six direction cosines.

        Examples
        --------
        PatientOrientation ['L', 'F'] on an AP film:

        >>> reader.initial_orientation()
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]

        No orientation information at all:

        >>> reader.initial_orientation()
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        """
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
    Read radio-fluoroscopy (RF) DICOM into a (z, y, x) array and register it.

    Single-frame and cine (multi-frame) data, grayscale or RGB. The plane is
    inferred from PatientOrientation and, for cine, the frame axis is put on
    the through-plane axis so scrolling slices steps through frames. No
    volumetric reconstruction is performed.

    Notes
    -----
    - Frame spacing is nominal (1 mm); real timing is in `frame_times`.
    - Without PatientOrientation the image is assumed to be a frontal
      (coronal) projection.

    Parameters
    ----------
    image_set : list or pydicom.Dataset
        RF dataset(s); only the first is read.

    only_tags : bool
        If True, only metadata is loaded.

    Attributes
    ----------
    array : np.ndarray or None
        (z, y, x[, 3]) array.

    spacing : np.ndarray
        (x, y, z) spacing in mm.

    dimensions : tuple
        (z, y, x) shape, also available with only_tags=True.

    frame_times : np.ndarray or None
        Per-frame time offsets in ms.

    rgb : bool
        True for 3-sample (color) data.

    image_name : str
        Name the image was registered under.

    Examples
    --------
    A 30-frame barium swallow, 1024 x 1024, PatientOrientation ['L', 'F']:

    >>> import pydicom
    >>> from medicalimageanalysis.read.dicom import ReadRF
    >>> reader = ReadRF(pydicom.dcmread('C:/Data/swallow.dcm'), only_tags=False)
    >>> reader.plane, reader.array.shape
    ('Coronal', (1024, 30, 1024))
    >>> reader.frame_times[:3]
    array([  0. ,  66.7, 133.4])

    Headers only still gives the shape:

    >>> ReadRF(ds, only_tags=True).dimensions
    (1024, 30, 1024)
    """

    def __init__(self, image_set, only_tags):
        """
        Infer plane and timing, optionally decode pixels, then register.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadRF
        >>> reader = ReadRF(ds, only_tags=True)
        >>> reader.image_name, reader.array
        ('RF 01', None)
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

    @staticmethod
    def _functional_group_spacing(img):
        """
        PixelSpacing from enhanced functional groups (shared first), or None.

        Examples
        --------
        >>> ReadRF._functional_group_spacing(enhanced_xrf)
        [0.308, 0.308]
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
        PatientOrientation joined into one string, e.g. 'LF'.

        Examples
        --------
        >>> ds.PatientOrientation = ['L', 'F']
        >>> reader._orientation_string()
        'LF'
        """
        orient = self.image_set[0].get('PatientOrientation', None)
        if not orient:
            return ''

        if isinstance(orient, str):
            return orient.replace('\\', '')

        return ''.join(str(o) for o in orient)

    def _rows_toward_feet(self):
        """
        True if increasing row index points to the feet (column letter has 'F').

        Examples
        --------
        >>> ds.PatientOrientation = ['L', 'F']
        >>> reader._rows_toward_feet()
        True
        """
        orient = self.image_set[0].get('PatientOrientation', None)
        if not orient or isinstance(orient, str) or len(orient) < 2:
            return False

        return 'F' in str(orient[1])

    def _target_dtype(self):
        """
        int16, or int32 for unsigned data with more than 15 bits stored.

        Examples
        --------
        >>> ds.PixelRepresentation, ds.BitsStored = 0, 16
        >>> reader._target_dtype()
        <class 'numpy.int32'>
        """
        ds = self.image_set[0]
        unsigned = int(ds.get('PixelRepresentation', 0)) == 0
        bits_stored = int(ds.get('BitsStored', 16))
        if unsigned and bits_stored > 15:
            return np.int32

        return np.int16

    def compute_array(self):
        """
        Decode RF pixels into a (z, y, x[, 3]) array.

        Steps:
            - normalize to (frames, rows, cols[, 3])
            - cast grayscale to a safe integer type
            - move the frame axis onto the through-plane axis for the plane
            - flip so +z points superior when rows run toward the feet
            - free PixelData and set `dimensions`

        Returns
        -------
        None

        Examples
        --------
        >>> reader = ReadRF(ds, only_tags=True)
        >>> reader.compute_array()
        >>> reader.array.shape, reader.dimensions
        ((1024, 30, 1024), (1024, 30, 1024))
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
            None for single-frame data or when no timing tag is present.

        Examples
        --------
        FrameTime of 33.3 ms over 4 frames:

        >>> reader.compute_frame_times()
        array([ 0. , 33.3, 66.6, 99.9])

        FrameTimeVector [0, 33, 33, 34] is accumulated:

        >>> reader.compute_frame_times()
        array([  0.,  33.,  66., 100.])
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
        Plane from PatientOrientation letters.

        H/F marks the vertical axis: with L/R it is coronal, with A/P sagittal.
        No H/F means axial; no PatientOrientation means coronal (frontal fluoro).

        Returns
        -------
        str
            'Axial', 'Coronal' or 'Sagittal'.

        Examples
        --------
        >>> ds.PatientOrientation = ['L', 'F']
        >>> reader.compute_plane()
        'Coronal'
        >>> ds.PatientOrientation = ['P', 'F']
        >>> reader.compute_plane()
        'Sagittal'
        >>> ds.PatientOrientation = ['L', 'P']
        >>> reader.compute_plane()
        'Axial'
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
        Voxel spacing (x, y, z) matched to the plane-dependent array layout.

        In-plane spacing comes from PixelSpacing, ImagerPixelSpacing (corrected
        by magnification), functional groups, then DetectorElementSpacing
        (1 mm if none). The frame axis gets 1 mm.

        Returns
        -------
        np.ndarray
            (x, y, z) spacing in mm.

        Examples
        --------
        Coronal with PixelSpacing [0.3, 0.3] (frames are y):

        >>> reader.compute_spacing()
        array([0.3, 1. , 0.3])
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
        (z, y, x) shape from Rows / Columns / NumberOfFrames, without pixels.

        Examples
        --------
        Coronal, 1024 rows x 1024 columns, 30 frames:

        >>> reader.dimensions_from_tags()
        (1024, 30, 1024)
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
    Read ultrasound DICOM into a standardized array and register it in `Data`.

    Output array layout:
        grayscale (default)   : (frames, rows, cols) uint8
        keep_color=True       : (frames, rows, cols, 3) uint8, `rgb` = True

    By default color overlays (Doppler, annotations) are removed: pixels
    whose RGB channels differ by more than `color_tolerance` are set to 0.
    Frames are time, not depth, so the z spacing is 1.0.

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

    Attributes
    ----------
    array : np.ndarray or None
        uint8 array as described above.

    spacing : np.ndarray
        (x, y, z) spacing in mm.

    spacing_known : bool
        False when no spacing source was found and 1 mm was assumed.

    frame_time : float or None
        Frame interval in ms.

    image_name : str
        Name the image was registered under.

    Examples
    --------
    A 120-frame liver cine, gray B-mode extracted from RGB:

    >>> import pydicom
    >>> from medicalimageanalysis.read.dicom import ReadUS
    >>> reader = ReadUS(pydicom.dcmread('C:/Data/liver_us.dcm'), only_tags=False)
    >>> reader.array.shape, reader.array.dtype
    ((120, 600, 800), dtype('uint8'))
    >>> reader.spacing, reader.spacing_known
    (array([0.2 , 0.25, 1.  ]), True)

    Keep the color Doppler overlay:

    >>> reader = ReadUS(ds, only_tags=False, keep_color=True)
    >>> reader.rgb, reader.array.shape
    (True, (120, 600, 800, 3))
    """

    def __init__(self, image_set, only_tags, keep_color=False, color_tolerance=3):
        """
        Read frame/sample counts, optionally decode pixels, then register.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadUS
        >>> reader = ReadUS(ds, only_tags=True)
        >>> reader.image_name, tuple(reader.dimensions)
        ('US 01', (120, 600, 800))
        """
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

    @staticmethod
    def _detector_spacing(ds):
        """
        DetectorElementSpacing from ContributingSourcesSequence, or None.

        Examples
        --------
        >>> ReadUS._detector_spacing(ds) is None
        True
        """
        seq = getattr(ds, 'ContributingSourcesSequence', None)
        if seq and 'DetectorElementSpacing' in seq[0]:
            return [float(v) for v in seq[0].DetectorElementSpacing]

        return None

    @staticmethod
    def _pixel_spacing(ds):
        """
        PixelSpacing from top level or functional groups, or None.

        Examples
        --------
        >>> ReadUS._pixel_spacing(ds)
        [0.25, 0.2]
        """
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
        [row, col] spacing in mm from the first 2D cm region, or None.

        Uses SequenceOfUltrasoundRegions items with RegionSpatialFormat 1 (2D)
        and physical units 3 (cm) in both directions.

        Examples
        --------
        PhysicalDeltaX 0.02 cm, PhysicalDeltaY 0.025 cm:

        >>> ReadUS._region_spacing(ds)
        [0.25, 0.2]
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
        """
        Scale an array down to 8 bits by dropping the extra low bits.

        Examples
        --------
        >>> import numpy as np
        >>> from medicalimageanalysis.read.dicom import ReadUS
        >>> ReadUS._to_uint8(np.array([0, 1024, 4095], dtype=np.uint16), bits=12)
        array([  0,  64, 255], dtype=uint8)
        """
        if arr.dtype == np.uint8:
            return arr
        shift = max(int(bits) - 8, 0)
        return np.clip(arr >> shift, 0, 255).astype(np.uint8)

    def compute_array(self):
        """
        Decode pixels and normalize to grayscale or RGB uint8.

        PALETTE COLOR is expanded through its LUT first. Layout is decided from
        SamplesPerPixel, not from the array's ndim.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the decoded array does not match the expected layout.

        Examples
        --------
        >>> reader = ReadUS(ds, only_tags=True)
        >>> reader.compute_array()
        >>> reader.array.shape, reader.rgb
        ((120, 600, 800), False)
        """
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
        (x, y, z) spacing; z is 1.0 because frames are time.

        Priority: PixelSpacing / functional groups, then the 2D tissue region in
        cm, then DetectorElementSpacing. Sets `spacing_known`.

        Returns
        -------
        np.ndarray
            (x, y, z) spacing in mm.

        Examples
        --------
        Region spacing [0.25, 0.2] (row, col):

        >>> reader.compute_spacing()
        array([0.2 , 0.25, 1.  ])

        No spacing information at all:

        >>> reader.compute_spacing(), reader.spacing_known
        (array([1., 1., 1.]), False)
        """
        ds = self.image_set[0]
        inplane = self._pixel_spacing(ds) or self._region_spacing(ds) or self._detector_spacing(ds)

        self.spacing_known = inplane is not None
        if inplane is None:
            inplane = [1.0, 1.0]

        return np.array([inplane[1], inplane[0], 1.0])  # [row, col] -> (x, y, z)


class ReadRTStruct:
    """
    Read an RTSTRUCT: planar ROIs, POINT POIs, and the image they belong to.

    ROIs are kept when their first contour is CLOSED_PLANAR, OPEN_PLANAR,
    OPEN_NONPLANAR or CLOSEDPLANAR_XOR; POINT structures become POIs. ROIs
    without a display color get a stable color derived from their name. The
    reader does not attach itself; `DicomReader.image_creation` passes it to
    `Image.input_rtstruct` of `match_image_name`.

    Parameters
    ----------
    image_set : pydicom.Dataset
        RTSTRUCT dataset.

    only_tags : bool
        If True, parse names/colors only (no coordinates).

    images : dict, optional
        name -> image object with `series_uid`, `sops`, and optionally
        `frame_of_reference_uid`. Defaults to the global `Data.image`.

    Attributes
    ----------
    roi_names, roi_colors, roi_types : list
        Per-ROI name, [R, G, B] color and geometric type.

    poi_names, poi_colors : list
        Per-POI name and color.

    contours : list
        One list per ROI of (N, 3) contour point arrays in patient mm.

    points : list
        One (1, 3) array per POI.

    match_image_name : str or None
        Loaded image the structures belong to.

    Examples
    --------
    Read a structure set against the images already loaded:

    >>> import pydicom
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import ReadRTStruct
    >>> rs = ReadRTStruct(pydicom.dcmread('C:/Data/Patient01/RS.dcm'), only_tags=False)
    >>> rs.match_image_name
    'CT 01'
    >>> rs.roi_names, rs.roi_types
    (['Liver', 'GTV'], ['CLOSED_PLANAR', 'CLOSED_PLANAR'])
    >>> len(rs.contours[0]), rs.contours[0][0].shape
    (61, (184, 3))

    Attach it the way the pipeline does:

    >>> Data.image[rs.match_image_name].input_rtstruct(rs)

    Names only, e.g. to list structures before loading:

    >>> ReadRTStruct(ds, only_tags=True).contours
    []
    """

    def __init__(self, image_set, only_tags, images=None):
        """
        Parse structure properties and match the structure set to an image.

        Image matching only runs when the file has at least one ROI or POI;
        coordinates are parsed as well unless only_tags is True.

        Examples
        --------
        Match against a custom image dictionary instead of `Data.image`:

        >>> from medicalimageanalysis.read.dicom import ReadRTStruct
        >>> rs = ReadRTStruct(ds, only_tags=True, images={'Planning CT': ct_image})
        >>> rs.match_image_name
        'Planning CT'
        """
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
    def _contour_values(raw):
        """
        ContourData as a flat float array, whether pydicom decoded it or not.

        Accepts decoded multi-values, lists of strings, or the raw backslash-
        separated string/bytes.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadRTStruct
        >>> ReadRTStruct._contour_values(['10.5', '-20.0', '35.25'])
        array([ 10.5 , -20.  ,  35.25])
        """
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("ascii", errors="ignore")
        if isinstance(raw, str):
            raw = [v for v in raw.strip("\x00 ").split("\\") if v.strip()]

        return np.asarray(raw, dtype=float).ravel()

    @staticmethod
    def _stable_color(name):
        """
        Deterministic fallback [R, G, B] color derived from the ROI name.

        The same name always gets the same color, across sessions and machines.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadRTStruct
        >>> ReadRTStruct._stable_color('GTV')
        [112, 143, 216]
        >>> ReadRTStruct._stable_color('Liver')
        [241, 163, 203]
        """
        h = hashlib.md5(name.encode("utf-8")).digest()

        return [int(h[0]), int(h[1]), int(h[2])]

    def get_frame_of_reference_uid(self):
        """
        Frame of Reference UID the structures are defined in, or None.

        Checks StructureSetROISequence first, then
        ReferencedFrameOfReferenceSequence.

        Returns
        -------
        str or None

        Examples
        --------
        >>> rs.get_frame_of_reference_uid()
        '1.2.840.113619.2.55.3.2831164355.12'
        """
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
        """
        Extract name, color, geometric type and referenced SOPs for each structure.

        Structures without a name, without any usable contour, or with an
        unsupported geometric type are skipped.

        Returns
        -------
        list[dict]
            Keys: 'index', 'number', 'name', 'color', 'type', 'sops'.

        Examples
        --------
        >>> props = rs.get_properties()
        >>> [(p['name'], p['type'], p['color']) for p in props]
        [('Liver', 'CLOSED_PLANAR', [0, 150, 255]), ('GTV', 'CLOSED_PLANAR', [255, 0, 0])]
        >>> len(props[0]['sops'])
        61
        """
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
                        if hasattr(c, "ContourData") and self._contour_values(c.ContourData).size >= 3]
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
        SeriesInstanceUID of the referenced image series (not the RTSTRUCT's own).

        Returns
        -------
        str or None

        Examples
        --------
        >>> rs.get_referenced_series_uid() == Data.image['CT 01'].series_uid
        True
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
        Find the loaded image the structure set belongs to.

        Priority:
            1. referenced series UID (confirmed by SOP overlap when SOPs exist)
            2. any image containing referenced SOPs
            3. same Frame of Reference UID

        Parameters
        ----------
        images : dict
            name -> image object.

        Returns
        -------
        str or None
            Name of the matching image.

        Examples
        --------
        >>> rs.match_with_image(Data.image)
        'CT 01'

        A structure set for a series that is not loaded:

        >>> rs.match_with_image({}) is None
        True
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
        Fill `contours` (per ROI, list of (N, 3) arrays) and `points` (per POI).

        Contour items whose type differs from the ROI's type, or whose value
        count is not a multiple of 3, are skipped.

        Returns
        -------
        None

        Examples
        --------
        >>> rs.structure_positions()
        >>> rs.roi_names[1], len(rs.contours[1])
        ('GTV', 9)
        >>> rs.poi_names, rs.points[0]
        (['Isocenter'], array([[ 12.4, -35.1, 102.5]]))
        """
        seqs = self.image_set.ROIContourSequence
        for p in self._properties:
            arrays = []
            for c in seqs[p["index"]].ContourSequence:
                if not hasattr(c, "ContourData"):
                    continue
                if str(c.ContourGeometricType).upper() != p["type"]:
                    continue
                vals = self._contour_values(c.ContourData)
                if vals.size % 3:
                    continue  # malformed item; skip rather than crash
                arrays.append(vals.reshape(-1, 3))

            if p["type"] == "POINT":
                self.points.append(arrays[0])
            else:
                self.contours.append(arrays)


class ReadREG:
    """
    Read a DICOM Spatial Registration (rigid) or Deformable Spatial Registration.

    Reference vs. moving is resolved per registration item:
        1. Distinct FoRs -> item matching the REG FrameOfReferenceUID is the
           reference.
        2. Shared FoR    -> identity-transform item is the reference.
        3. Ambiguous     -> first item is the reference.

    The reference is the Registered RCS (the REG's own FrameOfReferenceUID),
    the moving image is the Source RCS. Both images must already be loaded;
    they are found by SOP Instance UID.

    Rigid: `moving_matrix` is reference -> moving (for pull-resampling the
    moving image) and is passed to `Rigid` as `matrix`.

    Deformable: DICOM maps a registered (reference) point r to the source
    (moving) point

        m = Post @ (Pre @ r + V(r))

    which is the Deformable convention  m = rigid_matrix @ (r + dvf(r))  with

        rigid_matrix = Post,   dvf(r) = V(r) + (Pre - I) @ r

    so the vector grid is stored as is (no inversion, no name swap). A
    non-identity Pre is folded into the field; that assumes V is sampled at
    the registered point r, which is where the standard places the grid. NaN
    vectors (undefined points) are stored as zero displacement.

    dvf is (z, y, x, 3) float32 in (dx, dy, dz) mm; dimensions are
    (z, y, x); spacing/origin are (x, y, z); dvf_matrix rows are the grid's
    index axes in world (as Deformable expects).

    Parameters
    ----------
    image_set : pydicom.Dataset or list of pydicom.Dataset
        The REG dataset (first one used for a list).

    only_tags : bool, default False
        Only resolve the referenced images and the name.

    Attributes
    ----------
    deformable : bool
        True for a Deformable Spatial Registration.

    reference_name, moving_name : str or None
        Loaded images the registration connects.

    registration_name : str
        'CT 01_MR 02' (rigid) or 'DVF_CT 01_CBCT 02' (deformable), with
        '_1', '_2', ... added on collision.

    moving_matrix : np.ndarray or None
        Rigid: reference -> moving 4x4. Deformable: the Post matrix.

    dvf : np.ndarray or None
        (z, y, x, 3) displacement field in mm (deformable only).

    registration : Rigid or Deformable or None
        The created object (None when an image is missing or with only_tags).

    Examples
    --------
    A rigid CT-MR fusion:

    >>> import pydicom
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import ReadREG
    >>> Data.image_list()
    ['CT 01', 'MR 02']
    >>> reg = ReadREG(pydicom.dcmread('C:/Data/Patient01/REG.dcm'))
    >>> reg.registration_name, Data.rigid_list()
    ('CT 01_MR 02', ['CT 01_MR 02'])
    >>> np.round(reg.moving_matrix[:3, 3], 1)
    array([-10. ,   2.5,   0. ])

    A deformable CT-CBCT registration:

    >>> reg = ReadREG(pydicom.dcmread('C:/Data/Patient01/DREG.dcm'))
    >>> reg.deformable, reg.dvf.shape
    (True, (88, 128, 128, 3))
    >>> Data.deformable_list()
    ['DVF_CT 01_CBCT 03']

    Peek at which images a REG links without building anything:

    >>> reg = ReadREG(ds, only_tags=True)
    >>> reg.reference_name, reg.moving_name, reg.registration
    ('CT 01', 'MR 02', None)
    """

    def __init__(self, image_set, only_tags=False):
        """
        Parse a spatial registration object and create the matching registration.

        Parameters
        ----------
        image_set : pydicom.Dataset or list of pydicom.Dataset
            The REG dataset (the first one is used when a list is given).

        only_tags : bool, default False
            Only resolve the referenced images and the name; no matrices, no
            vector grid, no registration object.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadREG
        >>> reg = ReadREG(ds, only_tags=True)
        >>> reg.reference_name, reg.moving_name
        ('CT 01', 'MR 02')
        """
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
        self.pre_matrix = None
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
    def _deform_matrix(item, keyword):
        """
        4x4 Pre / Post Deformation matrix of a deformable item; identity when absent.

        Examples
        --------
        >>> ReadREG._deform_matrix(item, 'PreDeformationMatrixRegistrationSequence')
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        """
        if item is None or keyword not in item:
            return np.eye(4)

        return ReadREG._to_4x4(item[keyword][0].FrameOfReferenceTransformationMatrix)

    @staticmethod
    def _item_sops(item):
        """
        Referenced SOP Instance UIDs of a registration item ([] for None).

        Examples
        --------
        >>> ReadREG._item_sops(None)
        []
        >>> len(ReadREG._item_sops(reg._mov_item))
        88
        """
        if item is None:
            return []

        return [ref.ReferencedSOPInstanceUID for ref in item.get('ReferencedImageSequence', [])]

    @staticmethod
    def _to_4x4(values):
        """
        Flat DICOM transform (16 values, row major) to a 4x4 matrix.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadREG
        >>> ReadREG._to_4x4([1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])[:3, 3]
        array([10.,  0.,  0.])
        """
        return np.asarray(values, dtype=float).reshape(4, 4)

    def _is_identity_item(self, item):
        """
        True when a registration item is the identity transform.

        Deformable items must also have no vector grid.

        Examples
        --------
        >>> reg._is_identity_item(reg.ds.RegistrationSequence[0])
        True
        """
        if self.deformable:
            return ('DeformableRegistrationGridSequence' not in item
                    and np.allclose(self._deform_matrix(item, 'PreDeformationMatrixRegistrationSequence'), np.eye(4))
                    and np.allclose(self._deform_matrix(item, 'PostDeformationMatrixRegistrationSequence'), np.eye(4)))

        return np.allclose(self._rigid_matrix(item), np.eye(4))

    def _rigid_matrix(self, item):
        """
        Composite rigid transform of an item (each matrix applied in order).

        Identity for None or an item without MatrixRegistrationSequence.

        Examples
        --------
        >>> reg._rigid_matrix(None)[:3, 3]
        array([0., 0., 0.])
        >>> np.round(reg._rigid_matrix(reg._mov_item)[:3, 3], 1)
        array([ 10. ,  -2.5,   0. ])
        """
        if item is None or 'MatrixRegistrationSequence' not in item:
            return np.eye(4)

        m = np.eye(4)
        for mat in item.MatrixRegistrationSequence[0].MatrixSequence:
            m = self._to_4x4(mat.FrameOfReferenceTransformationMatrix) @ m

        return m

    def compute_deformable(self):
        """
        Read the deformation grid into the Deformable convention.

        Sets `dvf` (reference -> moving), `moving_matrix` (= Post), `dvf_matrix`,
        `origin`, `spacing`, `dimensions`, `pre_matrix` and `post_matrix`.

        The vector grid is copied out of the DICOM buffer and the buffer is
        deleted right after, so the raw bytes don't stay in RAM next to the
        field. A non-identity Pre is folded into the field one z slab at a time.

        Returns
        -------
        None

        Examples
        --------
        >>> reg = ReadREG(ds, only_tags=True)
        >>> reg.compute_deformable()
        >>> reg.dvf.shape, reg.dvf.dtype
        ((88, 128, 128, 3), dtype('float32'))
        >>> reg.spacing
        array([4., 4., 4.])
        """
        item = self._mov_item
        if item is None or 'DeformableRegistrationGridSequence' not in item:
            return

        self.pre_matrix = self._deform_matrix(item, 'PreDeformationMatrixRegistrationSequence')
        self.post_matrix = self._deform_matrix(item, 'PostDeformationMatrixRegistrationSequence')

        grid = item.DeformableRegistrationGridSequence[0]

        row = np.asarray(grid.ImageOrientationPatient[:3], dtype=float)
        col = np.asarray(grid.ImageOrientationPatient[3:], dtype=float)
        self.dvf_matrix = np.stack([row, col, np.cross(row, col)])      # rows = index axes in world

        self.origin = np.asarray(grid.ImagePositionPatient, dtype=float)
        self.dimensions = np.flip(np.asarray(grid.GridDimensions, dtype=int))
        self.spacing = np.asarray(grid.GridResolution, dtype=float)

        # Copy out of the read-only DICOM buffer (Pre is folded into it below), then free the buffer
        dvf = np.frombuffer(grid.VectorGridData, dtype='<f4').reshape(*self.dimensions, 3).copy()
        del grid.VectorGridData

        # NaN marks grid points where the transformation is undefined: no displacement there
        np.nan_to_num(dvf, copy=False, nan=0.0)

        # source = Post @ (Pre @ r + V(r)) = Post @ (r + V(r) + (Pre - I) @ r)
        if not np.allclose(self.pre_matrix, np.eye(4)):
            nz, ny, nx = (int(v) for v in self.dimensions)
            rotation = self.pre_matrix[:3, :3] - np.eye(3)
            translation = self.pre_matrix[:3, 3]
            jj, ii = np.mgrid[:ny, :nx]
            for k in range(nz):                                         # one slab at a time keeps memory low
                index = np.stack([ii, jj, np.full_like(ii, k)], axis=-1) * self.spacing
                r = self.origin + index @ self.dvf_matrix               # world point of each grid voxel
                dvf[k] += (r @ rotation.T + translation).astype(np.float32)

        self.dvf = dvf
        self.moving_matrix = self.post_matrix                           # rigid step after the deformation

    def compute_rigid(self):
        """
        Compute the rigid reference -> moving matrix into `moving_matrix`.

        DICOM stores each item's matrix as its own frame -> registered frame;
        the result is inv(inv(reference) @ moving).

        Returns
        -------
        None

        Examples
        --------
        A moving item that shifts +10 mm in x gives a -10 mm pull:

        >>> reg = ReadREG(ds, only_tags=True)
        >>> reg.compute_rigid()
        >>> reg.moving_matrix[:3, 3]
        array([-10.,   0.,   0.])
        """
        self.reference_matrix = self._rigid_matrix(self._ref_item)
        moving_to_ref = np.linalg.inv(self.reference_matrix) @ self._rigid_matrix(self._mov_item)
        self.moving_matrix = np.linalg.inv(moving_to_ref)

    def create_name(self):
        """
        Find the loaded reference / moving images by SOP and build a unique name.

        Missing images are named 'Unknown'.

        Returns
        -------
        None

        Examples
        --------
        >>> reg.create_name()
        >>> reg.registration_name
        'DVF_CT 01_CBCT 03'

        Loading the same REG twice:

        >>> reg.create_name()
        >>> reg.registration_name
        'DVF_CT 01_CBCT 03_1'
        """
        ref_first = self.reference_sops[0] if self.reference_sops else None
        mov_first = self.moving_sops[0] if self.moving_sops else None

        for image_name in Data.image_list():
            sops = Data.image[image_name].sops
            if self.reference_name is None and ref_first in sops:
                self.reference_name = image_name

            elif self.moving_name is None and mov_first in sops:
                self.moving_name = image_name

        prefix = 'DVF_' if self.deformable else ''
        base = f"{prefix}{self.reference_name or 'Unknown'}_{self.moving_name or 'Unknown'}"

        registry = Data.deformable_list() if self.deformable else Data.rigid_list()
        name, i = base, 1
        while name in registry:
            name = f'{base}_{i}'
            i += 1

        self.registration_name = name

    def create_registration(self):
        """
        Instantiate and register a `Rigid` or `Deformable` object.

        Skipped when either image isn't loaded, or when a deformable REG has no
        vector grid.

        Returns
        -------
        None

        Examples
        --------
        >>> reg = ReadREG(ds, only_tags=True)
        >>> reg.compute_rigid()
        >>> reg.create_registration()
        >>> Data.rigid_list()
        ['CT 01_MR 02']
        """
        if not (self.reference_name and self.moving_name):
            return

        if self.deformable:
            if self.dvf is None:
                return

            self.registration = Deformable(dvf=self.dvf, origin=self.origin, spacing=self.spacing,
                                           dimensions=tuple(int(v) for v in self.dimensions),
                                           rigid_matrix=self.moving_matrix, dvf_matrix=self.dvf_matrix,
                                           registration_name=self.registration_name,
                                           reference_name=self.reference_name, moving_name=self.moving_name)

        else:
            self.registration = Rigid(self.reference_name, self.moving_name, rigid_name=self.registration_name,
                                      matrix=self.moving_matrix)

    def order_items(self):
        """
        Decide which item is the reference (registered RCS) and which is moving.

        A single identity item is the reference only; a single non-identity
        item is the moving one.

        Returns
        -------
        None

        Examples
        --------
        >>> reg.order_items()
        >>> reg._is_identity_item(reg._ref_item), reg._mov_item is not None
        (True, True)
        """
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
        """
        Resolve series UIDs and SOP lists of the reference and moving images.

        Looks in ReferencedSeriesSequence and
        StudiesContainingOtherReferencedInstancesSequence; leftover series fill
        whichever side could not be matched by SOP.

        Returns
        -------
        None

        Examples
        --------
        >>> reg.resolve_series()
        >>> len(reg.reference_sops), len(reg.moving_sops)
        (142, 88)
        """
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
    Read an RTDOSE grid into canonical (z, y, x) Gy and register it.

    After loading, `array` is indexed [z, y, x] with every axis increasing
    in patient coordinates, and `origin` is the patient position of voxel
    [0, 0, 0]. `spacing` and `origin` are (x, y, z); `dimensions` matches
    `array.shape`. `plane` describes the grid as stored in the file, before
    reorientation. DoseGridScaling is applied, so values are in the file's
    DoseUnits (normally Gy).

    Parameters
    ----------
    image_set : pydicom.Dataset or list
        RTDOSE dataset (first used for a list).

    only_tags : bool, default False
        Geometry only, no dose values.

    Attributes
    ----------
    array : np.ndarray or None
        float32 dose grid (z, y, x).

    dose_name : str
        Name registered in `Data.dose`, e.g. 'RTDOSE 01'.

    Examples
    --------
    Load a plan dose:

    >>> import pydicom
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import ReadRTDose
    >>> dose = ReadRTDose(pydicom.dcmread('C:/Data/Patient01/RD.dcm'))
    >>> dose.dose_name, Data.dose_list()
    ('RTDOSE 01', ['RTDOSE 01'])
    >>> dose.array.shape, float(dose.array.max())
    ((96, 120, 150), 54.12)
    >>> dose.spacing
    array([2.5, 2.5, 2.5])

    Geometry only:

    >>> ReadRTDose(ds, only_tags=True).array is None
    True
    """

    def __init__(self, image_set, only_tags=False):
        """
        Decode (optionally), compute geometry, and register the dose grid.

        Examples
        --------
        >>> from medicalimageanalysis.read.dicom import ReadRTDose
        >>> dose = ReadRTDose([ds])
        >>> dose.modality, dose.plane
        ('RTDOSE', 'Axial')
        """
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

    def _num_frames(self):
        """
        Number of dose planes in the dataset (1 when NumberOfFrames is absent).

        Examples
        --------
        >>> dose._num_frames()
        96
        """
        return int(self.ds.get('NumberOfFrames') or 1)

    def _slice_step(self):
        """
        Signed distance between frames, from GridFrameOffsetVector.

        Falls back to SliceThickness (then 1.0) when the vector is missing or
        has a single entry.

        Examples
        --------
        >>> ds.GridFrameOffsetVector[:3]
        [0.0, -2.5, -5.0]
        >>> dose._slice_step()
        -2.5
        """
        offsets = self.ds.get('GridFrameOffsetVector')
        if offsets is not None and not isinstance(offsets, (str, float)) and len(offsets) > 1:
            step = float(offsets[1]) - float(offsets[0])
            if step != 0:
                return step

        return float(self.ds.get('SliceThickness') or 1.0)

    def compute_array(self):
        """
        Decode the grid as (frames, rows, cols) float32 times DoseGridScaling.

        Returns
        -------
        None

        Examples
        --------
        >>> dose = ReadRTDose(ds, only_tags=True)
        >>> dose.compute_array()
        >>> dose.array.dtype, dose.array.shape
        (dtype('float32'), (96, 120, 150))
        """
        scaling = float(self.ds.get('DoseGridScaling') or 1.0)
        shape = (self._num_frames(), int(self.ds.Rows), int(self.ds.Columns))
        self.array = (self.ds.pixel_array.astype(np.float32) * scaling).reshape(shape)

        del self.ds.PixelData

    def compute_geometry(self):
        """
        Reorder and flip the grid so array axes (0, 1, 2) run along +z, +y, +x.

        Works from tags alone, so it is valid with only_tags=True; the array is
        transformed alongside when it has been loaded. A decreasing
        GridFrameOffsetVector flips the frame axis and moves the origin to the
        most inferior plane.

        Returns
        -------
        None

        Examples
        --------
        Offsets [0, -2.5, -5, ...] with the first plane at z = 100 mm:

        >>> dose.compute_geometry()
        >>> float(dose.origin[2])
        -137.5
        >>> dose.image_matrix
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
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
        """
        Dose-grid ImageOrientationPatient; identity (flagged) when absent.

        Returns
        -------
        np.ndarray
            Six direction cosines.

        Examples
        --------
        >>> dose.compute_orientation()
        array([1., 0., 0., 0., 1., 0.])
        """
        if 'ImageOrientationPatient' in self.ds:
            return np.asarray(self.ds.ImageOrientationPatient, dtype=float)
        self.unverified = 'Orientation'
        return np.array([1, 0, 0, 0, 1, 0], dtype=float)

    def compute_plane(self):
        """
        Plane of the stored grid from its orientation.

        Returns
        -------
        str
            'Axial', 'Coronal' or 'Sagittal'.

        Examples
        --------
        >>> dose.compute_plane()
        'Axial'
        """
        normal = np.cross(self.orientation[:3], self.orientation[3:])
        return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(normal)))]


def create_image_name(modality):
    """
    Generate the next sequential image name for a modality.

    The index is the number of images already in `Data`, plus one, and is
    zero-padded to two digits. The index is shared across modalities, so a
    CT followed by an MR gives 'CT 01' and 'MR 02'.

    Parameters
    ----------
    modality : str
        Imaging modality (e.g. 'CT', 'MR', 'PT').

    Returns
    -------
    str
        Name such as 'CT 01'.

    Examples
    --------
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import create_image_name
    >>> Data.clear()
    >>> create_image_name('CT')
    'CT 01'

    With two images already loaded:

    >>> Data.image_list()
    ['CT 01', 'MR 02']
    >>> create_image_name('PT')
    'PT 03'

    Past nine images the padding stops:

    >>> len(Data.image_list())
    9
    >>> create_image_name('MR')
    'MR 10'
    """
    idx = len(Data.image_list())
    if idx < 9:
        image_name = modality + ' 0' + str(1 + idx)
    else:
        image_name = modality + ' ' + str(1 + idx)

    return image_name


def create_dose_name(modality):
    """
    Generate the next sequential dose name.

    Same scheme as `create_image_name`, counted over `Data.dose`.

    Parameters
    ----------
    modality : str
        Label for the dose (the RTDOSE reader passes 'RTDOSE').

    Returns
    -------
    str
        Name such as 'RTDOSE 01'.

    Examples
    --------
    >>> from medicalimageanalysis import Data
    >>> from medicalimageanalysis.read.dicom import create_dose_name
    >>> Data.dose_list()
    []
    >>> create_dose_name('RTDOSE')
    'RTDOSE 01'

    A second plan's dose:

    >>> Data.dose_list()
    ['RTDOSE 01']
    >>> create_dose_name('RTDOSE')
    'RTDOSE 02'
    """
    idx = len(Data.dose_list())
    if idx < 9:
        image_name = modality + ' 0' + str(1 + idx)
    else:
        image_name = modality + ' ' + str(1 + idx)

    return image_name