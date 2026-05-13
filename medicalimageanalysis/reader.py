"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Medical Imaging IO and File Orchestration Utility
=================================================

Description:
    This module serves as the primary entry point for parsing and loading
    multimodal medical datasets into the Morfeus environment. It provides
    robust file system traversal, memory safety checks, and specialized
    readers for clinical and research data formats.

Supported Formats:
    * DICOM (CT, MR, PT, RTSTRUCT, RTDOSE, REG, etc.)
    * MetaImage (.mhd, .raw)
    * NIfTI (.nii.gz)
    * Surface/Mesh Data (.stl, .vtk, .3mf)

Key Functionalities:
    1. **File Parsing**: Recursive directory searching and categorization
       by imaging modality and file extension.
    2. **Memory Management**: Predictive memory estimation via `check_memory`
       to prevent system crashes during large dataset ingestion.
    3. **Standardized Reading**: Unified wrappers for format-specific
       reader classes (e.g., DicomReader, ThreeMfReader).

Architecture Note:
    Most 'read' functions in this module interact with a global `Data`
    registry. Loading a file typically updates the shared state across
    the application.

Usage:
    >>> import medicalimageanalysis as mia
    >>> # Parse a directory
    >>> files = mia.file_parser(folder_path='path/to/patient_data')
    >>> # Load the DICOMs found
    >>> mia.read_dicoms(file_list=files['Dicom'])

"""

import os
import psutil

from pathlib import Path
from typing import Optional

from .read import DicomReader, MhdReader, StlReader, VtkReader, ThreeMfReader


def check_memory(files: dict[str, list[str]]) -> float:
    """
    Estimate remaining system memory after accounting for file sizes.

    Parameters
    ----------
    files : dict[str, list[str]]
        Dictionary where each key contains a list of file paths.
        Example:
        {
            'Dicom': [...],
            'Nifti': [...],
            'Raw': [...],
        }

    Returns
    -------
    float
        Remaining available memory in GB.

    Examples
    --------
    Estimate remaining memory after loading a patient dataset:

    >>> files = file_parser(folder_path='C:/Data/Patient01')
    >>> remaining_memory = check_memory(files)
    >>> print(f'{remaining_memory:.2f} GB available')
    24.81 GB available

    Estimate memory using a predefined file list:

    >>> files = {
    ...     'Dicom': ['image_001.dcm', 'image_002.dcm'],
    ...     'Nifti': ['segmentation.nii.gz'],
    ...     'Raw': [],
    ...     'Stl': [],
    ...     'Vtk': [],
    ...     '3mf': [],
    ... }
    >>> check_memory(files)
    23.4
    """

    # Sum the size of every file across all file categories
    total_size = sum(
        Path(file).stat().st_size
        for file_list in files.values()
        for file in file_list
    )

    # Get currently available system RAM in bytes
    available_memory = psutil.virtual_memory().available

    # Return remaining memory in gigabytes
    return (available_memory - total_size) / 1e9


def file_parser(
    folder_path: Optional[str] = None,
    file_list: Optional[list[str]] = None,
    exclude_files: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """
    Parse files by extension from a folder or provided file list.

    Parameters
    ----------
    folder_path : str | None
        Root folder to recursively search for files.

    file_list : list[str] | None
        Optional pre-generated list of file paths.
        If provided, folder traversal is skipped.

    exclude_files : list[str] | None
        Files to ignore during parsing.

    Returns
    -------
    dict[str, list[str]]
        Dictionary containing categorized file paths.

    Examples
    --------
    Parse all supported medical imaging files from a folder:

    >>> files = file_parser(folder_path='C:/Data/Patient01')
    >>> files.keys()
    dict_keys([
    ...     'Dicom',
    ...     'MHD',
    ...     'Raw',
    ...     'Nifti',
    ...     'Stl',
    ...     'Vtk',
    ...     '3mf',
    ...     'NoExtension',
    ... ])

    Parse a specific list of files:

    >>> file_list = [
    ...     'C:/Data/image_001.dcm',
    ...     'C:/Data/segmentation.nii.gz',
    ...     'C:/Data/model.stl',
    ... ]
    >>> files = file_parser(file_list=file_list)

    Exclude specific files during parsing:

    >>> files = file_parser(
    ...     folder_path='C:/Data',
    ...     exclude_files=['C:/Data/bad_scan.dcm'],
    ... )
    """

    # Initialize output dictionary
    files = {
        'Dicom': [],
        'MHD': [],
        'Raw': [],
        'Nifti': [],
        'Stl': [],
        'Vtk': [],
        '3mf': [],
        'NoExtension': [],
    }

    # Default empty exclusion list
    exclude_files = exclude_files or []

    # Generate file list from folder if not provided
    if file_list is None:
        file_list = []

        for root, _, filenames in os.walk(folder_path):
            file_list.extend(
                str(Path(root) / filename)
                for filename in filenames
            )

    # Categorize files by extension
    for filepath in file_list:

        if filepath in exclude_files:
            continue

        extension = Path(filepath).suffix.lower()

        if extension == '.dcm':
            files['Dicom'].append(filepath)

        elif extension == '.mhd':
            files['MHD'].append(filepath)

        elif extension == '.raw':
            files['Raw'].append(filepath)

        elif filepath.lower().endswith('.nii.gz'):
            files['Nifti'].append(filepath)

        elif extension == '.stl':
            files['Stl'].append(filepath)

        elif extension == '.vtk':
            files['Vtk'].append(filepath)

        elif extension == '.3mf':
            files['3mf'].append(filepath)

        elif extension == '':
            files['NoExtension'].append(filepath)

    return files


def read_dicoms(
    folder_path: Optional[str] = None,
    file_list: Optional[list[str]] = None,
    exclude_files: Optional[list[str]] = None,
    only_tags: bool = False,
    only_modality: Optional[list[str]] = None,
    only_load_roi_names: Optional[list[str]] = None,
    clear: bool = True,
):
    """
    Load DICOM files using DicomReader.

    Parameters
    ----------
    folder_path : str | None
        Path to a folder containing DICOM files or subfolders with dicom files.

    file_list : list[str] | None
        Explicit list of file paths to load.

    exclude_files : list[str] | None
        Files to exclude from loading.

    only_tags : bool
        If True, only load DICOM metadata tags.

    only_modality : list[str] | None
        Modalities to include.
        Example:
        ['CT', 'MR', 'PT', 'RTSTRUCT']

    only_load_roi_names : list[str] | None
        Specific ROI names to load from RTSTRUCT files.

    clear : bool
        If True, clear existing loaded DICOM data before loading.

    Returns
    -------
    None

    Examples
    --------
    Load all DICOMs from a folder:

    >>> read_dicoms(folder_path='C:/Data/Patient01')

    Load specific files only:

    >>> files = [
    ...     'C:/Data/image_001.dcm',
    ...     'C:/Data/image_002.dcm',
    ... ]
    >>> read_dicoms(file_list=files)

    Load only RTSTRUCT files:

    >>> read_dicoms(
    ...     folder_path='C:/Data',
    ...     only_modality=['CT'],
    ... )

    """

    # Default supported modalities
    if only_modality is None:
        only_modality = [
            'CT',
            'MR',
            'PT',
            'US',
            'DX',
            'RF',
            'CR',
            'RTSTRUCT',
            'REG',
            'RTDOSE',
        ]

    files = None

    # Parse files from folder or provided file list
    if folder_path is not None or file_list is not None:
        files = file_parser(
            folder_path=folder_path,
            file_list=file_list,
            exclude_files=exclude_files,
        )

    # Initialize DICOM reader
    dicom_reader = DicomReader(
        files,
        only_tags,
        only_modality,
        only_load_roi_names,
        clear,
    )

    # Load DICOM data
    dicom_reader.load()


def read_3mf(
    file: str,
    roi_name: str | None = None,
) -> None:
    """
    Load a 3MF mesh file using ThreeMfReader.

    Parameters
    ----------
    file : str
        Path to the 3MF file.

    roi_name : str | None
        Optional ROI name to associate with the mesh.

    Returns
    -------
    None

    Examples
    --------
    Load a 3MF implant mesh:

    >>> read_3mf('C:/Data/implant.3mf')

    Load a 3MF file and assign an ROI name:

    >>> read_3mf(
    ...     file='C:/Data/pelvis_model.3mf',
    ...     roi_name='Pelvis',
    ... )
    """

    # Initialize 3MF reader
    mf3_reader = ThreeMfReader(
        file,
        roi_name=roi_name,
    )

    # Load 3MF data
    mf3_reader.load()


def read_mhd(
    file: str | None = None,
    modality: str | None = None,
    reference_name: str | None = None,
    moving_name: str | None = None,
    roi_name: str | None = None,
    dose=None,
    dvf=None,
) -> None:
    """
    Load an MHD (MetaImage) file using MhdReader.

    Parameters
    ----------
    file : str | None
        Path to the MHD file.

    modality : str | None
        Imaging modality (e.g., CT, MR).

    reference_name : str | None
        Name of the reference image for registration workflows.

    moving_name : str | None
        Name of the moving image for registration workflows.

    roi_name : str | None
        ROI name to associate with this image.

    dose : optional
        Dose object to associate with this image.

    dvf : optional
        Deformation vector field (DVF) to associate with this image.

    Returns
    -------
    None

    Examples
    --------
    Load a basic MHD image:

    >>> read_mhd(file='C:/Data/scan.mhd')

    Load an MHD with modality labeling:

    >>> read_mhd(
    ...     file='C:/Data/scan.mhd',
    ...     modality='CT',
    ... )

    Load MHD for registration workflow:

    >>> read_mhd(
    ...     file='C:/Data/fixed.mhd',
    ...     reference_name='FixedImage',
    ...     moving_name='MovingImage',
    ... )

    Load MHD and associate ROI and dose:

    >>> read_mhd(
    ...     file='C:/Data/scan.mhd',
    ...     roi_name='PTV',
    ...     dose=dose_object,
    ... )
    """

    # Only proceed if file is provided
    if file is not None:

        # Initialize MHD reader with all optional metadata
        mhd_reader = MhdReader(
            file=file,
            modality=modality,
            reference_name=reference_name,
            moving_name=moving_name,
            roi_name=roi_name,
            dose=dose,
            dvf=dvf,
        )

        # Load image into system
        mhd_reader.load()


# def read_stl(self, files=None, create_image=False, match_image=None):
#     stl_reader = StlReader(self)
#     if files is not None:
#         stl_reader.input_files(files)
#     stl_reader.load()
#
#
# def read_vtk(self, files=None, create_image=False, match_image=None):
#         vtk_reader = VtkReader(self)
#         if files is not None:
#             vtk_reader.input_files(files)
#         vtk_reader.load()
