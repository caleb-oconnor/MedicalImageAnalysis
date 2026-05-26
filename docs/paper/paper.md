---
title: 'MedicalImageAnalysis: A Python Toolkit for Reading, Organizing, and Analyzing Medical Imaging Data'
tags:
  - Python
  - medical imaging
  - DICOM
  - radiology
  - radiation oncology
  - image processing
  - RTSTRUCT
authors:
  - name: Caleb OConnor
    corresponding: true
    affiliation: 1
affiliations:
  - name: The University of Texas MD Anderson Cancer Center, United States
    index: 1
date: 25 May 2026
bibliography: paper.bib
---

# Summary

`MedicalImageAnalysis` is an open-source Python library for reading and
organizing medical imaging data. It accepts a folder of image files, groups
them automatically by scan type and acquisition, and returns structured Python
objects containing the image data and associated metadata. Supported imaging
modalities include CT, MRI, ultrasound, fluoroscopy, and digital X-ray.
Radiotherapy structure files—which encode clinician-drawn boundaries around
tumors and healthy organs—are detected automatically and linked to their
corresponding scan. The library also includes tools for converting 2D contour
outlines into 3D surface meshes, aligning two image volumes to a common
coordinate space, and loading external mesh file formats.

# Statement of Need

Research scientists, medical physicists, and software engineers in radiotherapy
and quantitative radiology routinely receive imaging data as unstructured DICOM
archives—directories that mix scan types, patients, and acquisition sessions
with no consistent organization. Before any analysis can begin, these files
must be parsed, grouped into coherent series, spatially ordered, and linked to
associated annotation files. In radiation oncology this is especially
burdensome: a single patient study may contain a planning CT, one or more MRI
series acquired in a different patient orientation, and an RTSTRUCT file
containing tumor volumes and organs-at-risk drawn by a clinician, each
produced by a different vendor system with differing tag conventions.

No existing Python library handles this combination in a single step.
`pydicom` [@pydicom] provides low-level tag access but leaves series grouping,
slice ordering, orientation normalization, and RTSTRUCT association entirely to
the user. `SimpleITK` [@SimpleITK] offers powerful image I/O and registration
but requires pre-organized input and has no RTSTRUCT reader. `rt-utils`
[@rtutils] reads RTSTRUCT files and generates binary masks but does not handle
series grouping or orientation correction. `dicom2nifti` [@dicom2nifti]
converts organized DICOM series to NIfTI volumes but does not parse RTSTRUCTs
or perform multi-modality organization. The result is that researchers write
substantial custom parsing code for each new dataset or project—code that is
rarely shared, tested, or maintained.

`MedicalImageAnalysis` eliminates this overhead by providing a single entry
point that takes a raw directory and returns fully assembled, orientation-
normalized image objects with RTSTRUCTs already attached, allowing researchers
to focus on analysis rather than data wrangling.

# State of the Field

Several Python packages address overlapping aspects of medical image handling.
`pydicom` [@pydicom] is the standard low-level DICOM parser; it provides
tag access but not series organization or image assembly. `SimpleITK`
[@SimpleITK] excels at image I/O and filtering, including rigid and deformable
registration, but requires pre-organized input and has no RTSTRUCT reader.
`plastimatch` [@plastimatch] is a C++ toolkit with Python bindings that covers
dose-volume computation and image registration, but its Python API is lower
level and less focused on data organization. `rt-utils` [@rtutils] provides
RTSTRUCT reading and mask generation but does not handle series grouping or
orientation correction. `CERR` [@cerr] is a MATLAB-based radiotherapy research
platform with broad coverage, but its Python port is incomplete and the
MATLAB dependency limits accessibility.

`MedicalImageAnalysis` was built rather than extending an existing package for
three reasons. First, a unified reader that simultaneously groups multi-modality
DICOM series and attaches RTSTRUCTs to their reference images does not exist
in the Python ecosystem. Second, automatic FFS orientation normalization—
necessary when datasets originate from scanners configured with different
patient position conventions—requires coupling tag inspection with array
manipulation in ways that cut across the concerns of existing libraries.
Third, exposing mesh conversion and rigid registration under the same API
as DICOM reading reduces the number of dependencies and data-format
conversions required in a typical radiotherapy research workflow.

# Software Design

The library is organized into three layers. The *reader layer* consists of
the `mia.read_dicoms()` entry point, which accepts a folder path or explicit
file list and populates a module-level `mia.Data` registry with image
instances. File discovery, modality splitting, slice ordering, and FFS
reorientation are handled internally. An RTSTRUCT file found during
discovery is matched to an existing image instance via the DICOM frame-of-
reference UID; unmatched RTSTRUCTs are skipped without error. Optional
parameters allow the caller to suppress pixel-array loading (`only_tags`),
restrict reading to a subset of modalities (`only_modality`), exclude
specific files (`exclude_files`), and filter ROI loading to a named subset
(`only_load_roi_names`). Separate entry points (`mia.read_stl()`,
`mia.read_3mf()`) handle mesh file formats by generating a synthetic
bounding-box image from the mesh geometry.

The *data model layer* wraps each image series in an `ImageInstance` object
exposing the 3D NumPy array, per-slice DICOM tag dictionaries, and
pre-parsed metadata attributes (`spacing`, `origin`, `orientation`,
`patient_name`, `mrn`, `series_uid`, `rois`, `pois`, and others). Each
`ROIInstance` stores axial contour coordinates indexed by slice; each
`POIInstance` stores a single spatial point. Two-dimensional acquisitions
are stored as 3D arrays with a synthetic 1 mm slice thickness so that
downstream code can treat all modalities uniformly.

The *utilities layer* (`mia.utils`) provides stateless functions that
operate on NumPy arrays and mesh objects without requiring use of the reader.
Mesh utilities use `pyvista` [@pyvista], `pymeshfix` [@pymeshfix], and
`pyacvd` [@pyacvd] for surface generation, repair, and remeshing. Rigid
image registration delegates to `SimpleITK` [@SimpleITK]. The utilities
layer was deliberately decoupled from the reader so that users who have
already-organized data can use the processing functions in isolation.

A key design trade-off is the all-in-memory loading strategy: all image
arrays are loaded at read time rather than lazily. This simplifies the
API and avoids reference management complexity, but it means that users
must ensure sufficient RAM is available before pointing the reader at
large multi-patient directories. The documentation warns of this
constraint explicitly.

# Research Impact Statement

`MedicalImageAnalysis` was developed at The University of Texas MD Anderson
Cancer Center to support research workflows in the Department of Radiation
Physics. The library underpins internal projects involving automated
contouring, dose-response modeling, and multi-institutional image analysis.
It is available on PyPI and has accumulated installations across multiple
research institutions. The library is available for use in any institution
conducting radiotherapy or radiology research that requires automated
ingestion of heterogeneous DICOM archives.

# Acknowledgements

Funding here?

# References
