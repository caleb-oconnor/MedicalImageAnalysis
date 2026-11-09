# What is MedicalImageAnalysis?

MedicalImageAnalysis is built for workflows where medical imaging data needs to
be read from disk, organized into Python objects, and moved through a sequence
of ROI, mesh, and registration operations.

## Core capabilities

- Read DICOM studies and associated radiotherapy structures
- Load MetaImage volumes and deformable vector fields
- Work with STL, VTK, and 3MF geometry inputs
- Maintain loaded objects inside a shared registry
- Convert between contours, masks, and meshes
- Support rigid and deformable image-related workflows

## Public entry points

Many workflows begin with a small set of top-level functions:

- `medicalimageanalysis.file_parser`
- `medicalimageanalysis.read_dicoms`
- `medicalimageanalysis.read_mhd`
- `medicalimageanalysis.read_3mf`

These are intentionally simple and act as the “front door” of the package.
