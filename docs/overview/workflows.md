# Typical workflows

Below are the kinds of workflows this package is best suited for.

## Study ingestion

1. Parse a folder with `file_parser`
2. Load data with `read_dicoms` or `read_mhd`
3. Inspect results through the `Data` registry

## ROI processing

1. Load image and structure data
2. Extract or build contours
3. Convert contours to masks or meshes
4. Run statistics or geometry operations

## Registration-driven analysis

1. Load reference and moving data
2. Create or read rigid/deformable transforms
3. Apply transforms to downstream ROI or mesh tasks

## Geometry workflows

1. Load external models such as STL, VTK, or 3MF
2. Convert geometry into contours or masks
3. Refine surfaces or generate volume meshes
