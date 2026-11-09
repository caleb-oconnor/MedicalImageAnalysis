# Mesh workflows

The package includes utilities for turning imaging-derived objects into mesh
representations and for refining geometry after import.

## Workflow ideas

- import STL, VTK, or 3MF geometry
- generate meshes from contours
- create masks from mesh geometry
- refine surface meshes before downstream processing

## Useful API surface

- `Refinement`
- `Volume`
- `mesh_to_contours`
- `mesh_to_mask`
- `contours_to_mesh`
