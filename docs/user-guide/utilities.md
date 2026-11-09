# Utilities

The utility namespace exposes higher-level helpers for common medical imaging
operations, especially around ROI conversion and mesh workflows.

## Common utility groups

### Creation helpers

- `CreateDicomImage`
- `CreateImageFromMask`

### ROI conversion

- `mask_to_contours`
- `mesh_to_contours`
- `contours_to_mask`
- `mesh_to_mask`
- `contours_to_mesh`
- `contours_to_mask_and_mesh`

### ROI conversion classes

- `ContourToDiscreteMesh`
- `ContourToMask`
- `MaskToContour`
- `ModelToMask`

### Mesh and registration helpers

- `MeshSlicer`
- `Volume`
- `Refinement`
- `DeformableITK`

### Additional registration backends

- `DeformableANTs`
- `RigidITK`
- `ICP`

### Measurements

- `contour_longest_distance`
- `mesh_longest_distance`

## Suggested documentation growth

As the docs mature, this section is a good place to add worked examples with:

- contour-to-mask conversion
- mask-to-mesh generation
- registration workflows
- visualization screenshots from PyVista-based outputs
