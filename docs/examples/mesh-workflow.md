# Example: mesh-oriented workflow

Mesh workflows in this package can be done through structure objects or by
calling the lower-level utility functions directly.

## Structure-first workflow: create a mesh from an ROI

```python
import medicalimageanalysis as mia

mia.read_dicoms(folder_path="C:/Data/Patient01")

image = mia.Data.image["CT 01"]
roi = image.roi["Liver"]
roi.create_mesh(reduction=0.5)

print(roi.mesh.n_points)
print(roi.compute_bounding_box())
```

## Structure-first workflow: slice a mesh

```python
import medicalimageanalysis as mia

image = mia.Data.image["CT 01"]
roi = image.roi["Liver"]

slice_points = roi.compute_mesh_slice(
    origin=image.display.position["Axial"],
    plane="Axial",
)

print(slice_points)
```

## Direct utility workflow

```python
import medicalimageanalysis as mia

roi = mia.Data.image["CT 01"].roi["Liver"]

mesh = mia.contours_to_mesh(
    roi.contour_position,
    roi.image.origin,
    roi.image.spacing,
    roi.image.matrix,
)

slicer = mia.MeshSlicer(mesh)
cut = slicer.slice_transformed(normal=[0, 0, 1], origin=[0, 0, 0], matrix=None)
print(mesh.n_points)
print(cut)
```

## Useful API objects

- `Roi`
- `MeshSlicer`
- `Refinement`
- `Volume`
- `contours_to_mesh`
- `mesh_to_contours`
- `mesh_to_mask`
