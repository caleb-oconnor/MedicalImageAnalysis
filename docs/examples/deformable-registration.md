# Example: deformable registration workflows

This page follows the same pattern as the rigid example: use the structure API
first, then show the direct utility route.

## Structure-first workflow

```python
import medicalimageanalysis as mia

mia.read_dicoms(folder_path="C:/Data/Patient01")

deformable = mia.Deformable()
deformable.compute_registration(
    method="itk_fast_demons",
    reference_rois=["Liver"],
    moving_rois=["Liver"],
    reference_name="CT 01",
    moving_name="MR 01",
)

warped = deformable.create_image()
mesh = deformable.get_display_mesh("Liver")
print(warped.array.shape)
print(mesh.n_points)
```

## Direct utility workflow

```python
import medicalimageanalysis as mia

reference = mia.Data.image["CT 01"]
moving = mia.Data.image["MR 01"]

itk = mia.utils.deformable.simpleitk.DeformableITK(
    reference_image=reference.array,
    moving_image=moving.array,
)

field = itk.fast_demons(iterations=100, shrink_factors=(4, 2, 1))
print(field)
```

## What this example shows

- using `Deformable` as the main registration object
- creating downstream warped outputs from the structure layer
- calling deformable utilities directly when you want the lower-level API