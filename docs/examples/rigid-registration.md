# Example: rigid registration workflows

This page shows the preferred object-oriented workflow first, followed by a
direct utility workflow for users who want lower-level control.

## Structure-first workflow

```python
import medicalimageanalysis as mia

mia.read_dicoms(folder_path="C:/Data/Patient01")

rigid = mia.Rigid(reference_name="CT 01", moving_name="MR 01")
rigid.compute_registration(
    method="icp_point",
    reference_rois=["Liver"],
    moving_rois=["Liver"],
)

aligned = rigid.create_image(base_name="CT 01", image_name="MR 01")
overlay = rigid.compute_array("Axial", mia.Data.image["CT 01"].display.position["Axial"])
print(aligned.array.shape)
print(overlay["array"].shape)
```

## Direct utility workflow

```python
import medicalimageanalysis as mia

reference = mia.Data.image["CT 01"]
moving = mia.Data.image["MR 01"]

itk = mia.utils.rigid.simpleitk.RigidITK(
    reference_image=reference.array,
    moving_image=moving.array,
)

transform = itk.rigid(transform="Euler", metric="Intensity")
print(transform)
```

## What this example shows

- using a `Rigid` object as the main workflow surface
- creating an aligned image through the structure layer
- dropping down to the direct utility API when needed