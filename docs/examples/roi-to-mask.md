# Example: convert contours or masks

## Structure-first workflow

```python
import medicalimageanalysis as mia

mia.read_dicoms(folder_path="C:/Data/Patient01")

roi = mia.Data.image["CT 01"].roi["Liver"]
mask = roi.compute_mask()
mesh = roi.create_discrete_mesh()
print(mask.shape)
print(mesh.n_points)
```

## Direct utility workflow

```python
import numpy as np
import medicalimageanalysis as mia

mask = np.zeros((8, 32, 32), dtype=np.uint8)
contours = mia.mask_to_contours(mask, (0, 0, 0), (1, 1, 1), np.eye(3))
```

## Direct contour-to-mask workflow

```python
import medicalimageanalysis as mia

roi = mia.Data.image["CT 01"].roi["Liver"]

mask = mia.contours_to_mask(
    roi.contour_position,
    roi.image.array.shape,
    roi.image.origin,
    roi.image.spacing,
    roi.image.matrix,
)

print(mask.shape)
```

## Related functions

- `mask_to_contours`
- `contours_to_mask`
- `mesh_to_mask`

This area is a good candidate for richer tutorials later.
