# Example: load a MetaImage volume

## Basic image loading

```python
import medicalimageanalysis as mia

mia.read_mhd(file="C:/Data/scan.mhd", modality="CT")
print(mia.Data.image_list())
```

## Registration-oriented loading

```python
import medicalimageanalysis as mia

mia.read_mhd(
    file="C:/Data/dvf.mhd",
    reference_name="CT 01",
    moving_name="MR 01",
    dvf=True,
)

print(mia.Data.deformable_list())
```

## Inspect the created structure

```python
import medicalimageanalysis as mia

image = mia.Data.image[mia.Data.image_list()[0]]
print(image.spacing)
print(image.get_center())
```

## Variations

- load a standard image volume
- supply registration-related names for deformable workflows
- use MHD data as a stepping stone into ROI or mesh processing
