# Example: load DICOMs from a folder

## Simple folder-based loading

```python
import medicalimageanalysis as mia

mia.read_dicoms(folder_path="C:/Data/Patient01")

print(mia.Data.image_list())
print(mia.Data.roi_list())
print(mia.Data.dose_list())
```

## Parse first, then load selectively

```python
import medicalimageanalysis as mia

files = mia.file_parser(folder_path="C:/Data/Patient01")
mia.read_dicoms(
    file_list=files["Dicom"],
    only_modality=["CT", "MR", "RTSTRUCT"],
)

print(mia.Data.image_list())
print(mia.Data.roi_list())
```

## Inspect the structure objects after loading

```python
import medicalimageanalysis as mia

image = mia.Data.image[mia.Data.image_list()[0]]
print(image.array.shape)
print(image.get_bounds())
print(list(image.roi.keys()))
```

## What this example shows

- reading DICOMs directly from a folder
- optionally classifying files before loading
- how DICOM loading populates the shared structure registry
- how to inspect loaded `Image` objects afterward
