# Quickstart

The package is designed around a few high-level entry points:

- `medicalimageanalysis.file_parser`
- `medicalimageanalysis.read_dicoms`
- `medicalimageanalysis.read_mhd`
- the global `medicalimageanalysis.Data` registry

## Parse a folder of files

```python
import medicalimageanalysis as mia

files = mia.file_parser(folder_path="C:/Data/Patient01")
print(files.keys())
```

## Read DICOM data

```python
import medicalimageanalysis as mia

files = mia.file_parser(folder_path="C:/Data/Patient01")
mia.read_dicoms(file_list=files["Dicom"])

print(mia.Data.image_list())
print(mia.Data.roi_list())
```

## Read a MetaImage file

```python
import medicalimageanalysis as mia

mia.read_mhd(file="C:/Data/scan.mhd", modality="CT")
print(mia.Data.image_list())
```

## Access loaded structures

```python
import medicalimageanalysis as mia

image_name = mia.Data.image_list()[0]
image = mia.Data.image[image_name]

print(type(image))
print(image_name)
```

From here, continue to the user guide to understand how readers, structures,
and utility functions are organized.
