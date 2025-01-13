
# MedicalImageConverter

Version 2.0 and later releases contain a structure than in previous versions.

*MedicalImageConverter* is a Python package for working with medical image files. Currently, it
is only works with dicom files with future plans to read in .mhd, .stl, .3mf files. An image instance
is created for each respective image found, a 3D numpy array is created to contain the pixel data and
various variables exist for the tag information. If there is an associated RTSTRUCT file for an image
then they are added to the image instance. The user need only to give the top level folder and not read
in each image folder one by one. Also, the dicom files for multiple images can exist in a single folder,
it will separate them using the tag information.

The module currently imports 5 different modalities and RTSTRUCT files. The accepted
modalites are:
1. CT
2. MR
3. US
4. MG
5. DX

The CT and MR modalities have been tested extensively, along with their respective
ROIs. The other 3 modalities have been tested but only on a few datasets a piece.
For RTSTRUCTS, only those referencing CT and MR have been tested.

The images will be converted to Feet-First-Supine (if not so already), and the 
image position will be updated to reflect the needed rotations.

Disclaimer: All the files will be loaded into memory so be sure you have enough 
RAM available. Meaning don't select a folder path that contains 10s of different 
patient folders because you could possibly run out of RAM. Also, this module does 
not distinguish between patient IDs or patient names.


## Installation
Using [pip](https://pip.pypa.io/en/stable/):
```
pip install MedicalImageConverter
```

## Example 1
The user sets a path to the folder containing the dicom files or highest level folder with subfolders containing dicom
files.

```python
import MedicalImageConverter as mic

path = r'/path/to/folder'

reader = mic.Reader(folder_path=path)
reader.read_dicoms()

```

## Example 2
The user has more options if they are specifics requirements.
1. file_list - if the user already has the files wanted to read in, must be in type list
2. exclude_files - if the user wants to not read certain files
3. only_tags - does not read in the pixel array just the tags
4. only_modality - specify which modalities to read in, if not then all modalities will be read
5. only_load_roi_names - will only load rois with input name

Note: if *folder_path* and *file_list* are both input, then *folder_path* will be used and not both.

```python
import MedicalImageConverter as mic

file_list = ['filepath1.dcm', 'filepath2.dcm', ...]
exclude_files = ['filepath10.dcm', 'filepath11.dcm', ...]

reader = mic.Reader(file_list=file_list, exclude_files=exclude_files, only_tags=True, only_modality=['CT'],
                    only_load_roi_names=['Liver', 'Tumor'])
reader.read_dicoms()

```

## Retrieve image and tags:
The images are stored in a list. Each image instance contains a 3D array (None if *only_tags=True*), all tag information
and popular tags have their own respective variable.

Note: Even 2D images will contain a 3D array, along with a fake slice thickness of 1 mm.

```python
import MedicalImageConverter as mic

path = r'/path/to/folder'

reader = mic.Reader(folder_path=path)
reader.read_dicoms()

images = reader.images

array = images[0].array
tags = images[0].tags  # list of all the tags, for 100 slice CT scan the tags list would be 0-99 each containing a dict

name = images[0].patient_name  # or tags[0].PatientName
spacing = images[0].spacing  # inplane spacing followed by slice thickness

```

Instance variables:
<span style="font-size:.9em;">base_position, date, dimensions,
filepaths, frame_ref, image_matrix, mrn, orientation, 
origin, patient_name, plane, pois,
rgb, rois, sections, series_uid, skipped_slice, 
sops, spacing, tags, time, unverified</span>

## Retrieve ROI/POIs:
Each image contains a roi and poi dictionary, if a RTSTRUCT file associates with an image then each ROI/POI is added to
respective image dictionary.

```python
import MedicalImageConverter as mic

path = r'/path/to/folder'

reader = mic.Reader(folder_path=path)
reader.read_dicoms()

image = reader.images[0]

roi_names = list(image.rois.keys())
roi = image.rois[roi_names[0]]
contour_position = roi.contour_position

poi_names = list(image.pois.keys())
poi = image.rois[poi_names[0]]
point_position = poi.point_position

```