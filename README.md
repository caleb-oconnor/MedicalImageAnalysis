
# MedicalImageAnalysis

*MedicalImageAnalysis* is a comprehensive Python toolkit for managing and analyzing medical imaging data.
It can read unorganized DICOM and RTSTRUCTs files, automatically organizing them into separate image instances.
The platform supports both 2D and 3D modalities, with the ability to convert 2D contours to 3D surface meshes and 
vice versa. It also provides rigid registration between images, along with a variety of standard image processing 
techniques, enabling seamless integration into research and clinical workflows. There are additional reader file
types for stl and 3mf, where a fake image is generated based on the boundary of the input mesh.

The module currently imports 6 different modalities:
1. CT
2. MR
3. US
4. RF
5. DX
6. RTSTRUCT

CT and MR images will be converted to Feet-First-Supine (if not so already), and the 
image position will be updated to reflect the needed rotations.

Disclaimer: All the files will be loaded into memory so be sure you have enough 
RAM available. Meaning don't select a folder path that contains 100s of different 
patient folders because you could possibly run out of RAM. Also, this module does 
not distinguish between patient IDs or patient names, it will make an image instance for each image regardless
of patient id/name.


## Reader (Dicom)
The reader works by inputting a file path or a list of files. The files are read and split by modality, series instance 
uid, acquisition number (if it exists), and image orientation matrix. An image class instance is generated for each 
image. If an RTSTRUCT exist, it will match with an existing imported image, right now there is no method for importing
an RTSTRUCT that doesn't have a corresponding image. The match image class contains a ROI dictionary variable that will
create a ROI instance for each imported ROI name (the same process exists for POIs).

## Utils
The utils folder contains useful meshing, image conversion, and registration techniques. It doesn't require the images 
to be read in using the reader class or have image or ROI classes based on the reader, it exists completely independent.


## Installation
Using [pip](https://pip.pypa.io/en/stable/):
```
pip install medicalimageanalysis
```

## Reader Example 1
The user sets a path to the folder containing the dicom files or highest level folder with subfolders containing dicom
files.

```python
import medicalimageanalysis as mia

path = r'/path/to/folder'

reader = mia.Reader(folder_path=path)
reader.read_dicoms()

```

## Reader Example 2
The user has more options if they are specifics requirements.
1. file_list - if the user already has the files wanted to read in, must be in type list
2. exclude_files - if the user wants to not read certain files
3. only_tags - does not read in the pixel array just the tags
4. only_modality - specify which modalities to read in, if not then all modalities will be read
5. only_load_roi_names - will only load rois with input name

Note: if *folder_path* and *file_list* are both input, then *folder_path* will be used and not both.

```python
import medicalimageanalysis as mia

file_list = ['filepath1.dcm', 'filepath2.dcm', ...]
exclude_files = ['filepath10.dcm', 'filepath11.dcm', ...]

reader = mia.Reader(file_list=file_list, exclude_files=exclude_files, only_tags=True, only_modality=['CT'],
                    only_load_roi_names=['Liver', 'Tumor'])
reader.read_dicoms()

```

## Retrieve image and tags:
The images are stored in a list. Each image instance contains a 3D array (None if *only_tags=True*), all tag information
and popular tags have their own respective variable.

Note: Even 2D images will contain a 3D array, along with a fake slice thickness of 1 mm.

```python
import medicalimageanalysis as mia

path = r'/path/to/folder'

reader = mia.Reader(folder_path=path)
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
import medicalimageanalysis as mia

path = r'/path/to/folder'

reader = mia.Reader(folder_path=path)
reader.read_dicoms()

image = reader.images[0]

roi_names = list(image.rois.keys())
roi = image.rois[roi_names[0]]
contour_position = roi.contour_position

poi_names = list(image.pois.keys())
poi = image.rois[poi_names[0]]
point_position = poi.point_position

```
