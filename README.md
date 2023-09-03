
# MedicalImageConverter

*MedicalImageConverter* is a Python package for working with medical image files. It
lets the user read in images or ROIs (regions of interest), and converts them to 3D
numpy arrays. It also allows the user to input non-organized datasets (multiple images 
inside a single folder). Currently, this module only works for Dicom data with the hopes of expanding to 
other data formats in the future. 

The module currently imports 9 different modalites and RTSTRUCT files. The accepted
modalites are:
1. CT
2. MR
3. US
4. PT
5. MG
6. DX
7. NM
8. XA
9. CR

The CT and MR modalities have been tested extensively, along with their respective
ROIs. The other 7 modalities have been tested but only on a few datasets a piece.
For RTSTRUCTS, only those referencing CT and MR have been tested.

The images will be converted to Head-First-Supine (if not so already), and the 
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

## Example
The user sets a path to the folder containing the dicom files. If the user already
has each file path then they can continue to *DicomReader*, however if the user 
doesn't then use the *file_parsar* function. *file_parsar* will look through all 
the files inside the given file path and output a dictionary containing file paths 
for these types found:
1. Dicom (.dcm)
2. MetaHeader (.mhd)
3. Raw (.raw)
4. STL (.stl)

In the example below I used *file_parsar*, output the results into 
*file_dictionary*. Then the Dicom list was selected for the dictionary which was 
input into *DicomReader* class. Lastly, the data is then loaded in using 
*load_dicom*.

Note: The *exclude_files* allows the user not to get the file paths for any files
inside the list, generally the user will use an empty list. The 
*existing_image_info* is required when the user is trying to load in an RTSTRUCT 
file only, some tags are needed for the image it references to create a 3D volume
with the correct spacing.

```python
import MedicalImageConverter as mic

path = r'/path/to/folder'

exclude_files = []
existing_image_info = None
file_dictionary = mic.file_parsar(path, exclude_files)
dicom_reader = mic.DicomReader(file_dictionary['Dicom'], existing_image_info)
dicom_reader.load_dicom()
```

### Retrieve image information:
```python
image_data = dicom_reader.get_image_data()  # Returns a list of 3D arrays containing each image
image_info = dicom_reader.get_image_info()  # Returns a pandas dataframe containing important tag information
```

Tags in *image_info*:
<span style="font-size:.8em;">FilePath, SOPInstanceUID, PatientID,
PatientName, Modality, SeriesDescription, SeriesDate, SeriesTime, 
SeriesInstanceUID, SeriesNumber, AcquisitionNumber, SliceThickness,
PixelSpacing, Rows, Columns, ImagePositionPatient, Slices, DefaultWindow</span>

### Retrieve ROI information:
```python
roi_data = dicom_reader.get_roi_contour()  # Returns a list of lists containing each ROI contour per image
roi_info = dicom_reader.get_roi_info()  # Returns a pandas dataframe containing important tag information
```

Tags in *roi_info*:
<span style="font-size:.8em;">FilePath, RoiNames, PhysicalCoordinates,
ArrayCoordinates</span>

### Retrieve Sorted Files
```python
ds_images = dicom_reader.get_ds_images()  # Returns a list of dicom files sorted into each image  
```
