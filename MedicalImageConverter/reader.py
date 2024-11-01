"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


ThreeMfReader:
    Converts 3mf file to pyvista polydata mesh.


Dicom Reader:
    Reads in dicom files and creates 2D or 3D image data sets, depending on the modality type. Also, reads in rtstruct
    files and output roi data in either contour position or pixel location.

    Fully tested modalities:
    - CT
    - RTSTRUCT

    Limited tested modalities:
    - MR
    - US
    - MG
    - DX
    - MG
    - NM
    - PT

    Final variables:
    image_data = a list of list, where each index is either a 2D/3D array depending on the modality
    image_info = dataframe(index=images, columns=tags), 23 of what I think are the most important user and image
                 information tags. For images with multiple slices only a tag for the first slice is kept for that
                 image. Meaning for a 100 slice CT I don't store Pixel Spacing 100 times, only the 100 filepaths and
                 SOPInstanceUID (needed for RTSTRUCTS) are saved.

    # Note all images are converted to being FFS, the positioning is corrected as well as the roi locations.
"""

import os
import copy
import time
import gdcm
import threading

import zipfile
from PIL import ImageColor
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyvista as pv
import pydicom as dicom
from pydicom.uid import generate_uid


class ThreeMfReader:
    """
    Converts 3mf file to pyvista polydata mesh.
    """
    def __init__(self, path, load=True):
        self.path = path
        self.mesh = None

        if load:
            self.load_3mf()

    def load_3mf(self):
        """
        Loads in the 3mf file, gets the vertices/vertice colors/triangles and creates a polydata 3D model using pyvista.

        Returns
        -------

        """
        namespace = {"3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
                     "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"}

        archive = zipfile.ZipFile(self.path, "r")
        root = ET.parse(archive.open("3D/3dmodel.model"))
        color_list = list()
        colors = root.findall('.//m:color', namespace)
        if colors:
            for color in colors:
                color_list.append(color.get("color", 0))

        obj = root.findall("./3mf:resources/3mf:object", namespace)[0]
        triangles = obj.findall(".//3mf:triangle", namespace)

        vertex_list = []
        for vertex in obj.findall(".//3mf:vertex", namespace):
            vertex_list.append([vertex.get("x"), vertex.get("y"), vertex.get("z")])

        triangle_list = np.empty((1, 4 * len(triangles)))
        vertices_color = np.zeros((len(vertex_list), 3))
        for ii, triangle in enumerate(triangles):
            v1 = int(triangle.get("v1"))
            v2 = int(triangle.get("v2"))
            v3 = int(triangle.get("v3"))
            tricolor = self.color_avg(color_list, (triangle.get("p1")), (triangle.get("p2")), (triangle.get("p3")))
            rgb_color = list(ImageColor.getcolor(tricolor, "RGB")[0:3])
            vertices_color[v1] = rgb_color
            vertices_color[v2] = rgb_color
            vertices_color[v3] = rgb_color
            triangle_list[0, ii * 4:(ii + 1) * 4] = [3, v1, v2, v3]

        self.mesh = pv.PolyData(np.float64(np.asarray(vertex_list)), triangle_list[0, :].astype(int))
        self.mesh['colors'] = np.abs(255-vertices_color)

    @staticmethod
    def color_avg(color_list, p1, p2, p3):
        """
        Get the average color from color list.

        Parameters
        ----------
        color_list
        p1
        p2
        p3

        Returns
        -------

        """
        p2rgb = None
        p3rgb = None

        p1hex = color_list[int(p1)]
        value = p1hex.lstrip('#')
        lv = len(value)
        p1rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

        if isinstance(p2, int):
            p2hex = color_list[int(p2)]
            value = p2hex.lstrip('#')
            lv = len(value)
            p2rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

        if isinstance(p3, int):
            p3hex = color_list[int(p3)]
            value = p3hex.lstrip('#')
            lv = len(value)
            p3rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

        if p2rgb is not None and p3rgb is not None:
            rgbAvg = np.average(np.array(p1rgb), np.array(p2rgb), np.array(p3rgb))

        elif p2rgb is not None:
            rgbAvg = np.average(np.array(p1rgb), np.array(p2rgb))

        else:
            rgbAvg = p1rgb

        hexAvg = '#%02x%02x%02x' % (rgbAvg[0], rgbAvg[1], rgbAvg[2])

        return hexAvg

    def downsample_mesh(self, points):
        ds_mesh = self.mesh.decimate(1 - (points / len(self.mesh.points)))
        ds_points = np.asarray(ds_mesh.points)


def thread_process_dicom(path):
    try:
        datasets = dicom.dcmread(str(path))
    except:
        datasets = []

    return datasets


class DicomReader:
    """
    Reads in dicom files and creates 2D or 3D image data sets, depending on the modality type. Also, reads in rtstruct
    files and output roi data in either contour position or pixel location.

    Fully tested modalities:
    - CT
    - RTSTRUCT

    Limited tested modalities:
    - MR
    - US
    - MG
    - DX
    - MG
    - NM
    - PT

    Final variables:
    image_data = a list of list, where each index is either a 2D/3D array depending on the modality
    image_info = dataframe(index=images, columns=tags), 23 of what I think are the most important user and image
                 information tags. For images with multiple slices only a tag for the first slice is kept for that
                 image. Meaning for a 100 slice CT I don't store Pixel Spacing 100 times, only the 100 filepaths and
                 SOPInstanceUID (needed for RTSTRUCTS) are saved.

    # Note all images are converted to being FFS, the positioning is corrected as well as the roi locations.

    """
    def __init__(self, dicom_files, existing_image_info=None, only_load_roi_names=None):
        """

        Parameters
        ----------
        dicom_files - list of all the dicom paths
        existing_image_info - dataframe of image_info same as structure below (for when only loading RTSTRUCTS)
        only_load_roi_names - list of Roi names that will only be uploaded (so total segementator won't load all 100
                              rois)
        """
        self.dicom_files = dicom_files
        self.existing_image_info = existing_image_info
        self.only_load_roi_names = only_load_roi_names

        self.ds = []
        self.ds_images = []
        self.ds_dictionary = dict.fromkeys(['CT', 'MR', 'PT', 'US', 'DX', 'MG', 'NM', 'XA', 'CR', 'RTSTRUCT'])
        self.rt_df = pd.DataFrame(columns=['FilePath', 'SeriesInstanceUID', 'RoiSOP', 'RoiNames'])

        keep_tags = ['FilePath', 'SOPInstanceUID', 'PatientID', 'PatientName', 'Modality',
                     'SeriesDescription', 'SeriesDate', 'SeriesTime', 'SeriesInstanceUID', 'SeriesNumber',
                     'AcquisitionNumber', 'FrameOfReferenceUID', 'SliceThickness', 'PixelSpacing', 'Rows', 'Columns',
                     'PatientPosition', 'ImagePositionPatient', 'ImageOrientationPatient', 'ImageMatrix', 'ImagePlane',
                     'Slices', 'SkippedSlice', 'DefaultWindow', 'FullWindow', 'Unverified']
        self.image_info = pd.DataFrame(columns=keep_tags)
        self.image_data = []

        self.roi_info = pd.DataFrame(columns=['FilePath', 'RoiNames'])
        self.roi_contour = []
        self.roi_pixel_position = []

        self.contours = []

    def add_dicom_extension(self):
        """
        Will add .dcm extension to any file inside self.dicom_files that doesn't have an extension
        Returns
        -------

        """
        for ii, name in enumerate(self.dicom_files):
            a, b = os.path.splitext(name)
            if not b:
                self.dicom_files[ii] = name + '.dcm'

    def load_dicom(self, display_time=True):
        """
        Runs through all the base functions required to load in images/rois.

        Parameters
        ----------
        display_time - True if user wants to print load time

        Returns
        -------

        """
        t1 = time.time()
        self.read()
        self.separate_modalities()
        self.separate_images()
        self.separate_rt_images()
        self.standard_useful_tags()
        self.convert_images()
        self.fix_orientation()
        self.separate_contours()
        t2 = time.time()
        if display_time:
            print('Dicom Read Time: ', t2 - t1)

    def read(self):
        """
        Uses the threading to read in the data.

        self.ds -> contains tag/data from pydicom read-in

        Returns
        -------

        """
        threads = []

        def read_file_thread(file_path):
            self.ds.append(thread_process_dicom(file_path))

        for file_path in self.dicom_files:
            thread = threading.Thread(target=read_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def separate_modalities(self):
        """
        Separate read in files by their modality.

        ds_dictionary - dictionary of different modalities
        Returns
        -------

        """
        for modality in list(self.ds_dictionary.keys()):
            ds_modality = [d for d in self.ds if d['Modality'].value == modality]
            self.ds_dictionary[modality] = [ds_mod for ds_mod in ds_modality]

    def separate_images(self):
        """
        Runs through each modality and if multiple images exist per modality they are separated.

        Returns
        -------

        """
        for modality in list(self.ds_dictionary.keys()):
            if len(self.ds_dictionary[modality]) > 0 and modality not in ['RTSTRUCT', 'US', 'DX']:
                sorting_tags = np.asarray([[img['SeriesInstanceUID'].value, img['AcquisitionNumber'].value]
                                           if 'AcquisitionNumber' in img and img['AcquisitionNumber'].value is not None
                                           else [img['SeriesInstanceUID'].value, 1]
                                           for img in self.ds_dictionary[modality]])

                unique_tags = np.unique(sorting_tags, axis=0)
                for tag in unique_tags:
                    sorted_idx = np.where((sorting_tags[:, 0] == tag[0]) & (sorting_tags[:, 1] == tag[1]))
                    image_tags = [self.ds_dictionary[modality][idx] for idx in sorted_idx[0]]

                    if 'ImageOrientationPatient' in image_tags[0] and 'ImagePositionPatient' in image_tags[0]:
                        orientation = image_tags[0]['ImageOrientationPatient'].value
                        position_tags = np.asarray([t['ImagePositionPatient'].value for t in image_tags])

                        x = np.abs(orientation[0]) + np.abs(orientation[3])
                        y = np.abs(orientation[1]) + np.abs(orientation[4])
                        z = np.abs(orientation[2]) + np.abs(orientation[5])

                        if x < y and x < z:
                            slice_idx = np.argsort(position_tags[:, 0])
                        elif y < x and y < z:
                            slice_idx = np.argsort(position_tags[:, 1])
                        else:
                            slice_idx = np.argsort(position_tags[:, 2])

                        self.ds_images.append([image_tags[idx] for idx in slice_idx])

                    else:
                        self.ds_images.append(image_tags)

            elif len(self.ds_dictionary[modality]) > 0 and modality in ['US', 'DX']:
                for image in self.ds_dictionary[modality]:
                    self.ds_images.append([image])

    def separate_rt_images(self):
        """
        Loops through all RTSTRUCT files found. Some required information that will be used later in making the contours
        is pulled:
            SeriesInstanceUID
            RoiNames
            RoiSOP - this will be used to determine what slice the contour exist on
        Returns
        -------

        """
        for ii, rt_ds in enumerate(self.ds_dictionary['RTSTRUCT']):
            ref = rt_ds.ReferencedFrameOfReferenceSequence
            series_uid = ref[0]['RTReferencedStudySequence'][0]['RTReferencedSeriesSequence'][0][
                'SeriesInstanceUID'].value

            roi_sop = []
            for contour_list in rt_ds.ROIContourSequence:
                points = [c.NumberOfContourPoints for c in contour_list['ContourSequence']]
                if np.sum(np.asarray(points)) > 3:
                    roi_sop.append(contour_list['ContourSequence'][0]
                                   ['ContourImageSequence'][0]['ReferencedSOPInstanceUID'].value)

            self.rt_df.at[ii, 'FilePath'] = rt_ds.filename
            self.rt_df.at[ii, 'SeriesInstanceUID'] = series_uid
            self.rt_df.at[ii, 'RoiSOP'] = roi_sop
            self.rt_df.at[ii, 'RoiNames'] = [s.ROIName for s in rt_ds.StructureSetROISequence]

    def standard_useful_tags(self):
        """
        Important tags for each image that I use in DRAGON:
        Returns
        -------

        """
        for ii, image in enumerate(self.ds_images):
            for t in list(self.image_info.keys()):
                if t == 'FilePath':
                    self.image_info.at[ii, t] = [img.filename for img in image]

                elif t == 'SOPInstanceUID':
                    self.image_info.at[ii, t] = [img.SOPInstanceUID for img in image]

                elif t == 'PixelSpacing':
                    self.find_pixel_spacing(image[0], ii)

                elif t == 'ImagePositionPatient':
                    if image[0].Modality in ['US', 'CR', 'DX', 'MG', 'NM', 'XA']:
                        self.image_info.at[ii, t] = [0, 0, 0]
                    else:
                        self.image_info.at[ii, t] = image[0].ImagePositionPatient

                elif t == 'SliceThickness':
                    if len(image) > 1:
                        thickness = (np.asarray(image[1]['ImagePositionPatient'].value[2]).astype(float) -
                                     np.asarray(image[0]['ImagePositionPatient'].value[2]).astype(float))
                    elif t in image[0]:
                        thickness = np.asarray(image[0]['SliceThickness'].value).astype(float)
                    else:
                        thickness = 1

                    self.image_info.at[ii, t] = thickness

                elif t == 'Slices':
                    self.image_info.at[ii, t] = len(image)

                elif t == 'DefaultWindow':
                    if (0x0028, 0x1050) in image[0] and (0x0028, 0x1051) in image[0]:
                        center = image[0].WindowCenter
                        width = image[0].WindowWidth
                        if not isinstance(center, float):
                            center = center[0]

                        if not isinstance(width, float):
                            width = width[0]

                        self.image_info.at[ii, t] = [int(center), int(np.round(width/2))]

                    elif image[0].Modality == 'US':
                        self.image_info.at[ii, t] = [128, 128]

                    else:
                        self.image_info.at[ii, t] = None

                elif t == 'FullWindow':
                    self.image_info.at[ii, t] = None

                elif t == 'ImageMatrix':
                    pass

                elif t == 'SkippedSlice':
                    pass

                elif t == 'ImagePlane':
                    if image[0].Modality in ['US', 'CR', 'DX', 'MG', 'NM', 'XA']:
                        self.image_info.at[ii, t] = 'Axial'
                    else:
                        orientation = image[0]['ImageOrientationPatient'].value
                        x = np.abs(orientation[0]) + np.abs(orientation[3])
                        y = np.abs(orientation[1]) + np.abs(orientation[4])
                        z = np.abs(orientation[2]) + np.abs(orientation[5])

                        if x < y and x < z:
                            self.image_info.at[ii, t] = 'Sagittal'
                        elif y < x and y < z:
                            self.image_info.at[ii, t] = 'Coronal'
                        else:
                            self.image_info.at[ii, t] = 'Axial'

                elif t == 'Unverified':
                    pass

                else:
                    if t in image[0]:
                        self.image_info.at[ii, t] = image[0][t].value

                    else:
                        if t == 'SeriesDate':
                            if 'StudyDate' in image[0]:
                                self.image_info.at[ii, t] = image[0]['StudyDate'].value
                            else:
                                self.image_info.at[ii, t] = '0'

                        elif t == 'SeriesTime':
                            if 'StudyTime' in image[0]:
                                self.image_info.at[ii, t] = image[0]['StudyTime'].value
                            else:
                                self.image_info.at[ii, t] = '00000'

                        elif t == 'SeriesDescription':
                            self.image_info.at[ii, t] = 'None'

                        elif t == 'FrameOfReferenceUID':
                            if 'FrameOfReferenceUID' in image[0]:
                                self.image_info.at[ii, t] = image[0]['FrameOfReferenceUID'].value
                            else:
                                self.image_info.at[ii, t] = generate_uid()

    def find_pixel_spacing(self, image, ii):
        spacing = 'PixelSpacing'
        if image.Modality == 'US':
            if 'SequenceOfUltrasoundRegions' in image:
                if 'PhysicalDeltaX' in image.SequenceOfUltrasoundRegions[0]:
                    self.image_info.at[ii, spacing] = [
                        10 * np.round(image.SequenceOfUltrasoundRegions[0].PhysicalDeltaX, 4),
                        10 * np.round(image.SequenceOfUltrasoundRegions[0].PhysicalDeltaY, 4)]
                else:
                    self.image_info.at[ii, spacing] = [1, 1]
            else:
                self.image_info.at[ii, spacing] = [1, 1]

        elif image.Modality in ['DX', 'XA']:
            self.image_info.at[ii, spacing] = image.ImagerPixelSpacing

        elif 'PixelSpacing' in image:
            self.image_info.at[ii, spacing] = image.PixelSpacing

        elif 'ContributingSourcesSequence' in image:
            sequence = 'ContributingSourcesSequence'
            if 'DetectorElementSpacing' in image[sequence][0]:
                self.image_info.at[ii, spacing] = image[sequence][0]['DetectorElementSpacing'].value

        elif 'PerFrameFunctionalGroupsSequence' in image:
            sequence = 'PerFrameFunctionalGroupsSequence'
            if 'PixelMeasuresSequence' in image[sequence][0]:
                self.image_info.at[ii, spacing] = image[sequence][0]['PixelMeasuresSequence'][0]['PixelSpacing'].value

        else:
            self.image_info.at[ii, spacing] = [1, 1]

    def convert_images(self):
        """
        Gets the 2D slice for each image and combines them into a 3D array per each image. Uses the RescaleIntercept
        and RescaleSlope to adjust the HU.

        The US is a different story. The image was saved as an RGB value, which also contained like metadata and
        patient information embedded in the image itself. Luckily there was a simple way to get the actual US out, and
        that was using the fact that when all three RGB values are the same thing it corresponds to the image (this
        pulls some additional none image stuff but not nearly as bad). The quickest way I thought would be to find the
        standard deviation of all three values and if it is zero then it is a keeper.

        Sometimes the images are in a shape [1, 10, 512, 512] meaning 10 "slices" by 512x512 array. Not sure why the 1
        is there, so it checks if the shape is 4 and if so it only saves the image as a [10, 512, 512]
        Returns
        -------

        """
        for ii, image in enumerate(self.ds_images):
            image_slices = []
            if self.image_info.at[ii, 'Modality'] in ['CT', 'MR', 'PT', 'MG', 'NM', 'XA', 'CR']:
                for slice_ in image:
                    if (0x0028, 0x1052) in slice_:
                        intercept = slice_.RescaleIntercept
                    else:
                        intercept = 0

                    if (0x0028, 0x1053) in slice_:
                        slope = slice_.RescaleSlope
                    else:
                        slope = 1

                    image_slices.append(((slice_.pixel_array*slope)+intercept).astype('int16'))

            elif self.image_info.at[ii, 'Modality'] == 'DX':
                if (0x2050, 0x0020) in image[0]:
                    if image[0].PresentationLUTShape == 'INVERSE':
                        hold_array = image[0].pixel_array.astype('int16')
                        image_slices.append(16383 - hold_array)
                        self.image_info.at[ii, 'DefaultWindow'][0] = 16383 - self.image_info.at[ii, 'DefaultWindow'][0]
                    else:
                        image_slices.append(image[0].pixel_array.astype('int16'))
                else:
                    image_slices.append(image[0].pixel_array.astype('int16'))

            elif self.image_info.at[ii, 'Modality'] == 'US':
                if len(image) == 1:
                    us_data = image[0].pixel_array
                    if len(us_data.shape) == 3:
                        us_binary = (1 * (np.std(us_data, axis=2) == 0) == 1)
                        image_slices = (us_binary * us_data[:, :, 0]).astype('uint8')

                    else:
                        us_binary = (1 * (np.std(us_data, axis=3) == 0) == 1)
                        image_slices = (us_binary * us_data[:, :, :, 0]).astype('uint8')
                else:
                    print('Need to finish')

            image_hold = np.asarray(image_slices)
            if len(image_hold.shape) > 3:
                self.image_data.append(image_hold[0])
                self.image_info.at[ii, 'Slices'] = image_hold[0].shape[0]
            else:
                self.image_data.append(image_hold)

            image_min = np.min(self.image_data[-1])
            image_max = np.max(self.image_data[-1])
            self.image_info.at[ii, 'FullWindow'] = [image_min, image_max]

    def fix_orientation(self, convert_axial=True):
        """
        Corrects position for orientation fix. I force everything to be FFS so for non-FFS images the corner position
        is incorrect, below corrects for the position using the Pixel Spacing and Orientation Matrix
        Returns
        -------

        """
        for ii, image in enumerate(self.image_data):
            if self.image_info.at[ii, 'Modality'] in ['US', 'CR', 'DX', 'XA']:
                self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
                self.image_info.at[ii, 'Unverified'] = True

            elif self.image_info.at[ii, 'Modality'] == 'NM':
                self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
                self.image_info.at[ii, 'Unverified'] = True

            elif self.image_info.at[ii, 'Modality'] == 'MG':
                self.image_info.at[ii, 'Unverified'] = True
                if self.image_info.at[ii, 'Slices'] == 1:
                    self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                    self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]

                else:
                    if 'SharedFunctionalGroupsSequence' in self.ds_images[ii][0]:
                        sequence = 'SharedFunctionalGroupsSequence'
                        if 'PlaneOrientationSequence' in self.ds_images[ii][0][sequence][0]:
                            self.image_info.at[ii, 'ImageOrientationPatient'] = self.ds_images[ii][0][sequence][0]['PlaneOrientationSequence'][0]['ImageOrientationPatient'].value
                            self.compute_image_matrix(ii)
                        else:
                            self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                            self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
                    else:
                        self.image_info.at[ii, 'ImageMatrix'] = np.identity(4, dtype=np.float32)
                        self.image_info.at[ii, 'ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]

            else:
                if self.image_info.at[ii, 'PatientPosition']:
                    position = self.image_info.at[ii, 'PatientPosition']
                    rows = self.image_info.at[ii, 'Rows']
                    columns = self.image_info.at[ii, 'Columns']
                    spacing = self.image_info.at[ii, 'PixelSpacing']
                    coordinates = self.image_info.at[ii, 'ImagePositionPatient']
                    orientation = np.asarray(self.image_info.at[ii, 'ImageOrientationPatient'])

                    if self.image_info.at[ii, 'ImagePlane'] == 'Axial':
                        self.fix_axial(position, rows, columns, spacing, coordinates, orientation)

                    elif self.image_info.at[ii, 'ImagePlane'] == 'Sagittal':
                        self.fix_sagittal(position, rows, columns, spacing, coordinates, orientation)

                    self.compute_image_matrix(ii)

    def fix_axial(self, position, rows, columns, spacing, coordinates, orientation):
        if position in ['HFDR', 'FFDR']:
            self.image_data[ii] = np.rot90(image, 3, (1, 2))

            new_coordinates = np.double(coordinates[0]) - spacing[0] * (columns - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][0] = new_coordinates
            self.image_info.at[ii, 'ImageOrientationPatient'] = [-orientation[3],
                                                                 -orientation[4],
                                                                 -orientation[5],
                                                                 orientation[0],
                                                                 orientation[1],
                                                                 orientation[2]]
        elif position in ['HFP', 'FFP']:
            self.image_data[ii] = np.rot90(image, 2, (1, 2))

            new_coordinates = np.double(coordinates[0]) - spacing[0] * (columns - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][0] = new_coordinates

            new_coordinates = np.double(coordinates[1]) - spacing[1] * (rows - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][1] = new_coordinates
            self.image_info.at[ii, 'ImageOrientationPatient'] = [-orientation[0],
                                                                 -orientation[1],
                                                                 -orientation[2],
                                                                 -orientation[3],
                                                                 -orientation[4],
                                                                 -orientation[5]]
        elif position in ['HFDL', 'FFDL']:
            self.image_data[ii] = np.rot90(image, 1, (1, 2))

            new_coordinates = np.double(coordinates[1]) - spacing[1] * (rows - 1)
            self.image_info.at[ii, 'ImagePositionPatient'][1] = new_coordinates
            self.image_info.at[ii, 'ImageOrientationPatient'] = [orientation[3],
                                                                 orientation[4],
                                                                 orientation[5],
                                                                 -orientation[0],
                                                                 -orientation[1],
                                                                 -orientation[2]]

        if self.image_info.at[ii, 'ImageOrientationPatient'][0] < 0 or self.image_info.at[ii, 'ImageOrientationPatient'][5] < 0:
            self.image_info.at[ii, 'Unverified'] = True
        self.compute_image_matrix(ii)

    def fix_sagittal(self, position, rows, columns, spacing, coordinates, orientation):
        b = copy.deepcopy(self.image_data[ii])
        b = np.swapaxes(b, 0, 1)
        b = np.swapaxes(b, 1, 2)
        self.image_data[ii] = np.flip(b, axis=0)

    def compute_image_matrix(self, ii):
        """
        Computes the image rotation matrix, often seen in MR images where the image is tilted.


        Returns
        -------

        """
        row_direction = np.array(self.image_info.at[ii, 'ImageOrientationPatient'][:3])
        column_direction = np.array(self.image_info.at[ii, 'ImageOrientationPatient'][3:])
        translation_offset = np.asarray(self.image_info.at[ii, 'ImagePositionPatient'])

        # noinspection PyUnreachableCode
        slice_direction = np.cross(row_direction, column_direction)
        if len(self.ds_images[ii]) > 1:
            first = np.dot(slice_direction, self.ds_images[ii][0].ImagePositionPatient)
            second = np.dot(slice_direction, self.ds_images[ii][1].ImagePositionPatient)
            last = np.dot(slice_direction, self.ds_images[ii][-1].ImagePositionPatient)
            first_last_spacing = np.asarray((last - first) / (self.image_info.at[ii, 'Slices'] - 1))
            if np.abs((second - first) - first_last_spacing) > 0.01:
                self.find_skipped_slices(slice_direction, ii)
                slice_spacing = second - first
            else:
                slice_spacing = np.asarray((last - first) / (self.image_info.at[ii, 'Slices'] - 1))

            self.image_info.at[ii, 'SliceThickness'] = slice_spacing
            if slice_spacing < 0 and self.image_info.at[ii, 'ImagePlane']:
                self.image_info.at[ii, 'ImagePositionPatient'] = self.ds_images[ii][-1].ImagePositionPatient

        mat = np.identity(4, dtype=np.float32)
        mat[0, :3] = row_direction
        mat[1, :3] = column_direction
        mat[2, :3] = slice_direction
        mat[0:3, 3] = -translation_offset

        self.image_info.at[ii, 'ImageMatrix'] = mat

    def separate_contours(self):
        """
        existing_image_info is required if the users only loads a RTSTRUCT file, this is needed to match contours with
        the image they correspond to.

        It is pretty gross after that. For a given ROI each contour is read-in, matched with their image, then combined
        all the slices of each contour into their own numpy array.

        Returns
        -------

        """
        info = self.image_info
        if self.existing_image_info is not None:
            if len(list(info.index)) > 0:
                print('fix')
            else:
                info = self.existing_image_info

        index_list = list(info.index)
        for ii in range(len(info.index)):
            img_sop = info.at[index_list[ii], 'SOPInstanceUID']
            img_series = info.at[index_list[ii], 'SeriesInstanceUID']

            image_contour_list = []
            roi_names = []
            roi_filepaths = []
            for jj in range(len(self.rt_df.index)):
                if img_series == self.rt_df.at[jj, 'SeriesInstanceUID'] and self.rt_df.at[jj, 'RoiSOP'][0] in img_sop:
                    roi_sequence = self.ds_dictionary['RTSTRUCT'][jj].ROIContourSequence
                    for kk, sequence in enumerate(roi_sequence):
                        contour_list = []
                        if not self.only_load_roi_names or self.rt_df.RoiNames[jj][kk] in self.only_load_roi_names:
                            for c in sequence.ContourSequence:
                                if int(c.NumberOfContourPoints) > 1:
                                    contour_hold = np.round(np.array(c['ContourData'].value), 3)
                                    contour = contour_hold.reshape(int(len(contour_hold) / 3), 3)
                                    contour_list.append(contour)

                            if len(contour_list) > 0:
                                image_contour_list.append(contour_list)
                                roi_filepaths.append(self.rt_df.at[jj, 'FilePath'])
                                roi_names.append(self.rt_df.RoiNames[jj][kk])

            if len(roi_names) > 0:
                if not np.isnan(self.image_info.at[ii, 'SkippedSlice']):
                    image_contour_list = self.calculate_skipped_contours(ii, image_contour_list)
                self.roi_contour.append(image_contour_list)
                self.roi_info.at[ii, 'FilePath'] = roi_filepaths
                self.roi_info.at[ii, 'RoiNames'] = roi_names
            else:
                self.roi_contour.append([])
                self.roi_info.at[ii, 'FilePath'] = None
                self.roi_info.at[ii, 'RoiNames'] = None

    def find_skipped_slices(self, slice_direction, ii):
        image = self.ds_images[ii]
        base_spacing = None
        for jj in range(len(image) - 1):
            position_1 = np.dot(slice_direction, image[jj].ImagePositionPatient)
            position_2 = np.dot(slice_direction, image[jj + 1].ImagePositionPatient)
            if jj == 0:
                base_spacing = position_2 - position_1
            if jj > 0 and np.abs(base_spacing - (position_2 - position_1)) > 0.01:
                self.image_info.at[ii, 'SkippedSlice'] = jj + 1
                self.image_info.at[ii, 'Slices'] = len(image) + 1

                hold_data = copy.deepcopy(self.image_data[ii])
                interpolate_slice = np.mean(self.image_data[ii][jj:jj + 2, :, :], axis=0).astype(np.int16)
                self.image_data[ii] = np.insert(hold_data,
                                                self.image_info.at[ii, 'SkippedSlice'],
                                                interpolate_slice,
                                                axis=0)

    def calculate_skipped_contours(self, ii, image_contours):
        thickness = self.image_info.at[ii, 'SliceThickness']
        skipped = self.image_info.at[ii, 'SkippedSlice']
        z_positions = np.asarray([self.ds_images[ii][skipped - 1].ImagePositionPatient[2],
                                  self.ds_images[ii][skipped].ImagePositionPatient[2]])

        for jj, roi_contour in enumerate(image_contours):
            if len(roi_contour) > 1:
                for kk in range(len(roi_contour) - 1):
                    position_1 = roi_contour[kk][0][2]
                    position_2 = roi_contour[kk + 1][0][2]
                    if position_1 == z_positions[0] and position_2 == z_positions[1]:
                        roi_copy = copy.deepcopy(roi_contour)
                        roi_copy.insert(jj + 1, copy.deepcopy(roi_copy[kk]))
                        roi_copy[jj + 1][:, 2] = position_1 + thickness
                        self.image_info.at[ii, 'SOPInstanceUID'].insert(jj + 1, generate_uid())
                        image_contours[jj] = roi_copy

        return image_contours

    def convert_to_axial(self):
        for ii in range(len(self.ds_images)):
            if self.image_info.at[ii, 'ImagePlane'] == 'Coronal':
                b = copy.deepcopy(self.image_data[ii])
                self.image_data[ii] = np.flip(b, axis=0)

            elif self.image_info.at[ii, 'ImagePlane'] == 'Sagittal':
                b = copy.deepcopy(self.image_data[ii])
                b = np.swapaxes(b, 0, 1)
                b = np.swapaxes(b, 1, 2)
                self.image_data[ii] = np.flip(b, axis=0)
        #
        # b = np.swapaxes(self.image_data[2], 0, 1)
        # b = np.swapaxes(b, 1, 2)
        # c = np.flip(b, axis=0)
        print(1)

    def get_image_info(self):
        return self.image_info

    def get_image_data(self):
        return self.image_data

    def get_roi_contour(self):
        return self.roi_contour

    def get_roi_info(self):
        return self.roi_info

    def get_ds_images(self):
        return self.ds_images


if __name__ == '__main__':
    pass
