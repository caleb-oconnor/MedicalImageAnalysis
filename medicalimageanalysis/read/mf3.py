"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:

Functions:

"""

import os

import zipfile
from PIL import ImageColor
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv

from ..structure.image import Image
from ..utils.creation import CreateImageFromMask
from ..utils.conversion import ModelToMask

from ..data import Data


class ThreeMfReader(object):
    """
    Converts 3mf file to pyvista polydata mesh.
    """
    def __init__(self, reader, roi_names=None):
        self.reader = reader
        self.roi_names = roi_names

    def input_files(self, files):
        self.reader.files['3mf'] = files

    def load(self):
        for ii, file_path in enumerate(self.reader.files['3mf']):
            if len(self.roi_names) > 0 and self.roi_names[ii] is not None:
                roi_name = self.roi_names[ii]
            else:
                roi_name = self.reader.files['3mf'][ii].split('/')[-1].split('.3mf')[0]

            self.read(file_path, roi_name)

    def read(self, path, roi_name):
        """
        Loads in the 3mf file, gets the vertices/vertice colors/triangles and creates a polydata 3D model using pyvista.

        Returns
        -------

        """
        namespace = {"3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
                     "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"}

        archive = zipfile.ZipFile(path, "r")
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

        mesh = pv.PolyData(np.float64(np.asarray(vertex_list)), triangle_list[0, :].astype(int))
        mesh['colors'] = np.abs(255-vertices_color)

        decimate_mesh = mesh.decimate_pro(1 - (50000 / len(mesh.points)))

        image_name = 'CT ' + '0' + str(len(Data.image_list) + 1)

        model_to_mask = ModelToMask([decimate_mesh])
        mask = model_to_mask.mask

        new_image = CreateImageFromMask(mask, model_to_mask.origin, model_to_mask.spacing, image_name)
        Data.images[image_name] = Image(new_image)
        Data.image_list += [image_name]

        Data.images[image_name].create_roi(name=roi_name, visible=False, filepath=self.reader.files['3mf'])
        Data.images[image_name].rois[roi_name].add_mesh(decimate_mesh)
        Data.images[image_name].rois[roi_name].color = [128, 128, 128]
        Data.images[image_name].rois[roi_name].multi_color = True

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


