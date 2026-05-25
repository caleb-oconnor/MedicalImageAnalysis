"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

3D Manufacturing Format (3MF) Mesh Reader and Voxelizer
======================================================

Description:
    This module provides functionality to parse .3mf archives—a modern 3D
    printing format—and integrate them into a medical imaging workflow.
    The reader extracts mesh geometry and vertex-level color data, converting
    it into a PyVista PolyData object and subsequently voxelizing it into
    an internal Image/ROI structure.

Key Features:
    1. **Archive Handling**: Unzips the 3MF package to access the underlying
       XML model descriptions and texture images.
    2. **Color Resolution**: Supports two 3MF color workflows:
        - Texture-based: Maps UV coordinates to pixel colors in texture files.
        - Basematerials: Resolves hex-coded display colors via property IDs.
    3. **Mesh Optimization**: Automatically decimated high-density meshes
       (targeting ~50k points) to maintain application performance.
    4. **Voxelization**: Converts mesh geometry into a binary mask to create
       a registered `Image` and `ROI` within the global `Data` state.

Structure:
    * ThreeMfReader:
        - load(): The primary execution pipeline (Parse -> Build -> Voxelize).
        - get_color(): Inner helper to resolve vertex colors based on mode.

Usage:
    >>> # Load an implant mesh and register it as an ROI named 'Implant'
    >>> reader = ThreeMfReader("path/to/implant.3mf", roi_name="Implant")
    >>> reader.load()

"""

import zipfile
from PIL import Image as pil_image
from PIL import ImageColor
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv

from ..structure.image import Image
from ..utils.creation import CreateImageFromMask
from ..utils.convert.contour import ModelToMask

from ..data import Data


class ThreeMfReader(object):
    """
    Reader for `.3mf` (3D Manufacturing Format) files.

    This class converts a `.3mf` file into:
    - A PyVista mesh (PolyData)
    - Optional ROI creation
    - Internal image/mask representation

    It supports:
    - Vertex/triangle mesh extraction
    - Vertex coloring from textures or basematerials
    - Mesh decimation
    - Conversion into internal image + ROI structures

    Parameters
    ----------
    file : str
        Path to the `.3mf` file.
    roi_name : str, optional
        Name of ROI to create from the mesh.

    Examples
    --------
    Basic usage::

        reader = ThreeMfReader("model.3mf", roi_name="Tumor")
        reader.load()
    """

    def __init__(self, file, roi_name=None):
        """
        Initialize the 3MF reader.

        Parameters
        ----------
        file : str
            Path to `.3mf` file.
        roi_name : str, optional
            ROI name to assign to generated mesh.
        """
        self.file = file
        self.roi_name = roi_name

    def load(self):
        """
        Load and parse a `.3mf` file.

        Steps performed:
        1. Extract XML model from archive
        2. Parse vertices and triangles
        3. Handle color information (texture or basematerials)
        4. Build PyVista mesh
        5. Decimate mesh
        6. Convert mesh to internal mask/image
        7. Register ROI in global `Data` structure
        """

        ns = {
            "3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
            "m":   "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
        }

        archive = zipfile.ZipFile(self.file, "r")
        root = ET.parse(archive.open("3D/3dmodel.model"))

        obj = root.findall("./3mf:resources/3mf:object", ns)[0]

        # --- Vertex extraction ---
        vertex_list = np.array([
            [float(v.get("x")), float(v.get("y")), float(v.get("z"))]
            for v in obj.findall(".//3mf:vertex", ns)
        ], dtype=float)

        triangles = obj.findall(".//3mf:triangle", ns)
        n_tris = len(triangles)

        faces = np.empty(4 * n_tris, dtype=int)
        vertex_colors = np.full((len(vertex_list), 3), 200, dtype=np.uint8)
        vertex_hit = np.zeros(len(vertex_list), dtype=bool)

        # --- Detect color mode ---
        tex_group = root.find(".//m:texture2dgroup", ns)
        basematerials = root.find(".//m:basematerials", ns)

        if tex_group is not None:
            color_mode = "texture"
            group_id = tex_group.get("id")

            tex_el = root.find(".//m:texture2d", ns)
            tex_path = tex_el.get("path").lstrip("/")

            texture_img = pil_image.open(archive.open(tex_path)).convert("RGB")
            tex_w, tex_h = texture_img.size
            tex_pixels = np.array(texture_img)

            uv_list = [
                (float(tc.get("u")), float(tc.get("v")))
                for tc in tex_group.findall("m:tex2coord", ns)
            ]

            def get_color(tri, vi, pkey):
                pindex = tri.get(pkey)
                if pindex is None:
                    return None
                u, v = uv_list[int(pindex)]
                px = int(np.clip(u, 0, 1) * (tex_w - 1))
                py = int(np.clip(1.0 - v, 0, 1) * (tex_h - 1))
                return tex_pixels[py, px]

        elif basematerials is not None:
            color_mode = "basematerials"

            color_map = {}
            for bm in root.findall(".//m:basematerials", ns):
                gid = bm.get("id")
                for idx, base in enumerate(bm.findall("m:base", ns)):
                    hex_color = base.get("displaycolor", "#C8C8C8")
                    rgb = np.array(
                        ImageColor.getcolor(hex_color, "RGB")[:3],
                        dtype=np.uint8
                    )
                    color_map[(gid, idx)] = rgb

            mesh_el = obj.find(".//3mf:mesh", ns)
            default_pid = (mesh_el or obj).get("pid")
            default_pindex = int((mesh_el or obj).get("pindex", "0"))

            def get_color(tri, vi, pkey):
                pid = tri.get("pid", default_pid)
                if pid is None:
                    return None
                pindex = int(tri.get(pkey, default_pindex))
                return color_map.get((pid, pindex))

        else:
            color_mode = None

        # --- Build mesh ---
        for ii, tri in enumerate(triangles):
            v1, v2, v3 = map(int, (tri.get("v1"), tri.get("v2"), tri.get("v3")))
            faces[ii * 4:(ii + 1) * 4] = [3, v1, v2, v3]

            if color_mode is None:
                continue

            if color_mode == "texture" and tri.get("pid") != group_id:
                continue

            for vi, pkey in zip([v1, v2, v3], ["p1", "p2", "p3"]):
                if not vertex_hit[vi]:
                    rgb = get_color(tri, vi, pkey)
                    if rgb is not None:
                        vertex_colors[vi] = rgb
                        vertex_hit[vi] = True

        mesh = pv.PolyData(vertex_list, faces)
        mesh["colors"] = vertex_colors

        decimate_mesh = mesh.decimate_pro(1 - (50000 / len(mesh.points)))

        # --- Convert mesh to image/mask ---
        image_name = f"CT {len(Data.image_list) + 1:02d}"

        model_to_mask = ModelToMask([decimate_mesh])
        mask = model_to_mask.mask

        new_image = CreateImageFromMask(
            mask,
            model_to_mask.origin,
            model_to_mask.spacing,
            image_name
        )

        Data.image[image_name] = Image(new_image)
        Data.image_list.append(image_name)

        # --- ROI creation ---
        Data.image[image_name].create_roi(
            name=self.roi_name,
            visible=False,
            filepath=self.file
        )

        Data.image[image_name].rois[self.roi_name].add_mesh(decimate_mesh)
        Data.image[image_name].rois[self.roi_name].color = [128, 128, 128]
        Data.image[image_name].rois[self.roi_name].multi_color = True

        Data.match_rois()
