"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

3D Manufacturing Format (3MF) Mesh Reader and Voxelizer
=======================================================

Description:
    This module parses .3mf archives, a modern 3D printing format, and
    brings them into the imaging workflow. The reader extracts mesh
    geometry and per-vertex color, builds a PyVista PolyData mesh, and
    voxelizes it into a new Image that carries the mesh as an ROI.

Key Functionalities:
    1. **Archive Handling**: Opens the 3MF zip package to read the
       `3D/3dmodel.model` XML and any texture images it references.
    2. **Color Resolution**: Supports the two common 3MF color workflows:
        - Texture-based: UV coordinates sampled from a texture image.
        - Basematerials: hex `displaycolor` values looked up by property ID.
       Vertices without color stay light grey [200, 200, 200].
    3. **Mesh Optimization**: Decimates meshes above 50k points down to
       roughly 50k to keep rendering and voxelization fast; smaller meshes
       are kept as is.
    4. **Voxelization**: Converts the mesh into a binary mask, registers it
       as a new 'CT NN' image, and attaches the mesh as an ROI.

Architecture Note:
    Only the first `<object>` in the model is read. The result is written
    to the global `Data` registry, and `Data.match_rois()` is called so the
    new ROI name exists on every loaded image.

Usage:
    >>> from medicalimageanalysis.read.mf3 import ThreeMfReader
    >>> from medicalimageanalysis import Data
    >>> # Load an implant mesh and register it as an ROI named 'Implant'
    >>> reader = ThreeMfReader('C:/Data/implant.3mf', roi_name='Implant')
    >>> reader.load()
    >>> Data.image_list()

Examples
--------
>>> import medicalimageanalysis as mia
>>> from medicalimageanalysis import Data
>>> Data.clear()
>>> mia.read_3mf(file='C:/Data/pelvis_model.3mf', roi_name='Pelvis')
>>> Data.image_list()
['CT 01']
>>> roi = Data.image['CT 01'].roi['Pelvis']
>>> roi.color, roi.multi_color, roi.visible
([128, 128, 128], True, False)
"""

import zipfile
from pathlib import Path
from PIL import Image as pil_image
from PIL import ImageColor
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv

from ..structure.image import Image
from ..utils.creation import CreateImageFromMask
from ..utils.roi.discrete_converters import ModelToMask

from ..data import Data


class ThreeMfReader(object):
    """
    Reader for `.3mf` (3D Manufacturing Format) files.

    Converts one `.3mf` file into a decimated PyVista mesh with vertex
    colors, voxelizes it into a new image, and stores the mesh as an ROI on
    that image in the global `Data` registry.

    Parameters
    ----------
    file : str
        Path to the `.3mf` file.

    roi_name : str, optional
        Name of the ROI that holds the mesh. Defaults to the file stem
        (e.g. 'implant' for 'C:/Data/implant.3mf').

    Attributes
    ----------
    file : str
        Path to the `.3mf` file.

    roi_name : str
        ROI name used when registering the mesh (never None).

    Examples
    --------
    Load a colored tumor model and inspect the result:

    >>> from medicalimageanalysis.read.mf3 import ThreeMfReader
    >>> from medicalimageanalysis import Data
    >>> Data.clear()
    >>> reader = ThreeMfReader('C:/Data/tumor_model.3mf', roi_name='Tumor')
    >>> reader.load()
    >>> Data.image_list()
    ['CT 01']
    >>> list(Data.image['CT 01'].roi.keys())
    ['Tumor']

    The convenience wrapper does the same thing in one call:

    >>> import medicalimageanalysis as mia
    >>> mia.read_3mf(file='C:/Data/tumor_model.3mf', roi_name='Tumor')
    """

    def __init__(self, file, roi_name=None):
        """
        Store the file path and ROI name. Nothing is read until `load`.

        Parameters
        ----------
        file : str
            Path to `.3mf` file.

        roi_name : str, optional
            ROI name to assign to the generated mesh. Defaults to the file
            stem.

        Examples
        --------
        Constructing the reader is cheap and does not touch the file:

        >>> from medicalimageanalysis.read.mf3 import ThreeMfReader
        >>> reader = ThreeMfReader('C:/Data/mandible.3mf', roi_name='Mandible')
        >>> reader.file, reader.roi_name
        ('C:/Data/mandible.3mf', 'Mandible')

        Without a name the file stem is used:

        >>> ThreeMfReader('C:/Data/cutting_guide.3mf').roi_name
        'cutting_guide'
        """
        self.file = file
        self.roi_name = roi_name if roi_name is not None else Path(file).stem

    def load(self):
        """
        Parse the `.3mf` file and register it as a new image + ROI.

        Steps performed:
            1. Read `3D/3dmodel.model` from the archive.
            2. Parse vertices and triangles of the first object.
            3. Resolve vertex colors (texture or basematerials).
            4. Build the PyVista mesh and store colors in `mesh['colors']`.
            5. Decimate to roughly 50k points (only if larger).
            6. Voxelize into a mask image named 'CT NN'.
            7. Attach the mesh as an ROI and run `Data.match_rois()`.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the archive has no `3D/3dmodel.model` entry.

        IndexError
            If the model contains no `<object>`.

        Examples
        --------
        Load a mesh into an empty session:

        >>> from medicalimageanalysis.read.mf3 import ThreeMfReader
        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> ThreeMfReader('C:/Data/mandible.3mf', roi_name='Mandible').load()
        >>> Data.image_list()
        ['CT 01']

        The image name continues the session's numbering, and the new ROI
        is added (empty) to the images already loaded:

        >>> import medicalimageanalysis as mia
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> ThreeMfReader('C:/Data/guide.3mf', roi_name='CuttingGuide').load()
        >>> Data.image_list()
        ['CT 01', 'MR 02', 'CT 03']
        >>> 'CuttingGuide' in Data.image['MR 02'].roi
        True
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

        # Only reduce meshes above the target; smaller ones would need a negative reduction
        target_points = 50000
        if mesh.n_points > target_points:
            decimate_mesh = mesh.decimate_pro(1 - (target_points / mesh.n_points))
        else:
            decimate_mesh = mesh

        # --- Convert mesh to image/mask ---
        image_name = f"CT {len(Data.image_list()) + 1:02d}"

        model_to_mask = ModelToMask([decimate_mesh])
        mask = model_to_mask.mask

        new_image = CreateImageFromMask(
            mask,
            np.asarray(model_to_mask.origin),
            np.asarray(model_to_mask.spacing),
            image_name
        )

        Data.image[image_name] = Image(new_image)

        # --- ROI creation ---
        Data.image[image_name].create_roi(
            name=self.roi_name,
            visible=False,
            filepath=self.file
        )

        Data.image[image_name].roi[self.roi_name].add_mesh(decimate_mesh)
        Data.image[image_name].roi[self.roi_name].color = [128, 128, 128]
        Data.image[image_name].roi[self.roi_name].multi_color = True

        Data.match_rois()