"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org


Description:

Functions:

"""

import zipfile
from PIL import Image as pil_image
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
    def __init__(self, file, roi_name=None):
        self.file = file
        self.roi_name = roi_name

    def load(self):
        ns = {
            "3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
            "m":   "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
        }

        archive = zipfile.ZipFile(self.file, "r")
        root = ET.parse(archive.open("3D/3dmodel.model"))

        obj = root.findall("./3mf:resources/3mf:object", ns)[0]

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
            group_id = basematerials.get("id")

            color_map = {}  # (group_id, index) -> RGB
            for bm in root.findall(".//m:basematerials", ns):
                gid = bm.get("id")
                for idx, base in enumerate(bm.findall("m:base", ns)):
                    hex_color = base.get("displaycolor", "#C8C8C8")
                    rgb = np.array(ImageColor.getcolor(hex_color, "RGB")[:3], dtype=np.uint8)
                    color_map[(gid, idx)] = rgb

            mesh_el = obj.find(".//3mf:mesh", ns)
            default_pid    = (mesh_el or obj).get("pid")
            default_pindex = int((mesh_el or obj).get("pindex", "0"))

            def get_color(tri, vi, pkey):
                pid = tri.get("pid", default_pid)
                if pid is None:
                    return None
                pindex = int(tri.get(pkey, default_pindex))
                return color_map.get((pid, pindex))

        else:
            color_mode = None  # no color info found

        # --- Build faces and assign vertex colors ---
        for ii, tri in enumerate(triangles):
            v1, v2, v3 = int(tri.get("v1")), int(tri.get("v2")), int(tri.get("v3"))
            faces[ii*4:(ii+1)*4] = [3, v1, v2, v3]

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

        image_name = 'CT ' + '0' + str(len(Data.image_list) + 1)

        model_to_mask = ModelToMask([decimate_mesh])
        mask = model_to_mask.mask

        new_image = CreateImageFromMask(mask, model_to_mask.origin, model_to_mask.spacing, image_name)
        Data.image[image_name] = Image(new_image)
        Data.image_list += [image_name]

        Data.image[image_name].create_roi(name=self.roi_name, visible=False, filepath=self.file)
        Data.image[image_name].rois[self.roi_name].add_mesh(decimate_mesh)
        Data.image[image_name].rois[self.roi_name].color = [128, 128, 128]
        Data.image[image_name].rois[self.roi_name].multi_color = True
        Data.match_rois()
