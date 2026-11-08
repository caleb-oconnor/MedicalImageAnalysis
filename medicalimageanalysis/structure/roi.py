"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Region of Interest (ROI) Structure
==================================

Description:
    This module holds one ROI of an image: its planar contours (world and
    pixel), the closed surface mesh built from them, and the conversions
    between contours, masks and meshes. It also slices the mesh along any
    displayed plane, including oblique views and rigidly registered images,
    with an LRU cache so scrolling back over a slice is free.

Supported Inputs:
    * World contours: list of (N, 3) arrays in mm, x-y-z
    * Pixel contours: list of (N, 3) arrays of fractional pixel indices,
      x-y-z (closing point repeated at the end)
    * Masks: (z, y, x) arrays matching `image.dimensions`
    * Meshes: closed PyVista PolyData surfaces in mm

Key Functionalities:
    1. **Contours**: `update_position` / `update_pixel` replace the contours
       and rebuild the mesh; `compute_contour` returns the loops on one
       slice index.
    2. **Mesh**: `create_mesh` (smooth), `create_discrete_mesh` (voxel
       steps), and the `create_display_mesh` / `create_decimate_mesh` /
       `create_cluster_mesh` refinements.
    3. **Mask**: `compute_mask` / `create_sitk_mask` rasterize the contours;
       `convert_mask` goes back from a mask to contours and a mesh.
    4. **Slicing**: `compute_mesh_slice` cuts the mesh along an
       axial / sagittal / coronal or oblique view, in world or pixel units.

Coordinate Note:
    World points, `image.origin` and `image.spacing` are x-y-z. Pixel
    contours are x-y-z indices, so the axial slice index is column 2. Masks
    and `image.dimensions` are z-y-x, matching the image array.

Cache Note:
    Mesh slices are cached per ROI. Replacing `roi.mesh` (any setter,
    `update_mesh`, `create_mesh`, ...) clears the cache. Editing the mesh in
    place does not; call `clear_slice_cache` after that.

Usage:
    >>> import medicalimageanalysis as mia
    >>> files = mia.file_parser(folder_path='C:/Data/Patient01')
    >>> mia.read_dicoms(file_list=files['Dicom'])
    >>> image = mia.Data.image['CT 01']
    >>> # Contours of the axial slice currently shown, in display pixels
    >>> roi = image.roi['Liver']
    >>> loops, _ = roi.compute_mesh_slice(origin=image.display.position['Axial'], plane='Axial',
    ...                                   return_pixel=True)

Examples
--------
>>> import numpy as np
>>> import medicalimageanalysis as mia
>>> image = mia.Data.image['CT 01']
>>> image.create_roi('Box', color=[255, 0, 0])
>>> roi = image.roi['Box']
>>> # 40 x 40 mm square drawn on three axial slices
>>> z0 = image.origin[2] + 40 * image.spacing[2]
>>> square = np.array([[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], dtype=float)
>>> contours = [square + [0, 0, z0 + k * image.spacing[2]] for k in range(3)]
>>> roi.update_position(contours, plane='Axial')
>>> roi.mesh is not None
True
>>> mask = roi.compute_mask()
>>> mask.shape == tuple(image.dimensions)
True
"""

import random
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk

from ..utils.roi.discrete_converters import ContourToDiscreteMesh, MaskToContour
from ..utils.mesh.slicer import MeshSlicer
from ..utils.mesh.surface import Refinement, compute_components
from ..utils.roi.to_mask import contours_to_mask
from ..utils.roi.to_mesh import contours_to_mesh


def random_color(rgb_255=True):
    """
    Generate a random RGB color.

    Uses the `random` module, so `random.seed` makes it repeatable.

    Parameters
    ----------
    rgb_255 : bool
        True returns integers in [0, 255]. False returns floats in
        [0.0, 1.0).

    Returns
    -------
    tuple
        (r, g, b).

    Examples
    --------
    Integer color, repeatable with a seed:

    >>> import random
    >>> from medicalimageanalysis.structure.roi import random_color
    >>> random.seed(0)
    >>> random_color()
    (197, 215, 20)

    Float color for VTK / PyVista:

    >>> random.seed(0)
    >>> tuple(round(v, 3) for v in random_color(rgb_255=False))
    (0.844, 0.758, 0.421)
    """
    if rgb_255:
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    else:
        return (
            random.random(),
            random.random(),
            random.random()
        )


class Roi(object):
    """
    Region of interest of one image: contours, surface mesh and slicing.

    ROIs are normally made by the image (`image.add_roi`, `image.create_roi`,
    RTSTRUCT / MHD input) and stored in `image.roi[name]`.

    Attributes
    ----------
    contour_position : list[np.ndarray] | None
        World contours, (N, 3) in mm.

    contour_pixel : list[np.ndarray] | None
        Pixel contours, (N + 1, 3) x-y-z indices, closing point repeated.

    mesh : pv.PolyData | None
        Closed surface in mm. Setting it clears the slice cache.

    volume, com, bounds :
        Mesh volume (mm^3), center and bounds, set when a mesh is built.

    slice_cache_size : int
        Maximum number of cached slices per ROI, across all planes and
        views. Least recently used are dropped first.

    Examples
    --------
    Look up an ROI and read its geometry:

    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image['CT 01']
    >>> roi = image.roi['Liver']
    >>> roi.volume / 1000   # mL
    >>> roi.com

    Build an ROI directly from world contours:

    >>> from medicalimageanalysis.structure.roi import Roi
    >>> roi = Roi(image, position=contours, name='GTV', color=(255, 0, 0), plane='Axial')
    >>> roi.create_mesh()
    """
    slice_cache_size = 2500  # all planes/views together; least-recently-used dropped first

    def __init__(self, image, position=None, name=None, color=None, visible=False, filepaths=None, plane=None):
        """
        Initialize the ROI on an image.

        If `position` is given the pixel contours are computed right away;
        the mesh is not built until `create_mesh` (or an update method) is
        called.

        Parameters
        ----------
        image : Image
            Parent image; supplies origin, spacing, matrix, dimensions and
            display.

        position : list[np.ndarray] | None
            World contours, (N, 3) in mm.

        name : str | None
            ROI name, the key in `image.roi`.

        color : tuple | None
            RGB, 0-255. Random if None.

        visible : bool
            Show the ROI in the views.

        filepaths : list[str] | None
            Source files (RTSTRUCT, MHD, ...).

        plane : str | None
            Plane the contours were drawn on: 'Axial', 'Coronal' or
            'Sagittal'. Defaults to `image.plane`.

        Examples
        --------
        Empty ROI to be drawn later:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis.structure.roi import Roi
        >>> image = mia.Data.image['CT 01']
        >>> roi = Roi(image, name='GTV', color=(255, 0, 0))
        >>> roi.contour_position is None, roi.mesh is None
        (True, True)

        From world contours; pixel contours are filled in:

        >>> roi = Roi(image, position=contours, name='GTV', plane='Axial')
        >>> len(roi.contour_pixel) == len(contours)
        True
        """
        self.image = image

        self.name = name
        self.visible = visible
        self.color = color
        self.filepaths = filepaths

        self._slice_cache = OrderedDict()

        if plane is not None:
            self.plane = plane
        else:
            self.plane = self.image.plane

        if position is not None:
            self.contour_position = position
            self.contour_pixel = self.convert_position_to_pixel(position, base=True)
        else:
            self.contour_position = None
            self.contour_pixel = None

        if color is None:
            self.color = random_color()

        self.mesh = None
        self.cutter = None
        self.volume = None
        self.com = None
        self.bounds = None

        self.fixed_name = False
        self.rotation_center = None
        self.visual = {'2d': None, '3d': None, 'opacity': None, 'multicolor': None}
        self.misc = {}

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        """Any mesh replacement re-points the cached cutter and drops cached slices, so slices never use a stale
        mesh."""
        self._mesh = value

        cache = getattr(self, '_slice_cache', None)
        if cache is not None:
            cache.clear()

        if getattr(self, 'cutter', None) is not None:
            if value is None:
                self.cutter = None
            else:
                self.cutter.update_mesh(value)

    @staticmethod
    def _key_bytes(values, decimals):
        """Rounded bytes for cache keys. + 0.0 folds -0.0 into 0.0 so tiny sign flips don't cause misses."""
        return (np.round(np.asarray(values, dtype=float), decimals) + 0.0).tobytes()

    def add_mesh(self, mesh):
        """
        Attach an existing mesh and read its volume, center and bounds.

        Unlike `update_mesh`, the contours are kept.

        Parameters
        ----------
        mesh : pv.PolyData
            Closed surface in mm, in this image's world.

        Returns
        -------
        None

        Examples
        --------
        Attach a mesh loaded from disk:

        >>> import pyvista as pv
        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> roi.add_mesh(pv.read('C:/Data/liver.stl'))
        >>> roi.volume / 1000   # mL
        """
        self.mesh = mesh

        self.volume = mesh.volume
        self.com = mesh.center
        self.bounds = mesh.bounds

    def clear(self):
        """
        Remove the contours, mesh and derived geometry.

        Name, color, visibility and file paths are kept.

        Returns
        -------
        None

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> roi.clear()
        >>> roi.contour_position is None, roi.mesh is None, roi.volume is None
        (True, True, True)
        """
        self.contour_position = None
        self.contour_pixel = None

        self.mesh = None
        self.cutter = None
        self.volume = None
        self.com = None
        self.bounds = None

        self.fixed_name = False
        self.visual = {'2d': None, '3d': None, 'opacity': None, 'multicolor': None}
        self.misc = {}

    def clear_slice_cache(self):
        """
        Drop cached mesh slices.

        Only needed if the mesh is edited in place (mesh.points = ...,
        transform with inplace=True); replacing the mesh through the setter
        or `update_mesh` clears it automatically.

        Returns
        -------
        None

        Examples
        --------
        Shift the mesh 2 mm in place, then clear the stale slices:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> roi.mesh.points += [0.0, 0.0, 2.0]
        >>> roi.clear_slice_cache()
        """
        self._slice_cache.clear()

    def convert_position_to_pixel(self, points, base=True, plane=None, origin=None):
        """
        Convert world contours to pixel coordinates.

        With `base=True` the result is the image's own x-y-z pixel index.
        With `base=False` it is the 2D (column, row) pixel of a displayed
        plane, using the current view rotation (`image.display.matrix`) and
        the plane's top-left `origin`. Each returned loop is closed by
        repeating its first point.

        Parameters
        ----------
        points : list[np.ndarray]
            World contours, (N, 3) in mm.

        base : bool
            True for image pixel indices, False for display pixels.

        plane : str | None
            'Axial', 'Sagittal' or 'Coronal'. Required when `base=False`.

        origin : array-like (3,) | None
            World position of the plane's top-left pixel, e.g.
            `image.display.position[plane]`. Required when `base=False`.

        Returns
        -------
        list[np.ndarray]
            (N + 1, 3) x-y-z indices if `base=True`, else (N + 1, 2)
            (column, row).

        Examples
        --------
        Image pixel indices; the loop gains a closing point:

        >>> import numpy as np
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> roi = image.roi['Liver']
        >>> square = np.array([[-20, -20, 50], [20, -20, 50], [20, 20, 50], [-20, 20, 50]], dtype=float)
        >>> pixel = roi.convert_position_to_pixel([square])
        >>> pixel[0].shape
        (5, 3)

        Display pixels on the axial slice currently shown:

        >>> xy = roi.convert_position_to_pixel([square], base=False, plane='Axial',
        ...                                    origin=image.display.position['Axial'])
        >>> xy[0].shape
        (5, 2)
        """
        if base:
            position_to_pixel_matrix = self.image.compute_matrix_position_to_pixel()
            pixel = []
            for pos in points:
                p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
                pixel_3_axis = p_concat.dot(position_to_pixel_matrix.T)[:, :3]
                pixel += [np.vstack((pixel_3_axis, pixel_3_axis[0, :]))]

            return pixel

        else:
            axes = {'Axial': (0, 1, 2), 'Sagittal': (1, 2, 0), 'Coronal': (0, 2, 1)}
            xi, yi, ni = axes[plane]
            m = self.image.display.matrix
            x_axis, y_axis = m[:, xi], m[:, yi]
            sx, sy = self.image.spacing[xi], self.image.spacing[yi]

            pixel = []
            for pos in points:
                v = np.asarray(pos) - np.asarray(origin)
                col = v.dot(x_axis) / sx
                row = v.dot(y_axis) / sy
                xy = np.stack([col, row], axis=-1)
                pixel += [np.vstack((xy, xy[0, :]))]

            return pixel

    def convert_pixel_to_position(self, pixel=None):
        """
        Convert x-y-z pixel contours to world contours.

        Fractional indices are allowed. No closing point is added, so a
        round trip through `convert_position_to_pixel` keeps the repeated
        point.

        Parameters
        ----------
        pixel : list[np.ndarray]
            Pixel contours, (N, 3) x-y-z indices.

        Returns
        -------
        list[np.ndarray]
            World contours, (N, 3) in mm.

        Examples
        --------
        The first voxel maps to the image origin:

        >>> import numpy as np
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> roi = image.roi['Liver']
        >>> position = roi.convert_pixel_to_position([np.array([[0.0, 0.0, 0.0]])])
        >>> np.allclose(position[0][0], image.origin)
        True

        Round trip of the ROI's own contours:

        >>> back = roi.convert_pixel_to_position(roi.contour_pixel)
        >>> np.allclose(back[0][:-1], roi.contour_position[0])
        True
        """
        pixel_to_position_matrix = self.image.compute_matrix_pixel_to_position()

        position = []
        for ii, pix in enumerate(pixel):
            p_concat = np.concatenate((pix, np.ones((pix.shape[0], 1))), axis=1)
            position += [p_concat.dot(pixel_to_position_matrix.T)[:, :3]]

        return position

    def create_mesh(self, reduction=0):
        """
        Build a smooth closed surface mesh from the world contours.

        Sets `mesh`, `volume`, `com` and `bounds` (left unchanged if no mesh
        could be built).

        Parameters
        ----------
        reduction : float
            Decimation fraction applied once at the end, 0 to 1. 0 keeps
            every triangle; 0.5 removes about half.

        Returns
        -------
        None

        Examples
        --------
        Full-resolution mesh:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> roi.create_mesh()
        >>> full = roi.mesh.n_cells

        Lighter mesh for display or registration:

        >>> roi.create_mesh(reduction=0.5)
        >>> roi.mesh.n_cells < full
        True
        """
        self.mesh = contours_to_mesh(self.contour_position,
                                     self.image.origin,
                                     self.image.spacing,
                                     np.asarray(self.image.matrix, float),
                                     reduction=reduction)

        if self.mesh is not None:
            self.volume = self.mesh.volume
            self.com = self.mesh.center
            self.bounds = self.mesh.bounds

    def create_discrete_mesh(self):
        """
        Build a voxel-stepped (un-smoothed) mesh from the pixel contours.

        The surface follows the voxel faces, so its volume matches the voxel
        count of the mask. Sets `mesh`, `volume`, `com` and `bounds`.

        Returns
        -------
        None

        Examples
        --------
        Compare the stepped volume with the smooth one:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> roi.create_mesh()
        >>> smooth_ml = roi.volume / 1000
        >>> roi.create_discrete_mesh()
        >>> discrete_ml = roi.volume / 1000
        """
        meshing = ContourToDiscreteMesh(contour_pixel=self.contour_pixel,
                                        spacing=self.image.spacing,
                                        origin=self.image.origin,
                                        dimensions=self.image.dimensions,
                                        matrix=self.image.matrix,
                                        plane=self.plane)
        self.mesh = meshing.compute_mesh(discrete=True)

        self.volume = self.mesh.volume
        self.com = self.mesh.center
        self.bounds = self.mesh.bounds

    def create_display_mesh(self, iterations=20, angle=60, passband=0.001):
        """
        Smooth the current mesh in place with a windowed sinc filter.

        `volume`, `com` and `bounds` are not updated, so they still describe
        the mesh before smoothing.

        Parameters
        ----------
        iterations : int
            Number of smoothing passes.

        angle : float
            Feature angle in degrees; edges sharper than this are kept.

        passband : float
            Filter passband. Smaller values smooth more.

        Returns
        -------
        None

        Examples
        --------
        Smooth a stepped mesh for 3D rendering:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> roi.create_discrete_mesh()
        >>> roi.create_display_mesh(iterations=30, passband=0.0005)
        """
        refine = Refinement(self.mesh)
        self.mesh = refine.smooth(iterations=iterations, angle=angle, passband=passband)

    def create_decimate_mesh(self, percent=None, set_mesh=False):
        """
        Decimate the current mesh.

        Parameters
        ----------
        percent : float | None
            Fraction of triangles to remove, 0 to 1. If None, it is chosen
            so that about 10 * sqrt(n_points) points remain.

        set_mesh : bool
            If True, also replace `self.mesh` with the result.

        Returns
        -------
        pv.PolyData
            Decimated mesh.

        Examples
        --------
        Preview a 90 % reduction without touching the ROI:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> light = roi.create_decimate_mesh(percent=0.9)
        >>> light.n_cells < roi.mesh.n_cells
        True

        Automatic target, kept on the ROI:

        >>> mesh = roi.create_decimate_mesh(set_mesh=True)
        >>> roi.mesh is mesh
        True
        """
        if percent is None:
            points = np.round(10 * np.sqrt(self.mesh.number_of_points))
            percent = 1 - (points / self.mesh.number_of_points)

        mesh = self.mesh.decimate(percent)
        if set_mesh:
            self.mesh = mesh

        return mesh

    def create_cluster_mesh(self, points=None, set_mesh=False):
        """
        Remesh the current surface to near-uniform triangles (pyacvd).

        Parameters
        ----------
        points : int | None
            Target number of points. If None, it is chosen from the mesh
            volume and complexity.

        set_mesh : bool
            If True, also replace `self.mesh` with the result.

        Returns
        -------
        pv.PolyData
            Clustered closed surface.

        Examples
        --------
        Uniform 3000-point surface for registration or FE meshing:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> mesh = roi.create_cluster_mesh(points=3000)

        Automatic target, kept on the ROI:

        >>> roi.create_cluster_mesh(set_mesh=True)
        """
        refine = Refinement(self.mesh)
        mesh = refine.cluster(points=points)
        if set_mesh:
            self.mesh = mesh

        return mesh

    def compute_bounding_box(self, position=True):
        """
        Axis-aligned bounding box of the contours.

        Parameters
        ----------
        position : bool
            True uses the world contours (mm). False uses the pixel contours
            (x-y-z indices).

        Returns
        -------
        tuple
            (xmin, xmax, ymin, ymax, zmin, zmax).

        Examples
        --------
        Extent in mm:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> xmin, xmax, ymin, ymax, zmin, zmax = roi.compute_bounding_box()
        >>> width_mm = xmax - xmin

        First and last axial slice index the ROI is drawn on:

        >>> *_, kmin, kmax = roi.compute_bounding_box(position=False)
        >>> slices = range(int(round(kmin)), int(round(kmax)) + 1)
        """
        if position:
            contour_stacked = np.vstack(self.contour_position)
        else:
            contour_stacked = np.vstack(self.contour_pixel)

        contour_min = np.min(contour_stacked, axis=0)
        contour_max = np.max(contour_stacked, axis=0)

        return contour_min[0], contour_max[0], contour_min[1], contour_max[1], contour_min[2], contour_max[2]

    def compute_components(self):
        """
        Split the mesh into connected components.

        Returns
        -------
        list[pv.UnstructuredGrid] | None
            One mesh per connected region, largest first. None if the ROI
            has no mesh.

        Examples
        --------
        Count separate lesions in a GTV:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['GTV']
        >>> parts = roi.compute_components()
        >>> n_lesions = len(parts)
        >>> volumes_ml = [p.volume / 1000 for p in parts]
        """
        if self.mesh is not None:
            return compute_components(self.mesh)

        else:
            return None

    def compute_contour(self, slice_location, offset=0):
        """
        Pixel contours drawn on one slice index of the ROI's own plane.

        Only contours whose slice index (z for 'Axial', y for 'Coronal', x
        for 'Sagittal') rounds to `slice_location` are returned. This reads
        the stored contours; use `compute_mesh_slice` for other planes or
        oblique views.

        Parameters
        ----------
        slice_location : int
            Slice index along the ROI's plane normal.

        offset : float
            Added to both in-plane coordinates, e.g. 0.5 to move from voxel
            centers to voxel corners.

        Returns
        -------
        list[np.ndarray]
            (N, 2) in-plane pixel loops, closed. Empty if nothing is drawn
            on that slice.

        Examples
        --------
        Loops on axial slice 40:

        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> loops = roi.compute_contour(40)
        >>> loops[0].shape[1]
        2

        Shifted half a pixel for drawing on a pixel-corner grid:

        >>> loops = roi.compute_contour(40, offset=0.5)

        A slice outside the ROI:

        >>> roi.compute_contour(-1)
        []
        """
        contour_list = []
        if self.contour_pixel is not None:
            if self.plane == 'Axial':
                roi_z = [np.round(c[0, 2]).astype(int) for c in self.contour_pixel]
                keep_idx = np.argwhere(np.asarray(roi_z) == slice_location)

                if len(keep_idx) > 0:
                    for ii, idx in enumerate(keep_idx):
                        contour_corrected = np.vstack((self.contour_pixel[idx[0]][:, 0:2] + offset,
                                                       self.contour_pixel[idx[0]][0, 0:2] + offset))
                        contour_list += [contour_corrected]

            elif self.plane == 'Coronal':
                roi_y = [np.round(c[0, 1]).astype(int) for c in self.contour_pixel]
                keep_idx = np.argwhere(np.asarray(roi_y) == slice_location)

                if len(keep_idx) > 0:
                    for ii, idx in enumerate(keep_idx):
                        pixel_reshape = np.column_stack((self.contour_pixel[idx[0]][:, 0] + offset,
                                                         self.contour_pixel[idx[0]][:, 2] + offset))
                        stack = np.asarray([self.contour_pixel[idx[0]][0, 0], self.contour_pixel[idx[0]][0, 2]])
                        contour_corrected = np.vstack((pixel_reshape, stack))
                        contour_list += [contour_corrected]

            else:
                roi_x = [np.round(c[0, 0]).astype(int) for c in self.contour_pixel]
                keep_idx = np.argwhere(np.asarray(roi_x) == slice_location)

                if len(keep_idx) > 0:
                    for ii, idx in enumerate(keep_idx):
                        contour_corrected = np.vstack((self.contour_pixel[idx[0]][:, 1:] + offset,
                                                       self.contour_pixel[idx[0]][0, 1:] + offset))
                        contour_list += [contour_corrected]

        return contour_list

    def compute_mask(self):
        """
        Rasterize the world contours into a binary mask on the image grid.

        Returns
        -------
        np.ndarray
            (z, y, x) mask with the same shape as `image.dimensions`.

        Raises
        ------
        ValueError
            If the rasterized mask does not match `image.dimensions`.

        Examples
        --------
        Mask and voxel volume:

        >>> import numpy as np
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> mask = image.roi['Liver'].compute_mask()
        >>> mask.shape == tuple(image.dimensions)
        True
        >>> voxel_ml = mask.astype(bool).sum() * np.prod(image.spacing) / 1000

        Mean HU inside the ROI:

        >>> mean_hu = image.array[mask.astype(bool)].mean()
        """
        mask = contours_to_mask(self.contour_position,
                                self.image.dimensions,
                                self.image.origin,
                                self.image.spacing,
                                np.asarray(self.image.matrix, float))

        expected_shape = tuple(int(v) for v in np.asarray(self.image.dimensions).tolist())
        if mask.shape != expected_shape:
            raise ValueError(
                f"Contour-to-mask returned shape {mask.shape}, expected {expected_shape} in zyx order"
            )

        return mask

    def compute_mesh_slice(self, origin=None, plane=None, offset=0, rigid_matrix=None, return_pixel=False, grid=None,
                           use_cache=True):
        """
        Slice the ROI mesh along a displayed plane.

        The plane comes from `grid` (default: this image's own view of
        `plane` at `origin`), so oblique views work the same as axis-aligned
        ones. A `rigid_matrix` places the mesh in another image's world
        without moving it; the cutter's spatial index is reused.

        Results are cached in one LRU per ROI, keyed by the view geometry
        (axes, in-plane spacing), the rigid matrix, the slice origin
        (rounded to 1 um), offset and return_pixel. Scrolling back over a
        slice reuses the cached result. Rotating, flipping order or rigid
        dragging only creates new keys; old ones age out. Any mesh
        replacement clears the cache via the mesh setter.

        The returned tuple/list is the cached object; copy it before
        modifying it in place.

        Parameters
        ----------
        origin : array-like (3,) | None
            World position of the plane's top-left pixel, e.g.
            `image.display.position[plane]`. Used only when `grid` is None.

        plane : str | None
            'Axial', 'Sagittal' or 'Coronal'. Used only when `grid` is None.

        offset : float
            Added to both pixel coordinates when `return_pixel=True`.

        rigid_matrix : np.ndarray (4, 4) | None
            Maps this image's world to the displayed world, as returned by
            `Rigid.get_display_matrix`. Identity if None.

        return_pixel : bool
            False returns the cut as world-space PolyData. True returns 2D
            (column, row) pixel loops on the grid.

        grid : dict | None
            Plane in the displayed world, from
            `display.get_grid(plane, origin)`. Pass the base image's grid to
            draw this ROI on another image's view.

        use_cache : bool
            If True, read and store the result in the slice cache.

        Returns
        -------
        tuple
            (pv.PolyData, None) when `return_pixel=False`, or
            (list[np.ndarray], None) of (N, 2) loops when `return_pixel=True`.
            (None, None) / ([], []) if the ROI has no mesh.

        Examples
        --------
        Pixel loops on the axial slice currently shown:

        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> roi = image.roi['Liver']
        >>> loops, _ = roi.compute_mesh_slice(origin=image.display.position['Axial'], plane='Axial',
        ...                                   return_pixel=True)

        The same cut in world coordinates:

        >>> cut, _ = roi.compute_mesh_slice(origin=image.display.position['Axial'], plane='Axial')
        >>> cut.bounds

        An MR ROI drawn on the CT axial view through a rigid registration:

        >>> mr = mia.Data.image['MR 01']
        >>> rigid = mia.Data.rigid['CT 01_MR 01']
        >>> grid = image.display.get_grid('Axial', image.display.position['Axial'])
        >>> loops, _ = mr.roi['Liver'].compute_mesh_slice(
        ...     grid=grid,
        ...     rigid_matrix=rigid.get_display_matrix('CT 01', 'MR 01'),
        ...     return_pixel=True,
        ... )

        Skip the cache for a one-off cut:

        >>> cut, _ = roi.compute_mesh_slice(origin=image.display.position['Sagittal'], plane='Sagittal',
        ...                                 use_cache=False)
        """

        if self.mesh is None:
            return ([], []) if return_pixel else (None, None)
        elif self.cutter is None:
            self.cutter = MeshSlicer(self.mesh)

        rigid_matrix = rigid_matrix if rigid_matrix is not None else np.eye(4)

        # Grid defines the plane in DISPLAYED world. Default: this image's own view.
        if grid is None:
            grid = self.image.display.get_grid(plane, origin)

        x_axis, y_axis, normal = grid['x_axis'], grid['y_axis'], grid['normal']
        grid_origin = np.asarray(grid['origin'], dtype=float)
        sx, sy = grid['sx'], grid['sy']

        if use_cache:
            key = (self._key_bytes(x_axis, 9), self._key_bytes(y_axis, 9), self._key_bytes(normal, 9),
                   float(sx), float(sy),
                   self._key_bytes(rigid_matrix, 9),
                   self._key_bytes(grid_origin, 3),
                   float(offset), bool(return_pixel))
            hit = self._slice_cache.get(key)
            if hit is not None:
                self._slice_cache.move_to_end(key)
                return hit

        roi_slice = self.cutter.slice_transformed(normal=normal, origin=grid_origin, matrix=rigid_matrix)

        colors = None
        if not return_pixel:
            result = (roi_slice, colors)

        elif roi_slice.number_of_points == 0:
            result = ([], colors)

        else:
            roi_strip = roi_slice.strip(max_length=10000000)
            pixel_corrected = []
            for pts in (np.asarray(c.points) for c in roi_strip.cell):
                v = pts - grid_origin
                pixel_corrected += [np.stack([v.dot(x_axis) / sx + offset, v.dot(y_axis) / sy + offset], axis=-1)]
            result = (pixel_corrected, colors)

        if use_cache:
            self._slice_cache[key] = result
            while len(self._slice_cache) > self.slice_cache_size:
                self._slice_cache.popitem(last=False)

        return result[0], result[1]

    def create_sitk_mask(self):
        """
        Rasterize the contours into a SimpleITK image on the image grid.

        Origin, spacing and direction are copied from the parent image, so
        the mask can be written or resampled with SimpleITK directly.

        Returns
        -------
        SimpleITK.Image
            Binary mask with the image geometry.

        Examples
        --------
        Save the ROI as a NIfTI label:

        >>> import SimpleITK as sitk
        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> sitk_mask = roi.create_sitk_mask()
        >>> sitk.WriteImage(sitk_mask, 'C:/Data/liver.nii.gz')

        Geometry matches the image:

        >>> sitk_mask.GetSize() == tuple(int(d) for d in roi.image.dimensions[::-1])
        True
        """
        mask = self.compute_mask()

        matrix_flat = self.image.matrix.flatten(order='F')
        sitk_mask = sitk.GetImageFromArray(mask)
        sitk_mask.SetDirection([float(mat) for mat in matrix_flat])
        sitk_mask.SetOrigin(self.image.origin)
        sitk_mask.SetSpacing(self.image.spacing)

        return sitk_mask

    def convert_mask(self, mask):
        """
        Replace the ROI with the contours and mesh of a binary mask.

        The mask is contoured slice by slice on `self.plane`, then the
        mesh is rebuilt. An empty mask clears the mesh.

        Parameters
        ----------
        mask : np.ndarray
            (z, y, x) binary mask on this image's grid.

        Returns
        -------
        None

        Examples
        --------
        Load an nnU-Net prediction into an ROI:

        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> image.create_roi('Liver Pred', color=[0, 255, 0])
        >>> label = sitk.GetArrayFromImage(sitk.ReadImage('C:/Data/pred.nii.gz'))
        >>> image.roi['Liver Pred'].convert_mask((label == 1).astype(np.uint8))

        Threshold the CT for bone:

        >>> image.create_roi('Bone')
        >>> image.roi['Bone'].convert_mask((image.array > 300).astype(np.uint8))
        """
        mask_to_contour = MaskToContour(mask, spacing=self.image.spacing, origin=self.image.origin,
                                        matrix=self.image.matrix, plane=self.plane)
        self.contour_pixel, self.contour_position = mask_to_contour.create_contours()

        if len(self.contour_pixel) > 0:
            self.create_mesh()
        else:
            self.mesh = None
            self.volume = None
            self.com = None
            self.bounds = None

    def update_pixel(self, pixel, plane='Axial', ):
        """
        Replace the contours with new pixel contours and rebuild the mesh.

        World contours are computed from `pixel`. None or an empty list
        clears the contours and mesh.

        Parameters
        ----------
        pixel : list[np.ndarray] | None
            Pixel contours, (N, 3) x-y-z indices.

        plane : str
            Plane the contours were drawn on: 'Axial', 'Coronal' or
            'Sagittal'.

        Returns
        -------
        None

        Examples
        --------
        A 20 x 20 pixel square on axial slices 40-42:

        >>> import numpy as np
        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['GTV']
        >>> square = np.array([[200, 200], [220, 200], [220, 220], [200, 220]], dtype=float)
        >>> pixel = [np.c_[square, np.full(4, k)] for k in (40, 41, 42)]
        >>> roi.update_pixel(pixel, plane='Axial')
        >>> roi.mesh is not None
        True

        Clear the ROI:

        >>> roi.update_pixel(None)
        >>> roi.mesh is None
        True
        """
        self.plane = plane
        self.contour_pixel = pixel

        if pixel is not None and len(pixel) > 0:
            self.contour_position = self.convert_pixel_to_position(pixel=pixel)
            self.create_mesh()
        else:
            self.contour_pixel = None
            self.contour_position = None
            self.mesh = None

    def update_position(self, position, plane='Axial', ):
        """
        Replace the contours with new world contours and rebuild the mesh.

        Pixel contours are computed from `position`. None or an empty list
        clears the contours and mesh.

        Parameters
        ----------
        position : list[np.ndarray] | None
            World contours, (N, 3) in mm.

        plane : str
            Plane the contours were drawn on: 'Axial', 'Coronal' or
            'Sagittal'.

        Returns
        -------
        None

        Examples
        --------
        A 40 mm circle on three axial slices:

        >>> import numpy as np
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> roi = image.roi['GTV']
        >>> t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        >>> z0 = image.origin[2] + 40 * image.spacing[2]
        >>> circles = [np.c_[20 * np.cos(t), 20 * np.sin(t), np.full(64, z0 + k * image.spacing[2])]
        ...            for k in range(3)]
        >>> roi.update_position(circles, plane='Axial')
        >>> len(roi.contour_pixel)
        3

        Copy contours from another ROI:

        >>> roi.update_position(image.roi['Liver'].contour_position, plane=image.roi['Liver'].plane)
        """
        self.plane = plane
        self.contour_position = position

        if position is not None and len(position) > 0:
            self.contour_pixel = self.convert_position_to_pixel(points=position)
            self.create_mesh()
        else:
            self.contour_pixel = None
            self.contour_position = None
            self.mesh = None

    def update_mesh(self, mesh):
        """
        Replace the ROI with an external mesh and drop the contours.

        Use this when the mesh is now the source of truth (deformed,
        registered or edited in 3D). For a mesh that should sit alongside
        the existing contours, use `add_mesh`.

        Parameters
        ----------
        mesh : pv.PolyData
            Closed surface in mm, in this image's world.

        Returns
        -------
        None

        Examples
        --------
        Replace the liver with a deformed surface:

        >>> import pyvista as pv
        >>> import medicalimageanalysis as mia
        >>> roi = mia.Data.image['CT 01'].roi['Liver']
        >>> roi.update_mesh(pv.read('C:/Data/liver_deformed.vtk'))
        >>> roi.contour_position is None
        True
        """
        self.mesh = mesh
        self.volume = mesh.volume
        self.com = mesh.center
        self.bounds = mesh.bounds


        self.contour_pixel = None
        self.contour_position = None