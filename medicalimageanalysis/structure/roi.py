"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Module for defining and manipulating Region of Interest (ROI) structures,
    including coordinate conversions and mesh generations.

Structure:
    - random_color: Helper function to generate random RGB values.
    - Roi: Core class managing volumetric region data, contours, and surface meshes.

Examples
--------
>>> import medicalimageanalysis as mia
>>> mia.read_mhd(file='C:/Data/scan.mhd', modality='CT')
>>> image = mia.Data.image[mia.Data.image_list()[0]]
"""

import random
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk


from ..utils.mesh.surface import Refinement, compute_components
from ..utils.roi.to_mask import contours_to_mask
from ..utils.roi.to_mesh import contours_to_mesh
from ..utils.roi.discrete_converters import ContourToDiscreteMesh, ContourToMask, MaskToContour, MeshToContour


def random_color(rgb_255=True):
    """
    Generate a random color.

    Parameters
    ----------
    rgb_255 : bool, default True
        True  -> returns integers in [0, 255]
        False -> returns floats in [0.0, 1.0]

    Returns
    -------
    tuple
        A tuple of three elements containing (r, g, b) values.

    Examples
    --------
    >>> random_color(rgb_255=True)
    (142, 54, 210)
    >>> random_color(rgb_255=False)
    (0.421, 0.812, 0.119)
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
    slice_cache_size = 2500  # all planes/views together; least-recently-used dropped first

    """Region of Interest (ROI) manager for internal image contours and 3D meshes.
    
    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image[mia.Data.image_list()[0]]
    >>> roi = image.roi['PTV']
    """
    def __init__(self, image, position=None, name=None, color=None, visible=False, filepaths=None, plane=None):
        """Initialize the Roi object.
        
        Parameters
        ----------
        image : object
            The foundational image object with spatial attributes (spacing, origin, matrix).
        position : list of numpy.ndarray, optional
            A list containing spatial position contours.
        name : str, optional
            The display or identification name of the ROI.
        color : tuple, optional
            RGB representation of the ROI color. Generates a random color if None.
        visible : bool, default False
            Determines whether the ROI is visible within the viewport.
        filepaths : list of str, optional
            Associated source or configuration files for the ROI.
        plane : str, optional
            The orientation plane ('Axial', 'Coronal', or 'Sagittal'). Defaults to image plane.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.__init__(...)
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
        """Bind an existing mesh to the ROI and extract its spatial attributes.
        
        Parameters
        ----------
        mesh : object
            A mesh object containing structural parameters (volume, center, bounds).
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.add_mesh(...)
        """
        self.mesh = mesh

        self.volume = mesh.volume
        self.com = mesh.center
        self.bounds = mesh.bounds

    def clear(self):
        """Reset the geometric metrics, mesh, and structural bindings of the ROI.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.clear(...)
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
        """Drop cached mesh slices. Only needed if the mesh is edited in place (mesh.points = ..., transform with
        inplace=True); replacing the mesh through the setter or update_mesh clears it automatically."""
        self._slice_cache.clear()

    def convert_position_to_pixel(self, points, base=True, plane=None, origin=None):
        """Convert world-space contour points into image or display pixel coordinates.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.convert_position_to_pixel(...)
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
        """Convert pixel coordinates into physical spatial world coordinates.
        
        Parameters
        ----------
        pixel : list of numpy.ndarray, optional
            A list of image pixel-index matrices to be mapped.
        
        Returns
        -------
        list of numpy.ndarray
            The physical coordinates corresponding to the pixel indexes.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.convert_pixel_to_position(...)
        """
        pixel_to_position_matrix = self.image.compute_matrix_pixel_to_position()

        position = []
        for ii, pix in enumerate(pixel):
            p_concat = np.concatenate((pix, np.ones((pix.shape[0], 1))), axis=1)
            position += [p_concat.dot(pixel_to_position_matrix.T)[:, :3]]

        return position

    def create_mesh(self, reduction=0):
        """Generate a smoothed 3D surface mesh from the current pixel contours.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.create_mesh(...)
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
        """Compute an un-smoothed, step-like discrete mesh straight from pixel contour tracks.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.create_discrete_mesh(...)
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
        """Refine and smooth the loaded mesh for optimized presentation and rendering.
        
        Parameters
        ----------
        iterations : int, default 20
            The iteration loop limit assigned to the mesh smoothing system.
        angle : float/int, default 60
            Feature angle threshold to protect defined corners/edges.
        passband : float, default 0.001
            Low-pass filter frequency option specifying overall detail preservation.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.create_display_mesh(...)
        """
        refine = Refinement(self.mesh)
        self.mesh = refine.smooth(iterations=iterations, angle=angle, passband=passband)

    def create_decimate_mesh(self, percent=None, set_mesh=False):
        """Reduce the poly/triangle count of the active mesh framework.
        
        Parameters
        ----------
        percent : float, optional
            Decimation target coefficient. Calculated based on total nodes if None.
        set_mesh : bool, default False
            If True, overrides `self.mesh` directly with the decimated structure.
        
        Returns
        -------
        object
            The decimated mesh instance.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.create_decimate_mesh(...)
        """
        if percent is None:
            points = np.round(10 * np.sqrt(self.mesh.number_of_points))
            percent = 1 - (points / self.mesh.number_of_points)

        mesh = self.mesh.decimate(percent)
        if set_mesh:
            self.mesh = mesh

        return mesh

    def create_cluster_mesh(self, points=None, set_mesh=False):
        """Cluster vertices across the mesh workspace to modify structural density.
        
        Parameters
        ----------
        points : int, optional
            Target total cluster sample instances to evaluate.
        set_mesh : bool, default False
            If True, overrides `self.mesh` directly with the output cluster mesh.
        
        Returns
        -------
        object
            The clustered mesh instance.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.create_cluster_mesh(...)
        """
        refine = Refinement(self.mesh)
        mesh = refine.cluster(points=points)
        if set_mesh:
            self.mesh = mesh

        return mesh

    def compute_bounding_box(self, position=True):
        """Compute the 3D bounding box coordinates from the contour data.
        
        Parameters
        ----------
        position : bool, default True
            If True, evaluates physical position coordinates (`self.contour_position`).
            If False, evaluates pixel-space coordinates (`self.contour_pixel`).
        
        Returns
        -------
        tuple
            The bounding box boundaries ordered as (xmin, xmax, ymin, ymax, zmin, zmax).
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.compute_bounding_box(...)
        """
        if position:
            contour_stacked = np.vstack(self.contour_position)
        else:
            contour_stacked = np.vstack(self.contour_pixel)

        contour_min = np.min(contour_stacked, axis=0)
        contour_max = np.max(contour_stacked, axis=0)

        return contour_min[0], contour_max[0], contour_min[1], contour_max[1], contour_min[2], contour_max[2]

    def compute_components(self):
        """Return connected components of the current ROI mesh, when available.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.compute_components(...)
        """
        if self.mesh is not None:
            return compute_components(self.mesh)

        else:
            return None

    def compute_contour(self, slice_location, offset=0):
        """Extract specific loop coordinates along a given anatomical frame intersection.
        
        Parameters
        ----------
        slice_location : int
            The dimensional location slice coordinate to evaluate.
        offset : int/float, default 0
            Coordinate indexing padding offset adjustments.
        
        Returns
        -------
        list of numpy.ndarray
            Extracted closed loops representing the slice intersection.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.compute_contour(...)
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
        """Generate a binary voxel index mask reflecting the active contour perimeter profile.
        
        Returns
        -------
        numpy.ndarray
            A binary segmentation volume masking the underlying dataset space.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.compute_mask(...)
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
        """Slice the internal 3D mesh along a viewing plane.

        Results are cached in one LRU per ROI, keyed by the view geometry (axes, in-plane spacing), the rigid matrix,
        the slice origin (rounded to 1 um), offset and return_pixel. Scrolling back over a slice reuses the cached
        result. Rotating, flipping order or rigid dragging only creates new keys; old ones age out. Any mesh
        replacement clears the cache via the mesh setter.

        The returned tuple/list is the cached object; copy it before modifying it in place.
        
        Parameters
        ----------
        origin : float/list, optional
            The exact 3D origin positioning context mapping the cross-section plane.
        plane : str, optional
            Target orientation plane ('Axial', 'Coronal', or 'Sagittal').
        offset : int/float, default 0
            Padding vector adjustments applied to output indexing paths.
        rigid_matrix: 4x4 numpy array
            Lets you slice based on a rigid registration
        return_pixel : bool, default False
            If True, calculates and formats structural data outputs as pixel indexes.
        grid: image.display.get_grid(plane, origin)
            Grid defines the plane in DISPLAYED world. Default: this image's own view.
        
        Returns
        -------
        tuple
            (pixel_corrected, colors) if return_pixel is True, otherwise (roi_slice, colors).
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.compute_mesh_slice(...)
        """

        if self.mesh is None:
            return ([], []) if return_pixel else (None, None)
        elif self.cutter is None:
            self.cutter = MeshToContour(self.mesh)

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
        """Build an object instance conforming to a dedicated SimpleITK Image representation format.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        SimpleITK.Image
            An image structure mapped into SimpleITK space containing metadata properties.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.create_sitk_mask(...)
        """
        mask = self.compute_mask()

        matrix_flat = self.image.matrix.flatten(order='F')
        sitk_mask = sitk.GetImageFromArray(mask)
        sitk_mask.SetDirection([float(mat) for mat in matrix_flat])
        sitk_mask.SetOrigin(self.image.origin)
        sitk_mask.SetSpacing(self.image.spacing)

        return sitk_mask

    def convert_mask(self, mask):
        """Deconstruct a spatial binary mask array template back into contours and updated tracking meshes.
        
        Parameters
        ----------
        mask : numpy.ndarray
            The volumetric matrix profile used to update contour paths.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.convert_mask(...)
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
        """Manually assign a new configuration of pixel layers and regenerate corresponding meshes.
        
        Parameters
        ----------
        pixel : list of numpy.ndarray
            The target dataset coordinates to update the contours with.
        plane : str, default 'Axial'
            The new structural evaluation orientation workspace.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.update_pixel(...)
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

    def update_mesh(self, mesh):
        """Manually substitute an external tracking mesh structure and update metadata attributes.
        
        Parameters
        ----------
        mesh : object
            The physical alternative surface mesh properties database.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image[mia.Data.image_list()[0]]
        >>> roi = image.roi['PTV']
        >>> roi.update_mesh(...)
        """
        self.mesh = mesh
        self.volume = mesh.volume
        self.com = mesh.center
        self.bounds = mesh.bounds


        self.contour_pixel = None
        self.contour_position = None