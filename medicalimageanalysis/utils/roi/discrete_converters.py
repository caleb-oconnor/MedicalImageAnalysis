"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Discrete ROI Converters
=======================

Description:
    This module holds the voxel-based (discrete) ROI converters. Contours
    are filled slice by slice with OpenCV, and masks are traced with
    OpenCV contour finding, so all results snap to the pixel grid.

Classes:
    * ContourToDiscreteMesh   contours -> mask -> mesh (marching cubes or surface nets)
    * ContourToMask           contours -> binary mask
    * MaskToContour           binary mask -> pixel and world contours
    * ModelToMask             surface models -> mask on an auto-sized grid

Key Functionalities:
    1. **Multi-Plane Filling**: Contours can be axial, coronal or sagittal;
       holes are handled with XOR filling on each slice.
    2. **Mesh Generation**: Discrete marching cubes or smoothed VTK surface
       nets from the filled mask.
    3. **Mask Export**: `ModelToMask.save_image` writes the mask with
       SimpleITK.

Conventions:
    Masks and `dimensions` are z-y-x. Pixel contours are (n, 3) arrays of
    (x, y, z) voxel indices.

Usage:
    >>> from medicalimageanalysis.utils.roi.converters import ContourToMask
    >>> # Fill pixel-space contours into a mask
    >>> helper = ContourToMask(contour_pixel=contours, dimensions=(8, 16, 16))
    >>> helper.create_mask()

Examples
--------
>>> import numpy as np
>>> from medicalimageanalysis.utils.roi.converters import ContourToMask, MaskToContour
>>> square = np.array([[4, 4, 2], [12, 4, 2], [12, 12, 2], [4, 12, 2]], dtype=float)
>>> helper = ContourToMask(contour_pixel=[square], dimensions=(8, 16, 16))
>>> helper.create_mask()
>>> int(helper.mask[2].sum())
81
>>> pixel, position = MaskToContour(
...     mask=helper.mask, spacing=(1, 1, 2), origin=(0, 0, 0), matrix=np.eye(3),
... ).create_contours()
>>> len(pixel)
1
"""

import cv2
import numpy as np
import pyvista as pv
import SimpleITK as sitk

import vtk
from vtk.util import numpy_support


class ContourToDiscreteMesh(object):
    """
    Convert contours into a binary mask and then a discrete or smoothed surface mesh.

    The mask is built during construction (unless one is passed in);
    `compute_mesh` turns it into a mesh. Axial, coronal and sagittal
    contours are supported.

    Parameters
    ----------
    contour_position : list[np.ndarray] | None
        Contours as (n, 3) world points in mm.

    contour_pixel : list[np.ndarray] | None
        Contours as (n, 3) voxel indices (x, y, z). Used directly if given.

    spacing : tuple[float, float, float] | None
        Voxel spacing in mm (x-y-z).

    origin : tuple[float, float, float] | None
        World (x, y, z) of voxel (0, 0, 0).

    dimensions : tuple[int, int, int] | None
        Mask shape in z-y-x order.

    matrix : np.ndarray | None
        3x3 direction matrix. Identity if None.

    plane : str
        'Axial', 'Coronal' or 'Sagittal'.

    mask : np.ndarray | None
        Existing mask. If given, contour conversion and filling are skipped.

    Examples
    --------
    Mesh four stacked square contours (pixel space):

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.converters import ContourToDiscreteMesh
    >>> square = np.array([[4, 4], [12, 4], [12, 12], [4, 12]], dtype=float)
    >>> contours = [np.c_[square, np.full(4, z)] for z in range(2, 6)]
    >>> helper = ContourToDiscreteMesh(
    ...     contour_pixel=contours,
    ...     spacing=(1, 1, 2),
    ...     origin=(0, 0, 0),
    ...     dimensions=(8, 16, 16),
    ...     matrix=np.eye(3),
    ... )
    >>> mesh = helper.compute_mesh()
    """
    def __init__(self, contour_position=None, contour_pixel=None, spacing=None, origin=None, dimensions=None,
                 matrix=None, plane='Axial', mask=None):
        """
        Store geometry, convert contours to pixel space if needed and build the mask.

        Examples
        --------
        From world-space contours (converted to voxel indices internally):

        >>> import numpy as np
        >>> square = np.array([[4, 4], [12, 4], [12, 12], [4, 12]], dtype=float)
        >>> world = [np.c_[square, np.full(4, z * 2.0)] for z in range(2, 6)]
        >>> helper = ContourToDiscreteMesh(
        ...     contour_position=world,
        ...     spacing=(1, 1, 2),
        ...     origin=(0, 0, 0),
        ...     dimensions=(8, 16, 16),
        ... )
        >>> helper.mask.shape
        (8, 16, 16)

        From an existing mask (no contours needed):

        >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
        >>> mask[2:6, 4:12, 4:12] = 1
        >>> helper = ContourToDiscreteMesh(
        ...     mask=mask, spacing=(1, 1, 2), origin=(0, 0, 0), dimensions=(8, 16, 16),
        ... )
        """
        self.contour_position = contour_position
        self.contour_pixel = contour_pixel
        self.spacing = spacing
        self.origin = origin
        self.dimensions = dimensions
        self.plane = plane

        self.mask = mask

        if matrix is None:
            self.matrix = np.identity(3)
        else:
            self.matrix = matrix

        if self.contour_pixel is None and self.mask is None:
            self.convert_to_pixel_spacing()

        if self.mask is None:
            self.compute_mask()

    def convert_to_pixel_spacing(self):
        """
        Convert `contour_position` (world, mm) to `contour_pixel` (voxel indices).

        Uses a 4x4 world-to-voxel matrix built from `matrix`, `spacing` and
        `origin`. Called automatically by `__init__` when only world
        contours are given.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
        >>> helper = ContourToDiscreteMesh(mask=mask, spacing=(1, 1, 2), origin=(0, 0, 0))
        >>> helper.contour_position = [np.array([[4, 4, 6], [12, 4, 6], [12, 12, 6]], dtype=float)]
        >>> helper.convert_to_pixel_spacing()
        >>> helper.contour_pixel[0][0]          # z = 6 mm -> slice 3
        array([4., 4., 3.])
        """
        matrix = np.identity(3, dtype=np.float32)
        matrix[0, :] = self.matrix[0, :] / self.spacing[0]
        matrix[1, :] = self.matrix[1, :] / self.spacing[1]
        matrix[2, :] = self.matrix[2, :] / self.spacing[2]

        conversion_matrix = np.identity(4, dtype=np.float32)
        conversion_matrix[:3, :3] = matrix
        conversion_matrix[:3, 3] = np.asarray(self.origin).dot(-matrix.T)

        self.contour_pixel = []
        for ii, pos in enumerate(self.contour_position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            self.contour_pixel += [p_concat.dot(conversion_matrix.T)[:, :3]]

    def compute_mask(self):
        """
        Fill `contour_pixel` into a binary mask, slice by slice.

        Contours on the same slice are XOR-combined, so an inner ring cuts a
        hole. The result is stored in `self.mask` as uint8 0/1.

        Returns
        -------
        None

        Examples
        --------
        A ring with a hole on slice 2:

        >>> import numpy as np
        >>> outer = np.array([[2, 2, 2], [13, 2, 2], [13, 13, 2], [2, 13, 2]], dtype=float)
        >>> inner = np.array([[6, 6, 2], [9, 6, 2], [9, 9, 2], [6, 9, 2]], dtype=float)
        >>> helper = ContourToDiscreteMesh(
        ...     contour_pixel=[outer, inner], dimensions=(8, 16, 16),
        ... )
        >>> int(helper.mask[2, 3, 3]), int(helper.mask[2, 7, 7])
        (1, 0)
        """
        hold_mask = np.zeros([self.dimensions[0], self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
        if self.plane == 'Axial':
            image = np.zeros([self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
            for c in self.contour_pixel:
                contour_stacked = np.round(np.vstack((c[:, 0:2], c[0, 0:2])))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image.fill(0)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 2]))
                hold_mask[slice_num, :, :] = np.logical_xor(hold_mask[slice_num, :, :], image)

        elif self.plane == 'Coronal':
            image = np.zeros([self.dimensions[0], self.dimensions[2]], dtype=np.uint8)
            for c in self.contour_pixel:
                stack_1 = np.vstack((c[:, 0], c[:, 2])).T
                stack_2 = np.asarray([c[0, 0], c[0, 2]])
                contour_stacked = np.round(np.vstack((stack_1, stack_2)))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image.fill(0)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 1]))
                hold_mask[:, slice_num, :] = np.logical_xor(hold_mask[:, slice_num, :], image)

        else:
            image = np.zeros([self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
            for c in self.contour_pixel:
                contour_stacked = np.round(np.vstack((c[:, 1:], c[0, 1:])))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image.fill(0)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 0]))
                hold_mask[:, :, slice_num] = np.logical_xor(hold_mask[:, :, slice_num], image)

        self.mask = (hold_mask > 0).astype(np.uint8)

    def compute_mesh(self, discrete=False, smoothing_iterations=20, smoothing_relaxation=.5, smoothing_distance=1):
        """
        Convert the mask into a surface mesh in world coordinates.

        The mask is padded by one voxel so the surface closes at the image
        edge.

        Parameters
        ----------
        discrete : bool
            If True, use discrete marching cubes (blocky, voxel-exact). If
            False, use smoothed VTK surface nets.

        smoothing_iterations : int
            Surface nets smoothing iterations (discrete=False only).

        smoothing_relaxation : float
            Surface nets relaxation factor (discrete=False only).

        smoothing_distance : float
            Maximum distance a point may move while smoothing
            (discrete=False only).

        Returns
        -------
        pv.PolyData
            Closed surface mesh.

        Examples
        --------
        >>> import numpy as np
        >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
        >>> mask[2:6, 4:12, 4:12] = 1
        >>> helper = ContourToDiscreteMesh(
        ...     mask=mask, spacing=(1, 1, 2), origin=(0, 0, 0), dimensions=(8, 16, 16),
        ... )

        Smoothed surface (default):

        >>> smooth = helper.compute_mesh()

        Voxel-exact surface:

        >>> blocky = helper.compute_mesh(discrete=True)

        Lighter smoothing that stays closer to the voxels:

        >>> light = helper.compute_mesh(smoothing_iterations=5, smoothing_distance=0.5)
        """
        label = numpy_support.numpy_to_vtk(num_array=self.mask.ravel(),
                                           deep=True,
                                           array_type=vtk.VTK_FLOAT)

        img_vtk = vtk.vtkImageData()
        img_vtk.SetDimensions(self.dimensions[2], self.dimensions[1], self.dimensions[0])
        img_vtk.SetSpacing(self.spacing)
        img_vtk.SetOrigin(self.origin)
        img_vtk.SetDirectionMatrix(self.matrix.flatten(order='F'))
        img_vtk.GetPointData().SetScalars(label)

        pad_filter = vtk.vtkImageConstantPad()
        pad_filter.SetInputData(img_vtk)
        pad_filter.SetConstant(0)

        old_extent = img_vtk.GetExtent()
        new_extent = [old_extent[0] - 1, old_extent[1] + 1,
                      old_extent[2] - 1, old_extent[3] + 1,
                      old_extent[4] - 1, old_extent[5] + 1]
        pad_filter.SetOutputWholeExtent(new_extent)
        pad_filter.Update()
        pad_image = pad_filter.GetOutput()

        if discrete:
            vtk_mesh = vtk.vtkDiscreteMarchingCubes()
            vtk_mesh.SetInputData(pad_image)
            vtk_mesh.GenerateValues(1, 1, 1)
            vtk_mesh.Update()

            return pv.PolyData(vtk_mesh.GetOutput())

        else:
            # Uses VTK surface nets 3d
            img = pv.ImageData(pad_image)
            mesh = img.contour_labels(smoothing=True,
                                      output_mesh_type='triangles',
                                      smoothing_iterations=smoothing_iterations,
                                      smoothing_relaxation=smoothing_relaxation,
                                      smoothing_distance=smoothing_distance)

            return mesh


class ContourToMask(object):
    """
    Convert contours into a binary volumetric mask.

    Construction only stores the inputs; call `create_mask` to fill
    `self.mask`. Axial, coronal and sagittal contours are supported.

    Parameters
    ----------
    contour_position : list[np.ndarray] | None
        Contours as (n, 3) world points in mm.

    contour_pixel : list[np.ndarray] | None
        Contours as (n, 3) voxel indices (x, y, z). Used directly if given.

    spacing : tuple[float, float, float] | None
        Voxel spacing in mm (x-y-z).

    origin : tuple[float, float, float] | None
        World (x, y, z) of voxel (0, 0, 0).

    dimensions : tuple[int, int, int] | None
        Mask shape in z-y-x order.

    matrix : np.ndarray | None
        Direction matrix (at least 3x3). Required when converting
        `contour_position`.

    plane : str
        'Axial', 'Coronal' or 'Sagittal'.

    Examples
    --------
    Fill one axial square on slice 2:

    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.converters import ContourToMask
    >>> square = np.array([[4, 4, 2], [12, 4, 2], [12, 12, 2], [4, 12, 2]], dtype=float)
    >>> helper = ContourToMask(contour_pixel=[square], dimensions=(8, 16, 16))
    >>> helper.create_mask()
    >>> int(helper.mask[2].sum())               # 9 x 9 pixels, edges included
    81
    """
    def __init__(self, contour_position=None, contour_pixel=None, spacing=None, origin=None, dimensions=None,
                 matrix=None, plane='Axial'):
        """
        Store contours and geometry. No conversion is done until `create_mask`.

        Examples
        --------
        >>> import numpy as np
        >>> square = np.array([[4, 4, 2], [12, 4, 2], [12, 12, 2], [4, 12, 2]], dtype=float)
        >>> helper = ContourToMask(contour_pixel=[square], dimensions=(8, 16, 16))
        >>> helper.mask is None
        True
        """
        self.contour_position = contour_position
        self.contour_pixel = contour_pixel
        self.spacing = spacing
        self.origin = origin
        self.dimensions = dimensions
        self.matrix = matrix
        self.plane = plane

        self.mask = None

    def create_mask(self):
        """
        Convert contours to pixel space if needed, then fill `self.mask`.

        Returns
        -------
        None

        Examples
        --------
        Axial contours:

        >>> import numpy as np
        >>> square = np.array([[4, 4, 2], [12, 4, 2], [12, 12, 2], [4, 12, 2]], dtype=float)
        >>> helper = ContourToMask(contour_pixel=[square], dimensions=(8, 16, 16))
        >>> helper.create_mask()
        >>> helper.mask.shape, helper.mask.dtype
        ((8, 16, 16), dtype('uint8'))

        Sagittal contour on x = 8 (filled in the z-y plane):

        >>> sagittal = np.array([[8, 4, 1], [8, 12, 1], [8, 12, 6], [8, 4, 6]], dtype=float)
        >>> helper = ContourToMask(contour_pixel=[sagittal], dimensions=(8, 16, 16), plane='Sagittal')
        >>> helper.create_mask()
        >>> int(helper.mask[3, 8, 8])
        1
        """
        if self.contour_pixel is None:
            self.convert_to_pixel_spacing()

        self.compute_mask()

    def convert_to_pixel_spacing(self):
        """
        Convert `contour_position` (world, mm) to voxel indices using SimpleITK.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> world = [np.array([[4, 4, 4], [12, 4, 4], [12, 12, 4], [4, 12, 4]], dtype=float)]
        >>> helper = ContourToMask(
        ...     contour_position=world,
        ...     spacing=(1, 1, 2),
        ...     origin=(0, 0, 0),
        ...     dimensions=(16, 16, 8),
        ...     matrix=np.eye(3),
        ... )
        >>> helper.convert_to_pixel_spacing()
        >>> helper.contour_pixel[0][0]           # z = 4 mm -> slice 2
        (4.0, 4.0, 2.0)
        """
        sitk_image = sitk.Image([int(dim) for dim in self.dimensions], sitk.sitkUInt8)
        matrix_flat = self.matrix[0:3, 0:3].flatten(order='F')
        sitk_image.SetDirection([float(mat) for mat in matrix_flat])
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetSpacing(self.spacing)

        self.contour_pixel = [[]] * len(self.contour_position)
        for ii, contours in enumerate(self.contour_position):
            self.contour_pixel[ii] = [sitk_image.TransformPhysicalPointToContinuousIndex(contour) for contour in contours]

    def compute_mask(self):
        """
        Fill `contour_pixel` into a binary mask, slice by slice.

        Contours on the same slice are XOR-combined, so an inner ring cuts a
        hole. The result is stored in `self.mask` as uint8 0/1.

        Returns
        -------
        None

        Examples
        --------
        Two separate squares on different slices:

        >>> import numpy as np
        >>> a = np.array([[2, 2, 1], [6, 2, 1], [6, 6, 1], [2, 6, 1]], dtype=float)
        >>> b = np.array([[8, 8, 4], [14, 8, 4], [14, 14, 4], [8, 14, 4]], dtype=float)
        >>> helper = ContourToMask(contour_pixel=[a, b], dimensions=(8, 16, 16))
        >>> helper.compute_mask()
        >>> [int(helper.mask[k].any()) for k in range(8)]
        [0, 1, 0, 0, 1, 0, 0, 0]
        """
        hold_mask = np.zeros([self.dimensions[0], self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
        if self.plane == 'Axial':
            image = np.zeros([self.dimensions[1], self.dimensions[2]], dtype=np.uint8)
            for c in self.contour_pixel:
                contour_stacked = np.round(np.vstack((c[:, 0:2], c[0, 0:2])))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image.fill(0)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 2]))
                hold_mask[slice_num, :, :] = np.logical_xor(hold_mask[slice_num, :, :], image)

        elif self.plane == 'Coronal':
            image = np.zeros([self.dimensions[0], self.dimensions[2]], dtype=np.uint8)
            for c in self.contour_pixel:
                stack_1 = np.vstack((c[:, 0], c[:, 2])).T
                stack_2 = np.asarray([c[0, 0], c[0, 2]])
                contour_stacked = np.round(np.vstack((stack_1, stack_2)))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image.fill(0)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 1]))
                hold_mask[:, slice_num, :] = np.logical_xor(hold_mask[:, slice_num, :], image)

        else:
            image = np.zeros([self.dimensions[0], self.dimensions[1]], dtype=np.uint8)
            for c in self.contour_pixel:
                contour_stacked = np.round(np.vstack((c[:, 1:], c[0, 1:])))
                new_contour = np.array([contour_stacked], dtype=np.int32)
                image.fill(0)
                cv2.fillPoly(image, new_contour, 1)

                slice_num = int(np.round(c[0, 0]))
                hold_mask[:, :, slice_num] = np.logical_xor(hold_mask[:, :, slice_num], image)

        self.mask = (hold_mask > 0).astype(np.uint8)


class MaskToContour(object):
    """
    Trace a binary mask into slice-wise contours, in pixel and world coordinates.

    Only outer contours are kept (no holes), simplified to their corner
    points.

    Parameters
    ----------
    mask : np.ndarray | None
        Mask in z-y-x order. Any value > 0 counts as inside.

    spacing : tuple[float, float, float] | None
        Voxel spacing in mm (x-y-z).

    origin : tuple[float, float, float] | None
        World (x, y, z) of voxel (0, 0, 0).

    matrix : np.ndarray | None
        3x3 direction matrix.

    plane : str
        'axial', 'coronal' or 'sagittal' (case-insensitive).

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.converters import MaskToContour
    >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
    >>> mask[2:6, 4:12, 4:12] = 1
    >>> helper = MaskToContour(mask=mask, spacing=(1, 1, 2), origin=(0, 0, 0), matrix=np.eye(3))
    >>> pixel, position = helper.create_contours()
    >>> len(pixel)                               # one contour per slice
    4
    """

    def __init__(self, mask=None, spacing=None, origin=None, matrix=None, plane='axial'):
        """
        Store the mask and geometry, and initialize the output contour lists.

        Examples
        --------
        >>> import numpy as np
        >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
        >>> helper = MaskToContour(mask=mask, spacing=(1, 1, 2), origin=(0, 0, 0), matrix=np.eye(3))
        >>> helper.contour_pixel, helper.contour_position
        ([], [])
        """
        self.mask = mask
        self.spacing = spacing
        self.origin = origin
        self.matrix = matrix
        self.plane = plane

        self.contour_position = []
        self.contour_pixel = []

    def create_contours(self):
        """
        Trace pixel contours, then convert them to world contours.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            (contour_pixel, contour_position): (n, 3) voxel-index arrays and
            the matching (n, 3) world arrays in mm.

        Examples
        --------
        >>> import numpy as np
        >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
        >>> mask[2:6, 4:12, 4:12] = 1
        >>> helper = MaskToContour(mask=mask, spacing=(1, 1, 2), origin=(0, 0, 0), matrix=np.eye(3))
        >>> pixel, position = helper.create_contours()
        >>> pixel[0][:, 2]                       # slice index
        array([2, 2, 2, 2], dtype=int32)
        >>> position[0][:, 2]                    # world z in mm
        array([4., 4., 4., 4.])
        """
        self.compute_pixel()
        if (self.spacing, self.origin, self.matrix) is not None:
            self.compute_position()

        return self.contour_pixel, self.contour_position

    def compute_pixel(self):
        """
        Extract outer contours from each non-empty slice into `contour_pixel`.

        Returns
        -------
        None

        Examples
        --------
        Axial (default):

        >>> import numpy as np
        >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
        >>> mask[2:6, 4:12, 4:12] = 1
        >>> helper = MaskToContour(mask=mask)
        >>> helper.compute_pixel()
        >>> len(helper.contour_pixel)
        4

        Coronal (one contour per y slice):

        >>> helper = MaskToContour(mask=mask, plane='coronal')
        >>> helper.compute_pixel()
        >>> len(helper.contour_pixel)
        8
        """
        if self.plane.lower() == "axial":
            axis = 0
        elif self.plane.lower() == "coronal":
            axis = 1
        else:
            axis = 2

        num_slices = self.mask.shape[axis]
        for i in range(num_slices):
            if axis == 0:
                slice_2d = self.mask[i, :, :]
            elif axis == 1:
                slice_2d = self.mask[:, i, :]
            else:
                slice_2d = self.mask[:, :, i]

            slice_2d = (slice_2d > 0).astype(np.uint8) * 255
            if np.count_nonzero(slice_2d) == 0:
                continue

            contours, _ = cv2.findContours(slice_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) > 2:
                    contour = contour.squeeze(1)

                    n = contour.shape[0]
                    contour_xyz = np.zeros((n, 3), dtype=np.int32)
                    if self.plane.lower() == "axial":
                        contour_xyz[:, 0] = contour[:, 0]
                        contour_xyz[:, 1] = contour[:, 1]
                        contour_xyz[:, 2] = i

                    elif self.plane.lower() == "coronal":
                        contour_xyz[:, 0] = contour[:, 0]
                        contour_xyz[:, 1] = i
                        contour_xyz[:, 2] = contour[:, 1]

                    else:
                        contour_xyz[:, 0] = i
                        contour_xyz[:, 1] = contour[:, 0]
                        contour_xyz[:, 2] = contour[:, 1]

                    self.contour_pixel.append(contour_xyz)

    def compute_position(self):
        """
        Convert `contour_pixel` to world coordinates in `contour_position`.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> mask = np.zeros((8, 16, 16), dtype=np.uint8)
        >>> mask[2:6, 4:12, 4:12] = 1
        >>> helper = MaskToContour(
        ...     mask=mask, spacing=(0.5, 0.5, 2), origin=(-100, -100, 50), matrix=np.eye(3),
        ... )
        >>> helper.compute_pixel()
        >>> helper.compute_position()
        >>> helper.contour_position[0][0]        # pixel (4, 4, 2)
        array([-98., -98.,  54.])
        """
        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = self.matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = self.matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = self.matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        for ii, pix in enumerate(self.contour_pixel):
            p_concat = np.concatenate((pix, np.ones((pix.shape[0], 1))), axis=1)
            self.contour_position += [p_concat.dot(pixel_to_position_matrix.T)[:, :3]]


class ModelToMask(object):
    """
    Convert one or more surface models into a volumetric mask.

    The grid is sized automatically from the combined model bounds plus a
    5-voxel buffer on every side. The mask is either empty (default) or
    filled from axial cuts of the models.

    Parameters
    ----------
    models : list[pv.PolyData]
        Surface models in mm.

    origin : list[float] | None
        Grid origin (x, y, z). Recomputed from the bounds.

    spacing : list[float] | None
        Voxel spacing (x, y, z). Set automatically to [1, 1, 3] or [1, 1, 5]
        for models that fit, otherwise must be given.

    dims : list[int] | None
        Mask shape (z, y, x). Recomputed from the bounds.

    slice_locations : list[float] | None
        z positions of the mask slices. Recomputed from the bounds.

    matrix : np.ndarray | None
        Transformation matrix for the models. Identity if None.

    empty_array : bool
        If True, only an empty mask of the right shape is made. If False,
        the mask is filled with the model cross-sections.

    convert : bool
        If True, compute bounds, contours and mask during construction.

    Examples
    --------
    Empty mask sized to fit a model:

    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.roi.converters import ModelToMask
    >>> helper = ModelToMask(models=[pv.Sphere(radius=20.0)])
    >>> helper.spacing
    [1, 1, 3]
    >>> int(helper.mask.max())
    0

    Filled mask from two models:

    >>> models = [pv.Sphere(radius=10.0), pv.Sphere(radius=10.0, center=(30, 0, 0))]
    >>> helper = ModelToMask(models=models, empty_array=False)
    >>> helper.mask.ndim
    3
    """
    def __init__(self, models, origin=None, spacing=None, dims=None, slice_locations=None, matrix=None,
                 empty_array=True, convert=True):
        """
        Store the models and settings, and optionally run the full conversion.

        Examples
        --------
        Run each step manually:

        >>> import pyvista as pv
        >>> helper = ModelToMask(models=[pv.Sphere(radius=20.0)], empty_array=False, convert=False)
        >>> helper.compute_bounds()
        >>> helper.compute_contours()
        >>> helper.compute_mask()
        """
        self.models = models
        self.empty_array = empty_array

        self.spacing = spacing
        self.origin = origin
        self.dims = dims
        self.slice_locations = slice_locations

        if matrix is None:
            self.matrix = np.identity(4)
        else:
            self.matrix = matrix

        self.bounds = None

        self.contours = []
        self.mask = None

        if convert:
            self.compute_bounds()
            self.compute_contours()
            self.compute_mask()

    def compute_bounds(self):
        """
        Compute the grid bounds, origin, slice locations and mask dimensions.

        The combined bounding box of all models is padded by 5 voxels on each
        axis. If the models fit in 512 x 512 mm in-plane, the spacing is set
        to [1, 1, 3] (< 450 mm tall) or [1, 1, 5] (< 750 mm tall).

        Returns
        -------
        None

        Examples
        --------
        >>> import pyvista as pv
        >>> helper = ModelToMask(models=[pv.Sphere(radius=20.0)], convert=False)
        >>> helper.compute_bounds()
        >>> helper.spacing
        [1, 1, 3]
        >>> helper.bounds[4], helper.bounds[5]   # z: -20 - 15, 20 + 15
        (-35, 35)
        """
        model_bounds = [model.GetBounds() for model in self.models]
        model_min = np.min(model_bounds, axis=0)
        model_max = np.max(model_bounds, axis=0)

        model_min_max = [model_min[0], model_max[1], model_min[2], model_max[3], model_min[4], model_max[5]]

        if model_min_max[1] - model_min_max[0] < 512 and model_min_max[3] - model_min_max[2] < 512:
            if model_min_max[5] - model_min_max[4] < 450:
                self.spacing = [1, 1, 3]

            elif model_min_max[5] - model_min_max[4] < 750:
                self.spacing = [1, 1, 5]

        if self.spacing is not None:
            self.bounds = [int(model_min_max[0] - 5 * self.spacing[0]), int(model_min_max[1] + 5 * self.spacing[0]),
                           int(model_min_max[2] - 5 * self.spacing[1]), int(model_min_max[3] + 5 * self.spacing[1]),
                           int(model_min_max[4] - 5 * self.spacing[2]), int(model_min_max[5] + 5 * self.spacing[2])]
            self.origin = [self.bounds[0], self.bounds[2], self.bounds[4]]

            self.slice_locations = [i for i in range(self.bounds[4], self.bounds[5], self.spacing[2])]
            self.dims = [len(self.slice_locations), self.bounds[3] - self.bounds[2] + 1, self.bounds[1] - self.bounds[0] + 1]

    def compute_contours(self):
        """
        Cut each model with axial planes at `slice_locations`.

        Points are stored in pixel units relative to the grid. Slices outside
        a model get an empty list, so `self.contours[model][slice]` always
        lines up with the mask slices.

        Returns
        -------
        None

        Examples
        --------
        >>> import pyvista as pv
        >>> helper = ModelToMask(models=[pv.Sphere(radius=20.0)], convert=False)
        >>> helper.compute_bounds()
        >>> helper.compute_contours()
        >>> len(helper.contours[0]) == len(helper.slice_locations)
        True
        """
        for model in self.models:
            com = model.center
            org_bounds = model.GetBounds()

            model_contours = []
            for s in self.slice_locations:
                if org_bounds[4] < s < org_bounds[5]:
                    hold_contour = model.slice(normal='z', origin=[com[0], com[1], s])
                    model_contours.append((np.asarray(hold_contour.points)[:, 0:2] -
                                     (self.bounds[0], self.bounds[2])) / (self.spacing[0:2]))
                else:
                    model_contours.append([])

            self.contours.append(model_contours)

    def compute_mask(self):
        """
        Create the int8 mask, filling it from the contours if `empty_array` is False.

        Overlapping models are summed, so a voxel inside two models has
        value 2.

        Returns
        -------
        None

        Examples
        --------
        >>> import pyvista as pv
        >>> helper = ModelToMask(models=[pv.Sphere(radius=20.0)], empty_array=False, convert=False)
        >>> helper.compute_bounds()
        >>> helper.compute_contours()
        >>> helper.compute_mask()
        >>> helper.mask.dtype
        dtype('int8')
        """
        self.mask = np.zeros((self.dims[0], self.dims[1], self.dims[2]))
        if not self.empty_array:
            for ii, model in enumerate(self.models):

                model_contours = self.contours[ii]
                for jj, s in enumerate(self.slice_locations):
                    if len(model_contours[jj]) > 0:
                        frame = np.zeros((self.dims[1], self.dims[2]))
                        cv2.fillPoly(frame, np.array([model_contours[jj]], dtype=np.int32), 1)
                        self.mask[jj, :, :] = self.mask[jj, :, :] + frame

        self.mask = self.mask.astype(np.int8)

    def save_image(self, export_path):
        """
        Save the mask with SimpleITK, using the grid spacing and origin.

        Parameters
        ----------
        export_path : str
            Output file path. The extension picks the format (.nii.gz, .mhd,
            .nrrd, ...).

        Returns
        -------
        None

        Examples
        --------
        >>> import pyvista as pv
        >>> helper = ModelToMask(models=[pv.Sphere(radius=20.0)], empty_array=False)
        >>> helper.save_image('C:/Data/sphere_mask.nii.gz')

        Save as MetaImage:

        >>> helper.save_image('C:/Data/sphere_mask.mhd')
        """
        image = sitk.GetImageFromArray(self.mask)
        image.SetSpacing(self.spacing)
        image.SetOrigin([self.bounds[0], self.bounds[2], self.bounds[4]])
        sitk.WriteImage(image, export_path)