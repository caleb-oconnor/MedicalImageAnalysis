"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Provides display transformations, image reslicing, and rigid registration
    management (including VTK and Open3D ICP alignments) for multi-modal medical imaging.

Structure:
    - Display: Manages voxel-to-world coordination, spacing, slicing coordinates, and pixel metrics.
    - Rigid: Coordinates rigid alignment transformations, ROI mapping, and registration parameters.
"""

import os
import copy

import numpy as np
import pandas as pd

import vtk
from vtkmodules.util import numpy_support

from pydicom.uid import generate_uid
from scipy.spatial.transform import Rotation

from ..utils.rigid.icp import ICP
from ..data import Data


class Display(object):
    """
    Handles coordinate spaces, slice calculations, and multi-planar view metrics
    for registered volumes.

    Parameters
    ----------
    rigid : Rigid
        The parent rigid registration tracking instance.
    """
    def __init__(self, rigid):
        self.rigid = rigid

        self.origin = None
        self.spacing = None
        self.array = None
        self.matrix = np.identity(4)

        self.slice_location = [0, 0, 0]
        self.scroll_max = [0, 0, 0]
        self.offset = {'Axial': [0, 0], 'Coronal': [0, 0], 'Sagittal': [0, 0]}
        self.misc = {}

    def compute_array_slice(self, slice_plane):
        """
        Extracts a single 2D double-precision array slice along a specified viewport plane.

        Parameters
        ----------
        slice_plane : str
            The target viewing orientation. Must be 'Axial', 'Coronal', or 'Sagittal'.

        Returns
        -------
        numpy.ndarray or None
            The extracted 2D slice cast to float64, or None if outside bounds.
        """
        array_slice = None
        if slice_plane == 'Axial':
            if 0 <= self.slice_location[0] < self.array.shape[0]:
                array_slice = self.array[self.slice_location[0], :, :].astype(np.double)

        elif slice_plane == 'Coronal':
            if 0 <= self.slice_location[1] < self.array.shape[1]:
                array_slice = self.array[:, self.slice_location[1], :].astype(np.double)

        else:
            if 0 <= self.slice_location[2] < self.array.shape[2]:
                array_slice = self.array[:, :, self.slice_location[2]].astype(np.double)

        return array_slice

    def compute_offset(self):
        """
        Computes 2D viewport origin spacing offsets relative to the parent images.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.rigid.inverse:
            pos = Data.image[self.rigid.moving_name].origin
        else:
            pos = Data.image[self.rigid.reference_name].origin

        self.offset['Axial'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
        self.offset['Axial'][1] = (self.origin[1] - pos[1]) / self.spacing[1]
        self.offset['Coronal'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
        self.offset['Coronal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]
        self.offset['Sagittal'][0] = (self.origin[1] - pos[1]) / self.spacing[1]
        self.offset['Sagittal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]

    def compute_matrix_pixel_to_position(self):
        """
        Generates a 4x4 homogeneous transform mapping voxel coordinates to 3D world space.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            A 4x4 coordinate projection matrix in float32 format.
        """
        if self.rigid.inverse:
            matrix = copy.deepcopy(Data.image[self.rigid.reference_name].matrix)
        else:
            matrix = copy.deepcopy(Data.image[self.rigid.moving_name].matrix)

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
        """
        Generates a 4x4 homogeneous transform mapping 3D world space down to voxel indices.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            A 4x4 inverse coordinate projection matrix in float32 format.
        """
        if self.rigid.inverse:
            matrix = copy.deepcopy(Data.image[self.rigid.reference_name].matrix)
        else:
            matrix = copy.deepcopy(Data.image[self.rigid.moving_name].matrix)

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / self.spacing[0]
        hold_matrix[1, :] = matrix[1, :] / self.spacing[1]
        hold_matrix[2, :] = matrix[2, :] / self.spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        return position_to_pixel_matrix

    def compute_mesh_slice(self, roi_name=None, location=None, slice_plane=None, return_pixel=False):
        """
        Slices a 3D ROI surface mesh with a plane equation to yield 2D contours.

        Parameters
        ----------
        roi_name : str, optional
            The lookup key identifier for the region of interest.
        location : array_like, optional
            A 3D spatial vector tracking the intersection point of the plane.
        slice_plane : str, optional
            The slicing view orientation. Options: 'Axial', 'Coronal', 'Sagittal'.
        return_pixel : bool, default False
            If True, transforms output coordinates into 2D display pixel indices.

        Returns
        -------
        list or pyvista.PolyData
            List of 2D coordinates if return_pixel is True, otherwise a clipped VTK mesh object.
        """
        if self.rigid.rois[roi_name] is None:
            self.rigid.update_rois(roi_name=roi_name)

        if slice_plane == 'Axial':
            normal = self.matrix[:3, 2]
        elif slice_plane == 'Coronal':
            normal = self.matrix[:3, 1]
        else:
            normal = self.matrix[:3, 0]

        roi_slice = self.rigid.rois[roi_name].slice(normal=normal, origin=location)

        if return_pixel:
            if roi_slice.number_of_points > 0:
                roi_strip = roi_slice.strip(max_length=10000000)

                position = [np.asarray(c.points) for c in roi_strip.cell]
                pixels = self.convert_position_to_pixel(position=position)
                pixel_corrected = []
                for pixel in pixels:

                    if slice_plane in 'Axial':
                        pixel_reshape = pixel[:, :2]
                        pixel_corrected += [np.asarray([pixel_reshape[:, 0], pixel_reshape[:, 1]]).T]

                    elif slice_plane == 'Coronal':
                        pixel_reshape = np.column_stack((pixel[:, 0], pixel[:, 2]))
                        pixel_corrected += [pixel_reshape]

                    else:
                        pixel_reshape = pixel[:, 1:]
                        pixel_corrected += [pixel_reshape]

                return pixel_corrected

            else:
                return []

        else:
            return roi_slice

    def compute_reslice(self):
        """
        Executes volume reslicing back down to pixel spatial matrix representations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        image = self.rigid.create_image()

        self.origin = np.asarray(image.GetOrigin())
        self.spacing = image.GetSpacing()
        dimensions = image.GetDimensions()

        scalars = image.GetPointData().GetScalars()
        self.array = numpy_support.vtk_to_numpy(scalars).reshape(dimensions[2], dimensions[1], dimensions[0])

        self.compute_offset()
        self.compute_scroll_max()

    def compute_slice_location(self, position=None):
        """
        Maps a 3D world space vector position back to internal multi-axis index positions.

        Parameters
        ----------
        position : array_like, optional
            A 3D spatial position vector. Defaults to the tracking data origin points.

        Returns
        -------
        None
        """
        if position is None:
            if self.rigid.inverse:
                source_location = np.flip(Data.image[self.rigid.moving_name].display.slice_location)
                position = Data.image[self.rigid.moving_name].display.compute_index_positions(source_location)
            else:
                source_location = np.flip(Data.image[self.rigid.reference_name].display.slice_location)
                position = Data.image[self.rigid.reference_name].display.compute_index_positions(source_location)

        self.slice_location = np.flip(np.round((position - self.origin) / self.spacing).astype(np.int32))

    def compute_slice_origin(self, slice_plane):
        """
        Calculates the 3D position origin coordinate of a specific viewer matrix plane index.

        Parameters
        ----------
        slice_plane : str
            The viewing tracking plane. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        numpy.ndarray or None
            A 3D origin position coordinate, or None if requested locations exceed maximum bounds.
        """
        slice_origin = None
        if slice_plane == 'Axial' and 0 <= self.slice_location[0] <= self.scroll_max[0]:
            location = np.asarray([0, 0, self.slice_location[0]])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Coronal' and 0 <= self.slice_location[1] <= self.scroll_max[1]:
            location = np.asarray([0, self.slice_location[1], 0])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Sagittal' and 0 <= self.slice_location[2] <= self.scroll_max[2]:
            location = np.asarray([self.slice_location[2], 0, 0])
            slice_origin = self.origin + (location * self.spacing)

        return slice_origin

    def compute_scroll_max(self):
        """
        Calculates bound dimensional sizes of arrays to ensure tracking safety indices.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.scroll_max = [self.array.shape[0] - 1,
                           self.array.shape[1] - 1,
                           self.array.shape[2] - 1]

    def compute_vtk_slice(self, slice_plane):
        """
        Creates a vtkImageData instance tracking data for a standalone display view slice.

        Parameters
        ----------
        slice_plane : str
            The targeted viewing orientation. Must be 'Axial', 'Coronal', or 'Sagittal'.

        Returns
        -------
        vtk.vtkImageData or None
            The standalone VTK slice object data, or None if target coordinates match nothing.
        """
        if self.array is None:
            self.compute_reslice()
            self.compute_scroll_max()

        self.compute_slice_location()

        slice_array = None
        slice_origin = self.compute_slice_origin(slice_plane)
        array_shape = self.array.shape
        if slice_plane == 'Axial' and 0 <= self.slice_location[0] <= self.scroll_max[0]:
            slice_array = np.zeros((1, array_shape[1], array_shape[2]))
            slice_array[0, :, :] = self.array[self.slice_location[0], :, :]

        elif slice_plane == 'Coronal' and 0 <= self.slice_location[1] <= self.scroll_max[1]:
            slice_array = np.zeros((array_shape[0], 1, array_shape[2]))
            slice_array[:, 0, :] = self.array[:, self.slice_location[1], :]

        elif slice_plane == 'Sagittal' and 0 <= self.slice_location[2] <= self.scroll_max[2]:
            slice_array = np.zeros((array_shape[0], array_shape[1], 1))
            slice_array[:, :, 0] = self.array[:, :, self.slice_location[2]]

        vtk_image = None
        if slice_array is not None:
            vtk_image = vtk.vtkImageData()
            vtk_image.SetSpacing(self.spacing)
            vtk_image.SetDirectionMatrix(1, 0, 0, 0, 1, 0, 0, 0, 1)
            vtk_image.SetDimensions(np.flip(slice_array.shape))
            vtk_image.SetOrigin(slice_origin)
            vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(slice_array.flatten(order="C")))

        return vtk_image

    def convert_position_to_pixel(self, position=None):
        """
        Transforms a series of 3D spatial points into explicit pixel data coordinate indices.

        Parameters
        ----------
        position : list of numpy.ndarray, optional
            A collection of point coordinate array shapes to process.

        Returns
        -------
        list of numpy.ndarray
            An updated list containing 3D array index markers mapping onto display orientations.
        """
        if self.rigid.inverse:
            position_to_pixel_matrix = Data.image[self.rigid.moving_name].display.compute_matrix_position_to_pixel()
        else:
            position_to_pixel_matrix = Data.image[self.rigid.reference_name].display.compute_matrix_position_to_pixel()

        pixel = []
        for ii, pos in enumerate(position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            pixel_3_axis = p_concat.dot(position_to_pixel_matrix.T)[:, :3]
            pixel += [np.vstack((pixel_3_axis, pixel_3_axis[0, :]))]

        return pixel

    def update_slice_location(self, scroll, slice_plane):
        """
        Saves updated internal tracking index points for a localized viewport window plane.

        Parameters
        ----------
        scroll : int
            The absolute target pixel track slider position value.
        slice_plane : str
            The display viewport target tracker. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        None
        """
        if slice_plane == 'Axial':
            self.slice_location[0] = scroll
        elif slice_plane == 'Coronal':
            self.slice_location[1] = scroll
        else:
            self.slice_location[2] = scroll


class Rigid(object):
    """
    Coordinates 3D matrix-based rigid registration operations between tracking volumes.

    Parameters
    ----------
    reference_name : str
        Look-up reference key identification string for reference images.
    moving_name : str
        Look-up reference key identification string for moving target images.
    rigid_name : str, optional
        Unique assigned identifier name tracking this specific registration setup instance.
    roi_names : list of str, optional
        Target structural tracking tags. Defaults to `['Unknown']`.
    reference_sops : list, optional
        DICOM SOP Class Tracking identifiers matching reference series layers.
    moving_sops : list, optional
        DICOM SOP Class Tracking identifiers matching moving target series layers.
    reference_matrix : numpy.ndarray, optional
        A baseline fixed structural alignment matrix. Defaults to Identity.
    matrix : numpy.ndarray, optional
        The core active registration modification 4x4 matrix tracker. Defaults to Identity.
    combo_matrix : numpy.ndarray, optional
        An additive combined secondary structural step matrix transformation tracker. Defaults to Identity.
    combo_name : str, optional
        Identification name label matching any multi-stage composite transformations.
    """
    def __init__(self, reference_name, moving_name, rigid_name=None, roi_names=None, reference_sops=None,
                 moving_sops=None, reference_matrix=None, matrix=None, combo_matrix=None, combo_name=None):
        self.reference_name = reference_name
        self.moving_name = moving_name
        self.combo_name = combo_name
        self.rois = dict.fromkeys(Data.roi_list)
        self.local_uid = generate_uid()

        if roi_names is None:
            self.roi_names = ['Unknown']
        else:
            self.roi_names = roi_names

        if reference_matrix is None:
            self.reference_matrix = np.identity(4)
        else:
            self.reference_matrix = reference_matrix

        if matrix is None:
            self.matrix = np.identity(4)
        else:
            self.matrix = matrix

        if combo_matrix is None:
            self.combo_matrix = np.identity(4)
        else:
            self.combo_matrix = combo_matrix

        self.inverse = False
        self.slices = {'reference': ['All'], 'moving': ['All'], 'reference_sops': reference_sops,
                       'moving_sops': moving_sops}
        self.visual = {'reference': None, 'moving': None, 'opacity': 0.5, 'multicolor': None}

        self.misc = {}
        self.rotation_center = np.asarray([0, 0, 0])
        self.rigid_name = self.add_rigid(rigid_name)

        self.display = Display(self)
        if matrix is not None:
            self.update_rois()

    def add_rigid(self, rigid_name):
        """
        Saves the active registration initialization instances into globally accessible data scopes.

        Parameters
        ----------
        rigid_name : str or None
            A targeted name for tracker referencing. If None, automatically creates an informative name.

        Returns
        -------
        str
            The actual uniquely verified lookup key assigned to this tracking operation.
        """
        if rigid_name is None:
            if np.array_equal(self.combo_matrix, np.identity(4)):
                rigid_name = self.reference_name + '_' + self.moving_name
            else:
                rigid_name = self.reference_name + '_' + self.moving_name + '_combo'

            if rigid_name in Data.rigid_list:
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(rigid_name + '_' + str(n))
                    if new_name not in Data.rigid_list:
                        rigid_name = new_name
                        n = -100

        Data.rigid[rigid_name] = self
        Data.rigid_list += [rigid_name]

        return rigid_name

    def compute_aspect(self, slice_plane):
        """
        Calculates pixel rendering aspect ratios using voxel spacing dimension values.

        Parameters
        ----------
        slice_plane : str
            The target display viewport orientation ('Axial', 'Coronal', or 'Sagittal').

        Returns
        -------
        float
            The aspect ratio rounded down to exactly two decimal points.
        """
        if slice_plane == 'Axial':
            aspect = np.round(self.display.spacing[0] / self.display.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.display.spacing[0] / self.display.spacing[2], 2)
        else:
            aspect = np.round(self.display.spacing[1] / self.display.spacing[2], 2)

        return aspect

    def compute_icp_vtk(self, source_mesh, target_mesh, distance=1e-5, iterations=1000, landmarks=None,
                        com_matching=True, inverse=False, center=None):
        """
        Runs an Iterative Closest Point algorithm on mesh pairings via standard VTK backends.

        Parameters
        ----------
        source_mesh : pyvista.PolyData
            The stable source tracking spatial point cloud dataset mesh.
        target_mesh : pyvista.PolyData
            The floating destination alignment point cloud mesh tracking updates.
        distance : float, default 1e-5
            The threshold constraint tracking target minimum variance steps.
        iterations : int, default 1000
            The maximal computational cycle threshold iterations to try.
        landmarks : array_like, optional
            Explicit point tracking paired values ensuring localized regional priorities.
        com_matching : bool, default True
            Enables alignment matching starting values using Center of Mass matching parameters first.
        inverse : bool, default False
            If True, flips matrix operations inside target registration workflows.
        center : str, optional
            Sets specific origin balancing locations. If set to 'image', recalibrates around centers.

        Returns
        -------
        None
        """
        self.inverse = inverse
        if self.inverse:
            target_mesh.transform(self.matrix @ self.combo_matrix, inplace=True)
        else:
            target_mesh.transform(np.linalg.inv(self.matrix @ self.combo_matrix), inplace=True)

        icp = ICP(source_mesh, target_mesh)
        icp.compute_vtk(distance=distance, iterations=iterations, landmarks=landmarks, com_matching=com_matching,
                        inverse=inverse)

        if center == 'image':
            R_icp = np.asarray(icp.get_matrix(), dtype=float)
            old_center = np.array([0, 0, 0], dtype=float)
            new_center = np.array(Data.image[self.moving_name].compute_center(), dtype=float)

            T_neg = np.eye(4)
            T_neg[:3, 3] = -new_center
            T_pos = np.eye(4)
            T_pos[:3, 3] = new_center

            extra_rotation = np.eye(4)
            old_center_h = np.hstack([old_center, 1])
            new_center_h = np.hstack([new_center, 1])

            R_total = extra_rotation @ R_icp
            transformed_old_center = R_total @ old_center_h
            transformed_new_center = R_total @ new_center_h

            correction = (old_center_h - transformed_old_center) - (new_center_h - transformed_new_center)
            T_corr = np.eye(4)
            T_corr[:3, 3] = correction[:3]
            self.matrix = T_pos @ extra_rotation @ R_icp @ T_neg @ T_corr

        else:
            self.matrix = icp.get_matrix()

        self.update_rois()

    def compute_o3d(self, source_mesh, target_mesh, distance=10, iterations=1000, rmse=1e-7, fitness=1e-7,
                    method='point', com_matching=True, inverse=False, center=None):
        """
        Runs an Iterative Closest Point algorithm on mesh pairings via an Open3D computational backend.

        Parameters
        ----------
        source_mesh : pyvista.PolyData
            The stable source tracking spatial point cloud dataset mesh.
        target_mesh : pyvista.PolyData
            The floating destination alignment point cloud mesh tracking updates.
        distance : float, default 10
            The maximum correspondence search radius distance parameter.
        iterations : int, default 1000
            The maximal computational cycle threshold iterations to try.
        rmse : float, default 1e-7
            Root Mean Squared Error divergence tracking constraints.
        fitness : float, default 1e-7
            Overlapping verification matching target constraint values.
        method : str, default 'point'
            The surface approach metric to execute (e.g. 'point' or 'plane').
        com_matching : bool, default True
            Enables alignment matching starting values using Center of Mass matching parameters first.
        inverse : bool, default False
            If True, flips matrix operations inside target registration workflows.
        center : str, optional
            Sets specific origin balancing locations. If set to 'image', recalibrates around centers.

        Returns
        -------
        None
        """
        target_mesh.transform(self.matrix @ self.combo_matrix, inplace=True)

        icp = ICP(source_mesh, target_mesh)
        icp.compute_o3d(distance=distance, iterations=iterations, rmse=rmse, fitness=fitness, method=method,
                        com_matching=com_matching, inverse=inverse)

        if center == 'image':
            R_icp = np.asarray(icp.get_matrix(), dtype=float)
            old_center = np.array([0, 0, 0], dtype=float)
            new_center = np.array(Data.image[self.moving_name].compute_center(), dtype=float)

            T_neg = np.eye(4)
            T_neg[:3, 3] = -new_center
            T_pos = np.eye(4)
            T_pos[:3, 3] = new_center

            extra_rotation = np.eye(4)
            old_center_h = np.hstack([old_center, 1])
            new_center_h = np.hstack([new_center, 1])

            R_total = extra_rotation @ R_icp
            transformed_old_center = R_total @ old_center_h
            transformed_new_center = R_total @ new_center_h

            correction = (old_center_h - transformed_old_center) - (new_center_h - transformed_new_center)
            T_corr = np.eye(4)
            T_corr[:3, 3] = correction[:3]
            self.matrix = T_pos @ extra_rotation @ R_icp @ T_neg @ T_corr

        else:
            self.matrix = icp.get_matrix()

        self.update_rois()

    def copy_roi(self, roi_name=None):
        """
        Clones and projects an ROI structural mesh model alignment into different volume spaces.

        Parameters
        ----------
        roi_name : str, optional
            The targeted structure lookup key label mapping to the active tracking item.

        Returns
        -------
        None
        """
        if roi_name in list(self.rois.keys()):
            reference_roi = Data.image[self.reference_name].rois[roi_name]
            moving_roving = Data.image[self.moving_name].rois[roi_name]
            if self.inverse and self.rois[roi_name] is not None:
                reference_roi.mesh = self.rois[roi_name].transform(np.linalg.inv(self.matrix @ self.combo_matrix),
                                                                   inplace=False)
            elif reference_roi.mesh is not None:
                moving_roving.mesh = reference_roi.mesh.transform(self.matrix @ self.combo_matrix, inplace=False)
                self.update_rois(roi_name=roi_name)

    def create_image(self):
        """
        Applies active alignment transformation values to synthesize an explicit vtkImageData volume output.

        Parameters
        ----------
        None

        Returns
        -------
        vtk.vtkImageData
            The processed and resliced 3D image volume ready for viewport rendering.
        """
        if self.inverse:
            ref = self.moving_name
            mov = self.reference_name
        else:
            ref = self.reference_name
            mov = self.moving_name

        matrix_reshape = Data.image[mov].matrix.reshape(1, 9)[0]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(Data.image[mov].spacing)
        vtk_image.SetDirectionMatrix(matrix_reshape)
        vtk_image.SetDimensions(np.flip(Data.image[mov].array.shape))
        vtk_image.SetOrigin(Data.image[mov].origin)
        vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(Data.image[mov].array.flatten(order="C")))

        matrix = self.matrix @ self.combo_matrix
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, matrix[i, j])

        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_matrix)
        if self.inverse:
            transform.Inverse()

        vtk_reslice = vtk.vtkImageReslice()
        vtk_reslice.SetInputData(vtk_image)
        vtk_reslice.SetResliceTransform(transform)
        vtk_reslice.SetInterpolationModeToLinear()
        vtk_reslice.SetOutputSpacing(Data.image[ref].spacing)
        vtk_reslice.SetOutputDirection(1, 0, 0, 0, 1, 0, 0, 0, 1)
        vtk_reslice.AutoCropOutputOn()
        vtk_reslice.SetBackgroundLevel(-3001)
        vtk_reslice.Update()

        return vtk_reslice.GetOutput()

    def export_image(self, path=None):
        """
        Writes the current aligned 3D dataset out to disk as an mhd/raw MetaImage format pair.

        Parameters
        ----------
        path : str, optional
            The targeted filename destination write path string.

        Returns
        -------
        None
        """
        if self.moving_name is not None and path is not None:
            image = self.create_image()

            writer = vtk.vtkMetaImageWriter()
            writer.SetInputData(image)
            writer.SetFileName(path)
            writer.Write()

    def pre_alignment(self, superior=False, center=False, origin=False):
        """
        Applies rapid programmatic initializations to roughly align spatial orientation metrics.

        Parameters
        ----------
        superior : bool, default False
            Executes matching alignment targeting cranial top bounding parameters.
        center : bool, default False
            Executes concentric 3D center matching steps.
        origin : bool, default False
            Forces coordinate matching origins via translation offsets directly.

        Returns
        -------
        None
        """
        if superior:
            pass
        elif center:
            pass
        elif origin:
            self.matrix[:3, 3] = Data.image[self.moving_name].origin - Data.image[self.reference_name].origin

    def retrieve_angles(self, order='ZXY'):
        """
        Extracts Euler orientation values from the active registration transformation components.

        Parameters
        ----------
        order : str, default 'ZXY'
            The specific axes rotation order tracking matrix factorization.

        Returns
        -------
        numpy.ndarray
            A 3D spatial rotation angle vector formatted in degrees.
        """
        rotation = Rotation.from_matrix(self.matrix[:3, :3])
        return rotation.as_euler(order, degrees=True)

    def retrieve_array_plane(self, slice_plane, solo=None, position=None):
        """
        Fetches the active numeric matrix layer mapped to a specific viewport rendering view orientation.

        Parameters
        ----------
        slice_plane : str
            The viewer window viewing target profile ('Axial', 'Coronal', 'Sagittal').
        solo : bool, optional
            If True, skips running internal location auto-calculation metrics.
        position : array_like, optional
            Custom 3D coordinate values to center reslicing targets over.

        Returns
        -------
        numpy.ndarray or None
            A 2D array representation slice dataset, or None if parameters map outside targets.
        """
        if self.display.array is None:
            self.display.compute_reslice()
            self.display.compute_scroll_max()

        if solo is None:
            self.display.compute_slice_location(position=position)

        return self.display.compute_array_slice(slice_plane=slice_plane)

    def retrieve_center(self):
        """
        Calculates 3D spatial coordinate centers across composite structural transformations.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            A 3D vector coordinates array tracking the calculated geometric transform origin.
        """
        if self.inverse:
            image_name = self.moving_name
        else:
            image_name = self.reference_name

        original_center = Data.image[image_name].compute_center()
        center_h = np.array([original_center[0], original_center[1], original_center[2], 1.0])
        center = (self.matrix @ self.combo_matrix @ center_h)[:3]

        return center

    def retrieve_offset(self, slice_plane):
        """
        Fetches the tracking pixel offset parameters associated with a chosen plane direction.

        Parameters
        ----------
        slice_plane : str
            The layout plane direction key target.

        Returns
        -------
        list of float
            A 2D bounding offset pixel coordinate tracking metric.
        """
        return self.display.offset[slice_plane]

    def retrieve_slice_location(self, slice_plane):
        """
        Returns active index tracking tracking parameters across display viewing configurations.

        Parameters
        ----------
        slice_plane : str
            The layout plane tracker targeting. Options: 'Axial', 'Coronal', 'Sagittal'.

        Returns
        -------
        int
            The localized index pixel layer offset parameter.
        """
        if slice_plane == 'Axial':
            return self.display.slice_location[0]

        elif slice_plane == 'Coronal':
            return self.display.slice_location[1]

        else:
            return self.display.slice_location[2]

    def retrieve_slice_position(self, slice_plane=None):
        """
        Converts pixel layout values to 3D world space coordinates.

        Parameters
        ----------
        slice_plane : str, optional
            The viewer orientation tracker identifier. If None, computes along all 3 tracking axes.

        Returns
        -------
        numpy.ndarray
            A 3D spatial position tracking vector.
        """
        pixel_to_position_matrix = self.display.compute_matrix_pixel_to_position()

        if slice_plane is None:
            location = np.asarray([self.display.slice_location[2],
                                   self.display.slice_location[1],
                                   self.display.slice_location[0], 1])
        else:
            if slice_plane == 'Axial':
                location = np.asarray([0, 0, self.display.slice_location[0], 1])
            elif slice_plane == 'Coronal':
                location = np.asarray([0, self.display.slice_location[1], 0, 1])
                print(location)
            else:
                location = np.asarray([self.display.slice_location[2], 0, 0, 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def retrieve_scroll_max(self, slice_plane):
        """
        Finds index tracking limits across multi-axis data array structures safely.

        Parameters
        ----------
        slice_plane : str
            The viewing tracking configuration layer selection target.

        Returns
        -------
        int
            The absolute index count capacity tracking maximum limit value.
        """
        if slice_plane == 'Axial':
            return self.display.scroll_max[0]

        elif slice_plane == 'Coronal':
            return self.display.scroll_max[1]

        else:
            return self.display.scroll_max[2]

    def retrieve_translation(self):
        """
        Exposes current alignment spatial position offsets.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            A 3D vector coordinates array tracking translation distance variables.
        """
        return self.matrix[:3, 3]

    def retrieve_vtk_slice(self, slice_plane):
        """
        Generates clean display objects for individual spatial layers directly via internal displays.

        Parameters
        ----------
        slice_plane : str
            The active viewport plane configuration layout key target.

        Returns
        -------
        vtk.vtkImageData
            The processed display object configuration data layer.
        """
        return self.display.compute_vtk_slice(slice_plane)

    def save_rigid(self, path):
        """
        Serializes active internal class settings directly out into picked DataFrame file formats.

        Parameters
        ----------
        path : str
            The system directory output target location to export towards.

        Returns
        -------
        None
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['rois', 'pois', 'display']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))

    def update_rotation(self, center=None, r_x=0, r_y=0, r_z=0):
        """
        Applies incremental Euler adjustments around a focal origin space.

        Parameters
        ----------
        center : array_like, optional
            A 3D spatial rotation center point coordinates vector. Defaults to auto-calculated image centers.
        r_x : float, default 0
            The pitch rotation component tracking changes in degrees.
        r_y : float, default 0
            The roll rotation component tracking changes in degrees.
        r_z : float, default 0
            The yaw rotation component tracking changes in degrees.

        Returns
        -------
        None
        """
        if center is None:
            center = self.retrieve_center()

        R_mat = Rotation.from_euler('xyz', [r_x, r_y, r_z], degrees=True).as_matrix()
        R = np.identity(4)
        R[:3, :3] = R_mat

        T_neg = np.identity(4)
        T_neg[:3, 3] = -np.array(center)

        T_pos = np.identity(4)
        T_pos[:3, 3] = np.array(center)
        transform = T_pos @ R @ T_neg

        self.matrix = transform @ self.matrix

        self.display.compute_reslice()
        self.display.compute_scroll_max()
        self.update_rois()

    def update_translation(self, t_x=0, t_y=0, t_z=0):
        """
        Updates global transformation values with new structural translations.

        Parameters
        ----------
        t_x : float, default 0
            The linear displacement offset length traversing across X axes.
        t_y : float, default 0
            The linear displacement offset length traversing across Y axes.
        t_z : float, default 0
            The linear displacement offset length traversing across Z axes.

        Returns
        -------
        None
        """
        T = np.identity(4)
        T[0, 3] = t_x
        T[1, 3] = t_y
        T[2, 3] = t_z

        self.matrix = self.matrix @ T

        self.display.origin[0] -= t_x
        self.display.origin[1] -= t_y
        self.display.origin[2] -= t_z

        self.display.compute_offset()
        self.display.compute_scroll_max()
        self.update_rois()

    def update_rois(self, roi_name=None):
        """
        Re-evaluates surface mesh conversions across updated registration transformation settings.

        Parameters
        ----------
        roi_name : str, optional
            Target structural tracking label identifier to process. If None, refreshes all structural mappings.

        Returns
        -------
        None
        """
        for name in list(self.rois.keys()):
            if name not in Data.roi_list:
                del self.rois[name]

        for name in Data.roi_list:
            if name not in list(self.rois.keys()):
                self.rois[name] = None

        for name in Data.roi_list:
            if roi_name is None or name == roi_name:
                roi = Data.image[self.moving_name].rois[name]
                if roi.mesh is not None and roi.visible:
                    if self.inverse:
                        self.rois[name] = roi.mesh.transform(self.matrix @ self.combo_matrix, inplace=False)
                    else:
                        self.rois[name] = roi.mesh.transform(np.linalg.inv(self.matrix @ self.combo_matrix),
                                                             inplace=False)