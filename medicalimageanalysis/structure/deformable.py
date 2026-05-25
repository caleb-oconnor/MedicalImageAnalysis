"""
Morfeus Lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Deformable image registration management and multi-planar display slice processing
    for medical imaging datasets (CT/MR). This module handles displacement vector fields (DVFs),
    rigid transform updates, and ROI mesh warping workflows.

Structure:
    * Display: Handles pixel, slice, and mesh projections across Axial, Coronal, and Sagittal planes.
    * Deformable: Manages DVF computations, registration tracking, alignment corrections, and ROI transformations.
"""

import os
import copy

import numpy as np
import pandas as pd
import SimpleITK as sitk

from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

from ..data import Data
from ..utils.deformable.simpleitk import DeformableITK


class Display(object):
    """
    Handles coordinate mapping, data rendering, and slice extractions for viewing 3D volumes.

    This class decouples volumetric deformable properties from viewport orientation, mapping
    indices to physical coordinates across orthogonal imaging planes.

    Attributes
    ----------
    deformable : Deformable
        Parent deformable instance owning image metadata and DVF arrays.
    origin : list or tuple
        The physical world coordinate corresponding to index (0,0,0).
    spacing : list or tuple
        Voxel sizes in millimeters along the X, Y, and Z axes.
    array : list of np.ndarray
        Volumetric array instances captured at progressive transformation fractions.
    image : SimpleITK.Image
        Cached internal SimpleITK image reference.
    matrix : np.ndarray
        3x3 direction matrix relating voxel axes to physical spatial orientations.
    slice_location : list of int
        Current plane indices [Z, Y, X] selected for orthogonal viewing.
    scroll_max : list of int
        Maximum scroll index offsets allowed for each dimension.
    offset : dict
        Voxel coordinate translation shifts mapping the active array to a reference baseline image.
    misc : dict
        Storage area for arbitrary display properties or pipeline configurations.
    """

    def __init__(self, deformable):
        """
        Initializes a Display viewport tracking manager for an active alignment context.

        Parameters
        ----------
        deformable : Deformable
            Parent deformable dataset object to inspect or slice.
        """
        self.deformable = deformable

        self.origin = None
        self.spacing = None
        self.array = []
        self.image = None
        self.matrix = np.identity(3)

        self.slice_location = [0, 0, 0]
        self.scroll_max = None
        self.offset = {'Axial': [0, 0], 'Coronal': [0, 0], 'Sagittal': [0, 0]}
        self.misc = {}

        self.compute_scroll_max()

    def compute_array(self, slice_plane, portion=0):
        """
        Extracts a 2D intensity projection matrix matching a desired viewport orientation.

        Parameters
        ----------
        slice_plane : str
            The slicing orientation targeting array structures. Must be 'Axial', 'Coronal', or 'Sagittal'.
        portion : int, optional
            The target frame or incremental state index loaded inside `self.array`. Defaults to 0.

        Returns
        -------
        np.ndarray
            Floating-point 2D matrix representing structural voxel data across the selected section plane.
            Returns None if index definitions or bounds fall outside volume ranges.

        Examples
        --------
        >>> display_mgr.slice_location = [25, 0, 0]
        >>> axial_slice = display_mgr.compute_array('Axial')
        """
        array_slice = None
        if slice_plane == 'Axial':
            if 0 <= self.slice_location[0] < self.array[portion].shape[0]:
                array_slice = self.array[portion][self.slice_location[0], :, :].astype(np.double)

        elif slice_plane == 'Coronal':
            if 0 <= self.slice_location[1] < self.array[portion].shape[1]:
                array_slice = self.array[portion][:, self.slice_location[1], :].astype(np.double)

        else:
            if 0 <= self.slice_location[2] < self.array[portion].shape[2]:
                array_slice = self.array[portion][:, :, self.slice_location[2]].astype(np.double)

        return array_slice

    def compute_deformation(self, division=1):
        """
        Samples the displacement field continuously over fractional timeline increments to extract morphed configurations.

        Parameters
        ----------
        division : int, optional
            The total number of evaluation subdivisions evaluated to build progressive frame states. Defaults to 1.
        """
        for ii in range(division):
            ratio = (ii + 1) / division
            image = self.deformable.create_image(ratio=ratio)

            self.array += [sitk.GetArrayFromImage(image)]
            self.spacing = image.GetSpacing()
            self.origin = image.GetOrigin()

        self.compute_offset()

    def compute_grid(self, slice_plane='Axial', vector='x'):
        """
        Isolates directed scalar displacement components matching an explicit projection path.

        Parameters
        ----------
        slice_plane : str, optional
            Target spatial slicing plane. Must be 'Axial', 'Coronal', or 'Sagittal'. Defaults to 'Axial'.
        vector : str, optional
            Vector direction component to sample. Must be 'x', 'y', or 'z'. Defaults to 'x'.

        Returns
        -------
        np.ndarray
            Single-channel 2D matrix mapping local vector magnitudes (float32) intersecting the active viewer index.
        """
        if slice_plane == 'Axial':
            dvf_plane = self.deformable.dvf[self.slice_location[0], :, :, :]
        elif slice_plane == 'Coronal':
            dvf_plane = self.deformable.dvf[:, self.slice_location[1], :, :]
        else:
            dvf_plane = self.deformable.dvf[:, :, self.slice_location[2], :]

        if vector == 'x':
            dvf_vector = dvf_plane[:, :, 0]
        elif vector == 'y':
            dvf_vector = dvf_plane[:, :, 1]
        else:
            dvf_vector = dvf_plane[:, :, 2]

        return dvf_vector.astype(np.float32)

    def compute_matrix_pixel_to_position(self):
        """
        Calculates a 4x4 homogeneous matrix mapping discrete pixel indexes directly to continuous world coordinates.

        Returns
        -------
        np.ndarray
            Homogeneous 4x4 coordinate transformation array (float32).
        """
        matrix = copy.deepcopy(self.matrix)

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
        """
        Calculates a 4x4 homogeneous transform matrix mapping continuous physical spaces to matrix voxel indexes.

        Returns
        -------
        np.ndarray
            Homogeneous 4x4 mapping array (float32) handling spatial index inversions.
        """
        matrix = copy.deepcopy(self.matrix)

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
        Intersects 3D ROI structures using normal vectors to compute flat planar contours or local pixel contours.

        Parameters
        ----------
        roi_name : str, optional
            Key identifier pointing to the desired region of interest structure. Defaults to None.
        location : list or np.ndarray, optional
            Physical point intersection origin matching the cutting field. Defaults to None.
        slice_plane : str, optional
            Target viewing plane projection name. Must be 'Axial', 'Coronal', or 'Sagittal'. Defaults to None.
        return_pixel : bool, optional
            Converts structural 3D vertex contours back to local 2D view indexes if True. Defaults to False.

        Returns
        -------
        PolyData or list
            Returns a sliced surface model or a structured array list containing 2D cross-section coordinate steps.
        """
        if self.deformable.rois[roi_name] is None:
            self.deformable.update_rois(roi_name=roi_name)

        if slice_plane == 'Axial':
            normal = self.matrix[:3, 2]
        elif slice_plane == 'Coronal':
            normal = self.matrix[:3, 1]
        else:
            normal = self.matrix[:3, 0]

        roi_slice = self.deformable.rois[roi_name].slice(normal=normal, origin=location)

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

    def compute_offset(self):
        """
        Aligns internal display coordinate matrices to a master background image dataset profile.
        """
        if self.deformable.reference_name is not None:
            pos = Data.image[self.deformable.reference_name].origin

            self.offset['Axial'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
            self.offset['Axial'][1] = (self.origin[1] - pos[1]) / self.spacing[1]
            self.offset['Coronal'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
            self.offset['Coronal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]
            self.offset['Sagittal'][0] = (self.origin[1] - pos[1]) / self.spacing[1]
            self.offset['Sagittal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]

    def compute_slice_location(self, position=None):
        """
        Updates viewing indices by determining the closest voxel matrix intersection matching a physical landmark location.

        Parameters
        ----------
        position : list or np.ndarray, optional
            Physical 3D location target coordinates. If omitted, values sync
            to structural values loaded inside `Data.image`. Defaults to None.
        """
        if position is None:
            source_location = np.flip(Data.image[self.deformable.reference_name].display.slice_location)
            position = Data.image[self.deformable.reference_name].display.compute_index_positions(source_location)
        self.slice_location = np.flip(np.round((position - self.origin) / self.spacing).astype(np.int32))

    def compute_slice_origin(self, slice_plane):
        """
        Extracts continuous world position coordinates matching the precise origin boundaries of a selected slice index.

        Parameters
        ----------
        slice_plane : str
            Target viewing axis. Must be 'Axial', 'Coronal', or 'Sagittal'.

        Returns
        -------
        np.ndarray
            Vector containing the 3D continuous origin coordinate point, or None if conditions violate index limits.
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
        Calculates spatial matrix limits to restrict viewport index steps within valid image array data blocks.
        """
        if len(self.array) == 0:
            self.scroll_max = self.deformable.dimensions - 1
        else:
            self.scroll_max = [self.array[-1].shape[0] - 1,
                               self.array[-1].shape[1] - 1,
                               self.array[-1].shape[2] - 1]

    def convert_position_to_pixel(self, position=None):
        """
        Maps continuous physical 3D vertices into structural array pixel coordinates.

        Parameters
        ----------
        position : list of np.ndarray, optional
            Collection of point arrays containing physical spatial entries. Defaults to None.

        Returns
        -------
        list of np.ndarray
            Modulated coordinate indexes tailored for plotting canvas elements.
        """
        position_to_pixel_matrix = self.compute_matrix_position_to_pixel()

        pixel = []
        for ii, pos in enumerate(position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            pixel_3_axis = p_concat.dot(position_to_pixel_matrix.T)[:, :3]
            pixel += [np.vstack((pixel_3_axis, pixel_3_axis[0, :]))]

        return pixel

    def update_slice_location(self, scroll, slice_plane):
        """
        Manually forces a new tracking row or column view step offset index for display pipelines.

        Parameters
        ----------
        scroll : int
            The target matrix slice level index step parameter to set.
        slice_plane : str
            Orientation projection target. Must be 'Axial', 'Coronal', or 'Sagittal'.
        """
        if slice_plane == 'Axial':
            self.slice_location[0] = scroll
        elif slice_plane == 'Coronal':
            self.slice_location[1] = scroll
        else:
            self.slice_location[2] = scroll


class Deformable(object):
    """
    Manages non-rigid transformation workflows, holding non-rigid fields, parameters, and morph functions.

    This class provides wrappers around SimpleITK registration procedures, optimizing structural profiles
    and propagating vector field movements across biological regions of interest.

    Attributes
    ----------
    reference_name : str
        Key label matching the static reference/fixed baseline scan profile.
    reference_sops : list
        DICOM Service-Object Pair Instance UIDs identifying source reference items.
    moving_name : str
        Key label identifying the structural target image volume undergoing registration adjustments.
    moving_sops : list
        DICOM Instance UIDs mapping back to components within the moving scan profile.
    roi_names : list of str
        Descriptive structural ROI names associated with the target datasets.
    rigid_rois : dict
        Intermediary cache tracking rigid positional adjustments prior to applying non-rigid deformations.
    rois : dict
        Active morph result paths mapping structured mesh entities to their non-rigid deformation positions.
    reference_mesh : PolyData
        Master structural surface reference asset definition tracking target anatomy elements.
    moving_mesh : PolyData
        Deformable structural mesh surface parameters tracking moving anatomical targets.
    local_uid : str
        Process UID token generated for system pipeline identification tasks.
    modality : str
        Medical imaging scan parameter type (e.g., 'CT', 'MR').
    dvf : np.ndarray
        Displacement Vector Field tensor tracking displacement coordinates across matrix nodes.
    origin : list or tuple
        Continuous spatial world coordinates matching index origin foundations.
    spacing : list or tuple
        Scale dimensions mapping voxels to metrics in physical dimensions.
    dimensions : tuple or list
        Voxel count indicators structural bounds.
    rigid_matrix : np.ndarray
        4x4 matrix executing introductory linear or rigid spatial corrections.
    deformable_name : str
        Registry ID tracking active parameters in runtime contexts.
    display : Display
        Viewing interface mapping structures across viewport projections.
    """

    def __init__(self, dvf=None, origin=None, spacing=None, dimensions=None, roi_names=None,
                 rigid_matrix=None, dvf_matrix=None, registration_name=None, reference_name=None, moving_name=None,
                 reference_sops=None, moving_sops=None, reference_meshes=None, moving_meshes=None):
        """
        Initializes a Deformable registration record holding transformation parameters.
        """
        self.reference_name = reference_name
        self.reference_sops = reference_sops
        self.moving_name = moving_name
        self.moving_sops = moving_sops
        self.roi_names = roi_names
        self.rigid_rois = dict.fromkeys(Data.roi_list)
        self.rois = dict.fromkeys(Data.roi_list)
        self.reference_mesh = reference_meshes
        self.moving_mesh = moving_meshes
        self.local_uid = generate_uid()

        self.modality = None
        if dvf_matrix is not None:
            if np.allclose(dvf_matrix, np.identity(3), atol=1e-3):
                self.dvf = dvf
                self.origin = origin
                self.spacing = spacing
                self.dimensions = dimensions
            else:
                self.dvf, self.spacing, self.origin, self.dimensions = self.correct_dvf_direction(dvf, spacing, origin,
                                                                                                  dvf_matrix)

        else:
            self.dvf = dvf
            self.origin = origin
            self.spacing = spacing
            self.dimensions = dimensions

        if rigid_matrix is None:
            self.rigid_matrix = np.identity(4)
        else:
            self.rigid_matrix = rigid_matrix

        self.deformable_name = self.add_deformable(registration_name)

        self.display = Display(self)
        if self.dvf is not None:
            self.update_rois()

    def add_deformable(self, deformable_name):
        """
        Registers this transformation instance inside the repository storage tracking framework.

        Parameters
        ----------
        deformable_name : str
            Preferred tracking title identity. If None, an identifier is generated.

        Returns
        -------
        str
            The final tracking registry label used to locate this specific deformable context.
        """
        if deformable_name is None:
            if self.reference_name is None and self.moving_name is None:
                deformable_name = 'DVF_Unknown'
            else:
                deformable_name = 'DVF_' + self.reference_name + '_' + self.moving_name

            if deformable_name in Data.deformable_list:
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(deformable_name + '_' + str(n))
                    if new_name not in Data.deformable_list:
                        deformable_name = new_name
                        n = -100

        Data.deformable[deformable_name] = self
        Data.deformable_list += [deformable_name]

        return deformable_name

    def compute_aspect(self, slice_plane):
        """
        Extracts a pixel aspect scaling ratio tailored for displaying accurate image proportions.

        Parameters
        ----------
        slice_plane : str
            Targeting orientation axis. Must be 'Axial', 'Coronal', or 'Sagittal'.

        Returns
        -------
        float
            Voxel scale dimension multiplier ratio.
        """
        if slice_plane == 'Axial':
            aspect = np.round(self.spacing[0] / self.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.spacing[0] / self.spacing[2], 2)
        else:
            aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        return aspect

    def compute_biomechanical(self):
        """
        Placeholder method for biomechanical simulation algorithms or finite element extensions.
        """
        pass

    def compute_bspline(self, modality_gradient=True, sigma=2, control_spacing=None, mesh_size=None, gradient=1e-5,
                        iterations=100, crop=5):
        """
        Runs a B-Spline deformable registration workflow using metrics managed via SimpleITK components.

        Parameters
        ----------
        modality_gradient : bool, optional
            Evaluates multi-modality cross-corrections if True. Defaults to True.
        sigma : float, optional
            Gaussian blur kernel size applied across structural mask boundaries. Defaults to 2.
        control_spacing : list, optional
            Physical distance mapping parameters defining B-Spline grid densities. Defaults to None.
        mesh_size : list, optional
            Mesh node dimension parameters shaping transform density. Defaults to None.
        gradient : float, optional
            Optimization termination step tolerance. Defaults to 1e-5.
        iterations : int, optional
            Optimization limits capping evaluation routines. Defaults to 100.
        crop : int, optional
            Boundary truncation thickness processing parameters stripping edges. Defaults to 5.

        Examples
        --------
        >>> deform_ctx.compute_bspline(iterations=250, sigma=3)
        """
        ref_image = Data.image[self.reference_name].create_sitk_image()
        mov_image = Data.image[self.moving_name].create_sitk_image()

        euler = sitk.Euler3DTransform()
        euler.SetMatrix(self.rigid_matrix[:3, :3].flatten().tolist())
        euler.SetTranslation(self.rigid_matrix[:3, 3])

        mov_resampled = sitk.Resample(mov_image, ref_image, euler, sitk.sitkLinear, 0.0, mov_image.GetPixelID())

        ref_mask = None
        mov_mask = None
        for roi_name in self.roi_names:
            ref_roi = Data.image[self.reference_name].rois[roi_name]
            mov_roi = Data.image[self.moving_name].rois[roi_name]
            if ref_roi.mesh is not None or ref_roi.contour_pixel is not None:
                if mov_roi.mesh is not None or mov_roi.contour_pixel is not None:
                    if ref_mask is None:
                        ref_mask = ref_roi.compute_mask()
                    else:
                        ref_mask = ref_mask + ref_roi.compute_mask()

                        if mov_mask is None:
                            mov_mask = mov_roi.compute_mask()
                        else:
                            mov_mask = mov_mask + mov_roi.compute_mask()

        deform_itk = DeformableITK(reference_image=ref_image, moving_image=mov_resampled, reference_mask=None,
                                   moving_mask=None)

        if Data.image[self.reference_name].modality != Data.image[self.moving_name].modality and modality_gradient:
            deform_itk.cross_modality_correction()

        if ref_mask is not None and mov_mask is not None:
            deform_itk.create_sitk_image(ref_mask, ref_image.origin, ref_image.spacing, ref_image.matrix, mask=True)
            deform_itk.create_sitk_image(mov_mask, mov_image.origin, mov_image.spacing, mov_image.matrix, reference=False, mask=True)

            if sigma is not None:
                deform_itk.blur_mask(sigma=sigma)

        deform_itk.resample()
        dvf_image = deform_itk.bspline(control_spacing=control_spacing, mesh_size=mesh_size, gradient=gradient,
                                       iterations=iterations, crop=crop)

        self.origin = dvf_image.GetOrigin()
        self.spacing = dvf_image.GetSpacing()
        self.dvf = sitk.GetArrayFromImage(dvf_image)

    def compute_demons(self, method=None, modality_gradient=True, sigma=2, smooth=True, std=1, iterations=50,
                       intensity_threshold=0.001, step=2.0, crop=5):
        """
        Executes dense fluid registration variants including standard, fast, or diffeomorphic Demons models.

        Parameters
        ----------
        method : str, optional
            Target alignment flavor flag. Options include 'Demons' or 'Diffeomorphic'. Defaults to None.
        modality_gradient : bool, optional
            Integrates cross-modality intensity matching corrections if True. Defaults to True.
        sigma : float, optional
            Scale settings smoothing binary structural mask surfaces. Defaults to 2.
        smooth : bool, optional
            Smooths displacement entries post-iteration if True. Defaults to True.
        std : float, optional
            Standard deviation governing Gaussian field smoothing processes. Defaults to 1.
        iterations : int, optional
            Iteration runtime loop execution threshold limit bounds. Defaults to 50.
        intensity_threshold : float, optional
            Minimum intensity change required to continue evaluation steps. Defaults to 0.001.
        step : float, optional
            Field translation step scale factor. Defaults to 2.0.
        crop : int, optional
            Boundary exclusion padding pixel counts. Defaults to 5.
        """
        ref = Data.image[self.reference_name]
        mov = Data.image[self.moving_name]

        deform_itk = DeformableITK()
        deform_itk.create_sitk_image(ref.array, ref.origin, ref.spacing, ref.matrix)
        deform_itk.create_sitk_image(mov.array, mov.origin, mov.spacing, mov.matrix, reference=False)

        if Data.image[self.reference_name].modality != Data.image[self.moving_name].modality and modality_gradient:
            deform_itk.cross_modality_correction()

        ref_mask = None
        mov_mask = None
        for roi_name in self.roi_names:
            ref_roi = Data.image[self.reference_name].rois[roi_name]
            mov_roi = Data.image[self.moving_name].rois[roi_name]
            if ref_roi.mesh is not None or ref_roi.contour_pixel is not None:
                if mov_roi.mesh is not None or mov_roi.contour_pixel is not None:
                    if ref_mask is None:
                        ref_mask = ref_roi.compute_mask()
                    else:
                        ref_mask = ref_mask + ref_roi.compute_mask()

                        if mov_mask is None:
                            mov_mask = mov_roi.compute_mask()
                        else:
                            mov_mask = mov_mask + mov_roi.compute_mask()

        if ref_mask is not None and mov_mask is not None:
            deform_itk.create_sitk_image(ref_mask, ref.origin, ref.spacing, ref.matrix, mask=True)
            deform_itk.create_sitk_image(mov_mask, mov.origin, mov.spacing, mov.matrix, reference=False, mask=True)

            if sigma is not None:
                deform_itk.blur_mask(sigma=sigma)

        deform_itk.resample()
        if method in ['Demons', 'demons']:
            dvf_image = deform_itk.demons(smooth=smooth, std=std, iterations=iterations,
                                          intensity_threshold=intensity_threshold, crop=crop)

        elif method in ['Diffeomorphic', 'diffeomorphic']:
            dvf_image = deform_itk.diffeomorphic(smooth=smooth, std=std, iterations=iterations,
                                                 intensity_threshold=intensity_threshold, crop=crop)

        else:
            dvf_image = deform_itk.fast_demons(smooth=smooth, std=std, iterations=iterations,
                                               intensity_threshold=intensity_threshold, step=step, crop=crop)

        self.origin = dvf_image.GetOrigin()
        self.spacing = dvf_image.GetSpacing()
        self.dvf = sitk.GetArrayFromImage(dvf_image)

    @staticmethod
    def correct_dvf_direction(dvf, spacing, origin, matrix):
        """
        Rotates a DVF to a new direction, adjusting vectors and origin so that the DVF
        stays aligned in physical space (rotation about the center of the volume).

        Parameters
        ----------
        dvf : np.ndarray
            Original displacement vector field matrix tensor.
        spacing : list or tuple
            Dimensional voxel sizing array parameters.
        origin : list or tuple
            Starting structural boundary spatial point.
        matrix : np.ndarray
            Target directional rotation transformation matrix mapping properties.

        Returns
        -------
        tuple
            Contains updated (dvf_rotated, spacing, origin_new, dimensions) structures.
        """
        D_new = np.identity(3)
        R = D_new @ np.linalg.inv(matrix)

        center_index = (np.flip(dvf.shape)[1:] - 1) / 2.0
        center_phys = origin + matrix @ (center_index * spacing)

        Z, Y, X, _ = dvf.shape
        dvf_flat = dvf.reshape(-1, 3).T
        dvf_rotated_flat = R @ dvf_flat
        dvf_rotated = dvf_rotated_flat.T.reshape(Z, Y, X, 3)

        center_phys_new = center_phys  # center should not move
        origin_new = center_phys_new - D_new @ (center_index * spacing)

        dimensions = dvf_rotated.shape[0:3]

        return dvf_rotated, spacing, origin_new, dimensions

    def create_image(self, ratio=1):
        """
        Applies combined rigid parameters and displacement fields to resample the moving image volume.

        Parameters
        ----------
        ratio : float, optional
            Displacement vector field scaling factor. Defaults to 1.

        Returns
        -------
        SimpleITK.Image
            Resampled and transformed image data volume.
        """
        R = self.rigid_matrix[:3, :3]
        t = self.rigid_matrix[:3, 3]

        affine = sitk.AffineTransform(3)
        affine.SetMatrix(R.flatten())
        affine.SetTranslation(t)

        ref_image = Data.image[self.reference_name].create_sitk_image()
        moving_image = Data.image[self.moving_name].create_sitk_image()

        resampled_image = sitk.Resample(
            moving_image,
            ref_image,
            transform=affine,
            interpolator=sitk.sitkLinear,
            defaultPixelValue=-3001,
            outputPixelType=moving_image.GetPixelID()
        )

        dvf_initial_image = sitk.GetImageFromArray(self.dvf, isVector=True)
        # dvf_initial_image = sitk.GetImageFromArray(ratio * self.dvf, isVector=True)
        dvf_initial_image.SetSpacing(self.spacing)
        dvf_initial_image.SetOrigin(self.origin)

        invert_filter = sitk.InvertDisplacementFieldImageFilter()
        invert_dvf = invert_filter.Execute(dvf_initial_image)
        dis_tx = sitk.DisplacementFieldTransform(sitk.Cast(invert_dvf, sitk.sitkVectorFloat64))

        return sitk.Resample(resampled_image, dis_tx, sitk.sitkLinear, -3001, resampled_image.GetPixelID())

    def export_image(self, path=None):
        """
        Saves the transformed moving image to a specified file path.

        Parameters
        ----------
        path : str, optional
            Target file system write location destination path. Defaults to None.
        """
        if self.moving_name is not None and path is not None:
            image = self.create_image()

            sitk.WriteImage(image, path)

    def retrieve_array_plane(self, slice_plane, solo=None, position=None, vector=None):
        """
        Extracts 2D array matrix representations matching an explicit slicing plane projection request.

        Parameters
        ----------
        slice_plane : str
            Target viewing axis. Must be 'Axial', 'Coronal', or 'Sagittal'.
        solo : bool, optional
            Disables coordinate indexing recalculations if true. Defaults to None.
        position : list, optional
            Alternative physical space alignment center options. Defaults to None.
        vector : str, optional
            Requests DVF flow fields instead of image structural grids if specified ('x', 'y', 'z'). Defaults to None.

        Returns
        -------
        np.ndarray
            Evaluated slice visualization array data.
        """
        if len(self.display.array) == 0:
            self.display.compute_deformation()
            self.display.compute_slice_location()

        if solo is None:
            self.display.compute_slice_location(position=position)

        if vector is None:
            return self.display.compute_array(slice_plane)
        elif vector == 'x':
            return self.display.compute_grid(slice_plane=slice_plane, vector=vector)
        elif vector == 'y':
            return self.display.compute_grid(slice_plane=slice_plane, vector=vector)
        elif vector == 'z':
            return self.display.compute_grid(slice_plane=slice_plane, vector=vector)
        else:
            return None

    def retrieve_grid(self, slice_plane='Axial', vector='x'):
        """
        Queries displacement values mapping across explicit slices.

        Parameters
        ----------
        slice_plane : str, optional
            Viewer selection slice identifier tracking. Defaults to 'Axial'.
        vector : str, optional
            Targeted vector direction key parameter channel ('x', 'y', 'z'). Defaults to 'x'.

        Returns
        -------
        np.ndarray
            Displacement scalar cross-section matrix.
        """
        return self.display.compute_grid(slice_plane=slice_plane, vector=vector)

    def retrieve_offset(self, slice_plane):
        """
        Extracts translation pixel tracking indices relative to baseline configurations.

        Parameters
        ----------
        slice_plane : str
            Slice viewing target selection option parameter.

        Returns
        -------
        list
            Spatial indices indicating offset values.
        """
        return self.display.offset[slice_plane]

    def retrieve_slice_location(self, slice_plane):
        """
        Queries the active integer index steps selected across viewport tracking variables.

        Parameters
        ----------
        slice_plane : str
            Target axis identifier name.

        Returns
        -------
        int
            Discrete array matrix location track marker step value.
        """
        if slice_plane == 'Axial':
            return self.display.slice_location[0]

        elif slice_plane == 'Coronal':
            return self.display.slice_location[1]

        else:
            return self.display.slice_location[2]

    def retrieve_slice_position(self, slice_plane=None):
        """
        Maps current view tracking indices back into continuous physical world 3D position spaces.

        Parameters
        ----------
        slice_plane : str, optional
            Target viewport selection name profile tracker. Defaults to None.

        Returns
        -------
        np.ndarray
            Continuous 3D world coordinate vector.
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
        Queries tracking boundary limits capping viewport indices.

        Parameters
        ----------
        slice_plane : str
            Target viewport plane indicator tracking name.

        Returns
        -------
        int
            Maximum allowed scroll index step.
        """
        if slice_plane == 'Axial':
            return self.display.scroll_max[0]

        elif slice_plane == 'Coronal':
            return self.display.scroll_max[1]

        else:
            return self.display.scroll_max[2]

    def save_deformable(self, path):
        """
        Serializes structural transform attributes and displacement tensor files to disk.

        Parameters
        ----------
        path : str
            File system directory path destination target.
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['rois',
                                                                        'display',
                                                                        'dvf',
                                                                        'rigid_rois']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'dvf.npy'), self.dvf, allow_pickle=True)

    def update_rois(self, roi_name=None, percent=100):
        """
        Applies field deformations to morph 3D region of interest surface meshes via coordinate mapping.

        Transforms vertices through continuous spatial configurations using interpolation models
        to track structural changes across target organs or annotations.

        Parameters
        ----------
        roi_name : str, optional
            Target organ model tracking key label identifier. If None, all instances are updated. Defaults to None.
        percent : float, optional
            Magnitude modifier scaling field vector step sizes. Defaults to 100.
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
                    self.rigid_rois[name] = roi.mesh.transform(np.linalg.inv(self.rigid_matrix), inplace=False)

                    points = self.rigid_rois[name].points
                    voxel_coords = (points - self.origin) / self.spacing
                    deformed_points = copy.deepcopy(points)
                    for i in range(3):
                        deformed_points[:, i] += map_coordinates(
                            (percent * self.dvf[..., i]) / 100,
                            [voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]],
                            order=1,
                            mode='nearest'
                        )

                    self.rois[name] = copy.deepcopy(self.rigid_rois[name])
                    self.rois[name].points = deformed_points