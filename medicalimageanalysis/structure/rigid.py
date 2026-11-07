"""Morfeus lab
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

Examples
--------
>>> import medicalimageanalysis as mia
>>> mia.read_mhd(file='C:/Data/scan.mhd', modality='CT')
>>> image = mia.Data.image[mia.Data.image_list()[0]]
"""

import os
import copy

import numpy as np
import pandas as pd
import SimpleITK as sitk


import vtk
from vtkmodules.util import numpy_support

from pydicom.uid import generate_uid
from scipy.spatial.transform import Rotation

from ..utils.rigid.icp import ICP
from ..utils.rigid.simpleitk import RigidITK
from ..data import Data


class Rigid(object):
    """Coordinates 3D matrix-based rigid registration operations between tracking volumes.
    
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
    
    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> rigid = mia.Rigid('CT 01', 'MR 01')
    >>> rigid.retrieve_translation()
    """
    def __init__(self, reference_name, moving_name, rigid_name=None, roi_names=None, reference_sops=None,
                 moving_sops=None, reference_matrix=None, matrix=None, combo_matrix=None, combo_name=None,
                 inverse=False):
        """Create a rigid registration object and register it globally.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.__init__(...)
        """
        self.reference_name = reference_name
        self.moving_name = moving_name
        self.combo_name = combo_name
        self.roi = dict.fromkeys(Data.roi_list())
        self.local_uid = generate_uid()

        self.reference_matrix = np.identity(4) if reference_matrix is None else np.asarray(reference_matrix, float)
        self.matrix = np.identity(4) if matrix is None else np.asarray(matrix, float)
        self.combo_matrix = np.identity(4) if combo_matrix is None else np.asarray(combo_matrix, float)

        self.inverse = False
        self.current_ref = None
        self.current_mov = None
        self.set_current_order(inverse)

        self.slices = {'reference': ['All'], 'moving': ['All'], 'reference_sops': reference_sops,
                       'moving_sops': moving_sops}
        self.visual = {'reference': None, 'moving': None, 'opacity': 0.5, 'multicolor': None}
        self.misc = {}
        self.rigid_name = self.add_rigid(rigid_name)

    @staticmethod
    def _matrix_to_sitk(mat, center):
        """4x4 reference -> moving matrix to a centered sitk Euler3DTransform with the same mapping.

        Centering the rotation on the image (instead of the world origin) gives the optimizer well-scaled parameters.
        """
        mat = np.asarray(mat, dtype=float)
        center = np.asarray(center, dtype=float)

        # Re-orthonormalize so sitk's orthogonality check never trips on accumulated float drift.
        u, _, vt = np.linalg.svd(mat[:3, :3])
        rotation = u @ vt

        transform = sitk.Euler3DTransform()
        transform.SetCenter(center.tolist())
        transform.SetMatrix(rotation.ravel().tolist())
        # sitk: T(p) = R (p - c) + c + t'. Matching R p + t gives t' = t - c + R c.
        transform.SetTranslation((mat[:3, 3] - center + rotation @ center).tolist())

        return transform

    @staticmethod
    def _rigid_inv(mat):
        """Exact inverse of a rigid 4x4 (avoids drift from repeated np.linalg.inv)."""
        inv = np.identity(4)
        inv[:3, :3] = mat[:3, :3].T
        inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
        return inv

    @staticmethod
    def _sitk_to_matrix(transform):
        """Centered sitk rigid transform (Euler / Versor) to a 4x4 reference -> moving matrix."""
        if type(transform) is sitk.Transform:
            transform = transform.Downcast()

        rotation = np.asarray(transform.GetMatrix(), dtype=float).reshape(3, 3)
        center = np.asarray(transform.GetCenter(), dtype=float)
        translation = np.asarray(transform.GetTranslation(), dtype=float)

        mat = np.identity(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = translation + center - rotation @ center

        return mat

    def _apply_display_transform(self, T):
        """
        Apply T (in displayed world) to the resliced image: A_new = T @ A, then solve back for self.matrix so
        combo_matrix stays untouched.
        """
        A_new = T @ self.get_display_matrix()
        combined_new = A_new if self.inverse else self._rigid_inv(A_new)

        self.matrix = combined_new @ np.linalg.inv(self.combo_matrix)

    def _build_itk(self, reference_mask=None, moving_mask=None, images=True):
        """Create a RigidITK helper from the reference/moving images (and optional (z, y, x) mask arrays)."""
        helper = RigidITK()
        for name, is_reference, mask in ((self.reference_name, True, reference_mask),
                                         (self.moving_name, False, moving_mask)):
            image = Data.image[name]
            # image.matrix rows are the axes; sitk wants the axes as columns.
            direction = np.asarray(image.matrix, dtype=float).T

            if images:
                helper.create_sitk_image(image.array, image.origin, image.spacing, direction,
                                         reference=is_reference)
            if mask is not None:
                helper.create_sitk_image(mask, image.origin, image.spacing, direction,
                                         reference=is_reference, mask=True)

        return helper

    def add_rigid(self, rigid_name):
        """Saves the active registration initialization instances into globally accessible data scopes.
        
        Parameters
        ----------
        rigid_name : str or None
            A targeted name for tracker referencing. If None, automatically creates an informative name.
        
        Returns
        -------
        str
            The actual uniquely verified lookup key assigned to this tracking operation.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.add_rigid(...)
        """
        if rigid_name is None:
            if np.array_equal(self.combo_matrix, np.identity(4)):
                rigid_name = self.reference_name + '_' + self.moving_name
            else:
                rigid_name = self.reference_name + '_' + self.moving_name + '_combo'

            if rigid_name in Data.rigid_list():
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(rigid_name + '_' + str(n))
                    if new_name not in Data.rigid_list():
                        rigid_name = new_name
                        n = -100

        Data.rigid[rigid_name] = self

        return rigid_name

    def compute_array(self, plane, position, use_moving_spacing=False):
        """Same call as Display.compute_array: plane + the widget's position (top-left corner of the
        reference view). Returns the moved image sampled onto the reference view, plus 'scale'.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.compute_array(...)
        """
        base = Data.image[self.current_ref].display    # image the view is built on
        mover = Data.image[self.current_mov].display   # image being moved

        ref_grid = base.get_grid(plane, position)
        grid = ref_grid
        if use_moving_spacing:
            xi, yi, _ = mover.axes[plane]
            grid = mover.grid_with_spacing(ref_grid, mover.spacing[xi], mover.spacing[yi])

        res = mover.compute_array(plane, position, rigid_matrix=self.get_display_matrix(), grid=grid)
        if res is None:
            return None
        res['scale'] = (grid['sx'] / ref_grid['sx'], grid['sy'] / ref_grid['sy'])
        res['pixel_offset'] = (0.0, 0.0)

        return res

    def compute_aspect(self, plane):
        """Aspect for the resliced image, matching the (sx, sy) Display.compute_array samples with.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.compute_aspect(...)
        """
        spacing = Data.image[self.current_mov].display.spacing

        axes = {'Axial': (0, 1, 2), 'Sagittal': (1, 2, 0), 'Coronal': (0, 2, 1)}
        xi, yi, _ = axes[plane]

        return np.round(spacing[xi] / spacing[yi], 2)

    def compute_registration(self, method='icp_point', roi_name=None, reference_mesh=None, moving_mesh=None,
                             reference_mask=None, moving_mask=None, reference_points=None, moving_points=None,
                             use_current=True, **kwargs):
        """Runs a rigid registration and stores the result in ``self.matrix``.

        Every method solves for the full reference -> moving world transform. The ``combo_matrix`` is then factored
        out, so ``self.matrix @ self.combo_matrix`` is always the answer. ``inverse`` / ``set_current_order`` only
        decide which image is displayed resliced, so registration ignores them.

        Parameters
        ----------
        method : {'itk', 'itk_mask', 'landmark', 'icp_point', 'icp_plane', 'icp_vtk'}, default 'itk'
            * 'itk'       : intensity registration, ``RigidITK.rigid`` (masks restrict the metric only).
            * 'itk_mask'  : shape registration of binary masks via signed distance maps, ``RigidITK.rigid_mask``.
            * 'landmark'  : closed-form paired-point fit, ``RigidITK.landmark``.
            * 'icp_point' : Open3D point-to-point ICP, ``ICP.compute_o3d(method='point')``.
            * 'icp_plane' : Open3D point-to-plane ICP, ``ICP.compute_o3d(method='plane')``.
            * 'icp_vtk'   : VTK ICP, ``ICP.compute_vtk``.
        roi_name : str, optional
            ICP only. Uses this ROI's mesh on both images when the meshes are not passed explicitly.
        reference_mesh, moving_mesh : pyvista.PolyData, optional
            ICP only. Meshes in their own image's world coordinates. These override ``roi_name``. They are not
            modified.
        reference_mask, moving_mask : numpy.ndarray, optional
            ITK only. Binary (z, y, x) arrays on the reference / moving image grids. Required for 'itk_mask'.
            Optional for 'itk', where they restrict where the metric is evaluated.
        reference_points, moving_points : (N, 3) array_like, optional
            'landmark' only. Paired world points in (x, y, z) order, matched row by row.
        use_current : bool, default True
            Start from the current ``self.matrix @ self.combo_matrix`` when it is not identity, so you can do a
            manual rough alignment or chain methods. Otherwise each backend uses its own initializer (COM matching
            for ICP, the Geometry/Moments initializer for ITK).
        **kwargs
            Passed straight to the backend function. Anything not given uses that function's default.
            * 'icp_point' / 'icp_plane' : distance, iterations, rmse, fitness, com_matching, src_frac, tgt_frac,
              min_pts, max_pts, seed
            * 'icp_vtk' : distance, iterations, src_frac, min_pts, max_pts, com_matching
            * 'itk' : transform, initializer, metric, bins, radius, sampling, iterations, learning_rate, min_step,
              relaxation, shrink_factors, smoothing_sigmas, cross_modality, normalize, histogram_match
            * 'itk_mask' : transform, initializer, band, sampling, iterations, learning_rate, min_step, relaxation,
              shrink_factors, smoothing_sigmas
            * 'landmark' : transform, weights

        Returns
        -------
        dict
            Diagnostics from the backend.
            * ICP (Open3D): fitness, inlier_rmse, correspondences.
            * ICP (VTK): mean_distance, iterations.
            * itk / itk_mask: metric_value, stop_condition, iterations.
            * landmark: fre_rms, fre_max, residuals.

        Raises
        ------
        ValueError
            If the method is unknown or its required inputs are missing.
        TypeError
            If ``inverse`` is passed, or a kwarg does not belong to the chosen backend.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Rigid('CT 01', 'MR 01')

        Default: intensity registration with all RigidITK defaults (multimodal, so use mutual information)

        >>> info = rigid.compute_registration(metric='MI')
        >>> info['stop_condition']

        Intensity registration focused on the liver

        >>> liver_ct = mia.Data.image['CT 01'].roi['Liver'].create_mask()    # (z, y, x) binary arrays (your ROI mask source)
        >>> rigid.compute_registration('itk', reference_mask=liver_ct, metric='MI', sampling=0.25)

        ICP on an ROI surface, overriding only a few defaults

        >>> info = rigid.compute_registration('icp_plane', roi_name='Liver', distance=5, iterations=100)
        >>> info['fitness'], info['inlier_rmse']
        >>> rigid.compute_registration('icp_vtk', roi_name='Pelvis', src_frac=0.2, iterations=200)

        Explicit meshes instead of an ROI name

        >>> ref_mesh = mia.Data.image['CT 01'].roi['Femur'].mesh
        >>> mov_mesh = mia.Data.image['MR 01'].roi['Femur'].mesh
        >>> rigid.compute_registration('icp_point', reference_mesh=ref_mesh, moving_mesh=mov_mesh, distance=10)

        Coarse-to-fine chain (each step seeds the next through use_current)

        >>> ref_pts = [[12.1, -40.3, 101.0], [55.0, -38.2, 98.4], [30.2, 10.7, 140.9]]
        >>> mov_pts = [[10.4, -35.9, 96.2], [53.1, -34.0, 94.8], [28.9, 15.2, 136.1]]
        >>> rigid.compute_registration('landmark', reference_points=ref_pts, moving_points=mov_pts)
        >>> rigid.compute_registration('itk_mask', reference_mask=liver_ct, moving_mask=liver_mr, band=30)
        >>> rigid.compute_registration('itk', reference_mask=liver_ct, metric='MI')

        Ignore the current alignment and let the backend initialize itself

        >>> rigid.compute_registration('icp_point', roi_name='Liver', use_current=False)
        """
        if 'inverse' in kwargs:
            raise TypeError("inverse is not a registration option; use set_current_order to change the display.")

        start = self.get_combined_matrix()
        seed = use_current and not np.allclose(start, np.identity(4))

        if method in ('icp_point', 'icp_plane', 'icp_vtk'):
            if roi_name is not None:
                if reference_mesh is None:
                    reference_mesh = Data.image[self.reference_name].roi[roi_name].mesh
                if moving_mesh is None:
                    moving_mesh = Data.image[self.moving_name].roi[roi_name].mesh
            if reference_mesh is None or moving_mesh is None:
                raise ValueError(f"'{method}' needs roi_name or both reference_mesh and moving_mesh.")

            # source = reference, target = moving  ->  ICP matrix is reference -> moving.
            icp = ICP(reference_mesh, moving_mesh, matrix=start if seed else None)
            if method == 'icp_vtk':
                icp.compute_vtk(**kwargs)
            else:
                icp.compute_o3d(method='point' if method == 'icp_point' else 'plane', **kwargs)

            combined = icp.get_matrix()
            info = icp.get_metrics()

        elif method in ('itk', 'itk_mask'):
            helper = self._build_itk(reference_mask, moving_mask, images=(method == 'itk'))
            initial = None
            if seed:
                initial = self._matrix_to_sitk(start, Data.image[self.reference_name].get_center())

            run = helper.rigid if method == 'itk' else helper.rigid_mask
            # sitk convention is pull-back (reference point -> moving point), the same as self.matrix.
            transform, info = run(initial_transform=initial, return_info=True, **kwargs)
            combined = self._sitk_to_matrix(transform)

        elif method == 'landmark':
            if reference_points is None or moving_points is None:
                raise ValueError("'landmark' needs reference_points and moving_points.")

            transform, info = RigidITK().landmark(reference_points, moving_points, return_info=True, **kwargs)
            combined = self._sitk_to_matrix(transform)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Factor out the earlier stage so that self.matrix @ self.combo_matrix == combined.
        self.matrix = np.asarray(combined, dtype=float) @ self._rigid_inv(self.combo_matrix)

        return info

    def compute_mesh_slice(self, roi_name, plane, position, offset=0, return_pixel=True, use_cache=True):
        """Slice one of current_mov's ROIs in the same pixel space as compute_array. Cached per
        ROI (see Roi.compute_mesh_slice); pass use_cache=False while the user is actively dragging
        rotate/translate, since everyframe has a new matrix and would only push useful scroll
        entries out of the LRU.
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.compute_mesh_slice(...)
        """

        roi = Data.image[self.current_mov].roi[roi_name]
        grid = Data.image[self.current_ref].display.get_grid(plane, position)

        return roi.compute_mesh_slice(origin=position, plane=plane, offset=offset,
                                      rigid_matrix=self.get_display_matrix(),
                                      return_pixel=return_pixel, grid=grid, use_cache=use_cache)

    def copy_roi(self, roi_name=None):
        """Clones and projects an ROI structural mesh model alignment into different volume spaces.
        
        Parameters
        ----------
        roi_name : str, optional
            The targeted structure lookup key label mapping to the active tracking item.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.copy_roi(...)
        """
        if roi_name in list(self.roi.keys()):
            reference_roi = Data.image[self.reference_name].roi[roi_name]
            moving_roi = Data.image[self.moving_name].roi[roi_name]
            if self.inverse and self.roi[roi_name] is not None:
                reference_roi.mesh = self.roi[roi_name].transform(np.linalg.inv(self.matrix @ self.combo_matrix),
                                                                   inplace=False)
            elif reference_roi.mesh is not None:
                moving_roi.mesh = reference_roi.mesh.transform(self.matrix @ self.combo_matrix, inplace=False)

    def create_image(self):
        """Resample current_mov onto current_ref's world (same convention as the display).
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.create_image(...)
        """
        mov = Data.image[self.current_mov]
        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(mov.spacing)
        vtk_image.SetDirectionMatrix(mov.matrix.T.ravel())   # rows = axes in your convention; VTK wants columns
        vtk_image.SetDimensions(np.flip(mov.array.shape))
        vtk_image.SetOrigin(mov.origin)
        vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(mov.array.ravel(order="C"), deep=False))

        # ResliceTransform maps OUTPUT (displayed) coords -> INPUT (native) coords = inverse of display matrix
        out_to_in = self._rigid_inv(self.get_display_matrix())
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, out_to_in[i, j])
        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_matrix)

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(vtk_image)
        reslice.SetResliceTransform(transform)
        reslice.SetInterpolationModeToLinear()
        reslice.SetOutputSpacing(Data.image[self.current_ref].spacing)
        reslice.SetOutputDirection(1, 0, 0, 0, 1, 0, 0, 0, 1)
        reslice.AutoCropOutputOn()
        reslice.SetBackgroundLevel(-3001)
        reslice.Update()

        return reslice.GetOutput()

    def export_image(self, path=None):
        """Writes the current aligned 3D dataset out to disk as an mhd/raw MetaImage format pair.
        
        Parameters
        ----------
        path : str, optional
            The targeted filename destination write path string.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.export_image(...)
        """
        if self.moving_name is not None and path is not None:
            image = self.create_image()

            writer = vtk.vtkMetaImageWriter()
            writer.SetInputData(image)
            writer.SetFileName(path)
            writer.Write()

    def get_center(self, plane=None, position=None):
        """Pivot: current_mov's center as displayed, projected onto the active slice plane (when given).
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.get_center(...)
        """
        A = self.get_display_matrix()
        native_center = np.asarray(Data.image[self.current_mov].get_center(), dtype=float)
        center = A[:3, :3] @ native_center + A[:3, 3]

        if plane is None or position is None:
            return center

        normal = Data.image[self.current_ref].display.compute_plane_geometry(plane, position)['normal']
        n_hat = normal / np.linalg.norm(normal)
        return center - np.dot(center - np.asarray(position, dtype=float), n_hat) * n_hat

    def get_combined_matrix(self):
        """reference world -> moving world.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.get_combined_matrix(...)
        """

        return self.matrix @ self.combo_matrix

    def get_display_matrix(self):
        """current_mov native world -> current_ref (displayed) world. Pass this as rigid_matrix.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.get_display_matrix(...)
        """
        combined = self.get_combined_matrix()

        return combined if self.inverse else self._rigid_inv(combined)

    def get_view_center(self, plane, position):
        """Pivot in the widget's view coords (pixel centers at integers, matching the -0.5 placement).
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.get_view_center(...)
        """
        grid = Data.image[self.current_ref].display.get_grid(plane, position)
        d = self.get_center(plane, position) - grid['origin']

        return float(d @ grid['x_axis']) / grid['sx'], float(d @ grid['y_axis']) / grid['sy']

    def get_vtk_matrix(self):
        """For actor.SetUserMatrix on current_mov's ROI actors (or pyvista: actor.user_matrix = ...).
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.get_vtk_matrix(...)
        """
        A = self.get_display_matrix()
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, A[i, j])

        return vtk_matrix

    def pre_alignment(self, superior=False, center=False, origin=False):
        """Applies rapid programmatic initializations to roughly align spatial orientation metrics.
        
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
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.pre_alignment(...)
        """
        if superior:
            pass
        elif center:
            pass
        elif origin:
            self.matrix[:3, 3] = Data.image[self.moving_name].origin - Data.image[self.reference_name].origin

    def retrieve_angles(self, order='ZXY'):
        """Extracts Euler orientation values from the active registration transformation components.
        
        Parameters
        ----------
        order : str, default 'ZXY'
            The specific axes rotation order tracking matrix factorization.
        
        Returns
        -------
        numpy.ndarray
            A 3D spatial rotation angle vector formatted in degrees.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.retrieve_angles(...)
        """
        rotation = Rotation.from_matrix(self.matrix[:3, :3])
        return rotation.as_euler(order, degrees=True)

    def retrieve_translation(self):
        """Exposes current alignment spatial position offsets.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        numpy.ndarray
            A 3D vector coordinates array tracking translation distance variables.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.retrieve_translation(...)
        """
        return self.matrix[:3, 3]

    def save_rigid(self, path):
        """Serializes active internal class settings directly out into picked DataFrame file formats.
        
        Parameters
        ----------
        path : str
            The system directory output target location to export towards.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.save_rigid(...)
        """
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['roi', 'poi', 'display']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))

    def set_current_order(self, inverse):
        """Flips which image gets resliced. Does NOT change the registration itself, so the user can
        toggle at any time. Callers must re-request arrays/slices and move the actor matrix to the
        other image's ROI actors afterwards.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.set_current_order(...)
        """
        self.inverse = bool(inverse)
        if self.inverse:
            self.current_ref, self.current_mov = self.moving_name, self.reference_name
        else:
            self.current_ref, self.current_mov = self.reference_name, self.moving_name

    def update_rois(self):
        """Placeholder for future synchronized ROI updates under rigid transforms.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.update_rois(...)
        """
        pass

    def update_rotation(self, center=None, r_x=0, r_y=0, r_z=0, plane=None, position=None,
                        axis=None, angle=None):
        """Rotate about center (default: pivot on the current slice). Either Euler r_x/r_y/r_z (world axes, degrees),
        or axis + angle (degrees) to rotate about an arbitrary world axis, e.g. an oblique view's normal.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.update_rotation(...)
        """
        if center is None:
            center = self.get_center(plane=plane, position=position)
        center = np.asarray(center, dtype=float)

        R = np.identity(4)
        if axis is not None:
            axis = np.asarray(axis, dtype=float)
            R[:3, :3] = Rotation.from_rotvec(np.radians(angle) * axis / np.linalg.norm(axis)).as_matrix()
        else:
            R[:3, :3] = Rotation.from_euler('xyz', [r_x, r_y, r_z], degrees=True).as_matrix()

        T_neg = np.identity(4)
        T_neg[:3, 3] = -center
        T_pos = np.identity(4)
        T_pos[:3, 3] = center

        self._apply_display_transform(T_pos @ R @ T_neg)

    def update_translation(self, t_x=0, t_y=0, t_z=0):
        """Apply a pure translation in displayed world coordinates.
        
        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid[mia.Data.rigid_list()[0]]
        >>> rigid.update_translation(...)
        """
        T = np.identity(4)
        T[:3, 3] = [t_x, t_y, t_z]

        self._apply_display_transform(T)
