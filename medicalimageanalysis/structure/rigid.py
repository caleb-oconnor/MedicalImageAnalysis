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
import pyvista as pv


import vtk
from vtkmodules.util import numpy_support

from pydicom.uid import generate_uid
from scipy.spatial.transform import Rotation

from ..utils.rigid.icp import ICP
from ..utils.convert.contour import MeshToContour
from ..data import Data


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
                 moving_sops=None, reference_matrix=None, matrix=None, combo_matrix=None, combo_name=None,
                 inverse=False):
        self.reference_name = reference_name
        self.moving_name = moving_name
        self.combo_name = combo_name
        self.rois = dict.fromkeys(Data.roi_list)
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

    def _apply_display_transform(self, T):
        """
        Apply T (in displayed world) to the resliced image: A_new = T @ A, then solve back for self.matrix so
        combo_matrix stays untouched.
        """
        A_new = T @ self.get_display_matrix()
        combined_new = A_new if self.inverse else self._rigid_inv(A_new)

        self.matrix = combined_new @ np.linalg.inv(self.combo_matrix)

    @staticmethod
    def _rigid_inv(mat):
        """Exact inverse of a rigid 4x4 (avoids drift from repeated np.linalg.inv)."""
        inv = np.identity(4)
        inv[:3, :3] = mat[:3, :3].T
        inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
        return inv

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

    def compute_array(self, plane, position, use_moving_spacing=False):
        """
        Same call as Display.compute_array: plane + the widget's position (top-left corner of the
        reference view). Returns the moved image sampled onto the reference view, plus 'scale'.
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
        """
        Aspect for the resliced image, matching the (sx, sy) Display.compute_array samples with.
        """
        spacing = Data.image[self.current_mov].display.spacing

        axes = {'Axial': (0, 1, 2), 'Sagittal': (1, 2, 0), 'Coronal': (0, 2, 1)}
        xi, yi, _ = axes[plane]

        return np.round(spacing[xi] / spacing[yi], 2)

    def compute_icp_vtk(self, source_mesh, target_mesh, distance=1e-3, iterations=100, src_frac=0.1, min_pts=1000,
                        max_pts=20000, com_matching=True, inverse=False, center=None):
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
        icp.compute_vtk(distance=distance, iterations=iterations, src_frac=src_frac, min_pts=min_pts,
                        max_pts=max_pts, com_matching=com_matching, inverse=inverse)

        if center == 'image':
            R_icp = np.asarray(icp.get_matrix(), dtype=float)
            old_center = np.array([0, 0, 0], dtype=float)
            new_center = np.array(Data.image[self.moving_name].get_center(), dtype=float)

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

    def compute_o3d(self, source_mesh, target_mesh, distance=10, iterations=50, rmse=1e-6, fitness=1e-6, method='point',
                    com_matching=True, inverse=False, src_frac=0.1, tgt_frac=1.0, min_pts=1000, max_pts=20000, seed=0,
                    center=None):
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
                        com_matching=com_matching, inverse=inverse, src_frac=src_frac, tgt_frac=tgt_frac,
                        min_pts=min_pts, max_pts=max_pts, seed=seed)

        if center == 'image':
            R_icp = np.asarray(icp.get_matrix(), dtype=float)
            old_center = np.array([0, 0, 0], dtype=float)
            new_center = np.array(Data.image[self.moving_name].get_center(), dtype=float)

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

    def compute_mesh_slice(self, roi_name, plane, position, offset=0, return_pixel=True):
        """
        Slice one of current_mov's ROIs in the same pixel space as compute_array.
        """
        roi = Data.image[self.current_mov].rois[roi_name]

        return roi.compute_mesh_slice(origin=position, slice_plane=plane, offset=offset,
                                      rigid_matrix=self.get_display_matrix(), return_pixel=return_pixel)

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
            moving_roi = Data.image[self.moving_name].rois[roi_name]
            if self.inverse and self.rois[roi_name] is not None:
                reference_roi.mesh = self.rois[roi_name].transform(np.linalg.inv(self.matrix @ self.combo_matrix),
                                                                   inplace=False)
            elif reference_roi.mesh is not None:
                moving_roi.mesh = reference_roi.mesh.transform(self.matrix @ self.combo_matrix, inplace=False)

    def create_image(self):
        """
        Resample current_mov onto current_ref's world (same convention as the display).
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

    def get_center(self, plane=None, position=None):
        """
        Pivot: current_mov's center as displayed, projected onto the active slice plane (when given).
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
        """
        reference world -> moving world.
        """

        return self.matrix @ self.combo_matrix

    def get_display_matrix(self):
        """
        current_mov native world -> current_ref (displayed) world. Pass this as rigid_matrix.
        """
        combined = self.get_combined_matrix()

        return combined if self.inverse else self._rigid_inv(combined)

    def get_view_center(self, plane, position):
        """Pivot in the widget's view coords (pixel centers at integers, matching the -0.5 placement)."""
        grid = Data.image[self.current_ref].display.get_grid(plane, position)
        d = self.get_center(plane, position) - grid['origin']

        return float(d @ grid['x_axis']) / grid['sx'], float(d @ grid['y_axis']) / grid['sy']

    def get_vtk_matrix(self):
        """
        For actor.SetUserMatrix on current_mov's ROI actors (or pyvista: actor.user_matrix = ...).
        """
        A = self.get_display_matrix()
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, A[i, j])

        return vtk_matrix

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

    def set_current_order(self, inverse):
        """
        Flips which image gets resliced. Does NOT change the registration itself, so the user can
        toggle at any time. Callers must re-request arrays/slices and move the actor matrix to the
        other image's ROI actors afterwards.
        """
        self.inverse = bool(inverse)
        if self.inverse:
            self.current_ref, self.current_mov = self.moving_name, self.reference_name
        else:
            self.current_ref, self.current_mov = self.reference_name, self.moving_name

    def update_rotation(self, center=None, r_x=0, r_y=0, r_z=0, plane=None, position=None,
                        axis=None, angle=None):
        """
        Rotate about center (default: pivot on the current slice). Either Euler r_x/r_y/r_z (world axes, degrees),
        or axis + angle (degrees) to rotate about an arbitrary world axis, e.g. an oblique view's normal.
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
        T = np.identity(4)
        T[:3, 3] = [t_x, t_y, t_z]

        self._apply_display_transform(T)
