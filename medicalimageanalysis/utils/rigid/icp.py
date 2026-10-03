"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""
import copy

import vtk
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from scipy.spatial.transform import Rotation

from open3d.geometry import PointCloud, KDTreeSearchParamHybrid
from open3d.utility import Vector3dVector
from open3d.pipelines.registration import (registration_icp, ICPConvergenceCriteria,
                                           TransformationEstimationPointToPoint,
                                           TransformationEstimationPointToPlane)


class ICP(object):
    """
    Performs Iterative Closest Point (ICP) registration between a source and a target mesh or point cloud. Supports
    both VTK-based and Open3D-based ICP implementations.
    """
    def __init__(self, source: pv.PolyData, target: pv.PolyData, matrix: np.ndarray | None = None):
        """
        Initializes the ICP object.

        Parameters
        ----------
        source : pyvista.PolyData
            The moving/source mesh to align.
        target : pyvista.PolyData
            The reference/target mesh.
        matrix : (4, 4) ndarray, optional
            Initial transformation matrix. Defaults to identity.

        Raises
        ------
        TypeError
            If source or target is not a pyvista.PolyData.
        ValueError
            If matrix is not 4x4.
        """

        for name, mesh in (('source', source), ('target', target)):
            if not isinstance(mesh, pv.PolyData):
                raise TypeError(f"{name} must be a pyvista.PolyData, got {type(mesh).__name__}")

        self.source = source
        self.target = target

        self.matrix = matrix

        self.icp = None

    @staticmethod
    def _sample_count(n, frac, min_pts=1000, max_pts=None):
        """
        Computes how many points to keep when subsampling by fraction.

        Parameters
        ----------
        n : int
            Total number of available points.
        frac : float
            Fraction of points to keep (e.g. 0.1 for 10%).
        min_pts : int
            Lower bound on the returned count. Ignored if n is smaller.
        max_pts : int or None
            Upper bound on the returned count. None means no cap.

        Returns
        -------
        k : int
            Number of points to keep, in the range [min(min_pts, n), min(max_pts, n)].
        """

        k = int(round(n * frac))
        k = max(k, min(min_pts, n))
        if max_pts is not None:
            k = min(k, max_pts)

        return min(k, n)

    def _subsample(self, pts, frac, rng, min_pts=1000, max_pts=None, normals=None):
        """
        Randomly subsamples a point array (and optional normals) by fraction.

        Parameters
        ----------
        pts : (N, 3) ndarray
            Point coordinates.
        frac : float
            Fraction of points to keep, clamped by min_pts and max_pts.
        rng : numpy.random.Generator
            Random generator used to pick indices (for reproducibility).
        min_pts : int
            Minimum number of points to keep.
        max_pts : int or None
            Maximum number of points to keep. None means no cap.
        normals : (N, 3) ndarray or None
            Per-point normals subsampled with the same indices as pts.

        Returns
        -------
        pts_sub : (K, 3) ndarray
            Subsampled points (the original array if no subsampling was needed).
        normals_sub : (K, 3) ndarray or None
            Matching subsampled normals, or None if normals was None.
        """

        n = len(pts)
        k = self._sample_count(n, frac, min_pts, max_pts)
        if k >= n:
            return pts, normals

        idx = rng.choice(n, k, replace=False)

        return pts[idx], (normals[idx] if normals is not None else None)

    def compute_com(self):
        """
        Computes an initial translation by matching the center of mass (COM) of the source and target meshes.
        """
        translation = np.asarray(self.target.center) - np.asarray(self.source.center)

        self.matrix = np.identity(4)
        self.matrix[:3, 3] = translation

    def compute_o3d(self, distance=10, iterations=50, rmse=1e-6, fitness=1e-6, method='point', com_matching=True,
                    inverse=False, src_frac=0.1, tgt_frac=1.0, min_pts=1000, max_pts=20000, seed=0):
        """
        Performs rigid ICP using Open3D on subsampled source/target points.

        Parameters
        ----------
        distance : float
            Maximum correspondence distance (mesh units).
        iterations : int
            Maximum number of ICP iterations.
        rmse : float
            Convergence threshold for relative change in RMSE.
        fitness : float
            Convergence threshold for relative change in fitness.
        method : str
            'point' for point-to-point ICP, 'plane' for point-to-plane ICP.
        com_matching : bool
            If True, initializes with the translation between centers of mass.
        inverse : bool
            If True, stores the inverse of the resulting transformation matrix.
        src_frac : float
            Fraction of source points used, clamped to [min_pts, max_pts].
        tgt_frac : float
            Fraction of target points used, clamped below by min_pts (no upper cap).
        min_pts : int
            Minimum number of points kept per cloud.
        max_pts : int or None
            Maximum number of source points kept. None means no cap.
        seed : int
            Random seed for subsampling.

        Uses
        ----
        self.source : pyvista.PolyData
            Mesh that is moved onto the target.
        self.target : pyvista.PolyData
            Fixed mesh; must have point_normals for method='plane'.

        Returns
        -------
        None
            Sets self.icp (open3d RegistrationResult) and
            self.matrix ((4, 4) ndarray, source -> target, or its inverse if inverse=True).
        """

        src_pts = np.asarray(self.source.points)
        tgt_pts = np.asarray(self.target.points)
        tgt_nrm = np.asarray(self.target.point_normals)
        rng = np.random.default_rng(seed)

        src_sub, _ = self._subsample(src_pts, src_frac, rng, min_pts, max_pts)
        tgt_sub, tgt_nrm = self._subsample(tgt_pts, tgt_frac, rng, min_pts, None, tgt_nrm)

        ref_pcd = PointCloud(Vector3dVector(src_sub))
        mov_pcd = PointCloud(Vector3dVector(tgt_sub))
        mov_pcd.normals = Vector3dVector(tgt_nrm)

        initial_transform = np.identity(4)
        if com_matching:
            # use the full clouds for the centroid; it's cheap and more stable
            initial_transform[:3, 3] = tgt_pts.mean(0) - src_pts.mean(0)

        criteria = ICPConvergenceCriteria(max_iteration=iterations,
                                          relative_rmse=rmse,
                                          relative_fitness=fitness)

        estimator = (TransformationEstimationPointToPoint() if method == 'point'
                     else TransformationEstimationPointToPlane())

        self.icp = registration_icp(ref_pcd, mov_pcd, distance, initial_transform, estimator, criteria)

        self.matrix = np.linalg.inv(self.icp.transformation) if inverse else self.icp.transformation


    def compute_vtk(self, distance=1e-3, iterations=100, src_frac=0.1, min_pts=1000, max_pts=20000, com_matching=True,
                    inverse=False):
        """
        Performs rigid ICP using VTK, with landmark count set as a fraction of source points.

        Parameters
        ----------
        distance : float
            Absolute RMS distance threshold for convergence (mesh units).
        iterations : int
            Maximum number of ICP iterations.
        src_frac : float
            Fraction of source points used as landmarks, clamped to [min_pts, max_pts].
        min_pts : int
            Minimum number of landmarks.
        max_pts : int or None
            Maximum number of landmarks. None means no cap.
        com_matching : bool
            If True, starts by matching the source and target centroids.
        inverse : bool
            If True, stores the inverse of the resulting transformation matrix.

        Uses
        ----
        self.source : pyvista.PolyData
            Mesh that is moved onto the target (landmarks are taken from here).
        self.target : pyvista.PolyData
            Fixed mesh; full resolution is used for the closest-point search.

        Returns
        -------
        None
            Sets self.icp (vtkIterativeClosestPointTransform) and
            self.matrix ((4, 4) ndarray, source -> target, or its inverse if inverse=True).
        """

        landmarks = self._sample_count(self.source.n_points, src_frac, min_pts, max_pts)

        self.icp = vtk.vtkIterativeClosestPointTransform()
        self.icp.SetSource(self.source)
        self.icp.SetTarget(self.target)
        self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.SetMaximumNumberOfLandmarks(landmarks)
        self.icp.SetCheckMeanDistance(1)
        self.icp.SetMeanDistanceModeToRMS()
        self.icp.SetMaximumMeanDistance(distance)
        self.icp.SetMaximumNumberOfIterations(iterations)
        self.icp.SetStartByMatchingCentroids(com_matching)
        self.icp.Update()

        m = pv.array_from_vtkmatrix(self.icp.GetMatrix())
        self.matrix = np.linalg.inv(m) if inverse else m

    def get_matrix(self):
        """
        Returns the resulting transformation matrix.

        Returns
        -------
        np.ndarray
            4x4 transformation matrix.
        """

        return self.matrix

    def get_correspondence_set(self):
        """
        Returns the set of corresponding point indices (if available).

        Returns
        -------
        np.ndarray or None
            Correspondence set from ICP.
        """
        if hasattr(self.icp, 'correspondence_set'):
            return np.asarray(self.icp.correspondence_set)

        else:
            return None
