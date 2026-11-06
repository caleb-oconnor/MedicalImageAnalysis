"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Rigid surface registration with Iterative Closest Point (ICP), using either Open3D or VTK.

    Methods
    -------
    * Open3D: :meth:`ICP.compute_o3d` (point-to-point or point-to-plane, random subsampling, outlier rejection
      through a maximum correspondence distance)
    * VTK: :meth:`ICP.compute_vtk` (``vtkIterativeClosestPointTransform``, closest point on the target surface)
    * Utilities: :meth:`ICP.compute_com`, :meth:`ICP.get_matrix`, :meth:`ICP.get_correspondence_set`,
      :meth:`ICP.get_metrics`

    Conventions
    -----------
    * All matrices are (4, 4) homogeneous transforms that map source points onto the target
      (``target ~ matrix @ source``), unless ``inverse=True`` was used.
    * ``self.matrix`` holds the initial transform before a compute call and the result after it. If it is set when a
      compute method runs, it is used as the starting transform, so :meth:`compute_com` (or a ``matrix`` passed to
      the constructor) feeds directly into the ICP methods.
    * The "center" of a mesh is always its vertex mean, for both :meth:`compute_com` and the ``com_matching`` options.
    * The stored meshes are never modified.

Structure:
    ICP
    ├── __init__                   store meshes and optional initial matrix
    ├── _sample_count / _subsample random subsampling helpers
    ├── _initial_matrix            starting transform shared by both backends
    ├── compute_com                vertex-mean translation
    ├── compute_o3d                Open3D ICP
    ├── compute_vtk                VTK ICP
    └── get_matrix / get_correspondence_set / get_metrics

Examples
--------
>>> import numpy as np
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.rigid.icp import ICP
>>> source = pv.ParametricEllipsoid(30, 20, 10)
>>> target = source.rotate_z(5).translate((2, 1, 0))
>>> icp = ICP(source, target)
>>> icp.compute_o3d(method='plane')
>>> np.round(icp.get_matrix()[:3, 3], 1).tolist()
[2.0, 1.0, 0.0]
"""

import vtk
import numpy as np
import pyvista as pv

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.pipelines.registration import (registration_icp, ICPConvergenceCriteria,
                                           TransformationEstimationPointToPoint,
                                           TransformationEstimationPointToPlane)


class ICP:
    """Performs Iterative Closest Point (ICP) registration between a source and a target mesh or point cloud. Supports
    both VTK-based and Open3D-based ICP implementations.

    Parameters
    ----------
    source : pyvista.PolyData
        The moving/source mesh to align.
    target : pyvista.PolyData
        The reference/target mesh.
    matrix : (4, 4) ndarray, optional
        Initial source -> target transform. None means it is estimated in the compute call.

    Examples
    --------
    >>> import pyvista as pv
    >>> source = pv.ParametricEllipsoid(30, 20, 10)
    >>> icp = ICP(source, source.translate((2, 0, 0)))
    >>> icp.compute_vtk()
    >>> round(float(icp.get_matrix()[0, 3]), 1)
    2.0
    """

    def __init__(self, source: pv.PolyData, target: pv.PolyData, matrix: np.ndarray | None = None) -> None:
        """Stores the meshes and an optional initial transformation matrix.

        Parameters
        ----------
        source : pyvista.PolyData
            The moving/source mesh to align.
        target : pyvista.PolyData
            The reference/target mesh.
        matrix : (4, 4) ndarray, optional
            Initial source -> target transform. None (default) means no initial transform is stored; the compute
            methods then start from identity, or from the centroid offset if ``com_matching=True``.

        Raises
        ------
        TypeError
            If source or target is not a pyvista.PolyData.
        ValueError
            If matrix is given and is not 4x4.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> icp = ICP(pv.Sphere(), pv.Sphere(center=(2, 0, 0)), matrix=np.identity(4))
        >>> icp.get_matrix().shape
        (4, 4)
        >>> ICP(pv.Sphere(), pv.Sphere(), matrix=np.identity(3))
        Traceback (most recent call last):
        ...
        ValueError: matrix must be 4x4, got shape (3, 3)
        """

        for name, mesh in (('source', source), ('target', target)):
            if not isinstance(mesh, pv.PolyData):
                raise TypeError(f"{name} must be a pyvista.PolyData, got {type(mesh).__name__}")

        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"matrix must be 4x4, got shape {matrix.shape}")

        self.source = source
        self.target = target

        self.matrix = matrix

        # Backend result (open3d RegistrationResult or vtkIterativeClosestPointTransform).
        self.icp = None

        # Indices into the original source/target points that were used by the last compute_o3d call, so the
        # correspondence set can be mapped back onto the meshes.
        self.source_idx = None
        self.target_idx = None

    @staticmethod
    def _sample_count(n, frac, min_pts=1000, max_pts=None):
        """Computes how many points to keep when subsampling by fraction.

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

        Examples
        --------
        >>> ICP._sample_count(50000, 0.1)
        5000
        >>> ICP._sample_count(5000, 0.1)
        1000
        >>> ICP._sample_count(500, 0.1)
        500
        >>> ICP._sample_count(500000, 0.1, max_pts=20000)
        20000
        """

        k = int(round(n * frac))
        k = max(k, min(min_pts, n))
        if max_pts is not None:
            k = min(k, max_pts)

        return min(k, n)

    def _com_translation(self):
        """Returns the translation between the source and target vertex means (target - source)."""
        return np.asarray(self.target.points).mean(0) - np.asarray(self.source.points).mean(0)

    def _initial_matrix(self, com_matching):
        """Returns the starting transform used by both compute methods.

        ``self.matrix`` takes precedence when it is set. Otherwise the vertex-mean offset is used if ``com_matching``
        is True, and identity if not.

        Parameters
        ----------
        com_matching : bool
            Use the vertex-mean translation when no matrix is stored.

        Returns
        -------
        (4, 4) ndarray
            Initial source -> target transform.

        Examples
        --------
        >>> import pyvista as pv
        >>> icp = ICP(pv.Sphere(), pv.Sphere(center=(2, 0, 0)))
        >>> round(float(icp._initial_matrix(True)[0, 3]), 3), float(icp._initial_matrix(False)[0, 3])
        (2.0, 0.0)
        """

        if self.matrix is not None:
            return np.array(self.matrix, dtype=np.float64)

        initial = np.identity(4)
        if com_matching:
            initial[:3, 3] = self._com_translation()

        return initial

    def _subsample(self, pts, frac, rng, min_pts=1000, max_pts=None, normals=None):
        """Randomly subsamples a point array (and optional normals) by fraction.

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
        idx : (K,) ndarray
            Indices of the kept points in the original array.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> icp = ICP(pv.Sphere(), pv.Sphere())
        >>> pts = np.random.rand(5000, 3)
        >>> sub, _, idx = icp._subsample(pts, 0.5, np.random.default_rng(0))
        >>> sub.shape, bool(np.array_equal(sub, pts[idx]))
        ((2500, 3), True)
        """

        n = len(pts)
        k = self._sample_count(n, frac, min_pts, max_pts)
        if k >= n:
            return pts, normals, np.arange(n)

        idx = np.sort(rng.choice(n, k, replace=False))

        return pts[idx], (normals[idx] if normals is not None else None), idx

    def compute_com(self):
        """Computes an initial translation by matching the vertex means of the source and target meshes.

        The result is stored in ``self.matrix`` and is used as the starting transform by :meth:`compute_o3d` and
        :meth:`compute_vtk`.

        Returns
        -------
        None
            Sets self.matrix ((4, 4) ndarray, translation only).

        Examples
        --------
        >>> import pyvista as pv
        >>> icp = ICP(pv.Sphere(), pv.Sphere(center=(2, 0, 0)))
        >>> icp.compute_com()
        >>> round(float(icp.get_matrix()[0, 3]), 3)
        2.0
        """

        self.matrix = np.identity(4)
        self.matrix[:3, 3] = self._com_translation()

    def compute_o3d(self, distance=10, iterations=50, rmse=1e-6, fitness=1e-6, method='point', com_matching=True,
                    inverse=False, src_frac=0.1, tgt_frac=1.0, min_pts=1000, max_pts=20000, seed=0):
        """Performs rigid ICP using Open3D on subsampled source/target points.

        Parameters
        ----------
        distance : float
            Maximum correspondence distance (mesh units). Pairs further apart are rejected as outliers, so the
            initial offset must be within this distance.
        iterations : int
            Maximum number of ICP iterations.
        rmse : float
            Convergence threshold for relative change in RMSE.
        fitness : float
            Convergence threshold for relative change in fitness.
        method : {'point', 'plane'}
            'point' for point-to-point ICP, 'plane' for point-to-plane ICP (uses the target point normals).
        com_matching : bool
            If True and no matrix is stored, initializes with the translation between vertex means.
        inverse : bool
            If True, stores the inverse (target -> source) of the resulting transformation matrix.
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

        Returns
        -------
        None
            Sets self.icp (open3d RegistrationResult), self.source_idx / self.target_idx (subsample indices) and
            self.matrix ((4, 4) ndarray, source -> target, or its inverse if inverse=True).

        Raises
        ------
        ValueError
            If method is not 'point' or 'plane'.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> source = pv.ParametricEllipsoid(30, 20, 10)
        >>> icp = ICP(source, source.rotate_z(5).translate((2, 1, 0)))
        >>> icp.compute_o3d(method='plane')
        >>> np.round(icp.get_matrix()[:3, 3], 1).tolist()
        [2.0, 1.0, 0.0]
        >>> icp.get_metrics()['fitness']
        1.0
        """

        if method not in ('point', 'plane'):
            raise ValueError(f"Unknown method: {method}")

        src_pts = np.asarray(self.source.points)
        tgt_pts = np.asarray(self.target.points)
        # Normals are only needed (and only computed) for point-to-plane.
        tgt_nrm = np.asarray(self.target.point_normals) if method == 'plane' else None
        rng = np.random.default_rng(seed)

        src_sub, _, self.source_idx = self._subsample(src_pts, src_frac, rng, min_pts, max_pts)
        tgt_sub, tgt_nrm, self.target_idx = self._subsample(tgt_pts, tgt_frac, rng, min_pts, None, tgt_nrm)

        src_pcd = PointCloud(Vector3dVector(src_sub))
        tgt_pcd = PointCloud(Vector3dVector(tgt_sub))
        if tgt_nrm is not None:
            tgt_pcd.normals = Vector3dVector(tgt_nrm)

        initial_transform = self._initial_matrix(com_matching)

        criteria = ICPConvergenceCriteria(max_iteration=iterations,
                                          relative_rmse=rmse,
                                          relative_fitness=fitness)

        estimator = (TransformationEstimationPointToPoint() if method == 'point'
                     else TransformationEstimationPointToPlane())

        self.icp = registration_icp(src_pcd, tgt_pcd, distance, initial_transform, estimator, criteria)

        m = np.array(self.icp.transformation)
        self.matrix = np.linalg.inv(m) if inverse else m

    def compute_vtk(self, distance=1e-3, iterations=100, src_frac=0.1, min_pts=1000, max_pts=20000, com_matching=True,
                    inverse=False):
        """Performs rigid ICP using VTK, with landmark count set as a fraction of source points.

        VTK matches each landmark to the closest point on the target surface and has no outlier rejection, so it is
        best suited to meshes that overlap fully. ``vtkIterativeClosestPointTransform`` has no initial-transform
        input, so a stored matrix is applied to a copy of the source first and composed with the ICP result.

        Parameters
        ----------
        distance : float
            Absolute RMS distance threshold for convergence (mesh units). Note this is a convergence tolerance,
            not the correspondence radius used by :meth:`compute_o3d`.
        iterations : int
            Maximum number of ICP iterations.
        src_frac : float
            Fraction of source points used as landmarks, clamped to [min_pts, max_pts]. VTK picks landmarks with a
            regular stride through the point list, not randomly.
        min_pts : int
            Minimum number of landmarks.
        max_pts : int or None
            Maximum number of landmarks. None means no cap.
        com_matching : bool
            If True and no matrix is stored, starts by matching the source and target vertex means.
        inverse : bool
            If True, stores the inverse (target -> source) of the resulting transformation matrix.

        Returns
        -------
        None
            Sets self.icp (vtkIterativeClosestPointTransform) and
            self.matrix ((4, 4) ndarray, source -> target, or its inverse if inverse=True).

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> source = pv.ParametricEllipsoid(30, 20, 10)
        >>> icp = ICP(source, source.rotate_z(5).translate((2, 1, 0)))
        >>> icp.compute_vtk()
        >>> np.round(icp.get_matrix()[:3, 3], 1).tolist()
        [2.0, 1.0, 0.0]
        """

        landmarks = self._sample_count(self.source.n_points, src_frac, min_pts, max_pts)

        # Apply the starting transform to a copy so the same initialization is used as in compute_o3d.
        initial_transform = self._initial_matrix(com_matching)
        source = self.source.transform(initial_transform, inplace=False)

        self.icp = vtk.vtkIterativeClosestPointTransform()
        self.icp.SetSource(source)
        self.icp.SetTarget(self.target)
        self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.SetMaximumNumberOfLandmarks(landmarks)
        self.icp.SetCheckMeanDistance(1)
        self.icp.SetMeanDistanceModeToRMS()
        self.icp.SetMaximumMeanDistance(distance)
        self.icp.SetMaximumNumberOfIterations(iterations)
        # Centroids are already matched by the initial transform when requested.
        self.icp.SetStartByMatchingCentroids(False)
        self.icp.Update()

        m = pv.array_from_vtkmatrix(self.icp.GetMatrix()) @ initial_transform
        self.matrix = np.linalg.inv(m) if inverse else m

    def get_correspondence_set(self, original=True):
        """Returns the set of corresponding point indices from the last :meth:`compute_o3d` call.

        Parameters
        ----------
        original : bool, default True
            If True, indices refer to the original source/target mesh points. If False, they refer to the
            subsampled point clouds used inside the ICP.

        Returns
        -------
        (M, 2) ndarray or None
            Columns are (source index, target index). None if the last run was not Open3D.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.ParametricEllipsoid(30, 20, 10)
        >>> icp = ICP(source, source.translate((2, 0, 0)))
        >>> icp.compute_o3d()
        >>> pairs = icp.get_correspondence_set()
        >>> pairs.shape[1], bool(pairs[:, 0].max() < source.n_points)
        (2, True)
        """

        if not hasattr(self.icp, 'correspondence_set'):
            return None

        pairs = np.asarray(self.icp.correspondence_set)
        if original and len(pairs):
            pairs = np.column_stack((self.source_idx[pairs[:, 0]], self.target_idx[pairs[:, 1]]))

        return pairs

    def get_matrix(self):
        """Returns the resulting transformation matrix.

        Returns
        -------
        np.ndarray or None
            4x4 transformation matrix, or None if nothing has been set or computed.

        Examples
        --------
        >>> import pyvista as pv
        >>> icp = ICP(pv.Sphere(), pv.Sphere(center=(2, 0, 0)))
        >>> icp.get_matrix() is None
        True
        >>> icp.compute_com()
        >>> icp.get_matrix().shape
        (4, 4)
        """

        return self.matrix

    def get_metrics(self):
        """Returns quality metrics from the last ICP run, for QA and failure detection.

        Returns
        -------
        dict or None
            Open3D: ``{'fitness', 'inlier_rmse', 'correspondences'}`` (fitness is the inlier fraction of source
            points). VTK: ``{'mean_distance', 'iterations'}``. None if no ICP has been run.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.ParametricEllipsoid(30, 20, 10)
        >>> icp = ICP(source, source.translate((2, 0, 0)))
        >>> icp.compute_vtk()
        >>> sorted(icp.get_metrics())
        ['iterations', 'mean_distance']
        """

        if self.icp is None:
            return None

        if isinstance(self.icp, vtk.vtkIterativeClosestPointTransform):
            return {'mean_distance': float(self.icp.GetMeanDistance()),
                    'iterations': int(self.icp.GetNumberOfIterations())}

        return {'fitness': float(self.icp.fitness),
                'inlier_rmse': float(self.icp.inlier_rmse),
                'correspondences': len(self.icp.correspondence_set)}