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

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.pipelines.registration import (registration_icp, ICPConvergenceCriteria,
                                           TransformationEstimationPointToPlane, TransformationEstimationPointToPoint)


class ICP(object):
    """
    Performs Iterative Closest Point (ICP) registration between a source and a target mesh or point cloud. Supports
    both VTK-based and Open3D-based ICP implementations.
    """
    def __init__(self, source, target, matrix=None):
        """
        Initializes the ICP object.

        Parameters
        ----------
        source : pyvista.PolyData or point cloud
            The moving/source mesh or point cloud to align.
        target : pyvista.PolyData or point cloud
            The reference/target mesh or point cloud.
        matrix : np.ndarray, optional
            Initial 4x4 transformation matrix. Defaults to None (identity).
        """
        self.source = source
        self.target = target

        self.matrix = matrix

        self.icp = None

    def compute_com(self):
        """
        Computes an initial translation by matching the center of mass (COM) of the source and target meshes.
        """
        translation = np.asarray(self.mov.center) - np.asarray(self.ref.center)

        self.matrix = np.identity(4)
        self.matrix[:3, 3] = translation

    def compute_vtk(self, distance=1e-5, iterations=1000, landmarks=None, com_matching=True, inverse=False):
        """
        Performs ICP using VTK.

        Parameters
        ----------
        distance : float
            Maximum mean distance for convergence.
        iterations : int
            Maximum number of ICP iterations.
        landmarks : int, optional
            Number of landmarks to sample for ICP. Defaults to 1/10 of target points.
        com_matching : bool
            Whether to start by matching centroids of source and target.
        inverse : bool
            If True, stores the inverse of the resulting transformation matrix.
        """
        if landmarks is None:
            landmarks = int(np.round(len(self.target.points) / 10))

        self.icp = vtk.vtkIterativeClosestPointTransform()
        self.icp.SetSource(self.source)
        self.icp.SetTarget(self.target)
        self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.SetCheckMeanDistance(1)
        self.icp.SetMeanDistanceModeToRMS()
        if landmarks:
            self.icp.SetMaximumNumberOfLandmarks(landmarks)
        self.icp.SetMaximumMeanDistance(distance)
        self.icp.SetMaximumNumberOfIterations(iterations)
        if com_matching:
            self.icp.SetStartByMatchingCentroids(com_matching)
        self.icp.Modified()
        self.icp.Update()

        if inverse:
            self.matrix = np.linalg.inv(pv.array_from_vtkmatrix(self.icp.GetMatrix()))
        else:
            self.matrix = pv.array_from_vtkmatrix(self.icp.GetMatrix())

    def compute_o3d(self, distance=10, iterations=1000, rmse=1e-7, fitness=1e-7, method='point', com_matching=True,
                    inverse=False):
        """
        Performs ICP using Open3D.

        Parameters
        ----------
        distance : float
            Maximum correspondence distance.
        iterations : int
            Maximum number of ICP iterations.
        rmse : float
            Convergence threshold for relative RMSE.
        fitness : float
            Convergence threshold for relative fitness.
        method : str
            'point' for point-to-point ICP, 'plane' for point-to-plane ICP.
        com_matching : bool
            Whether to initialize with translation by centers of mass.
        inverse : bool
            If True, stores the inverse of the resulting transformation matrix.
        """
        ref_pcd = PointCloud()
        ref_pcd.points = Vector3dVector(np.asarray(self.source.points))

        mov_pcd = PointCloud()
        mov_pcd.points = Vector3dVector(np.asarray(self.target.points))
        mov_pcd.normals = Vector3dVector(np.asarray(self.target.point_normals))

        initial_transform = np.identity(4)
        if com_matching:
            translation = mov_pcd.get_center() - ref_pcd.get_center()
            initial_transform[:3, 3] = translation

        if method == 'point':
            self.icp = registration_icp(ref_pcd, mov_pcd, distance, initial_transform,
                                        TransformationEstimationPointToPoint(),
                                        ICPConvergenceCriteria(max_iteration=iterations,
                                                               relative_rmse=rmse,
                                                               relative_fitness=fitness))
        else:
            self.icp = registration_icp(ref_pcd, mov_pcd, distance, initial_transform,
                                        TransformationEstimationPointToPlane())

        if inverse:
            self.matrix = np.linalg.inv(self.icp.transformation)
        else:
            self.matrix = self.icp.transformation

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
