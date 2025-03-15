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

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.pipelines.registration import (registration_icp, ICPConvergenceCriteria,
                                           TransformationEstimationPointToPlane, TransformationEstimationPointToPoint)


class ICP(object):
    def __init__(self, source, target, matrix=None):
        self.source = source
        self.target = target

        self.matrix = matrix

        self.icp = None

    def compute_com(self):
        translation = np.asarray(self.mov.center - self.ref.center)

        self.matrix = np.identity(4)
        self.matrix[:3, 3] = translation

    def compute_vtk(self, distance=10, iterations=1000, landmarks=None, com_matching=True):
        if landmarks is None:
            landmarks = int(np.round(len(self.target.points) / 10))

        self.icp.SetSource(self.source)
        self.icp.SetTarget(self.target)
        self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.SetCheckMeanDistance(1)
        self.icp.SetMeanDistanceModeToRMS()
        self.icp.SetMaximumNumberOfLandmarks(landmarks)
        self.icp.SetMaximumMeanDistance(distance)
        self.icp.SetMaximumNumberOfIterations(iterations)
        self.icp.SetStartByMatchingCentroids(com_matching)
        self.icp.Modified()
        self.icp.Update()

        self.matrix = pv.array_from_vtkmatrix(self.icp.GetMatrix())

    def compute_o3d(self, distance=10, iterations=1000, rmse=1e-7, fitness=1e-7, method='point', com_matching=True):
        ref_pcd = PointCloud()
        ref_pcd.points = Vector3dVector(self.source.points)

        mov_pcd = PointCloud()
        mov_pcd.points = Vector3dVector(self.target.points)
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
        self.matrix = self.icp.transformation

    def get_matrix(self):
        return self.matrix

    def get_correspondence_set(self):
        if hasattr(self.icp, 'correspondence_set'):
            return self.icp.correspondence_set

        else:
            return None
