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
from open3d.pipelines.registration import registration_icp, TransformationEstimationPointToPoint, ICPConvergenceCriteria


class IcpVtk(object):
    def __init__(self, ref, mov):
        self.source = mov
        self.target = ref

        self.landmarks = int(np.round(len(mov.points)/10))
        self.distance = 1e-5
        self.iterations = 1000

        self.icp = vtk.vtkIterativeClosestPointTransform()

        self.reverse_transform = False

    def update_parameters(self, landmarks=None, distance=None, iterations=None):
        if landmarks:
            self.landmarks = landmarks

        if distance:
            self.distance = distance

        if iterations:
            self.iterations = iterations

    def compute_icp(self, com_matching=True):
        self.icp.SetSource(self.source)
        self.icp.SetTarget(self.target)
        self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.SetCheckMeanDistance(1)
        self.icp.SetMeanDistanceModeToRMS()
        self.icp.SetMaximumNumberOfLandmarks(self.landmarks)
        self.icp.SetMaximumMeanDistance(self.distance)
        self.icp.SetMaximumNumberOfIterations(self.iterations)
        self.icp.SetStartByMatchingCentroids(com_matching)
        self.icp.Modified()
        self.icp.Update()

        matrix = pv.array_from_vtkmatrix(self.icp.GetMatrix())
        if self.reverse_transform:
            matrix = np.linalg.inv(matrix)

        return matrix

    def compute_error(self):
        matrix = pv.array_from_vtkmatrix(self.icp.GetMatrix())
        new_source = self.source.transform(matrix, inplace=False)

        closest_cells, closest_points = self.target.find_closest_cell(new_source.points, return_closest_point=True)
        d_exact = np.linalg.norm(new_source.points - closest_points, axis=1)
        new_source["distances"] = d_exact
        return {'Min': np.min(d_exact), 'Mean': np.mean(d_exact), 'Max': np.max(d_exact)}


class IcpOpen3d(object):
    def __init__(self, ref_pv=None, mov_pv=None):
        self.ref_pcd = PointCloud()
        self.ref_pcd.points = Vector3dVector(ref_pv.points)
        self.mov_pcd = PointCloud()
        self.mov_pcd.points = Vector3dVector(mov_pv.points)

        self.distance = 1
        self.iterations = 1000
        self.rmse = 1e-7
        self.fitness = 1e-7

        self.initial_transform = None
        self.compute_com()

        self.registration = None
        self.parameter_results = {'transform': None, 'fitness': None, 'rmse': None}
        self.angles = None
        self.translation = None
        self.new_mesh = None

    def set_initial_transform(self, transform):
        self.initial_transform = transform

    def set_icp_settings(self, distance=None, iterations=None, rmse=None, fitness=None):
        if max_distancee:
            self.distance = distance

        if max_iteration:
            self.iterations = iterations

        if rmse:
            self.rmse = rmse

        if fitness:
            self.fitness = fitness

    def compute(self, com_matching=True):
        if com_matching:
            c = self.mov_pcd.get_center() - self.ref_pcd.get_center()
            self.initial_transform = np.asarray([[1, 0, 0, c[0]], [0, 1, 0, c[1]], [0, 0, 1, c[2]], [0, 0, 0, 1]])
        else:
            self.initial_transform = np.identity(4, dtye=np.float32)

        self.registration = registration_icp(self.ref_pcd, self.mov_pcd, self.distance, self.initial_transform,
                                             TransformationEstimationPointToPoint(),
                                             ICPConvergenceCriteria(max_iteration=self.iterations,
                                                                    relative_rmse=self.rmse,
                                                                    relative_fitness=self.fitness))

        self.parameter_results['transform'] = self.registration.transformation
        self.parameter_results['fitness'] = self.registration.fitness
        self.parameter_results['rmse'] = self.registration.inlier_rmse

        r11, r12, r13 = self.parameter_results['transform'][0][0:3]
        r21, r22, r23 = self.parameter_results['transform'][1][0:3]
        r31, r32, r33 = self.parameter_results['transform'][2][0:3]

        angle_z = np.arctan(r21 / r11)
        angle_y = np.arctan(-r31 * np.cos(angle_z) / r11)
        angle_x = np.arctan(r32 / r33)

        self.new_mesh = self.ref_pcd.transform(self.parameter_results['transform'])
        self.angles = [angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi]
        self.translation = self.new_mesh.get_center() - self.ref_com

    def correspondence_array(self):
        return self.registration.correspondence_set


class MeshCenterOfMass(object):
    def __init__(self, ref, mov):
        self.ref = ref
        self.mov = mov

        self.matrix = None

    def com_transfer(self):
        ref_com = self.ref.center
        mov_com = self.mov.center

        translation = mov_com - ref_com

        self.matrix = np.identity(4)
        self.matrix[0, 3] = -translation[0]
        self.matrix[1, 3] = -translation[1]
        self.matrix[2, 3] = -translation[2]

        return self.matrix
