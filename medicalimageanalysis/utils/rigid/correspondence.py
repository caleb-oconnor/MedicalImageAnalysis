"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import numpy as np

from scipy.spatial.distance import cdist

from .icp import ICP
from ..mesh.surface import only_main_component


class SurfaceMatching(object):
    def __init__(self, source_mesh, target_mesh, initial_correspondence=None, main_component=True):
        if main_component:
            self.source_mesh = only_main_component(source_mesh)
            self.target_mesh = only_main_component(target_mesh)
        else:
            self.source_mesh = source_mesh
            self.target_mesh = target_mesh

        self.initial_correspondence = initial_correspondence

        self.fix_idx = None
        self.boundary_condition = np.zeros((len(self.source_mesh.points), 3))

    def compute_rigid(self, distance=10, iterations=1000, rmse=1e-7, fitness=1e-7):
        icp = ICP(self.source_mesh, self.target_mesh)
        icp.compute_o3d(distance=distance, iterations=iterations, rmse=rmse, fitness=fitness, method='plane')
        matrix = np.linalg.inv(icp.matrix)

        self.target_mesh.transform(matrix, inplace=True)
        self.initial_correspondence = icp.get_correspondence_set()
        self.ignore_faces = []
        self.ignore_points = None

    def compute_matching(self):
        self.find_duplicates()
        self.simply_fix()
        self.surrounding_faces()

    def find_duplicates(self):
        unique_arr, indices, counts = np.unique(self.initial_correspondence[:, 1],
                                                return_index=True,
                                                return_counts=True)
        good_correspond = self.initial_correspondence[np.where(counts == 1)[0], :]

        total_correspondence = np.zeros((len(self.source_mesh.points), 1)) - 1
        total_correspondence[good_correspond[:, 0]] = good_correspond[:, 1].reshape(-1, 1)
        good_idx = np.where(total_correspondence >= 0)[0]
        self.fix_idx = np.where(total_correspondence == -1)[0]

        self.source_mesh.point_data['idx'] = np.arange(len(self.source_mesh.points))
        self.source_mesh.point_data['good'] = -1 * np.ones(len(self.source_mesh.points))
        self.source_mesh.point_data['good'][good_idx] = 1

        target_points = self.target_mesh.points[total_correspondence[good_idx].astype(np.int64)][:, 0, :]
        self.boundary_condition[good_idx] = np.asarray(target_points)- np.asarray(self.source_mesh.points[good_idx])

    def simply_fix(self):
        extracted_cells = [self.source_mesh.extract_points(idx, adjacent_cells=True) for idx in self.fix_idx]
        surrounding_point_indices = [np.asarray(extract.point_data['idx']) for extract in extracted_cells]
        check_surrounding = [self.source_mesh.point_data['good'][s] for s in surrounding_point_indices]
        simpy_idx = [ii for ii, s in enumerate(check_surrounding) if np.sum(s) == len(s) - 2]

        for idx in simpy_idx:
            fix = self.fix_idx[idx]

            surround_idx = surrounding_point_indices[idx][np.where(surrounding_point_indices[idx] != fix)[0]]
            surround_bc = np.mean(self.boundary_condition[surround_idx], axis=0)
            self.boundary_condition[idx] = surround_bc

        diff_set = np.setdiff1d(np.arange(len(self.fix_idx)), simpy_idx)
        self.fix_idx = self.fix_idx[diff_set]

    def surrounding_faces(self):
        extracted_cells = [self.source_mesh.extract_points(idx, adjacent_cells=True) for idx in self.fix_idx]
        indices = [np.asarray(extract.point_data['idx']) for extract in extracted_cells]
        connection = [extract.cell_connectivity for extract in extracted_cells]

        print(1)

    def face_check(self):
        faces = self.source_mesh.faces.reshape((-1, 4))[:, 1:]
        good_face_idx = np.where(np.sum(self.source_mesh.point_data['good'][faces], axis=1) == 3)[0]
        good_faces = faces[good_face_idx]

        good_face_points = np.asarray(self.source_mesh.points)[good_faces]

        a = np.linalg.norm(good_face_points[:, 0] - good_face_points[:, 1], axis=1)
        b = np.linalg.norm(good_face_points[:, 1] - good_face_points[:, 2], axis=1)
        c = np.linalg.norm(good_face_points[:, 2] - good_face_points[:, 0], axis=1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        base_quality = (4 * np.sqrt(3) * area) / (a ** 2 + b ** 2 + c ** 2)
        base_min = np.min(base_quality)

        deformed_quality = []
        deformed_area = []
        for idx in range(len(good_faces)):
            new_face = np.asarray(self.source_mesh.points)[good_faces[idx]] - self.boundary_condition[good_faces[idx]]
            a = np.linalg.norm(new_face[0, :] - new_face[1, :])
            b = np.linalg.norm(new_face[1, :] - new_face[2, :])
            c = np.linalg.norm(new_face[2, :] - new_face[0, :])
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            deformed_area += [area]
            deformed_quality += [(4 * np.sqrt(3) * area) / (a ** 2 + b ** 2 + c ** 2)]
        #     if deformed_quality[-1] < .5:
        #         self.ignore_faces += [good_faces[idx]]
        #
        # self.ignore_points = np.unique(self.ignore_faces)
        # self.boundary_condition[self.ignore_points] = [0, 0, 0]
