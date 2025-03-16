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

    def compute_rigid(self, distance=10, iterations=1000, rmse=1e-7, fitness=1e-7):
        icp = ICP(self.source_mesh, self.target_mesh)
        icp.compute_o3d(distance=distance, iterations=iterations, rmse=rmse, fitness=fitness)
        matrix = np.linalg.inv(icp.matrix)

        self.target_mesh.transform(matrix, inplace=True)
        self.initial_correspondence = icp.get_correspondence_set()



