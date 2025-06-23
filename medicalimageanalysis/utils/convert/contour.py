
import os
import sys
import copy
import time

import cv2
import vtk
import pyacvd
import numpy as np
import pyvista as pv

from scipy.spatial.distance import cdist
from matplotlib import path


class ContourToMesh(object):
    def __init__(self, contour, plane='Axial', existing_mesh=None):
        self.contour = contour
        self.plane = plane
        self.existing_mesh = existing_mesh

        self.contour_2d = self.convert_to_2d()
        self.slice_locations = self.get_slice_locations()
        self.slice_dict = self.create_slice_dictionary()
        self.slice_info = [np.where(self.slice_locations == s)[0] for s in list(self.slice_dict.keys())]

        self.mesh = None
        self.contour_dist = None

    def convert_to_2d(self):
        if self.plane == 'Axial':
            return [np.vstack((c[:, :2], c[0, :2])) for c in self.contour]

        elif self.plane == 'Coronal':
            return [np.vstack((np.hstack((c[:, 0], c[:, 2])), np.hstack((c[0, 0], c[0, 2])))) for c in self.contour]

        else:
            return [np.vstack((c[:, 1:], c[0, 1:])) for c in self.contour]

    def get_slice_locations(self):
        if self.plane == 'Axial':
            return [int(np.round(c[0, 2])) for c in self.contour]

        elif self.plane == 'Coronal':
            return [int(np.round(c[0, 1])) for c in self.contour]

        else:
            return [int(np.round(c[0, 0])) for c in self.contour]

    def create_slice_dictionary(self):
        unique_slices = np.unique(self.slice_locations)
        slice_dict = dict.fromkeys(unique_slices)
        for s in unique_slices:
            pre = np.where(unique_slices == s - 1)[0]
            post = np.where(unique_slices == s + 1)[0]
            if len(pre) == 0 and len(post) == 0:
                slice_dict[s] = 'Solo'
            elif len(pre) > 0 and len(post) > 0:
                slice_dict[s] = 'Both'
            elif len(pre) > 0:
                slice_dict[s] = 'Before'
            else:
                slice_dict[s] = 'After'

        return slice_dict

    def run(self):
        print(1)

    def distance(self, slice_idx=None):
        if slice_idx is None:
            c_next = [cdist(contour_loop[idx], contour_loop[idx + 1]) for idx in range(len(contour_loop) - 1)]


def dist_time(contour):
    t1 = time.time()
    contour_loop = [np.vstack((c[:, :2], c[0, :2])) for c in contour]
    if len(contour_loop) > 1:
        c_next = [cdist(contour_loop[idx], contour_loop[idx + 1]) for idx in range(len(contour_loop) - 1)]

    print(np.round(time.time() - t1, 3))


def mask_time():
    t1 = time.time()
    mask = mia.Data.images['CT 01'].rois['liver'].compute_mask()
    print(np.round(time.time() - t1, 3))


def grid_time(contour):
    t1 = time.time()
    for kk, c in enumerate(contour):
        if 20 < kk < 30:
            c = c[:-1, :2]
            c_idx = np.repeat(np.arange(len(c)), 2)[1:-1]
            c_idx = c_idx.reshape(int(len(c_idx) / 2), 2)
            c_lines = np.vstack((c_idx, [c_idx[-1, 1], c_idx[0, 0]]))

            c_lines_stacked = np.hstack((2 * np.ones((int(c_lines.shape[0]), 1)), c_lines)).astype(np.int64)
            points = pv.PolyData(np.hstack((c, np.zeros((len(c_lines_stacked), 1)))), lines=c_lines_stacked)

            b = points.bounds

            grid_interval = 5
            grid_x, grid_y, grid_z = np.mgrid[b[0]:b[1]:grid_interval, b[2]:b[3]:grid_interval, 0:1:1]
            # grid_x, grid_y, grid_z = np.mgrid[b[0]:b[1]:1, b[2]:b[3]:1, 0:1:1]
            structured_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
            grid_points = np.asarray(structured_grid.points)[:, :2]
            q = path.Path(c)
            a = q.contains_points(grid_points)
            inside_points = grid_points[np.where(a == 1)[0]]
            grid_idx = np.where(np.sort(cdist(inside_points, c))[:, 0] > grid_interval)[0]
            inside_points_2 = np.asarray(inside_points)[grid_idx, :2]
            inside_pv = pv.PolyData(np.hstack((inside_points_2, np.zeros((int(inside_points_2.shape[0]), 1)))))
            a = np.vstack((points.points, inside_pv.points))
            aa = pv.PolyData(a)
            # test = aa.delaunay_2d()
    print(np.round(time.time() - t1, 3))
