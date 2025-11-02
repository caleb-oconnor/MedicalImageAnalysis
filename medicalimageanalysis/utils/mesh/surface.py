"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Under construction

Structure:

"""

import cv2
import vtk
import pyacvd
import pymeshfix
import numpy as np
import pyvista as pv

from scipy.spatial import distance


class Refinement(object):
    """
    Provides tools for refining and processing surface meshes, including smoothing, clustering, decimation, and advanced
    face splitting for mesh quality improvement.
    """
    def __init__(self, mesh):
        self.mesh = mesh

        self.correct_faces = None

        self.points = np.asarray(self.mesh.points)
        self.face = mesh.faces.reshape(int(len(mesh.faces) / 4), 4)[:, 1:]
        self.face_centers = np.asarray(mesh.cell_centers().points)
        self.face_lines_sort = np.sort(np.vstack([[[ff[0], ff[1]], [ff[0], ff[2]], [ff[1], ff[2]]] for ff in self.face]), axis=1)
        self.face_lines = np.unique(self.face_lines_sort, axis=0)

    def smooth(self, iterations=20, angle=60, passband=0.001):
        """
        Smooths the mesh geometry using a windowed sinc filter.

        Parameters
        ----------
        iterations : int
            Number of smoothing iterations.
        angle : float
            Feature angle in degrees for smoothing sharp edges.
        passband : float
            Passband for the filter (controls smoothness).

        Returns
        -------
        mesh : pyvista.PolyData
            Smoothed mesh.
        """
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(self.mesh)
        smoother.SetNumberOfIterations(iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(angle)
        smoother.SetPassBand(passband)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOff()
        # smoother.SetRelaxationFactor(0.1)
        smoother.Update()
        self.mesh = pv.PolyData(smoother.GetOutput())

        return self.mesh

    def cluster(self, points=None):
        """
        Reduces the number of points on the mesh using clustering.

        Parameters
        ----------
        points : int, optional
            Number of target points. If None, computed automatically.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh after clustering.
        """
        if points is None:
            points = self.compute_points()
        clus = pyacvd.Clustering(self.mesh)
        clus.cluster(points)
        self.mesh = clus.create_mesh()

        return self.mesh

    def decimate(self, percent=None):
        """
        Decimates (reduces) mesh complexity by removing points and faces.

        Parameters
        ----------
        percent : float, optional
            Fraction of the mesh to remove. If None, computed automatically.

        Returns
        -------
        mesh : pyvista.PolyData
            Decimated mesh.
        """
        if percent is None:
            percent = self.compute_point_percentage()

        self.mesh.decimate(percent)

        return self.mesh

    def compute_points(self):
        """
        Computes a target number of points based on the mesh size.

        Returns
        -------
        int
            Approximate number of points for clustering/decimation.
        """
        return np.round(10 * np.sqrt(self.mesh.number_of_points))

    def compute_point_percentage(self):
        """
                Computes the fraction of points to remove for decimation.

                Returns
                -------
                float
                    Percentage of points to remove.
                """
        points = self.compute_points()

        return 1 - (points / self.mesh.number_of_points)

    def tri_split(self):
        """
        Splits selected faces into smaller triangles by adding face centroids.

        Returns
        -------
        pyvista.PolyData
            Mesh with refined triangular faces.
        """
        self.find_face_correction()

        base_faces = [f for ii, f in enumerate(self.face) if ii not in self.correct_faces]
        base_length = len(self.points)
        new_points = [self.face_centers[ii] for ii in self.correct_faces]
        total_points = np.concatenate((self.points, new_points))

        new_faces = []
        for ii, f in enumerate(self.correct_faces):
            hold_face = self.face[f]
            new_faces += [[hold_face[0], hold_face[1], base_length + ii]]
            new_faces += [[hold_face[1], hold_face[2], base_length + ii]]
            new_faces += [[hold_face[0], hold_face[2], base_length + ii]]

        total_faces = np.concatenate((base_faces, np.asarray(new_faces)))
        face_syntax = [[3, t[0], t[1], t[2]] for t in total_faces]

        return pv.PolyData(total_points, face_syntax)

    def advanced_split(self):
        """
        Advanced face splitting using midpoints of edges and face corrections. Intended for complex mesh refinement
        where tri_split is insufficient.
        """
        self.find_face_correction()

        midpoint, midline = self.compute_midpoints()
        all_points = np.concatenate(self.points, midpoint)

        face_line_help = np.asarray(
            [np.where((self.face_lines_sort[:, 0] == m[0]) & (self.face_lines_sort[:, 1] == m[1]))[0] for m in midline])
        face_help = [[np.sort(self.face[int(f[0] / 3)]), np.sort(self.face[int(f[1] / 3)])] for f in face_line_help]
        face_help_stack = np.vstack(face_help)
        face_unique = np.unique(face_help_stack, axis=0)

        for f in face_unique:
            mid_1 = np.where((midline[:, 0] == f[0]) & (midline[:, 1] == f[1]))[0]
            mid_2 = np.where((midline[:, 0] == f[1]) & (midline[:, 1] == f[2]))[0]
            mid_3 = np.where((midline[:, 0] == f[0]) & (midline[:, 1] == f[2]))[0]

            point_1 = self.points[f[0]]
            point_2 = self.points[f[1]]
            point_3 = self.points[f[2]]
            center = np.mean(np.asarray([point_1, point_2, point_3]), axis=0)
            if len(mid_1) + len(mid_2) + len(mid_3) == 3:
                print(1)

    def find_face_correction(self):
        """
        Identifies faces that require splitting based on distance between face centers.
        """
        dist = distance.cdist(self.face_centers, self.face_centers, metric='euclidean')
        dist_sort = np.sort(dist, axis=1)
        dist_sum = np.sum(dist_sort[:, :6], axis=1)
        dist_sum_sort = np.argsort(dist_sum)
        self.correct_faces = dist_sum_sort[:int(len(self.points) / 4)]

    def compute_midpoints(self):
        """
        Computes midpoints of mesh edges for advanced splitting.

        Returns
        -------
        midpoint_unique : np.ndarray
            Unique midpoint coordinates.
        midline_unique : np.ndarray
            Corresponding edge indices for midpoints.
        """
        line_midpoints = np.vstack([[(self.points[line[0]] + self.points[line[1]]) / 2] for line in self.face_lines])

        midpoint_list = []
        midline_list = []
        for ii, c in enumerate(correct_faces):
            f = face[c]
            midpoint_1 = (self.points[f[0]] + self.points[f[1]]) / 2
            midpoint_2 = (self.points[f[1]] + self.points[f[2]]) / 2
            midpoint_3 = (self.points[f[2]] + self.points[f[0]]) / 2

            dist_1 = np.linalg.norm(midpoint_1 - self.points[f[2]])
            dist_2 = np.linalg.norm(midpoint_2 - self.points[f[0]])
            dist_3 = np.linalg.norm(midpoint_3 - self.points[f[1]])

            if dist_1 < dist_2 and dist_1 < dist_3:
                f_sort = np.sort([f[0], f[1]])
                midpoint_list += [
                    line_midpoints[np.where((face_lines[:, 0] == f_sort[0]) & (face_lines[:, 1] == f_sort[1]))[0][0]]]
                midline_list += [f_sort]
            elif dist_2 < dist_1 and dist_2 < dist_3:
                f_sort = np.sort([f[1], f[2]])
                midpoint_list += [
                    line_midpoints[np.where((face_lines[:, 0] == f_sort[0]) & (face_lines[:, 1] == f_sort[1]))[0][0]]]
                midline_list += [f_sort]
            else:
                f_sort = np.sort([f[2], f[0]])
                midpoint_list += [
                    line_midpoints[np.where((face_lines[:, 0] == f_sort[0]) & (face_lines[:, 1] == f_sort[1]))[0][0]]]
                midline_list += [f_sort]

        midpoint_unique, indices = np.unique(midpoint_list, axis=0, return_index=True)
        midline_unique = np.asarray(midline_list)[indices]

        return midpoint_unique, midline_unique


def clean_mesh(mesh):
    """
    Cleans a mesh by removing holes, duplicate vertices, and other inconsistencies.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh to be cleaned.

    Returns
    -------
    corrected_mesh : pyvista.PolyData
        Cleaned mesh.
    """
    mesh = mesh.copy()

    meshfix = pymeshfix.PyTMesh()
    meshfix.load_array(mesh.points, mesh.faces.reshape((-1, 4))[:, 1:])
    meshfix.clean()

    verts, faces = meshfix.return_arrays()
    new_faces = np.insert(faces, 0, 3, axis=1)
    corrected_mesh = pv.PolyData(verts, new_faces)

    return corrected_mesh


def expansion(mesh, dist):
    """
    Expands a mesh along its vertex normals by a given distance and cleans the resulting mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh to be expanded.
    dist : float
        Distance to expand along vertex normals.

    Returns
    -------
    corrected_mesh : pyvista.PolyData
        Expanded and cleaned mesh.
    """
    mesh = mesh.copy()
    mesh.points += mesh.point_normals * dist

    meshfix = pymeshfix.PyTMesh()
    meshfix.load_array(mesh.points, mesh.faces.reshape((-1, 4))[:, 1:])
    meshfix.clean()

    verts, faces = meshfix.return_arrays()
    new_faces = np.insert(faces, 0, 3, axis=1)
    corrected_mesh = pv.PolyData(verts, new_faces)

    return corrected_mesh


def surface_boundary(source_meshes, target_meshes, points, matrix=None):
    """
    Aligns source and target meshes using point clustering to ensure that both meshes
    have the same number of points along corresponding boundaries.

    Parameters
    ----------
    source_meshes : list of pyvista.PolyData
        List of source meshes.
    target_meshes : list of pyvista.PolyData
        List of target meshes.
    points : list of arrays
        Reference points for clustering each mesh.
    matrix : np.array, optional
        Transformation matrix to apply to target meshes.

    Returns
    -------
    new_sources, new_targets : lists of pyvista.PolyData
        Lists of aligned and transformed meshes.
    """
    if matrix is None:
        matrix = np.identity(4)

    new_sources = []
    new_targets = []
    for ii, s in enumerate(source_meshes):
        n = -1
        while n >= -1:
            n += 1

            refine = Refinement(s)
            hold_s = refine.cluster(points=points[ii] + n)

            refine = Refinement(target_meshes[ii])
            hold_t = refine.cluster(points=points[ii] + n)

            if len(hold_s.points) == len(hold_t.points):
                n = -100

                new_sources += [hold_s]
                new_targets += [hold_t.transform(matrix, inplace=True)]

    return new_sources, new_targets


def only_main_component(mesh):
    """
    Extracts only the largest connected component of a mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh which may contain multiple disconnected components.

    Returns
    -------
    new_mesh : pyvista.PolyData
        The largest connected component, returned as a surface mesh.
    """
    multi_block = mesh.split_bodies()

    if len(multi_block) == 1:
        return mesh

    else:
        total_points = [len(m.points) for m in multi_block]
        idx = np.argmax(total_points)
        new_mesh = multi_block[int(idx)]

        return new_mesh.extract_surface()
