"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Persistent Mesh Slicing
=======================

Description:
    This module cuts a static ROI surface mesh with a plane, many times in a
    row, without rebuilding the cutter's spatial index on each call. When the
    ROI is under a rigid registration, the plane is moved into the mesh's own
    frame instead of moving the mesh, so the cached index stays valid while
    the user drags or rotates.

Supported Inputs:
    * Meshes: PyVista PolyData surfaces, or volume grids when
      `surface_mesh=False` (the surface is extracted first), in mm
    * Rigid matrices: 4x4 homogeneous arrays mapping mesh space to world

Key Functionalities:
    1. **Transformed Slice**: `MeshSlicer.slice_transformed` returns the
       intersection lines of the plane with the rigidly moved mesh, in world
       coordinates.
    2. **Mesh Swap**: `MeshSlicer.update_mesh` points the cutter at a new
       mesh when the geometry itself changes.

Transform Note:
    For a rigid matrix M, the world plane (n, p) is pulled back as
    n_local = M^T n and p_local = M^-1 p, the mesh is cut in its own frame,
    and the result is pushed forward by M. This is identical to cutting the
    transformed mesh with the world plane.

Usage:
    >>> import numpy as np
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.slicer import MeshSlicer
    >>> # Axial cut through the middle of a 10 mm radius sphere
    >>> slicer = MeshSlicer(pv.Sphere(radius=10.0))
    >>> cut = slicer.slice_transformed(normal=(0, 0, 1), origin=(0, 0, 0), matrix=np.eye(4))

Examples
--------
>>> import numpy as np
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.mesh.slicer import MeshSlicer
>>> sphere = pv.Sphere(radius=10.0, theta_resolution=64, phi_resolution=64)
>>> slicer = MeshSlicer(sphere)
>>> cut = slicer.slice_transformed(normal=(0, 0, 1), origin=(0, 0, 0), matrix=np.eye(4))
>>> round(float(np.linalg.norm(cut.points[:, :2], axis=1).max()), 1)
10.0
"""

import vtk
import numpy as np
import pyvista as pv


class MeshSlicer:
    """
    Persistent vtkPlaneCutter for static ROI surface meshes.

    The cutter and its spatial index are built once in `__init__`. Each
    `slice_transformed` call only moves the plane, so repeated slicing while
    scrolling or rigid dragging stays fast.

    Attributes
    ----------
    mesh : pv.PolyData
        Surface being cut, in its own (untransformed) frame.

    plane : vtk.vtkPlane
        Plane shared with the cutter; updated on every slice.

    cutter : vtk.vtkPlaneCutter
        Persistent cutter bound to `mesh` and `plane`.

    Examples
    --------
    Scroll axially through a sphere:

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.slicer import MeshSlicer
    >>> slicer = MeshSlicer(pv.Sphere(radius=10.0))
    >>> cuts = [slicer.slice_transformed((0, 0, 1), (0, 0, z), np.eye(4)) for z in range(-8, 9, 2)]
    >>> len(cuts)
    9

    Slice an ROI with the rigid registration currently applied to it:

    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image[mia.Data.image_list()[0]]
    >>> slicer = MeshSlicer(image.roi['Liver'].mesh)
    >>> rigid = np.eye(4)
    >>> rigid[:3, 3] = (5.0, -2.0, 0.0)
    >>> cut = slicer.slice_transformed(normal=(0, 0, 1), origin=(0, 0, 120.0), matrix=rigid)
    """

    def __init__(self, mesh, surface_mesh=True, silence=True):
        """
        Prepare a persistent plane cutter for repeated slicing of one mesh.

        The cutter is updated once on an axial plane through (0, 0, 0) to
        force the spatial index build up front.

        Parameters
        ----------
        mesh : pv.PolyData | pv.UnstructuredGrid
            Mesh to slice, in mm.

        surface_mesh : bool
            True if `mesh` is already a surface. False extracts the outer
            surface first (e.g. a tetrahedral mesh from `Volume`).

        silence : bool
            If True, VTK warning and error events from the cutter are
            consumed so empty cuts don't flood the console.

        Examples
        --------
        From a surface:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.slicer import MeshSlicer
        >>> slicer = MeshSlicer(pv.Sphere(radius=10.0))
        >>> type(slicer.mesh).__name__
        'PolyData'

        From a volume grid, keeping only its outer surface:

        >>> grid = pv.ImageData(dimensions=(5, 5, 5)).cast_to_unstructured_grid()
        >>> slicer = MeshSlicer(grid, surface_mesh=False)
        >>> type(slicer.mesh).__name__
        'PolyData'
        """
        if surface_mesh:
            self.mesh = mesh
        else:
            self.mesh = mesh.extract_surface() if hasattr(mesh, "extract_surface") else mesh
        self.plane = vtk.vtkPlane()

        self.cutter = vtk.vtkPlaneCutter()
        # Add dummy observer to consume warnings silently
        if silence:
            self.cutter.AddObserver("WarningEvent", lambda obj, event: None)
            self.cutter.AddObserver("ErrorEvent", lambda obj, event: None)
        self.cutter.SetInputData(self.mesh)
        self.cutter.SetPlane(self.plane)

        # Force initial build
        self.plane.SetNormal(0, 0, 1)
        self.plane.SetOrigin(0, 0, 0)
        self.cutter.Update()

    def slice_transformed(self, normal, origin, matrix):
        """
        Slice the rigidly transformed mesh with a world-space plane.

        The plane is moved into the mesh frame by the inverse of `matrix`,
        the static mesh is cut, and the cut is moved back by `matrix`.

        Parameters
        ----------
        normal : array-like (3,)
            Plane normal in world space.

        origin : array-like (3,)
            A point on the plane in world space, in mm.

        matrix : np.ndarray (4, 4)
            Rigid registration matrix mapping mesh space to world. Use
            np.eye(4) when no registration is applied.

        Returns
        -------
        pv.PolyData
            Intersection lines in world space, in mm. Empty PolyData if the
            plane misses the mesh.

        Examples
        --------
        Axial cut through the center of a 10 mm radius sphere:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.slicer import MeshSlicer
        >>> sphere = pv.Sphere(radius=10.0, theta_resolution=64, phi_resolution=64)
        >>> slicer = MeshSlicer(sphere)
        >>> cut = slicer.slice_transformed(normal=(0, 0, 1), origin=(0, 0, 0), matrix=np.eye(4))
        >>> round(float(np.linalg.norm(cut.points[:, :2], axis=1).max()), 1)
        10.0

        Sphere moved 5 mm in z; cutting at z = 5 mm hits its new equator:

        >>> rigid = np.eye(4)
        >>> rigid[:3, 3] = (0.0, 0.0, 5.0)
        >>> cut = slicer.slice_transformed(normal=(0, 0, 1), origin=(0, 0, 5), matrix=rigid)
        >>> round(float(cut.points[:, 2].mean()), 3)
        5.0
        >>> round(float(np.linalg.norm(cut.points[:, :2], axis=1).max()), 1)
        10.0

        A plane outside the mesh returns an empty result:

        >>> slicer.slice_transformed(normal=(0, 0, 1), origin=(0, 0, 20), matrix=np.eye(4)).n_points
        0
        """
        inv_mat = np.linalg.inv(matrix)

        normal_h = np.append(normal, 0.0)
        origin_h = np.append(origin, 1.0)

        local_normal = (matrix.T @ normal_h)[:3]
        local_origin = (inv_mat @ origin_h)[:3]

        self.plane.SetNormal(*local_normal)
        self.plane.SetOrigin(*local_origin)

        self.cutter.Update()

        output = self.cutter.GetOutput()
        if output is None or output.GetNumberOfCells() == 0:
            return pv.PolyData()

        return pv.wrap(output).transform(matrix, inplace=False)

    def update_mesh(self, new_mesh, surface_mesh=True):
        """
        Point the cutter at a new mesh.

        Call this only when the geometry itself changes (re-contour, smooth,
        decimate). A rigid move does not need it; pass the matrix to
        `slice_transformed` instead.

        Parameters
        ----------
        new_mesh : pv.PolyData | pv.UnstructuredGrid
            Replacement mesh, in mm.

        surface_mesh : bool
            True if `new_mesh` is already a surface. False extracts the outer
            surface first.

        Returns
        -------
        None

        Examples
        --------
        Swap a 10 mm sphere for a 5 mm one:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.slicer import MeshSlicer
        >>> slicer = MeshSlicer(pv.Sphere(radius=10.0, theta_resolution=64, phi_resolution=64))
        >>> slicer.update_mesh(pv.Sphere(radius=5.0, theta_resolution=64, phi_resolution=64))
        >>> cut = slicer.slice_transformed(normal=(0, 0, 1), origin=(0, 0, 0), matrix=np.eye(4))
        >>> round(float(np.linalg.norm(cut.points, axis=1).max()), 1)
        5.0
        """

        if surface_mesh:
            mesh = new_mesh
        else:
            mesh = new_mesh.extract_surface() if hasattr(new_mesh, "extract_surface") else new_mesh

        self.mesh = mesh
        self.cutter.SetInputData(mesh)
        self.cutter.Update()