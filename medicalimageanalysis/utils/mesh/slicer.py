"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
"""

import vtk
import numpy as np
import pyvista as pv


class MeshSlicer:
    """Persistent vtkPlaneCutter for static ROI surface meshes.
    Caches the sphere tree structure once and uses inverse plane transformations
    to slice without invalidating GPU/CPU spatial indices.

    Examples
    --------
    >>> from medicalimageanalysis.utils.roi.discrete_converters import MeshSlicer
    >>> obj = MeshToContour(...)
    """

    def __init__(self, mesh, surface_mesh=True, silence=True):
        """Prepare a persistent plane cutter for repeated slicing of one mesh.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> helper = MeshSlicer(pv.Sphere())
        >>> helper.__init__(...)
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
        """Slices the static mesh by transforming the plane in reverse.

        Parameters
        ----------
        normal : array-like (3,)
            Base viewing normal vector.
        origin : array-like (3,)
            Slice origin position in world space.
        matrix : np.ndarray (4, 4)
            The current active Rigid registration matrix.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> helper = MeshSlicer(pv.Sphere())
        >>> helper.slice_transformed(...)
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
        """Re-initializes the cutter only when mesh geometry actually changes.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> helper = MeshSlicer(pv.Sphere())
        >>> helper.update_mesh(...)
        """

        if surface_mesh:
            mesh = new_mesh
        else:
            mesh = new_mesh.extract_surface() if hasattr(new_mesh, "extract_surface") else new_mesh

        self.mesh = mesh
        self.cutter.SetInputData(mesh)
        self.cutter.Update()
