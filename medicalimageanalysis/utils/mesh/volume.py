"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Tetrahedral Volume Meshing
==========================

Description:
    This module turns a closed surface mesh into a volumetric tetrahedral
    mesh with PyTetWild, so an ROI surface can be used for finite element
    work (FEBio heat / biomechanics) or any volume-based measurement. The
    result is stored as a PyVista UnstructuredGrid of linear tetrahedra.

Supported Inputs:
    * Surfaces: closed, manifold PyVista PolyData triangle meshes, in mm

Key Functionalities:
    1. **Tetrahedralization**: `Volume.create` runs PyTetWild and rebuilds the
       result as a VTK_TETRA UnstructuredGrid with the node order
       (3, 1, 2, 0) of the PyTetWild output.
    2. **Export**: `Volume.write` saves the grid in ASCII to any format
       PyVista supports (.vtu, .vtk, ...).

Sizing Note:
    `edge_length` is a fraction of the surface bounding-box diagonal, not a
    length in mm. 0.02 on a 100 mm diagonal targets ~2 mm edges; halving it
    roughly multiplies the tetrahedron count by 8.

Usage:
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.volume import Volume
    >>> # Tetrahedralize a 10 mm radius sphere and save it
    >>> volume = Volume(pv.Sphere(radius=10.0))
    >>> volume.create(edge_length=0.1)
    >>> volume.write('C:/Data/sphere_tet.vtu')

Examples
--------
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.mesh.volume import Volume
>>> volume = Volume(pv.Sphere(radius=10.0))
>>> volume.create(edge_length=0.1)
>>> type(volume.mesh).__name__
'UnstructuredGrid'
>>> set(volume.mesh.celltypes) == {pv.CellType.TETRA}
True
"""

import pytetwild

import numpy as np
import pyvista as pv


class Volume(object):
    """
    Converts a closed surface mesh into a tetrahedral volume mesh.

    The surface is kept on `surface_mesh` and the tetrahedral result on
    `mesh` (None until `create` is called).

    Attributes
    ----------
    surface_mesh : pv.PolyData
        Source closed surface, in mm.

    mesh : pv.UnstructuredGrid | None
        Tetrahedral mesh built by `create`.

    Examples
    --------
    Tetrahedralize a sphere:

    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.volume import Volume
    >>> volume = Volume(pv.Sphere(radius=10.0))
    >>> volume.mesh is None
    True
    >>> volume.create(edge_length=0.1)
    >>> volume.mesh.n_cells > 0
    True

    Tetrahedralize an ROI surface from an image:

    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image[mia.Data.image_list()[0]]
    >>> volume = Volume(image.roi['Liver'].mesh)
    >>> volume.create(edge_length=0.02)
    >>> volume.write('C:/Data/liver_tet.vtu')
    """
    def __init__(self, surface_mesh):
        """
        Store the source surface mesh before tetrahedralization.

        Parameters
        ----------
        surface_mesh : pv.PolyData
            Closed, manifold triangle surface in mm.

        Examples
        --------
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> volume.surface_mesh.n_points
        842
        >>> volume.mesh is None
        True
        """
        self.surface_mesh = surface_mesh
        self.mesh = None

    def create(self, edge_length=.02):
        """
        Generate a tetrahedral volume mesh from the surface mesh.

        PyTetWild is run with optimization on, then the connectivity is
        rebuilt into a VTK_TETRA UnstructuredGrid and stored on `self.mesh`.

        Parameters
        ----------
        edge_length : float
            Target edge length as a fraction of the surface bounding-box
            diagonal. Smaller values give finer meshes.

        Returns
        -------
        None
            The result is stored on `self.mesh`.

        Examples
        --------
        Coarse mesh of a 10 mm radius sphere:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> volume.create(edge_length=0.1)
        >>> set(volume.mesh.celltypes) == {pv.CellType.TETRA}
        True

        A smaller `edge_length` gives more tetrahedra:

        >>> coarse = volume.mesh.n_cells
        >>> volume.create(edge_length=0.05)
        >>> volume.mesh.n_cells > coarse
        True
        """
        base_mesh = pytetwild.tetrahedralize_pv(self.surface_mesh, edge_length_fac=edge_length, optimize=True)
        tetra_connect = base_mesh.cell_connectivity.reshape(int(len(base_mesh.cell_connectivity) / 4), 4)
        tetra_reshape = np.asarray([4 * np.ones((tetra_connect.shape[0])),
                                    tetra_connect[:, 3],
                                    tetra_connect[:, 1],
                                    tetra_connect[:, 2],
                                    tetra_connect[:, 0]], dtype=pv.ID_TYPE).T
        tetra_reformat = tetra_reshape.reshape(int(len(base_mesh.cell_connectivity) * 1.25))
        tetra_total = int(len(base_mesh.cell_connectivity) / 4)
        cell_types = np.array([pv.CellType.TETRA] * tetra_total)
        self.mesh = pv.UnstructuredGrid(tetra_reformat, cell_types, np.asarray(base_mesh.points))

    def write(self, path):
        """
        Save the tetrahedral mesh to a file in ASCII.

        `create` must be called first.

        Parameters
        ----------
        path : str
            Output path. Any PyVista-supported extension (.vtu, .vtk, ...).

        Returns
        -------
        None

        Examples
        --------
        Write and read back a sphere:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> volume.create(edge_length=0.1)
        >>> volume.write('sphere_tet.vtu')
        >>> pv.read('sphere_tet.vtu').n_cells == volume.mesh.n_cells
        True
        """
        self.mesh.save(path, binary=False)