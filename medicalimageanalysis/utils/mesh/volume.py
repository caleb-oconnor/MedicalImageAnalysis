"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Under construction

Structure:

Examples
--------
>>> import numpy as np
>>> import medicalimageanalysis as mia
>>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
"""

import pytetwild

import numpy as np
import pyvista as pv


class Volume(object):
    """Converts a surface mesh into a volumetric tetrahedral mesh using PyTetWild and PyVista.
    Provides methods to create the volumetric mesh and save it to a file.
    
    Examples
    --------
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.volume import Volume
    >>> volume = Volume(pv.Sphere())
    >>> volume.create(edge_length=0.05)
    """
    def __init__(self, surface_mesh):
        """Store the source surface mesh before tetrahedralization.
        
        Examples
        --------
        >>> import pyvista as pv
        >>> volume = Volume(pv.Sphere())
        >>> volume.__init__(...)
        """
        self.surface_mesh = surface_mesh
        self.mesh = None

    def create(self, edge_length=.02):
        """Generates a tetrahedral volumetric mesh from the surface mesh.
        
        Parameters
        ----------
        edge_length : float
            Target edge length factor for tetrahedralization. Smaller values yield finer meshes.
        
        Examples
        --------
        >>> import pyvista as pv
        >>> volume = Volume(pv.Sphere())
        >>> volume.create(...)
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
        """Saves the tetrahedral mesh to a file.
        
        Parameters
        ----------
        path : str
            Path to save the mesh file. Can be .vtu or other PyVista-supported formats.
        
        Examples
        --------
        >>> import pyvista as pv
        >>> volume = Volume(pv.Sphere())
        >>> volume.write(...)
        """
        self.mesh.save(path, binary=False)
