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

import pytetwild


class Volume(object):
    def __init__(self, surface_mesh):
        self.surface_mesh = surface_mesh
        self.mesh = None

    def create(self, edge_length=.35):
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
        self.mesh.save(path, binary=False)
        