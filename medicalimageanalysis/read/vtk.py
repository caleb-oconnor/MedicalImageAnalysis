"""VTK mesh reader helpers.

This module provides a simple wrapper around :mod:`pyvista` mesh loading for
VTK geometry files used in the package's mesh import workflows.

Examples
--------
>>> import medicalimageanalysis as mia
>>> files = mia.file_parser(folder_path='C:/Data/Patient01')
>>> mia.read_dicoms(file_list=files['Dicom'])
"""

import os

import numpy as np
import pyvista as pv


class VtkReader(object):
    """Read VTK meshes into the parent reader's mesh collection.
    
    Examples
    --------
    >>> from medicalimageanalysis.read.vtk import VtkReader
    >>> vtk_reader = VtkReader(reader=mesh_reader)
    >>> vtk_reader.input_files(['C:/Data/organ.vtk'])
    """
    def __init__(self, reader):
        """Store the parent reader that owns the mesh list and file registry.
        
        Examples
        --------
        >>> vtk_reader = VtkReader(reader=mesh_reader)
        >>> vtk_reader.__init__()
        """
        self.reader = reader

    def input_files(self, files):
        """Register VTK file paths on the parent reader.
        
        Examples
        --------
        >>> vtk_reader = VtkReader(reader=mesh_reader)
        >>> vtk_reader.input_files(... )
        """
        self.reader.files['Vtk'] = files

    def load(self):
        """Load every registered VTK file.
        
        Examples
        --------
        >>> vtk_reader = VtkReader(reader=mesh_reader)
        >>> vtk_reader.load()
        """
        for file_path in self.reader.files['Vtk']:
            self.read(file_path)

    def read(self, path):
        """Read one VTK file and append the mesh to ``reader.meshes``.
        
        Examples
        --------
        >>> vtk_reader = VtkReader(reader=mesh_reader)
        >>> vtk_reader.read('C:/Data/model.vtk')
        """
        self.reader.meshes += [pv.read(path)]
