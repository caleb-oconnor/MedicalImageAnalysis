"""STL reader helpers.

This module provides a thin wrapper around :mod:`pyvista` mesh loading so STL
files can be incorporated into the package's higher-level reader workflows.

Examples
--------
>>> import medicalimageanalysis as mia
>>> files = mia.file_parser(folder_path='C:/Data/Patient01')
>>> mia.read_dicoms(file_list=files['Dicom'])
"""

import os

import numpy as np
import pyvista as pv


class StlReader(object):
    """Read STL meshes into the parent reader's mesh collection.
    
    Examples
    --------
    >>> from medicalimageanalysis.read.stl import StlReader
    >>> stl_reader = StlReader(reader=mesh_reader)
    >>> stl_reader.input_files(['C:/Data/organ.stl'])
    """
    def __init__(self, reader):
        """Store the parent reader that owns the mesh list and file registry.
        
        Examples
        --------
        >>> stl_reader = StlReader(reader=mesh_reader)
        >>> stl_reader.__init__()
        """
        self.reader = reader

    def input_files(self, files):
        """Register STL file paths on the parent reader.
        
        Examples
        --------
        >>> stl_reader = StlReader(reader=mesh_reader)
        >>> stl_reader.input_files(... )
        """
        self.reader.files['Stl'] = files

    def load(self):
        """Load every registered STL file.
        
        Examples
        --------
        >>> stl_reader = StlReader(reader=mesh_reader)
        >>> stl_reader.load()
        """
        for file_path in self.reader.files['Stl']:
            self.read(file_path)

    def read(self, path):
        """Read one STL file and append the mesh to ``reader.meshes``.
        
        Examples
        --------
        >>> stl_reader = StlReader(reader=mesh_reader)
        >>> stl_reader.read('C:/Data/model.stl')
        """
        self.reader.meshes += [pv.read(path)]
