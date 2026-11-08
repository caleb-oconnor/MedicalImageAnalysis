"""STL Mesh Reader Helper
======================

Description:
    Thin wrapper around `pyvista.read` that loads STL geometry files into
    a parent mesh reader. It does not touch the global `Data` registry; the
    parent decides what to do with the meshes it collects.

Parent Reader Contract:
    The `reader` passed in must expose:
        * `files`  : dict with a 'Stl' key holding a list of paths
        * `meshes` : list that loaded meshes are appended to

Key Functionalities:
    1. **Registration**: `input_files` stores the paths on the parent.
    2. **Batch Loading**: `load` reads every registered path in order.
    3. **Single Loading**: `read` loads one path on demand.

Usage:
    >>> from types import SimpleNamespace
    >>> from medicalimageanalysis.read.stl import StlReader
    >>> mesh_reader = SimpleNamespace(files={'Stl': []}, meshes=[])
    >>> stl_reader = StlReader(reader=mesh_reader)
    >>> stl_reader.input_files(['C:/Data/liver.stl'])
    >>> stl_reader.load()

Examples
--------
>>> import medicalimageanalysis as mia
>>> from types import SimpleNamespace
>>> from medicalimageanalysis.read.stl import StlReader
>>> files = mia.file_parser(folder_path='C:/Data/Patient01')
>>> files['Stl']
['C:/Data/Patient01/liver.stl', 'C:/Data/Patient01/gtv.stl']
>>> mesh_reader = SimpleNamespace(files=files, meshes=[])
>>> StlReader(reader=mesh_reader).load()
>>> len(mesh_reader.meshes)
2
>>> mesh_reader.meshes[0].n_points
48213
"""

import os

import numpy as np
import pyvista as pv


class StlReader(object):
    """
    Read STL meshes into the parent reader's mesh collection.

    Parameters
    ----------
    reader : object
        Parent reader exposing a `files` dict (with a 'Stl' key) and a
        `meshes` list.

    Attributes
    ----------
    reader : object
        The parent reader that owns the file registry and mesh list.

    Examples
    --------
    Register and load two meshes:

    >>> from types import SimpleNamespace
    >>> from medicalimageanalysis.read.stl import StlReader
    >>> mesh_reader = SimpleNamespace(files={'Stl': []}, meshes=[])
    >>> stl_reader = StlReader(reader=mesh_reader)
    >>> stl_reader.input_files(['C:/Data/liver.stl', 'C:/Data/gtv.stl'])
    >>> stl_reader.load()
    >>> len(mesh_reader.meshes)
    2

    Meshes are plain PyVista objects:

    >>> liver = mesh_reader.meshes[0]
    >>> type(liver).__name__, liver.n_points
    ('PolyData', 48213)
    """
    def __init__(self, reader):
        """
        Store the parent reader that owns the mesh list and file registry.

        Parameters
        ----------
        reader : object
            Parent reader with `files` and `meshes` attributes.

        Examples
        --------
        Nothing is read at construction time:

        >>> from types import SimpleNamespace
        >>> from medicalimageanalysis.read.stl import StlReader
        >>> mesh_reader = SimpleNamespace(files={'Stl': []}, meshes=[])
        >>> stl_reader = StlReader(reader=mesh_reader)
        >>> stl_reader.reader is mesh_reader
        True
        """
        self.reader = reader

    def input_files(self, files):
        """
        Register STL file paths on the parent reader.

        Replaces (does not extend) `reader.files['Stl']`.

        Parameters
        ----------
        files : list[str]
            Paths to STL files.

        Returns
        -------
        None

        Examples
        --------
        Register paths found by the file parser:

        >>> import medicalimageanalysis as mia
        >>> from types import SimpleNamespace
        >>> from medicalimageanalysis.read.stl import StlReader
        >>> mesh_reader = SimpleNamespace(files={'Stl': []}, meshes=[])
        >>> stl_reader = StlReader(reader=mesh_reader)
        >>> files = mia.file_parser(folder_path='C:/Data/Patient01')
        >>> stl_reader.input_files(files['Stl'])
        >>> mesh_reader.files['Stl']
        ['C:/Data/Patient01/liver.stl', 'C:/Data/Patient01/gtv.stl']

        A second call replaces the earlier list:

        >>> stl_reader.input_files(['C:/Data/pelvis.stl'])
        >>> mesh_reader.files['Stl']
        ['C:/Data/pelvis.stl']
        """
        self.reader.files['Stl'] = files

    def load(self):
        """
        Read every registered STL file and append it to `reader.meshes`.

        Returns
        -------
        None

        Examples
        --------
        Load all registered files in order:

        >>> from types import SimpleNamespace
        >>> from medicalimageanalysis.read.stl import StlReader
        >>> mesh_reader = SimpleNamespace(
        ...     files={'Stl': ['C:/Data/liver.stl', 'C:/Data/gtv.stl']},
        ...     meshes=[],
        ... )
        >>> StlReader(reader=mesh_reader).load()
        >>> [mesh.n_points for mesh in mesh_reader.meshes]
        [48213, 2410]

        Calling `load` twice appends the meshes again:

        >>> StlReader(reader=mesh_reader).load()
        >>> len(mesh_reader.meshes)
        4
        """
        for file_path in self.reader.files['Stl']:
            self.read(file_path)

    def read(self, path):
        """
        Read one STL file and append the mesh to `reader.meshes`.

        The path does not need to be registered first.

        Parameters
        ----------
        path : str
            Path to the STL file.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If `path` does not exist.

        Examples
        --------
        Load a single file on demand:

        >>> from types import SimpleNamespace
        >>> from medicalimageanalysis.read.stl import StlReader
        >>> mesh_reader = SimpleNamespace(files={'Stl': []}, meshes=[])
        >>> stl_reader = StlReader(reader=mesh_reader)
        >>> stl_reader.read('C:/Data/pelvis.stl')
        >>> mesh_reader.meshes[-1].n_cells
        96412
        """
        self.reader.meshes += [pv.read(path)]