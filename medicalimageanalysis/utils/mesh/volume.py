"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Tetrahedral Volume Meshing
==========================

Description:
    This module turns one or more closed surface meshes into a single
    conforming tetrahedral mesh with PyTetWild (>= 0.4), so ROI surfaces can
    be used for finite element work (FEBio heat / biomechanics) or any
    volume-based measurement. Surfaces may be nested (e.g. tumor inside
    liver), separate, or overlapping. Every interior surface is embedded in
    the mesh, so no tetrahedron crosses it, and each tetrahedron is labeled
    with the surface that most tightly encloses it.

Supported Inputs:
    * Surfaces: closed PyVista PolyData meshes, in mm. A single PolyData or a
      list of them.
    * Sizing fields: all-tetrahedral PyVista UnstructuredGrids with a point
      array of target edge lengths in mm, or the same data as raw arrays.

Key Functionalities:
    1. **Tetrahedralization**: `Volume.create` merges all surfaces, runs
       PyTetWild with VTK node ordering, and stores a VTK_TETRA
       UnstructuredGrid on `mesh`.
    2. **Region Labels**: `mesh.cell_data['Region']` holds, for every
       tetrahedron, the index of the smallest input surface that contains
       it (-1 if none, only possible with `disable_filtering=True`).
    3. **Per-Region Sizing**: `region_edge_lengths` gives each surface its
       own target edge length in mm; a sizing field is built automatically.
       A custom `sizing_field` grid or raw `bg_*` arrays can be used instead.
    4. **Region Extraction**: `Volume.extract_region` returns the
       tetrahedra of one region as its own grid.
    5. **Export**: `Volume.write` saves the grid in ASCII to any format
       PyVista supports (.vtu, .vtk, ...).

Sizing Note:
    `edge_length_fac` is a fraction of the bounding-box diagonal of all
    surfaces combined, not a length in mm. 0.02 on a 100 mm diagonal targets
    ~2 mm edges; halving it roughly multiplies the tetrahedron count by 8.
    `edge_length_abs`, `region_edge_lengths` and sizing field values are in
    mm. Sizing is a target, so elements near a finer region come out smaller.

Orientation Note:
    Every surface is oriented with outward normals before meshing. PyTetWild
    keeps tetrahedra by winding number, so an inward-facing nested surface
    would otherwise be carved out as a hole instead of meshed as a region.

Usage:
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.volume import Volume
    >>> # Mesh a 20 mm sphere with an 8 mm sphere inside, finer inside
    >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)])
    >>> volume.create(region_edge_lengths=[4.0, 1.5], quiet=True)
    >>> volume.write('C:/Data/nested_tet.vtu')

Examples
--------
>>> import numpy as np
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.mesh.volume import Volume
>>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)])
>>> volume.create(region_edge_lengths=[4.0, 1.5], quiet=True)
>>> set(volume.mesh.celltypes) == {pv.CellType.TETRA}
True
>>> sorted(np.unique(volume.mesh.cell_data['Region']).tolist())
[0, 1]
"""

import pytetwild

import numpy as np
import pyvista as pv


class Volume(object):
    """
    Converts one or more closed surface meshes into a single tetrahedral mesh.

    The prepared surfaces are kept on `surface_meshes` and the tetrahedral
    result on `mesh` (None until `create` is called). The index of a surface
    in `surface_meshes` is its region label in `mesh.cell_data['Region']`.

    Attributes
    ----------
    surface_meshes : list[pv.PolyData]
        Source surfaces, triangulated, cleaned, and oriented outward, in mm.

    mesh : pv.UnstructuredGrid | None
        Tetrahedral mesh built by `create`, with cell array 'Region'.

    Examples
    --------
    Tetrahedralize a single sphere:

    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.volume import Volume
    >>> volume = Volume(pv.Sphere(radius=10.0))
    >>> volume.mesh is None
    True
    >>> volume.create(edge_length_fac=0.1, quiet=True)
    >>> volume.mesh.n_cells > 0
    True

    Nested and separate surfaces in one mesh:

    >>> outer = pv.Sphere(radius=20.0)
    >>> inner = pv.Sphere(radius=8.0)
    >>> separate = pv.Sphere(radius=6.0, center=(40.0, 0.0, 0.0))
    >>> volume = Volume([outer, inner, separate])
    >>> volume.create(edge_length_abs=4.0, quiet=True)
    >>> sorted(set(volume.mesh.cell_data['Region'].tolist()))
    [0, 1, 2]

    Liver with a tumor, finer inside the tumor:

    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image[mia.Data.image_list()[0]]
    >>> volume = Volume([image.roi['Liver'].mesh, image.roi['GTV'].mesh])
    >>> volume.create(region_edge_lengths=[3.0, 1.0])
    >>> volume.write('C:/Data/liver_gtv_tet.vtu')
    """
    def __init__(self, surface_meshes):
        """
        Prepare and store the source surface meshes before tetrahedralization.

        Parameters
        ----------
        surface_meshes : pv.PolyData | list[pv.PolyData]
            One closed surface, or a list of them, in mm. List order sets the
            region labels.

        Examples
        --------
        A single surface is stored as a one-item list:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> len(volume.surface_meshes)
        1
        >>> volume.mesh is None
        True

        Several surfaces keep their order:

        >>> outer = pv.Sphere(radius=20.0)
        >>> inner = pv.Sphere(radius=8.0, theta_resolution=10, phi_resolution=10)
        >>> volume = Volume([outer, inner])
        >>> [surface.n_points for surface in volume.surface_meshes]
        [842, 82]
        """
        if isinstance(surface_meshes, pv.PolyData):
            surface_meshes = [surface_meshes]

        self.surface_meshes = [self.prepare_surface(surface) for surface in surface_meshes]
        self.mesh = None

    @staticmethod
    def prepare_surface(surface):
        """
        Triangulate, clean, and orient a surface with outward normals.

        Parameters
        ----------
        surface : pv.PolyData
            Closed surface, in mm.

        Returns
        -------
        pv.PolyData
            Triangle surface with outward cell normals in 'Normals'.

        Examples
        --------
        An inside-out sphere comes back facing outward:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> flipped = pv.Sphere(radius=10.0).flip_faces()
        >>> fixed = Volume.prepare_surface(flipped)
        >>> outward = np.einsum('ij,ij->i', fixed.cell_data['Normals'], fixed.cell_centers().points)
        >>> bool((outward > 0).all())
        True

        Quads are split into triangles:

        >>> fixed = Volume.prepare_surface(pv.Box())
        >>> fixed.is_all_triangles
        True
        """
        surface = surface.triangulate().clean()
        surface = surface.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True,
                                          consistent_normals=True)
        return surface

    def _build_sizing_field(self, region_edge_lengths, background_edge_length):
        """
        Build a background tetrahedral grid holding a target edge length.

        The grid covers all surfaces plus a margin. Each grid point takes the
        edge length of the smallest surface that contains it; points outside
        every surface, or inside a surface whose entry is None, take
        `background_edge_length`. The grid spacing is the finest edge length
        but is capped at ~200k points, since it only has to resolve where
        the regions are.

        Parameters
        ----------
        region_edge_lengths : list[float | None]
            Target edge length in mm for each surface, same order as
            `surface_meshes`.

        background_edge_length : float
            Edge length in mm used where no region value applies.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Background points (n, 3), tetrahedra (m, 4), and target edge
            length per point (n,).

        Examples
        --------
        Inner sphere at 1.5 mm, everything else at 4 mm:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)])
        >>> points, tets, values = volume._build_sizing_field([None, 1.5], 4.0)
        >>> tets.shape[1]
        4
        >>> np.unique(values).tolist()
        [1.5, 4.0]

        One entry is needed per surface:

        >>> volume._build_sizing_field([1.5], 4.0)
        Traceback (most recent call last):
            ...
        ValueError: region_edge_lengths needs one entry per surface mesh (None to use the default).
        """
        if len(region_edge_lengths) != len(self.surface_meshes):
            raise ValueError('region_edge_lengths needs one entry per surface mesh (None to use the default).')

        sizes = [size for size in region_edge_lengths if size is not None]
        bounds = np.asarray(pv.merge(self.surface_meshes).bounds).reshape(3, 2)
        pad = 2 * background_edge_length
        lower = bounds[:, 0] - pad
        extent = bounds[:, 1] + pad - lower

        max_points = 200000
        spacing = max(min(sizes + [background_edge_length]), (np.prod(extent) / max_points) ** (1 / 3))
        dims = np.ceil(extent / spacing).astype(int) + 1
        grid = pv.ImageData(dimensions=dims, spacing=[spacing] * 3, origin=lower).triangulate()

        region = self._label_points(grid.points)
        values = np.full(grid.n_points, background_edge_length, dtype=np.float64)
        for index, size in enumerate(region_edge_lengths):
            if size is not None:
                values[region == index] = size

        tets = grid.cells_dict[pv.CellType.TETRA].astype(np.int32)

        return np.asarray(grid.points, dtype=np.float64), tets, values

    def _label_points(self, points):
        """
        Label points with the index of the smallest surface containing them.

        Smallest is by enclosed volume, so a point inside a tumor that is
        inside the liver is labeled with the tumor, and a point in the
        overlap of two separate surfaces goes to the smaller one.

        Parameters
        ----------
        points : np.ndarray
            (n, 3) query points, in mm.

        Returns
        -------
        np.ndarray
            (n,) int32 labels, -1 for points outside every surface.

        Examples
        --------
        Center, outer shell, and outside of two nested spheres:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)])
        >>> points = np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        >>> volume._label_points(points).tolist()
        [1, 0, -1]
        """
        cloud = pv.PolyData(np.asarray(points, dtype=np.float64))
        labels = np.full(cloud.n_points, -1, dtype=np.int32)
        best_volume = np.full(cloud.n_points, np.inf)
        for index, surface in enumerate(self.surface_meshes):
            inside = cloud.select_enclosed_points(surface, check_surface=False)['SelectedPoints'].astype(bool)
            surface_volume = abs(surface.volume)
            take = inside & (surface_volume < best_volume)
            labels[take] = index
            best_volume[take] = surface_volume

        return labels

    def create(self, edge_length_fac=0.05, edge_length_abs=None, region_edge_lengths=None, optimize=True,
               simplify=True, epsilon=1e-3, stop_energy=10.0, coarsen=False, num_threads=0, num_opt_iter=80,
               loglevel=6, quiet=False, disable_filtering=False, sizing_field=None, sizing_field_scalars=None,
               bg_vertices=None, bg_tets=None, bg_values=None):
        """
        Generate one conforming tetrahedral mesh from all surface meshes.

        All surfaces are merged and run through PyTetWild together with VTK
        node ordering, so every tetrahedron has positive volume. The result
        is stored on `self.mesh` with each cell labeled in 'Region'.

        Sizing priority: `region_edge_lengths`, then `sizing_field`, then
        `bg_*` arrays, then a uniform `edge_length_abs` / `edge_length_fac`.

        Parameters
        ----------
        edge_length_fac : float
            Target edge length as a fraction of the combined bounding-box
            diagonal. Ignored when `edge_length_abs` is given.

        edge_length_abs : float, optional
            Target edge length in mm.

        region_edge_lengths : list[float | None], optional
            Target edge length in mm per surface, same order as
            `surface_meshes`. None entries, and space outside all surfaces,
            use `edge_length_abs` (or `edge_length_fac`). Requires
            `optimize=True`.

        optimize : bool
            Improve element quality (minimum scaled Jacobian). Controlled by
            `stop_energy` and `num_opt_iter`.

        simplify : bool
            Simplify the input surfaces before tetrahedralization.

        epsilon : float
            Envelope size: max distance of output surface from input,
            relative to the bounding-box size.

        stop_energy : float
            Optimization stops when the conformal AMIPS energy reaches this.

        coarsen : bool
            Coarsen the output as much as possible while keeping quality.

        num_threads : int
            Threads used; 0 uses all cores.

        num_opt_iter : int
            Maximum optimization iterations when `optimize=True`.

        loglevel : int
            0 = most verbose, 6 = minimal.

        quiet : bool
            Disable all output. Overrides `loglevel`.

        disable_filtering : bool
            Keep all tetrahedra of the bounding box instead of only those
            inside the surfaces. Outside cells get Region -1.

        sizing_field : pv.UnstructuredGrid, optional
            All-tetrahedral grid with a point array of target edge lengths
            in mm. Requires `optimize=True`.

        sizing_field_scalars : str, optional
            Name of the point array on `sizing_field`; defaults to its
            active point scalars.

        bg_vertices, bg_tets, bg_values : np.ndarray, optional
            The same sizing field as raw arrays: points (n, 3), tetrahedra
            (m, 4), and target edge length per point (n,).

        Returns
        -------
        None
            The result is stored on `self.mesh`.

        Examples
        --------
        Uniform mesh of a 10 mm radius sphere:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> volume.create(edge_length_fac=0.1, quiet=True)
        >>> sizes = volume.mesh.compute_cell_sizes(length=False, area=False)
        >>> bool((sizes.cell_data['Volume'] > 0).all())
        True

        A smaller `edge_length_fac` gives more tetrahedra:

        >>> coarse = volume.mesh.n_cells
        >>> volume.create(edge_length_fac=0.05, quiet=True)
        >>> volume.mesh.n_cells > coarse
        True

        Nested spheres, 1.5 mm inside the inner sphere and 4 mm elsewhere;
        the inner region has the smaller tetrahedra:

        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)])
        >>> volume.create(region_edge_lengths=[None, 1.5], edge_length_abs=4.0, quiet=True)
        >>> sizes = volume.mesh.compute_cell_sizes(length=False, area=False).cell_data['Volume']
        >>> region = volume.mesh.cell_data['Region']
        >>> bool(sizes[region == 1].mean() < sizes[region == 0].mean())
        True

        A custom sizing field, finer toward +x:

        >>> field = pv.ImageData(dimensions=(12, 12, 12), spacing=(5.0, 5.0, 5.0),
        ...                      origin=(-27.5, -27.5, -27.5)).triangulate()
        >>> field.point_data['size'] = np.interp(field.points[:, 0], [-25.0, 25.0], [5.0, 1.5])
        >>> volume.create(sizing_field=field, sizing_field_scalars='size', quiet=True)
        >>> volume.mesh.n_cells > 0
        True

        A sizing field needs optimization on:

        >>> volume.create(region_edge_lengths=[None, 1.5], optimize=False, quiet=True)
        Traceback (most recent call last):
            ...
        ValueError: A sizing field requires optimize=True.
        """
        merged = pv.merge(self.surface_meshes)
        vertices = np.asarray(merged.points, dtype=np.float64)
        faces = merged.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

        if region_edge_lengths is not None:
            if edge_length_abs is not None:
                background_edge_length = edge_length_abs
            else:
                diagonal = np.linalg.norm(np.ptp(vertices, axis=0))
                background_edge_length = edge_length_fac * diagonal
            bg_vertices, bg_tets, bg_values = self._build_sizing_field(region_edge_lengths, background_edge_length)

        elif sizing_field is not None:
            if sizing_field_scalars is None:
                sizing_values = sizing_field.active_scalars
            else:
                sizing_values = sizing_field.point_data[sizing_field_scalars]
            bg_vertices = np.asarray(sizing_field.points, dtype=np.float64)
            bg_tets = sizing_field.cells_dict[pv.CellType.TETRA].astype(np.int32)
            bg_values = np.asarray(sizing_values, dtype=np.float64)

        if bg_vertices is not None and not optimize:
            raise ValueError('A sizing field requires optimize=True.')

        points, tets = pytetwild.tetrahedralize(vertices, faces, edge_length_fac=edge_length_fac,
                                                edge_length_abs=edge_length_abs, optimize=optimize,
                                                simplify=simplify, epsilon=epsilon, stop_energy=stop_energy,
                                                coarsen=coarsen, num_threads=num_threads,
                                                num_opt_iter=num_opt_iter, loglevel=loglevel, quiet=quiet,
                                                vtk_ordering=True, disable_filtering=disable_filtering,
                                                bg_vertices=bg_vertices, bg_tets=bg_tets, bg_values=bg_values)

        self.mesh = pv.UnstructuredGrid({pv.CellType.TETRA: np.asarray(tets)}, np.asarray(points))
        self.mesh.cell_data['Region'] = self._label_points(self.mesh.cell_centers().points)

    def extract_region(self, index):
        """
        Return the tetrahedra labeled with one region.

        `create` must be called first.

        Parameters
        ----------
        index : int
            Surface index in `surface_meshes`, or -1 for unlabeled cells.

        Returns
        -------
        pv.UnstructuredGrid
            Tetrahedra of that region, with 'Region' carried over. Empty if
            no cell has that label.

        Examples
        --------
        Split nested spheres into the outer shell and the inner core:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)])
        >>> volume.create(edge_length_abs=3.0, quiet=True)
        >>> shell = volume.extract_region(0)
        >>> core = volume.extract_region(1)
        >>> shell.n_cells + core.n_cells == volume.mesh.n_cells
        True

        The core's bounds match the 8 mm inner sphere:

        >>> round(core.bounds[1])
        8
        """
        return self.mesh.extract_cells(np.flatnonzero(self.mesh.cell_data['Region'] == index))

    def write(self, path, binary=False):
        """
        Save the tetrahedral mesh to a file in ASCII.

        `create` must be called first. The 'Region' array is saved with it.

        Parameters
        ----------
        path : str
            Output path. Any PyVista-supported extension (.vtu, .vtk, ...).
        binary : bool
            Sets if the output is binary or not during save.

        Returns
        -------
        None

        Examples
        --------
        Write and read back a sphere:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> volume.create(edge_length_fac=0.1, quiet=True)
        >>> volume.write('sphere_tet.vtu')
        >>> reloaded = pv.read('sphere_tet.vtu')
        >>> reloaded.n_cells == volume.mesh.n_cells
        True
        >>> 'Region' in reloaded.cell_data
        True
        """
        self.mesh.save(path, binary=binary)
