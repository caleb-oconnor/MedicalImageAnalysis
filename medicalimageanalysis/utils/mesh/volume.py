"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Tetrahedral Volume Meshing
==========================

Description:
    This module turns one or more closed surface meshes into a single
    conforming tetrahedral mesh with PyTetWild (>= 0.4), ready for finite
    element work (FEBio heat / biomechanics) or any volume-based measurement.
    Surfaces may be nested (e.g. tumor inside liver), separate, overlapping,
    or marked as cavities (e.g. a vessel lumen) that are left empty. Every
    interior surface is embedded in the mesh, so no tetrahedron crosses it,
    and each tetrahedron is labeled with the surface that most tightly
    encloses it.

Supported Inputs:
    * Surfaces: closed, manifold PyVista PolyData meshes, in mm. A single
      PolyData or a list of them.
    * Sizing fields: all-tetrahedral PyVista UnstructuredGrids with a point
      array of target edge lengths in mm, or the same data as raw arrays.

Key Functionalities:
    1. **Tetrahedralization**: `Volume.create` merges all surfaces, runs
       PyTetWild with VTK node ordering, and stores a VTK_TETRA
       UnstructuredGrid on `mesh`.
    2. **Region Labels**: `mesh.cell_data['Region']` holds, for every
       tetrahedron, the index of the smallest input surface that contains
       it (-1 if none, only possible with `disable_filtering=True`). Region
       names are saved in `mesh.field_data['RegionNames']`.
    3. **Cavities**: surfaces listed in `cavities` are meshed as holes.
    4. **Per-Region Sizing**: `region_edge_lengths` gives each surface its
       own target edge length in mm. A custom `sizing_field` grid or raw
       `bg_*` arrays can be used instead.
    5. **FEBio Sets**: `Volume.extract_boundary` returns a region's outer
       surface, or only the faces it shares with one neighbor, oriented
       outward and carrying global node and element ids.
       `Volume.region_node_ids` returns a region's node set.
    6. **Quality**: `Volume.compute_quality` stores a per-cell quality
       measure (scaled Jacobian by default) on the mesh.
    7. **Export**: `Volume.write` saves the grid in ASCII to any format
       PyVista supports (.vtu, .vtk, ...).

Sizing Note:
    `edge_length_fac` and `epsilon` are fractions of the bounding-box
    diagonal of the LARGEST single surface, not of all surfaces combined, so
    adding a distant surface does not change the mesh of the others. With a
    100 mm diagonal, edge_length_fac=0.02 targets ~2 mm edges; halving it
    roughly multiplies the tetrahedron count by 8. `edge_length_abs`,
    `region_edge_lengths` and sizing field values are in mm and are targets:
    elements next to a finer region come out smaller.

Orientation Note:
    Every surface is oriented with outward normals before meshing, and
    cavities with inward normals. PyTetWild keeps tetrahedra by winding
    number, so the orientation decides whether the inside of a surface is
    meshed or carved out.

Region Index Note:
    Methods that take a region accept its index in `surface_meshes` or its
    name. For neighbors, -1 means the outside of the mesh; a cavity's index
    means the cavity wall.

Usage:
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.volume import Volume
    >>> # Liver-like sphere with a finer tumor and an empty vessel
    >>> liver = pv.Sphere(radius=40.0)
    >>> tumor = pv.Sphere(radius=8.0, center=(10.0, 0.0, 0.0))
    >>> vessel = pv.Cylinder(radius=3.0, height=30.0, center=(-15.0, 0.0, 0.0))
    >>> volume = Volume([liver, tumor, vessel], names=['Liver', 'GTV', 'Vessel'], cavities=['Vessel'])
    >>> volume.create(edge_length_abs=5.0, region_edge_lengths=[None, 1.5, 1.5], quiet=True)
    >>> tumor_surface = volume.extract_boundary('GTV')
    >>> volume.write('C:/Data/liver_tet.vtu')

Examples
--------
>>> import numpy as np
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.mesh.volume import Volume
>>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)], names=['Outer', 'Inner'])
>>> volume.create(edge_length_abs=4.0, region_edge_lengths=[None, 1.5], quiet=True)
>>> set(volume.mesh.celltypes) == {pv.CellType.TETRA}
True
>>> sorted(np.unique(volume.mesh.cell_data['Region']).tolist())
[0, 1]
>>> interface = volume.extract_boundary('Inner', neighbor='Outer')
>>> bool(abs(np.linalg.norm(interface.points, axis=1).mean() - 8.0) < 0.2)
True
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
        Source surfaces, triangulated, cleaned, and oriented (outward, or
        inward for cavities), in mm.

    names : list[str]
        One name per surface; 'Region_0', 'Region_1', ... if not given.

    cavities : list[int]
        Indices of the surfaces meshed as holes.

    mesh : pv.UnstructuredGrid | None
        Tetrahedral mesh built by `create`, with cell array 'Region' and
        field array 'RegionNames'.

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

    Liver with a tumor, finer inside the tumor, and the tumor surface and
    nodes for FEBio:

    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image[mia.Data.image_list()[0]]
    >>> volume = Volume([image.roi['Liver'].mesh, image.roi['GTV'].mesh], names=['Liver', 'GTV'])
    >>> volume.create(edge_length_abs=3.0, region_edge_lengths=[None, 1.0])
    >>> gtv_surface = volume.extract_boundary('GTV', neighbor='Liver')
    >>> gtv_nodes = volume.region_node_ids('GTV')
    >>> volume.write('C:/Data/liver_gtv_tet.vtu')
    """
    def __init__(self, surface_meshes, names=None, cavities=None):
        """
        Prepare and store the source surface meshes before tetrahedralization.

        Parameters
        ----------
        surface_meshes : pv.PolyData | list[pv.PolyData]
            One closed surface, or a list of them, in mm. List order sets the
            region labels.

        names : list[str], optional
            One name per surface. Defaults to 'Region_0', 'Region_1', ...

        cavities : list[int | str], optional
            Surfaces (by index or name) to leave empty, e.g. a vessel lumen.
            A cavity should sit inside another surface; one that does not
            simply produces no tetrahedra.

        Examples
        --------
        A single surface is stored as a one-item list:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> len(volume.surface_meshes)
        1
        >>> volume.names
        ['Region_0']

        Several named surfaces, the second one a cavity:

        >>> outer = pv.Sphere(radius=20.0)
        >>> lumen = pv.Sphere(radius=8.0, theta_resolution=10, phi_resolution=10)
        >>> volume = Volume([outer, lumen], names=['Tissue', 'Lumen'], cavities=['Lumen'])
        >>> [surface.n_points for surface in volume.surface_meshes]
        [842, 82]
        >>> volume.cavities
        [1]
        """
        if isinstance(surface_meshes, pv.PolyData):
            surface_meshes = [surface_meshes]

        if names is None:
            names = ['Region_' + str(index) for index in range(len(surface_meshes))]
        if len(names) != len(surface_meshes):
            raise ValueError('names needs one entry per surface mesh.')
        self.names = list(names)

        self.cavities = []
        if cavities is not None:
            self.cavities = sorted(self._region_index(cavity) for cavity in cavities)

        self.surface_meshes = [self.prepare_surface(surface, inward=index in self.cavities)
                               for index, surface in enumerate(surface_meshes)]
        self.mesh = None

    @staticmethod
    def prepare_surface(surface, inward=False):
        """
        Triangulate, clean, and orient a surface.

        Parameters
        ----------
        surface : pv.PolyData
            Closed surface, in mm.

        inward : bool
            False orients normals outward (meshed region). True orients them
            inward (cavity).

        Returns
        -------
        pv.PolyData
            Triangle surface with consistently oriented cell normals in
            'Normals'.

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

        A cavity faces inward:

        >>> cavity = Volume.prepare_surface(pv.Sphere(radius=10.0), inward=True)
        >>> inward = np.einsum('ij,ij->i', cavity.cell_data['Normals'], cavity.cell_centers().points)
        >>> bool((inward < 0).all())
        True

        Quads are split into triangles:

        >>> Volume.prepare_surface(pv.Box()).is_all_triangles
        True
        """
        surface = surface.triangulate().clean()
        surface = surface.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True,
                                          consistent_normals=True)
        if inward:
            surface = surface.flip_faces()
            surface.cell_data['Normals'] = -surface.cell_data['Normals']
        return surface

    def _build_sizing_field(self, region_edge_lengths, background_edge_length, epsilon, num_threads):
        """
        Build a background tetrahedral mesh holding a target edge length.

        The background is a fast, unoptimized PyTetWild run over the whole
        bounding box with filtering off, so it has nodes on every input
        surface no matter how small that surface is. Each node takes the
        edge length of the smallest surface that contains it; nodes on a
        surface also take that surface's edge length if it is finer, so
        every region is sized right up to its wall and the transition
        happens on the coarser side. Nodes outside every surface, or inside
        a surface whose entry is None, take `background_edge_length`.

        Parameters
        ----------
        region_edge_lengths : list[float | None]
            Target edge length in mm for each surface, same order as
            `surface_meshes`.

        background_edge_length : float
            Edge length in mm used where no region value applies, and the
            edge length of the background mesh itself.

        epsilon : float
            PyTetWild envelope, relative to the merged bounding-box diagonal.

        num_threads : int
            Threads used; 0 uses all cores.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Background points (n, 3), tetrahedra (m, 4), and target edge
            length per point (n,).

        Examples
        --------
        A 4 mm tumor inside a 120 mm organ still gets sampled:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> organ = pv.Sphere(radius=120.0, theta_resolution=60, phi_resolution=60)
        >>> tumor = pv.Sphere(radius=4.0, center=(30.0, 0.0, 0.0))
        >>> volume = Volume([organ, tumor])
        >>> points, tets, values = volume._build_sizing_field([None, 0.8], 12.0, 1e-3, 0)
        >>> tets.shape[1]
        4
        >>> bool((values == 0.8).sum() > 50)
        True
        >>> sorted(np.unique(values).tolist())
        [0.8, 12.0]

        One entry is needed per surface:

        >>> volume._build_sizing_field([0.8], 12.0, 1e-3, 0)
        Traceback (most recent call last):
            ...
        ValueError: region_edge_lengths needs one entry per surface mesh (None to use the default).
        """
        if len(region_edge_lengths) != len(self.surface_meshes):
            raise ValueError('region_edge_lengths needs one entry per surface mesh (None to use the default).')

        merged = pv.merge(self.surface_meshes)
        points, tets = pytetwild.tetrahedralize(np.asarray(merged.points, dtype=np.float64),
                                                merged.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
                                                edge_length_abs=background_edge_length, optimize=False,
                                                epsilon=epsilon, num_threads=num_threads, quiet=True,
                                                vtk_ordering=True, disable_filtering=True)
        points = np.asarray(points, dtype=np.float64)

        labels = self._label_points(points)
        values = np.full(len(points), background_edge_length, dtype=np.float64)
        for index, size in enumerate(region_edge_lengths):
            if size is not None:
                values[labels == index] = size

        cloud = pv.PolyData(points)
        tolerance = 2 * epsilon * merged.length
        for index, size in enumerate(region_edge_lengths):
            if size is not None:
                distance = np.abs(cloud.compute_implicit_distance(self.surface_meshes[index])['implicit_distance'])
                on_surface = distance <= tolerance
                values[on_surface] = np.minimum(values[on_surface], size)

        return points, np.asarray(tets, dtype=np.int32), values

    def _face_table(self):
        """
        List every tetrahedron face that lies on a region boundary.

        A face is kept when the tetrahedron on its other side belongs to a
        different region, or when there is no tetrahedron on its other side.
        Faces are ordered so their normal points out of their own
        tetrahedron. Faces with no tetrahedron on the other side get
        neighbor -1, or the index of a cavity when they are closer to that
        cavity's surface than to any other surface.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Faces (f, 3) as global node ids, owning cell id (f,), and
            neighbor region (f,).

        Examples
        --------
        A single sphere only has outside faces:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> volume.create(edge_length_fac=0.1, quiet=True)
        >>> faces, cells, neighbors = volume._face_table()
        >>> faces.shape[1]
        3
        >>> np.unique(neighbors).tolist()
        [-1]
        """
        tets = self.mesh.cells_dict[pv.CellType.TETRA]
        regions = self.mesh.cell_data['Region']

        # Local faces ordered so the normal points away from the opposite vertex.
        local = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        faces = tets[:, local].reshape(-1, 3)
        cells = np.repeat(np.arange(len(tets)), 4)

        keys = np.sort(faces, axis=1)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        same = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
        partner = np.full(len(faces), -1)
        partner[order[:-1][same]] = order[1:][same]
        partner[order[1:][same]] = order[:-1][same]

        neighbors = np.full(len(faces), -1, dtype=np.int32)
        shared = partner >= 0
        neighbors[shared] = regions[cells[partner[shared]]]

        keep = ~shared | (neighbors != regions[cells])
        faces, cells, neighbors = faces[keep], cells[keep], neighbors[keep]

        outside = neighbors == -1
        if self.cavities and outside.any():
            centers = pv.PolyData(self.mesh.points[faces[outside]].mean(axis=1))
            distances = np.column_stack([np.abs(centers.compute_implicit_distance(surface)['implicit_distance'])
                                         for surface in self.surface_meshes])
            nearest = distances.argmin(axis=1)
            outside_neighbors = np.where(np.isin(nearest, self.cavities), nearest, -1)
            neighbors[np.flatnonzero(outside)] = outside_neighbors

        return faces, cells, neighbors

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

    def _region_index(self, region):
        """
        Convert a region name or index to an index.

        Parameters
        ----------
        region : int | str
            Index in `surface_meshes`, -1, or a name in `names`.

        Returns
        -------
        int

        Examples
        --------
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)], names=['Liver', 'GTV'])
        >>> volume._region_index('GTV'), volume._region_index(0), volume._region_index(-1)
        (1, 0, -1)
        """
        if isinstance(region, str):
            return self.names.index(region)
        return int(region)

    def compute_quality(self, quality_measure='scaled_jacobian'):
        """
        Compute a per-cell quality measure and store it on the mesh.

        The values are stored in `mesh.cell_data['Quality']`. For the scaled
        Jacobian, 1 is a regular tetrahedron, values near 0 are slivers, and
        negative values are inverted elements that FEBio will reject.

        Parameters
        ----------
        quality_measure : str
            Any measure accepted by `pv.UnstructuredGrid.cell_quality`, e.g.
            'scaled_jacobian', 'aspect_ratio', 'min_angle', 'radius_ratio'.

        Returns
        -------
        np.ndarray
            (n_cells,) quality values.

        Examples
        --------
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume(pv.Sphere(radius=10.0))
        >>> volume.create(edge_length_fac=0.1, quiet=True)
        >>> quality = volume.compute_quality()
        >>> bool(quality.min() > 0)
        True
        >>> 'Quality' in volume.mesh.cell_data
        True

        Count slivers per region:

        >>> int((quality[volume.mesh.cell_data['Region'] == 0] < 0.1).sum()) >= 0
        True
        """
        quality = np.asarray(self.mesh.cell_quality(quality_measure).cell_data[quality_measure])
        self.mesh.cell_data['Quality'] = quality
        return quality

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
            Target edge length as a fraction of the bounding-box diagonal of
            the largest single surface. Ignored when `edge_length_abs` is
            given.

        edge_length_abs : float, optional
            Target edge length in mm.

        region_edge_lengths : list[float | None], optional
            Target edge length in mm per surface, same order as
            `surface_meshes`. None entries, and space outside all surfaces,
            use `edge_length_abs` (or `edge_length_fac`). A cavity's entry
            sizes the elements on its wall. Requires `optimize=True`.

        optimize : bool
            Improve element quality (minimum scaled Jacobian). Controlled by
            `stop_energy` and `num_opt_iter`.

        simplify : bool
            Simplify the input surfaces before tetrahedralization.

        epsilon : float
            Envelope size: max distance of the output surfaces from the
            input, relative to the bounding-box diagonal of the largest
            single surface.

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
            in mm, covering all surfaces. Requires `optimize=True`.

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

        A distant second surface does not change the first one's edge
        length (within ~10%, from PyTetWild's own tolerances):

        >>> def mean_edge(grid):
        ...     return grid.extract_all_edges().compute_cell_sizes()['Length'].mean()
        >>> pair = Volume([pv.Sphere(radius=10.0), pv.Sphere(radius=10.0, center=(200.0, 0.0, 0.0))])
        >>> pair.create(edge_length_fac=0.05, quiet=True)
        >>> bool(abs(mean_edge(pair.extract_region(0)) / mean_edge(volume.mesh) - 1) < 0.15)
        True

        Nested spheres, 1.5 mm inside the inner sphere and 4 mm elsewhere:

        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)])
        >>> volume.create(edge_length_abs=4.0, region_edge_lengths=[None, 1.5], quiet=True)
        >>> sizes = volume.mesh.compute_cell_sizes(length=False, area=False).cell_data['Volume']
        >>> region = volume.mesh.cell_data['Region']
        >>> bool(sizes[region == 1].mean() < sizes[region == 0].mean())
        True

        An inner cavity is left empty:

        >>> hollow = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)], cavities=[1])
        >>> hollow.create(edge_length_abs=3.0, quiet=True)
        >>> centers = np.linalg.norm(hollow.mesh.cell_centers().points, axis=1)
        >>> int((centers < 7.5).sum())
        0

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

        # Size and envelope relative to the largest single surface, not the merged box.
        surface_diagonal = max(surface.length for surface in self.surface_meshes)
        if edge_length_abs is None:
            edge_length_abs = edge_length_fac * surface_diagonal
        merged_epsilon = epsilon * surface_diagonal / merged.length

        if region_edge_lengths is not None:
            if not optimize:
                raise ValueError('A sizing field requires optimize=True.')
            bg_vertices, bg_tets, bg_values = self._build_sizing_field(region_edge_lengths, edge_length_abs,
                                                                       merged_epsilon, num_threads)

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

        points, tets = pytetwild.tetrahedralize(vertices, faces, edge_length_abs=edge_length_abs,
                                                optimize=optimize, simplify=simplify, epsilon=merged_epsilon,
                                                stop_energy=stop_energy, coarsen=coarsen, num_threads=num_threads,
                                                num_opt_iter=num_opt_iter, loglevel=loglevel, quiet=quiet,
                                                vtk_ordering=True, disable_filtering=disable_filtering,
                                                bg_vertices=bg_vertices, bg_tets=bg_tets, bg_values=bg_values)

        self.mesh = pv.UnstructuredGrid({pv.CellType.TETRA: np.asarray(tets)}, np.asarray(points))
        self.mesh.cell_data['Region'] = self._label_points(self.mesh.cell_centers().points)
        self.mesh.field_data['RegionNames'] = np.array(self.names)

    def extract_boundary(self, region, neighbor=None):
        """
        Extract a region's boundary faces as a surface, e.g. for an FEBio
        surface, contact pair, or load.

        Faces are oriented with normals pointing out of `region`. Global ids
        refer to `mesh`, so they can be written directly into FEBio sets.

        Parameters
        ----------
        region : int | str
            Region whose boundary is wanted.

        neighbor : int | str, optional
            None returns the full boundary of `region`. A region returns only
            the faces shared with it. -1 returns only the faces on the
            outside of the mesh. A cavity returns only the cavity wall.

        Returns
        -------
        pv.PolyData
            Triangle surface with point array 'GlobalNodeIds', and cell
            arrays 'GlobalElementIds' (the tetrahedron each face belongs to)
            and 'Neighbor' (region on the other side).

        Examples
        --------
        Interface between a tumor and the surrounding tissue:

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)], names=['Tissue', 'Tumor'])
        >>> volume.create(edge_length_abs=3.0, quiet=True)
        >>> interface = volume.extract_boundary('Tumor', neighbor='Tissue')
        >>> bool(abs(np.linalg.norm(interface.points, axis=1).mean() - 8.0) < 0.2)
        True

        Normals point out of the region asked for:

        >>> normals = interface.compute_normals(cell_normals=True, point_normals=False,
        ...                                     auto_orient_normals=False).cell_data['Normals']
        >>> bool((np.einsum('ij,ij->i', normals, interface.cell_centers().points) > 0).all())
        True

        The tissue's full boundary is its outer skin plus the tumor wall:

        >>> boundary = volume.extract_boundary('Tissue')
        >>> sorted(np.unique(boundary.cell_data['Neighbor']).tolist())
        [-1, 1]

        Global node ids index straight into the volume mesh:

        >>> ids = interface.point_data['GlobalNodeIds']
        >>> bool(np.allclose(volume.mesh.points[ids], interface.points))
        True

        A cavity wall, e.g. for a pressure load:

        >>> hollow = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)], names=['Wall', 'Lumen'],
        ...                 cavities=['Lumen'])
        >>> hollow.create(edge_length_abs=3.0, quiet=True)
        >>> lumen_wall = hollow.extract_boundary('Wall', neighbor='Lumen')
        >>> bool(abs(np.linalg.norm(lumen_wall.points, axis=1).mean() - 8.0) < 0.2)
        True
        """
        faces, cells, neighbors = self._face_table()
        keep = self.mesh.cell_data['Region'][cells] == self._region_index(region)
        if neighbor is not None:
            keep &= neighbors == self._region_index(neighbor)
        faces, cells, neighbors = faces[keep], cells[keep], neighbors[keep]

        node_ids, local_faces = np.unique(faces, return_inverse=True)
        surface = pv.PolyData.from_regular_faces(self.mesh.points[node_ids], local_faces.reshape(-1, 3))
        surface.point_data['GlobalNodeIds'] = node_ids
        surface.cell_data['GlobalElementIds'] = cells
        surface.cell_data['Neighbor'] = neighbors
        return surface

    def extract_region(self, region):
        """
        Return the tetrahedra labeled with one region.

        Parameters
        ----------
        region : int | str
            Region index or name, or -1 for unlabeled cells.

        Returns
        -------
        pv.UnstructuredGrid
            Tetrahedra of that region, with 'Region' carried over and
            'vtkOriginalPointIds' / 'vtkOriginalCellIds' pointing back into
            `mesh`. Empty if no cell has that label.

        Examples
        --------
        Split nested spheres into the outer shell and the inner core:

        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)], names=['Shell', 'Core'])
        >>> volume.create(edge_length_abs=3.0, quiet=True)
        >>> shell = volume.extract_region('Shell')
        >>> core = volume.extract_region(1)
        >>> shell.n_cells + core.n_cells == volume.mesh.n_cells
        True
        >>> round(core.bounds[1])
        8
        """
        return self.mesh.extract_cells(np.flatnonzero(self.mesh.cell_data['Region'] == self._region_index(region)))

    def region_node_ids(self, region):
        """
        Return the global node ids used by a region, e.g. for an FEBio node
        set or to assign initial temperature.

        Nodes on an interface belong to both regions.

        Parameters
        ----------
        region : int | str
            Region index or name.

        Returns
        -------
        np.ndarray
            Sorted unique node ids into `mesh.points`.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> from medicalimageanalysis.utils.mesh.volume import Volume
        >>> volume = Volume([pv.Sphere(radius=20.0), pv.Sphere(radius=8.0)], names=['Tissue', 'Tumor'])
        >>> volume.create(edge_length_abs=3.0, quiet=True)
        >>> tumor_nodes = volume.region_node_ids('Tumor')
        >>> bool(np.linalg.norm(volume.mesh.points[tumor_nodes], axis=1).max() < 8.5)
        True

        The interface nodes are the overlap of the two sets:

        >>> shared = np.intersect1d(tumor_nodes, volume.region_node_ids('Tissue'))
        >>> interface = volume.extract_boundary('Tumor', neighbor='Tissue')
        >>> bool(np.array_equal(shared, np.sort(interface.point_data['GlobalNodeIds'])))
        True
        """
        tets = self.mesh.cells_dict[pv.CellType.TETRA]
        return np.unique(tets[self.mesh.cell_data['Region'] == self._region_index(region)])

    def write(self, path):
        """
        Save the tetrahedral mesh to a file in ASCII.

        `create` must be called first. 'Region', 'RegionNames', and
        'Quality' (if computed) are saved with it.

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
        >>> volume = Volume(pv.Sphere(radius=10.0), names=['Ball'])
        >>> volume.create(edge_length_fac=0.1, quiet=True)
        >>> volume.write('sphere_tet.vtu')
        >>> reloaded = pv.read('sphere_tet.vtu')
        >>> reloaded.n_cells == volume.mesh.n_cells
        True
        >>> reloaded.field_data['RegionNames'].tolist()
        ['Ball']
        """
        self.mesh.save(path, binary=False)