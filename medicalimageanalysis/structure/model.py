"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Models: meshes derived from an image's ROIs for one computation, such as a clustered surface for CPD / NRICP
    surface matching, or a tetrahedral mesh for FEBio (biomechanical registration, ablation heat). A model never
    replaces ``roi.mesh``, which stays the full-resolution anatomy used for display, masks and volumes. One ROI can
    have any number of models (one per recipe), and one model can come from several ROIs (liver + tumor regions, a
    vessel cavity).

    Conventions
    -----------
    * A model belongs to one image (``image.model``), like markups and planes: never synced to other images. Its
      ``id`` is unique across every image (``Data.new_model_id``); its ``name`` is unique on its image only.
    * ``roi_ids`` are the source ROIs, in region order: region i of the mesh is ``roi_ids[i]``. Every region label
      (``mesh.cell_data['Region']``) is an index into ``roi_ids``; -1 = no region. Ids, not names, so renaming an ROI
      never breaks a model. ``cavity_ids`` are source ROIs meshed as holes.
    * The recipe (``cluster_points``, ``edge_length_abs``, ``region_edge_lengths``, ...) is kept as plain attributes,
      so a model can be rebuilt or compared with another one. :meth:`Model.build` runs it: cluster each ROI mesh
      (pyacvd, ``roi.create_cluster_mesh``), then tetrahedralize them together (pytetwild through
      :class:`~medicalimageanalysis.utils.mesh.volume.Volume`).
    * ``kind``: 'tet' (``mesh`` is a VTK_TETRA UnstructuredGrid) or 'surface' (``mesh`` is the clustered surfaces
      merged into one PolyData, with 'Region' on points and cells). ``surfaces`` always holds the clustered surface of
      each ROI, in region order.
    * Staleness: ``source_keys`` holds a hash of every source ROI mesh when the model was built. If an ROI is edited
      afterwards the model is stale (:meth:`Model.is_stale`) but is never rebuilt or dropped automatically: results
      computed on it only make sense on that mesh.
    * ``build_count`` goes up by one every time the mesh is replaced (:meth:`Model.build`, :meth:`Model.update_mesh`)
      and is saved. The id stays the same through a rebuild, so anything that used the model records the build_count
      it saw (``Deformable.set_model``) and can tell when the model has been rebuilt since
      (``Deformable.check_model``). To keep an old mesh next to a new recipe, make a new model instead of rebuilding.
    * ``group_id`` puts the model in a group (``Data.group``, None = ungrouped), like any other item. Set it with
      ``group=`` here or ``Data.set_group('model', ...)``.
    * Results that belong to the mesh (temperature, material maps) go in ``mesh.point_data`` / ``mesh.cell_data`` and
      are saved with it. Results that belong to one registration (nodal displacements, the DVF) stay in that
      Deformable, which refers to the model by id and keeps a frozen copy of the mesh it used.

Structure:
    - Model: source ROIs, recipe, build, display slice, FEBio helpers (boundaries, node sets, quality) and saving.

Examples
--------
>>> import medicalimageanalysis as mia
>>> image = mia.Data.image['CT 01']
>>> fe = image.add_model(['Liver', 'GTV'], name='Liver FE', cluster_points=2500, edge_length_abs=4.0,
...                      region_edge_lengths=[None, 1.5])
>>> fe.build()
>>> fe.mesh.n_points, fe.compute_quality().min()

A second recipe from the same ROI, and a surface-only one for CPD

>>> fine = image.add_model('Liver', name='Liver FE fine', cluster_points=5000, edge_length_abs=2.0)
>>> cpd = image.add_model('Liver', name='Liver CPD', kind='surface', cluster_points=2500)
>>> cpd.build()

FEBio sets

>>> gtv_wall = fe.extract_boundary('GTV', neighbor='Liver')
>>> gtv_nodes = fe.region_node_ids('GTV')

After the Liver contour is edited

>>> fe.is_stale(), fe.get_stale_rois()
(True, [1042])
"""

import io
import os
import json
import hashlib

import numpy as np
import pyvista as pv

from .roi import random_color
from ..data import Data
from ..utils.mesh.volume import Volume


class Model(object):
    """One derived mesh (clustered surface or tetrahedral mesh) built from ROIs of one image.

    Made by the image (:meth:`Image.add_model`) and stored in ``image.model``, which looks it up by permanent id or by
    name (``image.model[7]``, ``image.model['Liver FE']``).

    Parameters
    ----------
    image : Image
        Image whose ROIs the model is built from.
    roi_names : str, int or list of them
        Source ROIs (names or ids), in region order. Kept as ``roi_ids``.
    kind : {'tet', 'surface'}, default 'tet'
        Tetrahedral mesh, or clustered surfaces only.
    name : str, optional
        Replaced by the image's registry when the model is added.
    cavities : str, int or list of them, optional
        Source ROIs meshed as holes (e.g. a vessel lumen). Kept as ``cavity_ids``.
    cluster_points : int or list of (int or None), optional
        Target points of the clustered surface: one value for every ROI, or one per ROI. None (or a None entry) uses
        the ROI mesh as it is.
    edge_length_fac, edge_length_abs, region_edge_lengths, epsilon, optimize, simplify, coarsen, stop_energy,
    num_opt_iter
        pytetwild recipe, passed to :meth:`Volume.create` (see there). ``region_edge_lengths`` is one entry per ROI,
        in region order.
    color : list of int, optional
        RGB, 0-255. Random if None.
    visible : bool, default True
        Show in the views.
    group : str or int, optional
        Group (name or id) to put the model in; a name is created when missing. None = ungrouped.

    Attributes
    ----------
    id : int or None
        Permanent id, written by ``image.model``. None until the model is added.
    group_id : int or None
        Group the model is in (``Data.group``), None = ungrouped.
    surfaces : list of pyvista.PolyData or None
        Clustered surface of each source ROI, in region order. None until built.
    mesh : pyvista.UnstructuredGrid, pyvista.PolyData or None
        The model. None until built.
    source_keys : list of str or None
        Hash of each source ROI mesh when built.
    build_count : int
        Number of times the mesh has been replaced (0 = never built).
    misc : dict
        Free space for the application.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image['CT 01']
    >>> model = mia.Model(image, ['Liver', 'GTV', 'Portal vein'], cavities='Portal vein', cluster_points=[2500, 800,
    ...                   600], edge_length_abs=4.0, region_edge_lengths=[None, 1.5, 1.5])
    >>> image.model.add(model, 'Liver FE vessels')
    >>> model.build()
    """
    KINDS = ('tet', 'surface')
    SCHEMA_VERSION = 1

    def __init__(self, image, roi_names, kind='tet', name=None, cavities=None, cluster_points=None,
                 edge_length_fac=0.05, edge_length_abs=None, region_edge_lengths=None, epsilon=1e-3, optimize=True,
                 simplify=True, coarsen=False, stop_energy=10.0, num_opt_iter=80, color=None, visible=True,
                 group=None):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {list(self.KINDS)}; got {kind!r}")

        self.image = image

        self.id = None
        self.name = name
        self.kind = kind
        self.color = list(random_color()) if color is None else list(color)
        self.visible = visible
        self.group_id = Data.resolve_group(group)

        self.roi_ids = [Data.get_roi_id(roi) for roi in self._as_list(roi_names)]
        self.cavity_ids = [] if cavities is None else [Data.get_roi_id(roi) for roi in self._as_list(cavities)]
        for roi_id in self.cavity_ids:
            if roi_id not in self.roi_ids:
                raise ValueError(f"Cavity ROI {Data.get_roi_name(roi_id)!r} must also be one of roi_names.")

        # recipe
        if cluster_points is None or np.isscalar(cluster_points):
            cluster_points = [cluster_points] * len(self.roi_ids)
        if len(cluster_points) != len(self.roi_ids):
            raise ValueError("cluster_points needs one value, or one entry per ROI.")
        if region_edge_lengths is not None and len(region_edge_lengths) != len(self.roi_ids):
            raise ValueError("region_edge_lengths needs one entry per ROI (None to use the default).")

        self.cluster_points = [None if p is None else int(p) for p in cluster_points]
        self.edge_length_fac = edge_length_fac
        self.edge_length_abs = edge_length_abs
        self.region_edge_lengths = None if region_edge_lengths is None else list(region_edge_lengths)
        self.epsilon = epsilon
        self.optimize = optimize
        self.simplify = simplify
        self.coarsen = coarsen
        self.stop_energy = stop_energy
        self.num_opt_iter = num_opt_iter

        # result
        self.surfaces = None
        self.mesh = None
        self.source_keys = None
        self.build_count = 0

        self._volume_cache = None       # Volume wrapper around mesh, rebuilt when mesh is replaced
        self.misc = {}

    @staticmethod
    def _as_list(names):
        """A single name or id becomes a one-item list, anything else a list."""
        return [names] if isinstance(names, (str, int, np.integer)) else list(names)

    @staticmethod
    def _cell_edges(flat, closed):
        """(edges (M, 2), owner cell (M,)) from a VTK flat cell array [n, id0, ..., n, ...]. closed=True also joins
        each cell's last point to its first (polygons); False for lines."""
        flat = np.asarray(flat)
        edges, owners = [], []
        i, cell = 0, 0
        while i < len(flat):
            n = int(flat[i])
            ids = flat[i + 1:i + 1 + n]
            if closed and n > 2:
                pairs = np.column_stack((ids, np.roll(ids, -1)))
            else:
                pairs = np.column_stack((ids[:-1], ids[1:]))
            edges.append(pairs)
            owners.append(np.full(len(pairs), cell))
            i += n + 1
            cell += 1

        if not edges:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)

        return np.vstack(edges), np.concatenate(owners)

    @staticmethod
    def _mesh_key(mesh):
        """Short hash of a surface's points and faces, to tell whether a source ROI changed since the build."""
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(mesh.points, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())

        return digest.hexdigest()

    @classmethod
    def from_bytes(cls, image, data):
        """Rebuild a model from :meth:`to_bytes` output, not yet added to the image.

        ``id`` and ``name`` hold the saved values; :meth:`Image.add_saved_model` adds it with them (or a new id when the
        saved one is taken). The source ROIs do not have to be loaded. The saved group goes through
        ``Data.live_group_id``, so load the groups first (:meth:`Image.load_markups` before :meth:`Image.load_models`).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> with open('C:/Data/CT 01/model_7.npz', 'rb') as f:
        ...     model = mia.Model.from_bytes(image, f.read())
        """
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            record = json.loads(str(arrays['record']))
            schema = record.get('schema', 1)
            if schema > cls.SCHEMA_VERSION:
                raise ValueError(f"Record schema {schema} is newer than supported ({cls.SCHEMA_VERSION}).")

            model = cls.__new__(cls)
            model.image = image
            model.id = record['id']
            model.name = record['name']
            model.kind = record['kind']
            model.color = record['color']
            model.visible = record['visible']
            model.group_id = Data.live_group_id(record.get('group_id'))
            model.roi_ids = list(record['roi_ids'])
            model.cavity_ids = list(record['cavity_ids'])
            for key in ('cluster_points', 'edge_length_fac', 'edge_length_abs', 'region_edge_lengths', 'epsilon',
                        'optimize', 'simplify', 'coarsen', 'stop_energy', 'num_opt_iter', 'source_keys'):
                setattr(model, key, record[key])
            model.build_count = int(record.get('build_count', 0))
            model._volume_cache = None
            model.misc = {}

            model.surfaces = None
            if record['n_surfaces']:
                model.surfaces = [pv.PolyData(arrays[f'surface_{i}_points'], arrays[f'surface_{i}_faces'])
                                  for i in range(record['n_surfaces'])]

            model.mesh = None
            if 'points' in arrays:
                if model.kind == 'tet':
                    model.mesh = pv.UnstructuredGrid({pv.CellType.TETRA: arrays['tets']}, arrays['points'])
                else:
                    model.mesh = pv.PolyData(arrays['points'], arrays['faces'])
                for key in arrays.files:
                    if key.startswith('point_data_'):
                        model.mesh.point_data[key[len('point_data_'):]] = arrays[key]
                    elif key.startswith('cell_data_'):
                        model.mesh.cell_data[key[len('cell_data_'):]] = arrays[key]

        return model

    @classmethod
    def load(cls, image, path):
        """Read a model written by :meth:`save`, not yet added to the image (see :meth:`from_bytes`).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> model = mia.Model.load(image, 'C:/Data/liver_fe.npz')
        >>> image.add_saved_model(model)
        """
        with open(path, 'rb') as f:
            return cls.from_bytes(image, f.read())

    def _record(self):
        """State as plain JSON types (everything but the arrays)."""
        return {'schema': self.SCHEMA_VERSION, 'id': self.id, 'name': self.name,
                'image_id': getattr(self.image, 'id', None), 'kind': self.kind,
                'color': [int(c) for c in self.color], 'visible': bool(self.visible), 'group_id': self.group_id,
                'roi_ids': list(self.roi_ids), 'cavity_ids': list(self.cavity_ids),
                'cluster_points': self.cluster_points, 'edge_length_fac': self.edge_length_fac,
                'edge_length_abs': self.edge_length_abs, 'region_edge_lengths': self.region_edge_lengths,
                'epsilon': self.epsilon, 'optimize': bool(self.optimize), 'simplify': bool(self.simplify),
                'coarsen': bool(self.coarsen), 'stop_energy': self.stop_energy, 'num_opt_iter': self.num_opt_iter,
                'source_keys': self.source_keys, 'build_count': int(self.build_count),
                'n_surfaces': 0 if self.surfaces is None else len(self.surfaces)}

    def _region_index(self, region):
        """Region index of an ROI name or id (or -1, the outside of the mesh)."""
        if not isinstance(region, str) and int(region) == -1:
            return -1

        roi_id = Data.get_roi_id(region)
        if roi_id not in self.roi_ids:
            raise ValueError(f"ROI {Data.get_roi_name(roi_id)!r} is not a region of model '{self.name}'.")

        return self.roi_ids.index(roi_id)

    def _source_meshes(self):
        """Mesh of every source ROI on this image, in region order; raises when one has no mesh."""
        meshes = []
        for roi_id in self.roi_ids:
            roi = self.image.roi.get(roi_id)
            if roi is None or roi.mesh is None:
                label = Data.get_roi_name(roi_id) if Data.has_roi(roi_id) else f'id {roi_id}'
                raise ValueError(f"ROI {label!r} has no mesh on '{self.image.image_name}'.")
            meshes.append(roi.mesh)

        return meshes

    def _volume(self):
        """:class:`Volume` wrapped around the current mesh, for its FEBio helpers (cached until mesh is replaced)."""
        if self.kind != 'tet' or self.mesh is None:
            raise ValueError(f"Model '{self.name}' has no tetrahedral mesh; build it first (kind='tet').")

        if self._volume_cache is None or self._volume_cache.mesh is not self.mesh:
            cavities = [self.roi_ids.index(roi_id) for roi_id in self.cavity_ids]
            surfaces = self.surfaces if self.surfaces is not None else self._source_meshes()
            volume = Volume(surfaces, cavities=cavities)
            volume.mesh = self.mesh
            self._volume_cache = volume

        return self._volume_cache

    def build(self, num_threads=0, quiet=True):
        """Run the recipe: cluster each source ROI mesh, then (kind='tet') tetrahedralize them together.

        Replaces ``surfaces`` and ``mesh``, records ``source_keys`` and adds one to ``build_count``. Anything stored on
        the old mesh (results, quality) is gone, and a registration that used the old mesh reports it through
        ``Deformable.check_model``. To try another recipe while keeping this mesh, make a new model instead.

        Parameters
        ----------
        num_threads : int, default 0
            pytetwild threads; 0 = all cores.
        quiet : bool, default True
            Silence pytetwild.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> model = mia.Data.image['CT 01'].model['Liver FE']
        >>> model.build()
        >>> model.mesh.n_cells
        """
        sources = self._source_meshes()

        surfaces = []
        for roi_id, mesh, points in zip(self.roi_ids, sources, self.cluster_points):
            if points is None:
                surfaces.append(mesh.triangulate().clean())
            else:
                surfaces.append(self.image.roi[roi_id].create_cluster_mesh(points=points).triangulate().clean())

        if self.kind == 'tet':
            volume = Volume(surfaces, cavities=[self.roi_ids.index(roi_id) for roi_id in self.cavity_ids])
            volume.create(edge_length_fac=self.edge_length_fac, edge_length_abs=self.edge_length_abs,
                          region_edge_lengths=self.region_edge_lengths, optimize=self.optimize,
                          simplify=self.simplify, epsilon=self.epsilon, stop_energy=self.stop_energy,
                          coarsen=self.coarsen, num_threads=num_threads, num_opt_iter=self.num_opt_iter,
                          quiet=quiet)
            mesh = volume.mesh
            mesh.field_data.pop('RegionNames', None)        # regions are roi_ids; names come from Data when needed
            self._volume_cache = volume
        else:
            labeled = []
            for index, surface in enumerate(surfaces):
                surface = surface.copy()
                surface.point_data['Region'] = np.full(surface.n_points, index, dtype=np.int32)
                surface.cell_data['Region'] = np.full(surface.n_cells, index, dtype=np.int32)
                labeled.append(surface)
            mesh = pv.merge(labeled, merge_points=False)
            self._volume_cache = None

        self.surfaces = surfaces
        self.mesh = mesh
        self.source_keys = [self._mesh_key(source) for source in sources]
        self.build_count += 1

    def compute_mesh_slice(self, grid, return_pixel=True):
        """Element edges (tet) or contours (surface) where a view cuts the model, for checking the mesh in 2D.

        Parameters
        ----------
        grid : dict
            The view's grid (``display.get_grid``, or the 'grid' of any compute_array result on this image).
        return_pixel : bool, default True
            False returns the raw pyvista slice (world coordinates).

        Returns
        -------
        dict, pyvista.PolyData or None
            'segments' ((M, 2, 2) end points in (col, row), pixel centers at integers, each edge once) and 'roi_id'
            ((M,) source ROI of each segment, -1 for none). Draw with one call, e.g. pyqtgraph connect='pairs'. None
            when the model is not built.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> grid = image.display.get_grid('Axial', image.display.position['Axial'])
        >>> cut = image.model['Liver FE'].compute_mesh_slice(grid)
        >>> xy = cut['segments'].reshape(-1, 2)
        >>> curve.setData(xy[:, 0], xy[:, 1], connect='pairs')
        """
        if self.mesh is None:
            return None

        origin = np.asarray(grid['origin'], dtype=float)
        cut = self.mesh.slice(normal=np.asarray(grid['normal'], dtype=float), origin=origin)
        if not return_pixel:
            return cut

        if cut.n_points == 0:
            return {'segments': np.empty((0, 2, 2)), 'roi_id': np.empty(0, dtype=int)}

        if cut.faces.size > 0:
            edges, owners = self._cell_edges(cut.faces, closed=True)
        else:
            edges, owners = self._cell_edges(cut.lines, closed=False)

        _, first = np.unique(np.sort(edges, axis=1), axis=0, return_index=True)     # shared edges drawn once
        edges, owners = edges[first], owners[first]

        d = np.asarray(cut.points) - origin
        pixel = np.column_stack((d @ np.asarray(grid['x_axis'], dtype=float) / grid['sx'],
                                 d @ np.asarray(grid['y_axis'], dtype=float) / grid['sy']))

        region = np.asarray(cut.cell_data['Region'])[owners]
        roi_ids = np.append(np.asarray(self.roi_ids), -1)            # region -1 -> last entry -> -1

        return {'segments': pixel[edges], 'roi_id': roi_ids[region]}

    def compute_quality(self, quality_measure='scaled_jacobian'):
        """Per-tet quality, stored in ``mesh.cell_data['Quality']`` (see :meth:`Volume.compute_quality`).

        Scaled Jacobian: 1 = regular tetrahedron, near 0 = sliver, negative = inverted (FEBio rejects it).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> quality = mia.Data.image['CT 01'].model['Liver FE'].compute_quality()
        >>> int((quality < 0.1).sum())          # slivers
        """
        return self._volume().compute_quality(quality_measure)

    def compute_region_volumes(self):
        """Volume (mm^3) of each region of a tet model, in region order, next to its source ROI's own volume.

        A quick check of how much clustering and meshing changed each structure. A region does not include the
        regions nested inside it (the Liver region of a liver + GTV model excludes the GTV), while an ROI's own volume
        does, so compare an outer ROI against the sum of its region and the regions inside it.

        Returns
        -------
        list of (roi_id, model volume, roi volume)
            roi volume is None when the ROI is not loaded or has no mesh.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> (liver, liver_mm3, liver_roi), (gtv, gtv_mm3, gtv_roi) = mia.Data.image['CT 01'].model['Liver FE'] \
        ...     .compute_region_volumes()
        >>> round((liver_mm3 + gtv_mm3) / liver_roi, 3), round(gtv_mm3 / gtv_roi, 3)
        (0.997, 0.987)
        """
        self._volume()
        sizes = self.mesh.compute_cell_sizes(length=False, area=False, volume=True).cell_data['Volume']
        region = np.asarray(self.mesh.cell_data['Region'])

        result = []
        for index, roi_id in enumerate(self.roi_ids):
            roi = self.image.roi.get(roi_id)
            roi_volume = None if roi is None or roi.mesh is None else float(abs(roi.mesh.volume))
            result.append((roi_id, float(np.sum(sizes[region == index])), roi_volume))

        return result

    def export(self, path):
        """Write the mesh to any format pyvista writes (.vtu, .vtk, ...) for FEBio pre-processing or ParaView.

        'Region' holds region indices; the ROI of region i is ``roi_ids[i]``.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].model['Liver FE'].export('C:/Data/liver_fe.vtu')
        """
        if self.mesh is None:
            raise ValueError(f"Model '{self.name}' is not built.")

        self.mesh.save(path, binary=False)

    def extract_boundary(self, region, neighbor=None):
        """A region's boundary faces with global node / element ids, for FEBio surfaces, contacts and loads.

        Same as :meth:`Volume.extract_boundary`, with regions named by ROI (name or id). neighbor -1 = the outside of
        the mesh; a cavity ROI = its wall.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> model = mia.Data.image['CT 01'].model['Liver FE']
        >>> liver_skin = model.extract_boundary('Liver', neighbor=-1)
        >>> gtv_wall = model.extract_boundary('GTV', neighbor='Liver')
        >>> node_ids = gtv_wall.point_data['GlobalNodeIds']
        """
        neighbor = None if neighbor is None else self._region_index(neighbor)

        return self._volume().extract_boundary(self._region_index(region), neighbor=neighbor)

    def extract_region(self, region):
        """The tetrahedra of one region (ROI name or id), with ids pointing back into ``mesh``.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> gtv_tets = mia.Data.image['CT 01'].model['Liver FE'].extract_region('GTV')
        """
        return self._volume().extract_region(self._region_index(region))

    def get_region_names(self):
        """Current ROI name of each region, in region order (renaming an ROI is picked up).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].model['Liver FE'].get_region_names()
        ['Liver', 'GTV']
        """
        return [Data.get_roi_name(roi_id) if Data.has_roi(roi_id) else f'ROI {roi_id}' for roi_id in self.roi_ids]

    def get_stale_rois(self):
        """Ids of the source ROIs whose mesh changed (or is gone) since the model was built; all of them if it was
        never built.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].model['Liver FE'].get_stale_rois()
        [1042]
        """
        if self.source_keys is None:
            return list(self.roi_ids)

        stale = []
        for roi_id, key in zip(self.roi_ids, self.source_keys):
            roi = self.image.roi.get(roi_id)
            if roi is None or roi.mesh is None or self._mesh_key(roi.mesh) != key:
                stale.append(roi_id)

        return stale

    def is_stale(self):
        """True when a source ROI changed since the build (or it was never built). Nothing is rebuilt.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> model = mia.Data.image['CT 01'].model['Liver FE']
        >>> if model.is_stale():
        ...     print('Rebuild', model.name)
        """
        return len(self.get_stale_rois()) > 0

    def region_node_ids(self, region):
        """Global node ids of one region (ROI name or id), e.g. an FEBio node set or initial temperature.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> gtv_nodes = mia.Data.image['CT 01'].model['Liver FE'].region_node_ids('GTV')
        """
        return self._volume().region_node_ids(self._region_index(region))

    def save(self, path):
        """Write the model to one .npz file (record, surfaces, mesh and its data arrays). Written to a temporary name
        and swapped in, so a crash never leaves a half-written file. No pickle.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].model['Liver FE'].save('C:/Data/liver_fe.npz')
        """
        temp = path + '.tmp'
        with open(temp, 'wb') as f:
            f.write(self.to_bytes())
        os.replace(temp, path)

    def to_bytes(self):
        """The model as .npz bytes (what :meth:`save` writes), e.g. to encrypt or store in the database.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> data = mia.Data.image['CT 01'].model['Liver FE'].to_bytes()
        """
        arrays = {'record': np.array(json.dumps(self._record(), separators=(',', ':')))}

        if self.surfaces is not None:
            for i, surface in enumerate(self.surfaces):
                arrays[f'surface_{i}_points'] = np.asarray(surface.points)
                arrays[f'surface_{i}_faces'] = np.asarray(surface.faces)

        if self.mesh is not None:
            arrays['points'] = np.asarray(self.mesh.points)
            if self.kind == 'tet':
                arrays['tets'] = np.asarray(self.mesh.cells_dict[pv.CellType.TETRA])
            else:
                arrays['faces'] = np.asarray(self.mesh.faces)
            for key in self.mesh.point_data.keys():
                arrays['point_data_' + key] = np.asarray(self.mesh.point_data[key])
            for key in self.mesh.cell_data.keys():
                arrays['cell_data_' + key] = np.asarray(self.mesh.cell_data[key])

        buffer = io.BytesIO()
        np.savez(buffer, **arrays)

        return buffer.getvalue()

    def update_mesh(self, mesh, surfaces=None):
        """Use a mesh built elsewhere (another mesher, a script) as this model, and record the current source ROI
        meshes as its source. Counts as a rebuild (``build_count`` goes up by one).

        A tet mesh needs cell array 'Region' (indices into ``roi_ids``).

        Examples
        --------
        >>> import pyvista as pv
        >>> import medicalimageanalysis as mia
        >>> model = mia.Data.image['CT 01'].model['Liver FE']
        >>> model.update_mesh(pv.read('C:/Data/liver_fe_edited.vtu'))
        """
        if self.kind == 'tet' and 'Region' not in mesh.cell_data:
            raise ValueError("A tet model's mesh needs cell array 'Region' (indices into roi_ids).")

        self.mesh = mesh
        if surfaces is not None:
            self.surfaces = list(surfaces)
        self.source_keys = [self._mesh_key(source) for source in self._source_meshes()]
        self.build_count += 1
        self._volume_cache = None