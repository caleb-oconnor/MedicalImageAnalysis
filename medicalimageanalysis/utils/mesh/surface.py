"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Surface mesh refinement (smoothing, clustering, decimation, fin pruning), mesh quality checks, and connected
    component utilities.

    Methods
    -------
    * Smoothing: :meth:`Refinement.smooth` (``vtkWindowedSincPolyDataFilter``)
    * Point reduction: :meth:`Refinement.cluster` (pyacvd with retries, fin pruning and a ``decimate_pro``
      fallback until the surface is closed and manifold), :meth:`Refinement.decimate`
    * Target point counts: :meth:`Refinement.compute_points` (10 * sqrt(n)), :meth:`Refinement.compute_volume_points`
      (points per mL, optionally scaled by surface complexity)
    * Repair: :meth:`Refinement.prune_fins`
    * Metrics: :meth:`Refinement.surface_defects`, :meth:`Refinement.complexity_score`,
      :meth:`Refinement.mean_edge_length`
    * Components: :func:`only_main_component`, :func:`compute_components`, :func:`component_overlap`

    Conventions
    -----------
    * Input meshes are assumed to be closed (watertight) surfaces, e.g. ROI meshes. Open boundaries are treated as
      defects by :meth:`Refinement.cluster` and eroded by :meth:`Refinement.prune_fins`.
    * Mesh units are assumed to be millimeters, so volumes are converted to mL with ``mm^3 / 1000``.
    * ``Refinement.mesh`` holds the current mesh. Each refinement method replaces it with a new mesh and returns it,
      so calls can be chained (``refine.smooth(); refine.cluster()``). The mesh passed to the constructor
      is never modified.
    * A "clean" surface is closed, edge-manifold and all triangles; :meth:`Refinement.surface_defects` returns 0 for
      it, a positive count of boundary + non-manifold edges otherwise, and -1 if the mesh is empty or not all
      triangles.
    * Inputs may be ``pyvista.PolyData`` or any pyvista dataset (e.g. ``UnstructuredGrid``), which is converted with
      ``extract_surface``.

Structure:
    Refinement
    ├── __init__                     store a surface copy of the mesh
    ├── smooth                       windowed sinc smoothing
    ├── cluster                      pyacvd retries -> prune fins -> decimate_pro fallback
    ├── decimate                     quadric decimation
    ├── prune_fins                   strip faces with a free edge
    ├── compute_points / compute_point_percentage / compute_volume_points
    ├── surface_defects / complexity_score / mean_edge_length
    └── _tidy / _defect_rank         static helpers
    only_main_component              largest connected component
    component_overlap                match components by overlap or distance
    compute_components               split into connected components

Examples
--------
>>> import pyvista as pv
>>> from medicalimageanalysis.utils.mesh.surface import Refinement
>>> refine = Refinement(pv.Sphere(radius=30, theta_resolution=100, phi_resolution=100))
>>> _ = refine.smooth()
>>> mesh = refine.cluster(points=2000)
>>> refine.method, refine.surface_defects()
('pyacvd', 0)
"""

import math
import warnings

import vtk
import pyacvd
import numpy as np
import pyvista as pv


def _ensure_polydata(mesh):
    """Returns the mesh as a pyvista.PolyData surface.

    Parameters
    ----------
    mesh : pyvista.DataSet or vtk.vtkDataSet
        Input mesh. PolyData is returned as-is; other datasets are converted with ``extract_surface``.

    Returns
    -------
    pyvista.PolyData
        Surface mesh.

    Raises
    ------
    TypeError
        If mesh cannot be wrapped as a pyvista dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> grid = pv.Sphere().cast_to_unstructured_grid()
    >>> type(_ensure_polydata(grid)).__name__
    'PolyData'
    """

    if isinstance(mesh, pv.PolyData):
        return mesh

    if not isinstance(mesh, pv.DataSet):
        try:
            mesh = pv.wrap(mesh)

        except Exception:
            raise TypeError(f"mesh must be a pyvista dataset, got {type(mesh).__name__}") from None

    return mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()


class Refinement(object):
    """Provides tools for refining surface meshes: smoothing, clustering, decimation and fin pruning, plus quality
    metrics used to check the result.

    Parameters
    ----------
    mesh : pyvista.PolyData or pyvista.DataSet
        Input surface. Non-PolyData datasets are converted with ``extract_surface``. A copy is stored, so the input is
        never modified.

    Examples
    --------
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.surface import Refinement
    >>> refine = Refinement(pv.Sphere(theta_resolution=32, phi_resolution=32))
    >>> smooth_mesh = refine.smooth()
    >>> smooth_mesh.n_points
    962
    """

    def __init__(self, mesh):
        """Stores a copy of the mesh as the current surface.

        Parameters
        ----------
        mesh : pyvista.PolyData or pyvista.DataSet
            Input surface.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=32, phi_resolution=32)
        >>> refine = Refinement(sphere)
        >>> refine.mesh.n_points, refine.mesh is sphere
        (962, False)
        """

        self.mesh = _ensure_polydata(mesh).copy(deep=True)

        # Which path cluster took ('native', 'pyacvd', 'pyacvd(retry k)', 'pyacvd+prune_fins(...)',
        # 'decimate_pro_fallback'). None until cluster is called.
        self.method = None

    @staticmethod
    def _count_defects(mesh):
        """Static version of :meth:`surface_defects`."""

        if mesh.n_cells == 0 or mesh.n_lines or mesh.n_verts or not mesh.is_all_triangles:
            return -1

        fe = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=True, feature_edges=False,
                                        manifold_edges=False)

        # A closed triangle mesh always has an even face count.
        return int(fe.n_cells) + int(mesh.n_cells % 2)

    @staticmethod
    def _defect_rank(n_bad):
        """Sort key for defect counts: invalid meshes (-1) rank last."""
        return np.iinfo(np.int32).max if n_bad < 0 else n_bad

    @staticmethod
    def _prune_fins(mesh, max_iter=20):
        """Static version of :meth:`prune_fins`; returns a new mesh."""

        faces = mesh.regular_faces
        for _ in range(max_iter):
            nf = len(faces)
            edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            _, inv, cnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True)

            # Faces-per-edge for each face's 3 edges; any edge used by a single face is free.
            edge_cnt = cnt[inv.ravel()].reshape(3, nf).T
            dangling = (edge_cnt == 1).any(axis=1)
            if not dangling.any():
                break

            faces = faces[~dangling]

        return pv.PolyData.from_regular_faces(mesh.points, faces).clean()

    @staticmethod
    def _tidy(mesh):
        """Triangulates and merges duplicate points, then drops any cells clean() turned into lines/vertices.

        Examples
        --------
        >>> import pyvista as pv
        >>> Refinement._tidy(pv.Cube()).n_cells
        12
        """

        mesh = mesh.triangulate().clean()
        if mesh.n_lines or mesh.n_verts:
            # mesh.faces only holds polygons; lines/vertices are stored separately.
            tris = mesh.faces.reshape(-1, 4)[:, 1:]
            mesh = pv.PolyData.from_regular_faces(mesh.points, tris).clean()

        return mesh

    def cluster(self, points=None, points_per_ml=None, min_points=2000, max_points=7000, complexity_weight=0.0,
                complexity_metric='compactness', min_multiplier=0.9, max_multiplier=1.1, max_tries=5):
        """Clusters the mesh with pyacvd and makes sure the result is a closed, edge-manifold triangle surface.

        Workflow
        --------
        0. Target point count: ``points`` if given; otherwise :meth:`compute_volume_points` if ``points_per_ml`` is
           given (scaled by complexity when ``complexity_weight > 0``); otherwise :meth:`compute_points`.
        1. If the target is >= the current point count, the mesh is kept as-is ('native').
        2. pyacvd is run up to ``max_tries`` times with target counts of 0, -1%, +2%, -3%, +4%, ... around the target
           (a different count gives a different partition). The first clean result is returned.
        3. Otherwise every failed candidate is passed through :meth:`prune_fins`, fewest defects first, and the first
           one that comes out clean is returned.
        4. Last resort: ``decimate_pro`` with topology preservation, pruned if needed. A warning is raised if it is
           still not clean.

        Parameters
        ----------
        points : int, optional
            Fixed target number of points. Cannot be combined with ``points_per_ml``.
        points_per_ml : float, optional
            Volume-based target, see :meth:`compute_volume_points`. Only used when ``points`` is None.
        min_points, max_points : int
            Bounds on the volume-based target.
        complexity_weight : float
            Complexity scaling of the volume-based target. 0 (default) disables it.
        complexity_metric : str
            Metric passed to :meth:`complexity_score`.
        min_multiplier, max_multiplier : float
            Bounds on the complexity multiplier.
        max_tries : int
            Number of pyacvd attempts before fin pruning.

        Returns
        -------
        mesh : pyvista.PolyData
            Clustered mesh (also stored in self.mesh). The path that produced it is stored in self.method.

        Raises
        ------
        ValueError
            If both ``points`` and ``points_per_ml`` are given.

        Examples
        --------
        >>> import pyvista as pv
        >>> refine = Refinement(pv.Sphere(radius=30, theta_resolution=100, phi_resolution=100))
        >>> mesh = refine.cluster(points_per_ml=8.0, min_points=1500, max_points=2500)
        >>> refine.method, refine.surface_defects(), mesh.n_points
        ('pyacvd', 0, 1500)
        >>> _ = refine.cluster(points=50000)
        >>> refine.method
        'native'
        >>> refine.cluster(points=1000, points_per_ml=8.0)
        Traceback (most recent call last):
        ...
        ValueError: Give either points or points_per_ml, not both
        """

        if points is not None and points_per_ml is not None:
            raise ValueError('Give either points or points_per_ml, not both')

        if points is not None:
            target_points = int(points)

        elif points_per_ml is not None:
            target_points = self.compute_volume_points(points_per_ml=points_per_ml, min_points=min_points,
                                                       max_points=max_points, complexity_weight=complexity_weight,
                                                       complexity_metric=complexity_metric,
                                                       min_multiplier=min_multiplier, max_multiplier=max_multiplier)
        else:
            target_points = self.compute_points()

        mesh = self.mesh

        if target_points >= mesh.n_points:
            self.method = 'native'
            return self.mesh

        # 1) Clustering attempts: keep the first clean result, otherwise keep every candidate as (n_bad, k, mesh).
        candidates = []
        last_err = None
        for k in range(max_tries):
            tp = int(round(target_points * (1.0 + 0.01 * k * (-1) ** k)))
            try:
                clusterer = pyacvd.Clustering(mesh)
                clusterer.cluster(tp)
                clustered = self._tidy(clusterer.create_mesh())

            except Exception as e:
                last_err = e
                continue

            n_bad = self._count_defects(clustered)
            if n_bad == 0:
                self.mesh = clustered
                self.method = 'pyacvd' if k == 0 else f'pyacvd(retry {k})'
                return self.mesh

            candidates.append((n_bad, k, clustered))
            last_err = f'{n_bad} defects at target {tp}'

        # 2) No clean partition: prune fins, fewest defects first.
        for n_bad, k, clustered in sorted(candidates, key=lambda c: self._defect_rank(c[0])):
            pruned = self._prune_fins(clustered)
            if self._count_defects(pruned) == 0:
                self.mesh = pruned
                self.method = f'pyacvd+prune_fins(try {k}, {n_bad} defects)'
                return self.mesh

        # 3) Last resort: decimate_pro, pruned if needed.
        reduction = float(np.clip(1.0 - target_points / mesh.n_points, 0.0, 0.99))
        clustered = self._tidy(mesh.decimate_pro(reduction, preserve_topology=True))
        if self._count_defects(clustered) != 0:
            clustered = self._prune_fins(clustered)

        n_bad = self._count_defects(clustered)
        if n_bad != 0:
            warnings.warn(f'cluster: decimate_pro fallback still has {n_bad} defects '
                          f'(last pyacvd error: {last_err})')

        self.mesh = clustered
        self.method = 'decimate_pro_fallback'

        return self.mesh

    def complexity_score(self, metric='compactness'):
        """Measures how far the shape is from a sphere.

        ``'compactness'`` is ``(area / volume^(2/3)) / (36 pi)^(1/3) - 1``: 0 for a sphere and larger for elongated
        or folded shapes. Requires a closed surface for the volume to be meaningful.

        Parameters
        ----------
        metric : {'compactness'}
            Complexity metric.

        Returns
        -------
        float
            Complexity score (>= 0).

        Raises
        ------
        ValueError
            If metric is not supported.

        Examples
        --------
        >>> import pyvista as pv
        >>> round(Refinement(pv.Sphere(theta_resolution=100, phi_resolution=100)).complexity_score(), 3)
        0.0
        >>> round(Refinement(pv.Cube()).complexity_score(), 3)
        0.241
        """

        if metric != 'compactness':
            raise ValueError(f'Unsupported complexity metric: {metric}')

        mesh = self.mesh.triangulate().clean()
        area = max(float(abs(mesh.area)), 1e-8)
        volume = max(float(abs(mesh.volume)), 1e-8)

        sphere_compactness = (36.0 * math.pi) ** (1.0 / 3.0)
        compactness = area / (volume ** (2.0 / 3.0))

        return float(max(0.0, (compactness / sphere_compactness) - 1.0))

    def compute_points(self):
        """Computes a target number of points as 10 * sqrt(current point count).

        Returns
        -------
        int
            Target number of points for clustering/decimation.

        Examples
        --------
        >>> import pyvista as pv
        >>> refine = Refinement(pv.Sphere(theta_resolution=32, phi_resolution=32))
        >>> refine.compute_points()
        310
        """

        return int(np.round(10 * np.sqrt(self.mesh.n_points)))

    def compute_point_percentage(self):
        """Computes the fraction of points to remove so that :meth:`compute_points` points remain.

        Returns
        -------
        float
            Fraction to remove (0-1).

        Examples
        --------
        >>> import pyvista as pv
        >>> refine = Refinement(pv.Sphere(theta_resolution=32, phi_resolution=32))
        >>> round(refine.compute_point_percentage(), 3)
        0.678
        """

        return float(1 - (self.compute_points() / self.mesh.n_points))

    def compute_volume_points(self, points_per_ml=8.0, min_points=2000, max_points=7000, complexity_weight=0.0,
                              complexity_metric='compactness', min_multiplier=0.9, max_multiplier=1.1):
        """Computes a target point count from the enclosed volume, optionally scaled up for complex shapes.

        ``base = clip(points_per_ml * volume_ml, min_points, max_points)``, then
        ``points = clip(base * clip(1 + complexity_weight * complexity_score, min_multiplier, max_multiplier),
        min_points, max_points)``. With ``complexity_weight=0`` (default) the result is just the volume estimate.

        Parameters
        ----------
        points_per_ml : float
            Points per mL of enclosed volume (mesh volume in mm^3 / 1000).
        min_points, max_points : int
            Bounds on the returned count, applied before and after the complexity scaling.
        complexity_weight : float
            Weight on :meth:`complexity_score`. 0 disables the scaling.
        complexity_metric : str
            Metric passed to :meth:`complexity_score`.
        min_multiplier, max_multiplier : float
            Bounds on the complexity multiplier. The score is >= 0, so the multiplier never drops below 1 and
            ``min_multiplier`` only matters if it is set above 1.

        Returns
        -------
        int
            Target number of points.

        Examples
        --------
        >>> import pyvista as pv
        >>> refine = Refinement(pv.ParametricEllipsoid(60, 30, 20))
        >>> refine.compute_volume_points(points_per_ml=30)
        4520
        >>> refine.compute_volume_points(points_per_ml=30, complexity_weight=0.12)
        4605
        """

        volume_ml = float(abs(self.mesh.volume)) / 1000.0
        base = int(np.clip(int(round(points_per_ml * volume_ml)), min_points, max_points))

        if not complexity_weight:
            return base

        multiplier = 1.0 + float(complexity_weight) * self.complexity_score(complexity_metric)
        multiplier = float(np.clip(multiplier, min_multiplier, max_multiplier))

        return int(np.clip(int(round(base * multiplier)), min_points, max_points))

    def decimate(self, percent=None):
        """Decimates (reduces) mesh complexity by removing points and faces.

        Parameters
        ----------
        percent : float, optional
            Fraction of the faces to remove (0-1). If None, :meth:`compute_point_percentage` is used.

        Returns
        -------
        mesh : pyvista.PolyData
            Decimated mesh (also stored in self.mesh).

        Examples
        --------
        >>> import pyvista as pv
        >>> refine = Refinement(pv.Sphere(theta_resolution=32, phi_resolution=32))
        >>> refine.decimate(0.5).n_cells
        960
        """

        if percent is None:
            percent = self.compute_point_percentage()

        self.mesh = self.mesh.decimate(float(percent))

        return self.mesh

    def mean_edge_length(self):
        """Computes the mean length of the unique edges of the mesh.

        Returns
        -------
        float
            Mean edge length (mesh units).

        Examples
        --------
        >>> import pyvista as pv
        >>> round(Refinement(pv.Cube()).mean_edge_length(), 3)
        1.138
        """

        mesh = self.mesh.triangulate().clean()
        edges = mesh.extract_all_edges()

        return float(np.mean(edges.compute_cell_sizes(length=True)['Length']))

    def prune_fins(self, max_iter=20):
        """Iteratively strips faces that have a free edge (zero-thickness flaps), then drops unused points.

        Faces are only removed, never added, so holes are not filled; a surface with real holes is eroded until it is
        closed or ``max_iter`` is reached.

        Parameters
        ----------
        max_iter : int
            Maximum number of strip passes.

        Returns
        -------
        mesh : pyvista.PolyData
            Pruned mesh (also stored in self.mesh).

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> fin = np.vstack([sphere.points, [[2.0, 0.0, 0.0]]])
        >>> faces = np.vstack([sphere.regular_faces, [[0, 1, len(fin) - 1]]])
        >>> refine = Refinement(pv.PolyData.from_regular_faces(fin, faces))
        >>> refine.surface_defects()
        4
        >>> refine.prune_fins().n_cells == sphere.n_cells, refine.surface_defects()
        (True, 0)
        """

        self.mesh = self._prune_fins(self.mesh, max_iter)

        return self.mesh

    def smooth(self, iterations=20, angle=60, passband=0.001):
        """Smooths the mesh geometry using a windowed sinc filter.

        Parameters
        ----------
        iterations : int
            Number of smoothing iterations.
        angle : float
            Feature angle in degrees for smoothing sharp edges.
        passband : float
            Passband for the filter (lower is smoother).

        Returns
        -------
        mesh : pyvista.PolyData
            Smoothed mesh (also stored in self.mesh). Point count and connectivity are unchanged.

        Examples
        --------
        >>> import pyvista as pv
        >>> refine = Refinement(pv.Sphere(theta_resolution=32, phi_resolution=32))
        >>> refine.smooth(iterations=10).n_cells
        1920
        """

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(self.mesh)
        smoother.SetNumberOfIterations(iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(angle)
        smoother.SetPassBand(passband)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOff()
        smoother.Update()
        self.mesh = pv.PolyData(smoother.GetOutput())

        return self.mesh

    def surface_defects(self):
        """Counts surface defects of the current mesh, for QA and failure detection.

        Returns
        -------
        int
            0 for a closed, edge-manifold, all-triangle surface; otherwise the number of boundary + non-manifold
            edges (+1 if the face count is odd). -1 if the mesh is empty, has lines/vertices, or is not all
            triangles.

        Examples
        --------
        >>> import pyvista as pv
        >>> Refinement(pv.Sphere()).surface_defects()
        0
        >>> Refinement(pv.Plane()).surface_defects()
        -1
        >>> Refinement(pv.Plane().triangulate()).surface_defects()
        40
        """

        return self._count_defects(self.mesh)


def component_overlap(components_1, components_2, min_distance=0):
    """Matches connected components from two meshes by overlap or proximity.

    Workflow
    --------
    1. Pairs whose bounding boxes intersect are tested for overlap: surface intersection, or any point of one
       component enclosed by the other. Overlapping pairs are matched with distance 0.
    2. Components left unmatched on both sides are paired if their nearest surface distance is <= ``min_distance``.

    Parameters
    ----------
    components_1, components_2 : list[pyvista.DataSet]
        Connected mesh components to compare (e.g. from :func:`compute_components`). Surfaces are expected to be
        closed for the enclosed-point test.
    min_distance : float, optional
        Maximum nearest-surface distance (mm) allowed when no direct overlap is found.

    Returns
    -------
    list[dict]
        Match records ``{'Components_1': i, 'Components_2': j, 'distance_mm': d}``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.surface import component_overlap
    >>> a = [pv.Sphere(radius=1), pv.Sphere(radius=1, center=(10, 0, 0))]
    >>> b = [pv.Sphere(radius=1, center=(0.5, 0, 0)), pv.Sphere(radius=1, center=(12.5, 0, 0))]
    >>> [(m['Components_1'], m['Components_2']) for m in component_overlap(a, b)]
    [(0, 0)]
    >>> [(m['Components_1'], m['Components_2']) for m in component_overlap(a, b, min_distance=1)]
    [(0, 0), (1, 1)]
    """

    matches = []
    for ii, c1 in enumerate(components_1):
        for jj, c2 in enumerate(components_2):

            ax0, ax1, ay0, ay1, az0, az1 = c1.bounds
            bx0, bx1, by0, by1, bz0, bz1 = c2.bounds
            if not (ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0 and az0 <= bz1 and az1 >= bz0):
                continue

            surf1 = _ensure_polydata(c1)
            surf2 = _ensure_polydata(c2)

            surface_intersection = surf1.intersection(surf2)[0].n_cells > 0

            inside2 = surf2.select_enclosed_points(pv.PolyData(c1.points), tolerance=1e-5, check_surface=False)
            frac1_in_2 = (inside2['SelectedPoints'] > 0).mean()

            inside1 = surf1.select_enclosed_points(pv.PolyData(c2.points), tolerance=1e-5, check_surface=False)
            frac2_in_1 = (inside1['SelectedPoints'] > 0).mean()

            if surface_intersection or frac1_in_2 > 0 or frac2_in_1 > 0:
                matches += [{'Components_1': ii, 'Components_2': jj, 'distance_mm': 0}]

    matched1 = {m['Components_1'] for m in matches}
    matched2 = {m['Components_2'] for m in matches}

    unmatched1 = [ii for ii in range(len(components_1)) if ii not in matched1]
    unmatched2 = [jj for jj in range(len(components_2)) if jj not in matched2]

    for ii in unmatched1:
        for jj in unmatched2:

            surf1 = _ensure_polydata(components_1[ii])
            surf2 = _ensure_polydata(components_2[jj])

            # Distance from every point on surf1 to the surf2 surface.
            dist = surf1.compute_implicit_distance(surf2)
            d = float(np.abs(dist['implicit_distance']).min())

            if d <= min_distance:
                matches += [{'Components_1': ii, 'Components_2': jj, 'distance_mm': d}]

    return matches


def compute_components(mesh):
    """Splits a mesh into connected components.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.

    Returns
    -------
    list[pyvista.UnstructuredGrid]
        One mesh per connected region, ordered by region id (largest first).

    Examples
    --------
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.surface import compute_components
    >>> mesh = pv.Sphere().merge(pv.Sphere(center=(5, 0, 0)))
    >>> len(compute_components(mesh))
    2
    """

    conn = mesh.connectivity(label_regions=True)
    region_ids = np.unique(conn['RegionId'])

    return [conn.threshold((rid, rid), scalars='RegionId') for rid in region_ids]


def only_main_component(mesh):
    """Extracts the largest connected component (by point count) of a mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh which may contain multiple disconnected components.

    Returns
    -------
    pyvista.PolyData
        The largest connected component as a surface, or the input mesh if it has a single component.

    Examples
    --------
    >>> import pyvista as pv
    >>> from medicalimageanalysis.utils.mesh.surface import only_main_component
    >>> big = pv.Sphere(radius=2, theta_resolution=40, phi_resolution=40)
    >>> small = pv.Sphere(radius=0.5, center=(5, 0, 0))
    >>> only_main_component(big.merge(small)).n_points == big.n_points
    True
    """

    multi_block = mesh.split_bodies()

    if len(multi_block) == 1:
        return mesh

    idx = int(np.argmax([m.n_points for m in multi_block]))

    return multi_block[idx].extract_surface()

