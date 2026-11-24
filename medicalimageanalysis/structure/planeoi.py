"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Planes of interest (PlaneOI): cut planes for surgery (resection, osteotomy), reference planes (midsagittal, ...)
    and any other plane placed on an image. A plane has no inside, so it is not an ROI: instead it acts on ROIs
    (:meth:`PlaneOI.cut_roi` writes the part that is kept as a new ROI). Like a markup it belongs to one image and is
    never synced to other images. It can be copied to another image through a Rigid (:meth:`PlaneOI.copy_to`), which
    makes a new, independent plane there; there is no live link and no copy through a Deformable (a plane does not
    stay a plane through a DVF).

    Conventions
    -----------
    * ``origin`` is a world point (x, y, z, mm) on the plane and the center of its extent. ``normal`` and ``x_axis``
      are unit vectors; ``y_axis = normal x x_axis``, so (x_axis, y_axis, normal) is right-handed.
    * ``extent`` is (width, height) in mm along x_axis and y_axis, or None for an infinite plane. It only bounds
      the display; cutting always uses the whole plane.
    * ``keep_side`` is +1 (keep the side the normal points to), -1 (keep the other side) or None (not a cut plane).
    * ``group`` is a free label tying planes together, e.g. the two to four planes of one resection.
    * Display: a plane crosses any view that is not parallel to it in a straight line, so it is shown on every view
      (:meth:`PlaneOI.compute_display`, analytic: no mesh slicing). A view lying in the plane shows its extent (or the
      whole view for an infinite plane); a parallel view elsewhere shows nothing. :meth:`PlaneOI.go_to` turns a view
      to look straight at the plane.
    * ``id`` is permanent and unique across every image (``Data.new_planeoi_id``); ``name`` is unique on its image
      only.

Structure:
    - PlaneOI: geometry, display on a view grid, 3D quad, mesh / ROI cutting, copy through a Rigid and a JSON record.

Examples
--------
>>> import medicalimageanalysis as mia
>>> image = mia.Data.image['CT 01']
>>> cut = image.add_planeoi(origin=[12.0, -80.0, 410.0], normal=[0.0, 0.3, 0.95], extent=(120, 90), keep_side=-1,
...                          group='Resection 1', name='Superior cut')

Drawing it on a view (a line on a crossing view, the rectangle on a view lying in it)

>>> grid = image.display.get_grid('Sagittal', image.display.position['Sagittal'])
>>> shown = cut.compute_display(grid)
>>> shown['kind'], shown['pixel']                  # 'line', (2, 2) end points in (col, row)

Look straight at it (the other two views turn with it, perpendicular to the cut)

>>> cut.go_to('Axial')

Cut the bone with every plane of the resection

>>> mesh = image.roi['Pelvis_L'].mesh
>>> for plane in image.planeoi.values():
...     if plane.group == 'Resection 1':
...         mesh = plane.cut_mesh(mesh)

Plane from three clicked points, then copied to the MR through a registration

>>> plane = mia.PlaneOI.from_points(image, [p0, p1, p2], name='Osteotomy')
>>> image.planeoi.add(plane, plane.name)
>>> on_mr = plane.copy_to('MR 01', rigid_name='CT 01_MR 01')
"""

import json

import numpy as np
import pyvista as pv

from .roi import random_color
from ..data import Data


class PlaneOI(object):
    """One plane of interest on one image.

    Made by the image (:meth:`Image.add_planeoi`) and stored in ``image.planeoi``, which looks it up by permanent id
    or by name (``image.planeoi[4]``, ``image.planeoiplaneoi['Superior cut']``).

    Parameters
    ----------
    image : Image
        Image the plane belongs to.
    origin : (3,) array_like
        World point on the plane (mm); the center of its extent.
    normal : (3,) array_like
        Plane normal (any length, stored as a unit vector).
    x_axis : (3,) array_like, optional
        In-plane x direction (projected into the plane). None picks the world axis furthest from the normal.
    name : str, optional
        Replaced by the image's registry when the plane is added.
    extent : (2,) array_like, optional
        (width, height) in mm along x_axis / y_axis. None = infinite.
    keep_side : {1, -1, None}, optional
        Side kept when cutting: +1 the side the normal points to, -1 the other. None = not a cut plane.
    group : str, optional
        Label tying planes together (e.g. one resection).
    color : list of int, optional
        RGB, 0-255. Random if None.
    visible : bool, default True
        Show in the views.

    Attributes
    ----------
    id : int or None
        Permanent id, written by ``image.planeoi``. None until the plane is added.
    y_axis : (3,) ndarray
        normal x x_axis.
    misc : dict
        Free space for the application.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image['CT 01']
    >>> plane = mia.PlaneOI(image, origin=image.display.crosshair, normal=[0, 0, 1])
    >>> image.planeoi.add(plane, 'Axial reference')
    """
    SCHEMA_VERSION = 1

    def __init__(self, image, origin, normal, x_axis=None, name=None, extent=None, keep_side=None, group=None,
                 color=None, visible=True):
        if keep_side not in (1, -1, None):
            raise ValueError(f"keep_side must be 1, -1 or None; got {keep_side!r}")

        self.image = image

        self.id = None
        self.name = name
        self.color = list(random_color()) if color is None else list(color)
        self.visible = visible

        self.origin = np.asarray(origin, dtype=float).reshape(3)
        self.x_axis, self.y_axis, self.normal = self._orthonormal(normal, x_axis)
        self.extent = None if extent is None else (float(extent[0]), float(extent[1]))
        self.keep_side = keep_side
        self.group = group

        self.misc = {}

    @staticmethod
    def _clip_interval(offsets, slopes, lows, highs):
        """Range [lo, hi] of t where low <= offset + slope * t <= high holds for every row, or None when empty."""
        lo, hi = -np.inf, np.inf
        for offset, slope, low, high in zip(offsets, slopes, lows, highs):
            if abs(slope) < 1e-12:
                if offset < low or offset > high:
                    return None
                continue

            t0, t1 = sorted(((low - offset) / slope, (high - offset) / slope))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return None

        return lo, hi

    @staticmethod
    def _orthonormal(normal, x_axis=None):
        """(x_axis, y_axis, normal) unit and right-handed. x_axis is projected into the plane; None (or one along the
        normal) is replaced by the world axis furthest from the normal."""
        normal = np.asarray(normal, dtype=float).reshape(3)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            raise ValueError("Plane normal has zero length.")
        normal = normal / length

        if x_axis is not None:
            x_axis = np.asarray(x_axis, dtype=float).reshape(3)
            x_axis = x_axis - np.dot(x_axis, normal) * normal

        if x_axis is None or np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.identity(3)[int(np.argmin(np.abs(normal)))]
            x_axis = x_axis - np.dot(x_axis, normal) * normal

        x_axis = x_axis / np.linalg.norm(x_axis)

        return x_axis, np.cross(normal, x_axis), normal

    @classmethod
    def from_json(cls, image, text):
        """Rebuild a plane from :meth:`to_json` output, not yet added to the image.

        ``id`` and ``name`` hold the saved values; :meth:`Image.markups_from_json` adds it with them (or with a new id
        when the saved one is taken).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> plane = mia.PlaneOI.from_json(image, text)
        >>> image.planeoi.add(plane, plane.name)
        """
        record = json.loads(text)
        schema = record.get('schema', 1)
        if schema > cls.SCHEMA_VERSION:
            raise ValueError(f"Record schema {schema} is newer than supported ({cls.SCHEMA_VERSION}).")

        plane = cls(image, record['origin'], record['normal'], x_axis=record['x_axis'], name=record['name'],
                    extent=record['extent'], keep_side=record['keep_side'], group=record['group'],
                    color=record['color'], visible=record['visible'])
        plane.id = record.get('id')

        return plane

    @classmethod
    def from_points(cls, image, points, name=None, extent=None, keep_side=None, group=None, color=None,
                    visible=True):
        """Plane through three or more world points (least squares for more than three), not yet added to the image.

        The origin is the points' centroid and x_axis their main in-plane direction. The normal's sign follows the
        right-hand rule around the first three points (counter-clockwise seen from the side the normal points to);
        use :meth:`flip` to turn it.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> plane = mia.PlaneOI.from_points(image, [p0, p1, p2], name='Osteotomy', keep_side=1)
        >>> image.planeoi.add(plane, plane.name)
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) < 3:
            raise ValueError(f"A plane needs at least 3 points; got {len(points)}.")

        center = points.mean(axis=0)
        _, singular, vt = np.linalg.svd(points - center)
        if singular[1] < 1e-9 * max(singular[0], 1.0):
            raise ValueError("The points lie on one line; they do not define a plane.")

        normal = vt[2]
        if np.dot(normal, np.cross(points[1] - points[0], points[2] - points[0])) < 0:
            normal = -normal

        return cls(image, center, normal, x_axis=vt[0], name=name, extent=extent, keep_side=keep_side, group=group,
                   color=color, visible=visible)

    def _clip_mesh(self, mesh, normal, cap):
        """One pyvista clip of mesh by the plane with this normal. Which side pyvista keeps is checked by cut_mesh."""
        origin = tuple(float(v) for v in self.origin)
        normal = tuple(float(v) for v in normal)
        if cap:
            return mesh.clip_closed_surface(normal=normal, origin=origin)

        result = mesh.clip(normal=normal, origin=origin, invert=False)
        if not isinstance(result, pv.PolyData):
            result = result.extract_surface()

        return result

    def _corners(self, width, height):
        """(4, 3) world corners of a width x height rectangle on the plane, around the origin, in order."""
        a = 0.5 * width * self.x_axis
        b = 0.5 * height * self.y_axis

        return np.array([self.origin - a - b, self.origin + a - b, self.origin + a + b, self.origin - a + b])

    def _record(self):
        """State as plain JSON types."""
        return {'schema': self.SCHEMA_VERSION, 'id': self.id, 'name': self.name,
                'image_id': getattr(self.image, 'id', None), 'origin': self.origin.tolist(),
                'normal': self.normal.tolist(), 'x_axis': self.x_axis.tolist(),
                'extent': None if self.extent is None else list(self.extent), 'keep_side': self.keep_side,
                'group': self.group, 'color': [int(c) for c in self.color], 'visible': bool(self.visible)}

    def _registration_matrix(self, target, rigid_name=None):
        """4x4 this image's world -> target image's world (target is an Image).

        Without a Rigid the two images must share a frame of reference. With one, the full registration is used (its
        display animation fraction is ignored), in whichever direction carries this image onto the target.
        """
        if rigid_name is None:
            if Data.frame_of(target.id) != Data.frame_of(self.image.id):
                raise ValueError(f"'{target.image_name}' is not in the frame of reference of "
                                 f"'{self.image.image_name}'; pass the rigid_name of a registration that connects "
                                 f"them.")
            return np.identity(4)

        rigid = Data.rigid[rigid_name]
        if not rigid.connects(self.image.id, target.id):
            raise ValueError(f"'{rigid.rigid_name}' does not connect '{self.image.image_name}' and "
                             f"'{target.image_name}'.")

        if self.image.id == rigid.reference.image_id:
            on_reference = True
        elif self.image.id == rigid.moving.image_id:
            on_reference = False
        else:
            on_reference = Data.frame_of(self.image.id) == rigid.reference.frame

        combined = rigid.get_combined_matrix()          # reference world -> moving world
        if on_reference:
            return combined

        inverse = np.identity(4)
        inverse[:3, :3] = combined[:3, :3].T
        inverse[:3, 3] = -combined[:3, :3].T @ combined[:3, 3]

        return inverse

    def _tolerance(self):
        """Default distance (mm) a parallel view may be from the plane and still count as lying in it."""
        return 0.5 * float(np.min(np.asarray(self.image.spacing, dtype=float)))

    def compute_angle(self, other):
        """Angle (degrees, 0-90) between this plane and another PlaneOI (or a normal vector).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> planes = mia.Data.image['CT 01'].planeoi
        >>> planes['Superior cut'].compute_angle(planes['Inferior cut'])
        24.6
        """
        normal = other.normal if isinstance(other, PlaneOI) else np.asarray(other, dtype=float)
        cosine = abs(np.dot(self.normal, normal)) / np.linalg.norm(normal)

        return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))

    def compute_display(self, grid, tolerance=None):
        """What to draw on a view, or None when there is nothing to draw.

        Parameters
        ----------
        grid : dict
            The view's grid (``display.get_grid``, or the 'grid' of any compute_array result on this image).
        tolerance : float, optional
            For a view parallel to the plane: largest distance (mm) at which it still counts as lying in it. None =
            half the image's smallest voxel spacing.

        Returns
        -------
        dict or None
            'kind' and 'pixel' ((col, row) floats, pixel centers at integers):

            * 'line'    : (2, 2) end points where the plane crosses the view, clipped to the view and to the extent.
            * 'polygon' : (4, 2) corners of the extent, for a view lying in the plane.
            * 'fill'    : (4, 2) corners of the whole view, for a view lying in an infinite plane.

            None when the plane misses the view (parallel and elsewhere, or the extent ends before the view).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> grid = image.display.get_grid('Coronal', image.display.position['Coronal'])
        >>> for plane in image.planeoi.values():
        ...     shown = plane.compute_display(grid)
        ...     if shown is not None and plane.visible:
        ...         draw(shown['pixel'], closed=shown['kind'] != 'line', color=plane.color)
        """
        tolerance = self._tolerance() if tolerance is None else float(tolerance)

        grid_origin = np.asarray(grid['origin'], dtype=float)
        x_axis = np.asarray(grid['x_axis'], dtype=float)
        y_axis = np.asarray(grid['y_axis'], dtype=float)
        normal = np.asarray(grid['normal'], dtype=float)
        normal = normal / np.linalg.norm(normal)
        sx, sy = float(grid['sx']), float(grid['sy'])
        h, w = grid['shape']

        cross = np.cross(self.normal, normal)
        cross_sq = np.dot(cross, cross)

        if cross_sq < 1e-10:                                      # parallel to the view
            if abs(np.dot(grid_origin - self.origin, self.normal)) > tolerance:
                return None

            if self.extent is None:
                pixel = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
                return {'kind': 'fill', 'pixel': pixel}

            d = self._corners(self.extent[0], self.extent[1]) - grid_origin
            return {'kind': 'polygon', 'pixel': np.column_stack((d @ x_axis / sx, d @ y_axis / sy))}

        # a point on both planes, and the direction of the line they share (same math as Display._intersection_line)
        d1, d2 = np.dot(self.normal, self.origin), np.dot(normal, grid_origin)
        point = (d1 * np.cross(normal, cross) + d2 * np.cross(cross, self.normal)) / cross_sq
        direction = cross / np.sqrt(cross_sq)

        offsets = [np.dot(point - grid_origin, x_axis) / sx, np.dot(point - grid_origin, y_axis) / sy]
        slopes = [np.dot(direction, x_axis) / sx, np.dot(direction, y_axis) / sy]
        lows = [-0.5, -0.5]
        highs = [w - 0.5, h - 0.5]
        if self.extent is not None:
            offsets += [np.dot(point - self.origin, self.x_axis), np.dot(point - self.origin, self.y_axis)]
            slopes += [np.dot(direction, self.x_axis), np.dot(direction, self.y_axis)]
            lows += [-0.5 * self.extent[0], -0.5 * self.extent[1]]
            highs += [0.5 * self.extent[0], 0.5 * self.extent[1]]

        interval = self._clip_interval(offsets, slopes, lows, highs)
        if interval is None:
            return None

        ends = np.array([point + interval[0] * direction, point + interval[1] * direction]) - grid_origin

        return {'kind': 'line', 'pixel': np.column_stack((ends @ x_axis / sx, ends @ y_axis / sy))}

    def compute_distance(self, points):
        """Signed distance (mm) of world points from the plane, positive on the side the normal points to.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> margin = image.planeoi['Superior cut'].compute_distance(image.roi['GTV'].mesh.points).min()
        """
        points = np.asarray(points, dtype=float)
        distance = (points.reshape(-1, 3) - self.origin) @ self.normal

        return float(distance[0]) if points.ndim == 1 else distance

    def compute_mesh(self, size=None):
        """The plane as a pyvista quad for the 3D view: its extent, or a size x size square for an infinite plane.

        Parameters
        ----------
        size : float, optional
            Side (mm) of the square for an infinite plane. None = the diagonal of the image's bounds.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> quad = mia.Data.image['CT 01'].planeoi['Superior cut'].compute_mesh()
        >>> plotter.add_mesh(quad, color='red', opacity=0.4)
        """
        if self.extent is not None:
            width, height = self.extent
        else:
            if size is None:
                bounds = np.asarray(self.image.get_bounds(), dtype=float)
                size = float(np.linalg.norm(bounds[1::2] - bounds[0::2]))
            width = height = float(size)

        return pv.PolyData(self._corners(width, height), faces=np.array([4, 0, 1, 2, 3]))

    def copy_to(self, image_name, rigid_name=None, name=None):
        """Copy the plane onto another image as a new, independent plane (no live link), and return it.

        Parameters
        ----------
        image_name : str or int
            Target image in ``Data.image`` (name or id).
        rigid_name : str or int, optional
            Registration in ``Data.rigid`` connecting the two frames of reference. None = same frame (no transform).
        name : str, optional
            Name on the target image. None = this plane's name.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> plane = mia.Data.image['MR 01'].planeoi['Margin plane']
        >>> on_ct = plane.copy_to('CT 01', rigid_name='CT 01_MR 01')
        >>> on_ct.cut_roi('Pelvis_L')
        """
        target = Data.image[image_name]
        matrix = self._registration_matrix(target, rigid_name)
        rotation = matrix[:3, :3]

        return target.add_planeoi(origin=rotation @ self.origin + matrix[:3, 3], normal=rotation @ self.normal,
                                  x_axis=rotation @ self.x_axis, name=self.name if name is None else name,
                                  extent=self.extent, keep_side=self.keep_side, group=self.group,
                                  color=list(self.color), visible=self.visible)

    def cut_mesh(self, mesh, cap=True):
        """The part of a mesh on the ``keep_side`` of the plane.

        Parameters
        ----------
        mesh : pyvista.PolyData
            Surface in this image's world (e.g. ``roi.mesh``).
        cap : bool, default True
            Close the cut with a flat cap, so a closed surface stays closed (needs a closed, manifold surface).
            False leaves the cut open.

        Returns
        -------
        pyvista.PolyData
            Empty when nothing of the mesh is on the kept side.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> kept = image.planeoi['Superior cut'].cut_mesh(image.roi['Pelvis_L'].mesh)
        """
        if self.keep_side is None:
            raise ValueError(f"'{self.name}' has no keep_side: set it to 1 (the side the normal points to) or -1 "
                             f"before cutting.")

        keep = self.keep_side * self.normal
        if cap:
            mesh = mesh.triangulate()

        distance = (np.asarray(mesh.points) - self.origin) @ keep
        if not np.any(distance > 0):
            return pv.PolyData()
        if not np.any(distance < 0):
            return mesh.copy()

        # Which side pyvista's clip filters keep differs between filters and versions, so check it and flip if needed
        result = self._clip_mesh(mesh, keep, cap)
        if result.n_points == 0 or np.mean((np.asarray(result.points) - self.origin) @ keep) < 0:
            result = self._clip_mesh(mesh, -keep, cap)

        return result

    def cut_roi(self, roi_name, new_name=None, color=None, cap=True):
        """Cut one of this image's ROIs with the plane and store the kept part as an ROI on the image.

        The result is mesh only (``update_mesh`` drops contours), like any ROI replaced by a mesh.

        Parameters
        ----------
        roi_name : str or int
            ROI on this image (name or id).
        new_name : str, optional
            ROI to write the result to; created when missing. None = '<roi>_<plane>'.
        color : list of int, optional
            Color of a new ROI. None = the source ROI's color.
        cap : bool, default True
            See :meth:`cut_mesh`.

        Returns
        -------
        Roi

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> kept = image.planeoi['Superior cut'].cut_roi('Pelvis_L', new_name='Pelvis_L_resected')
        >>> kept.volume / 1000          # mL
        """
        roi = self.image.roi[roi_name]
        if roi.mesh is None:
            raise ValueError(f"ROI '{roi.name}' has no mesh to cut.")

        new_name = f'{roi.name}_{self.name}' if new_name is None else new_name
        mesh = self.cut_mesh(roi.mesh, cap=cap)

        if new_name not in self.image.roi:
            self.image.add_roi(roi_name=new_name, color=roi.color if color is None else color, visible=True)
        self.image.roi[new_name].update_mesh(mesh)

        return self.image.roi[new_name]

    def flip(self):
        """Turn the normal around. keep_side flips with it, so the kept half of a cut stays the same.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].planeoi['Osteotomy'].flip()
        """
        self.normal = -self.normal
        self.y_axis = np.cross(self.normal, self.x_axis)
        if self.keep_side is not None:
            self.keep_side = -self.keep_side

    def go_to(self, view_plane='Axial'):
        """Turn the image's display so view_plane looks straight at the plane, centered on its origin.

        The other two planes turn with the shared rotation, so they show the plane edge-on (perpendicular cuts
        through it). Returns the new position of view_plane.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> position = mia.Data.image['CT 01'].planeoi['Superior cut'].go_to('Axial')
        """
        return self.image.display.go_to_plane(view_plane, self.origin, self.normal, x_axis=self.x_axis)

    def to_json(self):
        """State as one JSON string (one database row or one line in a file). Read it back with :meth:`from_json`.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> text = mia.Data.image['CT 01'].planeoi['Superior cut'].to_json()
        """
        return json.dumps(self._record(), separators=(',', ':'))

    def update_geometry(self, origin=None, normal=None, x_axis=None):
        """Move or turn the plane (e.g. dragging it in a view). Anything left as None is kept; a new normal keeps the
        current x_axis as far as possible.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> plane = mia.Data.image['CT 01'].planeoi['Superior cut']
        >>> plane.update_geometry(origin=plane.origin + 5 * plane.normal)        # 5 mm along the normal
        >>> plane.update_geometry(normal=[0.0, 0.2, 1.0])                        # tilt
        """
        if origin is not None:
            self.origin = np.asarray(origin, dtype=float).reshape(3)

        self.x_axis, self.y_axis, self.normal = self._orthonormal(self.normal if normal is None else normal,
                                                                  self.x_axis if x_axis is None else x_axis)