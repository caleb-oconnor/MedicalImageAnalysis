"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Markups of interest (MOI): the lines, arrows, angles, polylines, polygons, ellipses and text a physician draws on
    an image to point something out or measure it. A markup is not an ROI (no inside, no mask, no DVH) and it belongs
    to one image only: it is never synced to other images (no match step) and never carried through a Rigid or
    Deformable registration.

    Conventions
    -----------
    * Points are world (x, y, z) in mm, like ``roi.contour_position``.
    * A markup keeps the view it was drawn on: ``view_plane`` ('Axial', 'Sagittal' or 'Coronal'), ``plane_origin`` (a
      world point on that view, its top-left corner when drawn) and the view's ``plane_x_axis``, ``plane_y_axis`` and
      ``plane_normal`` at that moment. A markup drawn on an oblique view stays oblique. Points are projected onto
      that plane whenever they are set, so float error from the clicks never takes them off the view.
    * A markup is shown only on views that lie in its plane (within half a slice by default, :meth:`Moi.is_on_view`).
      Other views hide it; :meth:`Moi.go_to` takes the image's display back to the view it was drawn on.
    * Measurements (:meth:`Moi.compute_measurement`) are in the image's own world: mm, mm^2 and degrees.
    * ``group_id`` puts the markup in a markup group (``image.moi_group``, None = ungrouped), e.g. every measurement
      of one lesion. Set it with ``group=`` here or ``Data.set_group('moi', ...)``.
    * ``id`` is permanent and unique across every image (``Data.new_moi_id``). ``name`` is unique on its image only,
      so two images can each have a 'Line 1'. Rename with ``Data.rename_moi`` (or ``image.moi.rename``); no other image
      has to follow.
    * Points per kind (``Moi.POINTS``): line 2, arrow 2 (tail, head), angle 3 (end, vertex, end), polyline 2+, polygon
      3+ (closed, first point not repeated), ellipse 3 (center, end of the first axis, end of the second axis; the
      second axis is made perpendicular to the first), text 1 (anchor).

Structure:
    - Moi: kind, points, drawing plane, display on a view grid, measurements and a JSON record.

Examples
--------
>>> import medicalimageanalysis as mia
>>> image = mia.Data.image['CT 01']
>>> position = image.display.position['Axial']
>>> ruler = image.add_moi('line', [[10.0, -40.0, z], [52.0, -40.0, z]], view_plane='Axial', position=position)
>>> ruler.name, ruler.compute_measurement()
('Line 1', ('Length', 42.0, 'mm'))

Points from clicks on a view (pixel col, row)

>>> grid = image.display.get_grid('Axial', position)
>>> points = mia.Moi.convert_pixel_to_position([[200, 180], [260, 230]], grid)
>>> arrow = image.add_moi('arrow', points, view_plane='Axial', position=position, text='Fracture')

Drawing it in a widget

>>> shown = arrow.compute_display(grid)          # None when the view is not on the arrow's slice
>>> shown['pixel'], shown['closed'], shown['text']

Back to the slice (and rotation) it was drawn on

>>> arrow.go_to()
"""

import json

import numpy as np

from ..data import Data


class Moi(object):
    """One markup of interest on one image.

    Made by the image (:meth:`Image.add_moi`) and stored in ``image.moi``, which looks it up by permanent id or by name
    (``image.moi[12]``, ``image.moi['Line 1']``).

    Parameters
    ----------
    image : Image
        Image the markup belongs to.
    kind : str
        'line', 'arrow', 'angle', 'polyline', 'polygon', 'ellipse' or 'text'.
    points : (N, 3) array_like
        World points (mm). See ``POINTS`` for how many each kind takes.
    view_plane : str, default 'Axial'
        Plane of the view it was drawn on.
    plane_origin, plane_x_axis, plane_y_axis, plane_normal : (3,) array_like
        That view's grid (``display.get_grid(view_plane, position)``): a point on it and its axes.
    name : str, optional
        Replaced by the image's registry when the markup is added.
    color : list of int, optional
        RGB, 0-255. Defaults to yellow.
    visible : bool, default True
        Show in the views.
    text : str, optional
        Label drawn with the markup (the whole content for kind 'text').
    group : str or int, optional
        Markup group (path or id) to put the markup in; a path is created when missing. None = ungrouped.

    Attributes
    ----------
    id : int or None
        Permanent id, written by ``image.moi``. None until the markup is added.
    group_id : int or None
        Markup group the markup is in (``image.moi_group``), None = ungrouped.
    misc : dict
        Free space for the application.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> image = mia.Data.image['CT 01']
    >>> grid = image.display.get_grid('Axial', image.display.position['Axial'])
    >>> moi = mia.Moi(image, 'angle', points, view_plane='Axial', plane_origin=grid['origin'],
    ...               plane_x_axis=grid['x_axis'], plane_y_axis=grid['y_axis'], plane_normal=grid['normal'])
    >>> moi.compute_angle()
    """
    # kind -> (fewest points, most points); None = no maximum
    POINTS = {'line': (2, 2), 'arrow': (2, 2), 'angle': (3, 3), 'polyline': (2, None), 'polygon': (3, None),
              'ellipse': (3, 3), 'text': (1, 1)}

    # view plane -> index of its normal in (x, y, z), same as Display.AXES[plane][2]
    NORMAL_INDEX = {'Axial': 2, 'Sagittal': 0, 'Coronal': 1}

    SCHEMA_VERSION = 1

    def __init__(self, image, kind, points, view_plane='Axial', plane_origin=None, plane_x_axis=None,
                 plane_y_axis=None, plane_normal=None, name=None, color=None, visible=True, text=None, group=None):
        if kind not in self.POINTS:
            raise ValueError(f"kind must be one of {list(self.POINTS)}; got {kind!r}")
        if view_plane not in self.NORMAL_INDEX:
            raise ValueError(f"view_plane must be one of {list(self.NORMAL_INDEX)}; got {view_plane!r}")
        if plane_origin is None or plane_x_axis is None or plane_y_axis is None or plane_normal is None:
            raise ValueError("A markup needs the plane of the view it is drawn on (plane_origin and its axes); "
                             "use Image.add_moi to take them from the display.")

        self.image = image

        self.id = None
        self.name = name
        self.kind = kind
        self.color = [255, 255, 0] if color is None else list(color)
        self.visible = visible
        self.text = text
        self.group_id = Data.resolve_group('moi', group)

        self.view_plane = view_plane
        self.plane_origin = np.asarray(plane_origin, dtype=float)
        self.plane_x_axis = self._unit(plane_x_axis)
        self.plane_y_axis = self._unit(plane_y_axis)
        self.plane_normal = self._unit(plane_normal)

        self.points = self._clean_points(points)

        self.misc = {}

    # ---- Compute ----

    @staticmethod
    def _ellipse_outline(center, axis_a, axis_b, samples=64):
        """(samples, 3) world points around the ellipse center + cos(t) axis_a + sin(t) axis_b."""
        t = np.linspace(0, 2 * np.pi, samples, endpoint=False)

        return center + np.outer(np.cos(t), axis_a) + np.outer(np.sin(t), axis_b)

    @staticmethod
    def _unit(vector):
        """Vector scaled to length 1."""
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            raise ValueError("Axis has zero length.")

        return vector / norm

    @staticmethod
    def convert_pixel_to_position(pixel, grid):
        """World points of (col, row) pixels on a view grid (pixel centers at integers), e.g. mouse clicks.

        Parameters
        ----------
        pixel : (N, 2) array_like
            (col, row) on the view.
        grid : dict
            The view's grid, from ``display.get_grid`` (or the 'grid' of a compute_array result).

        Returns
        -------
        (N, 3) ndarray

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> grid = mia.Data.image['CT 01'].display.get_grid('Axial', position)
        >>> mia.Moi.convert_pixel_to_position([[256, 256]], grid)
        """
        pixel = np.asarray(pixel, dtype=float).reshape(-1, 2)
        x_axis = np.asarray(grid['x_axis'], dtype=float)
        y_axis = np.asarray(grid['y_axis'], dtype=float)

        return (np.asarray(grid['origin'], dtype=float) + np.outer(pixel[:, 0] * grid['sx'], x_axis) +
                np.outer(pixel[:, 1] * grid['sy'], y_axis))

    def _clean_points(self, points):
        """(N, 3) float points checked against the kind's point count and projected onto the drawing plane."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)

        fewest, most = self.POINTS[self.kind]
        if len(points) < fewest or (most is not None and len(points) > most):
            count = str(fewest) if fewest == most else f"at least {fewest}"
            raise ValueError(f"A {self.kind} takes {count} points; got {len(points)}.")

        points = points - np.outer((points - self.plane_origin) @ self.plane_normal, self.plane_normal)

        if self.kind == 'ellipse':
            center = points[0]
            axis_a = points[1] - center
            if np.linalg.norm(axis_a) < 1e-9:
                raise ValueError("The ellipse's first axis has zero length.")

            axis_a_hat = axis_a / np.linalg.norm(axis_a)
            axis_b = points[2] - center
            axis_b = axis_b - np.dot(axis_b, axis_a_hat) * axis_a_hat       # perpendicular part
            if np.linalg.norm(axis_b) < 1e-9:
                raise ValueError("The ellipse's second axis has zero length (it lies along the first).")
            points[2] = center + axis_b

        return points

    def _in_plane(self, points):
        """(N, 2) mm coordinates of world points along the drawing plane's x and y axes."""
        d = np.asarray(points, dtype=float) - self.plane_origin

        return np.column_stack((d @ self.plane_x_axis, d @ self.plane_y_axis))

    def _outline(self):
        """World points to draw: the points themselves, or the sampled outline for an ellipse."""
        if self.kind == 'ellipse':
            center = self.points[0]
            return self._ellipse_outline(center, self.points[1] - center, self.points[2] - center)

        return self.points

    def _tolerance(self):
        """Default distance (mm) a view may be from the drawing plane and still show the markup: half a slice."""
        return 0.5 * float(self.image.display.spacing[self.NORMAL_INDEX[self.view_plane]])

    def compute_angle(self):
        """Angle (degrees) at the vertex of an 'angle' markup (points: end, vertex, end); None for other kinds.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].moi['Angle 1'].compute_angle()
        37.4
        """
        if self.kind != 'angle':
            return None

        a = self.points[0] - self.points[1]
        b = self.points[2] - self.points[1]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def compute_area(self):
        """Area (mm^2) of a 'polygon' or 'ellipse' markup; None for other kinds.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].moi['Ellipse 1'].compute_area()
        """
        if self.kind == 'polygon':
            uv = self._in_plane(self.points)
            u, v = uv[:, 0], uv[:, 1]
            return float(0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))

        if self.kind == 'ellipse':
            a = np.linalg.norm(self.points[1] - self.points[0])
            b = np.linalg.norm(self.points[2] - self.points[0])
            return float(np.pi * a * b)

        return None

    def compute_display(self, grid, tolerance=None):
        """What to draw on a view, or None when the view does not lie in the markup's plane.

        Parameters
        ----------
        grid : dict
            The view's grid (``display.get_grid``, or the 'grid' of any compute_array result on this image).
        tolerance : float, optional
            Largest distance (mm) between the view and the drawing plane. None = half a slice.

        Returns
        -------
        dict or None
            'kind', 'pixel' ((N, 2) (col, row) floats, pixel centers at integers; an ellipse comes as its sampled
            outline), 'closed' (True for polygon and ellipse: join the last point to the first) and 'text'. For an
            arrow the head is the last point.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> grid = image.display.get_grid('Axial', image.display.position['Axial'])
        >>> for moi in image.moi.values():
        ...     shown = moi.compute_display(grid)
        ...     if shown is not None and moi.visible:
        ...         draw(shown['pixel'], closed=shown['closed'], color=moi.color)
        """
        if not self.is_on_view(grid, tolerance=tolerance):
            return None

        d = self._outline() - np.asarray(grid['origin'], dtype=float)
        pixel = np.column_stack((d @ np.asarray(grid['x_axis'], dtype=float) / grid['sx'],
                                 d @ np.asarray(grid['y_axis'], dtype=float) / grid['sy']))

        return {'kind': self.kind, 'pixel': pixel, 'closed': self.kind in ('polygon', 'ellipse'), 'text': self.text}

    def compute_length(self):
        """Length (mm): line or arrow length, polyline length, polygon or ellipse perimeter. None for angle and text.

        The ellipse perimeter uses Ramanujan's approximation (well under 0.01 % error for clinical shapes).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].moi['Line 1'].compute_length()
        42.0
        """
        if self.kind in ('line', 'arrow', 'polyline'):
            return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

        if self.kind == 'polygon':
            closed = np.vstack((self.points, self.points[:1]))
            return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())

        if self.kind == 'ellipse':
            a = np.linalg.norm(self.points[1] - self.points[0])
            b = np.linalg.norm(self.points[2] - self.points[0])
            return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))

        return None

    def compute_measurement(self):
        """The one number a viewer shows next to the markup: (label, value, unit), or None for text.

        line / arrow / polyline -> ('Length', mm, 'mm'); polygon / ellipse -> ('Area', mm^2, 'mm2');
        angle -> ('Angle', degrees, 'deg').

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> label, value, unit = mia.Data.image['CT 01'].moi['Line 1'].compute_measurement()
        >>> f'{label}: {value:.1f} {unit}'
        'Length: 42.0 mm'
        """
        if self.kind in ('line', 'arrow', 'polyline'):
            return 'Length', self.compute_length(), 'mm'
        if self.kind in ('polygon', 'ellipse'):
            return 'Area', self.compute_area(), 'mm2'
        if self.kind == 'angle':
            return 'Angle', self.compute_angle(), 'deg'

        return None

    def is_on_view(self, grid, tolerance=None):
        """True when the view is parallel to the drawing plane and within tolerance (mm, None = half a slice) of it.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> image.moi['Line 1'].is_on_view(image.display.get_grid('Axial', image.display.position['Axial']))
        True
        """
        tolerance = self._tolerance() if tolerance is None else float(tolerance)

        normal = self._unit(grid['normal'])
        if abs(np.dot(normal, self.plane_normal)) < 1 - 1e-5:
            return False

        distance = abs(np.dot(np.asarray(grid['origin'], dtype=float) - self.plane_origin, self.plane_normal))

        return bool(distance <= tolerance)

    # ---- Set ----

    def update_points(self, points):
        """Replace the points (e.g. after dragging a handle); checked and projected onto the drawing plane again.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> moi = mia.Data.image['CT 01'].moi['Line 1']
        >>> moi.update_points([moi.points[0], new_end])
        """
        self.points = self._clean_points(points)

    # ---- Load ----

    @classmethod
    def from_json(cls, image, text):
        """Rebuild a markup from :meth:`to_json` output, not yet added to the image.

        ``id`` and ``name`` hold the saved values; :meth:`Image.markups_from_json` adds it with them (or with a new id
        when the saved one is taken). ``group_id`` holds the saved value; :meth:`Image.markups_from_json` maps it onto
        the session's group.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> image = mia.Data.image['CT 01']
        >>> moi = mia.Moi.from_json(image, text)
        >>> image.moi.add(moi, moi.name)
        """
        record = json.loads(text)
        schema = record.get('schema', 1)
        if schema > cls.SCHEMA_VERSION:
            raise ValueError(f"Record schema {schema} is newer than supported ({cls.SCHEMA_VERSION}).")

        moi = cls(image, record['kind'], record['points'], view_plane=record['view_plane'],
                  plane_origin=record['plane_origin'], plane_x_axis=record['plane_x_axis'],
                  plane_y_axis=record['plane_y_axis'], plane_normal=record['plane_normal'], name=record['name'],
                  color=record['color'], visible=record['visible'], text=record['text'])
        moi.id = record.get('id')
        moi.group_id = record.get('group_id')

        return moi

    # ---- Save ----

    def _record(self):
        """State as plain JSON types."""
        return {'schema': self.SCHEMA_VERSION, 'id': self.id, 'name': self.name,
                'image_id': getattr(self.image, 'id', None), 'kind': self.kind,
                'points': self.points.tolist(), 'view_plane': self.view_plane,
                'plane_origin': self.plane_origin.tolist(), 'plane_x_axis': self.plane_x_axis.tolist(),
                'plane_y_axis': self.plane_y_axis.tolist(), 'plane_normal': self.plane_normal.tolist(),
                'color': [int(c) for c in self.color], 'visible': bool(self.visible), 'text': self.text,
                'group_id': self.group_id}

    def to_json(self):
        """State as one JSON string (one database row or one line in a file). Read it back with :meth:`from_json`.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> text = mia.Data.image['CT 01'].moi['Line 1'].to_json()
        """
        return json.dumps(self._record(), separators=(',', ':'))

    # ---- Other ----

    def go_to(self):
        """Turn the image's display back to the view the markup was drawn on, centered on it.

        Rotates the shared view (so an oblique markup comes back oblique), scrolls every plane through the markup's
        center and puts the crosshair there. Returns the new position of ``view_plane``.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> position = mia.Data.image['CT 01'].moi['Line 1'].go_to()
        """
        return self.image.display.go_to_plane(self.view_plane, self.points.mean(axis=0), self.plane_normal,
                                              x_axis=self.plane_x_axis)