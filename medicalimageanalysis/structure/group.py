"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Groups: named, nested sets of one kind of item (ROIs, POIs, markups, planes or models), e.g. 'Liver' > 'Vessels',
    'Liver' > 'GTV' > GTV_1 ... GTV_n, 'Airways' > Point 1 ... 100. A group is an overlay: every member stays a normal
    item in its own registry (``image.roi['GTV_1']`` still works) and only carries a ``group_id``.

    Conventions
    -----------
    * One kind per group (``kind``: 'roi', 'poi', 'moi', 'planeoi' or 'model'), and each image holds one group
      container per kind (``image.roi_group``, ``image.poi_group``, ...), each with any number of nested groups.
    * Like an ROI, a group exists on every image (``Data.match_groups``), empty where that image has no members. The
      same path has the same permanent ``id`` on every image; ids are unique across every kind.
    * ``name`` is the group's own name, unique among its siblings. ``path`` is the registry key, written by the
      container: the names from the top down joined by '/' (``image.roi_group['Liver/Vessels']``). Names cannot
      contain '/'.
    * Two types of group:

      - folder (``main_id`` None): organizes independent items; nothing is rebuilt from anything.
      - composite (``main_id`` = an ROI id, ROI groups only): the members are parts of that main ROI (GTV_1 ... GTV_n
        of GTV). The main ROI is not a member of its own composite, and a composite holds no subgroups.

    * Same on every image (identity): kind, name, path, parent_id, main_id, method, method_version, settings. Per
      image: color, visible, match. ``Data`` methods change identity fields on every image at once.
    * ``method`` names the algorithm that makes or updates the group (e.g. 'components', 'nnunet_liver_segments'),
      with its ``method_version`` and keyword ``settings``. The package only stores it; the application runs it.
      None = manual.
    * ``match`` is this image's record of the last time the group was made or matched on it (``Data.record_group_match``):
      the image it was matched against, the registrations it went through, the method and a key of every member's
      geometry. ``Data.group_status`` compares it with the current data: 'manual', 'not run', 'current' or 'stale'.
    * Membership is stored on the member (``item.group_id``, None = ungrouped). An ROI's or POI's group is the same on
      every image; a markup, plane or model has its own.

Structure:
    - Group: identity, per-image display and match record, and a JSON record.

Examples
--------
>>> import medicalimageanalysis as mia
>>> from medicalimageanalysis import Data
>>> Data.add_group('roi', 'Liver/Vessels')
>>> Data.set_group('roi', ['Portal vein', 'Hepatic artery'], 'Liver/Vessels')
>>> mia.Data.image['CT 01'].roi_group['Liver/Vessels'].name
'Vessels'
>>> Data.add_group('roi', 'Liver/GTV', main='GTV')                           # composite: parts of GTV
>>> Data.set_group('roi', ['GTV_1', 'GTV_2'], 'Liver/GTV')
"""

import copy
import json

from .roi import random_color


class Group(object):
    """One group on one image. Made by ``Data.add_group`` (on every image at once) and stored in the image's container
    for its kind, which looks it up by permanent id or by path (``image.roi_group[12]``,
    ``image.roi_group['Liver/GTV']``).

    Parameters
    ----------
    kind : {'roi', 'poi', 'moi', 'planeoi', 'model'}
        Kind of item the group holds.
    name : str
        Own name, unique among its siblings.
    color : list of int, optional
        RGB, 0-255. Random if None.
    visible : bool, default True
        Per-image display setting for the group node (members keep their own ``visible``).
    parent_id : int, optional
        Id of the group this one is nested in. None = top level.
    main_id : int, optional
        ROI id of the main ROI of a composite. None = folder.

    Attributes
    ----------
    id : int or None
        Permanent id, written by the container. None until the group is added.
    path : str or None
        Registry key, written by the container: 'Liver/GTV'.
    method, method_version : str or None
        Algorithm that makes or updates the group, and its version. None = manual.
    settings : dict
        Keyword settings for ``method`` (stored data; the method is called as ``method(image, group, **settings)``).
    match : dict or None
        This image's match record (see ``Data.record_group_match``). None = never made or matched on this image.
    misc : dict
        Free space for the application.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> group = mia.Data.image['CT 01'].roi_group['Liver/GTV']
    >>> group.kind, group.name, group.path, group.main_id is not None
    ('roi', 'GTV', 'Liver/GTV', True)
    """
    SCHEMA_VERSION = 1

    def __init__(self, kind, name, color=None, visible=True, parent_id=None, main_id=None):
        self.id = None
        self.path = None
        self.kind = kind
        self.name = name
        self.color = list(random_color()) if color is None else list(color)
        self.visible = visible
        self.parent_id = parent_id
        self.main_id = main_id

        self.method = None
        self.method_version = None
        self.settings = {}

        self.match = None
        self.misc = {}

    # ---- Set ----

    def copy_identity(self, other):
        """Take the fields that are the same on every image (kind, name, parent, main, method) from another copy of
        the group. Color, visibility and the match record are left as they are.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> group.copy_identity(mia.Data.image['CT 01'].roi_group['Liver/GTV'])
        """
        self.kind = other.kind
        self.name = other.name
        self.parent_id = other.parent_id
        self.main_id = other.main_id
        self.method = other.method
        self.method_version = other.method_version
        self.settings = copy.deepcopy(other.settings)

    # ---- Load ----

    @classmethod
    def from_json(cls, text):
        """Rebuild a group from :meth:`to_json` output, not yet added to a container.

        ``id``, ``path``, ``parent_id`` and ``main_id`` hold the saved values; ``Image.groups_from_json`` adds the
        group and maps them onto the session's ids.

        Examples
        --------
        >>> from medicalimageanalysis.structure.group import Group
        >>> group = Group.from_json(text)
        """
        record = json.loads(text)
        schema = record.get('schema', 1)
        if schema > cls.SCHEMA_VERSION:
            raise ValueError(f"Record schema {schema} is newer than supported ({cls.SCHEMA_VERSION}).")

        group = cls(record['kind'], record['name'], color=record['color'], visible=record['visible'],
                    parent_id=record['parent_id'], main_id=record['main_id'])
        group.id = record.get('id')
        group.path = record.get('path')
        group.method = record.get('method')
        group.method_version = record.get('method_version')
        group.settings = dict(record.get('settings') or {})
        group.match = record.get('match')

        return group

    # ---- Save ----

    def _record(self):
        """State as plain JSON types."""
        return {'schema': self.SCHEMA_VERSION, 'id': self.id, 'path': self.path, 'kind': self.kind,
                'name': self.name, 'color': [int(c) for c in self.color], 'visible': bool(self.visible),
                'parent_id': self.parent_id, 'main_id': self.main_id, 'method': self.method,
                'method_version': self.method_version, 'settings': self.settings, 'match': self.match}

    def to_json(self):
        """State as one JSON string (one database row or one line in a file). Read it back with :meth:`from_json`.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> text = mia.Data.image['CT 01'].roi_group['Liver/GTV'].to_json()
        """
        return json.dumps(self._record(), separators=(',', ':'))