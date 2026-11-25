"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Groups: named sets of ROIs, POIs, markups, planes and models (e.g. 'Liver vessels' > 'Portal', the airway points,
    the planes of one resection). A group is an overlay: every member stays a normal, flat item in its own registry
    (``image.roi['GTV_1']`` still works) and only carries a ``group_id``. Nothing is grouped unless someone groups it.

    Conventions
    -----------
    * Groups live in ``Data.group`` (one registry for the whole session), with a permanent ``id`` and a user ``name``
      unique across the session. One group can hold items of every kind (the portal vein ROI and its bifurcation
      POIs).
    * Membership is stored on the member (``item.group_id``, None = ungrouped), never as a list on the group, so a
      deleted item never leaves a dangling id. ``Data.group_members`` finds the members.
    * Each item is in at most one group. Groups nest through ``parent_id`` (None = top level).
    * ROIs and POIs have the same group on every image, like their names (``Data.set_group`` sets it everywhere and
      ``Data.match_rois`` / ``match_pois`` copy it). Markups, planes and models are per image, so their group is set
      on the one item.
    * A group has no visibility of its own: a viewer shows it as checked when its members are visible.
    * Where geometry came from is not a group: an ROI made from another one (split into components, cut by a plane)
      records it in ``roi.source_id``.

Structure:
    - Group: name, color, parent and a JSON record.

Examples
--------
>>> import medicalimageanalysis as mia
>>> from medicalimageanalysis import Data
>>> vessels = Data.add_group('Liver vessels')
>>> portal = Data.add_group('Portal', parent='Liver vessels')
>>> Data.set_group('roi', 'Portal vein', 'Portal')
>>> Data.set_group('poi', [f'Bifurcation {n}' for n in range(1, 9)], 'Portal')
>>> Data.group_members('Liver vessels', 'poi')            # nested groups included
[12, 13, 14, 15, 16, 17, 18, 19]
"""

import json

from .roi import random_color


class Group(object):
    """One named set of items. Made by ``Data.add_group`` and stored in ``Data.group``, which looks it up by permanent
    id or by name (``Data.group[3]``, ``Data.group['Portal']``).

    Parameters
    ----------
    name : str, optional
        Replaced by ``Data.group`` when the group is added.
    color : list of int, optional
        RGB, 0-255, e.g. for the group's icon or for coloring its members. Random if None.
    parent_id : int, optional
        Id of the group this one is nested in. None = top level.

    Attributes
    ----------
    id : int or None
        Permanent id, written by ``Data.group``. None until the group is added.
    misc : dict
        Free space for the application.

    Examples
    --------
    >>> from medicalimageanalysis import Data
    >>> group = Data.add_group('Airways', color=[0, 160, 255])
    >>> group.id, group.name, group.parent_id
    (1, 'Airways', None)
    """
    SCHEMA_VERSION = 2

    def __init__(self, name=None, color=None, parent_id=None):
        self.id = None
        self.name = name
        self.color = list(random_color()) if color is None else list(color)
        self.parent_id = parent_id

        self.misc = {}

    @classmethod
    def from_json(cls, text):
        """Rebuild a group from :meth:`to_json` output, not yet added to ``Data.group``.

        ``id`` and ``parent_id`` hold the saved values; ``Data.add_saved_groups`` adds it and maps them onto the
        session's ids.

        Examples
        --------
        >>> from medicalimageanalysis.structure.group import Group
        >>> group = Group.from_json(text)
        """
        record = json.loads(text)
        schema = record.get('schema', 1)
        if schema > cls.SCHEMA_VERSION:
            raise ValueError(f"Record schema {schema} is newer than supported ({cls.SCHEMA_VERSION}).")

        group = cls(name=record['name'], color=record['color'], parent_id=record.get('parent_id'))
        group.id = record.get('id')

        return group

    def _record(self):
        """State as plain JSON types."""
        return {'schema': self.SCHEMA_VERSION, 'id': self.id, 'name': self.name,
                'color': [int(c) for c in self.color], 'parent_id': self.parent_id}

    def to_json(self):
        """State as one JSON string (one database row or one line in a file). Read it back with :meth:`from_json`.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> text = Data.group['Portal'].to_json()
        """
        return json.dumps(self._record(), separators=(',', ':'))
