"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Permanent-Id / Editable-Name Registry
=====================================

Description:
    A dict-like container that stores every item under a permanent integer id
    and a user name. The id never changes and is never reused, so it is safe
    to use as a database key. The name can be changed at any time with
    `rename`.

Lookup Rule:
    * int (Python or numpy)  -> permanent id
    * str                    -> user name

Dict Compatibility:
    Iterating, `keys()`, `items()` and `in` work with names like the plain
    dictionaries they replace, so `for name in Data.image`,
    `list(image.roi.keys())` and `image.roi['Liver'] = Roi(...)` keep working.
    Use `ids()` to iterate by permanent id.

Examples
--------
>>> from medicalimageanalysis.registry import Registry
>>> class Item(object):
...     pass
>>> rois = Registry()
>>> roi_id = rois.add(Item(), 'Liver')
>>> rois['Liver'] is rois[roi_id]
True
>>> rois.rename('Liver', 'Liver_Whole')
>>> rois.get_id('Liver_Whole') == roi_id
True
"""

import numbers


class Registry(object):
    """
    Container that stores objects by permanent integer id and by editable name.

    The registry is the source of truth for names. When `name_attribute` is
    set, the current name is also mirrored onto each object (e.g. `roi.name`,
    `image.image_name`) so existing code can keep reading it. Always change a
    name with `rename` (or the Data-level rename methods); never assign the
    mirrored attribute directly.

    Parameters
    ----------
    name_attribute : str | None
        Attribute the name is mirrored onto. None leaves the object untouched.

    id_attribute : str | None
        Attribute the permanent id is written onto. None leaves the object
        untouched.

    id_source : callable | None
        Function `id_source(name, item) -> int | None` that supplies the id
        of a new item, e.g. from the application's database. Returning None
        falls back to `next_id`. Every image's ROI/POI registry uses it so the
        id comes from the session-wide `Data.roi` / `Data.poi` and is the same
        on every image. None always hands out `next_id`.

    start_id : int
        First id handed out when no id is given and there is no id_source.

    Attributes
    ----------
    next_id : int
        Next id to hand out. Save it with the patient and set it back after
        loading so deleted ids are never reused.

    Examples
    --------
    >>> rois = Registry()
    >>> liver_id = rois.add(Item(), 'Liver')
    >>> gtv_id = rois.add(Item(), 'GTV')
    >>> rois.ids()
    [1, 2]
    >>> rois.keys()
    ['Liver', 'GTV']

    Restore from the database with the saved ids:

    >>> rois = Registry()
    >>> rois.add(Item(), 'Liver', item_id=7)
    7
    >>> rois.next_id = 12   # saved value, so ids 8-11 stay retired
    """
    def __init__(self, name_attribute='name', id_attribute='id', id_source=None, start_id=1):
        self.name_attribute = name_attribute
        self.id_attribute = id_attribute
        self.id_source = id_source

        self.objects = {}
        self.id_to_name = {}
        self.name_to_id = {}
        self.next_id = start_id

    def __contains__(self, key):
        try:
            self._resolve(key)
        except (KeyError, TypeError):
            return False

        return True

    def __delitem__(self, key):
        self.remove(key)

    def __getitem__(self, key):
        return self.objects[self._resolve(key)]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.objects)

    def __setitem__(self, key, item):
        """
        A new name adds the item (new id); an existing id or name replaces the
        object and keeps its id and name.
        """
        if isinstance(key, str) and key not in self.name_to_id:
            self.add(item, key)
            return

        item_id = self._resolve(key)
        self.objects[item_id] = item
        self._tag(item, item_id, self.id_to_name[item_id])

    @staticmethod
    def _check_id(item_id):
        """
        Validates an explicit id and returns it as a Python int.
        """
        if isinstance(item_id, bool) or not isinstance(item_id, numbers.Integral):
            raise TypeError(f'Id must be an int, got {type(item_id).__name__}')

        return int(item_id)

    @staticmethod
    def _check_name(name):
        """
        Validates a user name.
        """
        if not isinstance(name, str):
            raise TypeError(f'Name must be a str, got {type(name).__name__}')

        if not name.strip():
            raise ValueError('Name cannot be empty')

        return name

    def _resolve(self, key):
        """
        Converts an id or a name into the permanent id.

        Parameters
        ----------
        key : int | str
            Permanent id or user name.

        Returns
        -------
        int
            Permanent id.
        """
        if isinstance(key, bool):
            raise TypeError('Key cannot be a bool')

        if isinstance(key, numbers.Integral):
            key = int(key)
            if key not in self.objects:
                raise KeyError(f'No item with id {key}')

            return key

        if isinstance(key, str):
            if key not in self.name_to_id:
                raise KeyError(f'No item named {key!r}')

            return self.name_to_id[key]

        raise TypeError(f'Key must be int (id) or str (name), got {type(key).__name__}')

    def _tag(self, item, item_id, name):
        """
        Writes the id and name onto the object, when those attributes are set.
        """
        if self.id_attribute is not None:
            setattr(item, self.id_attribute, item_id)

        if self.name_attribute is not None:
            setattr(item, self.name_attribute, name)

    def add(self, item, name, item_id=None):
        """
        Adds an item and returns its permanent id.

        Parameters
        ----------
        item : object
            Object to store.

        name : str
            User name, unique within this registry.

        item_id : int | None
            Explicit id, e.g. the database id. None asks `id_source`; if that
            is missing or returns None, `next_id` is handed out.

        Returns
        -------
        int
            Permanent id of the item.

        Examples
        --------
        >>> rois = Registry()
        >>> rois.add(Item(), 'Liver')
        1
        >>> rois.add(Item(), 'GTV', item_id=10)
        10
        >>> rois.add(Item(), 'Spleen')
        11
        """
        name = self._check_name(name)
        if name in self.name_to_id:
            raise KeyError(f'Name {name!r} is already in use')

        if item_id is None and self.id_source is not None:
            item_id = self.id_source(name, item)

        if item_id is None:
            item_id = self.next_id
        else:
            item_id = self._check_id(item_id)
            if item_id in self.objects:
                raise KeyError(f'Id {item_id} is already in use')

        self.objects[item_id] = item
        self.id_to_name[item_id] = name
        self.name_to_id[name] = item_id
        self.next_id = max(self.next_id, item_id + 1)
        self._tag(item, item_id, name)

        return item_id

    def change_id(self, key, new_id):
        """
        Re-keys an item under a new permanent id; its name and position are kept.

        Only for re-keying to an id from outside (e.g. the database assigned
        one after the item was made). Anything else that stores the old id
        must be updated too, so for ROIs, POIs, images and doses use the
        `Data.change_*_id` methods, which do that.

        Parameters
        ----------
        key : int | str
            Current id or name.

        new_id : int
            New id, not in use in this registry.

        Returns
        -------
        int
            The old id.

        Examples
        --------
        >>> rois.change_id('Liver', 17)
        1
        >>> rois['Liver'].id
        17
        """
        old_id = self._resolve(key)
        new_id = self._check_id(new_id)
        if new_id == old_id:
            return old_id

        if new_id in self.objects:
            raise KeyError(f'Id {new_id} is already in use')

        # rebuild instead of pop/insert so the item keeps its place in the order
        self.objects = {(new_id if k == old_id else k): v for k, v in self.objects.items()}
        self.id_to_name = {(new_id if k == old_id else k): v for k, v in self.id_to_name.items()}
        name = self.id_to_name[new_id]
        self.name_to_id[name] = new_id
        self.next_id = max(self.next_id, new_id + 1)
        self._tag(self.objects[new_id], new_id, name)

        return old_id

    def get(self, key, default=None):
        """
        Returns the item for an id or name, or `default` when it is missing.
        """
        try:
            return self.objects[self._resolve(key)]
        except KeyError:
            return default

    def get_id(self, key):
        """
        Returns the permanent id for an id or name.

        Examples
        --------
        >>> rois.get_id('Liver')
        1
        """
        return self._resolve(key)

    def get_name(self, key):
        """
        Returns the current user name for an id or name.

        Examples
        --------
        >>> rois.get_name(1)
        'Liver'
        """
        return self.id_to_name[self._resolve(key)]

    def ids(self):
        """
        Returns all permanent ids in insertion order.
        """
        return list(self.objects.keys())

    def items(self):
        """
        Returns (name, item) pairs in insertion order, like dict.items().
        """
        return [(self.id_to_name[item_id], item) for item_id, item in self.objects.items()]

    def keys(self):
        """
        Returns all current user names in insertion order, like dict.keys().
        """
        return [self.id_to_name[item_id] for item_id in self.objects]

    def remove(self, key):
        """
        Removes an item and returns it. Its id is retired and never reused.

        Examples
        --------
        >>> rois.remove('GTV')
        >>> 'GTV' in rois
        False
        """
        item_id = self._resolve(key)
        item = self.objects.pop(item_id)
        name = self.id_to_name.pop(item_id)
        del self.name_to_id[name]

        return item

    def rename(self, key, new_name):
        """
        Changes the user name of an item; the permanent id is untouched.

        Parameters
        ----------
        key : int | str
            Permanent id or current user name.

        new_name : str
            New user name, unique within this registry.

        Examples
        --------
        >>> rois.rename('Liver', 'Liver_Whole')
        >>> rois.get_name(1)
        'Liver_Whole'
        """
        item_id = self._resolve(key)
        new_name = self._check_name(new_name)
        old_name = self.id_to_name[item_id]

        if new_name == old_name:
            return

        if new_name in self.name_to_id:
            raise KeyError(f'Name {new_name!r} is already in use')

        del self.name_to_id[old_name]
        self.name_to_id[new_name] = item_id
        self.id_to_name[item_id] = new_name
        self._tag(self.objects[item_id], item_id, new_name)

    def values(self):
        """
        Returns all items in insertion order, like dict.values().
        """
        return list(self.objects.values())


if __name__ == '__main__':
    class Dummy(object):
        def __init__(self):
            self.id = None
            self.name = None

    rois = Registry()
    liver_id = rois.add(Dummy(), 'Liver')
    gtv_id = rois.add(Dummy(), 'GTV')

    rois.rename('Liver', 'Liver_Whole')
    assert rois[liver_id] is rois['Liver_Whole']
    assert rois[liver_id].name == 'Liver_Whole'
    assert 'Liver' not in rois

    del rois['GTV']
    rois['GTV'] = Dummy()
    assert rois.get_id('GTV') == 3

    replacement = Dummy()
    rois['Liver_Whole'] = replacement
    assert replacement.id == liver_id

    rois.change_id('Liver_Whole', 17)
    assert rois[17] is replacement and replacement.id == 17 and rois.keys() == ['Liver_Whole', 'GTV']
    assert rois.add(Dummy(), 'Spleen') == 18

    from_db = Registry(id_source=lambda name, item: {'Liver': 40}.get(name))
    assert from_db.add(Dummy(), 'Liver') == 40
    assert from_db.add(Dummy(), 'New ROI') == 41

    print(list(zip(rois.ids(), rois.keys())))