"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Data Management Module for Medical Imaging and Transformations
==============================================================

Description:
    This module provides a centralized, global state container for medical
    imaging data, including volumetric images, dose distributions, and
    spatial transformations (rigid and deformable).

Stored Data:
    * Images (CT, MR, PT, US, XRay, MHD, NIfTI, ...)
    * Rigid registrations
    * Deformable registrations (DVFs)
    * Dose distributions (RTDOSE)
    * Markups, planes and models of interest (kept on their images)

Key Functionalities:
    1. **Global Registry**: Class-level `Registry` containers shared by every
       module that imports `Data`, so no explicit object passing is required.
       Each entry has a permanent integer id (for the database) and a user
       name that can be changed; either one looks it up.
    2. **Listing**: Convenience methods (`image_list`, `roi_list`, ...) that
       return the names currently held in the registry.
    3. **Annotation Sync**: ROIs and POIs only exist on images, so the
       images are the only record of them; `Data` keeps no separate table.
       A name has one id on every image: a name already on any image keeps
       that id ('Liver' is the same id on the CT and the MR). `get_roi_id`,
       `get_roi_name`, `roi_ids` and `roi_list` read it from the images.
       `match_rois` / `match_pois` give every image an
       entry for every ROI/POI. Color and visibility stay per image; a
       newly added empty ROI/POI copies them from another image. Doses
       live in `dose`, so they are never given ROIs or POIs.
    4. **Rename**: `rename_roi`, `rename_poi` and `rename_image` change the
       user name everywhere at once; the id never changes.
    5. **Lookup / Removal**: `volume` returns an image or dose by name for
       display code (Rigid, Deformable); `delete` removes an entry from any
       registry, `delete_roi` / `delete_poi` remove an ROI/POI from every
       image.
    6. **Markups and Planes**: markups of interest (``image.moi``: lines, arrows, angles, ...) and planes of interest
       (``image.planeoi``: cut and reference planes) belong to one image and are never synced or matched. `Data`
       only keeps their ids unique across every image (`new_moi_id`, `new_planeoi_id`) and finds, renames, re-keys
       and deletes them by id (`find_moi`, `rename_moi`, `change_moi_id`, `delete_moi`, and the same for planes).
       Their names are unique per image only, so a name needs the image (``image_name=``).
       Models (``image.model``: clustered / tetrahedral meshes derived from ROIs) follow the same rules: `find_model`,
       `rename_model`, `change_model_id`, `delete_model`.
    7. **Groups**: named, nested sets of one kind of item ('Liver' > 'Vessels',
       'Liver' > 'GTV' > GTV_1 ... GTV_n). Each image has one container per
       kind (``image.roi_group``, ``poi_group``, ``moi_group``,
       ``planeoi_group``, ``model_group``) and, like ROIs, every group exists
       on every image (`match_groups`), empty where it has no members. A
       group is an overlay: members stay normal items and only carry
       ``item.group_id``, so code that ignores groups is unaffected. Groups are
       looked up by path ('Liver/GTV') or id. A folder group organizes; a
       composite (ROIs only) names the main ROI its members are parts of.
       `add_group`, `set_group`, `group_members`, `ungrouped_members`,
       `rename_group`, `set_group_parent`, `delete_group`, `set_group_main`.
    8. **Group methods and status**: a group can name the algorithm that makes
       it (`set_group_method`). `record_group_match` stores, per image, what a
       run used (image matched against, registrations, member geometry), and
       `group_status` reports 'manual', 'not run', 'current' or 'stale'.
       `find_registration_path` and `map_points` carry points between frames
       of reference through a chain of Rigid / Deformable registrations.

Id Note:
    Without help, new ids come from counters (1, 2, 3, ...) that only go
    up, so a deleted ROI's number is not reused in the session.
    The package doesn't know the application's database or numbering
    convention, so the application decides ids in one of two ways:

    * **Provider (recommended):** set `Data.roi_id_provider` (and
      `volume_id_provider`, `poi_id_provider`, `rigid_id_provider`,
      `deformable_id_provider`, `moi_id_provider`, `planeoi_id_provider`,
      `model_id_provider`, `group_id_provider`) to a function `provider(name, item)` that
      returns the database id: the existing one when the name / series is
      already known, otherwise a newly created one. Returning None falls back
      to the counter. Every new ROI anywhere (RTSTRUCT, mask, drawing,
      match_rois) goes through it, once per name.
    * **Re-key afterwards:** `Data.change_roi_id`, `change_poi_id` and
      `change_volume_id` move an item to the database's id and update every
      image, Rigid and Deformable that stores the old one.

Architecture Note:
    The module utilizes a Singleton-style pattern via class-level attributes
    in the `Data` class. Data loaded in one part of the application (e.g. a
    reader) is immediately available to other components (e.g. a
    visualization or registration module).

Warning:
    Because this class uses global state, calling `Data.clear()` will affect
    all modules currently importing this class.

Usage:
    >>> import medicalimageanalysis as mia
    >>> from medicalimageanalysis import Data
    >>> # Load a patient
    >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
    >>> # Inspect what was loaded
    >>> Data.image_list()

Examples
--------
>>> import medicalimageanalysis as mia
>>> from medicalimageanalysis import Data
>>> Data.clear()
>>> Data.image_list()
[]
>>> mia.read_dicoms(folder_path='C:/Data/Patient01')
>>> Data.image_list()
['CT 01', 'MR 01']
>>> ct = Data.image['CT 01']
>>> list(ct.roi.keys())
['Liver', 'GTV']
>>> Data.get_roi_id('Liver')
1
>>> Data.rename_roi('Liver', 'Liver_Whole')
>>> ct.roi[1].name
'Liver_Whole'
"""

import hashlib
import weakref

import numpy as np

from .registry import Registry


class Data(object):
    """
    Global orchestrator for medical imaging data and transformations.

    This class acts as a centralized, static repository (Singleton pattern)
    for managing volumetric data, spatial transformations, and clinical
    annotations. Because data is stored at the class level, all
    modifications are reflected globally across the application.

    Every attribute is a `Registry`: look entries up by permanent id (int)
    or user name (str). Iterating and `keys()` give names, like the plain
    dictionaries they replace; `ids()` gives the permanent ids.

    Attributes
    ----------
    image : Registry
        Images; the name is mirrored onto `image.image_name`.

    rigid : Registry
        Rigid registration objects; the name is mirrored onto
        `rigid.rigid_name`.

    deformable : Registry
        Deformable registration objects; the name is mirrored onto
        `deformable.deformable_name`.

    dose : Registry
        Dose distribution objects; the name is mirrored onto
        `dose.image_name`. Image and dose ids share one counter, so a
        volume id is unique across both.

    roi_next_id, poi_next_id : int
        Fallback counters for new ROI / POI ids (no provider). ROIs and POIs
        themselves live only on images (``image.roi`` / ``image.poi``).

    moi_next_id, planeoi_next_id, model_next_id, group_next_id : int
        Fallback counters for new markup / plane / model / group ids (no provider). They themselves live only on
        images (``image.moi`` / ``image.planeoi`` / ``image.model`` / ``image.roi_group``, ...).

    GROUP_KINDS : tuple of str
        Kinds of item a group can hold; each image has a container ``image.<kind>_group`` for each.

    Examples
    --------
    Access the registry from any module without passing objects around:

    >>> import medicalimageanalysis as mia
    >>> from medicalimageanalysis import Data
    >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
    >>> Data.image_list()
    ['CT 01', 'MR 01']

    Grab an image by name or by id (always z-y-x):

    >>> ct = Data.image['CT 01']
    >>> ct is Data.image[ct.id]
    True
    >>> ct.array.shape
    (142, 512, 512)

    The same ROI id on every image:

    >>> Data.image['CT 01'].roi['GTV'].id == Data.image['MR 01'].roi['GTV'].id
    True
    """

    # Id providers: functions the application sets so new ids come from its database instead of the counters. Each
    # is called as provider(name, item) and returns an int, or None to fall back to the counter. Not reset by clear().
    volume_id_provider = None        # images and doses; item is the Image / Dose (series_uid, mrn, frame_ref, ...)
    roi_id_provider = None           # item is None: an ROI's identity is its name
    poi_id_provider = None           # item is None
    rigid_id_provider = None         # item is the Rigid
    deformable_id_provider = None    # item is the Deformable
    moi_id_provider = None           # item is the Moi (its image, kind, ... are set)
    planeoi_id_provider = None      # item is the PlaneOI
    model_id_provider = None         # item is the Model (its image, roi_ids, recipe are set)
    group_id_provider = None         # item is the Group (its kind, name, parent_id are set); name is its path

    # Kinds of item a group can hold: each image has a container '<kind>_group' for each
    GROUP_KINDS = ('roi', 'poi', 'moi', 'planeoi', 'model')

    # Fallback counters for new ROI / POI ids when there is no provider (or it returns None). They only go up, so a
    # number is never handed out twice in a session. Reset by clear().
    roi_next_id = 1
    poi_next_id = 1
    moi_next_id = 1
    planeoi_next_id = 1
    model_next_id = 1
    group_next_id = 1

    # Registries, built by clear() (run once at import, below the class). Images and doses draw ids from one counter,
    # so a volume id is unique across both and Data.volume(id) is never ambiguous.
    image = None
    rigid = None
    deformable = None
    dose = None

    # Fingerprints of item geometry / registrations for group_status, kept until the object they were taken from is
    # replaced (contours, mesh, DVF, ...). Weak keys, so a deleted item drops its entry.
    _key_cache = weakref.WeakKeyDictionary()

    # ---- Registry ----

    @staticmethod
    def _provided_id(provider, name, item):
        """The provider's id for a new item, or None (counter) when there is no provider or it returns None."""
        if provider is None:
            return None

        return provider(name, item)

    @classmethod
    def _annotation_name(cls, kind, key):
        """Current name of an ROI/POI (name or id) on the loaded images; KeyError when no image has it."""
        for image in cls.image.values():
            registry = getattr(image, kind)
            if key in registry:
                return registry.get_name(key)

        raise KeyError(f"No {kind.upper()} {key!r} on any loaded image.")

    @classmethod
    def _annotation_pairs(cls, kind):
        """(id, name) of every ROI (kind='roi') or POI (kind='poi') on the loaded images, in first-seen order, each id
        once. The images are the only record: nothing is kept for a name no image has."""
        pairs = []
        seen = set()
        for image in cls.image.values():
            registry = getattr(image, kind)
            for annotation_id, name in zip(registry.ids(), registry.keys()):
                if annotation_id not in seen:
                    seen.add(annotation_id)
                    pairs.append((annotation_id, name))

        return pairs

    @classmethod
    def _change_annotation_id(cls, kind, key, new_id):
        """Re-key an ROI/POI on every image. Returns the old id, or None when new_id is already its id."""
        old_id = cls._resolve_annotation(kind, key)
        new_id = Registry._check_id(new_id)
        if new_id == old_id:
            return None

        if cls._has_annotation(kind, new_id):
            raise KeyError(f"{kind.upper()} id {new_id} is already used by "
                           f"{cls._annotation_name(kind, new_id)!r}.")

        for image in cls.image.values():
            registry = getattr(image, kind)
            if old_id in registry:
                registry.change_id(old_id, new_id)

        counter = kind + '_next_id'
        setattr(cls, counter, max(getattr(cls, counter), new_id + 1))

        return old_id

    @classmethod
    def _change_markup_id(cls, kind, key, new_id, image_name=None):
        """Re-key a markup (kind='moi') or plane (kind='planeoi') on its image. Returns the old id, or None when new_id
        is already its id."""
        item = cls._find_markup(kind, key, image_name)
        new_id = Registry._check_id(new_id)
        if new_id == item.id:
            return None

        if new_id in cls._markup_ids(kind):
            raise KeyError(f"{kind.upper()} id {new_id} is already in use.")

        old_id = getattr(item.image, kind).change_id(item.id, new_id)

        counter = kind + '_next_id'
        setattr(cls, counter, max(getattr(cls, counter), new_id + 1))

        return old_id

    @classmethod
    def _check_annotations(cls, kind):
        """Raise ValueError if one ROI/POI name has two ids across the images, or one id two names.

        Can only happen when ids were set by hand past `Data` (e.g. ``image.roi.add(..., item_id=...)``) or saved
        images disagree; the loaders and every `Data` method keep one id per name.
        """
        ids_of_name = {}
        names_of_id = {}
        for image in cls.image.values():
            registry = getattr(image, kind)
            for annotation_id, name in zip(registry.ids(), registry.keys()):
                ids_of_name.setdefault(name, set()).add(annotation_id)
                names_of_id.setdefault(annotation_id, set()).add(name)

        problems = [f"{name!r} has ids {sorted(ids)}" for name, ids in ids_of_name.items() if len(ids) > 1]
        problems += [f"id {annotation_id} is named {sorted(names)}" for annotation_id, names in names_of_id.items()
                     if len(names) > 1]
        if problems:
            raise ValueError(f"{kind.upper()} ids disagree between images: " + "; ".join(problems) +
                             f". Fix with Data.change_{kind}_id or Data.rename_{kind} on the image that is wrong.")

    @classmethod
    def _delete_annotation(cls, kind, key):
        """Remove an ROI/POI from every image. Its id is retired: the counter never hands it out again."""
        annotation_id = cls._resolve_annotation(kind, key)
        for image in cls.image.values():
            registry = getattr(image, kind)
            if annotation_id in registry:
                registry.remove(annotation_id)

        return annotation_id

    @classmethod
    def _delete_markup(cls, kind, key, image_name=None):
        """Remove a markup / plane from its image and return it. Its id is retired: the counter never hands it out
        again."""
        item = cls._find_markup(kind, key, image_name)

        return getattr(item.image, kind).remove(item.id)

    @classmethod
    def _find_markup(cls, kind, key, image_name=None):
        """The markup / plane with this id on any loaded image, or with this name (or id) on image_name.

        Names are unique per image only, so a name needs image_name.
        """
        if image_name is not None:
            return getattr(cls.image[image_name], kind)[key]

        if isinstance(key, str):
            raise ValueError(f"{kind.upper()} names are only unique on one image; pass image_name= to find "
                             f"{key!r}, or use its id.")

        for image in cls.image.values():
            registry = getattr(image, kind)
            if key in registry:
                return registry[key]

        raise KeyError(f"No {kind.upper()} with id {key!r} on any loaded image.")

    @classmethod
    def _has_annotation(cls, kind, key):
        """True when some loaded image has the ROI/POI (name or id)."""
        return any(key in getattr(image, kind) for image in cls.image.values())

    @classmethod
    def _markup_ids(cls, kind):
        """Ids of every markup / plane on the loaded images, image by image."""
        return [item_id for image in cls.image.values() for item_id in getattr(image, kind).ids()]

    @classmethod
    def _match_annotations(cls, kind):
        """Give every image an entry for every ROI/POI on any image, under the same id (see match_rois)."""
        cls._check_annotations(kind)
        images = cls.image.values()

        cls.match_groups()

        for annotation_id, name in cls._annotation_pairs(kind):
            color, visible, group_id = None, False, None
            for image in images:
                annotation = getattr(image, kind).get(annotation_id)
                if annotation is None:
                    continue
                if color is None and annotation.color is not None:
                    color, visible = annotation.color, annotation.visible
                if group_id is None:
                    group_id = getattr(annotation, 'group_id', None)

            if color is None:
                color = [128, 128, 128]

            for image in images:
                if annotation_id not in getattr(image, kind):
                    if kind == 'roi':
                        image.add_roi(roi_name=name, color=color, visible=visible, match=False)
                    else:
                        image.add_poi(poi_name=name, color=color, visible=visible, match=False)
                    getattr(image, kind)[annotation_id].group_id = group_id

    @classmethod
    def _new_annotation_id(cls, kind, name):
        """Id for an ROI/POI name being added to an image.

        A name already on a loaded image keeps that id. A new name asks the provider (``roi_id_provider`` /
        ``poi_id_provider``); if there is none or it returns None, the counter (``roi_next_id`` / ``poi_next_id``)
        gives a number past every id in use. The counter never goes down, so a number is not handed out twice in a
        session.
        """
        for image in cls.image.values():
            registry = getattr(image, kind)
            if name in registry:
                return registry.get_id(name)

        used = {annotation_id for annotation_id, _ in cls._annotation_pairs(kind)}
        counter = kind + '_next_id'

        annotation_id = cls._provided_id(getattr(cls, kind + '_id_provider'), name, None)
        if annotation_id is None:
            annotation_id = max([getattr(cls, counter)] + [i + 1 for i in used])
        else:
            annotation_id = Registry._check_id(annotation_id)
            if annotation_id in used:
                raise KeyError(f"{kind}_id_provider gave id {annotation_id} for {name!r}, but it is already used by "
                               f"{cls._annotation_name(kind, annotation_id)!r}.")

        setattr(cls, counter, max(getattr(cls, counter), annotation_id + 1))

        return annotation_id

    @classmethod
    def _new_markup_id(cls, kind, name, item):
        """Id for a markup / plane being added to an image: the provider's (``moi_id_provider`` /
        ``planeoi_id_provider``), else the counter past every id in use. No name matching: two markups with the
        same name on different images get different ids."""
        used = set(cls._markup_ids(kind))
        counter = kind + '_next_id'

        item_id = cls._provided_id(getattr(cls, kind + '_id_provider'), name, item)
        if item_id is None:
            item_id = max([getattr(cls, counter)] + [i + 1 for i in used])
        else:
            item_id = Registry._check_id(item_id)
            if item_id in used:
                raise KeyError(f"{kind}_id_provider gave id {item_id} for {name!r}, but it is already in use.")

        setattr(cls, counter, max(getattr(cls, counter), item_id + 1))

        return item_id

    @classmethod
    def _rename_annotation(cls, kind, key, new_name):
        """Rename an ROI/POI on every image that has it; checked before anything changes."""
        annotation_id = cls._resolve_annotation(kind, key)
        new_name = Registry._check_name(new_name)

        if cls._has_annotation(kind, new_name) and cls._resolve_annotation(kind, new_name) != annotation_id:
            raise KeyError(f"{kind.upper()} name {new_name!r} is already in use.")

        for image in cls.image.values():
            registry = getattr(image, kind)
            if annotation_id in registry:
                registry.rename(annotation_id, new_name)

    @classmethod
    def _rename_markup(cls, kind, key, new_name, image_name=None):
        """Rename a markup / plane on its image (names only need to be unique there)."""
        item = cls._find_markup(kind, key, image_name)
        getattr(item.image, kind).rename(item.id, new_name)

    @classmethod
    def _resolve_annotation(cls, kind, key):
        """Id of an ROI/POI (name or id) on the loaded images; KeyError when no image has it."""
        for image in cls.image.values():
            registry = getattr(image, kind)
            if key in registry:
                return registry.get_id(key)

        raise KeyError(f"No {kind.upper()} {key!r} on any loaded image.")

    @classmethod
    def change_model_id(cls, key, new_id, image_name=None) -> None:
        """
        Give a model a new permanent id (e.g. the database's). The name is unchanged. Every Deformable that refers to
        the model is updated too.

        Parameters
        ----------
        key : int | str
            Current id, or name with `image_name`.

        new_id : int
            New id, not used by another model on any image.

        image_name : str | int, optional
            Image holding the model; needed when `key` is a name.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.change_model_id('Liver FE', 3301, image_name='CT 01')
        """
        old_id = cls._change_markup_id('model', key, new_id, image_name)
        if old_id is None:
            return
        new_id = Registry._check_id(new_id)

        for deform in cls.deformable.values():
            for side in (deform.reference, deform.moving):
                if side.model_id == old_id:
                    side.model_id = new_id

    @classmethod
    def change_moi_id(cls, key, new_id, image_name=None) -> None:
        """
        Give a markup of interest a new permanent id (e.g. the database's). The name is unchanged.

        Parameters
        ----------
        key : int | str
            Current id, or name with `image_name`.

        new_id : int
            New id, not used by another markup on any image.

        image_name : str | int, optional
            Image holding the markup; needed when `key` is a name.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.change_moi_id('Line 1', 5521, image_name='CT 01')
        >>> Data.image['CT 01'].moi['Line 1'].id
        5521
        """
        cls._change_markup_id('moi', key, new_id, image_name)

    @classmethod
    def change_planeoi_id(cls, key, new_id, image_name=None) -> None:
        """
        Give a plane of interest a new permanent id (e.g. the database's). The name is unchanged.

        Parameters
        ----------
        key : int | str
            Current id, or name with `image_name`.

        new_id : int
            New id, not used by another plane on any image.

        image_name : str | int, optional
            Image holding the plane; needed when `key` is a name.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.change_planeoi_id('Superior cut', 812, image_name='CT 01')
        """
        cls._change_markup_id('planeoi', key, new_id, image_name)

    @classmethod
    def change_poi_id(cls, key, new_id) -> None:
        """
        Give a POI a new permanent id everywhere it is stored. The name is unchanged.

        Updates every image's POI. Use it to adopt the database's id for a POI made before the database knew it.

        Parameters
        ----------
        key : str | int
            POI name or current id.

        new_id : int
            New id, not used by another POI.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If no image has `key`, or `new_id` is used by another POI. Nothing is changed in that case.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.change_poi_id('Iso', 230)
        >>> Data.image['CT 01'].poi['Iso'].id
        230
        """
        old_id = cls._change_annotation_id('poi', key, new_id)
        if old_id is None:
            return
        new_id = Registry._check_id(new_id)

        for image in cls.image.values():
            for group in image.poi_group.values():
                if group.match is not None:
                    group.match['members'] = [[new_id if m == old_id else m, k] for m, k in group.match['members']]

    @classmethod
    def change_roi_id(cls, key, new_id) -> None:
        """
        Give an ROI a new permanent id everywhere it is stored. The name is unchanged.

        Updates every image's ROI, every Rigid's ROI pairs, every Deformable's guiding ROIs and warped meshes, every
        Model's source ROIs and every composite group it is the main ROI of. Use it to adopt the database's id for an ROI made before the database knew it.

        Parameters
        ----------
        key : str | int
            ROI name or current id.

        new_id : int
            New id, not used by another ROI.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If no image has `key`, or `new_id` is used by another ROI. Nothing is changed in that case.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.change_roi_id('Liver', 1042)
        >>> Data.image['CT 01'].roi['Liver'].id
        1042
        """
        old_id = cls._change_annotation_id('roi', key, new_id)
        if old_id is None:
            return
        new_id = Registry._check_id(new_id)

        for rigid in cls.rigid.values():
            for side in (rigid.reference, rigid.moving):
                side.rois = [new_id if roi_id == old_id else roi_id for roi_id in side.rois]

        for deform in cls.deformable.values():
            if deform.roi_ids is not None:
                deform.roi_ids = [new_id if roi_id == old_id else roi_id for roi_id in deform.roi_ids]
            if old_id in deform.roi:
                deform.roi = {(new_id if k == old_id else k): v for k, v in deform.roi.items()}
            deform.invalidate_display()

        for image in cls.image.values():
            for model in image.model.values():
                model.roi_ids = [new_id if roi_id == old_id else roi_id for roi_id in model.roi_ids]
                model.cavity_ids = [new_id if roi_id == old_id else roi_id for roi_id in model.cavity_ids]
            for group in image.roi_group.values():
                if group.main_id == old_id:
                    group.main_id = new_id
                if group.match is not None:
                    group.match['members'] = [[new_id if m == old_id else m, k] for m, k in group.match['members']]

    @classmethod
    def change_volume_id(cls, key, new_id) -> None:
        """
        Give an image or dose a new permanent id everywhere it is stored. The name is unchanged.

        Updates `Data.image` / `Data.dose` and every Rigid / Deformable side registered with it, including the
        'volume:<id>' stand-in frame of a volume with no Frame of Reference UID. Use it to adopt the database's id for
        an image read before the database knew it.

        Parameters
        ----------
        key : str | int
            Image / dose name or current id.

        new_id : int
            New id, not used by another image or dose.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `key` is unknown or `new_id` is already used by an image or dose. Nothing is changed in that case.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.change_volume_id('CT 01', 88123)
        >>> Data.image[88123].image_name
        'CT 01'
        """
        old_id = cls.volume(key).id
        new_id = Registry._check_id(new_id)
        if new_id == old_id:
            return

        if new_id in cls.image or new_id in cls.dose:
            raise KeyError(f"Id {new_id} is already used by '{cls.volume_label(new_id)}'.")

        registry = cls.image if old_id in cls.image else cls.dose
        registry.change_id(old_id, new_id)

        old_frame, new_frame = f'volume:{old_id}', f'volume:{new_id}'
        for registration in cls.rigid.values() + cls.deformable.values():
            for side in (registration.reference, registration.moving):
                if side.image_id == old_id:
                    side.image_id = new_id
                if side.frame == old_frame:
                    side.frame = new_frame

        for deform in cls.deformable.values():
            deform.invalidate_display()

        for image in cls.image.values():
            for kind in cls.GROUP_KINDS:
                for group in getattr(image, kind + '_group').values():
                    if group.match is not None and group.match.get('against') == old_id:
                        group.match['against'] = new_id

    @classmethod
    def clear(cls) -> None:
        """
        Wipe all data from the global registry.

        Performs a full reset of the class-level storage, clearing the
        current session's images (with their groups), registrations, dose
        objects and ROI/POI/MOI/PlaneOI/Model/group ids. Ids start again at 1. This action is
        non-reversible and affects every module importing `Data`.

        Returns
        -------
        None

        Examples
        --------
        Reset the session before loading a new patient:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> Data.image_list()
        ['CT 01', 'MR 01']
        >>> Data.clear()
        >>> Data.image_list()
        []

        Every registry is emptied, not just images:

        >>> Data.clear()
        >>> Data.rigid_list(), Data.deformable_list(), Data.dose_list(), Data.roi_list()
        ([], [], [], [])
        """

        # Replace (not mutate) each registry so old references are released. The id providers are kept.
        cls.image = Registry(name_attribute='image_name', id_source=cls.new_volume_id)
        cls.dose = Registry(name_attribute='image_name', id_source=cls.new_volume_id)
        cls.rigid = Registry(name_attribute='rigid_name',
                             id_source=lambda name, item: cls._provided_id(cls.rigid_id_provider, name, item))
        cls.deformable = Registry(name_attribute='deformable_name',
                                  id_source=lambda name, item: cls._provided_id(cls.deformable_id_provider, name, item))
        cls.roi_next_id = 1
        cls.poi_next_id = 1
        cls.moi_next_id = 1
        cls.planeoi_next_id = 1
        cls.model_next_id = 1
        cls.group_next_id = 1
        cls._key_cache = weakref.WeakKeyDictionary()

    @classmethod
    def deformable_list(cls) -> list[str]:
        """
        Return the names of all stored deformable registrations.

        Returns
        -------
        list[str]
            Deformable registration names, in insertion order.

        Examples
        --------
        Nothing loaded:

        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.deformable_list()
        []

        After loading a DVF through the MHD reader:

        >>> import medicalimageanalysis as mia
        >>> mia.read_mhd(
        ...     file='C:/Data/dvf.mhd',
        ...     reference_name='CT 01',
        ...     moving_name='MR 01',
        ...     dvf=True,
        ... )
        >>> Data.deformable_list()
        ['Deformable_0']
        """
        return cls.deformable.keys()

    @classmethod
    def delete(cls, name, registry: str | None = None) -> None:
        """
        Remove an image, dose, rigid or deformable registration from the registry.

        Registrations are not removed with their images: Rigid and Deformable keep their frames of reference (and
        Deformable its meshes), so they keep working when the image is loaded again. Use `delete_roi` / `delete_poi`
        for ROIs and POIs.

        Parameters
        ----------
        name : str | int
            User name or permanent id of the entry to delete. An id needs `registry`, since ids are only unique
            within one registry.
        registry : {'image', 'dose', 'rigid', 'deformable'}, optional
            Registry to delete from. None searches all four by name; the name must then be in exactly one.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `name` is not in the registry (or in any registry when `registry` is None).
        ValueError
            If `registry` is not a known registry, `registry` is None and `name` is in more than one, or `name` is
            an id and `registry` is None.

        Examples
        --------
        Remove a dose, a rigid and a deformable registration:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> Data.delete('Dose_0')
        >>> Data.delete('CT 01_MR 01', registry='rigid')
        >>> Data.delete('DVF_CT 01_CBCT 01')

        Remove an image by id while keeping the rest of the session:

        >>> Data.delete(Data.image.get_id('MR 01'), registry='image')
        >>> Data.image_list()
        ['CT 01']
        """
        registries = {'image': cls.image, 'dose': cls.dose, 'rigid': cls.rigid, 'deformable': cls.deformable}

        if registry is not None:
            if registry not in registries:
                raise ValueError(f"registry must be one of {list(registries)}; got {registry!r}")
            del registries[registry][name]
            return

        if not isinstance(name, str):
            raise ValueError(f"Deleting by id ({name!r}) needs registry=, ids are only unique within one registry.")

        found = [key for key, entries in registries.items() if name in entries]
        if not found:
            raise KeyError(f"'{name}' is not in any registry.")
        if len(found) > 1:
            raise ValueError(f"'{name}' is in {found}; pass registry= to choose one.")

        del registries[found[0]][name]

    @classmethod
    def delete_image(cls, image_name) -> None:
        """
        Remove a specific image from the registry. Same as ``delete(image_name, registry='image')``.

        Parameters
        ----------
        image_name : str | int
            Name or permanent id of the image to delete.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `image_name` does not exist in the registry.

        Examples
        --------
        Remove a single image while keeping the rest of the session:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> Data.image_list()
        ['CT 01', 'MR 01']
        >>> Data.delete_image('MR 01')
        >>> Data.image_list()
        ['CT 01']

        Guard against a missing name:

        >>> if 'PT 01' in Data.image:
        ...     Data.delete_image('PT 01')
        """
        cls.delete(image_name, registry='image')

    @classmethod
    def delete_model(cls, key, image_name=None) -> None:
        """
        Remove a model from its image and retire its id. A Deformable that used it keeps its frozen copy
        (``side.model_mesh``), so the registration still works and can still be viewed.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.delete_model('Liver FE fine', image_name='CT 01')
        """
        cls._delete_markup('model', key, image_name)

    @classmethod
    def delete_moi(cls, key, image_name=None) -> None:
        """
        Remove a markup of interest from its image and retire its id.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.delete_moi(5521)
        >>> Data.delete_moi('Line 2', image_name='CT 01')
        """
        cls._delete_markup('moi', key, image_name)

    @classmethod
    def delete_planeoi(cls, key, image_name=None) -> None:
        """
        Remove a plane of interest from its image and retire its id.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.delete_planeoi('Superior cut', image_name='CT 01')
        """
        cls._delete_markup('planeoi', key, image_name)

    @classmethod
    def delete_poi(cls, key) -> None:
        """
        Remove a POI from every image and retire its id.

        Parameters
        ----------
        key : str | int
            POI name or permanent id.

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.delete_poi('Iso')
        >>> Data.has_poi('Iso')
        False
        """
        cls._delete_annotation('poi', key)

    @classmethod
    def delete_roi(cls, key) -> None:
        """
        Remove an ROI from every image and retire its id.

        Parameters
        ----------
        key : str | int
            ROI name or permanent id.

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.delete_roi('Liver')
        >>> Data.has_roi('Liver')
        False
        """
        roi_id = cls._delete_annotation('roi', key)

        for image in cls.image.values():                    # a composite whose main ROI is gone is a folder
            for group in image.roi_group.values():
                if group.main_id == roi_id:
                    group.main_id = None

    @classmethod
    def dose_list(cls) -> list[str]:
        """
        Return the names of all stored dose objects.

        Returns
        -------
        list[str]
            Dose names, in insertion order.

        Examples
        --------
        Nothing loaded:

        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.dose_list()
        []

        After loading a plan with an RTDOSE file:

        >>> import medicalimageanalysis as mia
        >>> mia.read_dicoms(
        ...     folder_path='C:/Data/Patient01',
        ...     only_modality=['CT', 'RTDOSE'],
        ... )
        >>> Data.dose_list()
        ['Dose_0']
        """
        return cls.dose.keys()

    @classmethod
    def find_model(cls, key, image_name=None):
        """
        A model by id (searched on every loaded image), or by name / id on `image_name`. Its image is ``model.image``.

        Raises
        ------
        KeyError
            If it is not found.
        ValueError
            If `key` is a name and `image_name` is None (names are unique per image only).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> model = Data.find_model(3301)
        >>> model = Data.find_model('Liver FE', image_name='CT 01')
        """
        return cls._find_markup('model', key, image_name)

    @classmethod
    def find_moi(cls, key, image_name=None):
        """
        A markup of interest by id (searched on every loaded image), or by name / id on `image_name`.

        Its image is ``moi.image``.

        Raises
        ------
        KeyError
            If it is not found.
        ValueError
            If `key` is a name and `image_name` is None (names are unique per image only).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> moi = Data.find_moi(5521)
        >>> moi.image.image_name, moi.name
        ('CT 01', 'Line 1')
        """
        return cls._find_markup('moi', key, image_name)

    @classmethod
    def find_planeoi(cls, key, image_name=None):
        """
        A plane of interest by id (searched on every loaded image), or by name / id on `image_name`.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> plane = Data.find_planeoi(812)
        >>> plane = Data.find_planeoi('Superior cut', image_name='CT 01')
        """
        return cls._find_markup('planeoi', key, image_name)

    @classmethod
    def get_poi_id(cls, key) -> int:
        """
        Permanent id of a POI (name or id) on the loaded images.

        Raises
        ------
        KeyError
            If no loaded image has it.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.get_poi_id('Iso')
        230
        """
        return cls._resolve_annotation('poi', key)

    @classmethod
    def get_poi_name(cls, key) -> str:
        """
        Current name of a POI (id or name) on the loaded images.

        Raises
        ------
        KeyError
            If no loaded image has it.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.get_poi_name(230)
        'Iso'
        """
        return cls._annotation_name('poi', key)

    @classmethod
    def get_roi_id(cls, key) -> int:
        """
        Permanent id of an ROI (name or id) on the loaded images.

        Raises
        ------
        KeyError
            If no loaded image has it.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.get_roi_id('Liver')
        1042
        """
        return cls._resolve_annotation('roi', key)

    @classmethod
    def get_roi_name(cls, key) -> str:
        """
        Current name of an ROI (id or name) on the loaded images.

        Raises
        ------
        KeyError
            If no loaded image has it.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.get_roi_name(1042)
        'Liver'
        """
        return cls._annotation_name('roi', key)

    @classmethod
    def has_poi(cls, key) -> bool:
        """
        True when some loaded image has the POI (name or id).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.has_poi('Iso'), Data.has_poi(230)
        (True, True)
        """
        return cls._has_annotation('poi', key)

    @classmethod
    def has_roi(cls, key) -> bool:
        """
        True when some loaded image has the ROI (name or id).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.has_roi('Liver'), Data.has_roi(1042)
        (True, True)
        """
        return cls._has_annotation('roi', key)

    @classmethod
    def image_list(cls) -> list[str]:
        """
        Return the names of all currently loaded images.

        Returns
        -------
        list[str]
            Image names, in insertion order.

        Examples
        --------
        Nothing loaded:

        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.image_list()
        []

        After loading a patient folder:

        >>> import medicalimageanalysis as mia
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> Data.image_list()
        ['CT 01', 'MR 01']

        Loop over every loaded image:

        >>> for image_name in Data.image_list():
        ...     print(image_name, Data.image[image_name].array.shape)
        CT 01 (142, 512, 512)
        MR 01 (88, 256, 256)
        """
        return cls.image.keys()

    @classmethod
    def match_pois(cls) -> None:
        """
        Synchronize POI definitions across all loaded images.

        Gives every image an entry for every POI found on any image, under the same permanent id. Existing POIs
        keep their own color and visibility; a copy added to an image takes the color/visibility of the first image
        where the POI has a color, else grey [128, 128, 128] and hidden, and the POI's group. Groups are matched
        first (`match_groups`).

        Raises
        ------
        ValueError
            If one POI name has different ids on different images (or one id different names). Fix with
            `change_poi_id` or `rename_poi` first.

        Returns
        -------
        None

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> list(Data.image['MR 01'].poi.keys())
        []
        >>> Data.match_pois()
        >>> Data.image['MR 01'].poi['Iso'].id == Data.image['CT 01'].poi['Iso'].id
        True
        """
        cls._match_annotations('poi')

    @classmethod
    def match_rois(cls) -> None:
        """
        Synchronize ROI definitions across all loaded images.

        Gives every image an entry for every ROI found on any image, under the same permanent id. Existing ROIs
        keep their own color and visibility; a copy added to an image takes the color/visibility of the first image
        where the ROI has a color, else grey [128, 128, 128] and hidden, and the ROI's group. Groups are matched
        first (`match_groups`).

        Raises
        ------
        ValueError
            If one ROI name has different ids on different images (or one id different names). Fix with
            `change_roi_id` or `rename_roi` first.

        Returns
        -------
        None

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> list(Data.image['MR 01'].roi.keys())
        []
        >>> Data.match_rois()
        >>> Data.image['MR 01'].roi['Liver'].id == Data.image['CT 01'].roi['Liver'].id
        True
        """
        cls._match_annotations('roi')

    @classmethod
    def model_ids(cls) -> list[int]:
        """
        Permanent ids of every model on the loaded images.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.model_ids()
        [3301, 3302]
        """
        return cls._markup_ids('model')

    @classmethod
    def moi_ids(cls) -> list[int]:
        """
        Permanent ids of every markup of interest on the loaded images.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.moi_ids()
        [5521, 5522]
        """
        return cls._markup_ids('moi')

    @classmethod
    def new_model_id(cls, name=None, item=None) -> int:
        """
        Id for a new model; every image's ``model`` registry uses it as its id_source.

        ``model_id_provider(name, item)``, or the next ``model_next_id`` when there is no provider or it returns None.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.new_model_id()
        3303
        """
        return cls._new_markup_id('model', name, item)

    @classmethod
    def new_moi_id(cls, name=None, item=None) -> int:
        """
        Id for a new markup of interest; every image's ``moi`` registry uses it as its id_source.

        ``moi_id_provider(name, item)`` (e.g. the database's id), or the next ``moi_next_id`` when there is no
        provider or it returns None. Names are not matched across images.

        Raises
        ------
        KeyError
            If the provider returns an id already used by another markup.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.new_moi_id()
        5523
        """
        return cls._new_markup_id('moi', name, item)

    @classmethod
    def new_planeoi_id(cls, name=None, item=None) -> int:
        """
        Id for a new plane of interest; every image's ``planeoi`` registry uses it as its id_source.

        ``planeoi_id_provider(name, item)``, or the next ``planeoi_next_id`` when there is no provider or it
        returns None. Names are not matched across images.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.new_planeoi_id()
        813
        """
        return cls._new_markup_id('planeoi', name, item)

    @classmethod
    def new_poi_id(cls, name) -> int:
        """
        Id for a POI name being added to an image; every image's ``poi`` registry uses it as its id_source.

        A name already on a loaded image keeps that id, so a name has one id on every image. A new name gets
        ``poi_id_provider(name, None)`` (e.g. the database's id), or the next ``poi_next_id`` when there is no
        provider or it returns None.

        Parameters
        ----------
        name : str
            POI name.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If the provider returns an id already used by another POI.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.new_poi_id('Iso')        # already on an image
        230
        """
        return cls._new_annotation_id('poi', name)

    @classmethod
    def new_roi_id(cls, name) -> int:
        """
        Id for an ROI name being added to an image; every image's ``roi`` registry uses it as its id_source.

        A name already on a loaded image keeps that id, so a name has one id on every image. A new name gets
        ``roi_id_provider(name, None)`` (e.g. the database's id), or the next ``roi_next_id`` when there is no
        provider or it returns None.

        Parameters
        ----------
        name : str
            ROI name.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If the provider returns an id already used by another ROI.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.new_roi_id('Liver')        # already on an image
        1042
        """
        return cls._new_annotation_id('roi', name)

    @classmethod
    def new_volume_id(cls, name=None, item=None) -> int:
        """
        Id for a new image or dose, used as the `id_source` of `image` and `dose`.

        Asks `volume_id_provider` first (e.g. look the series up in the database). Without one, or when it returns
        None, the next free id shared by `image` and `dose`.

        Parameters
        ----------
        name : str | None
            Name the volume is being added under.

        item : Image | Dose | None
            The volume being added (``series_uid``, ``mrn``, ``frame_ref``, ... are set).

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If the provider returns an id already used by another image or dose.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.image.ids(), Data.dose.ids()
        ([1, 2], [3])
        >>> Data.new_volume_id()
        4
        """
        volume_id = cls._provided_id(cls.volume_id_provider, name, item)
        if volume_id is None:
            return max(cls.image.next_id, cls.dose.next_id)

        volume_id = Registry._check_id(volume_id)
        if volume_id in cls.image or volume_id in cls.dose:
            raise KeyError(f"volume_id_provider gave id {volume_id} for {name!r}, but it is already used by "
                           f"'{cls.volume_label(volume_id)}'.")

        return volume_id

    @classmethod
    def planeoi_ids(cls) -> list[int]:
        """
        Permanent ids of every plane of interest on the loaded images.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.planeoi_ids()
        [812]
        """
        return cls._markup_ids('planeoi')

    @classmethod
    def poi_ids(cls) -> list[int]:
        """
        Permanent ids of every POI on the loaded images, in the same order as `poi_list`.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> list(zip(Data.poi_ids(), Data.poi_list()))
        [(230, 'Iso')]
        """
        return [annotation_id for annotation_id, _ in cls._annotation_pairs('poi')]

    @classmethod
    def poi_list(cls) -> list[str]:
        """
        Unique POI names across all loaded images, in the order they are first found.

        Returns
        -------
        list[str]

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.poi_list()
        []
        """
        return [name for _, name in cls._annotation_pairs('poi')]

    @classmethod
    def rename_deformable(cls, key, new_name) -> None:
        """
        Change a deformable registration's user name; its id is unchanged.

        Parameters
        ----------
        key : str | int
            Current name or permanent id.

        new_name : str
            New name, unique among deformable registrations.

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_deformable('DVF_CT 01_CBCT 01', 'Fraction 3 DIR')
        """
        cls.deformable.rename(key, new_name)

    @classmethod
    def rename_dose(cls, key, new_name) -> None:
        """
        Change a dose's user name (``image_name`` and ``dose_name``); its id is unchanged.

        Parameters
        ----------
        key : str | int
            Current name or permanent id.

        new_name : str
            New name, unique among doses.

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_dose('Dose_0', 'Plan Sum')
        >>> Data.dose['Plan Sum'].dose_name
        'Plan Sum'
        """
        dose_id = cls.dose.get_id(key)
        cls.dose.rename(dose_id, new_name)
        cls.dose[dose_id].dose_name = new_name

    @classmethod
    def rename_image(cls, key, new_name) -> None:
        """
        Change an image's user name; its id is unchanged.

        Rigid and Deformable store image ids, so every registration keeps
        working. Only their default names (e.g. 'CT 01_MR 01'), which are
        plain labels, keep the old image name.

        Parameters
        ----------
        key : str | int
            Current name or permanent id.

        new_name : str
            New name, unique among images.

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_image('CT 01', 'Planning CT')
        >>> Data.image['Planning CT'].image_name
        'Planning CT'
        """
        cls.image.rename(key, new_name)

    @classmethod
    def rename_model(cls, key, new_name, image_name=None) -> None:
        """
        Change a model's name on its image; its id is unchanged.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_model('Model 1', 'Liver FE', image_name='CT 01')
        """
        cls._rename_markup('model', key, new_name, image_name)

    @classmethod
    def rename_moi(cls, key, new_name, image_name=None) -> None:
        """
        Change a markup's name on its image; its id is unchanged. Only that image is affected.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_moi('Line 1', 'Tumor length', image_name='CT 01')
        >>> Data.rename_moi(5521, 'Tumor length')
        """
        cls._rename_markup('moi', key, new_name, image_name)

    @classmethod
    def rename_planeoi(cls, key, new_name, image_name=None) -> None:
        """
        Change a plane's name on its image; its id is unchanged. Only that image is affected.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_planeoi('Plane 1', 'Superior cut', image_name='CT 01')
        """
        cls._rename_markup('planeoi', key, new_name, image_name)

    @classmethod
    def rename_poi(cls, key, new_name) -> None:
        """
        Change a POI's user name on every image; its id is unchanged.

        Always rename here rather than through one image's ``poi`` registry, or the images would disagree.

        Parameters
        ----------
        key : str | int
            Current name or permanent id.

        new_name : str
            New name, not used by another POI.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If no image has `key`, or `new_name` belongs to another POI. Nothing is renamed in that case.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_poi('Iso', 'Isocenter')
        >>> Data.image['CT 01'].poi['Isocenter'].id
        230
        """
        cls._rename_annotation('poi', key, new_name)

    @classmethod
    def rename_rigid(cls, key, new_name) -> None:
        """
        Change a rigid registration's user name; its id is unchanged.

        Child registrations find their parent by uid, so they are unaffected.

        Parameters
        ----------
        key : str | int
            Current name or permanent id.

        new_name : str
            New name, unique among rigid registrations.

        Returns
        -------
        None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_rigid('CT 01_CBCT 01', 'right hip')
        >>> Data.rigid['right hip'].rigid_name
        'right hip'
        """
        cls.rigid.rename(key, new_name)

    @classmethod
    def rename_roi(cls, key, new_name) -> None:
        """
        Change an ROI's user name on every image; its id is unchanged.

        Always rename here rather than through one image's ``roi`` registry, or the images would disagree.

        Parameters
        ----------
        key : str | int
            Current name or permanent id.

        new_name : str
            New name, not used by another ROI.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If no image has `key`, or `new_name` belongs to another ROI. Nothing is renamed in that case.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_roi('Liver', 'Liver_Whole')
        >>> Data.image['CT 01'].roi['Liver_Whole'].id
        1042

        Renaming the main ROI of a composite renames what was named after it, and nothing the user named: the
        composite when it still has the ROI's old name, and the parts still named '<old name>_<n>':

        >>> Data.rename_roi('GTV', 'Tumor')            # 'Liver/GTV' -> 'Liver/Tumor', GTV_1 -> Tumor_1, ...
        """
        roi_id = cls._resolve_annotation('roi', key)
        old_name = cls.get_roi_name(roi_id)
        new_name = Registry._check_name(new_name)
        if new_name == old_name:
            return
        if cls.has_roi(new_name) and cls.get_roi_id(new_name) != roi_id:
            raise KeyError(f"ROI name {new_name!r} is already in use.")

        # everything named after the ROI, checked before anything is renamed
        groups, parts = {}, []
        for group in cls._group_templates('roi').values():
            if group.main_id != roi_id:
                continue
            if group.name == old_name:
                groups[group.id] = new_name
            for member_id in cls.group_members('roi', group.id, include_children=False):
                suffix = cls.get_roi_name(member_id)[len(old_name) + 1:]
                if cls.get_roi_name(member_id).startswith(old_name + '_') and suffix.isdigit():
                    parts.append((member_id, f'{new_name}_{suffix}'))

        targets = [name for _, name in parts]
        for member_id, name in parts:
            if targets.count(name) > 1 or (cls.has_roi(name) and cls.get_roi_id(name) != member_id):
                raise KeyError(f"Renaming part {cls.get_roi_name(member_id)!r} to {name!r} would clash with an "
                               f"existing ROI; rename it first.")

        if groups:
            cls._apply_group_paths('roi', names=groups)          # raises on a sibling clash before any change
        cls._rename_annotation('roi', roi_id, new_name)
        for member_id, name in parts:
            cls._rename_annotation('roi', member_id, name)

    @classmethod
    def rigid_list(cls) -> list[str]:
        """
        Return the names of all stored rigid registrations.

        Returns
        -------
        list[str]
            Rigid registration names, in insertion order.

        Examples
        --------
        Nothing loaded:

        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.rigid_list()
        []

        After loading a DICOM REG file alongside its images:

        >>> import medicalimageanalysis as mia
        >>> mia.read_dicoms(
        ...     folder_path='C:/Data/Patient01',
        ...     only_modality=['CT', 'MR', 'REG'],
        ... )
        >>> Data.rigid_list()
        ['Rigid_0']
        """
        return cls.rigid.keys()

    @classmethod
    def roi_ids(cls) -> list[int]:
        """
        Permanent ids of every ROI on the loaded images, in the same order as `roi_list`.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> list(zip(Data.roi_ids(), Data.roi_list()))
        [(1042, 'Liver')]
        """
        return [annotation_id for annotation_id, _ in cls._annotation_pairs('roi')]

    @classmethod
    def roi_list(cls) -> list[str]:
        """
        Unique ROI names across all loaded images, in the order they are first found.

        Returns
        -------
        list[str]

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.roi_list()
        []
        """
        return [name for _, name in cls._annotation_pairs('roi')]

    @classmethod
    def volume(cls, name):
        """
        Return an image or dose by name, for anything that is only displayed or resliced.

        Looks in `image` first, then `dose`. Rigid and Deformable use this for their display paths so a dose can be
        shown through a registration. Registration, masks and ROI/POI matching keep using `image` directly, so a dose
        is never registered, contoured on or given synced ROIs.

        Image and dose ids are counted separately, so look up by name here; an id is checked against `image` first.

        Parameters
        ----------
        name : str | int
            Name (or id) in `image` or `dose`.

        Returns
        -------
        Image or Dose

        Raises
        ------
        KeyError
            If `name` is in neither registry.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01', only_modality=['CT', 'RTDOSE'])
        >>> Data.volume('CT 01') is Data.image['CT 01']
        True
        >>> Data.volume('Dose_0') is Data.dose['Dose_0']
        True
        """
        if name in cls.image:
            return cls.image[name]
        if name in cls.dose:
            return cls.dose[name]

        raise KeyError(f"'{name}' is not in Data.image or Data.dose.")

    @classmethod
    def volume_id(cls, key):
        """
        Permanent id of an image or dose from whatever the user passed.

        This is the one place user input becomes an id: Rigid, Deformable and Dose call it at their entry points and
        work with ids from then on. A name must be loaded; an id is passed through even when its volume is not loaded
        (e.g. a registration restored before its images).

        Parameters
        ----------
        key : str | int | None
            Volume name or permanent id. None stays None.

        Returns
        -------
        int | None

        Raises
        ------
        KeyError
            If `key` is a name that is in neither `image` nor `dose`.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.volume_id('CT 01')
        1
        >>> Data.volume_id(1)
        1
        """
        if key is None:
            return None

        if isinstance(key, str):
            return cls.volume(key).id

        return Registry._check_id(key)

    @classmethod
    def volume_label(cls, key) -> str:
        """
        Current user name of an image or dose for messages and default names, or 'id <n>' when it is not loaded.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.volume_label(1)
        'CT 01'
        >>> Data.volume_label(99)
        'id 99'
        """
        try:
            return cls.volume(key).image_name
        except (KeyError, TypeError):
            return str(key) if isinstance(key, str) else f'id {key}'

    # ---- Group ----

    @staticmethod
    def _geometry_key(kind, item):
        """Short hash of an item's geometry, to tell whether it changed since a group was matched."""
        digest = hashlib.blake2b(digest_size=12)

        def add(values):
            if values is not None:
                digest.update(np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes())
            digest.update(b'|')

        if kind == 'roi':
            if item.contour_position is not None and len(item.contour_position) > 0:
                for contour in item.contour_position:              # contours are saved exactly; the mesh is rebuilt
                    add(contour)
            else:
                add(None if item.mesh is None else item.mesh.points)  # mesh-only ROI
        elif kind == 'poi':
            add(item.point_position)
        elif kind == 'moi':
            add(item.points)
        elif kind == 'planeoi':
            add(np.concatenate((item.origin, item.normal, item.x_axis)))
            add(item.extent)
        else:
            add([item.build_count])
            add(None if item.mesh is None else item.mesh.points)

        return digest.hexdigest()

    @staticmethod
    def _registration_key(registration):
        """Short hash of what a Rigid / Deformable maps with, to tell whether it changed since a group was matched.
        A DVF is sampled every 4th voxel, so the key stays cheap for a viewer to ask often."""
        digest = hashlib.blake2b(digest_size=12)
        if hasattr(registration, 'get_combined_matrix'):
            digest.update(np.round(registration.get_combined_matrix(), 9).tobytes())
        else:
            digest.update(np.round(np.asarray(registration.rigid_matrix, dtype=float), 9).tobytes())
            for values in (registration.origin, registration.spacing):
                digest.update(b'|' if values is None else np.asarray(values, dtype=np.float64).tobytes())
            if registration.dvf is not None:
                digest.update(str(registration.dvf.shape).encode())
                digest.update(np.ascontiguousarray(registration.dvf[::4, ::4, ::4]).tobytes())

        return digest.hexdigest()

    @classmethod
    def _apply_group_paths(cls, kind, names=None, parents=None, remove=None):
        """Change group names (id -> name) and parents (id -> parent id) on every image and re-key every path that
        changes; groups in remove are dropped. Every resulting path is checked (unique, no cycle, parent exists)
        before anything changes."""
        names = {} if names is None else names
        parents = {} if parents is None else parents
        remove = set() if remove is None else set(remove)

        templates = cls._group_templates(kind)
        name_of = {gid: names.get(gid, group.name) for gid, group in templates.items() if gid not in remove}
        parent_of = {gid: parents.get(gid, group.parent_id) for gid, group in templates.items() if gid not in remove}
        for gid, name in name_of.items():
            cls._check_group_name(name)
            if parent_of[gid] is not None and parent_of[gid] not in name_of:
                raise ValueError(f"Group {templates[gid].path!r} would be nested in a group that does not exist.")

        paths = {}
        for gid in name_of:
            chain, current = [], gid
            while current is not None:
                if current in chain:
                    raise ValueError(f"Group {templates[gid].path!r} would be nested in itself.")
                chain.append(current)
                current = parent_of[current]
            paths[gid] = '/'.join(name_of[g] for g in reversed(chain))

        seen = {}
        for gid, path in paths.items():
            if path in seen:
                raise ValueError(f"Two {kind.upper()} groups would be named {path!r}: names must differ among "
                                 f"siblings.")
            seen[path] = gid

        for image in cls.image.values():
            registry = getattr(image, kind + '_group')
            for gid in [g for g in registry.ids() if g in remove]:
                registry.remove(gid)

            changed = [gid for gid in registry.ids() if registry[gid].path != paths[gid]]
            for gid in changed:                                         # two steps, so swapped paths never clash
                registry.rename(gid, f'__renaming__{gid}')
            for gid in changed:
                registry.rename(gid, paths[gid])

            for gid in registry.ids():
                registry[gid].name = name_of[gid]
                registry[gid].parent_id = parent_of[gid]

    @classmethod
    def _cached_geometry_key(cls, kind, item):
        """_geometry_key, reused while the item still holds the same contour / mesh / point objects. Edits through
        the item's own methods (update_position, convert_mask, update_points, ...) replace those objects, so they are
        always seen; an array edited in place is not (like Roi's slice cache)."""
        if kind == 'roi':
            signature = (item.contour_position, item.mesh)
        elif kind == 'poi':
            signature = (item.point_position,)
        elif kind == 'moi':
            signature = (item.points,)
        elif kind == 'planeoi':
            signature = (item.origin, item.normal, item.x_axis, item.extent)
        else:
            signature = (item.mesh, item.build_count)

        cached = cls._key_cache.get(item)
        if cached is not None and len(cached[0]) == len(signature) and all(
                a is b for a, b in zip(cached[0], signature)):
            return cached[1]

        key = cls._geometry_key(kind, item)
        cls._key_cache[item] = (signature, key)

        return key

    @classmethod
    def _cached_registration_key(cls, registration):
        """_registration_key, reused for a Deformable while it holds the same DVF, rigid matrix and grid (sampling
        the DVF is the costly part). A Rigid is recomputed every time: its key is one 4x4 matrix."""
        if hasattr(registration, 'get_combined_matrix'):
            return cls._registration_key(registration)

        signature = (registration.dvf, registration.rigid_matrix, registration.origin, registration.spacing)
        cached = cls._key_cache.get(registration)
        if cached is not None and all(a is b for a, b in zip(cached[0], signature)):
            return cached[1]

        key = cls._registration_key(registration)
        cls._key_cache[registration] = (signature, key)

        return key

    @classmethod
    def _check_group_kind(cls, kind):
        """Raise ValueError unless kind is one of GROUP_KINDS."""
        if kind not in cls.GROUP_KINDS:
            raise ValueError(f"kind must be one of {list(cls.GROUP_KINDS)}; got {kind!r}")

    @classmethod
    def _check_group_name(cls, name):
        """Raise ValueError for a group name that is empty, holds the path separator '/' or starts / ends with a
        space (nothing is stripped silently, so 'Liver' and 'Liver ' can never both exist)."""
        if not isinstance(name, str) or not name.strip() or '/' in name or name != name.strip():
            raise ValueError(f"A group name must be non-empty text without '/' or surrounding spaces; got {name!r}.")

    @classmethod
    def _group_of_roi(cls, roi_id):
        """Group id of an ROI (the same on every image), or None."""
        for image in cls.image.values():
            if roi_id in image.roi:
                return getattr(image.roi[roi_id], 'group_id', None)

        return None

    @classmethod
    def _group_template(cls, kind, key):
        """A group (path or id) as held by the first loaded image that has it; its identity fields are the same on
        every image. KeyError when no image has it."""
        cls._check_group_kind(kind)
        for image in cls.image.values():
            registry = getattr(image, kind + '_group')
            if key in registry:
                return registry[key]

        raise KeyError(f"No {kind.upper()} group {key!r} on any loaded image.")

    @classmethod
    def _group_templates(cls, kind):
        """Group id -> group, each group once (first image that has it), parents before children."""
        templates = {}
        for image in cls.image.values():
            registry = getattr(image, kind + '_group')
            for gid in registry.ids():
                if gid not in templates:
                    templates[gid] = registry[gid]

        return dict(sorted(templates.items(), key=lambda item: item[1].path.count('/')))

    @classmethod
    def add_group(cls, kind, path, color=None, main=None) -> int:
        """
        Create a group on every loaded image and return its id; the groups along the path are created too, like
        ``mkdir -p``. A path that already exists is returned as it is.

        Parameters
        ----------
        kind : {'roi', 'poi', 'moi', 'planeoi', 'model'}

        path : str
            Names from the top down, joined by '/': 'Liver/Vessels'.

        color : list of int, optional
            RGB, 0-255, for the groups created. Random if None.

        main : str | int, optional
            ROI groups only: make the new group a composite of this ROI (see `set_group_main`).

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If no image is loaded (groups live on images), a name is empty, holds '/' or has surrounding spaces, or a
            group would be created inside a composite.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.add_group('roi', 'Liver/Vessels')
        2
        >>> Data.add_group('roi', 'Liver/GTV', main='GTV')        # composite of GTV
        >>> Data.add_group('poi', 'Liver/Vessels/Bifurcations')   # a POI group may share an ROI group's path
        """
        cls._check_group_kind(kind)
        names = str(path).split('/')
        for name in names:
            cls._check_group_name(name)
        if len(cls.image) == 0:
            raise ValueError("Groups live on images: load an image before adding a group.")

        from .structure.group import Group
        from .structure.roi import random_color

        parent_id, current = None, ''
        for name in names:
            current = name if not current else current + '/' + name
            if cls.has_group(kind, current):
                parent_id = cls.get_group_id(kind, current)
                continue

            if parent_id is not None and cls._group_template(kind, parent_id).main_id is not None:
                raise ValueError(f"{cls._group_template(kind, parent_id).path!r} is a composite: it holds ROIs, not "
                                 f"groups.")

            group_color = list(random_color()) if color is None else list(color)
            group_id = None
            for image in cls.image.values():
                group = Group(kind, name, color=group_color, parent_id=parent_id)
                group_id = getattr(image, kind + '_group').add(group, current, item_id=group_id)
            parent_id = group_id

        if main is not None:
            cls.set_group_main(parent_id, main)

        return parent_id

    @classmethod
    def change_group_id(cls, kind, key, new_id) -> None:
        """
        Give a group a new permanent id (e.g. the database's) on every image. The path is unchanged; members and
        nested groups are updated.

        Parameters
        ----------
        kind : {'roi', 'poi', 'moi', 'planeoi', 'model'}

        key : str | int
            Group path or current id.

        new_id : int
            New id, not used by another group of any kind.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.change_group_id('roi', 'Liver/GTV', 4100)
        """
        old_id = cls.get_group_id(kind, key)
        new_id = Registry._check_id(new_id)
        if new_id == old_id:
            return

        for other_kind in cls.GROUP_KINDS:
            if new_id in cls._group_templates(other_kind):
                raise KeyError(f"Group id {new_id} is already used by "
                               f"{cls._group_template(other_kind, new_id).path!r} ({other_kind}).")

        for image in cls.image.values():
            registry = getattr(image, kind + '_group')
            if old_id in registry:
                registry.change_id(old_id, new_id)
            for group in registry.values():
                if group.parent_id == old_id:
                    group.parent_id = new_id
            for item in getattr(image, kind).values():
                if getattr(item, 'group_id', None) == old_id:
                    item.group_id = new_id

        cls.group_next_id = max(cls.group_next_id, new_id + 1)

    @classmethod
    def delete_group(cls, kind, key) -> None:
        """
        Remove a group from every image. Its members and nested groups move up to its parent (ungrouped / top level
        when it has none); no item is deleted.

        Raises
        ------
        ValueError
            If moving the nested groups up would give two siblings the same name. Nothing changes in that case.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.delete_group('roi', 'Liver/Vessels')        # the vessel ROIs are now in 'Liver'
        """
        group = cls._group_template(kind, key)
        group_id, parent_id = group.id, group.parent_id

        children = {gid: parent_id for gid, other in cls._group_templates(kind).items() if other.parent_id == group_id}
        cls._apply_group_paths(kind, parents=children, remove=[group_id])

        for image in cls.image.values():
            for item in getattr(image, kind).values():
                if getattr(item, 'group_id', None) == group_id:
                    item.group_id = parent_id

    @classmethod
    def get_group_id(cls, kind, key) -> int:
        """
        Permanent id of a group (path or id) on the loaded images.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.get_group_id('roi', 'Liver/GTV')
        5
        """
        return cls._group_template(kind, key).id

    @classmethod
    def group_descendants(cls, kind, key) -> list[int]:
        """
        Id of a group followed by the ids of every group nested under it (any depth).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.group_descendants('roi', 'Liver')
        [1, 2, 5]
        """
        templates = cls._group_templates(kind)
        found = [cls.get_group_id(kind, key)]
        index = 0
        while index < len(found):
            found += [gid for gid, group in templates.items() if group.parent_id == found[index] and gid not in found]
            index += 1

        return found

    @classmethod
    def group_list(cls, kind) -> list[str]:
        """
        Paths of every group of one kind, parents before children.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.group_list('roi')
        ['Liver', 'Lungs', 'Liver/Vessels', 'Liver/GTV', 'Lungs/Vessels']
        """
        return [group.path for group in cls._group_templates(kind).values()]

    @classmethod
    def group_members(cls, kind, key, include_children=True, image_name=None) -> list[int]:
        """
        Ids of the items in a group, each id once.

        Parameters
        ----------
        kind : {'roi', 'poi', 'moi', 'planeoi', 'model'}

        key : str | int
            Group path or id.

        include_children : bool, default True
            Also the members of the groups nested in it.

        image_name : str | int, optional
            Only this image. None = every loaded image (ROI/POI ids are the same on every image).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> [Data.get_roi_name(i) for i in Data.group_members('roi', 'Liver/GTV')]
        ['GTV_1', 'GTV_2', 'GTV_3']
        >>> Data.group_members('planeoi', 'Resection 1', image_name='CT 01')
        """
        group_ids = set(cls.group_descendants(kind, key)) if include_children else {cls.get_group_id(kind, key)}
        images = cls.image.values() if image_name is None else [cls.image[image_name]]

        found = []
        for image in images:
            registry = getattr(image, kind)
            for item_id in registry.ids():
                if getattr(registry[item_id], 'group_id', None) in group_ids and item_id not in found:
                    found.append(item_id)

        return found

    @classmethod
    def group_status(cls, kind, key, image_name):
        """
        Whether a group on one image is still what its last run made (see `record_group_match`).

        Returns
        -------
        (str, list of str)
            Status and the reasons it is stale:

            * 'manual'  : no method and never recorded; nothing to run.
            * 'not run' : the group has a method but was never made or matched on this image (e.g. a new CT).
            * 'current' : nothing it used has changed.
            * 'stale'   : something changed: 'method changed', 'registration <kind> <id> changed' / 'deleted',
              'member <id> added' / 'removed' / 'edited'.

            An ROI's geometry is fingerprinted by its contours (its mesh only when it has none), so reloading, which
            rebuilds the mesh, never makes a group stale. Fingerprints are cached until the item's contours / mesh /
            points object (or a Deformable's DVF) is replaced; an array edited in place is not seen.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.group_status('roi', 'Liver/GTV', 'CT 04')
        ('not run', [])
        >>> Data.group_status('roi', 'Liver/GTV', 'CT 02')
        ('stale', ['registration rigid 2 changed'])
        """
        image = cls.image[image_name]
        group = getattr(image, kind + '_group')[key]
        record = group.match
        if record is None:
            return ('manual' if group.method is None else 'not run'), []

        reasons = []
        if record.get('method') != group.method or record.get('method_version') != group.method_version:
            reasons.append('method changed')

        for reg_kind, reg_id, reg_key in record.get('registrations', []):
            registry = cls.rigid if reg_kind == 'rigid' else cls.deformable
            if reg_id not in registry:
                reasons.append(f'registration {reg_kind} {reg_id} deleted')
            elif cls._cached_registration_key(registry[reg_id]) != reg_key:
                reasons.append(f'registration {reg_kind} {reg_id} changed')

        items = getattr(image, kind)
        saved = {int(member_id): member_key for member_id, member_key in record.get('members', [])}
        current = cls.group_members(kind, group.id, include_children=False, image_name=image.id)
        for member_id in current:
            if member_id not in saved:
                reasons.append(f'member {member_id} added')
            elif cls._cached_geometry_key(kind, items[member_id]) != saved[member_id]:
                reasons.append(f'member {member_id} edited')
        for member_id in saved:
            if member_id not in current:
                reasons.append(f'member {member_id} removed')

        return ('stale' if reasons else 'current'), reasons

    @classmethod
    def group_statuses(cls, kind, image_name) -> dict:
        """
        `group_status` of every group of one kind on one image: {path: (status, reasons)}, parents before children.

        Fingerprints are cached per item and per registration (see `group_status`), so calling this after every edit
        or load is cheap; still, call it when something changes, not on every repaint.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.group_statuses('roi', 'CT 04')
        {'Liver': ('manual', []), 'Liver/GTV': ('not run', []), 'Liver/Segments': ('current', [])}
        """
        registry = getattr(cls.image[image_name], kind + '_group')
        groups = sorted(registry.values(), key=lambda group: group.path.count('/'))

        return {group.path: cls.group_status(kind, group.id, image_name) for group in groups}

    @classmethod
    def has_group(cls, kind, key) -> bool:
        """
        True when some loaded image has the group (path or id).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.has_group('roi', 'Liver/GTV')
        True
        """
        try:
            cls._group_template(kind, key)
        except KeyError:
            return False

        return True

    @classmethod
    def match_groups(cls) -> None:
        """
        Give every loaded image every group, of every kind, under the same id and path.

        A copy added to an image takes the group's identity (name, parent, main, method) and the color/visibility of
        the first image that has it; it has no match record (its status is 'not run' or 'manual'). Called by
        `match_rois` / `match_pois` and by `add_group`; call it after adding an image by other means.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.match_groups()
        >>> Data.image['CT 04'].roi_group['Liver/GTV'].match is None
        True
        """
        from .structure.group import Group

        for kind in cls.GROUP_KINDS:
            for group_id, template in cls._group_templates(kind).items():
                for image in cls.image.values():
                    registry = getattr(image, kind + '_group')
                    if group_id not in registry:
                        group = Group(kind, template.name, color=template.color, visible=template.visible)
                        group.copy_identity(template)
                        registry.add(group, template.path, item_id=group_id)

    @classmethod
    def new_group_id(cls, name=None, item=None) -> int:
        """
        Id for a group being added to an image; every image's group containers use it as their id_source.

        A path already on a loaded image (same kind) keeps that id, so a group has one id on every image. A new path
        gets ``group_id_provider(path, group)``, or the next ``group_next_id`` (past every group id of every kind).

        Parameters
        ----------
        name : str
            The group's path.

        item : Group
            The group being added (its kind is set).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.new_group_id('Liver/GTV', group)
        """
        kind = getattr(item, 'kind', None)
        if kind is not None and cls.has_group(kind, name):
            return cls.get_group_id(kind, name)

        used = set()
        for other_kind in cls.GROUP_KINDS:
            used |= set(cls._group_templates(other_kind))

        group_id = cls._provided_id(cls.group_id_provider, name, item)
        if group_id is None:
            group_id = max([cls.group_next_id] + [i + 1 for i in used])
        else:
            group_id = Registry._check_id(group_id)
            if group_id in used:
                raise KeyError(f"group_id_provider gave id {group_id} for {name!r}, but it is already in use.")

        cls.group_next_id = max(cls.group_next_id, group_id + 1)

        return group_id

    @classmethod
    def record_group_match(cls, kind, key, image_name, against=None, path=None) -> None:
        """
        Record on one image that its group was just made or matched, so `group_status` can tell later when that
        result goes stale. A group method calls it at the end of a run; so can the viewer after a manual assignment.

        Stores the group's method and version, the image it was matched against, every registration on the way (by
        id, with a key of what it maps with) and a key of every direct member's geometry on this image.

        Parameters
        ----------
        kind : {'roi', 'poi', 'moi', 'planeoi', 'model'}

        key : str | int
            Group path or id.

        image_name : str | int
            Image the group was made or matched on.

        against : str | int, optional
            Image it was matched against (e.g. the CT the members were defined on). None = made on this image alone.

        path : list of (str, int), optional
            Registrations used, from `find_registration_path`. None with `against` = the shortest chain.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.record_group_match('roi', 'Liver/GTV', 'CT 01')                       # defined on CT 01
        >>> Data.record_group_match('roi', 'Liver/GTV', 'CT 04', against='CT 03')      # matched through CT 03
        """
        image = cls.image[image_name]
        group = getattr(image, kind + '_group')[key]

        against_id = None if against is None else cls.volume_id(against)
        if path is None and against_id is not None:
            path = cls.find_registration_path(image.id, against_id) or []

        registrations = []
        for reg_kind, reg_key in (path or []):
            registry = cls.rigid if reg_kind == 'rigid' else cls.deformable
            registrations.append([reg_kind, registry.get_id(reg_key), cls._cached_registration_key(registry[reg_key])])

        items = getattr(image, kind)
        members = [[member_id, cls._cached_geometry_key(kind, items[member_id])]
                   for member_id in cls.group_members(kind, group.id, include_children=False, image_name=image.id)]

        group.match = {'against': against_id, 'registrations': registrations, 'method': group.method,
                       'method_version': group.method_version, 'members': members}

    @classmethod
    def rename_group(cls, kind, key, new_name) -> None:
        """
        Change a group's own name on every image; the paths of the groups nested in it follow. Its id is unchanged.

        Raises
        ------
        ValueError
            If a sibling already has that name, or the name holds '/'.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.rename_group('roi', 'Liver/GTV', 'Lesions')
        >>> Data.has_group('roi', 'Liver/Lesions')
        True
        """
        group_id = cls.get_group_id(kind, key)
        cls._apply_group_paths(kind, names={group_id: new_name})

    @classmethod
    def resolve_group(cls, kind, group):
        """
        Group id from whatever was passed: None stays None, a path gives that group's id (created when missing, see
        `add_group`), an id must exist.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.resolve_group('poi', 'Airways')          # created on first use
        7
        >>> Data.resolve_group('poi', 7), Data.resolve_group('poi', None)
        (7, None)
        """
        if group is None:
            return None
        if isinstance(group, str):
            return cls.add_group(kind, group)

        return cls.get_group_id(kind, group)

    @classmethod
    def set_group(cls, kind, key, group, image_name=None) -> None:
        """
        Put one or more items in a group, or take them out (group=None).

        An ROI or POI is set on every image (its group is part of its identity, like its name). A markup, plane or
        model is set on its one item.

        Parameters
        ----------
        kind : {'roi', 'poi', 'moi', 'planeoi', 'model'}

        key : str | int | list of them
            Item name(s) or id(s). Markup / plane / model names need `image_name`.

        group : str | int | None
            Group path (created when missing) or id, of the same kind. None = ungrouped.

        image_name : str | int, optional
            Image holding the markups / planes / models; needed when `key` is a name.

        Raises
        ------
        ValueError
            If an ROI would join a composite while it is the main ROI of a composite (its own or another).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.set_group('roi', ['Portal vein', 'Hepatic artery'], 'Liver/Vessels')
        >>> Data.set_group('poi', [f'Bif {n}' for n in range(1, 21)], 'Liver/Vessels/Bifurcations')
        >>> Data.set_group('planeoi', ['Superior cut', 'Inferior cut'], 'Resection 1', image_name='CT 01')
        """
        cls._check_group_kind(kind)
        group_id = cls.resolve_group(kind, group)
        keys = list(key) if isinstance(key, (list, tuple, set)) else [key]

        if kind in ('roi', 'poi'):
            item_ids = [cls._resolve_annotation(kind, k) for k in keys]        # all checked before anything changes
            if kind == 'roi' and group_id is not None and cls._group_template('roi', group_id).main_id is not None:
                mains = {g.main_id: g.path for g in cls._group_templates('roi').values() if g.main_id is not None}
                for item_id in item_ids:
                    if item_id in mains:
                        raise ValueError(f"ROI {cls.get_roi_name(item_id)!r} is the main ROI of {mains[item_id]!r}; "
                                         f"a main ROI cannot be a part of a composite.")
            for image in cls.image.values():
                registry = getattr(image, kind)
                for item_id in item_ids:
                    if item_id in registry:
                        registry[item_id].group_id = group_id
        else:
            items = [cls._find_markup(kind, k, image_name) for k in keys]
            for item in items:
                item.group_id = group_id

    @classmethod
    def set_group_main(cls, key, main) -> None:
        """
        Make an ROI group a composite of a main ROI (its members are parts of that ROI), or a folder again (None).

        Raises
        ------
        ValueError
            If the group holds subgroups, the ROI is already the main ROI of another composite (one composite per
            main ROI), or the ROI is a part of any composite (a main ROI is never a part).

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.set_group_main('Liver/GTV', 'GTV')
        >>> Data.set_group_main('Liver/GTV', None)           # folder
        """
        group = cls._group_template('roi', key)
        main_id = None if main is None else cls._resolve_annotation('roi', main)

        if main_id is not None:
            templates = cls._group_templates('roi')
            if any(other.parent_id == group.id for other in templates.values()):
                raise ValueError(f"{group.path!r} holds groups; a composite holds only ROIs.")
            for other in templates.values():
                if other.main_id == main_id and other.id != group.id:
                    raise ValueError(f"ROI {cls.get_roi_name(main_id)!r} is already the main ROI of {other.path!r}; "
                                     f"an ROI has one composite.")
            own_group = cls._group_template('roi', cls._group_of_roi(main_id)) if cls._group_of_roi(main_id) else None
            if own_group is not None and own_group.main_id is not None:
                raise ValueError(f"ROI {cls.get_roi_name(main_id)!r} is a part of {own_group.path!r}; a part cannot "
                                 f"be a main ROI. Move it out first.")

        for image in cls.image.values():
            if group.id in image.roi_group:
                image.roi_group[group.id].main_id = main_id

    @classmethod
    def set_group_method(cls, kind, key, method, version=None, **settings) -> None:
        """
        Name the algorithm that makes or updates a group, on every image. The package only stores it; the application
        looks the name up and calls it as ``method(image, group, **settings)``.

        Images matched with another method or version report 'stale' ('method changed').

        Parameters
        ----------
        kind : {'roi', 'poi', 'moi', 'planeoi', 'model'}

        key : str | int
            Group path or id.

        method : str | None
            Algorithm name. None = manual.

        version : str, optional
            Algorithm version.

        **settings
            Keyword settings passed to the method.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.set_group_method('roi', 'Liver/GTV', 'components', version='1.0', min_volume=50.0, match='overlap')
        >>> Data.set_group_method('roi', 'Liver/Segments', 'nnunet_liver_segments', version='2026.11')
        """
        group_id = cls.get_group_id(kind, key)
        for image in cls.image.values():
            registry = getattr(image, kind + '_group')
            if group_id in registry:
                registry[group_id].method = method
                registry[group_id].method_version = version
                registry[group_id].settings = dict(settings)

    @classmethod
    def set_group_parent(cls, kind, key, parent) -> None:
        """
        Nest a group in another one (parent=None = top level), on every image. Its path, and the paths under it,
        follow.

        Raises
        ------
        ValueError
            If parent is the group itself or nested in it, is a composite, or already holds a group with this name.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.set_group_parent('roi', 'Vessels', 'Liver')         # 'Vessels' -> 'Liver/Vessels'
        """
        group_id = cls.get_group_id(kind, key)
        parent_id = cls.resolve_group(kind, parent)
        if parent_id is not None and cls._group_template(kind, parent_id).main_id is not None:
            raise ValueError(f"{cls._group_template(kind, parent_id).path!r} is a composite: it holds ROIs, not "
                             f"groups.")

        cls._apply_group_paths(kind, parents={group_id: parent_id})

    @classmethod
    def ungrouped_members(cls, kind, image_name=None) -> list[int]:
        """
        Ids of the items of one kind in no group (or in a group that no longer exists), each id once.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> [Data.get_roi_name(i) for i in Data.ungrouped_members('roi')]
        ['External', 'Iso ring']
        """
        cls._check_group_kind(kind)
        group_ids = set(cls._group_templates(kind))
        images = cls.image.values() if image_name is None else [cls.image[image_name]]

        found = []
        for image in images:
            registry = getattr(image, kind)
            for item_id in registry.ids():
                group_id = getattr(registry[item_id], 'group_id', None)
                if (group_id is None or group_id not in group_ids) and item_id not in found:
                    found.append(item_id)

        return found

    # ---- Other ----

    @classmethod
    def find_registration_path(cls, image_a, image_b, registrations=None):
        """
        Registrations that carry image_a's frame of reference to image_b's, fewest steps first: [] for the same frame,
        None when no chain connects them.

        Each step is ('rigid', id) or ('deformable', id), in order from image_a. A chain may mix both and run in
        either direction of each registration (CT 04 -> CT 03 -> CT 02 -> CT 01, or CT 04 -> CT 01 directly). Among
        equally short chains a deformable step is preferred. A Deformable with no DVF yet is skipped.

        Parameters
        ----------
        image_a, image_b : str | int
            Images or doses (name or id).

        registrations : list of (str, str | int), optional
            Only these: ('rigid', name or id) / ('deformable', name or id). None = every registration.

        Returns
        -------
        list of (str, int) or None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.find_registration_path('CT 04', 'CT 01')
        [('deformable', 3), ('rigid', 2), ('rigid', 1)]
        >>> Data.find_registration_path('CT 04', 'CT 01', registrations=[('rigid', 'CT 01_CT 04')])
        [('rigid', 4)]
        """
        start, goal = cls.frame_of(image_a), cls.frame_of(image_b)
        if start == goal:
            return []

        allowed = None
        if registrations is not None:
            allowed = {(kind, (cls.rigid if kind == 'rigid' else cls.deformable).get_id(reg_key))
                       for kind, reg_key in registrations}

        edges = []
        for kind, registry in (('deformable', cls.deformable), ('rigid', cls.rigid)):
            for reg_id in registry.ids():
                registration = registry[reg_id]
                if allowed is not None and (kind, reg_id) not in allowed:
                    continue
                if kind == 'deformable' and registration.dvf is None:
                    continue
                if registration.reference.frame == registration.moving.frame:
                    continue
                edges.append((kind, reg_id, registration.reference.frame, registration.moving.frame))

        previous = {start: None}
        queue = [start]
        while queue:
            frame = queue.pop(0)
            if frame == goal:
                break
            for kind, reg_id, frame_1, frame_2 in edges:
                for here, there in ((frame_1, frame_2), (frame_2, frame_1)):
                    if here == frame and there not in previous:
                        previous[there] = (frame, kind, reg_id)
                        queue.append(there)

        if goal not in previous:
            return None

        path, frame = [], goal
        while previous[frame] is not None:
            frame, kind, reg_id = previous[frame]
            path.append((kind, reg_id))

        return path[::-1]

    @classmethod
    def frame_of(cls, key):
        """
        Frame of Reference UID of an image or dose, or a stand-in built from its id.

        One rule for the whole package (Rigid, Deformable and Dose all use it). A missing or placeholder UID (e.g.
        '00000.00000') must never make unrelated images share a frame, so such a volume is its own frame,
        'volume:<id>'. Built from the id, not the name, so renaming never changes a frame.

        Parameters
        ----------
        key : str | int | None
            Volume name or permanent id. An id that is not loaded gives 'volume:<id>'; None gives None.

        Returns
        -------
        str | None

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> Data.frame_of('CT 01')
        '1.2.840.113619.2.55.3.1234'
        >>> Data.frame_of('US 01')          # ultrasound has no Frame of Reference UID
        'volume:4'
        """
        if key is None:
            return None

        try:
            volume = cls.volume(key)
        except KeyError:
            return f'volume:{key}'

        frame = getattr(volume, 'frame_ref', None)
        if frame is None or not str(frame).strip(' .0'):
            return f'volume:{volume.id}'

        return str(frame)

    @classmethod
    def map_points(cls, points, from_image, to_image, path=None):
        """
        World points of one image carried into another image's world through a chain of registrations.

        Each step is applied in the direction the chain runs: a Rigid's combined matrix (or its inverse), a
        Deformable's full warp (``Deformable.warp_points``). Use it in a group method to compare pieces or points on
        a new image with the members on an image already matched.

        Parameters
        ----------
        points : (N, 3) array_like
            World points (mm) in from_image's frame of reference.

        from_image, to_image : str | int
            Images or doses (name or id).

        path : list of (str, int), optional
            Steps from `find_registration_path`. None = the shortest chain.

        Returns
        -------
        (N, 3) ndarray

        Raises
        ------
        ValueError
            If no chain connects the two frames, or `path` does not lead from one to the other.

        Examples
        --------
        >>> from medicalimageanalysis import Data
        >>> on_ct1 = Data.map_points(Data.image['CT 04'].poi['Bif 3'].point_position, 'CT 04', 'CT 01')
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if path is None:
            path = cls.find_registration_path(from_image, to_image)
            if path is None:
                raise ValueError(f"No registration chain connects '{cls.volume_label(cls.volume_id(from_image))}' "
                                 f"and '{cls.volume_label(cls.volume_id(to_image))}'.")

        frame = cls.frame_of(from_image)
        for reg_kind, reg_key in path:
            registration = (cls.rigid if reg_kind == 'rigid' else cls.deformable)[reg_key]
            if frame == registration.reference.frame:
                forward, frame = True, registration.moving.frame
            elif frame == registration.moving.frame:
                forward, frame = False, registration.reference.frame
            else:
                raise ValueError(f"Step {reg_kind} {reg_key!r} does not start in frame {frame!r}.")

            if reg_kind == 'rigid':
                matrix = registration.get_combined_matrix()
                if not forward:
                    rotation = matrix[:3, :3].T
                    matrix = np.identity(4)
                    matrix[:3, :3] = rotation
                    matrix[:3, 3] = -rotation @ registration.get_combined_matrix()[:3, 3]
                points = points @ matrix[:3, :3].T + matrix[:3, 3]
            else:
                points = registration.warp_points(points, to_reference=not forward)

        if frame != cls.frame_of(to_image):
            raise ValueError(f"The path ends in frame {frame!r}, not in the frame of the target image.")

        return points


# Build the registries once at import
Data.clear()