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

Id Note:
    Without help, new ids come from counters (1, 2, 3, ...) that only go
    up, so a deleted ROI's number is not reused in the session.
    The package doesn't know the application's database or numbering
    convention, so the application decides ids in one of two ways:

    * **Provider (recommended):** set `Data.roi_id_provider` (and
      `volume_id_provider`, `poi_id_provider`, `rigid_id_provider`,
      `deformable_id_provider`) to a function `provider(name, item)` that
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

    # Fallback counters for new ROI / POI ids when there is no provider (or it returns None). They only go up, so a
    # number is never handed out twice in a session. Reset by clear().
    roi_next_id = 1
    poi_next_id = 1

    # Registries, built by clear() (run once at import, below the class). Images and doses draw ids from one counter,
    # so a volume id is unique across both and Data.volume(id) is never ambiguous.
    image = None
    rigid = None
    deformable = None
    dose = None

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
    def _has_annotation(cls, kind, key):
        """True when some loaded image has the ROI/POI (name or id)."""
        return any(key in getattr(image, kind) for image in cls.image.values())

    @classmethod
    def _match_annotations(cls, kind):
        """Give every image an entry for every ROI/POI on any image, under the same id (see match_rois)."""
        cls._check_annotations(kind)
        images = cls.image.values()

        for annotation_id, name in cls._annotation_pairs(kind):
            color, visible = None, False
            for image in images:
                annotation = getattr(image, kind).get(annotation_id)
                if annotation is not None and annotation.color is not None:
                    color, visible = annotation.color, annotation.visible
                    break

            if color is None:
                color = [128, 128, 128]

            for image in images:
                if annotation_id not in getattr(image, kind):
                    if kind == 'roi':
                        image.add_roi(roi_name=name, color=color, visible=visible, match=False)
                    else:
                        image.add_poi(poi_name=name, color=color, visible=visible, match=False)

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
    def _resolve_annotation(cls, kind, key):
        """Id of an ROI/POI (name or id) on the loaded images; KeyError when no image has it."""
        for image in cls.image.values():
            registry = getattr(image, kind)
            if key in registry:
                return registry.get_id(key)

        raise KeyError(f"No {kind.upper()} {key!r} on any loaded image.")

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

    @classmethod
    def change_roi_id(cls, key, new_id) -> None:
        """
        Give an ROI a new permanent id everywhere it is stored. The name is unchanged.

        Updates every image's ROI, every Rigid's ROI pairs and every Deformable's guiding ROIs and warped meshes. Use it to adopt the database's id for an ROI made before the database knew it.

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

    @classmethod
    def clear(cls) -> None:
        """
        Wipe all data from the global registry.

        Performs a full reset of the class-level storage, clearing the
        current session's images, registrations, dose objects and ROI/POI
        ids. Ids start again at 1. This action is non-reversible and affects
        every module importing `Data`.

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
        cls._delete_annotation('roi', key)

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
        where the POI has a color, else grey [128, 128, 128] and hidden.

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
        where the ROI has a color, else grey [128, 128, 128] and hidden.

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
        """
        cls._rename_annotation('roi', key, new_name)

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


# Build the registries once at import
Data.clear()