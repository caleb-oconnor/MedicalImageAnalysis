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
    1. **Global Registry**: Class-level dictionaries shared by every module
       that imports `Data`, so no explicit object passing is required.
    2. **Listing**: Convenience methods (`image_list`, `roi_list`, ...) that
       return the names currently held in the registry.
    3. **Annotation Sync**: `match_rois` / `match_pois` make every image carry
       the same set of ROI/POI names with consistent color and visibility.
       Doses live in `dose`, so they are never given ROIs or POIs.
    4. **Lookup / Removal**: `volume` returns an image or dose by name for
       display code (Rigid, Deformable); `delete` removes an entry from any
       registry.

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
"""


class Data(object):
    """
    Global orchestrator for medical imaging data and transformations.

    This class acts as a centralized, static repository (Singleton pattern)
    for managing volumetric data, spatial transformations, and clinical
    annotations. Because data is stored at the class level, all
    modifications are reflected globally across the application.

    Attributes
    ----------
    image : dict
        Mapping of image names to image objects.

    rigid : dict
        Mapping of registration names to rigid registration objects.

    deformable : dict
        Mapping of registration names to deformable registration objects.

    dose : dict
        Mapping of dose names to dose distribution objects.

    Examples
    --------
    Access the registry from any module without passing objects around:

    >>> import medicalimageanalysis as mia
    >>> from medicalimageanalysis import Data
    >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
    >>> Data.image_list()
    ['CT 01', 'MR 01']

    Grab an image and inspect its array (always z-y-x):

    >>> ct = Data.image['CT 01']
    >>> ct.array.shape
    (142, 512, 512)

    Check every registry at once:

    >>> {
    ...     'Images': Data.image_list(),
    ...     'Rigid': Data.rigid_list(),
    ...     'Deformable': Data.deformable_list(),
    ...     'Dose': Data.dose_list(),
    ... }
    {'Images': ['CT 01', 'MR 01'], 'Rigid': [], 'Deformable': [], 'Dose': []}
    """

    image = {}
    rigid = {}
    deformable = {}
    dose = {}

    @classmethod
    def clear(cls) -> None:
        """
        Wipe all data from the global registry.

        Performs a full reset of the class-level storage, clearing the
        current session's images, registrations, and dose objects. This
        action is non-reversible and affects every module importing `Data`.

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
        >>> Data.rigid_list(), Data.deformable_list(), Data.dose_list()
        ([], [], [])
        """

        # Replace (not mutate) each registry so old references are released
        cls.image = {}
        cls.rigid = {}
        cls.deformable = {}
        cls.dose = {}

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
        return list(cls.deformable.keys())

    @classmethod
    def delete(cls, name: str, registry: str | None = None) -> None:
        """
        Remove an image, dose, rigid or deformable registration from the registry.

        Registrations are not removed with their images: Rigid and Deformable keep their frames of reference (and
        Deformable its meshes), so they keep working when the image is loaded again.

        Parameters
        ----------
        name : str
            Key of the entry to delete (as returned by `image_list`, `dose_list`, `rigid_list` or
            `deformable_list`).
        registry : {'image', 'dose', 'rigid', 'deformable'}, optional
            Registry to delete from. None searches all four; the name must then be in exactly one.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `name` is not in the registry (or in any registry when `registry` is None).
        ValueError
            If `registry` is not a known registry, or `registry` is None and `name` is in more than one.

        Examples
        --------
        Remove a dose, a rigid and a deformable registration:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> Data.delete('Dose_0')
        >>> Data.delete('CT 01_MR 01', registry='rigid')
        >>> Data.delete('DVF_CT 01_CBCT 01')

        Remove an image while keeping the rest of the session:

        >>> Data.delete('MR 01', registry='image')
        >>> Data.image_list()
        ['CT 01']
        """
        registries = {'image': cls.image, 'dose': cls.dose, 'rigid': cls.rigid, 'deformable': cls.deformable}

        if registry is not None:
            if registry not in registries:
                raise ValueError(f"registry must be one of {list(registries)}; got {registry!r}")
            del registries[registry][name]
            return

        found = [key for key, entries in registries.items() if name in entries]
        if not found:
            raise KeyError(f"'{name}' is not in any registry.")
        if len(found) > 1:
            raise ValueError(f"'{name}' is in {found}; pass registry= to choose one.")

        del registries[found[0]][name]

    @classmethod
    def delete_image(cls, image_name: str) -> None:
        """
        Remove a specific image from the registry. Same as ``delete(image_name, registry='image')``.

        Parameters
        ----------
        image_name : str
            Key of the image to delete (as returned by `image_list`).

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
        return list(cls.dose.keys())

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
        return list(cls.image.keys())

    @classmethod
    def match_pois(cls) -> None:
        """
        Synchronize POI definitions across all loaded images.

        Similar to `match_rois`, this ensures Point of Interest consistency.
        It is particularly useful when comparing landmarks across different
        modalities (e.g., CT vs MR) where a point identified in one image
        should exist (even if unplaced) in the other.

        Logic:
            1. Aggregates all unique POI names from all images.
            2. Takes color/visibility from an image where the POI has a color.
            3. Adds missing POIs to images that lack them.

        Missing POIs default to grey [128, 128, 128] and visible=False when
        no image provides a color.

        Returns
        -------
        None

        Examples
        --------
        Landmarks placed only on the CT get created on the MR:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> list(Data.image['CT 01'].poi.keys())
        ['ASIS_L', 'ASIS_R']
        >>> list(Data.image['MR 01'].poi.keys())
        []
        >>> Data.match_pois()
        >>> list(Data.image['MR 01'].poi.keys())
        ['ASIS_L', 'ASIS_R']

        The new POIs inherit the CT color and visibility:

        >>> Data.image['MR 01'].poi['ASIS_L'].color
        [255, 0, 0]
        """

        # Union of POI names across every image
        image_pois = [list(cls.image[image_name].poi.keys()) for image_name in list(cls.image.keys())]
        poi_names = list({x for r in image_pois for x in r})

        # Determine color/visibility for each POI (grey and hidden by default)
        color = [[128, 128, 128]] * len(poi_names)
        visible = [False] * len(poi_names)
        for ii, poi_name in enumerate(poi_names):
            for image_name in list(cls.image.keys()):
                pois_on_image = list(cls.image[image_name].poi.keys())
                if poi_name in pois_on_image:
                    if cls.image[image_name].poi[poi_name].color is not None:
                        color[ii] = cls.image[image_name].poi[poi_name].color
                        visible[ii] = cls.image[image_name].poi[poi_name].visible

        # Add missing POIs to each image
        for ii, poi_name in enumerate(poi_names):
            for image_name in list(cls.image.keys()):
                pois_on_image = list(cls.image[image_name].poi.keys())
                if poi_name not in pois_on_image:
                    cls.image[image_name].add_poi(poi_name=poi_name, color=color[ii], visible=visible[ii])

    @classmethod
    def match_rois(cls) -> None:
        """
        Synchronize ROI definitions across all loaded images.

        Performs a 'union' of all Region of Interest names found across the
        dataset so every image has an ROI object for every name in the
        master list.

        Logic:
            1. Aggregates all unique ROI names from all images.
            2. For each ROI, uses the first image with a color as the
               authoritative color and visibility.
            3. Adds missing ROI definitions to images that lack them,
               preserving visual consistency across the session.

        Missing ROIs default to grey [128, 128, 128] and visible=False when
        no image provides a color.

        Returns
        -------
        None

        Examples
        --------
        Contours drawn on the CT get empty counterparts on the MR:

        >>> import medicalimageanalysis as mia
        >>> from medicalimageanalysis import Data
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> list(Data.image['CT 01'].roi.keys())
        ['Liver', 'GTV']
        >>> list(Data.image['MR 01'].roi.keys())
        ['GTV']
        >>> Data.match_rois()
        >>> list(Data.image['MR 01'].roi.keys())
        ['GTV', 'Liver']

        Run after every load so the ROI panel shows the same names per image:

        >>> mia.read_dicoms(folder_path='C:/Data/Patient01_followup', clear=False)
        >>> Data.match_rois()
        >>> all(
        ...     set(Data.image[name].roi) == set(Data.roi_list())
        ...     for name in Data.image_list()
        ... )
        True
        """

        # Union of ROI names across every image
        roi_names = list(cls.roi_list())

        for image_name in cls.image:
            for roi_name in cls.image[image_name].roi:
                if roi_name not in roi_names:
                    roi_names.append(roi_name)

        # Determine authoritative color/visibility for each ROI
        color = [[128, 128, 128] for _ in roi_names]
        visible = [False for _ in roi_names]

        for ii, roi_name in enumerate(roi_names):
            for image_name in cls.image:
                if roi_name in cls.image[image_name].roi:
                    roi = cls.image[image_name].roi[roi_name]

                    if roi.color is not None:
                        color[ii] = roi.color
                        visible[ii] = roi.visible
                        break

        # Add missing ROIs to each image
        for ii, roi_name in enumerate(roi_names):
            for image_name in cls.image:
                if roi_name not in cls.image[image_name].roi:
                    cls.image[image_name].add_roi(roi_name=roi_name, color=color[ii], visible=visible[ii])

    @classmethod
    def poi_list(cls) -> list[str]:
        """
        Return the unique POI names present across all loaded images.

        Returns
        -------
        list[str]
            Unique POI names, in the order they are first found.

        Examples
        --------
        Nothing loaded:

        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.poi_list()
        []

        Names are de-duplicated across images:

        >>> import medicalimageanalysis as mia
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> list(Data.image['CT 01'].poi.keys())
        ['ASIS_L', 'ASIS_R']
        >>> list(Data.image['MR 01'].poi.keys())
        ['ASIS_L', 'PubicSymphysis']
        >>> Data.poi_list()
        ['ASIS_L', 'ASIS_R', 'PubicSymphysis']
        """

        # Collect names in first-seen order without duplicates
        poi_names = []
        for image in cls.image.values():
            for poi_name in image.poi:
                if poi_name not in poi_names:
                    poi_names.append(poi_name)
        return poi_names

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
        return list(cls.rigid.keys())

    @classmethod
    def roi_list(cls) -> list[str]:
        """
        Return the unique ROI names present across all loaded images.

        Returns
        -------
        list[str]
            Unique ROI names, in the order they are first found.

        Examples
        --------
        Nothing loaded:

        >>> from medicalimageanalysis import Data
        >>> Data.clear()
        >>> Data.roi_list()
        []

        Names are de-duplicated across images:

        >>> import medicalimageanalysis as mia
        >>> mia.read_dicoms(folder_path='C:/Data/Patient01')
        >>> list(Data.image['CT 01'].roi.keys())
        ['Liver', 'GTV']
        >>> list(Data.image['MR 01'].roi.keys())
        ['GTV', 'Ablation']
        >>> Data.roi_list()
        ['Liver', 'GTV', 'Ablation']

        Load only specific ROIs from the RTSTRUCT:

        >>> Data.clear()
        >>> mia.read_dicoms(
        ...     folder_path='C:/Data/Patient01',
        ...     only_load_roi_names=['Liver'],
        ... )
        >>> Data.roi_list()
        ['Liver']
        """

        # Collect names in first-seen order without duplicates
        roi_names = []
        for image in cls.image.values():
            for roi_name in image.roi:
                if roi_name not in roi_names:
                    roi_names.append(roi_name)
        return roi_names

    @classmethod
    def volume(cls, name: str):
        """
        Return an image or dose by name, for anything that is only displayed or resliced.

        Looks in `image` first, then `dose`. Rigid and Deformable use this for their display paths so a dose can be
        shown through a registration. Registration, masks and ROI/POI matching keep using `image` directly, so a dose
        is never registered, contoured on or given synced ROIs.

        Parameters
        ----------
        name : str
            Key in `image` or `dose`.

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