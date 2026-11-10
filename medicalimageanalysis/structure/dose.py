"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Radiation dose volumes (RTDOSE). A Dose is an :class:`Image` subclass, so it shares the image display (reslicing,
    oblique views, crosshairs), geometry, encrypted saving and the Rigid / Deformable display paths. What it adds is
    dose scaling (Gy), a 0 Gy background, a dose window, dose resampled onto any image grid (optionally through a
    Rigid) and partial-volume dose-volume histogram metrics for many ROIs at once. Doses live in
    ``Data.dose`` (not ``Data.image``), so they are never registered, contoured on or given synced ROIs/POIs.

    Conventions
    -----------
    * Same as :mod:`image`: the array is (z, y, x); origin, spacing and world points are (x, y, z); ``dose.matrix``
      rows are the dose grid axes.
    * The array is float32 in Gy (stored pixel values times DoseGridScaling).
    * Outside the dose grid the display returns ``Dose.background`` (0 Gy).
    * ``dose_name`` is the key in ``Data.dose``; ``image_name`` is the same name, so inherited Image code works.
    * DVH metrics are named as strings (see :meth:`Dose.compute_roi_metrics`): 'Volume', 'Dmin', 'Dmax', 'Dmean',
      'Dmedian', 'Dstd', 'D95' (dose to the hottest 95 % of the volume), 'D0.03cc' (hottest 0.03 cc), 'V20Gy' (% of the
      volume receiving at least 20 Gy), 'V20Gy_cc', 'VS20Gy' / 'VS20Gy_cc' (volume receiving less than 20 Gy), and with
      a prescription 'V95%' / 'V95%_cc' (at least 95 % of the prescription).

Structure:
    - Dose: Image subclass with dose scaling, a dose window, dose resampling, DVH metrics and annotation methods
      disabled.

Examples
--------
>>> import medicalimageanalysis as mia
>>> dose = mia.Data.dose[mia.Data.dose_list()[0]]
>>> res = dose.display.compute_array('Axial', dose.display.position['Axial'])

Dose on its planning CT (same frame of reference, no registration)

>>> ct = mia.Data.image['CT 01']
>>> grid = ct.display.get_grid('Axial', ct.display.position['Axial'])
>>> res = dose.display.compute_array('Axial', ct.display.position['Axial'], grid=grid)

Dose on a registered MR, or deformed onto another CT

>>> res = mia.Data.rigid['CT 01_MR 01'].compute_array('Axial', mr_position, base_name='MR 01', image_name='Dose_0')
>>> res = mia.Data.deformable['DVF_CT 02_CT 01'].compute_array('Axial', ct2_position, image_name='Dose_0')

DVH metrics for several ROIs in one pass (the dose is resampled once)

>>> table = dose.compute_roi_metrics('CT 01', roi_names=['Parotid_L', 'Parotid_R'],
...                                  metrics=['Dmean', 'D2', 'D0.03cc', 'V26Gy', 'V30Gy_cc'])

The same on a CBCT registered to the planning CT

>>> table = dose.compute_roi_metrics('CBCT 01', roi_names=['Parotid_L', 'Parotid_R'], rigid_name='CT 01_CBCT 01')
"""

import re

import numpy as np
import pandas as pd

from pydicom.uid import generate_uid

from .image import Display, Image
from ..data import Data
from ..utils import contours_to_mask


class Dose(Image):
    """One RTDOSE volume: an :class:`Image` with dose scaling, a 0 Gy background and DVH statistics.

    Parameters
    ----------
    dose : object
        Reader output providing image_set, array, dose_name, modality, filepaths, sops, plane, spacing, orientation,
        origin and image_matrix.

    Attributes
    ----------
    background : float
        Class attribute: 0 Gy outside the dose grid.
    DEFAULT_METRICS : list of str
        Class attribute: metrics :meth:`compute_roi_metrics` reports when none are named: Volume, Dmin, Dmax, Dmean,
        Dmedian, Dstd, D1, D2, D5 ... D95, D98, D99, and V0Gy ... V150Gy in % and cc.
    dose_name : str
        Key in ``Data.dose`` (``image_name`` holds the same name).
    display : Display or None
        None for a tag-only read (no array).

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> dose = mia.Data.dose[mia.Data.dose_list()[0]]
    >>> dose.origin, dose.spacing, dose.array.shape
    >>> dose.get_window()
    [0.0, 72.4]
    """
    background = 0.0
    DEFAULT_METRICS = (['Volume', 'Dmin', 'Dmax', 'Dmean', 'Dmedian', 'Dstd'] +
                       [f'D{p}' for p in [1, 2] + list(range(5, 100, 5)) + [98, 99]] +
                       [f'V{x}Gy' for x in range(0, 155, 5)] +
                       [f'V{x}Gy_cc' for x in range(0, 155, 5)])

    def __init__(self, dose):
        self.roi = {}
        self.poi = {}

        self.tags = dose.image_set
        self.array = None if dose.array is None else self._to_gy(dose.array)    # None for a tag-only read
        self.array_dtype = 'float32'

        self.dose_name = dose.dose_name
        self.image_name = dose.dose_name
        self.modality = dose.modality

        patient_name = self._first_tag(['PatientName'])
        self.patient_name = 'missing' if patient_name is None else str(patient_name).split('^')[:3]
        self.mrn = str(self._first_tag(['PatientID'], 'missing'))
        self.birthdate = str(self._first_tag(['PatientBirthDate'], ''))
        self.date = self._first_tag(['SeriesDate', 'ContentDate', 'AcquisitionDate', 'StudyDate'], '00000')
        self.time = self._first_tag(['SeriesTime', 'ContentTime', 'AcquisitionTime', 'StudyTime'], '00000')
        self.local_uid = generate_uid()
        self.study_uid = self._first_tag(['StudyInstanceUID'], '00000.00000')
        self.series_uid = self._first_tag(['SeriesInstanceUID'], '00000.00000')
        self.acq_number = self._first_tag(['AcquisitionNumber'], '1')
        self.frame_ref = self._first_tag(['FrameOfReferenceUID'], '00000.00000')
        self.window = self.get_window()

        self.filepaths = dose.filepaths
        self.sops = dose.sops

        self.plane = dose.plane
        self.orientation = dose.orientation
        self.spacing = np.asarray(dose.spacing, dtype=float)
        self.origin = np.asarray(dose.origin, dtype=float)
        self.matrix = np.asarray(dose.image_matrix, dtype=float)
        shape = dose.dimensions if self.array is None else self.array.shape[:3]
        self.dimensions = tuple(int(d) for d in shape)                  # always (z, y, x)

        self.skipped_slice = False
        self.rgb = False

        self.local_name = None
        self.serialization_version = 1

        self.camera_position = None
        self.visual = {'colormap': 'jet', 'bounds': None}
        self.misc = {}

        self.display = None if self.array is None else Display(self)

    # ------------------------------------------------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _frame(volume, name):
        """Frame of Reference UID of an image or dose, or its name when the UID is missing or a placeholder
        (the same rule as Rigid / Deformable)."""
        frame = getattr(volume, 'frame_ref', None)
        if frame is None or not str(frame).strip(' .0'):
            return name

        return str(frame)

    @staticmethod
    def _metric_values(dose, weight, metrics, prescription=None):
        """Metric name -> value for one ROI.

        dose : (N,) dose (Gy) of each voxel touching the ROI.
        weight : (N,) volume (cc) of each voxel that lies inside the ROI (partial-volume weighted).

        Voxels are sorted hottest first and their volumes accumulated, so 'Dx' is the dose at which the hottest x % (or
        x cc for 'Dxcc') of the volume is reached, interpolated between voxels. A 'Dxcc' larger than the ROI gives Dmin.
        """
        if dose.size == 0 or float(np.sum(weight)) <= 0:
            return {m: np.nan for m in metrics}

        order = np.argsort(dose)[::-1]
        d = np.asarray(dose, dtype=np.float64)[order]
        w = np.asarray(weight, dtype=np.float64)[order]

        cum = np.r_[0.0, np.cumsum(w)]          # volume (cc) of the voxels hotter than each one
        volume = cum[-1]
        mean = float(np.dot(w, d) / volume)
        dose_axis = np.r_[d[0], d]

        def dose_at(cc):
            return float(np.interp(cc, cum, dose_axis))

        stats = {'Volume': volume, 'Dmin': float(d[-1]), 'Dmax': float(d[0]), 'Dmean': mean,
                 'Dmedian': dose_at(0.5 * volume), 'Dstd': float(np.sqrt(np.dot(w, (d - mean) ** 2) / volume))}

        values = {}
        for metric in metrics:
            kind, x, unit, volume_unit = Dose._parse_metric(metric)
            if kind == 'stat':
                values[metric] = stats[x]
            elif kind == 'D':
                values[metric] = dose_at(x / 100.0 * volume if unit == '%' else x)
            else:
                level = x if unit == 'Gy' else x / 100.0 * prescription
                received = cum[np.searchsorted(-d, -level, side='right')]     # volume with dose >= level
                result = received if kind == 'V' else volume - received
                values[metric] = result if volume_unit == 'cc' else 100.0 * result / volume

        return values

    @staticmethod
    def _parse_metric(metric):
        """(kind, value, unit, volume unit) of a metric name; raises ValueError for an unknown name.

        'Dmean' -> ('stat', 'Dmean', None, None); 'D95' -> ('D', 95.0, '%', None); 'D0.03cc' -> ('D', 0.03, 'cc', None);
        'V20Gy' -> ('V', 20.0, 'Gy', 'percent'); 'VS20Gy_cc' -> ('VS', 20.0, 'Gy', 'cc'); 'V95%' -> ('V', 95.0, '%',
        'percent').
        """
        if metric in ('Volume', 'Dmin', 'Dmax', 'Dmean', 'Dmedian', 'Dstd'):
            return 'stat', metric, None, None

        match = re.fullmatch(r'D(\d+(?:\.\d+)?)(cc)?', metric)
        if match:
            return 'D', float(match[1]), 'cc' if match[2] else '%', None

        match = re.fullmatch(r'(VS|V)(\d+(?:\.\d+)?)(Gy|%)(_cc)?', metric)
        if match:
            return match[1], float(match[2]), match[3], 'cc' if match[4] else 'percent'

        raise ValueError(f"Unknown metric {metric!r}: use Volume, Dmin, Dmax, Dmean, Dmedian, Dstd, D<x>, D<x>cc, "
                         f"V<x>Gy, V<x>Gy_cc, VS<x>Gy, VS<x>Gy_cc, V<x>% or V<x>%_cc.")

    # ------------------------------------------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------------------------------------------
    def _raise_no_annotations(self):
        """Doses carry no ROIs or POIs: contour on the image in the dose's frame of reference instead."""
        raise TypeError(f"'{self.dose_name}' is a dose; ROIs and POIs belong on an image in its frame of reference.")

    def _register(self):
        """Store this dose in ``Data.dose`` (used by :meth:`Image.load_metadata`)."""
        Data.dose[self.dose_name] = self

    def _registration_matrix(self, image_name, rigid_name=None):
        """4x4 dose native world -> image_name world.

        Without a Rigid the image must be in the dose's frame of reference. With one, the full registration is used
        (its display animation fraction is ignored), in whichever direction carries the dose onto the image.
        """
        image = Data.image[image_name]
        dose_frame = self._frame(self, self.dose_name)

        if rigid_name is None:
            if self._frame(image, image_name) != dose_frame:
                raise ValueError(f"'{image_name}' is not in the frame of reference of '{self.dose_name}'; pass the "
                                 f"rigid_name of a registration that connects them.")
            return np.identity(4)

        rigid = Data.rigid[rigid_name]
        if not rigid.connects(image_name, self.dose_name):
            raise ValueError(f"'{rigid_name}' does not connect '{image_name}' and '{self.dose_name}'.")

        combined = rigid.get_combined_matrix()      # reference world -> moving world
        if dose_frame != rigid.moving.frame:
            return combined

        inverse = np.identity(4)                    # dose on the moving side: moving world -> reference world
        inverse[:3, :3] = combined[:3, :3].T
        inverse[:3, 3] = -combined[:3, :3].T @ combined[:3, 3]

        return inverse

    def _roi_weights(self, image, roi_names, partial_volume=True):
        """Per ROI: (name, box, weight). box is (z0, z1, y0, y1, x0, x1) on the image grid, weight the (z, y, x) volume
        in cc of each box voxel inside the ROI. box and weight are None for an ROI with no contours or no voxels.

        partial_volume=True weights each voxel by the fraction inside the contour (mask / 255); False counts a voxel
        fully when at least half of it is inside (mask >= 128).
        """
        voxel_cc = float(np.prod(np.asarray(image.spacing, dtype=float))) / 1000.0

        rois = []
        for name in roi_names:
            roi = image.roi.get(name)
            if roi is None or roi.contour_position is None or len(roi.contour_position) == 0:
                rois.append((name, None, None))
                continue

            mask = contours_to_mask(roi.contour_position, image.dimensions, image.origin, image.spacing, image.matrix)
            inside = mask > 0 if partial_volume else mask >= 128
            if not inside.any():
                rois.append((name, None, None))
                continue

            box = []
            for axes in ((1, 2), (0, 2), (0, 1)):       # z, y, x extents
                index = np.flatnonzero(inside.any(axis=axes))
                box += [int(index[0]), int(index[-1]) + 1]
            z0, z1, y0, y1, x0, x1 = box

            crop = mask[z0:z1, y0:y1, x0:x1]
            if partial_volume:
                weight = crop.astype(np.float32) * np.float32(voxel_cc / 255.0)
            else:
                weight = (crop >= 128).astype(np.float32) * np.float32(voxel_cc)

            rois.append((name, tuple(box), weight))

        return rois

    def _sample_box(self, image, box, matrix):
        """(z, y, x) float32 dose on image's own voxel grid inside box (z0, z1, y0, y1, x0, x1), one image slice at a
        time through the display's trilinear kernel. matrix maps dose native world -> image world."""
        z0, z1, y0, y1, x0, x1 = box
        axes = np.asarray(image.matrix, dtype=float)            # rows = x, y, z index axes
        spacing = np.asarray(image.spacing, dtype=float)

        out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
        for k in range(z0, z1):
            origin = image.compute_position([x0, y0, k])
            grid = {'origin': origin, 'x_axis': axes[0], 'y_axis': axes[1], 'normal': axes[2],
                    'sx': float(spacing[0]), 'sy': float(spacing[1]), 'shape': (y1 - y0, x1 - x0)}
            out[k - z0] = self.display.compute_array('Axial', origin, rigid_matrix=matrix, grid=grid)['array']

        return out

    def _to_gy(self, array):
        """Contiguous (z, y, x) float32 dose in Gy.

        Integer arrays are raw stored values and get DoseGridScaling applied. Float arrays are taken as already
        scaled by the reader, so scaling is never applied twice.
        """
        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.integer):
            scaling = float(self._first_tag(['DoseGridScaling'], 1.0))
            array = array.astype(np.float32) * np.float32(scaling)

        return np.ascontiguousarray(array, dtype=np.float32)

    # ------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------
    def add_poi(self, poi_name=None, color=None, visible=False, path=None, point=None):
        """Not available on a dose; raises TypeError. Add the POI to an image in the dose's frame instead.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].add_poi('Iso', point=[0.0, -20.0, 15.5])
        """
        self._raise_no_annotations()

    def add_roi(self, roi_name=None, color=None, visible=False, path=None, contour=None, plane='Axial'):
        """Not available on a dose; raises TypeError. Add the ROI to an image in the dose's frame instead.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].add_roi('Liver', contour=contours)
        """
        self._raise_no_annotations()

    def compute_dose_on_image(self, image_name, rigid_name=None, bounds=None):
        """Dose (Gy) resampled trilinearly onto an image's voxel grid, 0 outside the dose grid.

        Parameters
        ----------
        image_name : str
            Image in ``Data.image`` whose grid the dose is sampled on.
        rigid_name : str, optional
            Registration in ``Data.rigid`` connecting the dose's frame of reference with the image's (e.g. planning CT
            -> CBCT). None = the image is in the dose's frame of reference.
        bounds : tuple of int, optional
            (z0, z1, y0, y1, x0, x1) index box on the image grid (end exclusive). None = the whole image.

        Returns
        -------
        (z, y, x) float32 ndarray
            The image's shape, or the box's shape when bounds is given.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> dose = mia.Data.dose['RTDOSE 01']
        >>> on_ct = dose.compute_dose_on_image('CT 01')
        >>> on_cbct = dose.compute_dose_on_image('CBCT 01', rigid_name='CT 01_CBCT 01')
        >>> patch = dose.compute_dose_on_image('CT 01', bounds=(40, 60, 200, 300, 150, 260))
        """
        image = Data.image[image_name]
        if bounds is None:
            nz, ny, nx = (int(v) for v in image.dimensions[:3])
            bounds = (0, nz, 0, ny, 0, nx)

        return self._sample_box(image, tuple(int(v) for v in bounds), self._registration_matrix(image_name, rigid_name))

    def compute_roi_metrics(self, image_name, roi_names=None, metrics=None, rigid_name=None, partial_volume=True,
                            prescription=None):
        """DVH metrics for several ROIs on one image, with the dose resampled once.

        The dose is sampled onto the image grid only inside the box that covers every requested ROI, then each ROI's
        metrics come from its own partial-volume mask. Voxels are weighted by the fraction inside the contour, so small
        structures (e.g. parotids) get accurate volumes and dose-volume points.

        Parameters
        ----------
        image_name : str
            Image in ``Data.image`` holding the ROIs.
        roi_names : str or list of str, optional
            ROIs on that image. None = every ROI on it. ROIs with no contours get NaN.
        metrics : str or list of str, optional
            Metric names (see the module conventions), reported in this order. None = ``DEFAULT_METRICS``.

            * 'Volume' (cc), 'Dmin', 'Dmax', 'Dmean', 'Dmedian', 'Dstd' (Gy)
            * 'D<x>'      : dose (Gy) to the hottest x % of the volume, e.g. 'D95', 'D2'
            * 'D<x>cc'    : dose (Gy) to the hottest x cc, e.g. 'D0.03cc', 'D2cc'
            * 'V<x>Gy'    : % of the volume receiving at least x Gy ('V<x>Gy_cc' in cc)
            * 'VS<x>Gy'   : % of the volume receiving less than x Gy ('VS<x>Gy_cc' in cc)
            * 'V<x>%'     : % of the volume receiving at least x % of the prescription ('V<x>%_cc' in cc)
        rigid_name : str, optional
            Registration in ``Data.rigid`` connecting the dose's frame of reference with the image's. None = the image
            is in the dose's frame of reference.
        partial_volume : bool, default True
            Weight voxels by the fraction inside the contour. False counts a voxel when at least half of it is inside.
        prescription : float, optional
            Prescription dose (Gy), needed for 'V<x>%' metrics.

        Returns
        -------
        pandas.DataFrame
            One row per ROI: 'ROI' then the metrics in the requested order.

        Raises
        ------
        ValueError
            For an unknown metric name, a 'V<x>%' metric without a prescription, or an image in another frame of
            reference with no (or the wrong) rigid_name.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> dose = mia.Data.dose['RTDOSE 01']
        >>> dose.compute_roi_metrics('CT 01')                                                # every ROI, defaults
        >>> dose.compute_roi_metrics('CT 01', roi_names=['Parotid_L', 'Parotid_R'],
        ...                          metrics=['Dmean', 'D0.03cc', 'V26Gy', 'V30Gy_cc'])
        >>> dose.compute_roi_metrics('CT 01', roi_names='PTV', metrics=['D95', 'V95%', 'V107%'], prescription=70)
        >>> dose.compute_roi_metrics('CBCT 01', roi_names=['Parotid_L'], rigid_name='CT 01_CBCT 01')
        """
        image = Data.image[image_name]
        if roi_names is None:
            roi_names = list(image.roi)
        roi_names = [roi_names] if isinstance(roi_names, str) else list(roi_names)

        if metrics is None:
            metrics = self.DEFAULT_METRICS
        metrics = [metrics] if isinstance(metrics, str) else list(metrics)

        # check every name before the (slower) mask and sampling work
        for metric in metrics:
            kind, _, unit, _ = self._parse_metric(metric)
            if kind != 'D' and unit == '%' and prescription is None:
                raise ValueError(f"{metric!r} needs a prescription (Gy).")

        matrix = self._registration_matrix(image_name, rigid_name)
        rois = self._roi_weights(image, roi_names, partial_volume=partial_volume)

        boxes = np.array([box for _, box, _ in rois if box is not None])
        if len(boxes):
            union = (int(boxes[:, 0].min()), int(boxes[:, 1].max()), int(boxes[:, 2].min()), int(boxes[:, 3].max()),
                     int(boxes[:, 4].min()), int(boxes[:, 5].max()))
            dose = self._sample_box(image, union, matrix)

        rows = []
        for name, box, weight in rois:
            if box is None:
                values = {m: np.nan for m in metrics}
            else:
                z0, z1, y0, y1, x0, x1 = box
                sub = dose[z0 - union[0]:z1 - union[0], y0 - union[2]:y1 - union[2], x0 - union[4]:x1 - union[4]]
                inside = weight > 0
                values = self._metric_values(sub[inside], weight[inside], metrics, prescription)

            rows.append({'ROI': name, **values})

        return pd.DataFrame(rows, columns=['ROI'] + metrics)

    def create_external(self, name='External', color=None, visible=False, filepaths=None, threshold=-250):
        """Not available on a dose; raises TypeError. Create the external on the planning image instead.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].create_external()
        """
        self._raise_no_annotations()

    def create_roi(self, name=None, color=None, visible=False, filepath=None):
        """Not available on a dose; raises TypeError. Create the ROI on an image in the dose's frame instead.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].create_roi('GTV')
        """
        self._raise_no_annotations()

    def get_window(self):
        """[lower, upper] display window from WindowCenter/WindowWidth, else 0 to the maximum dose (Gy).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.dose['Dose_0'].get_window()
        [0.0, 72.4]
        """
        if (0x0028, 0x1050) in self.tags[0] and (0x0028, 0x1051) in self.tags[0]:
            return super().get_window()

        if self.array is not None:
            return [0.0, float(np.max(self.array))]

        return [0, 1]

    def input_mhd(self, filename, roi_names, values, plane='Axial'):
        """Not available on a dose; raises TypeError. Load the label map onto an image in the dose's frame instead.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].input_mhd('C:/Data/labels.mhd', ['Liver'], [1])
        """
        self._raise_no_annotations()

    def input_rtstruct(self, rtstruct):
        """Not available on a dose; raises TypeError. Load the RTSTRUCT onto an image in the dose's frame instead.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.image['CT 01'].input_rtstruct(rtstruct)
        """
        self._raise_no_annotations()