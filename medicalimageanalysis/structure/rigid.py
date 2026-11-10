"""The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Rigid registration between two frames of reference, computed from two images of any modality (CT, MR, PT, US,
    X-ray, tomo): display reslicing, manual rotate/translate, automatic registration (ICP or SimpleITK), registrations
    built on a parent registration, and a small JSON state for saving. Logging and when to save are left to the
    application.

    Conventions
    -----------
    * Every image array is (z, y, x), including 2D X-ray (z = 1) and ultrasound frames. Color data is (z, y, x, c).
    * Origin, spacing and world points are (x, y, z). ``image.matrix`` rows are the image axes.
    * ``get_combined_matrix() = matrix @ base`` maps a reference-world point to the matching moving-world point.
      ``matrix`` is this registration's own step; ``base`` is the parent's result (identity without a parent).
    * The registration maps the reference image's frame of reference to the moving image's. Any image in either frame
      can be displayed through it (e.g. an RTDose in the planning CT's frame shown on the MR). Display paths look
      images up with ``Data.volume`` (image or dose); registration paths use ``Data.image`` only.
    * Display calls name their own images: ``base_name`` is the image whose view is shown, ``image_name`` the image
      resliced onto it. Each view (and each overlay in a view) is its own call, so any number of views can share one
      registration. Leaving both out shows the registered moving image on the registered reference image.
    * An image with no usable Frame of Reference UID is its own frame. When both registered images share a frame, the
      registration only applies to those two images.
    * The JSON state holds only 4x4 matrices, names, frames, indices and ROI names. Image arrays, masks,
      points and meshes are never saved.

Structure:
    - RigidSide: image name, frame, z-slice selection and ROI names for one side. Built by Rigid.
    - RigidVisual: display settings (opacity, multicolor, step animation fraction). Not saved.
    - Rigid: matrices, parent link, display, registration backends and JSON state.

Examples
--------
>>> import medicalimageanalysis as mia
>>> rigid = mia.Rigid('CT 01', 'MR 01')
>>> info = rigid.compute_registration('itk', metric='MI')
>>> rigid.retrieve_angles(), rigid.retrieve_translation()

One registration, several views (each its own call)

>>> res = rigid.compute_array('Axial', ct_position)                                       # MR on the CT
>>> res = rigid.compute_array('Axial', mr_position, base_name='MR 01', image_name='CT 01')  # CT on the MR
>>> res = rigid.compute_array('Axial', mr_position, base_name='MR 01', image_name='Dose 01')  # dose in the CT frame
>>> mia.Rigid.find_rigids('Dose 01', 'MR 01')
['CT 01_MR 01']

Second registration built on a first one (both exist, so they can be viewed side by side)

>>> right = mia.Rigid('CT 01', 'CBCT 01', rigid_name='right hip')
>>> right.compute_registration('icp_plane', reference_rois='Femur_R')
>>> left = mia.Rigid('CT 01', 'CBCT 01', rigid_name='left hip', parent_name='right hip')
>>> left.compute_registration('icp_plane', reference_rois=['Femur_L', 'Acetabulum_L'])
>>> left.visual.display_fraction = 0.0     # left hip unaligned, slide to 1.0 to watch it align

A few ultrasound frames registered to a CT

>>> rigid = mia.Rigid('CT 01', 'US 01', moving_indices=[40, 41, 42])
"""

import os
import json
import hashlib

from dataclasses import dataclass, asdict, field

import numpy as np
import pyvista as pv
import SimpleITK as sitk

import vtk
from vtkmodules.util import numpy_support

from pydicom.uid import generate_uid
from scipy.spatial.transform import Rotation

from ..utils.rigid.icp import ICP
from ..utils.rigid.simpleitk import RigidITK
from ..data import Data


@dataclass
class RigidSide:
    """One side (reference or moving) of a registration. Built by :class:`Rigid`, not created directly.

    Attributes
    ----------
    image_name : str
        The image this side was registered with (key into ``Data.image``).
    frame : str
        Frame of Reference UID shared by every image on this side; the image name when the image has none.
    indices : list of int or None
        z-indices of ``image_name`` used for computation. None = all slices. The matching SOP Instance UIDs are
        ``Data.image[image_name].sops[i]``.
    rois : list of str
        ROI names on ``image_name``, paired by position with the other side's ``rois``.
    """
    image_name: str
    frame: str
    indices: list | None = None
    rois: list = field(default_factory=list)


@dataclass
class RigidVisual:
    """Display settings for a registration. Application state: never saved.

    Attributes
    ----------
    reference, moving : object
        Per-image display settings (e.g. colormap names).
    opacity : float
        Blend opacity of the moved image.
    multicolor : object
        Multicolor overlay setting.
    display_fraction : float
        Step animation: 0 shows only the parent's result (structure unaligned), 1 the full registration.
    """
    reference: object = None
    moving: object = None
    opacity: float = 0.5
    multicolor: object = None
    display_fraction: float = 1.0


class Rigid(object):
    """Rigid registration between the frames of reference of a reference and a moving image.

    Parameters
    ----------
    reference_name, moving_name : str
        Images in ``Data.image`` used to compute the registration. Their frames define what it applies to.
    rigid_name : str, optional
        Lookup key in ``Data.rigid``. None builds one from the image names.
    parent_name : str, optional
        Registration this one is built on; it must connect the same two frames. Its result is the base of this one
        and is followed live while it exists; the last known base is kept (and saved) so this registration survives
        the parent being deleted.
    reference_indices, moving_indices : list of int, optional
        z-slices used on each side (e.g. ultrasound frames). None = all slices.
    matrix : (4, 4) array_like, optional
        This registration's own step (reference -> moving). Defaults to identity.
    local_uid : str, optional
        Existing uid when loading (:meth:`from_json`). None generates a new one.

    Examples
    --------
    >>> import medicalimageanalysis as mia
    >>> rigid = mia.Rigid('CT 01', 'MR 01')
    >>> rigid = mia.Rigid('CT 01', 'US 01', moving_indices=[10, 11, 12])
    >>> left = mia.Rigid('CT 01', 'CBCT 01', rigid_name='left hip', parent_name='right hip')
    """
    SCHEMA_VERSION = 1

    def __init__(self, reference_name, moving_name, rigid_name=None, parent_name=None, reference_indices=None,
                 moving_indices=None, matrix=None, local_uid=None):
        self.reference = self._make_side(reference_name, reference_indices)
        self.moving = self._make_side(moving_name, moving_indices)
        self.local_uid = generate_uid() if local_uid is None else local_uid

        self.matrix = np.identity(4) if matrix is None else np.asarray(matrix, dtype=float)

        self.parent_uid = None
        self.base_matrix = np.identity(4)
        if parent_name is not None:
            parent = Data.rigid[parent_name]
            if (parent.reference.frame, parent.moving.frame) != (self.reference.frame, self.moving.frame):
                raise ValueError(f"Parent '{parent_name}' does not connect the same frames of reference.")
            self.parent_uid = parent.local_uid
            self.base_matrix = parent.get_combined_matrix().copy()

        self.visual = RigidVisual()

        self.rigid_name = self.add_rigid(rigid_name)

    @staticmethod
    def _as_list(names):
        """None stays None, a single name becomes a one-item list, anything else becomes a list."""
        if names is None:
            return None
        return [names] if isinstance(names, str) else list(names)

    @staticmethod
    def _frame_of(image_name):
        """Frame of Reference UID of an image, or its own name when the UID is missing or a placeholder.

        A placeholder (e.g. '00000.00000' from Image.get_frame_ref) must never make unrelated images share a frame.
        """
        try:
            volume = Data.volume(image_name)
        except KeyError:
            volume = None

        frame = None if volume is None else getattr(volume, 'frame_ref', None)
        if frame is None or not str(frame).strip(' .0'):
            return image_name

        return str(frame)

    @staticmethod
    def _make_side(image_name, indices=None, frame=None):
        """RigidSide with clean int indices; frame read from the image unless given."""
        return RigidSide(image_name, Rigid._frame_of(image_name) if frame is None else frame,
                         None if indices is None else [int(i) for i in indices])

    @staticmethod
    def _matrix_list(mat):
        """4x4 matrix to a flat list of 16 floats for JSON."""
        return np.asarray(mat, dtype=float).ravel().tolist()

    @staticmethod
    def _matrix_to_sitk(mat, center):
        """4x4 reference -> moving matrix to a centered sitk Euler3DTransform with the same mapping."""
        mat = np.asarray(mat, dtype=float)
        center = np.asarray(center, dtype=float)

        # Re-orthonormalize so sitk's orthogonality check never trips on accumulated float drift.
        u, _, vt = np.linalg.svd(mat[:3, :3])
        rotation = u @ vt

        transform = sitk.Euler3DTransform()
        transform.SetCenter(center.tolist())
        transform.SetMatrix(rotation.ravel().tolist())
        # sitk: T(p) = R (p - c) + c + t'. Matching R p + t gives t' = t - c + R c.
        transform.SetTranslation((mat[:3, 3] - center + rotation @ center).tolist())

        return transform

    @staticmethod
    def _rigid_inv(mat):
        """Exact inverse of a rigid 4x4 (avoids drift from repeated np.linalg.inv)."""
        inv = np.identity(4)
        inv[:3, :3] = mat[:3, :3].T
        inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
        return inv

    @staticmethod
    def _scalar_array(array):
        """(z, y, x) float32 array for registration; (z, y, x, c) color is converted to luminance."""
        array = np.asarray(array)
        if array.ndim == 4:
            if array.shape[3] >= 3:
                weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                array = array[..., :3].astype(np.float32) @ weights
            else:
                array = array.astype(np.float32).mean(axis=3)

        return array.astype(np.float32, copy=False)

    @staticmethod
    def _sitk_to_matrix(transform):
        """Centered sitk rigid transform (Euler / Versor) to a 4x4 reference -> moving matrix."""
        if type(transform) is sitk.Transform:
            transform = transform.Downcast()

        rotation = np.asarray(transform.GetMatrix(), dtype=float).reshape(3, 3)
        center = np.asarray(transform.GetCenter(), dtype=float)
        translation = np.asarray(transform.GetTranslation(), dtype=float)

        mat = np.identity(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = translation + center - rotation @ center

        return mat

    @staticmethod
    def _slice_mask(side, shape):
        """(z, y, x) uint8 mask of the side's selected z-slices, or None when the whole image is used."""
        if side.indices is None:
            return None

        indices = [i for i in side.indices if 0 <= i < shape[0]]
        if not indices:
            raise ValueError(f"None of the indices {side.indices} are inside '{side.image_name}' (z = {shape[0]}).")

        mask = np.zeros(shape, dtype=np.uint8)
        mask[indices] = 1

        return mask

    @staticmethod
    def _vtk_matrix(mat):
        """4x4 numpy matrix to vtkMatrix4x4."""
        vtk_matrix = vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(tuple(np.asarray(mat, dtype=float).ravel()))
        return vtk_matrix

    @staticmethod
    def find_rigids(image_a, image_b):
        """Names of every registration in ``Data.rigid`` that connects the frames of two images (either direction).

        Several can match, e.g. 'right hip' and 'left hip' between the same CT and CBCT.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Rigid.find_rigids('Dose 01', 'MR 01')
        ['CT 01_MR 01']
        """
        return [name for name, rigid in Data.rigid.items() if rigid.connects(image_a, image_b)]

    @classmethod
    def from_json(cls, text):
        """Rebuild a registration from :meth:`to_json` output and register it in ``Data.rigid``.

        The saved frames are kept, so it works before its images are loaded. The parent is found again by uid when it
        is loaded too; otherwise the saved base matrix is used.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> with open('C:/Data/ct_mr.json') as f:
        ...     rigid = mia.Rigid.from_json(f.read())
        """
        record = json.loads(text)
        if record.get('schema', 1) > cls.SCHEMA_VERSION:
            raise ValueError(f"Record schema {record['schema']} is newer than supported ({cls.SCHEMA_VERSION}).")

        reference, moving = record['reference'], record['moving']
        rigid = cls(reference['image_name'], moving['image_name'], rigid_name=record['name'],
                    matrix=np.reshape(record['matrix'], (4, 4)), local_uid=record['uid'])

        rigid.reference = cls._make_side(reference['image_name'], reference['indices'], reference.get('frame'))
        rigid.moving = cls._make_side(moving['image_name'], moving['indices'], moving.get('frame'))
        rigid.reference.rois = list(reference['rois'])
        rigid.moving.rois = list(moving['rois'])
        rigid.parent_uid = record['parent_uid']
        rigid.base_matrix = np.reshape(np.asarray(record['base_matrix'], dtype=float), (4, 4))

        return rigid

    def _apply_display_transform(self, T, base_name, image_name):
        """Apply T (in the base image's world) to the resliced image: A_new = T @ A, then solve back for self.matrix
        so the parent's base stays untouched. Manual edits act on the full registration, so the animation snaps to 1."""
        _, moved_side = self._sides(base_name, image_name, require_different=True)

        self.visual.display_fraction = 1.0
        A_new = T @ self.get_display_matrix(base_name, image_name)
        combined_new = self._rigid_inv(A_new) if moved_side is self.moving else A_new

        self.matrix = combined_new @ self._rigid_inv(self.get_base_matrix())

    def _build_itk(self, reference_mask=None, moving_mask=None, images=True):
        """RigidITK helper from both registered images, optional (z, y, x) masks and the sides' slice subsets.

        A slice subset restricts the metric to those slices, intersected with a user mask when one is given.
        """
        helper = RigidITK()
        for side, is_reference, mask in ((self.reference, True, reference_mask), (self.moving, False, moving_mask)):
            image = Data.image[side.image_name]
            direction = np.asarray(image.matrix, dtype=float).T     # rows = axes -> sitk columns = axes

            if images:
                helper.create_sitk_image(self._scalar_array(image.array), image.origin, image.spacing, direction,
                                         reference=is_reference)

            subset = self._slice_mask(side, np.shape(image.array)[:3])
            if mask is not None and subset is not None:
                mask = np.logical_and(np.asarray(mask) != 0, subset)
            elif subset is not None and images:
                mask = subset

            if mask is not None:
                helper.create_sitk_image(mask, image.origin, image.spacing, direction, reference=is_reference,
                                         mask=True)

        return helper

    def _displayed_pivot(self, base_display, moved_side):
        """Pivot where it appears in the base image's world before this step is applied."""
        pivot = self._pivot(moved_side)
        return base_display[:3, :3] @ pivot + base_display[:3, 3]

    def _extend_grid(self, grid, base_name, image_name):
        """grid grown (by whole pixels, so it stays aligned) to cover image_name's full extent as displayed on base_name.

        Returns (grid, (col0, row0)): col0/row0 (<= 0) are where the grown grid's top-left pixel sits, in grid pixels,
        relative to the original grid's top-left pixel.
        """
        A = self.get_display_matrix(base_name, image_name)
        corners = Data.volume(image_name).compute_corner_positions() @ A[:3, :3].T + A[:3, 3]
        rel = corners - grid['origin']
        u = rel @ grid['x_axis'] / grid['sx']
        v = rel @ grid['y_axis'] / grid['sy']

        h, w = grid['shape']
        col0, row0 = int(np.floor(min(u.min(), 0))), int(np.floor(min(v.min(), 0)))
        col1, row1 = int(np.ceil(max(u.max(), w - 1))), int(np.ceil(max(v.max(), h - 1)))

        new = dict(grid)
        new['origin'] = grid['origin'] + col0 * grid['sx'] * grid['x_axis'] + row0 * grid['sy'] * grid['y_axis']
        new['shape'] = (row1 - row0 + 1, col1 - col0 + 1)

        return new, (col0, row0)

    def _find_parent(self):
        """Parent Rigid by uid, or None if there is no parent or it no longer exists."""
        if self.parent_uid is None:
            return None

        for rigid in Data.rigid.values():
            if rigid.local_uid == self.parent_uid and rigid is not self:
                return rigid

        return None

    def _merge_roi_meshes(self, image_name, roi_names):
        """One PolyData from several ROI meshes on the same image (world coordinates, never modified)."""
        meshes = []
        for name in roi_names:
            roi = Data.image[image_name].roi.get(name)
            if roi is None or roi.mesh is None:
                raise ValueError(f"ROI '{name}' has no mesh on '{image_name}'.")
            meshes.append(roi.mesh)

        if len(meshes) == 1:
            return meshes[0]

        merged = pv.merge(meshes, merge_points=False)
        # ICP requires PolyData; older pyvista versions return an UnstructuredGrid from merge.
        return merged if isinstance(merged, pv.PolyData) else merged.extract_surface()

    def _oriented(self, combined, moved_side):
        """reference -> moving matrix to the matrix that carries moved_side's world onto the other side's world."""
        return self._rigid_inv(combined) if moved_side is self.moving else combined

    def _pivot(self, side):
        """Center of the side's paired ROIs (on its registered image, native world), else that image's center.

        Uses the registered image even when another image of the same frame is displayed, so every image moved
        through this registration animates about the same point.
        """
        image = Data.image[side.image_name]

        bounds = []
        for name in side.rois:
            roi = image.roi.get(name)
            if roi is not None and roi.mesh is not None:
                bounds.append(roi.mesh.bounds)

        if bounds:
            bounds = np.asarray(bounds, dtype=float)
            return (bounds[:, 0::2].min(axis=0) + bounds[:, 1::2].max(axis=0)) / 2

        return np.asarray(image.get_center(), dtype=float)

    def _record(self):
        """Registration state as plain JSON types (no arrays other than the 4x4 matrices as lists)."""
        return {'schema': self.SCHEMA_VERSION, 'uid': self.local_uid, 'name': self.rigid_name,
                'reference': asdict(self.reference), 'moving': asdict(self.moving),
                'matrix': self._matrix_list(self.matrix), 'parent_uid': self.parent_uid,
                'base_matrix': self._matrix_list(self.get_base_matrix())}

    def _roi_on_image(self, roi_name, image_name, moved_side):
        """Name of the ROI to slice on image_name: a paired name from the other side is mapped when image_name is
        the moved side's registered image; otherwise roi_name is used as given."""
        other = self.reference if moved_side is self.moving else self.moving
        if image_name == moved_side.image_name and roi_name not in moved_side.rois and roi_name in other.rois:
            return moved_side.rois[other.rois.index(roi_name)]

        return roi_name

    def _side_of(self, image_name):
        """The side an image belongs to: by name for the registered images, otherwise by frame of reference.

        None when the image is in neither frame, or when both registered images share one frame (then only those two
        images can tell the sides apart).
        """
        if image_name == self.reference.image_name:
            return self.reference
        if image_name == self.moving.image_name:
            return self.moving
        if self.reference.frame == self.moving.frame:
            return None

        frame = self._frame_of(image_name)
        if frame == self.reference.frame:
            return self.reference
        if frame == self.moving.frame:
            return self.moving

        return None

    def _sides(self, base_name, image_name, require_different=False):
        """(base side, moved side) for a display call; None names default to the registered reference / moving."""
        base_name = self.reference.image_name if base_name is None else base_name
        image_name = self.moving.image_name if image_name is None else image_name

        base_side, moved_side = self._side_of(base_name), self._side_of(image_name)
        if base_side is None or moved_side is None:
            missing = base_name if base_side is None else image_name
            raise ValueError(f"'{missing}' is not in a frame of reference connected by '{self.rigid_name}'.")
        if require_different and base_side is moved_side:
            raise ValueError(f"'{base_name}' and '{image_name}' are on the same side of '{self.rigid_name}'.")

        return base_side, moved_side

    def _view_grid(self, plane, position, base_name, image_name, spacing=None):
        """(grid, base_grid): the grid to sample on and the base image's own grid for this view."""
        base_grid = Data.volume(base_name).display.get_grid(plane, position)
        if spacing is None:
            return base_grid, base_grid

        sx, sy = self._view_spacing(plane, base_name, image_name, spacing)
        return Data.volume(image_name).display.grid_with_spacing(base_grid, sx, sy), base_grid

    def _view_spacing(self, plane, base_name, image_name, spacing=None):
        """In-plane (sx, sy): None = base image, 'native' = moved image, float or (sx, sy)."""
        if spacing is None or isinstance(spacing, str):
            if spacing not in (None, 'native'):
                raise ValueError(f"spacing must be None, 'native', a number or (sx, sy); got {spacing!r}")
            display = Data.volume(base_name if spacing is None else image_name).display
            xi, yi, _ = display.AXES[plane]
            return float(display.spacing[xi]), float(display.spacing[yi])

        if np.isscalar(spacing):
            return float(spacing), float(spacing)

        return float(spacing[0]), float(spacing[1])

    def add_rigid(self, rigid_name):
        """Store this registration in ``Data.rigid``; None builds a unique name from the image names.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Rigid('CT 01', 'MR 01').rigid_name
        'CT 01_MR 01'
        """
        if rigid_name is None:
            rigid_name = self.reference.image_name + '_' + self.moving.image_name
            if rigid_name in Data.rigid_list():
                n = 1
                while rigid_name + '_' + str(n) in Data.rigid_list():
                    n += 1
                rigid_name = rigid_name + '_' + str(n)

        Data.rigid[rigid_name] = self

        return rigid_name

    def align_origins(self):
        """Set this step so the two registered images' origins coincide (a rough start before compute_registration).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.rigid['CT 01_MR 01'].align_origins()
        """
        combined = np.identity(4)
        combined[:3, 3] = (np.asarray(Data.image[self.moving.image_name].origin, dtype=float) -
                           np.asarray(Data.image[self.reference.image_name].origin, dtype=float))

        self.matrix = combined @ self._rigid_inv(self.get_base_matrix())

    def compute_array(self, plane, position, base_name=None, image_name=None, spacing=None, extend=True):
        """image_name resliced onto base_name's view: Display.compute_array's result plus 'scale' and 'offset'.

        Parameters
        ----------
        plane : str
            View plane, e.g. 'Axial'.
        position : array_like
            The widget's position (base image's view).
        base_name : str, optional
            Image whose view is shown. Any image in either frame. None = the registered reference image.
        image_name : str, optional
            Image resliced onto the view. Any image in either frame. None = the registered moving image. An image on
            the same side as the base is in the same world, so it is sampled with no registration applied.
        spacing : None, 'native', float or (sx, sy), optional
            None = base image grid, 'native' = moved image spacing, or any spacing. Use the same value in
            :meth:`compute_mesh_slice` and :meth:`compute_aspect`.
        extend : bool, default True
            Grow the grid past the base image so the whole of image_name is shown (fusion with a larger moving image).
            Place the result at 'offset' instead of the base image's top-left pixel.

        Returns
        -------
        dict
            Display.compute_array's result plus 'scale' (sample spacing / base spacing) and 'pixel_offset' (col, row) of the
            result's top-left pixel in the base view's pixel coords; (0, 0) unless extend=True.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid['CT 01_MR 01']
        >>> res = rigid.compute_array('Axial', position)
        >>> res = rigid.compute_array('Sagittal', position, extend=True)    # whole MR, even past the CT's z range
        >>> res = rigid.compute_array('Axial', position, base_name='MR 01', image_name='Dose 01')
        """
        base_name = self.reference.image_name if base_name is None else base_name
        image_name = self.moving.image_name if image_name is None else image_name

        grid, base_grid = self._view_grid(plane, position, base_name, image_name, spacing)
        col0, row0 = 0, 0
        if extend:
            grid, (col0, row0) = self._extend_grid(grid, base_name, image_name)

        res = Data.volume(image_name).display.compute_array(plane, position,
                                                            rigid_matrix=self.get_display_matrix(base_name, image_name),
                                                            grid=grid)
        res['scale'] = (grid['sx'] / base_grid['sx'], grid['sy'] / base_grid['sy'])
        res['pixel_offset'] = (col0 * res['scale'][0], row0 * res['scale'][1])

        return res

    def compute_aspect(self, plane, base_name=None, image_name=None, spacing=None):
        """Pixel aspect (sx / sy) of the view, matching compute_array with the same images and spacing.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.rigid['CT 01_MR 01'].compute_aspect('Sagittal')
        """
        base_name = self.reference.image_name if base_name is None else base_name
        image_name = self.moving.image_name if image_name is None else image_name
        sx, sy = self._view_spacing(plane, base_name, image_name, spacing)

        return np.round(sx / sy, 2)

    def compute_mesh_slice(self, roi_name, plane, position, base_name=None, image_name=None, offset=0,
                           return_pixel=True, use_cache=True, spacing=None):
        """Slice one of image_name's ROIs on the same grid compute_array samples, so image and contour line up.

        On the registered images, roi_name can be the paired name from either image. Pass use_cache=False while
        dragging or animating.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.rigid['CT 01_MR 01'].compute_mesh_slice('Liver', 'Axial', position)
        """
        base_name = self.reference.image_name if base_name is None else base_name
        image_name = self.moving.image_name if image_name is None else image_name
        _, moved_side = self._sides(base_name, image_name)

        roi = Data.volume(image_name).roi[self._roi_on_image(roi_name, image_name, moved_side)]
        grid, _ = self._view_grid(plane, position, base_name, image_name, spacing)

        return roi.compute_mesh_slice(origin=position, plane=plane, offset=offset,
                                      rigid_matrix=self.get_display_matrix(base_name, image_name),
                                      return_pixel=return_pixel, grid=grid, use_cache=use_cache)

    def compute_registration(self, method='icp_point', reference_rois=None, moving_rois=None, reference_mesh=None,
                             moving_mesh=None, reference_mask=None, moving_mask=None, reference_points=None,
                             moving_points=None, use_current=True, **kwargs):
        """Run a rigid registration on the registered images and store the result in ``self.matrix`` (the parent's
        base is factored out).

        Parameters
        ----------
        method : {'itk', 'itk_mask', 'landmark', 'icp_point', 'icp_plane', 'icp_vtk'}, default 'itk'
            * 'itk'       : intensity registration, ``RigidITK.rigid``. Masks and slice subsets restrict the metric.
            * 'itk_mask'  : binary mask shape registration, ``RigidITK.rigid_mask``.
            * 'landmark'  : paired-point fit, ``RigidITK.landmark``.
            * 'icp_point' / 'icp_plane' : Open3D ICP, ``ICP.compute_o3d``.
            * 'icp_vtk'   : VTK ICP, ``ICP.compute_vtk``.
        reference_rois, moving_rois : str or list of str, optional
            ROI names paired by position (moving None = same names). Merged into one surface for ICP. They become
            this registration's ROI pairs and pivot (see :meth:`pair_rois`).
        reference_mesh, moving_mesh : pyvista.PolyData, optional
            ICP only; override the ROIs.
        reference_mask, moving_mask : numpy.ndarray, optional
            ITK only; binary (z, y, x). Required for 'itk_mask'.
        reference_points, moving_points : (N, 3) array_like, optional
            'landmark' only; paired world points.
        use_current : bool, default True
            Start from the current combined matrix when it is not identity.
        **kwargs
            Passed straight to the backend.

        Returns
        -------
        dict
            Backend diagnostics.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Rigid('CT 01', 'MR 01')
        >>> rigid.compute_registration(metric='MI')
        >>> rigid.compute_registration('icp_plane', reference_rois='Liver', distance=5)
        >>> rigid.compute_registration('icp_point', reference_rois=['Liver', 'Kidney_R'],
        ...                            moving_rois=['Liver_2', 'R Kidney'])
        """
        reference_rois = self._as_list(reference_rois)
        moving_rois = self._as_list(moving_rois)
        if moving_rois is not None and reference_rois is None:
            raise ValueError("moving_rois needs reference_rois to pair with.")
        if reference_rois:
            self.pair_rois(reference_rois, moving_rois)
            moving_rois = list(self.moving.rois)

        start = self.get_combined_matrix()
        seed = use_current and not np.allclose(start, np.identity(4))

        if method in ('icp_point', 'icp_plane', 'icp_vtk'):
            if reference_mesh is None and reference_rois:
                reference_mesh = self._merge_roi_meshes(self.reference.image_name, reference_rois)
            if moving_mesh is None and moving_rois:
                moving_mesh = self._merge_roi_meshes(self.moving.image_name, moving_rois)
            if reference_mesh is None or moving_mesh is None:
                raise ValueError(f"'{method}' needs reference_rois or both reference_mesh and moving_mesh.")

            # source = reference, target = moving  ->  the ICP matrix is reference -> moving.
            icp = ICP(reference_mesh, moving_mesh, matrix=start if seed else None)
            if method == 'icp_vtk':
                icp.compute_vtk(**kwargs)
            else:
                icp.compute_o3d(method='point' if method == 'icp_point' else 'plane', **kwargs)

            combined = icp.get_matrix()
            info = icp.get_metrics()

        elif method in ('itk', 'itk_mask'):
            helper = self._build_itk(reference_mask, moving_mask, images=(method == 'itk'))
            initial = None
            if seed:
                initial = self._matrix_to_sitk(start, Data.image[self.reference.image_name].get_center())

            run = helper.rigid if method == 'itk' else helper.rigid_mask
            # sitk is pull-back (reference point -> moving point), the same convention as self.matrix.
            transform, info = run(initial_transform=initial, return_info=True, **kwargs)
            combined = self._sitk_to_matrix(transform)

        elif method == 'landmark':
            if reference_points is None or moving_points is None:
                raise ValueError("'landmark' needs reference_points and moving_points.")

            transform, info = RigidITK().landmark(reference_points, moving_points, return_info=True, **kwargs)
            combined = self._sitk_to_matrix(transform)

        else:
            raise ValueError(f"Unknown method: {method}")

        self.matrix = np.asarray(combined, dtype=float) @ self._rigid_inv(self.get_base_matrix())

        return info

    def connects(self, image_a, image_b):
        """True when the two images are on opposite sides of this registration (by name or frame of reference).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.rigid['CT 01_MR 01'].connects('Dose 01', 'MR 01')
        True
        """
        side_a, side_b = self._side_of(image_a), self._side_of(image_b)

        return side_a is not None and side_b is not None and side_a is not side_b

    def create_image(self, base_name=None, image_name=None):
        """Resample image_name onto base_name's world (same convention as the display). Color keeps channels; outside
        image_name is its ``background`` (0 for color).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> vtk_image = mia.Data.rigid['CT 01_MR 01'].create_image()
        """
        base_name = self.reference.image_name if base_name is None else base_name
        image_name = self.moving.image_name if image_name is None else image_name

        mov = Data.volume(image_name)
        array = np.ascontiguousarray(mov.array)
        components = array.shape[3] if array.ndim == 4 else 1

        vtk_image = vtk.vtkImageData()
        vtk_image.SetSpacing(mov.spacing)
        vtk_image.SetDirectionMatrix(mov.matrix.T.ravel())   # rows = axes in this convention; VTK wants columns
        vtk_image.SetDimensions([int(v) for v in array.shape[2::-1]])
        vtk_image.SetOrigin(mov.origin)

        flat = array.reshape(-1, components) if components > 1 else array.ravel()
        # deep copy only when ascontiguousarray had to make a temporary copy
        scalars = numpy_support.numpy_to_vtk(flat, deep=array is not mov.array)
        vtk_image.GetPointData().SetScalars(scalars)

        # ResliceTransform maps OUTPUT (displayed) coords -> INPUT (native) coords = inverse of display matrix
        transform = vtk.vtkTransform()
        transform.SetMatrix(self._vtk_matrix(self._rigid_inv(self.get_display_matrix(base_name, image_name))))

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(vtk_image)
        reslice.SetResliceTransform(transform)
        reslice.SetInterpolationModeToLinear()
        reslice.SetOutputSpacing(Data.volume(base_name).spacing)
        reslice.SetOutputDirection(1, 0, 0, 0, 1, 0, 0, 0, 1)
        reslice.AutoCropOutputOn()
        reslice.SetBackgroundLevel(0 if components > 1 else mov.background)
        reslice.Update()

        return reslice.GetOutput()

    def export_image(self, path, base_name=None, image_name=None):
        """Write the resampled image (create_image) as an mhd/raw MetaImage pair.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.rigid['CT 01_MR 01'].export_image('C:/Data/aligned.mhd')
        """
        writer = vtk.vtkMetaImageWriter()
        writer.SetInputData(self.create_image(base_name, image_name))
        writer.SetFileName(path)
        writer.Write()

    def get_base_matrix(self):
        """reference -> moving result of the parent (identity without a parent).

        Follows the parent live while it exists and keeps the last result, so this registration survives the parent
        being deleted. The steps don't have to be run back to back.
        """
        parent = self._find_parent()
        if parent is not None:
            self.base_matrix = parent.get_combined_matrix().copy()

        return self.base_matrix

    def get_center(self, base_name=None, image_name=None, plane=None, position=None):
        """image_name's center as displayed on base_name, projected onto the active slice plane when given."""
        base_name = self.reference.image_name if base_name is None else base_name
        image_name = self.moving.image_name if image_name is None else image_name

        A = self.get_display_matrix(base_name, image_name)
        native_center = np.asarray(Data.volume(image_name).get_center(), dtype=float)
        center = A[:3, :3] @ native_center + A[:3, 3]

        if plane is None or position is None:
            return center

        normal = Data.volume(base_name).display.compute_plane_geometry(plane, position)['normal']
        n_hat = normal / np.linalg.norm(normal)
        return center - np.dot(center - np.asarray(position, dtype=float), n_hat) * n_hat

    def get_combined_matrix(self):
        """reference world -> moving world, including the parent's result."""
        return self.matrix @ self.get_base_matrix()

    def get_display_key(self, base_name=None, image_name=None):
        """Short hash of the display matrix for cache keys. Also changes when the parent changes."""
        matrix = np.round(self.get_display_matrix(base_name, image_name), 9) + 0.0     # + 0.0 turns -0.0 into 0.0
        return hashlib.blake2b(matrix.tobytes(), digest_size=8).hexdigest()

    def get_display_matrix(self, base_name=None, image_name=None):
        """image_name native world -> base_name world, including the step animation. Pass this as rigid_matrix.

        Identity when both images are on the same side (same frame of reference).
        """
        base_side, moved_side = self._sides(base_name, image_name)
        if base_side is moved_side:
            return np.identity(4)

        if self.visual.display_fraction >= 1.0:
            return self._oriented(self.get_combined_matrix(), moved_side)

        base_display = self._oriented(self.get_base_matrix(), moved_side)
        return self.get_motion_matrix(base_name, image_name, self.visual.display_fraction) @ base_display

    def get_indices(self, image_name):
        """z-indices registered for this image (e.g. ultrasound frames), or None for all slices / other images.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Rigid('CT 01', 'US 01', moving_indices=[40, 41, 42]).get_indices('US 01')
        [40, 41, 42]
        """
        for side in (self.reference, self.moving):
            if image_name == side.image_name:
                return side.indices

        return None

    def get_motion_matrix(self, base_name=None, image_name=None, fraction=1.0):
        """This step's motion M in base_name's world (full display = M @ base display), interpolated by fraction.

        The pivot moves in a straight line while the rotation angle scales linearly, so the structure slides and
        turns in place instead of swinging around the world origin.
        """
        _, moved_side = self._sides(base_name, image_name, require_different=True)

        base_display = self._oriented(self.get_base_matrix(), moved_side)
        full_display = self._oriented(self.get_combined_matrix(), moved_side)
        motion = full_display @ self._rigid_inv(base_display)
        if fraction >= 1.0:
            return motion

        fraction = max(float(fraction), 0.0)
        c = self._displayed_pivot(base_display, moved_side)
        d = motion[:3, :3] @ c + motion[:3, 3] - c          # how far the pivot travels

        partial = np.identity(4)
        partial[:3, :3] = Rotation.from_rotvec(fraction * Rotation.from_matrix(motion[:3, :3]).as_rotvec()).as_matrix()
        partial[:3, 3] = c + fraction * d - partial[:3, :3] @ c

        return partial

    def get_view_center(self, plane, position, base_name=None, image_name=None):
        """Center in the base widget's view coords (pixel centers at integers, matching the -0.5 placement)."""
        base_name = self.reference.image_name if base_name is None else base_name

        grid = Data.volume(base_name).display.get_grid(plane, position)
        d = self.get_center(base_name, image_name, plane, position) - grid['origin']

        return float(d @ grid['x_axis']) / grid['sx'], float(d @ grid['y_axis']) / grid['sy']

    def get_vtk_matrix(self, base_name=None, image_name=None):
        """For actor.SetUserMatrix on image_name's ROI actors in base_name's 3D view."""
        return self._vtk_matrix(self.get_display_matrix(base_name, image_name))

    def pair_rois(self, reference_rois, moving_rois=None):
        """Set which reference ROI corresponds to which moving ROI (paired by position, replaces earlier pairs).

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Rigid('CT 01', 'MR 01')
        >>> rigid.pair_rois(['Liver', 'Kidney_R'], ['Liver_2', 'R Kidney'])
        >>> rigid.compute_mesh_slice('Liver', 'Axial', position)      # slices 'Liver_2' on the MR
        """
        reference_rois = self._as_list(reference_rois)
        moving_rois = list(reference_rois) if moving_rois is None else self._as_list(moving_rois)
        if len(reference_rois) != len(moving_rois):
            raise ValueError(f"reference_rois ({len(reference_rois)}) and moving_rois ({len(moving_rois)}) must "
                             f"pair one-to-one.")

        self.reference.rois = list(reference_rois)
        self.moving.rois = list(moving_rois)

    def retrieve_angles(self, order='ZXY'):
        """Euler angles (degrees) of this registration's own step."""
        return Rotation.from_matrix(self.matrix[:3, :3]).as_euler(order, degrees=True)

    def retrieve_motion(self, base_name=None, image_name=None, order='ZXY'):
        """This step's rotation (degrees) and the displacement of the pivot, in base_name's world.

        Returns
        -------
        angles : (3,) ndarray
        translation : (3,) ndarray
        """
        _, moved_side = self._sides(base_name, image_name, require_different=True)

        base_display = self._oriented(self.get_base_matrix(), moved_side)
        motion = self.get_motion_matrix(base_name, image_name)
        c = self._displayed_pivot(base_display, moved_side)

        angles = Rotation.from_matrix(motion[:3, :3]).as_euler(order, degrees=True)
        translation = motion[:3, :3] @ c + motion[:3, 3] - c

        return angles, translation

    def retrieve_translation(self):
        """Translation of this registration's own step."""
        return self.matrix[:3, 3]

    def save(self, path):
        """Write the JSON state to a file. Written to a temporary name and swapped in, so a crash never leaves a
        half-written file.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> mia.Data.rigid['CT 01_MR 01'].save('C:/Data/ct_mr.json')
        """
        temp = path + '.tmp'
        with open(temp, 'w') as f:
            f.write(self.to_json())
        os.replace(temp, path)

    def to_json(self):
        """Compact JSON of the registration state (no images, meshes or display settings), typically under 1 KB."""
        return json.dumps(self._record(), separators=(',', ':'))

    def update_rotation(self, base_name=None, image_name=None, center=None, r_x=0, r_y=0, r_z=0, plane=None,
                        position=None, axis=None, angle=None):
        """Rotate image_name as seen on base_name, about center (default: its center on the current slice), by Euler
        r_x/r_y/r_z (world axes, degrees) or axis + angle (degrees).

        The registration itself changes, so every view using it follows.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid['CT 01_MR 01']
        >>> rigid.update_rotation(r_z=2, plane='Axial', position=position)
        >>> rigid.update_rotation('MR 01', 'CT 01', axis=[0, 0.7, 0.7], angle=1.5, plane='Axial', position=position)
        """
        if center is None:
            center = self.get_center(base_name, image_name, plane=plane, position=position)
        center = np.asarray(center, dtype=float)

        R = np.identity(4)
        if axis is not None:
            axis = np.asarray(axis, dtype=float)
            R[:3, :3] = Rotation.from_rotvec(np.radians(angle) * axis / np.linalg.norm(axis)).as_matrix()
        else:
            R[:3, :3] = Rotation.from_euler('xyz', [r_x, r_y, r_z], degrees=True).as_matrix()

        T_neg = np.identity(4)
        T_neg[:3, 3] = -center
        T_pos = np.identity(4)
        T_pos[:3, 3] = center

        self._apply_display_transform(T_pos @ R @ T_neg, base_name, image_name)

    def update_translation(self, base_name=None, image_name=None, t_x=0, t_y=0, t_z=0):
        """Translate image_name as seen on base_name, in base_name's world coordinates.

        Examples
        --------
        >>> import medicalimageanalysis as mia
        >>> rigid = mia.Data.rigid['CT 01_MR 01']
        >>> rigid.update_translation(t_x=1.5)
        >>> rigid.update_translation('MR 01', 'CT 01', t_x=1.5)
        """
        T = np.identity(4)
        T[:3, 3] = [t_x, t_y, t_z]

        self._apply_display_transform(T, base_name, image_name)