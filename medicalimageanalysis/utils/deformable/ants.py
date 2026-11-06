"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Deformable image registration wrappers around ANTsPy (``antspyx``).

    Methods
    -------
    * :meth:`DeformableANTs.registration`: the ANTs deformable transforms (SyN, BSplineSyN, Gaussian / B-spline
      displacement field, time-varying velocity fields) with any ANTs metric (CC, Mattes, MI,
      MeanSquares, Demons, GC), optional label guidance (organ masks drive the deformation), extra metric pairs and
      an optional affine stage, in one call. Also runs any ANTsPy ``ants.registration`` preset
      (e.g. ``'antsRegistrationSyNQuick[so]'``).
    * :meth:`DeformableANTs.landmark`: displacement field fitted to corresponding point pairs.
    * :meth:`DeformableANTs.groupwise`: unbiased template building, with one field per image to the template.

    Conventions
    -----------
    * Input NumPy arrays are indexed (z, y, x), the same as the SimpleITK class. They are transposed internally
      because ANTsPy arrays are indexed (x, y, z). Origin, spacing, direction and landmark points are (x, y, z).
    * Every method returns an ``ants.ANTsImage`` displacement field with 3 components (dx, dy, dz) in mm, on the
      **full reference image grid** (even when mask cropping is used). ``field.numpy()`` is indexed
      (x, y, z, 3); use ``field.numpy().transpose(2, 1, 0, 3)`` for (z, y, x, 3).
    * The field is a pull-back (fixed -> moving) field: for a point ``p`` in the reference image, the corresponding
      point in the moving image is ``p + dvf(p)``. This is the same convention and physical (LPS) space as the
      SimpleITK class, so fields from both are interchangeable.
    * Any affine stage is included in the returned field (it is collapsed into one displacement field).
    * The stored images and masks are never modified. Preprocessing is applied to working copies inside each call.
    * Masks restrict the metric natively in ANTs, so no image blending is needed.

Examples
--------
>>> import numpy as np
>>> from medicalimageanalysis.utils.deformable.ants import DeformableANTs
>>> z, y, x = np.mgrid[:16, :32, :32]
>>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
>>> helper = DeformableANTs()
>>> _ = helper.create_ants_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
>>> _ = helper.create_ants_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
>>> dvf = helper.registration(iterations=(40, 20, 10), metric='MeanSquares')
>>> dvf.shape, dvf.components
((32, 32, 16), 3)
"""

import contextlib
import io
import os
import tempfile

import ants
import numpy as np
from ants.internal import get_lib_fn, get_pointer_string, process_arguments


class DeformableANTs:
    """Deformable registration between a reference (fixed) and moving image with ANTs, with optional masks.

    All image-based methods share the same preprocessing path (:meth:`_prepare`): the moving image and masks are
    resampled onto the reference grid, optionally histogram matched, and optionally cropped to the union of both
    masks. :meth:`_field` turns the ANTs transform files into one displacement field on the full reference grid.

    Parameters
    ----------
    reference_image : ants.ANTsImage, optional
        Fixed image. Its grid defines the output displacement field.
    moving_image : ants.ANTsImage, optional
        Image to be registered onto the reference.
    reference_mask : ants.ANTsImage, optional
        Region of interest on the reference image. Any nonzero voxel is treated as inside.
    moving_mask : ants.ANTsImage, optional
        Region of interest on the moving image. Any nonzero voxel is treated as inside.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.deformable.ants import DeformableANTs
    >>> helper = DeformableANTs()
    >>> image = helper.create_ants_image(np.zeros((16, 32, 32), dtype=np.float32), (0, 0, 0), (1, 1, 2), np.eye(3))
    >>> helper.reference_image.shape
    (32, 32, 16)
    """

    def __init__(self, reference_image=None, moving_image=None, reference_mask=None, moving_mask=None):
        """Store ANTs images and optional masks for later registration calls.

        Parameters
        ----------
        reference_image : ants.ANTsImage, optional
            Fixed image.
        moving_image : ants.ANTsImage, optional
            Moving image.
        reference_mask : ants.ANTsImage, optional
            Binary mask on the reference image (nonzero = inside). Stored as ``unsigned char``.
        moving_mask : ants.ANTsImage, optional
            Binary mask on the moving image (nonzero = inside). Stored as ``unsigned char``.

        Examples
        --------
        >>> import ants
        >>> import numpy as np
        >>> image = ants.from_numpy(np.zeros((32, 32, 16), dtype=np.float32))
        >>> helper = DeformableANTs(reference_image=image, moving_image=image)
        >>> helper.moving_mask is None
        True
        """
        self.reference_image = reference_image
        self.moving_image = moving_image

        # Masks are normalized to binary unsigned char up front, which is what the ANTs mask options expect
        self.reference_mask = None if reference_mask is None else (reference_mask != 0).astype('uint8')
        self.moving_mask = None if moving_mask is None else (moving_mask != 0).astype('uint8')

    def _field(self, transforms, moving, folder, reference=None):
        """Collapse ANTs transform files into one displacement field on the full reference grid.

        Displacement fields defined on a cropped grid give zero displacement outside it; affine parts apply
        everywhere. The result is the pull-back (fixed -> moving) field.

        Parameters
        ----------
        transforms : list of str or ants.ANTsTransform
            Forward transforms in ANTs order (as returned by ``ants.registration(...)['fwdtransforms']``).
        moving : ants.ANTsImage
            Any image in moving space (only needed by ``ants.apply_transforms``).
        folder : str
            Directory for the temporary composed field file.
        reference : ants.ANTsImage, optional
            Grid of the output field. Defaults to the stored reference image.

        Returns
        -------
        ants.ANTsImage
            3-component displacement field (dx, dy, dz) in mm.

        Examples
        --------
        >>> import tempfile
        >>> import ants
        >>> import numpy as np
        >>> reference = ants.from_numpy(np.zeros((32, 32, 16), dtype=np.float32))
        >>> shift = ants.create_ants_transform(transform_type='AffineTransform', translation=(2, 0, 0))
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, 'shift.mat')
        ...     ants.write_transform(shift, path)
        ...     dvf = DeformableANTs(reference_image=reference)._field([path], reference, folder)
        >>> dvf.components, float(dvf.numpy()[8, 8, 8, 0])
        (3, 2.0)
        """
        reference = self.reference_image if reference is None else reference

        # compose= makes antsApplyTransforms write the composite displacement field instead of a warped image
        path = ants.apply_transforms(fixed=reference, moving=moving, transformlist=list(transforms),
                                     compose=os.path.join(folder, 'composed_'))
        field = ants.image_read(path)

        # Read fully into memory before the temporary folder is removed
        return field.clone()

    def _prepare(self, crop=10, histogram_match=False):
        """Build working copies of the images/masks for a registration call (stored data is left untouched).

        Steps, in order: cast to float -> resample moving image and masks onto the reference grid -> histogram
        matching -> crop to the union of both masks.

        Parameters
        ----------
        crop : float, default 10
            Crop margin in mm (converted per axis, so it is isotropic for anisotropic voxels). ``0`` disables
            cropping. Only applied when both masks exist.
        histogram_match : bool, default False
            Histogram-match the moving image to the reference.

        Returns
        -------
        tuple
            ``(fixed, moving, fixed_mask, moving_mask)``; masks may be None.

        Examples
        --------
        >>> import numpy as np
        >>> helper = DeformableANTs()
        >>> array = np.zeros((16, 32, 32), dtype=np.float32)
        >>> mask = np.zeros((16, 32, 32), dtype=np.uint8)
        >>> mask[4:12, 8:24, 8:24] = 1
        >>> for reference in (True, False):
        ...     _ = helper.create_ants_image(array, (0, 0, 0), (1, 1, 2), np.eye(3), reference=reference)
        ...     _ = helper.create_ants_image(mask, (0, 0, 0), (1, 1, 2), np.eye(3), reference=reference, mask=True)
        >>> fixed, moving, fixed_mask, moving_mask = helper._prepare(crop=4)
        >>> fixed.shape, moving.shape == fixed.shape
        ((24, 24, 12), True)
        """
        if self.reference_image is None or self.moving_image is None:
            raise ValueError('Both reference_image and moving_image must be set before registering.')

        # Everything is brought onto the reference grid so a single crop region applies to all images
        fixed = self.reference_image.clone('float')
        moving = ants.resample_image_to_target(self.moving_image.clone('float'), fixed, interp_type='linear')
        # Masks are resampled as float: ANTs returns an empty image when resampling unsigned char directly
        fixed_mask, moving_mask = None, None
        if self.reference_mask is not None:
            fixed_mask = ants.resample_image_to_target(self.reference_mask.clone('float'), fixed,
                                                       interp_type='nearestNeighbor')
            fixed_mask = (fixed_mask > 0.5).astype('uint8')

        if self.moving_mask is not None:
            moving_mask = ants.resample_image_to_target(self.moving_mask.clone('float'), fixed,
                                                        interp_type='nearestNeighbor')
            moving_mask = (moving_mask > 0.5).astype('uint8')

        if histogram_match:
            moving = ants.histogram_match_image(moving, fixed)

        if crop > 0 and fixed_mask is not None and moving_mask is not None:
            union = (fixed_mask.numpy() > 0) | (moving_mask.numpy() > 0)
            if union.any():
                # Bounding box of the union in index space (x, y, z), padded by the margin converted mm -> voxels
                indices = np.argwhere(union)
                pad = np.ceil(crop / np.array(fixed.spacing)).astype(int)
                lower = np.maximum(indices.min(axis=0) - pad, 0)
                upper = np.minimum(indices.max(axis=0) + 1 + pad, np.array(fixed.shape))

                # crop_indices keeps the physical origin, so cropped images stay correctly placed in space
                fixed, moving, fixed_mask, moving_mask = [ants.crop_indices(image, lower.tolist(), upper.tolist())
                                                          for image in (fixed, moving, fixed_mask, moving_mask)]

        return fixed, moving, fixed_mask, moving_mask

    def create_ants_image(self, array, origin, spacing, direction, reference=True, mask=False):
        """Convert a (z, y, x) NumPy array to an ANTs image and store it as the reference/moving image or mask.

        Parameters
        ----------
        array : numpy.ndarray
            Volume indexed (z, y, x). Transposed to ANTs (x, y, z) order internally.
        origin : sequence of float
            Physical position of voxel (0, 0, 0), in (x, y, z) order.
        spacing : sequence of float
            Voxel size in (x, y, z) order, i.e. the slice spacing is the **last** entry.
        direction : numpy.ndarray
            3x3 direction cosine matrix whose columns are the x, y, z axis directions.
        reference : bool, default True
            Store as reference (True) or moving (False).
        mask : bool, default False
            Store as a mask. Masks are binarized (nonzero = inside) and cast to ``unsigned char``.

        Returns
        -------
        ants.ANTsImage
            The created (and stored) image.

        Examples
        --------
        >>> import numpy as np
        >>> helper = DeformableANTs()
        >>> image = helper.create_ants_image(np.zeros((16, 32, 48), dtype=np.float32), (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> image.shape, image.spacing
        ((48, 32, 16), (1.0, 1.0, 2.0))
        >>> mask = helper.create_ants_image(np.ones((16, 32, 48)), (0, 0, 0), (1, 1, 2), np.eye(3), mask=True)
        >>> mask.pixeltype
        'unsigned char'
        """
        # ANTsPy arrays are (x, y, z); this package uses (z, y, x)
        array = np.ascontiguousarray(np.asarray(array).transpose(2, 1, 0))
        if mask:
            array = (array != 0).astype(np.uint8)

        else:
            array = array.astype(np.float32)

        image = ants.from_numpy(array, origin=[float(v) for v in origin], spacing=[float(v) for v in spacing],
                                direction=np.asarray(direction, dtype=np.float64))

        if reference:
            if mask:
                self.reference_mask = image

            else:
                self.reference_image = image

        else:
            if mask:
                self.moving_mask = image

            else:
                self.moving_image = image

        return image

    def groupwise(self, images, iterations=3, transform='SyN', gradient_step=0.2, blending_weight=0.75,
                  initial_template=None, verbose=False):
        """Build an unbiased template from several images and return one displacement field per image.

        Uses ``ants.build_template`` (iterative average of registered images, then sharpened), then registers each
        image to the final template. No image is the reference, so the template has no bias toward one scan.
        Independent of the stored images/masks.

        Parameters
        ----------
        images : list of ants.ANTsImage
            Two or more images, already rigidly aligned.
        iterations : int, default 3
            Template-building iterations (each registers every image to the current template).
        transform : str, default 'SyN'
            ``ants.registration`` type used while building the template and for the final fields (e.g. 'SyN',
            'SyNOnly', 'antsRegistrationSyNQuick[so]'). Types with an affine stage include it in the fields.
        gradient_step : float, default 0.2
            Template update step (how far the template moves toward the average deformation each iteration).
        blending_weight : float, default 0.75
            Weight of the plain average vs. the sharpened average in the template (1 = no sharpening).
        initial_template : ants.ANTsImage, optional
            Starting template. Defaults to the average of ``images``; its grid is the template grid.
        verbose : bool, default False
            Print ANTs progress.

        Returns
        -------
        tuple
            ``(template, fields)``: the template ``ants.ANTsImage`` and one displacement field per image on the
            template grid. Field ``i`` maps a template point to image ``i`` (pull-back), so warping image ``i`` with
            it moves it into template space.

        Examples
        --------
        >>> import ants
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> base = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> images = [ants.from_numpy(np.roll(base, shift, axis=2).transpose(2, 1, 0).copy()) for shift in (-2, 0, 2)]
        >>> template, fields = DeformableANTs().groupwise(images, iterations=2, transform='SyNOnly')
        >>> len(fields), fields[0].shape == template.shape
        (3, True)
        """
        if len(images) < 2:
            raise ValueError('Groupwise registration needs at least two images.')
        images = [image.clone('float') for image in images]

        # build_template prints progress unconditionally; silence it unless verbose
        output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with output:
            template = ants.build_template(initial_template=initial_template, image_list=images,
                                           iterations=iterations, gradient_step=gradient_step,
                                           blending_weight=blending_weight, type_of_transform=transform,
                                           verbose=verbose)

        # Final fields: each image registered to the finished template (template = fixed, so fields are pull-back)
        fields = []
        with tempfile.TemporaryDirectory() as folder:
            for index, image in enumerate(images):
                result = ants.registration(template, image, type_of_transform=transform,
                                           outprefix=os.path.join(folder, f'group{index}_'), verbose=verbose)
                fields.append(self._field(result['fwdtransforms'], image, folder, reference=template))

        return template, fields

    def landmark(self, reference_points, moving_points, transform='bspline', mesh_size=4, levels=4,
                 regularization=1e-6, verbose=False):
        """Fit a displacement field to corresponding point pairs (no image intensities used).

        Useful for turning a surface correspondence (e.g. from a mesh registration) into a volumetric DVF. Only the
        reference image grid is needed; the moving image is not used.

        Parameters
        ----------
        reference_points : array_like, shape (N, 3)
            Points on the reference, physical (x, y, z) in mm.
        moving_points : array_like, shape (N, 3)
            Corresponding points on the moving image, same order and convention.
        transform : {'bspline', 'tps', 'diffeo', 'syn', 'tv', 'affine', 'rigid'}, default 'bspline'
            'bspline' = one multi-level B-spline fit (fastest). 'tps' = thin-plate spline (passes exactly through
            the points; slow for many thousands of points). 'diffeo' = composition of small B-spline fits
            (invertible). 'syn' = symmetric version of 'diffeo'. 'tv' = time-varying velocity field (largest
            deformations). 'affine'/'rigid' fit a linear transform.
        mesh_size : int, default 4
            B-spline mesh size (control grid cells per axis) at the coarsest level. Not used by 'tps'.
        levels : int, default 4
            Number of multi-level B-spline fitting levels; the mesh doubles at each level. Not used by 'tps'.
        regularization : float, default 1e-6
            Ridge regularization for the 'affine'/'rigid' fits.
        verbose : bool, default False
            Print fitting progress.

        Returns
        -------
        ants.ANTsImage
            3-component displacement field (fixed -> moving) in mm on the full reference grid.

        Examples
        --------
        >>> import ants
        >>> import numpy as np
        >>> helper = DeformableANTs(reference_image=ants.from_numpy(np.zeros((32, 32, 16), dtype=np.float32)))
        >>> reference_points = np.random.default_rng(0).uniform(4, 14, size=(50, 3))
        >>> dvf = helper.landmark(reference_points, reference_points + [2, 0, 0])
        >>> round(float(dvf.numpy()[8, 8, 8, 0]))
        2
        """
        if self.reference_image is None:
            raise ValueError('reference_image must be set before registering.')

        reference_points = np.asarray(reference_points, dtype=np.float64)
        moving_points = np.asarray(moving_points, dtype=np.float64)
        if reference_points.shape != moving_points.shape or reference_points.shape[1] != 3:
            raise ValueError('reference_points and moving_points must both be (N, 3) with matching N.')

        reference = self.reference_image

        # Thin-plate spline: displacement (moving - reference) at each reference point, interpolated over the grid
        if transform == 'tps':
            return ants.fit_thin_plate_spline_displacement_field(
                displacement_origins=reference_points, displacements=moving_points - reference_points,
                origin=reference.origin, spacing=reference.spacing, size=reference.shape,
                direction=reference.direction)

        # Reference points as ANTs "fixed" points gives the pull-back (fixed -> moving) convention
        result = ants.fit_transform_to_paired_points(moving_points=moving_points, fixed_points=reference_points,
                                                     transform_type=transform, regularization=regularization,
                                                     domain_image=reference, number_of_fitting_levels=levels,
                                                     mesh_size=mesh_size, verbose=verbose)
        if isinstance(result, dict):
            result = result['forward_transform']

        # Linear fits are written to file and collapsed like any other transform; deformable fits are fields already
        if transform in ('affine', 'rigid'):
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, 'landmark.mat')
                ants.write_transform(result, path)
                return self._field([path], reference, folder)

        return ants.transform_to_displacement_field(result, reference)

    def registration(self, transform='SyN', metric='CC', radius=4, bins=32, iterations=(100, 70, 50, 20),
                     shrink_factors=(8, 4, 2, 1), smoothing_sigmas=(3, 2, 1, 0), gradient_step=0.1,
                     update_variance=3.0, total_variance=0.0, mesh_spacing=26, time_points=4, convergence=1e-6,
                     affine=False, label_weight=0.0, labels=None, label_sigma=2.0, extra_metrics=None,
                     histogram_match=False, winsorize=(0.005, 0.995), crop=10, seed=42, preset=None, verbose=False):
        """Deformable registration with ``antsRegistration`` (any ANTs deformable transform and metric).

        Runs one deformable stage (optionally after an affine stage) with full control over the transform, metric,
        pyramid and regularization. The defaults reproduce the deformable stage of ``antsRegistrationSyN.sh``
        (SyN with CC, radius 4). Assumes the images are already rigidly aligned unless ``affine=True``.

        Parameters
        ----------
        transform : str, default 'SyN'
            ANTs deformable transform:

            * ``'SyN'``: symmetric diffeomorphic normalization (Gaussian regularization). The standard choice.
            * ``'BSplineSyN'``: SyN with B-spline regularization (smoother, fewer folding artifacts).
            * ``'GaussianDisplacementField'``: dense field, Gaussian regularized (elastic/fluid; not symmetric).
            * ``'BSplineDisplacementField'``: dense field, B-spline regularized.
            * ``'TimeVaryingVelocityField'``: velocity field integrated over time; handles very large
              deformations, but is several times slower than SyN and needs more iterations to converge.
            * ``'TimeVaryingBSplineVelocityField'``: B-spline version of the above.

            ANTs' ``Exponential``/``BSplineExponential`` transforms are not offered: their optimizer barely moves
            with these settings (tested), so SyN is the better diffeomorphic choice.
        metric : {'CC', 'Mattes', 'MI', 'MeanSquares', 'Demons', 'GC'}, default 'CC'
            'CC' = neighborhood cross correlation (best for same modality, robust to intensity drift; slowest).
            'Mattes'/'MI' = mutual information (multimodal, faster). 'MeanSquares'/'Demons' = same modality with
            matching intensities (fastest). 'GC' = global correlation; one value for the whole image, so it is a
            weak driver for local deformation and mainly useful in the ``affine`` stage or as an extra metric.
        radius : int, default 4
            Neighborhood radius in voxels for 'CC'.
        bins : int, default 32
            Histogram bins for 'Mattes' and 'MI'.
        iterations : sequence of int, default (100, 70, 50, 20)
            Iterations per resolution level, coarse to fine.
        shrink_factors : sequence of int, default (8, 4, 2, 1)
            Image downsampling per level; same length as ``iterations``.
        smoothing_sigmas : sequence of float, default (3, 2, 1, 0)
            Gaussian image smoothing per level, in voxels; same length as ``iterations``.
        gradient_step : float, default 0.1
            Step size of each update (0.1-0.25 is typical; larger is faster but less stable).
        update_variance : float, default 3.0
            Gaussian variance (voxels^2) of the update field (fluid-like regularization). Used by 'SyN',
            'GaussianDisplacementField' and 'TimeVaryingVelocityField'.
        total_variance : float, default 0.0
            Gaussian variance (voxels^2) of the total field (elastic-like regularization; 0 = off, as in
            ``antsRegistrationSyN.sh``). Used by 'SyN', 'GaussianDisplacementField' and 'TimeVaryingVelocityField'.
        mesh_spacing : float, default 26
            B-spline control point spacing in mm at the base level (the ``antsRegistrationSyN.sh -s`` default).
            Used by the B-spline transforms.
        time_points : int, default 4
            Number of time points of the velocity field. Used by the time-varying transforms.
        convergence : float, default 1e-6
            Stop a level when the metric changes less than this over the last 10 iterations.
        affine : bool, default False
            Run a masked affine stage (MI) before the deformable stage. The affine is included in the output field.
        label_weight : float, default 0.0
            Label guidance: weight of a MeanSquares term on Gaussian-blurred label images, relative to the intensity
            metric (weight 1.0). The labels drive the deformation where intensity contrast is poor (e.g. liver in
            non-contrast CT). 0 disables. With no ``labels``, the stored ``reference_mask``/``moving_mask`` are used.
            Start around 0.5-1.
        labels : list of tuple, optional
            Label pairs ``(reference_label, moving_label)`` (binary ``ants.ANTsImage``, one pair per structure) used
            when ``label_weight > 0``. Each pair adds its own term with weight ``label_weight``.
        label_sigma : float, default 2.0
            Gaussian blur of the labels in mm before matching, so the term has a gradient beyond the label edge.
        extra_metrics : list of tuple, optional
            Additional metric terms for the deformable stage as ``(metric, fixed_image, moving_image, weight,
            parameter)``, e.g. ``('MeanSquares', fixed_label_blurred, moving_label_blurred, 0.5, 0)``. Images must be
            on the reference grid (they are cropped with the main images). ``parameter`` is the radius for 'CC' and
            the number of bins for 'Mattes'/'MI'.
        histogram_match : bool, default False
            Histogram-match the moving image to the reference (useful for same modality with 'MeanSquares'/'Demons';
            leave off for 'CC'/'MI').
        winsorize : tuple of float or None, default (0.005, 0.995)
            Lower and upper quantiles at which ANTs clips intensities before registering; None disables.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        seed : int or None, default 42
            Random seed for metric sampling, for reproducible results. None uses ANTs' default.
        preset : str, optional
            Instead of the custom stage above, run an ``ants.registration`` preset (e.g. ``'SyNOnly'``,
            ``'ElasticSyN'``, ``'antsRegistrationSyNQuick[so]'``, ``'antsRegistrationSyN[bo]'``). Only
            ``metric``, ``radius``/``bins``, ``iterations``, ``gradient_step``, the variances, masks and ``crop``
            are passed on, and only where that preset uses them; ``transform``, ``affine``, the label and extra
            metric terms, ``winsorize`` and ``seed`` are not used. Presets with an affine/rigid stage include it in
            the output field. ``Quick`` presets skip the full-resolution level, which can leave small cropped
            regions almost unregistered.
        verbose : bool, default False
            Print the ANTs command line and progress.

        Returns
        -------
        ants.ANTsImage
            3-component displacement field (fixed -> moving) in mm on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableANTs()
        >>> _ = helper.create_ants_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_ants_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.registration(metric='MeanSquares', iterations=(40, 20, 10), shrink_factors=(4, 2, 1),
        ...                           smoothing_sigmas=(2, 1, 0))
        >>> dvf.shape, round(float(dvf.numpy()[16, 16, 8, 0]))
        ((32, 32, 16), 2)
        >>> dvf = helper.registration(transform='BSplineSyN', metric='MeanSquares', mesh_spacing=10,
        ...                           iterations=(40, 20, 10), shrink_factors=(4, 2, 1), smoothing_sigmas=(2, 1, 0))
        >>> round(float(dvf.numpy()[16, 16, 8, 0]))
        2

        Label-guided, using the stored masks:

        >>> mask = (fixed > 30).astype(np.uint8)
        >>> _ = helper.create_ants_image(mask, (0, 0, 0), (1, 1, 2), np.eye(3), mask=True)
        >>> _ = helper.create_ants_image(np.roll(mask, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False,
        ...                              mask=True)
        >>> dvf = helper.registration(metric='MeanSquares', label_weight=1.0, iterations=(40, 20, 10),
        ...                           shrink_factors=(4, 2, 1), smoothing_sigmas=(2, 1, 0))
        >>> round(float(dvf.numpy()[16, 16, 8, 0]))
        2
        """
        fixed, moving, fixed_mask, moving_mask = self._prepare(crop=crop, histogram_match=histogram_match)

        with tempfile.TemporaryDirectory() as folder:
            prefix = os.path.join(folder, 'reg_')

            # Preset path: let ANTsPy build the command line for its named registration types
            if preset is not None:
                result = ants.registration(fixed, moving, type_of_transform=preset, outprefix=prefix,
                                           mask=fixed_mask, moving_mask=moving_mask, grad_step=gradient_step,
                                           flow_sigma=update_variance, total_sigma=total_variance,
                                           syn_metric=metric, syn_sampling=radius if metric == 'CC' else bins,
                                           reg_iterations=tuple(iterations), verbose=verbose)
                return self._field(result['fwdtransforms'], self.moving_image, folder)

            # Keep references to every image passed by pointer, so none is freed while ANTs runs
            keep = [fixed, moving, fixed_mask, moving_mask]

            def metric_string(name, fixed_image, moving_image, weight, parameter):
                # ANTs metric syntax: Name[fixed, moving, weight, radius/bins(, sampling strategy, percentage)]
                keep.extend([fixed_image, moving_image])
                pointers = f'{get_pointer_string(fixed_image)},{get_pointer_string(moving_image)},{weight}'
                if name in ('Mattes', 'MI'):
                    return f'{name}[{pointers},{parameter},Regular,0.25]'
                if name == 'CC':
                    return f'CC[{pointers},{parameter}]'
                if name in ('MeanSquares', 'Demons', 'GC'):
                    return f'{name}[{pointers},0]'
                raise ValueError(f'Unknown metric: {name}')

            # The time-varying B-spline transform takes a mesh size (cells per axis + time), not a spacing in mm,
            # so convert mesh_spacing using the physical extent of the (possibly cropped) image
            velocity_mesh = 'x'.join(str(max(1, int(round(size * spacing / mesh_spacing))))
                                     for size, spacing in zip(fixed.shape, fixed.spacing)) + f'x{time_points}'

            # Transform syntax per ANTs; see antsRegistration --help
            transforms = {
                'SyN': f'SyN[{gradient_step},{update_variance},{total_variance}]',
                'BSplineSyN': f'BSplineSyN[{gradient_step},{mesh_spacing},0,3]',
                'GaussianDisplacementField': f'GaussianDisplacementField[{gradient_step},{update_variance},'
                                             f'{total_variance}]',
                'BSplineDisplacementField': f'BSplineDisplacementField[{gradient_step},{mesh_spacing},0,3]',
                'TimeVaryingVelocityField': f'TimeVaryingVelocityField[{gradient_step},{time_points},'
                                            f'{update_variance},0,{total_variance},0]',
                'TimeVaryingBSplineVelocityField': f'TimeVaryingBSplineVelocityField[{gradient_step},{velocity_mesh},'
                                                   f'{time_points},3]',
            }

            if transform not in transforms:
                raise ValueError(f'Unknown transform: {transform}')

            # Masks (binary, nonzero = inside); 'NA' tells ANTs there is no mask for that image
            masks = (f'[{get_pointer_string(fixed_mask) if fixed_mask is not None else "NA"},'
                     f'{get_pointer_string(moving_mask) if moving_mask is not None else "NA"}]')
            levels = len(iterations)

            args = ['--dimensionality', str(fixed.dimension), '--float', '1',
                    '--output', prefix, '--write-composite-transform', '1', '--collapse-output-transforms', '1',
                    '--interpolation', 'Linear', '--use-histogram-matching', '0',
                    '--initial-moving-transform', 'Identity']

            if winsorize is not None:
                args += ['--winsorize-image-intensities', f'[{winsorize[0]},{winsorize[1]}]']

            if seed is not None:
                args += ['--random-seed', str(seed)]

            # Optional affine stage (masked, mutual information), standard ANTs pyramid
            if affine:
                args += ['--transform', 'Affine[0.1]',
                         '--metric', metric_string('MI', fixed, moving, 1, 32),
                         '--convergence', '[1000x500x250x100,1e-6,10]',
                         '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox',
                         '--masks', masks]

            # Deformable stage: main metric plus any extra metric terms
            args += ['--transform', transforms[transform],
                     '--metric', metric_string(metric, fixed, moving, 1, radius if metric == 'CC' else bins)]
            # Label guidance: MeanSquares on blurred labels (default: the stored masks), one term per label pair
            label_pairs = list(labels) if labels is not None else [(self.reference_mask, self.moving_mask)]
            terms = list(extra_metrics or [])
            if label_weight > 0:
                for reference_label, moving_label in label_pairs:
                    if reference_label is None or moving_label is None:
                        raise ValueError('label_weight > 0 needs labels, or reference_mask and moving_mask.')
                    terms.append(('MeanSquares', ants.smooth_image(reference_label.clone('float'), label_sigma),
                                  ants.smooth_image(moving_label.clone('float'), label_sigma), label_weight, 0))

            for name, fixed_extra, moving_extra, weight, parameter in terms:
                # Extra images are brought onto the (possibly cropped) working grid
                fixed_extra = ants.resample_image_to_target(fixed_extra.clone('float'), fixed)
                moving_extra = ants.resample_image_to_target(moving_extra.clone('float'), fixed)
                args += ['--metric', metric_string(name, fixed_extra, moving_extra, weight, parameter)]

            args += ['--convergence', f'[{"x".join(str(int(i)) for i in iterations)},{convergence},10]',
                     '--shrink-factors', 'x'.join(str(int(s)) for s in shrink_factors[:levels]),
                     '--smoothing-sigmas', 'x'.join(str(s) for s in smoothing_sigmas[:levels]) + 'vox',
                     '--masks', masks]

            if verbose:
                args += ['--verbose', '1']

            exit_code = get_lib_fn('antsRegistration')(process_arguments(args))
            if exit_code != 0:
                raise RuntimeError(f'antsRegistration failed with exit code {exit_code}; rerun with verbose=True.')

            # The composite file holds every stage (affine + deformable); collapse it onto the full reference grid
            return self._field([prefix + 'Composite.h5'], self.moving_image, folder)


