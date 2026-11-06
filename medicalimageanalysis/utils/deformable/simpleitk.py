"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Deformable image registration wrappers around SimpleITK.

    Methods
    -------
    * ImageRegistrationMethod: :meth:`DeformableITK.bspline` (multi-level grid refinement),
      :meth:`DeformableITK.displacement_field` (dense field, native masks)
    * Landmarks: :meth:`DeformableITK.landmark` (B-spline fitted to point pairs)
    * Elastix: :meth:`DeformableITK.elastix` (BSpline / RecursiveBSpline / SplineKernel, bending energy, point
      correspondences), :meth:`DeformableITK.elastix_groupwise`
    * Demons family: :meth:`DeformableITK.demons`, :meth:`DeformableITK.symmetric_demons`,
      :meth:`DeformableITK.fast_demons`, :meth:`DeformableITK.diffeomorphic`, :meth:`DeformableITK.level_set_motion`

    Conventions
    -----------
    * NumPy arrays are indexed (z, y, x). SimpleITK origin, spacing, direction and landmark points are (x, y, z).
    * Every registration method returns a ``sitk.sitkVectorFloat32`` displacement field on the **full reference
      image grid** (even when mask cropping is used). Vector components are (dx, dy, dz) in mm.
    * The field is a pull-back (fixed -> moving) field: for a point ``p`` in the reference image, the corresponding
      point in the moving image is ``p + dvf(p)``.
    * The stored images and masks are never modified. Preprocessing is applied to working copies inside each call.
    * Masks: the ImageRegistrationMethod and Elastix methods restrict the metric to the masks. The Demons family has
      no mask input, so the area outside the (blurred) union of both masks is blended to one constant in **both**
      images. The boundary is then in the same place in both images and creates no registration force, so the
      anatomy inside the masks drives the result rather than the mask outlines.

Examples
--------
>>> import numpy as np
>>> from medicalimageanalysis.utils.deformable.simpleitk import DeformableITK
>>> z, y, x = np.mgrid[:16, :32, :32]
>>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
>>> helper = DeformableITK()
>>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
>>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
>>> dvf = helper.fast_demons()
>>> dvf.GetSize(), dvf.GetNumberOfComponentsPerPixel()
((32, 32, 16), 3)
"""

import os
import tempfile

import numpy as np
import SimpleITK as sitk


class DeformableITK:
    """Deformable registration between a reference (fixed) and moving image, with optional masks.

    All image-based methods share the same preprocessing path (:meth:`_prepare`): the moving image and masks are
    resampled onto the reference grid, then optionally converted to gradient magnitude, intensity normalized,
    histogram matched, mask blended (Demons family only) and cropped to the union of both masks.

    Parameters
    ----------
    reference_image : SimpleITK.Image, optional
        Fixed image. Its grid defines the output displacement field.
    moving_image : SimpleITK.Image, optional
        Image to be registered onto the reference.
    reference_mask : SimpleITK.Image, optional
        Region of interest on the reference image. Any nonzero voxel is treated as inside.
    moving_mask : SimpleITK.Image, optional
        Region of interest on the moving image. Any nonzero voxel is treated as inside.

    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.deformable.simpleitk import DeformableITK
    >>> fixed = np.zeros((16, 32, 32), dtype=np.float32)
    >>> fixed[4:12, 8:24, 8:24] = 100
    >>> helper = DeformableITK()
    >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
    >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
    >>> helper.reference_image.GetSize()
    (32, 32, 16)
    """

    def __init__(self, reference_image=None, moving_image=None, reference_mask=None, moving_mask=None):
        """Store SimpleITK images and optional masks for later registration calls.

        Parameters
        ----------
        reference_image, moving_image : SimpleITK.Image, optional
            Fixed and moving images.
        reference_mask, moving_mask : SimpleITK.Image, optional
            Binary masks (nonzero = inside). Stored as ``sitkUInt8``.

        Examples
        --------
        >>> import SimpleITK as sitk
        >>> image = sitk.Image(32, 32, 16, sitk.sitkFloat32)
        >>> helper = DeformableITK(reference_image=image, moving_image=image)
        >>> helper.moving_mask is None
        True
        """
        self.reference_image = reference_image
        self.moving_image = moving_image

        # Masks are normalized to binary UInt8 up front, which is what ITK and Elastix mask APIs expect
        self.reference_mask = None if reference_mask is None else sitk.Cast(reference_mask != 0, sitk.sitkUInt8)
        self.moving_mask = None if moving_mask is None else sitk.Cast(moving_mask != 0, sitk.sitkUInt8)

    def _demons(self, filter_class, options, std, iterations, intensity_threshold, shrink_factors, smoothing_sigmas,
                crop, cross_modality, normalize, histogram_match, blend_sigma):
        """Shared multi-resolution driver for the Demons family (one public method per algorithm calls this).

        Each level smooths and downsamples both images, upsamples the previous level's field onto the new grid, and
        uses it to initialize the next run. Displacements are in mm, so no rescaling is needed between levels.

        Parameters
        ----------
        filter_class : type
            A SimpleITK PDE deformable registration filter class (e.g. ``sitk.DemonsRegistrationFilter``).
        options : dict
            Extra filter settings as ``{'SetterNameWithoutSet': value}``, e.g. ``{'MaximumUpdateStepLength': 1.0}``.
        std : float
            Gaussian regularization of the displacement field, in **voxels**.
        iterations : int
            Iterations per resolution level.
        intensity_threshold : float
            Intensity differences below this are treated as matched (normalized [0, 1] units when ``normalize``).
        shrink_factors : sequence of int
            Downsampling factor per level.
        smoothing_sigmas : sequence of float
            Image smoothing per level, in mm; same length as ``shrink_factors``.
        crop : float
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool
            Register gradient magnitude images instead of raw intensities.
        normalize : bool
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`).
        histogram_match : bool
            Histogram-match the moving image to the reference.
        blend_sigma : float
            Blur (mm) of the mask-union edge used to blend the outside region; 0 uses a hard edge.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper._demons(sitk.FastSymmetricForcesDemonsRegistrationFilter, {'MaximumUpdateStepLength': 1.0},
        ...                      1.5, 50, 0.001, (2, 1), (1, 0), 10, False, True, True, 2)
        >>> dvf.GetSize()
        (32, 32, 16)
        """
        # Demons has no mask input: normalize, histogram match and blend outside the mask union in both images
        fixed, moving, _, _ = self._prepare(crop=crop, cross_modality=cross_modality, normalize=normalize,
                                            histogram_match=histogram_match, blend_masks=True, blend_sigma=blend_sigma)

        demons = filter_class()
        demons.SetNumberOfIterations(iterations)

        # Field regularization: Gaussian std in voxels (ITK applies it per axis without image spacing)
        demons.SetStandardDeviations(std)
        demons.SmoothDisplacementFieldOn()
        demons.SetIntensityDifferenceThreshold(intensity_threshold)
        for name, value in options.items():
            getattr(demons, 'Set' + name)(value)

        field = None
        for shrink, sigma in zip(shrink_factors, smoothing_sigmas):
            # Cap the shrink per axis so thin axes (few slices) are never reduced below 2 voxels
            factors = [max(1, min(int(shrink), size // 2)) for size in fixed.GetSize()]

            level_fixed = sitk.SmoothingRecursiveGaussian(fixed, sigma) if sigma > 0 else fixed
            level_moving = sitk.SmoothingRecursiveGaussian(moving, sigma) if sigma > 0 else moving
            level_fixed = sitk.Shrink(level_fixed, factors)
            level_moving = sitk.Shrink(level_moving, factors)

            if field is None:
                field = demons.Execute(level_fixed, level_moving)
            else:
                # Upsample the previous level's field to this level's grid to initialize the next run
                field = sitk.Resample(field, level_fixed, sitk.Transform(), sitk.sitkLinear, 0.0, field.GetPixelID())
                field = demons.Execute(level_fixed, level_moving, field)

        # Final level may still be coarser than the reference (if shrink_factors doesn't end at 1); resample handles it
        dvf = sitk.Cast(field, sitk.sitkVectorFloat32)
        return sitk.Resample(dvf, self.reference_image, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkVectorFloat32)

    def _prepare(self, crop=10, cross_modality=False, normalize=False, histogram_match=False, blend_masks=False,
                 blend_sigma=2):
        """Build working copies of the images/masks for a registration call (stored data is left untouched).

        Steps, in order: cast to float -> resample moving image and masks onto the reference grid -> gradient
        magnitude -> intensity normalization -> histogram matching -> mask blending -> crop to mask union.

        Parameters
        ----------
        crop : float, default 10
            Crop margin in mm. ``0`` disables cropping. Only applied when both masks exist.
        cross_modality : bool, default False
            Apply :meth:`cross_modality_correction` to both images.
        normalize : bool, default False
            Clip each image to its 0.5-99.5 percentile range (inside the mask union, if masks exist) and rescale to
            [0, 1]. Makes intensity thresholds and step sizes comparable across CT/MR.
        histogram_match : bool, default False
            Histogram-match the moving image to the reference.
        blend_masks : bool, default False
            Blend the area outside the (blurred) mask union to the median reference intensity inside it, in both
            images. Used by the Demons family, which has no mask input.
        blend_sigma : float, default 2
            Blur of the union mask in mm before blending; ``0`` uses a hard edge.

        Returns
        -------
        tuple
            ``(fixed, moving, fixed_mask, moving_mask)``; masks may be None.

        Examples
        --------
        >>> import numpy as np
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(np.random.rand(16, 32, 32).astype(np.float32), (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.random.rand(16, 32, 32).astype(np.float32), (0, 0, 0), (1, 1, 2), np.eye(3),
        ...                              reference=False)
        >>> fixed, moving, fixed_mask, moving_mask = helper._prepare(normalize=True)
        >>> fixed.GetSize() == moving.GetSize(), fixed_mask is None
        (True, True)
        >>> 0 <= float(sitk.GetArrayViewFromImage(fixed).min()) and float(sitk.GetArrayViewFromImage(fixed).max()) <= 1
        True
        """
        if self.reference_image is None or self.moving_image is None:
            raise ValueError('Both reference_image and moving_image must be set before registering.')

        # Everything is brought onto the reference grid so index-space operations (crop, Demons) are valid
        fixed = sitk.Cast(self.reference_image, sitk.sitkFloat32)
        moving = self.resample(sitk.Cast(self.moving_image, sitk.sitkFloat32), fixed)
        fixed_mask = None if self.reference_mask is None else self.resample(self.reference_mask, fixed, mask=True)
        moving_mask = None if self.moving_mask is None else self.resample(self.moving_mask, fixed, mask=True)

        # Union of whichever masks exist; used for normalization statistics and blending
        union = None
        if fixed_mask is not None and moving_mask is not None:
            union = sitk.Or(fixed_mask, moving_mask)

        elif fixed_mask is not None or moving_mask is not None:
            union = fixed_mask if fixed_mask is not None else moving_mask

        union_array = None if union is None else sitk.GetArrayViewFromImage(union) > 0
        if union_array is not None and not union_array.any():
            union, union_array = None, None

        # Gradient magnitude before cropping so the crop boundary doesn't create artificial edges
        if cross_modality:
            fixed = self.cross_modality_correction(fixed)
            moving = self.cross_modality_correction(moving)

        # Robust percentile normalization to [0, 1]; statistics from inside the masks so background/air is ignored
        if normalize:
            normalized = []
            for image in (fixed, moving):
                values = sitk.GetArrayViewFromImage(image)
                values = values[union_array] if union_array is not None else values
                low, high = np.percentile(values, [0.5, 99.5])
                if high <= low:
                    high = low + 1.0

                image = sitk.Clamp(image, sitk.sitkFloat32, float(low), float(high))
                image = sitk.Cast((image - float(low)) / float(high - low), sitk.sitkFloat32)
                normalized.append(sitk.Clamp(image, sitk.sitkFloat32, 0.0, 1.0))
            fixed, moving = normalized

        if histogram_match:
            moving = sitk.HistogramMatching(moving, fixed, numberOfHistogramLevels=256, numberOfMatchPoints=15,
                                            thresholdAtMeanIntensity=False)

        # The same weight is used for both images, so the blended boundary is identical in both and adds no force.
        # Blending to the median (instead of multiplying to 0) also avoids a strong edge at the mask boundary.
        if blend_masks and union is not None:
            weight = self.blur_mask(union, blend_sigma) if blend_sigma > 0 else sitk.Cast(union, sitk.sitkFloat32)
            fill = float(np.median(sitk.GetArrayViewFromImage(fixed)[union_array]))
            fixed = sitk.Cast(fixed * weight + fill * (1 - weight), sitk.sitkFloat32)
            moving = sitk.Cast(moving * weight + fill * (1 - weight), sitk.sitkFloat32)

        if crop > 0 and fixed_mask is not None and moving_mask is not None:
            fixed, moving, fixed_mask, moving_mask = self.mask_crop([fixed, moving, fixed_mask, moving_mask],
                                                                    fixed_mask, moving_mask, margin=crop)

        return fixed, moving, fixed_mask, moving_mask

    def blur_mask(self, mask, sigma=2):
        """Return a Gaussian-smoothed mask normalized to [0, 1], used as a soft blending weight.

        Parameters
        ----------
        mask : SimpleITK.Image
            Binary mask.
        sigma : float, default 2
            Gaussian standard deviation in mm.

        Returns
        -------
        SimpleITK.Image
            Float32 weight image in [0, 1]. An empty or uniform mask is returned unchanged (as float).

        Examples
        --------
        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> helper = DeformableITK()
        >>> array = np.zeros((16, 32, 32), dtype=np.uint8)
        >>> array[4:12, 8:24, 8:24] = 1
        >>> weight = helper.blur_mask(sitk.GetImageFromArray(array), sigma=2)
        >>> float(sitk.GetArrayViewFromImage(weight).max()), weight.GetPixelIDTypeAsString()
        (1.0, '32-bit float')
        """
        blurred = sitk.SmoothingRecursiveGaussian(sitk.Cast(mask, sitk.sitkFloat32), sigma=sigma)

        # Guard against divide-by-zero on empty/uniform masks
        view = sitk.GetArrayViewFromImage(blurred)
        min_val, max_val = float(view.min()), float(view.max())
        if max_val - min_val <= 0:
            return sitk.Cast(mask, sitk.sitkFloat32)

        # Arithmetic with Python floats promotes to float64; cast back so it can multiply Float32 images
        return sitk.Cast((blurred - min_val) / (max_val - min_val), sitk.sitkFloat32)

    def bspline(self, control_spacing=(20, 20, 20), scale_factors=(1, 2, 4), shrink_factors=(4, 2, 1),
                smoothing_sigmas=(2, 1, 0), iterations=100, accuracy=1e-5, metric='Intensity', bins=32, radius=2,
                sampling=0.1, crop=10, cross_modality=False):
        """Multi-resolution B-spline registration with control-grid refinement (ImageRegistrationMethod + LBFGS2).

        The control grid starts coarse and is refined by ``scale_factors`` at each pyramid level, so large
        deformations are captured first and local detail last.

        Parameters
        ----------
        control_spacing : sequence of float, default (20, 20, 20)
            **Final** control point spacing in mm, (x, y, z). The first level uses
            ``control_spacing * max(scale_factors)``.
        scale_factors : sequence of int, default (1, 2, 4)
            Control grid refinement per level; must have the same length as ``shrink_factors``.
        shrink_factors : sequence of int, default (4, 2, 1)
            Image downsampling per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Gaussian image smoothing per level, in mm.
        iterations : int, default 100
            Maximum LBFGS2 iterations per level.
        accuracy : float, default 1e-5
            LBFGS2 solution accuracy (convergence tolerance on the gradient norm).
        metric : {'Intensity', 'Correlation', 'ANTS', 'MI', 'JointMI'}, default 'Intensity'
            'Intensity' = mean squares (same modality, fastest). 'ANTS' = ANTS neighborhood correlation (robust for
            CT-CT/MR-MR with intensity drift). 'MI'/'JointMI' = Mattes / joint histogram mutual information
            (multimodal).
        bins : int, default 32
            Histogram bins for 'MI' and 'JointMI'.
        radius : int, default 2
            Neighborhood radius in voxels for 'ANTS'.
        sampling : float, default 0.1
            Fraction of voxels randomly sampled for the metric. ``1.0`` uses every voxel.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.bspline(control_spacing=(8, 8, 8), sampling=1.0)
        >>> dvf.GetSize(), dvf.GetPixelIDTypeAsString()
        ((32, 32, 16), 'vector of 32-bit float')
        >>> round(float(sitk.GetArrayViewFromImage(dvf)[8, 16, 16, 0]))
        2
        """
        fixed, moving, fixed_mask, moving_mask = self._prepare(crop=crop, cross_modality=cross_modality)

        # Initial (coarsest) mesh; refined by scale_factors so the last level reaches control_spacing.
        # At least 1 cell per axis so thin axes don't produce a zero mesh.
        coarsest = max(scale_factors)
        mesh_size = [max(1, int(round(size * spacing / (control * coarsest))))
                     for size, spacing, control in zip(fixed.GetSize(), fixed.GetSpacing(), control_spacing)]
        transform = sitk.BSplineTransformInitializer(fixed, mesh_size, order=3)

        registration = sitk.ImageRegistrationMethod()
        if metric == 'Intensity':
            registration.SetMetricAsMeanSquares()

        elif metric == 'Correlation':
            registration.SetMetricAsCorrelation()

        elif metric == 'ANTS':
            registration.SetMetricAsANTSNeighborhoodCorrelation(radius=radius)

        elif metric == 'MI':
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)

        elif metric == 'JointMI':
            registration.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=bins)

        else:
            raise ValueError(f'Unknown metric: {metric}')

        # Random sampling is the biggest speed win; fixed seed keeps runs reproducible
        if sampling < 1.0:
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(sampling, 42)

        # Masks restrict where the metric is evaluated (binary UInt8, nonzero = inside)
        if fixed_mask is not None:
            registration.SetMetricFixedMask(fixed_mask)

        if moving_mask is not None:
            registration.SetMetricMovingMask(moving_mask)

        # LBFGS2 (not LBFGSB): LBFGSB can't handle the parameter count changing when the grid is refined
        registration.SetOptimizerAsLBFGS2(solutionAccuracy=accuracy, numberOfIterations=iterations,
                                          deltaConvergenceTolerance=0.01)

        # Coarse-to-fine image pyramid
        registration.SetShrinkFactorsPerLevel(list(shrink_factors))
        registration.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Control-grid refinement per level (replaces SetInitialTransform)
        registration.SetInitialTransformAsBSpline(transform, inPlace=True, scaleFactors=list(scale_factors))
        registration.SetInterpolator(sitk.sitkLinear)
        registration.Execute(fixed, moving)

        dvf = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat32, fixed.GetSize(),
                                                fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())

        # Place the (possibly cropped) field back on the full reference grid; zero displacement outside the crop
        return sitk.Resample(dvf, self.reference_image, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkVectorFloat32)

    def create_sitk_image(self, array, origin, spacing, direction, reference=True, mask=False):
        """Convert a NumPy array to a SimpleITK image and store it as the reference/moving image or mask.

        Parameters
        ----------
        array : numpy.ndarray
            Volume indexed (z, y, x).
        origin : sequence of float
            Physical position of voxel (0, 0, 0), in (x, y, z) order.
        spacing : sequence of float
            Voxel size in (x, y, z) order, i.e. the slice spacing is the **last** entry.
        direction : numpy.ndarray
            3x3 direction cosine matrix whose columns are the x, y, z axis directions.
        reference : bool, default True
            Store as reference (True) or moving (False).
        mask : bool, default False
            Store as a mask. Masks are binarized (nonzero = inside) and cast to ``sitkUInt8``.

        Returns
        -------
        SimpleITK.Image
            The created (and stored) image.

        Examples
        --------
        >>> import numpy as np
        >>> helper = DeformableITK()
        >>> image = helper.create_sitk_image(np.zeros((16, 32, 32), dtype=np.float32), (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> image.GetSize(), image.GetSpacing()
        ((32, 32, 16), (1.0, 1.0, 2.0))
        >>> mask = helper.create_sitk_image(np.ones((16, 32, 32)), (0, 0, 0), (1, 1, 2), np.eye(3), mask=True)
        >>> mask.GetPixelIDTypeAsString()
        '8-bit unsigned integer'
        """
        image = sitk.GetImageFromArray(array)
        image.SetOrigin([float(v) for v in origin])
        image.SetSpacing([float(v) for v in spacing])
        image.SetDirection(np.asarray(direction, dtype=np.float64).flatten().tolist())

        if mask:
            image = sitk.Cast(image != 0, sitk.sitkUInt8)

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

    def cross_modality_correction(self, image):
        """Return the gradient magnitude of an image to reduce intensity differences between modalities.

        For multimodal registration, prefer ``metric='MI'`` in :meth:`bspline`, :meth:`displacement_field` or
        :meth:`elastix`; this is mainly useful for the Demons family, which assumes matching intensities.

        Parameters
        ----------
        image : SimpleITK.Image
            Input image.

        Returns
        -------
        SimpleITK.Image
            Float32 gradient magnitude image.

        Examples
        --------
        >>> import SimpleITK as sitk
        >>> helper = DeformableITK()
        >>> gradient = helper.cross_modality_correction(sitk.Image(32, 32, 16, sitk.sitkFloat32))
        >>> gradient.GetPixelIDTypeAsString()
        '32-bit float'
        """
        return sitk.GradientMagnitude(sitk.Cast(image, sitk.sitkFloat32))

    def demons(self, std=1.5, iterations=100, intensity_threshold=0.001, shrink_factors=(4, 2, 1),
               smoothing_sigmas=(2, 1, 0), crop=10, cross_modality=False, normalize=True, histogram_match=True,
               blend_sigma=2):
        """Classic (Thirion) Demons registration, multi-resolution.

        Parameters
        ----------
        std : float, default 1.5
            Gaussian regularization of the displacement field, in **voxels**.
        iterations : int, default 100
            Iterations per resolution level.
        intensity_threshold : float, default 0.001
            Intensity differences below this are treated as matched (normalized [0, 1] units when ``normalize``).
        shrink_factors : sequence of int, default (4, 2, 1)
            Downsampling factor per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Image smoothing per level, in mm.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.
        normalize : bool, default True
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`).
        histogram_match : bool, default True
            Histogram-match the moving image to the reference.
        blend_sigma : float, default 2
            Blur (mm) of the mask-union edge used to blend the outside region; 0 uses a hard edge.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.demons()
        >>> dvf.GetSize()
        (32, 32, 16)
        """
        return self._demons(sitk.DemonsRegistrationFilter, {}, std, iterations, intensity_threshold, shrink_factors,
                            smoothing_sigmas, crop, cross_modality, normalize, histogram_match, blend_sigma)

    def diffeomorphic(self, std=1.5, iterations=100, intensity_threshold=0.001, step=0.5, shrink_factors=(4, 2, 1),
                      smoothing_sigmas=(2, 1, 0), crop=10, cross_modality=False, normalize=True, histogram_match=True,
                      blend_sigma=2):
        """Diffeomorphic Demons registration (invertible, fold-free fields), multi-resolution.

        Parameters
        ----------
        std : float, default 1.5
            Gaussian regularization of the displacement field, in **voxels**.
        iterations : int, default 100
            Iterations per resolution level.
        intensity_threshold : float, default 0.001
            Intensity differences below this are treated as matched (normalized [0, 1] units when ``normalize``).
        step : float, default 0.5
            Maximum update step length per iteration, in voxels. Kept at the ITK default because large steps can
            break the diffeomorphic update.
        shrink_factors : sequence of int, default (4, 2, 1)
            Downsampling factor per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Image smoothing per level, in mm.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.
        normalize : bool, default True
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`).
        histogram_match : bool, default True
            Histogram-match the moving image to the reference.
        blend_sigma : float, default 2
            Blur (mm) of the mask-union edge used to blend the outside region; 0 uses a hard edge.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.diffeomorphic()
        >>> dvf.GetSize()
        (32, 32, 16)
        """
        return self._demons(sitk.DiffeomorphicDemonsRegistrationFilter, {'MaximumUpdateStepLength': step}, std,
                            iterations, intensity_threshold, shrink_factors, smoothing_sigmas, crop, cross_modality,
                            normalize, histogram_match, blend_sigma)


    def displacement_field(self, metric='Demons', bins=32, radius=2, iterations=200, learning_rate=1.0,
                           update_variance=0.0, total_variance=3.0, shrink_factors=(4, 2, 1),
                           smoothing_sigmas=(2, 1, 0), crop=10, cross_modality=False, normalize=True):
        """Dense displacement field registration (ImageRegistrationMethod + gradient descent).

        Every voxel has its own displacement, regularized by Gaussian smoothing of the update and/or total field.
        Unlike the Demons family, any metric can be used and the masks restrict the metric natively. Slower than
        Demons, but the most flexible dense option. ITK's gradient descent converges slowly on smooth anatomy, so
        use it for moderate deformations or as a refinement, and raise ``iterations`` for 'ANTS'/'MI'.

        Parameters
        ----------
        metric : {'Demons', 'ANTS', 'Intensity', 'Correlation', 'MI', 'JointMI'}, default 'Demons'
            Similarity metric. 'Demons' converges fastest for same modality. 'ANTS' (neighborhood correlation) is
            more robust to local intensity changes but needs more iterations. 'MI'/'JointMI' for multimodal.
        bins : int, default 32
            Histogram bins for 'MI' and 'JointMI'.
        radius : int, default 2
            Neighborhood radius in voxels for 'ANTS'.
        iterations : int, default 200
            Maximum gradient descent iterations per level.
        learning_rate : float, default 1.0
            Gradient descent step; scaled per voxel by ``SetOptimizerScalesFromPhysicalShift``.
        update_variance : float, default 0.0
            Gaussian variance (voxels^2) applied to each update (fluid-like regularization). 0 disables.
        total_variance : float, default 3.0
            Gaussian variance (voxels^2) applied to the total field (elastic-like regularization). Larger is
            smoother but recovers less of large deformations.
        shrink_factors : sequence of int, default (4, 2, 1)
            Image downsampling per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Gaussian image smoothing per level, in mm.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.
        normalize : bool, default True
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`); keeps the step size and the
            'Demons'/'Intensity' metrics on a consistent scale.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.displacement_field(iterations=50)
        >>> dvf.GetSize()
        (32, 32, 16)
        """
        fixed, moving, fixed_mask, moving_mask = self._prepare(crop=crop, cross_modality=cross_modality,
                                                               normalize=normalize)

        registration = sitk.ImageRegistrationMethod()
        if metric == 'ANTS':
            registration.SetMetricAsANTSNeighborhoodCorrelation(radius=radius)

        elif metric == 'Demons':
            registration.SetMetricAsDemons()

        elif metric == 'Intensity':
            registration.SetMetricAsMeanSquares()

        elif metric == 'Correlation':
            registration.SetMetricAsCorrelation()

        elif metric == 'MI':
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)

        elif metric == 'JointMI':
            registration.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=bins)

        else:
            raise ValueError(f'Unknown metric: {metric}')

        # Native metric masks (no image multiplication needed)
        if fixed_mask is not None:
            registration.SetMetricFixedMask(fixed_mask)
        if moving_mask is not None:
            registration.SetMetricMovingMask(moving_mask)

        # Zero initial field on the (cropped) reference grid; SimpleITK adapts it to each pyramid level
        field = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
        field.CopyInformation(fixed)
        transform = sitk.DisplacementFieldTransform(field)
        transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=update_variance,
                                               varianceForTotalField=total_variance)

        registration.SetInitialTransform(transform, inPlace=True)
        registration.SetInterpolator(sitk.sitkLinear)
        # Wide convergence window: with a short window the checker stops dense fields after ~10 iterations.
        # Learning rate re-estimated every iteration so each step moves at most about one voxel.
        registration.SetOptimizerAsGradientDescent(learningRate=learning_rate, numberOfIterations=iterations,
                                                   convergenceMinimumValue=1e-6, convergenceWindowSize=50,
                                                   estimateLearningRate=registration.EachIteration)
        registration.SetOptimizerScalesFromPhysicalShift()

        registration.SetShrinkFactorsPerLevel(list(shrink_factors))
        registration.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.Execute(fixed, moving)

        dvf = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat32, fixed.GetSize(),
                                                fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())

        # Place the (possibly cropped) field back on the full reference grid
        return sitk.Resample(dvf, self.reference_image, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkVectorFloat32)

    def elastix(self, transform='BSpline', metric='Intensity', bins=32, resolution=4, spacing=10, iterations=500,
                order=3, bending_weight=0.0, reference_points=None, moving_points=None, points_weight=0.1,
                parameter=None, crop=10, cross_modality=False):
        """Nonrigid registration with Elastix; the deformation field is computed with Transformix.

        Assumes the images are already rigidly/affinely aligned. Requires a SimpleITK build with Elastix
        (``SimpleITK-SimpleElastix`` on PyPI).

        Parameters
        ----------
        transform : {'BSpline', 'RecursiveBSpline', 'SplineKernel'}, default 'BSpline'
            'RecursiveBSpline' gives the same result as 'BSpline' but is faster. 'SplineKernel' is a thin-plate
            spline through the landmarks; it requires ``reference_points`` and ``moving_points`` and refines their
            positions with the intensity metric.
        metric : {'Intensity', 'MI'}, default 'Intensity'
            ``AdvancedMeanSquares`` or ``AdvancedMattesMutualInformation``.
        bins : int, default 32
            Histogram bins (mutual information only).
        resolution : int, default 4
            Number of pyramid levels.
        spacing : float, default 10
            Final B-spline grid spacing in mm (B-spline transforms only).
        iterations : int, default 500
            Maximum iterations per level.
        order : int, default 3
            B-spline order (B-spline transforms only).
        bending_weight : float, default 0.0
            Weight of ``TransformBendingEnergyPenalty``, which penalizes non-smooth deformations. 0 disables. The
            right value depends on the metric scale; start around 0.01-1.
        reference_points, moving_points : array_like, shape (N, 3), optional
            Corresponding points, physical (x, y, z) in mm. With a B-spline transform they add a
            ``CorrespondingPointsEuclideanDistanceMetric`` term; with 'SplineKernel' they are the spline landmarks.
        points_weight : float, default 0.1
            Weight of the corresponding-points term relative to the intensity metric (weight 1.0).
        parameter : SimpleITK.ParameterMap, optional
            Full Elastix parameter map. When given, all settings above except the points are ignored.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:32, :64, :64]
        >>> fixed = (100 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 160 - (z - 16) ** 2 / 40)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.elastix(transform='RecursiveBSpline', resolution=2, iterations=100)  # doctest: +SKIP
        >>> dvf.GetSize()  # doctest: +SKIP
        (64, 64, 32)
        """
        fixed, moving, fixed_mask, moving_mask = self._prepare(crop=crop, cross_modality=cross_modality)
        use_points = reference_points is not None and moving_points is not None
        if transform == 'SplineKernel' and not use_points:
            raise ValueError("transform='SplineKernel' requires reference_points and moving_points.")

        elastix = sitk.ElastixImageFilter()
        elastix.LogToConsoleOff()
        elastix.SetFixedImage(fixed)
        elastix.SetMovingImage(moving)

        # Elastix masks must be binary unsigned char
        if fixed_mask is not None:
            elastix.SetFixedMask(fixed_mask)

        if moving_mask is not None:
            elastix.SetMovingMask(moving_mask)

        if parameter is None:
            parameter = sitk.GetDefaultParameterMap('bspline')
            parameter['NumberOfResolutions'] = [str(resolution)]

            # The default grid schedule is fixed at 4 levels; rebuild it (Elastix's sqrt(2) factor) for any count
            parameter['GridSpacingSchedule'] = [str(1.41 ** level) for level in reversed(range(resolution))]
            parameter['MaximumNumberOfIterations'] = [str(iterations)]
            parameter['AutomaticParameterEstimation'] = ['true']
            parameter['ResultImagePixelType'] = ['float']

            # The warped image isn't used; only the transform is needed
            parameter['WriteResultImage'] = ['false']

            # Elastix erodes masks at each pyramid level by default, which can delete small organ masks entirely
            parameter['ErodeMask'] = ['false']

            if transform == 'SplineKernel':
                parameter['Transform'] = ['SplineKernelTransform']
                parameter['SplineKernelType'] = ['ThinPlateSpline']
                parameter['SplineRelaxationFactor'] = ['0.0']
            else:
                parameter['Transform'] = ['RecursiveBSplineTransform' if transform == 'RecursiveBSpline'
                                          else 'BSplineTransform']
                parameter['FinalGridSpacingInPhysicalUnits'] = [str(spacing)]
                parameter['BSplineTransformSplineOrder'] = [str(order)]

            # Metric list: intensity first, then optional point and bending energy terms
            metrics = ['AdvancedMeanSquares' if metric == 'Intensity' else 'AdvancedMattesMutualInformation']
            weights = [1.0]
            if use_points and transform != 'SplineKernel':
                metrics.append('CorrespondingPointsEuclideanDistanceMetric')
                weights.append(points_weight)

            if bending_weight > 0:
                metrics.append('TransformBendingEnergyPenalty')
                weights.append(bending_weight)

            parameter['Metric'] = metrics
            if metric != 'Intensity':
                parameter['NumberOfHistogramBins'] = [str(bins)]

            if len(metrics) > 1:
                parameter['Registration'] = ['MultiMetricMultiResolutionRegistration']
                for index, weight in enumerate(weights):
                    parameter[f'Metric{index}Weight'] = [str(weight)]

        elastix.SetParameterMap(parameter)

        # Elastix/Transformix write parameter and log files; keep them out of the working directory
        with tempfile.TemporaryDirectory() as folder:
            if use_points:
                # Elastix point file format: 'point', count, then one 'x y z' line per point (physical mm)
                files = []
                for name, points in (('fixed.txt', reference_points), ('moving.txt', moving_points)):
                    path = os.path.join(folder, name)
                    np.savetxt(path, np.asarray(points, dtype=np.float64).reshape(-1, 3),
                               header=f'point\n{len(points)}', comments='')
                    files.append(path)
                elastix.SetFixedPointSetFileName(files[0])
                elastix.SetMovingPointSetFileName(files[1])

            elastix.SetOutputDirectory(folder)
            elastix.Execute()

            transformix = sitk.TransformixImageFilter()
            transformix.LogToConsoleOff()
            transformix.SetOutputDirectory(folder)
            transformix.SetMovingImage(moving)
            transformix.SetTransformParameterMap(elastix.GetTransformParameterMap())
            transformix.ComputeDeformationFieldOn()
            transformix.Execute()
            dvf = sitk.Cast(transformix.GetDeformationField(), sitk.sitkVectorFloat32)

        # Place the (possibly cropped) field back on the full reference grid
        return sitk.Resample(dvf, self.reference_image, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkVectorFloat32)

    def elastix_groupwise(self, images, resolution=4, spacing=10, iterations=500):
        """Groupwise Elastix registration of several images to their implicit mean (BSplineStackTransform).

        No image is the reference: all images are deformed towards a common mean space. Useful for building
        atlases or longitudinal series without biasing toward one scan. Independent of the stored images/masks.

        Parameters
        ----------
        images : list of SimpleITK.Image
            Two or more 3D images, already rigidly aligned. All are resampled onto the grid of ``images[0]``.
        resolution : int, default 4
            Number of pyramid levels.
        spacing : float, default 10
            Final B-spline grid spacing in mm.
        iterations : int, default 500
            Maximum iterations per level.

        Returns
        -------
        list of SimpleITK.Image
            One ``sitkVectorFloat32`` displacement field per image, on the grid of ``images[0]``. Field ``i`` maps a
            point in the mean space to image ``i`` (pull-back), so warping image ``i`` with it moves it to the mean.

        Examples
        --------
        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> z, y, x = np.mgrid[:32, :64, :64]
        >>> base = (100 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 160 - (z - 16) ** 2 / 40)).astype(np.float32)
        >>> images = [sitk.GetImageFromArray(np.roll(base, shift, axis=2)) for shift in (-2, 0, 2)]
        >>> dvfs = DeformableITK().elastix_groupwise(images, resolution=2, iterations=100)  # doctest: +SKIP
        >>> len(dvfs), dvfs[0].GetSize()  # doctest: +SKIP
        (3, (64, 64, 32))
        """
        if len(images) < 2:
            raise ValueError('Groupwise registration needs at least two images.')

        # Stack the 3D images into one 4D image; the 4th axis is the group index
        reference = sitk.Cast(images[0], sitk.sitkFloat32)
        stack = sitk.JoinSeries([self.resample(sitk.Cast(image, sitk.sitkFloat32), reference) for image in images])

        parameter = sitk.GetDefaultParameterMap('groupwise')
        parameter['NumberOfResolutions'] = [str(resolution)]
        parameter['GridSpacingSchedule'] = [str(1.41 ** level) for level in reversed(range(resolution))]
        parameter['FinalGridSpacingInPhysicalUnits'] = [str(spacing)]
        parameter['MaximumNumberOfIterations'] = [str(iterations)]
        parameter['WriteResultImage'] = ['false']

        # Smooth/downsample only the 3 spatial axes; the group axis (4th) must not be smoothed (and needs >= 4
        # images if it is)
        schedule = []
        for level in reversed(range(resolution)):
            schedule += [str(2 ** level)] * 3 + ['0']
        parameter['FixedImagePyramidSchedule'] = schedule
        parameter['MovingImagePyramidSchedule'] = schedule

        elastix = sitk.ElastixImageFilter()
        elastix.LogToConsoleOff()

        # Groupwise registration uses the same 4D stack as fixed and moving image
        elastix.SetFixedImage(stack)
        elastix.SetMovingImage(stack)
        elastix.SetParameterMap(parameter)

        with tempfile.TemporaryDirectory() as folder:
            elastix.SetOutputDirectory(folder)
            elastix.Execute()

            transformix = sitk.TransformixImageFilter()
            transformix.LogToConsoleOff()
            transformix.SetOutputDirectory(folder)
            transformix.SetMovingImage(stack)
            transformix.SetTransformParameterMap(elastix.GetTransformParameterMap())
            transformix.ComputeDeformationFieldOn()
            transformix.Execute()
            field = transformix.GetDeformationField()

        # Split the 4D field (4 components) into one 3D (dx, dy, dz) field per image on the reference grid
        size = list(field.GetSize())
        dvfs = []
        for index in range(len(images)):
            volume = sitk.Extract(field, size[:3] + [0], [0, 0, 0, index])
            components = [sitk.VectorIndexSelectionCast(volume, axis, sitk.sitkFloat32) for axis in range(3)]
            dvf = sitk.Compose(components)
            dvf.CopyInformation(reference)
            dvfs.append(dvf)

        return dvfs

    def fast_demons(self, std=1.5, iterations=100, intensity_threshold=0.001, step=1.0, shrink_factors=(4, 2, 1),
                    smoothing_sigmas=(2, 1, 0), crop=10, cross_modality=False, normalize=True, histogram_match=True,
                    blend_sigma=2):
        """Fast Symmetric Forces Demons registration, multi-resolution.

        Parameters
        ----------
        std : float, default 1.5
            Gaussian regularization of the displacement field, in **voxels**.
        iterations : int, default 100
            Iterations per resolution level.
        intensity_threshold : float, default 0.001
            Intensity differences below this are treated as matched (normalized [0, 1] units when ``normalize``).
        step : float, default 1.0
            Maximum update step length per iteration, in voxels (ITK default 0.5; larger converges faster but is
            less stable).
        shrink_factors : sequence of int, default (4, 2, 1)
            Downsampling factor per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Image smoothing per level, in mm.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.
        normalize : bool, default True
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`).
        histogram_match : bool, default True
            Histogram-match the moving image to the reference.
        blend_sigma : float, default 2
            Blur (mm) of the mask-union edge used to blend the outside region; 0 uses a hard edge.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.fast_demons()
        >>> round(float(sitk.GetArrayViewFromImage(dvf)[8, 16, 16, 0]))
        2
        """
        return self._demons(sitk.FastSymmetricForcesDemonsRegistrationFilter, {'MaximumUpdateStepLength': step},
                            std, iterations, intensity_threshold, shrink_factors, smoothing_sigmas, crop,
                            cross_modality, normalize, histogram_match, blend_sigma)

    def landmark(self, reference_points, moving_points, control_points=8, order=3):
        """Fit a B-spline displacement field to corresponding point pairs (no image intensities used).

        Useful for turning a surface correspondence (e.g. from a mesh registration) into a volumetric DVF. Only the
        reference image grid is needed; the moving image is not used.

        Parameters
        ----------
        reference_points : array_like, shape (N, 3)
            Points on the reference, physical (x, y, z) in mm.
        moving_points : array_like, shape (N, 3)
            Corresponding points on the moving image, same order and convention.
        control_points : int, default 8
            Number of B-spline control points per axis (must be greater than ``order``). More points follow the
            landmarks more closely but extrapolate less smoothly away from them.
        order : int, default 3
            B-spline order.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> helper = DeformableITK(reference_image=sitk.Image(32, 32, 16, sitk.sitkFloat32))
        >>> reference_points = np.random.default_rng(0).uniform(4, 14, size=(50, 3))
        >>> moving_points = reference_points + [2, 0, 0]
        >>> dvf = helper.landmark(reference_points, moving_points)
        >>> round(float(sitk.GetArrayViewFromImage(dvf)[8, 8, 8, 0]))
        2
        """
        if self.reference_image is None:
            raise ValueError('reference_image must be set before registering.')

        reference_points = np.asarray(reference_points, dtype=np.float64)
        moving_points = np.asarray(moving_points, dtype=np.float64)
        if reference_points.shape != moving_points.shape or reference_points.shape[1] != 3:
            raise ValueError('reference_points and moving_points must both be (N, 3) with matching N.')

        # Fixed landmarks -> moving landmarks gives the pull-back (fixed -> moving) convention used everywhere
        initializer = sitk.LandmarkBasedTransformInitializerFilter()
        initializer.SetFixedLandmarks(reference_points.flatten().tolist())
        initializer.SetMovingLandmarks(moving_points.flatten().tolist())
        initializer.SetReferenceImage(self.reference_image)
        initializer.SetBSplineNumberOfControlPoints(int(control_points))
        transform = initializer.Execute(sitk.BSplineTransform(3, order))

        reference = self.reference_image
        return sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat32, reference.GetSize(),
                                                 reference.GetOrigin(), reference.GetSpacing(),
                                                 reference.GetDirection())

    def level_set_motion(self, std=1.5, iterations=200, intensity_threshold=0.001, alpha=0.01, gradient_std=1.0,
                         max_rms_error=0.0, shrink_factors=(4, 2, 1), smoothing_sigmas=(2, 1, 0), crop=10,
                         cross_modality=False, normalize=True, histogram_match=True, blend_sigma=2):
        """Level Set Motion registration, multi-resolution.

        Moves voxels along the image gradient like a level-set front; more robust than Thirion Demons to large
        intensity differences, but each step is small, so it needs more iterations than the Demons methods.

        Parameters
        ----------
        std : float, default 1.5
            Gaussian regularization of the displacement field, in **voxels**.
        iterations : int, default 200
            Iterations per resolution level.
        intensity_threshold : float, default 0.001
            Intensity differences below this are treated as matched (normalized [0, 1] units when ``normalize``).
        alpha : float, default 0.01
            Stabilizing constant added to the gradient magnitude (prevents large steps in flat regions). Tuned for
            normalized [0, 1] intensities; ITK's default of 0.1 barely moves on normalized images.
        gradient_std : float, default 1.0
            Gaussian smoothing (voxels) applied to the image before computing gradients.
        max_rms_error : float, default 0.0
            Stop a level early when the RMS change of the field per iteration falls below this. 0 disables (ITK's
            default of 0.02 stops after a few iterations on normalized images).
        shrink_factors : sequence of int, default (4, 2, 1)
            Downsampling factor per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Image smoothing per level, in mm.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.
        normalize : bool, default True
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`).
        histogram_match : bool, default True
            Histogram-match the moving image to the reference.
        blend_sigma : float, default 2
            Blur (mm) of the mask-union edge used to blend the outside region; 0 uses a hard edge.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.level_set_motion()
        >>> dvf.GetSize()
        (32, 32, 16)
        """
        options = {'Alpha': alpha, 'GradientSmoothingStandardDeviations': gradient_std,
                   'MaximumRMSError': max_rms_error}

        return self._demons(sitk.LevelSetMotionRegistrationFilter, options, std, iterations, intensity_threshold,
                            shrink_factors, smoothing_sigmas, crop, cross_modality, normalize, histogram_match,
                            blend_sigma)

    def mask_crop(self, images, reference_mask, moving_mask, margin=10):
        """Crop images to the bounding box of the union of two masks, padded by a margin.

        All images must already share the same grid (see :meth:`resample`).

        Parameters
        ----------
        images : list of SimpleITK.Image
            Images (and/or masks) to crop with the same region.
        reference_mask, moving_mask : SimpleITK.Image
            Binary masks on the shared grid.
        margin : float, default 10
            Padding around the bounding box, in mm (converted per axis so it is isotropic for anisotropic voxels).

        Returns
        -------
        list of SimpleITK.Image
            Cropped images, in the same order. Returned uncropped if both masks are empty.

        Examples
        --------
        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> helper = DeformableITK()
        >>> array = np.zeros((16, 32, 32), dtype=np.uint8)
        >>> array[4:12, 8:24, 8:24] = 1
        >>> mask = sitk.GetImageFromArray(array)
        >>> mask.SetSpacing((1, 1, 2))
        >>> cropped = helper.mask_crop([mask], mask, mask, margin=4)
        >>> cropped[0].GetSize()
        (24, 24, 12)
        """
        combined = sitk.Or(sitk.Cast(reference_mask != 0, sitk.sitkUInt8), sitk.Cast(moving_mask != 0, sitk.sitkUInt8))

        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(combined)
        if not label_stats.HasLabel(1):
            return list(images)

        # Bounding box is (x0, y0, z0, sx, sy, sz) in index space
        bbox = label_stats.GetBoundingBox(1)
        start = np.array(bbox[:3])
        end = start + np.array(bbox[3:])

        # Margin in mm -> voxels per axis
        pad = np.ceil(margin / np.array(combined.GetSpacing())).astype(int)
        start = np.maximum(start - pad, 0)
        end = np.minimum(np.array(combined.GetSize()), end + pad)
        size = end - start

        # RegionOfInterest keeps the physical origin, so cropped images stay correctly placed in space
        return [sitk.RegionOfInterest(image, size.tolist(), start.tolist()) for image in images]

    def resample(self, image, reference, mask=False):
        """Resample an image onto the grid of a reference image (identity transform).

        Parameters
        ----------
        image : SimpleITK.Image
            Image to resample.
        reference : SimpleITK.Image
            Image defining the output grid (size, origin, spacing, direction).
        mask : bool, default False
            Use nearest-neighbor interpolation and return a binary ``sitkUInt8`` image.

        Returns
        -------
        SimpleITK.Image
            Image on the reference grid. Returned without resampling if the grids already match.

        Examples
        --------
        >>> import SimpleITK as sitk
        >>> helper = DeformableITK()
        >>> reference = sitk.Image(32, 32, 16, sitk.sitkFloat32)
        >>> moving = sitk.Image(64, 64, 32, sitk.sitkFloat32)
        >>> moving.SetSpacing((0.5, 0.5, 0.5))
        >>> helper.resample(moving, reference).GetSize()
        (32, 32, 16)
        """
        if mask:
            image = sitk.Cast(image != 0, sitk.sitkUInt8)

        # Skip the resample when the grids are already identical
        same_grid = (image.GetSize() == reference.GetSize()
                     and np.allclose(image.GetSpacing(), reference.GetSpacing())
                     and np.allclose(image.GetOrigin(), reference.GetOrigin())
                     and np.allclose(image.GetDirection(), reference.GetDirection()))

        if same_grid:
            return image

        interpolator = sitk.sitkNearestNeighbor if mask else sitk.sitkLinear
        return sitk.Resample(image, reference, sitk.Transform(), interpolator, 0.0, image.GetPixelID())

    def symmetric_demons(self, std=1.5, iterations=100, intensity_threshold=0.001, shrink_factors=(4, 2, 1),
                         smoothing_sigmas=(2, 1, 0), crop=10, cross_modality=False, normalize=True,
                         histogram_match=True, blend_sigma=2):
        """Symmetric Forces Demons registration (forces from both image gradients), multi-resolution.

        Older and slower than :meth:`fast_demons`; mainly useful for comparison.

        Parameters
        ----------
        std : float, default 1.5
            Gaussian regularization of the displacement field, in **voxels**.
        iterations : int, default 100
            Iterations per resolution level.
        intensity_threshold : float, default 0.001
            Intensity differences below this are treated as matched (normalized [0, 1] units when ``normalize``).
        shrink_factors : sequence of int, default (4, 2, 1)
            Downsampling factor per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Image smoothing per level, in mm.
        crop : float, default 10
            Mask-union crop margin in mm (0 disables).
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.
        normalize : bool, default True
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`).
        histogram_match : bool, default True
            Histogram-match the moving image to the reference.
        blend_sigma : float, default 2
            Blur (mm) of the mask-union edge used to blend the outside region; 0 uses a hard edge.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field (fixed -> moving) on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = DeformableITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> dvf = helper.symmetric_demons()
        >>> dvf.GetSize()
        (32, 32, 16)
        """
        return self._demons(sitk.SymmetricForcesDemonsRegistrationFilter, {}, std, iterations, intensity_threshold,
                            shrink_factors, smoothing_sigmas, crop, cross_modality, normalize, histogram_match,
                            blend_sigma)
