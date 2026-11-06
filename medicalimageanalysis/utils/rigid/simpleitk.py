"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
    Rigid image registration wrappers around SimpleITK.

    Methods
    -------
    * ImageRegistrationMethod: :meth:`RigidITK.rigid` (Euler / Versor rigid registration with masks,
      multi-resolution and several metrics)
    * Paired landmarks: :meth:`RigidITK.landmark` (closed-form weighted Kabsch / Horn fit, no images needed)
    * Mask shapes: :meth:`RigidITK.rigid_mask` (registration of signed distance maps built from binary masks, which
      ignores intensities entirely)
    * Utilities: :meth:`RigidITK.resample_moving`, :meth:`RigidITK.transform_to_displacement_field`,
      :meth:`RigidITK.landmark_error`, :meth:`RigidITK.index_to_physical`, :meth:`RigidITK.distance_map`

    Workflow
    --------
    The three registration methods return the same kind of transform, and both :meth:`RigidITK.rigid` and
    :meth:`RigidITK.rigid_mask` accept ``initial_transform``. A typical coarse-to-fine chain is
    ``landmark -> rigid_mask -> rigid``, each step seeding the next.

    Conventions
    -----------
    * NumPy arrays are indexed (z, y, x). SimpleITK origin, spacing, direction and points are (x, y, z).
    * The optimized transform follows the same pull-back convention used by ``sitk.Resample``: for a point ``p``
      in the reference image, ``transform.TransformPoint(p)`` gives the corresponding point in the moving image.
    * The stored images and masks are never modified. Preprocessing is applied to working copies inside each call.
    * Masks restrict only the metric evaluation in :meth:`RigidITK.rigid`. In :meth:`RigidITK.rigid_mask` the masks are
      the registered shapes. They are stored as binary ``sitkUInt8`` images in their native image spaces.
    * Landmarks are (N, D) arrays of physical points in (x, y, z) order, paired row by row. Use
      :meth:`RigidITK.index_to_physical` to convert (z, y, x) voxel indices first.

Examples
--------
>>> import numpy as np
>>> from medicalimageanalysis.utils.rigid.simpleitk import RigidITK
>>> z, y, x = np.mgrid[:16, :32, :32]
>>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
>>> helper = RigidITK()
>>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
>>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
>>> transform = helper.rigid(sampling=1.0, iterations=50)
>>> round(float(transform.GetTranslation()[0]))
2
"""

import numpy as np
import SimpleITK as sitk


class RigidITK:
    """Rigid registration between a reference (fixed) and moving image, with optional masks.

    This is intentionally smaller than the deformable helper: one main registration method covers the common rigid
    use cases, while a couple of utility methods convert the result to a resampled image or displacement field.

    Parameters
    ----------
    reference_image : SimpleITK.Image, optional
        Fixed image. Its grid defines the registration domain.
    moving_image : SimpleITK.Image, optional
        Image to be registered onto the reference.
    reference_mask : SimpleITK.Image, optional
        Region of interest on the reference image. Any nonzero voxel is treated as inside.
    moving_mask : SimpleITK.Image, optional
        Region of interest on the moving image. Any nonzero voxel is treated as inside.

    Examples
    --------
    >>> import numpy as np
    >>> fixed = np.zeros((16, 32, 32), dtype=np.float32)
    >>> fixed[4:12, 8:24, 8:24] = 100
    >>> helper = RigidITK()
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
        >>> helper = RigidITK(reference_image=image, moving_image=image)
        >>> helper.reference_mask is None
        True
        """
        self.reference_image = reference_image
        self.moving_image = moving_image

        # Masks are normalized once up front, which is what the metric mask APIs expect.
        self.reference_mask = None if reference_mask is None else sitk.Cast(reference_mask != 0, sitk.sitkUInt8)
        self.moving_mask = None if moving_mask is None else sitk.Cast(moving_mask != 0, sitk.sitkUInt8)

    @staticmethod
    def _base_transform(dimension, transform):
        """Return an identity rigid transform of the requested family for a 2D/3D problem."""
        if dimension not in (2, 3):
            raise ValueError('RigidITK registration is implemented only for 2D or 3D images.')

        if transform == 'Euler':
            return sitk.Euler2DTransform() if dimension == 2 else sitk.Euler3DTransform()

        if transform == 'Versor':
            if dimension != 3:
                raise ValueError("transform='Versor' is only available for 3D images.")

            return sitk.VersorRigid3DTransform()

        raise ValueError(f'Unknown transform: {transform}')

    @staticmethod
    def _copy_rigid(source, base):
        """Copy the center, rotation and translation of a rigid transform into ``base``.

        This lets any rigid result (for example from :meth:`landmark`, or one read from disk) seed a registration
        that uses a different rigid family, e.g. a Euler landmark result refined with a Versor optimization.

        Parameters
        ----------
        source : SimpleITK.Transform
            Rigid transform to copy. A generic ``sitk.Transform`` wrapper is downcast first.
        base : SimpleITK.Transform
            Euler or Versor transform that receives the parameters.

        Returns
        -------
        SimpleITK.Transform
            ``base``, updated in place.

        Examples
        --------
        >>> import SimpleITK as sitk
        >>> source = sitk.Euler3DTransform((1, 2, 3), 0.0, 0.0, 0.1, (4, 0, 0))
        >>> copied = RigidITK._copy_rigid(sitk.Transform(source), sitk.VersorRigid3DTransform())
        >>> copied.GetName()
        'VersorRigid3DTransform'
        >>> np.allclose(copied.TransformPoint((5, 5, 5)), source.TransformPoint((5, 5, 5)))
        True
        """
        if type(source) is sitk.Transform:
            source = source.Downcast()

        try:
            center, matrix, translation = source.GetCenter(), source.GetMatrix(), source.GetTranslation()
        except AttributeError:
            raise TypeError(f'initial_transform must be a rigid Euler/Versor transform, got {source.GetName()}.')

        if len(center) != base.GetDimension():
            raise ValueError('initial_transform dimension does not match the registration dimension.')

        base.SetCenter(center)
        base.SetMatrix(matrix)
        base.SetTranslation(translation)
        return base

    @staticmethod
    def distance_map(mask, band=None):
        """Signed Euclidean distance map of a binary mask, in physical units.

        Negative inside, positive outside, zero on the boundary. Clamping to ``band`` keeps far-away voxels from
        dominating the metric while still giving a smooth gradient everywhere within ``band`` of the surface.

        Parameters
        ----------
        mask : SimpleITK.Image
            Binary mask (nonzero = inside).
        band : float, optional
            Clamp distances to [-band, band] (mm). None leaves them unclamped.

        Returns
        -------
        SimpleITK.Image
            Float32 signed distance map on the mask grid.

        Examples
        --------
        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> array = np.zeros((16, 32, 32), dtype=np.uint8)
        >>> array[4:12, 8:24, 8:24] = 1
        >>> distance = RigidITK.distance_map(sitk.GetImageFromArray(array), band=5)
        >>> values = sitk.GetArrayViewFromImage(distance)
        >>> float(values.min()), float(values.max()), bool(values[8, 16, 16] < 0)
        (-3.0, 5.0, True)
        """
        mask = sitk.Cast(mask != 0, sitk.sitkUInt8)
        distance = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False, squaredDistance=False,
                                                useImageSpacing=True)
        distance = sitk.Cast(distance, sitk.sitkFloat32)
        if band is not None:
            distance = sitk.Clamp(distance, sitk.sitkFloat32, -float(band), float(band))

        return distance

    @staticmethod
    def _execute(registration, fixed, moving, fixed_mask, moving_mask, initial_transform, return_info):
        """Run a configured registration and return the transform (and diagnostics if requested)."""
        if fixed_mask is not None:
            registration.SetMetricFixedMask(fixed_mask)

        if moving_mask is not None:
            registration.SetMetricMovingMask(moving_mask)

        # Optimize the rigid transform in place so the returned object is the actual Euler/Versor transform,
        # not a CompositeTransform wrapper around it.
        registration.SetInitialTransform(initial_transform, inPlace=True)
        registration.Execute(fixed, moving)

        if not return_info:
            return initial_transform

        info = {'metric_value': float(registration.GetMetricValue()),
                'stop_condition': registration.GetOptimizerStopConditionDescription(),
                'iterations': int(registration.GetOptimizerIteration())}

        return initial_transform, info

    @staticmethod
    def landmark_error(transform, reference_points, moving_points):
        """Distance between transformed reference landmarks and their moving partners.

        On landmarks used in :meth:`landmark` this is the fiducial registration error; on landmarks left out of the
        fit (or for a transform from :meth:`rigid` / :meth:`rigid_mask`) it is the target registration error.

        Parameters
        ----------
        transform : SimpleITK.Transform
            Transform mapping reference points to the moving image.
        reference_points, moving_points : (N, D) array_like
            Paired physical points in (x, y, z) order.

        Returns
        -------
        (N,) ndarray
            Euclidean error per landmark, in physical units.

        Examples
        --------
        >>> import SimpleITK as sitk
        >>> transform = sitk.Euler3DTransform((0, 0, 0), 0, 0, 0, (1, 0, 0))
        >>> RigidITK.landmark_error(transform, [[0, 0, 0], [5, 5, 5]], [[1, 0, 0], [6, 5, 8]]).tolist()
        [0.0, 3.0]
        """
        reference_points = np.atleast_2d(np.asarray(reference_points, dtype=np.float64))
        moving_points = np.atleast_2d(np.asarray(moving_points, dtype=np.float64))
        if reference_points.shape != moving_points.shape:
            raise ValueError('reference_points and moving_points must have the same shape.')

        mapped = np.array([transform.TransformPoint(p.tolist()) for p in reference_points])
        return np.linalg.norm(mapped - moving_points, axis=1)

    @staticmethod
    def _registration_method(metric, bins, radius, sampling, iterations, learning_rate, min_step, relaxation,
                             shrink_factors, smoothing_sigmas):
        """Build the ``ImageRegistrationMethod`` shared by :meth:`rigid` and :meth:`rigid_mask`.

        Masks, the initial transform and the images are set by the caller.
        """
        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError('shrink_factors and smoothing_sigmas must have the same length.')

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

        # Random sampling is usually the best speed/robustness tradeoff for rigid alignment.
        if sampling < 1.0:
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(float(sampling), 42)

        registration.SetInterpolator(sitk.sitkLinear)

        registration.SetOptimizerAsRegularStepGradientDescent(learningRate=learning_rate, minStep=min_step,
                                                              numberOfIterations=iterations,
                                                              relaxationFactor=relaxation,
                                                              gradientMagnitudeTolerance=1e-6)
        registration.SetOptimizerScalesFromPhysicalShift()

        registration.SetShrinkFactorsPerLevel(list(shrink_factors))
        registration.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        return registration

    def _create_initial_transform(self, fixed, moving, transform, initializer, initial_transform=None):
        """Create a centered rigid transform for 2D/3D registration.

        ``initial_transform``, when given, takes precedence over ``initializer``.
        """
        base = self._base_transform(fixed.GetDimension(), transform)
        if initial_transform is not None:
            return self._copy_rigid(initial_transform, base)

        if initializer == 'Geometry':
            mode = sitk.CenteredTransformInitializerFilter.GEOMETRY

        elif initializer == 'Moments':
            mode = sitk.CenteredTransformInitializerFilter.MOMENTS

        else:
            raise ValueError(f'Unknown initializer: {initializer}')

        return sitk.CenteredTransformInitializer(fixed, moving, base, mode)

    def _default_value(self, image, mask=False, default_value=None):
        """Return a sensible resampling fill value."""
        if default_value is not None:
            return float(default_value)

        if mask:
            return 0.0

        return float(sitk.GetArrayViewFromImage(image).min())

    def _normalize_image(self, image, mask=None):
        """Robustly clip an image to its 0.5-99.5 percentile range and rescale to [0, 1]."""
        values = sitk.GetArrayViewFromImage(image)
        if mask is not None:
            inside = sitk.GetArrayViewFromImage(mask) > 0
            if inside.any():
                values = values[inside]

        low, high = np.percentile(values, [0.5, 99.5])
        if high <= low:
            high = low + 1.0

        image = sitk.Clamp(image, sitk.sitkFloat32, float(low), float(high))
        return sitk.Cast((image - float(low)) / float(high - low), sitk.sitkFloat32)

    def _prepare(self, cross_modality=False, normalize=False, histogram_match=False):
        """Build working copies of the images and masks for a registration call.

        Unlike the deformable helper, the moving image is kept in its native grid. SimpleITK's rigid registration
        handles different image geometries directly, so avoiding a pre-resample preserves the moving FOV, keeps the
        initializer meaningful, and avoids an extra interpolation.

        Parameters
        ----------
        cross_modality : bool, default False
            Apply :meth:`cross_modality_correction` to both images.
        normalize : bool, default False
            Clip each image to its 0.5-99.5 percentile range in its own image space and rescale to [0, 1].
        histogram_match : bool, default False
            Histogram-match the moving image to the reference.

        Returns
        -------
        tuple
            ``(fixed, moving, fixed_mask, moving_mask)``; masks may be None.

        Examples
        --------
        >>> import numpy as np
        >>> helper = RigidITK()
        >>> _ = helper.create_sitk_image(np.random.rand(16, 32, 32).astype(np.float32), (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.random.rand(16, 32, 32).astype(np.float32), (0, 0, 0), (1, 1, 2), np.eye(3),
        ...                              reference=False)
        >>> fixed, moving, fixed_mask, moving_mask = helper._prepare(normalize=True)
        >>> fixed.GetDimension() == moving.GetDimension(), fixed_mask is None
        (True, True)
        """
        if self.reference_image is None or self.moving_image is None:
            raise ValueError('Both reference_image and moving_image must be set before registering.')
        if self.reference_image.GetDimension() != self.moving_image.GetDimension():
            raise ValueError('reference_image and moving_image must have the same dimension.')

        fixed = sitk.Cast(self.reference_image, sitk.sitkFloat32)
        moving = sitk.Cast(self.moving_image, sitk.sitkFloat32)
        fixed_mask = None if self.reference_mask is None else sitk.Cast(self.reference_mask != 0, sitk.sitkUInt8)
        moving_mask = None if self.moving_mask is None else sitk.Cast(self.moving_mask != 0, sitk.sitkUInt8)

        if cross_modality:
            fixed = self.cross_modality_correction(fixed)
            moving = self.cross_modality_correction(moving)

        if normalize:
            fixed = self._normalize_image(fixed, fixed_mask)
            moving = self._normalize_image(moving, moving_mask)

        if histogram_match:
            moving = sitk.HistogramMatching(moving, fixed, numberOfHistogramLevels=256, numberOfMatchPoints=15,
                                            thresholdAtMeanIntensity=True)

        return fixed, moving, fixed_mask, moving_mask

    def create_sitk_image(self, array, origin, spacing, direction, reference=True, mask=False):
        """Convert a NumPy array to a SimpleITK image and store it as the reference/moving image or mask.

        Parameters
        ----------
        array : numpy.ndarray
            Volume indexed (z, y, x).
        origin : sequence of float
            Physical position of voxel (0, 0, 0), in (x, y, z) order.
        spacing : sequence of float
            Voxel size in (x, y, z) order.
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
        >>> helper = RigidITK()
        >>> image = helper.create_sitk_image(np.zeros((16, 32, 32), dtype=np.float32), (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> image.GetSize(), image.GetSpacing()
        ((32, 32, 16), (1.0, 1.0, 2.0))
        >>> mask = helper.create_sitk_image(np.ones((16, 32, 32)), (0, 0, 0), (1, 1, 2), np.eye(3), mask=True)
        >>> mask.GetPixelIDTypeAsString()
        '8-bit unsigned integer'
        """
        if mask:
            # SimpleITK does not accept NumPy bool arrays directly.
            array = (np.asarray(array) != 0).astype(np.uint8)

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

        For multimodal rigid registration, prefer ``metric='MI'`` or ``metric='JointMI'`` in :meth:`rigid`; this is
        mainly useful when rigid alignment still benefits from edge-based matching. A recursive Gaussian version is
        used so the gradients are less noise-sensitive than a plain voxelwise gradient magnitude.

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
        >>> helper = RigidITK()
        >>> gradient = helper.cross_modality_correction(sitk.Image(32, 32, 16, sitk.sitkFloat32))
        >>> gradient.GetPixelIDTypeAsString()
        '32-bit float'
        """
        sigma = float(np.mean(image.GetSpacing()))
        return sitk.GradientMagnitudeRecursiveGaussian(sitk.Cast(image, sitk.sitkFloat32), sigma=sigma)

    def index_to_physical(self, indices, reference=True):
        """Convert (z, y, x) voxel indices to physical (x, y, z) points using the reference or moving image.

        Continuous (fractional) indices are allowed, so sub-voxel landmark positions are kept.

        Parameters
        ----------
        indices : (N, D) array_like
            Voxel indices in NumPy (z, y, x) order.
        reference : bool, default True
            Use the reference image geometry (True) or the moving image geometry (False).

        Returns
        -------
        (N, D) ndarray
            Physical points in (x, y, z) order.

        Examples
        --------
        >>> import numpy as np
        >>> helper = RigidITK()
        >>> _ = helper.create_sitk_image(np.zeros((16, 32, 32), dtype=np.float32), (10, 0, 0), (1, 1, 2), np.eye(3))
        >>> helper.index_to_physical([[3, 2, 1]]).tolist()
        [[11.0, 2.0, 6.0]]
        """
        image = self.reference_image if reference else self.moving_image
        if image is None:
            raise ValueError(f"{'reference' if reference else 'moving'}_image must be set to convert indices.")

        dimension = image.GetDimension()
        indices = np.atleast_2d(np.asarray(indices, dtype=np.float64))
        if indices.shape[1] != dimension:
            raise ValueError(f'indices must have {dimension} columns, got {indices.shape[1]}.')

        direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(dimension, dimension)
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        origin = np.asarray(image.GetOrigin(), dtype=np.float64)

        # Reverse to (x, y, z) index order, then p = origin + direction @ (spacing * index).
        return origin + (indices[:, ::-1] * spacing) @ direction.T

    def landmark(self, reference_points, moving_points, transform='Euler', weights=None, return_info=False):
        """Closed-form rigid registration from paired landmarks (weighted Kabsch / Horn).

        No images are needed. Each row of ``reference_points`` is paired with the same row of ``moving_points``,
        and the least-squares rotation and translation are solved with an SVD, with the reflection case removed so
        the result is always a proper rotation. The result can be used directly or passed as ``initial_transform``
        to :meth:`rigid` or :meth:`rigid_mask`.

        Parameters
        ----------
        reference_points : (N, D) array_like
            Physical landmark positions in the reference image, (x, y, z) order.
        moving_points : (N, D) array_like
            Matching physical landmark positions in the moving image, same order as ``reference_points``.
        transform : {'Euler', 'Versor'}, default 'Euler'
            Rigid transform family of the returned transform.
        weights : (N,) array_like, optional
            Non-negative per-landmark weights, e.g. lower for landmarks that are hard to localize. None weights all
            landmarks equally.
        return_info : bool, default False
            Return a diagnostics dictionary along with the transform.

        Returns
        -------
        SimpleITK.Transform or tuple
            Rigid transform mapping reference points to moving points (the same convention as :meth:`rigid`),
            centered on the weighted reference centroid. If ``return_info=True``, returns ``(transform, info)``
            where ``info`` holds the fiducial registration error: ``'fre_rms'``, ``'fre_max'`` and the per-point
            ``'residuals'``.

        Raises
        ------
        ValueError
            If the point arrays do not match, there are too few points, or the reference points are collinear
            (the rotation about that line is undetermined).

        Notes
        -----
        The fiducial registration error (FRE) measures how well the landmarks used in the fit agree with each other.
        It is not an accuracy estimate for the rest of the anatomy; use :meth:`landmark_error` on landmarks that were
        left out of the fit (target registration error) for that.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> reference = rng.uniform(0, 100, (6, 3))
        >>> angle = np.radians(10)
        >>> rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        >>> moving = reference @ rotation.T + (5, -3, 2)
        >>> helper = RigidITK()
        >>> transform, info = helper.landmark(reference, moving, return_info=True)
        >>> np.allclose(transform.TransformPoint(reference[0].tolist()), moving[0])
        True
        >>> info['fre_rms'] < 1e-6
        True
        """
        reference_points = np.asarray(reference_points, dtype=np.float64)
        moving_points = np.asarray(moving_points, dtype=np.float64)
        if reference_points.ndim != 2 or reference_points.shape != moving_points.shape:
            raise ValueError('reference_points and moving_points must be (N, D) arrays of the same shape.')

        n, dimension = reference_points.shape
        if dimension not in (2, 3):
            raise ValueError('Landmarks must be 2D or 3D points.')

        if n < dimension:
            raise ValueError(f'At least {dimension} landmark pairs are needed for {dimension}D, got {n}.')

        if weights is None:
            weights = np.ones(n)

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('weights must be a non-negative (N,) array with a positive sum.')

        weights = weights / weights.sum()

        reference_center = weights @ reference_points
        moving_center = weights @ moving_points
        reference_centered = reference_points - reference_center
        moving_centered = moving_points - moving_center

        # The fit needs D - 1 independent directions in the reference points (non-collinear in 3D).
        spread = np.linalg.svd(reference_centered * np.sqrt(weights)[:, None], compute_uv=False)
        if spread[dimension - 2] <= 1e-9 * max(spread[0], 1.0):
            raise ValueError('reference_points are degenerate (coincident or collinear); rotation is undetermined.')

        covariance = (reference_centered * weights[:, None]).T @ moving_centered
        u, _, vt = np.linalg.svd(covariance)
        correction = np.ones(dimension)
        correction[-1] = np.sign(np.linalg.det(vt.T @ u.T))  # -1 removes a reflection
        rotation = vt.T @ np.diag(correction) @ u.T

        # sitk rigid transforms are T(p) = R (p - c) + c + t, so centering on the reference centroid gives
        # t = moving_center - reference_center.
        result = self._base_transform(dimension, transform)
        result.SetCenter(reference_center.tolist())
        result.SetMatrix(rotation.flatten().tolist())
        result.SetTranslation((moving_center - reference_center).tolist())

        if not return_info:
            return result

        residuals = self.landmark_error(result, reference_points, moving_points)
        info = {'fre_rms': float(np.sqrt(weights @ residuals ** 2)),
                'fre_max': float(residuals.max()),
                'residuals': residuals}

        return result, info

    def resample(self, image, reference, mask=False, transform=None, default_value=None):
        """Resample an image onto the grid of a reference image.

        Parameters
        ----------
        image : SimpleITK.Image
            Image to resample.
        reference : SimpleITK.Image
            Image defining the output grid.
        mask : bool, default False
            Use nearest-neighbor interpolation and return a binary ``sitkUInt8`` image.
        transform : SimpleITK.Transform, optional
            Transform from the reference grid to ``image``. ``None`` uses the identity transform.
        default_value : float, optional
            Fill value outside the moving image domain. ``None`` uses 0 for masks and the image minimum otherwise.

        Returns
        -------
        SimpleITK.Image
            Resampled image on the reference grid.

        Examples
        --------
        >>> import SimpleITK as sitk
        >>> helper = RigidITK()
        >>> reference = sitk.Image(32, 32, 16, sitk.sitkFloat32)
        >>> moving = sitk.Image(64, 64, 32, sitk.sitkFloat32)
        >>> moving.SetSpacing((0.5, 0.5, 0.5))
        >>> helper.resample(moving, reference).GetSize()
        (32, 32, 16)
        """
        if mask:
            image = sitk.Cast(image != 0, sitk.sitkUInt8)

        if transform is None:
            same_grid = (image.GetSize() == reference.GetSize()
                         and np.allclose(image.GetSpacing(), reference.GetSpacing())
                         and np.allclose(image.GetOrigin(), reference.GetOrigin())
                         and np.allclose(image.GetDirection(), reference.GetDirection()))

            if same_grid:
                return image

            transform = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)

        interpolator = sitk.sitkNearestNeighbor if mask else sitk.sitkLinear
        return sitk.Resample(image, reference, transform, interpolator,
                             self._default_value(image, mask=mask, default_value=default_value), image.GetPixelID())

    def resample_moving(self, transform, interpolator='linear', default_value=None):
        """Resample the moving image onto the reference grid with a rigid transform.

        Parameters
        ----------
        transform : SimpleITK.Transform
            Transform mapping reference points to the moving image.
        interpolator : {'linear', 'nearest'}, default 'linear'
            Interpolation used for the moving image.
        default_value : float, optional
            Fill value outside the moving image domain. ``None`` uses the moving-image minimum.

        Returns
        -------
        SimpleITK.Image
            Moving image resampled onto the reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> helper = RigidITK()
        >>> fixed = np.zeros((16, 32, 32), dtype=np.float32)
        >>> fixed[4:12, 8:24, 8:24] = 1
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> transform = helper.rigid(sampling=1.0, iterations=50)
        >>> aligned = helper.resample_moving(transform)
        >>> aligned.GetSize()
        (32, 32, 16)
        """
        if self.reference_image is None or self.moving_image is None:
            raise ValueError('Both reference_image and moving_image must be set before resampling.')

        if interpolator == 'linear':
            interpolator_value = sitk.sitkLinear

        elif interpolator == 'nearest':
            interpolator_value = sitk.sitkNearestNeighbor

        else:
            raise ValueError(f'Unknown interpolator: {interpolator}')

        return sitk.Resample(self.moving_image, self.reference_image, transform, interpolator_value,
                             self._default_value(self.moving_image, default_value=default_value),
                             self.moving_image.GetPixelID())

    def rigid(self, transform='Euler', initializer='Geometry', metric='Intensity', bins=32, radius=2, sampling=0.1,
              iterations=200, learning_rate=2.0, min_step=1e-4, relaxation=0.5, shrink_factors=(4, 2, 1),
              smoothing_sigmas=(2, 1, 0), cross_modality=False, normalize=False, histogram_match=False,
              initial_transform=None, return_info=False):
        """Rigid registration with ``sitk.ImageRegistrationMethod``.

        This method stays intentionally opinionated: the transform is rigid, the optimizer is regular-step gradient
        descent, and the main choices exposed are the metric, initializer and multi-resolution settings.

        Parameters
        ----------
        transform : {'Euler', 'Versor'}, default 'Euler'
            Rigid transform family. 'Versor' is available only for 3D images.
        initializer : {'Geometry', 'Moments'}, default 'Geometry'
            How the initial transform center and translation are estimated. 'Moments' is intensity-weighted and can
            be unreliable on raw CT because large air regions dominate the moment calculation. Ignored when
            ``initial_transform`` is given.
        metric : {'Intensity', 'Correlation', 'ANTS', 'MI', 'JointMI'}, default 'Intensity'
            Similarity metric. 'Intensity' = mean squares, 'Correlation' = global correlation, 'ANTS' = local
            neighborhood correlation, 'MI'/'JointMI' = mutual information for multimodal cases.
        bins : int, default 32
            Histogram bins for 'MI' and 'JointMI'.
        radius : int, default 2
            Neighborhood radius in voxels for 'ANTS'.
        sampling : float, default 0.1
            Fraction of voxels randomly sampled for the metric. ``1.0`` uses every voxel. With very small masks, a
            larger value may be needed because only samples inside the mask contribute to the metric.
        iterations : int, default 200
            Maximum regular-step gradient descent iterations.
        learning_rate : float, default 2.0
            Initial optimizer step size, in physical units.
        min_step : float, default 1e-4
            Smallest optimizer step before convergence is declared.
        relaxation : float, default 0.5
            Factor used to reduce the step size when the optimizer changes direction.
        shrink_factors : sequence of int, default (4, 2, 1)
            Image downsampling per level.
        smoothing_sigmas : sequence of float, default (2, 1, 0)
            Gaussian image smoothing per level, in mm.
        cross_modality : bool, default False
            Register gradient magnitude images instead of raw intensities.
        normalize : bool, default False
            Percentile-normalize both images to [0, 1] (see :meth:`_prepare`).
        histogram_match : bool, default False
            Histogram-match the moving image to the reference before optimization.
        initial_transform : SimpleITK.Transform, optional
            Rigid starting transform, e.g. the result of :meth:`landmark` or :meth:`rigid_mask`. It is copied, so
            the object passed in is not modified.
        return_info : bool, default False
            Return a small diagnostics dictionary along with the transform.

        Returns
        -------
        SimpleITK.Transform or tuple
            Rigid transform mapping points from the reference image to the moving image. If ``return_info=True``,
            returns ``(transform, info)`` where ``info`` contains the final metric value, stop condition and number
            of iterations.

        Examples
        --------
        >>> import numpy as np
        >>> z, y, x = np.mgrid[:16, :32, :32]
        >>> fixed = (100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40 - (z - 8) ** 2 / 10)).astype(np.float32)
        >>> helper = RigidITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> transform = helper.rigid(sampling=1.0, iterations=50)
        >>> round(float(transform.GetTranslation()[0]))
        2
        """
        fixed, moving, fixed_mask, moving_mask = self._prepare(cross_modality=cross_modality, normalize=normalize,
                                                               histogram_match=histogram_match)

        registration = self._registration_method(metric, bins, radius, sampling, iterations, learning_rate, min_step,
                                                 relaxation, shrink_factors, smoothing_sigmas)
        initial = self._create_initial_transform(fixed, moving, transform, initializer, initial_transform)

        return self._execute(registration, fixed, moving, fixed_mask, moving_mask, initial, return_info)

    def rigid_mask(self, transform='Euler', initializer='Moments', band=20.0, sampling=0.25, iterations=200,
                   learning_rate=2.0, min_step=1e-4, relaxation=0.5, shrink_factors=(4, 2, 1),
                   smoothing_sigmas=(2, 1, 0), reference_mask=None, moving_mask=None, initial_transform=None,
                   return_info=False):
        """Rigid registration of two binary masks through their signed distance maps.

        The images are not used, only the shapes, so this works across modalities (CT, CBCT, MR) and is a robust way
        to align one structure (a bone, the liver) before an intensity refinement with :meth:`rigid`. Distance maps
        give the optimizer a smooth, nearly quadratic metric that converges from much further away than raw binary
        masks. Each mask is converted in its own native grid, so no pre-resampling is needed.

        Parameters
        ----------
        transform : {'Euler', 'Versor'}, default 'Euler'
            Rigid transform family. 'Versor' is available only for 3D.
        initializer : {'Moments', 'Geometry'}, default 'Moments'
            Applied to the binary masks. 'Moments' aligns the mask centroids, which is reliable here because a
            binary mask has no background intensity to bias it. Ignored when ``initial_transform`` is given.
        band : float, default 20.0
            Distances are clamped to [-band, band] mm and the metric is evaluated only where the reference distance
            is inside the band. This roughly sets the capture range beyond the initializer, so raise it for large
            initial offsets.
        sampling : float, default 0.25
            Fraction of voxels randomly sampled (only samples inside the reference band contribute).
        iterations, learning_rate, min_step, relaxation, shrink_factors, smoothing_sigmas
            Optimizer and multi-resolution settings, as in :meth:`rigid`.
        reference_mask, moving_mask : SimpleITK.Image, optional
            Masks to register. None uses the stored ``self.reference_mask`` / ``self.moving_mask``. Passing them here
            lets the shapes differ from the ROI masks used by :meth:`rigid`.
        initial_transform : SimpleITK.Transform, optional
            Rigid starting transform, e.g. from :meth:`landmark`. It is copied, not modified.
        return_info : bool, default False
            Return a diagnostics dictionary along with the transform. ``metric_value`` is the mean squared distance
            difference in mm², so its square root is a rough RMS surface mismatch.

        Returns
        -------
        SimpleITK.Transform or tuple
            Rigid transform mapping reference points to the moving image (same convention as :meth:`rigid`).

        Raises
        ------
        ValueError
            If either mask is missing or empty, or the masks have different dimensions.

        Examples
        --------
        >>> import numpy as np
        >>> fixed = np.zeros((24, 48, 48), dtype=np.uint8)
        >>> fixed[6:18, 10:30, 14:34] = 1
        >>> fixed[6:18, 10:20, 34:40] = 1
        >>> helper = RigidITK()
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3), mask=True)
        >>> _ = helper.create_sitk_image(np.roll(fixed, (3, 2), axis=(2, 1)), (0, 0, 0), (1, 1, 2), np.eye(3),
        ...                              reference=False, mask=True)
        >>> transform = helper.rigid_mask(sampling=1.0)
        >>> (np.round(transform.TransformPoint((0, 0, 0)), 1) + 0.0).tolist()
        [3.0, 2.0, 0.0]
        """
        reference_mask = self.reference_mask if reference_mask is None else reference_mask
        moving_mask = self.moving_mask if moving_mask is None else moving_mask
        if reference_mask is None or moving_mask is None:
            raise ValueError('Both reference and moving masks must be set for rigid_mask.')

        if reference_mask.GetDimension() != moving_mask.GetDimension():
            raise ValueError('reference and moving masks must have the same dimension.')

        reference_mask = sitk.Cast(reference_mask != 0, sitk.sitkUInt8)
        moving_mask = sitk.Cast(moving_mask != 0, sitk.sitkUInt8)
        for name, mask in (('reference', reference_mask), ('moving', moving_mask)):
            if not sitk.GetArrayViewFromImage(mask).any():
                raise ValueError(f'The {name} mask is empty.')

        fixed = self.distance_map(reference_mask, band)
        moving = self.distance_map(moving_mask, band)

        # Evaluate only near the reference surface; flat clamped regions add no gradient and would dilute samples.
        fixed_band = sitk.Cast(sitk.Abs(fixed) < float(band), sitk.sitkUInt8)

        registration = self._registration_method('Intensity', 32, 2, sampling, iterations, learning_rate, min_step,
                                                 relaxation, shrink_factors, smoothing_sigmas)
        # The initializer sees the binary masks, so 'Moments' is the mask centroid offset.
        initial = self._create_initial_transform(sitk.Cast(reference_mask, sitk.sitkFloat32),
                                                 sitk.Cast(moving_mask, sitk.sitkFloat32), transform, initializer,
                                                 initial_transform)

        return self._execute(registration, fixed, moving, fixed_band, None, initial, return_info)

    def transform_to_displacement_field(self, transform):
        """Convert a rigid transform to a displacement field on the reference image grid.

        Parameters
        ----------
        transform : SimpleITK.Transform
            Transform mapping reference points to the moving image.

        Returns
        -------
        SimpleITK.Image
            ``sitkVectorFloat32`` displacement field on the full reference grid.

        Examples
        --------
        >>> import numpy as np
        >>> import SimpleITK as sitk
        >>> helper = RigidITK()
        >>> fixed = np.zeros((16, 32, 32), dtype=np.float32)
        >>> fixed[4:12, 8:24, 8:24] = 1
        >>> _ = helper.create_sitk_image(fixed, (0, 0, 0), (1, 1, 2), np.eye(3))
        >>> _ = helper.create_sitk_image(np.roll(fixed, 2, axis=2), (0, 0, 0), (1, 1, 2), np.eye(3), reference=False)
        >>> field = helper.transform_to_displacement_field(helper.rigid(sampling=1.0, iterations=50))
        >>> field.GetPixelIDTypeAsString(), field.GetSize()
        ('vector of 32-bit float', (32, 32, 16))
        """
        if self.reference_image is None:
            raise ValueError('reference_image must be set before creating a displacement field.')

        reference = self.reference_image
        return sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat32, reference.GetSize(),
                                                 reference.GetOrigin(), reference.GetSpacing(),
                                                 reference.GetDirection())
