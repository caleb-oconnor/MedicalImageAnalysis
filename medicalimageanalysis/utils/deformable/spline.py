"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
"""

import numpy as np
import SimpleITK as sitk


class Spline(object):
    def __init__(self, reference_image=None, moving_image=None, reference_mask=None, moving_mask=None):
        self.reference_image = reference_image
        self.reference_mask = reference_mask
        self.moving_image = moving_image
        self.moving_mask = moving_mask

    def create_sitk_image(self, array, origin, spacing, direction, reference=True, mask=False):
        image = sitk.GetImageFromArray(array)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction.flatten().astype(np.float64))

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

    def cross_modality_correction(self):
        if self.reference_image is not None:
            self.reference_image = sitk.GradientMagnitude(self.reference_image)

        if self.moving_image is not None:
            self.moving_image = sitk.GradientMagnitude(self.moving_image)

    def blur_mask(self, reference=True, sigma=2):
        if reference and self.reference_mask is not None:
            mask = sitk.Cast(self.reference_mask, sitk.sitkFloat32)
            blurred_mask = sitk.SmoothingRecursiveGaussian(mask, sigma=sigma)
            min_val = sitk.GetArrayViewFromImage(blurred_mask).min()
            max_val = sitk.GetArrayViewFromImage(blurred_mask).max()
            self.reference_mask = (blurred_mask - min_val) / (max_val - min_val)

        elif not reference and self.moving_mask is not None:
            mask = sitk.Cast(self.moving_mask, sitk.sitkFloat32)
            blurred_mask = sitk.SmoothingRecursiveGaussian(mask, sigma=sigma)
            min_val = sitk.GetArrayViewFromImage(blurred_mask).min()
            max_val = sitk.GetArrayViewFromImage(blurred_mask).max()
            self.moving_mask = (blurred_mask - min_val) / (max_val - min_val)

    def resample(self):
        if self.reference_image is not None and self.moving_image is not None:
            self.moving_image = sitk.Resample(self.moving_image,
                                              self.reference_image,
                                              sitk.Transform(),
                                              sitk.sitkLinear,
                                              0.0,
                                              self.moving_image.GetPixelIDValue())

        if self.reference_mask is not None and self.moving_mask is not None:
            self.moving_mask = sitk.Resample(self.moving_mask,
                                             self.reference_mask,
                                             sitk.Transform(),
                                             sitk.sitkLinear,
                                             0.0,
                                             self.moving_mask.GetPixelIDValue())

    def bspline(self, control_spacing=None, mesh_size=None, gradient=1e-5, iterations=100):
        fixed = sitk.Cast(self.reference_image, sitk.sitkFloat32)
        moving = sitk.Cast(self.moving_image, sitk.sitkFloat32)

        if control_spacing is None:
            control_spacing = [25.0, 25.0, 25.0]

        if mesh_size is None:
            image_physical_size = [size * spacing for size, spacing in zip(fixed.GetSize(), fixed.GetSpacing())]
            mesh_size = [int(sz / sp) for sz, sp in zip(image_physical_size, control_spacing)]
        bspline_transform = sitk.BSplineTransformInitializer(fixed, mesh_size)

        bspline = sitk.ImageRegistrationMethod()
        bspline.SetMetricAsMeanSquares()
        if self.reference_mask:
            bspline.SetMetricFixedMask(sitk.Cast(self.reference_mask, sitk.sitkFloat32))
        if self.moving_mask:
            bspline.SetMetricMovingMask(sitk.Cast(self.moving_mask, sitk.sitkFloat32))
        bspline.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=gradient, numberOfIterations=iterations)
        bspline.SetInitialTransform(bspline_transform, inPlace=True)
        bspline.SetInterpolator(sitk.sitkLinear)

        final_transform = bspline.Execute(fixed, moving)

        return sitk.TransformToDisplacementField(final_transform, sitk.sitkVectorFloat32, fixed.GetSize(),
                                                 fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())

    def elastix(self, parameter=None, metric='Intensity', bins=6, resolution=4, spacing=10, iterations=2000, order=3):
        fixed = sitk.Cast(self.reference_image, sitk.sitkFloat32)
        moving = sitk.Cast(self.moving_image, sitk.sitkFloat32)

        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(fixed)
        elastix.SetMovingImage(moving)
        if self.reference_mask is not None:
            elastix.SetFixedMask(sitk.Cast(self.reference_mask, sitk.sitkFloat32))
        if self.moving_mask is not None:
            elastix.SetMovingMask(sitk.Cast(self.moving_mask, sitk.sitkFloat32))

        if parameter is None:
            parameter = sitk.GetDefaultParameterMap("nonrigid")
            if metric == 'Intensity':
                parameter["Metric"] = ['AdvancedMeanSquares']
            else:
                parameter["Metric"] = ["AdvancedMattesMutualInformation"]
            parameter["NumberOfHistogramBins"] = [str(bins)]
            parameter["NumberOfResolutions"] = [str(resolution)]
            parameter["FinalGridSpacingInPhysicalUnits"] = [str(spacing)]
            parameter["MaximumNumberOfIterations"] = [str(iterations)]
            parameter["BSplineTransformSplineOrder"] = [str(order)]
            parameter["AutomaticParameterEstimation"] = ["true"]
            parameter["ResultImagePixelType"] = ["float"]
            parameter["WriteResultImage"] = ["true"]

        elastix.SetParameterMap(parameter)
        elastix.Execute()

        transform_param_map = elastix.GetTransformParameterMap()
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(moving)
        transformix.SetTransformParameterMap(transform_param_map)
        transform_param_map[0]["ComputeDeformationField"] = ["true"]
        transformix.SetTransformParameterMap(transform_param_map)
        transformix.Execute()

        return transformix.GetDeformationField()
