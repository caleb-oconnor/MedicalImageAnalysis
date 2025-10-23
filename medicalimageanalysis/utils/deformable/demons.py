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


class Demons(object):
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

    def demons(self, smooth=True, std=1, iterations=50, intensity_threshold=0.001):
        fixed = sitk.Cast(self.reference_image, sitk.sitkFloat32)
        moving = sitk.Cast(self.moving_image, sitk.sitkFloat32)

        if self.reference_mask is not None:
            fixed = fixed * sitk.Cast(self.reference_mask, sitk.sitkFloat32)
        if self.moving_mask is not None:
            moving = moving * sitk.Cast(self.moving_mask, sitk.sitkFloat32)

        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(iterations)
        demons.SetStandardDeviations(std)
        demons.SetIntensityDifferenceThreshold(intensity_threshold)
        if smooth:
            demons.SmoothDisplacementFieldOn()
        else:
            demons.SmoothDisplacementFieldOff()

        return demons.Execute(fixed, moving)

    def fast_demons(self, smooth=True, std=1, iterations=50, intensity_threshold=0.001, step=2.0):
        fixed = sitk.Cast(self.reference_image, sitk.sitkFloat32)
        moving = sitk.Cast(self.moving_image, sitk.sitkFloat32)

        if self.reference_mask is not None:
            fixed = fixed * sitk.Cast(self.reference_mask, sitk.sitkFloat32)
        if self.moving_mask is not None:
            moving = moving * sitk.Cast(self.moving_mask, sitk.sitkFloat32)

        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(iterations)
        demons.SetStandardDeviations(std)
        demons.SetIntensityDifferenceThreshold(intensity_threshold)
        demons.SetMaximumUpdateStepLength(step)
        if smooth:
            demons.SmoothDisplacementFieldOn()
        else:
            demons.SmoothDisplacementFieldOff()

        return demons.Execute(fixed, moving)

    def diffeomorphic(self, smooth=True, std=1, iterations=50, intensity_threshold=0.001, step=2.0):
        fixed = sitk.Cast(self.reference_image, sitk.sitkFloat32)
        moving = sitk.Cast(self.moving_image, sitk.sitkFloat32)

        if self.reference_mask is not None:
            fixed = fixed * sitk.Cast(self.reference_mask, sitk.sitkFloat32)
        if self.moving_mask is not None:
            moving = moving * sitk.Cast(self.moving_mask, sitk.sitkFloat32)

        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(iterations)
        demons.SetStandardDeviations(std)
        demons.SetIntensityDifferenceThreshold(intensity_threshold)
        demons.SetMaximumUpdateStepLength(step)
        if smooth:
            demons.SmoothDisplacementFieldOn()
        else:
            demons.SmoothDisplacementFieldOff()

        return demons.Execute(fixed, moving)
