"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
"""

import numpy as np

from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops


def external(array, threshold=-250, min_volume=100, only_mask=True, less_than=False):
    if less_than:
        binary = array < threshold
    else:
        binary = array > threshold
    label_image = label(binary)
    label_regions = regionprops(label_image)
    region_areas = [region.area for region in label_regions]
    max_idx = np.argmax(region_areas)
    bounds = label_regions[max_idx].bbox
    box_image = label_regions[max_idx].image

    mask = np.zeros(array.shape)
    centroid_external = np.zeros((box_image.shape[0], 2))
    external_components = np.zeros((box_image.shape[0], 1))
    for ii in range(box_image.shape[0]):
        filled_image = binary_fill_holes(box_image[ii, :, :])
        fill_image = label(filled_image)
        fill_regions = regionprops(fill_image)
        external_components[ii] = len([region.area for region in fill_regions if region.area > min_volume])

        centroid_external[ii, :] = np.round(np.mean(np.argwhere(filled_image), axis=0))
        mask[ii + bounds[0], bounds[1]:bounds[4], bounds[2]:bounds[5]] = 1 * filled_image

    if only_mask:
        return mask
    else:
        return mask, centroid_external, external_components, bounds
