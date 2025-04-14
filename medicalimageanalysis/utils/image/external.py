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


def compute_external(array, threshold=-250, only_mask=True):
    binary = array > threshold
    label_image = label(binary)
    label_regions = regionprops(label_image)
    region_areas = [region.area for region in label_regions]
    max_idx = np.argmax(region_areas)
    bounds = label_regions[max_idx].bbox

    box_image = label_regions[max_idx].image

    mask = np.zeros(array.shape)
    reduce_mask = np.zeros(box_image.shape)
    centroid_external = np.zeros((box_image.shape[2], 2))
    external_components = np.zeros((box_image.shape[2], 1))
    for ii in range(box_image.shape[2]):
        filled_image = binary_fill_holes(box_image[:, :, ii])
        fill_image = label(filled_image)
        fill_regions = regionprops(fill_image)
        external_components[ii] = len([region.area for region in fill_regions if region.area > 100000])

        centroid_external[ii, :] = np.round(np.mean(np.argwhere(filled_image), axis=0))
        reduce_mask[:, :, ii] = filled_image * array[bounds[0]:bounds[3], bounds[1]:bounds[4], ii + bounds[2]]
        mask[bounds[0]:bounds[3], bounds[1]:bounds[4], ii + bounds[2]] = 1 * filled_image

    if only_mask:
        return mask
    else:
        return mask, reduce_mask, centroid_external, external_components, bounds
