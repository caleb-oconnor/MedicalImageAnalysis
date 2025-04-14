"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
"""

import time

import math
import numpy as np

from skimage.filters import sato
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops

from ..conversion import ContourToDiscreteMesh


def compute_external(array_slice, threshold=-250):
    binary = array_slice > threshold
    label_image = label(binary)
    label_regions = regionprops(label_image)
    region_areas = [region.area for region in label_regions]
    max_idx = np.argmax(region_areas)
    bounds = label_regions[max_idx].bbox

    box_image = label_regions[max_idx].image

    mask = np.zeros(array_slice.shape)
    external = np.zeros(box_image.shape)
    centroid_external = np.zeros((box_image.shape[2], 2))
    external_components = np.zeros((box_image.shape[2], 1))
    for ii in range(box_image.shape[2]):
        filled_image = binary_fill_holes(box_image[:, :, ii])
        fill_image = label(filled_image)
        fill_regions = regionprops(fill_image)
        external_components[ii] = len([region.area for region in fill_regions if region.area > 100000])

        centroid_external[ii, :] = np.round(np.mean(np.argwhere(filled_image), axis=0))
        external[:, :, ii] = filled_image * array_slice[bounds[0]:bounds[3], bounds[1]:bounds[4], ii + bounds[2]]
        mask[bounds[0]:bounds[3], bounds[1]:bounds[4], ii + bounds[2]] = 1 * filled_image
        # external[:, :, ii] = filled_image * array_slice[bounds[0]:bounds[3], bounds[1]:bounds[4], ii + bounds[2]]

    return external, mask, centroid_external, external_components, bounds


class CT(object):
    def __init__(self, image, external_threshold=-250):
        self.image = image
        self.hu_lines = []
        self.hu_lines_full = []

        t1 = time.time()
        self.external, self.mask, self.centroid, components, self.bounds = compute_external(self.image.array,
                                                                                            threshold=external_threshold)
        self.compute_hu_lines(same_centroid=True)
        print(np.round(time.time() - t1, 3))

    def compute_hu_lines(self, x_center=300, y_center=300, radius=300, interval=5, steps=300, same_centroid=False):
        if same_centroid:
            centroids = np.repeat([np.mean(self.centroid, axis=0).astype(np.int32)], repeats=len(self.centroid), axis=0)
        else:
            centroids = self.centroid

        point_idx = []
        for angle in range(0, 360, interval):
            angle_rad = math.radians(angle)
            x = x_center + radius * math.cos(angle_rad)
            y = y_center + + radius * math.sin(angle_rad)

            point_diff = np.asarray([x, y]) - np.asarray([x_center, y_center])
            step = point_diff / steps
            point_idx += [np.flip([tuple(np.asarray([x_center, y_center]) + i * step) for i in range(steps + 1)])]

        x_shape, y_shape, _ = self.image.array.shape
        centroid_unique = np.unique(centroids, axis=0)
        centroid_idx = []
        for ii, cent in enumerate(centroid_unique):
            center = cent + [self.bounds[0], self.bounds[1]]
            diff = np.asarray([x_center, y_center]) - center
            point_correction = [idx - diff for idx in point_idx]

            point_hold = []
            for correct in point_correction:
                point_hold += [np.asarray([[int(c[0]), int(c[1])] for c in correct if 0 <= c[0] < x_shape and 0 <= c[1] < y_shape])]

            centroid_idx += [point_hold]

        for ii, c in enumerate(centroids):
            if same_centroid:
                lines_idx = centroid_idx[0]
            else:
                select_lines = int(np.where((centroid_unique[:, 0] == c[0]) & (centroid_unique[:, 1] == c[1]))[0][0])
                lines_idx = centroid_idx[select_lines]

            mask_slice = self.mask[:, :, ii]
            array_slice = self.image.array[:, :, ii]

            mask_idx = [np.where(mask_slice[line[:, 0], line[:, 1]] > 0) for line in lines_idx]
            all_hu_lines = [array_slice[line[:, 0], line[:, 1]] for line in lines_idx]
            keep_hu_lines = [all_hu_lines[ii][m[0][0]:] if len(m) > 0 else all_hu_lines[ii][-1] for ii, m in enumerate(mask_idx)]
            self.hu_lines += [keep_hu_lines]
            self.hu_lines_full += [all_hu_lines]

    def fat_removal(self, threshold=0):
        t1 = time.time()
        binary = self.external > threshold
        label_image = label(binary)
        label_regions = regionprops(label_image)
        region_areas = [region.area for region in label_regions]
        print(np.round(time.time() - t1, 3))
        print(1)

    def slice_central_deviation(self):
        central_std = np.zeros((self.external.shape[2], 1))
        central_mean = np.zeros((self.external.shape[2], 1))
        central_array = np.zeros((50, 50, self.external.shape[2]))
        for ii in range(self.external.shape[2]):
            central_array[:, :, ii] = self.external[int(self.centroid[ii, 0] - 25):int(self.centroid[ii, 0] + 25),
                                      int(self.centroid[ii, 1] - 25):int(self.centroid[ii, 1] + 25), ii]
            central_mean[ii] = np.mean(central_array[:, :, ii])
            central_std[ii] = np.std(central_array[:, :, ii])
        central_median_mean = np.mean(central_mean)
        central_median_std = np.median(central_std)

    def external_bining(self):
        low_regions = np.arange(-251, 1, 25)
        high_regions = np.arange(1, 501, 25)

        low_bins = np.zeros((len(low_regions), 1))
        for ii, low in enumerate(low_regions):
            low_bins[ii] = np.sum(self.external < low)

        high_bins = np.zeros((len(high_regions), 1))
        for ii, high in enumerate(high_regions):
            high_bins[ii] = np.sum(self.external > high)

    def sato_filter(self, sigmas):
        return sato(self.array, sigmas=range(1, 3, 5), black_ridges=True, mode='reflect', cval=0)
