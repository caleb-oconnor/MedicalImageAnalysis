"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
"""

import cv2
import numpy as np


def contours_from_mask(mask, plane='Axial'):
    if plane == 'Axial':
        slices = mask.shape[2]
    elif plane == 'Coronal':
        slices = mask.shape[1]
    else:
        slices = mask.shape[0]

    contours = []
    for ii in range(slices):
        if plane == 'Axial':
            tuple_contour, _ = cv2.findContours(mask[:, :, ii], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours += [np.concatenate((np.vstack(t), ii * np.ones((len(t), 1))), axis=1) for t in tuple_contour]
        elif plane == 'Coronal':
            tuple_contour, _ = cv2.findContours(mask[:, ii, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for t in tuple_contour:
                stack = np.vstack(t)
                contours += [np.vstack((stack[:, 0], ii * np.ones(len(t)), stack[:, 1])).T]
        else:
            tuple_contour, _ = cv2.findContours(mask[ii, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours += [np.concatenate((ii * np.ones((len(t), 1)), np.vstack(t)), axis=1) for t in tuple_contour]

    return contours
