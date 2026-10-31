"""Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Examples
--------
>>> import numpy as np
>>> import medicalimageanalysis as mia
>>> mask = np.zeros((8, 32, 32), dtype=np.uint8)
"""

import cv2
import numpy as np


def contours_from_mask(mask, plane='Axial'):
    """Extract contour polylines from a binary mask.
    
    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask in ``(z, y, x)`` order.
    plane : str, optional
        Plane along which contours are extracted: ``'Axial'``, ``'Coronal'``,
        or ``'Sagittal'``.
    
    Returns
    -------
    list[numpy.ndarray]
        Contours represented in voxel coordinates, one array per connected
        contour on each processed slice.
    
    Examples
    --------
    >>> import numpy as np
    >>> from medicalimageanalysis.utils.roi.contour import contours_from_mask
    >>> mask = np.zeros((4, 16, 16), dtype=np.uint8)
    >>> contours = contours_from_mask(mask, plane='Axial')
    """
    mask = mask.astype(np.uint8)

    if plane == 'Axial':
        slices = mask.shape[0]
    elif plane == 'Coronal':
        slices = mask.shape[1]
    else:
        slices = mask.shape[2]

    contours = []
    for ii in range(slices):
        if plane == 'Axial':
            tuple_contour, _ = cv2.findContours(mask[ii, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours += [np.concatenate((np.vstack(t), ii * np.ones((len(t), 1))), axis=1) for t in tuple_contour]
        elif plane == 'Coronal':
            tuple_contour, _ = cv2.findContours(mask[:, ii, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for t in tuple_contour:
                stack = np.vstack(t)
                contours += [np.vstack((stack[:, 0], ii * np.ones(len(t)), stack[:, 1])).T]
        else:
            tuple_contour, _ = cv2.findContours(mask[:, :, ii], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours += [np.concatenate((ii * np.ones((len(t), 1)), np.vstack(t)), axis=1) for t in tuple_contour]

    return contours
