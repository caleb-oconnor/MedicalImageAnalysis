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

import numpy as np
import SimpleITK as sitk


def euler_transform(matrix=None, angles=None, translation=None, rotation_center=None, zyx=False):
    """Creates a 3D Euler transformation that can combine rotation, translation, and a rotation center. Useful for rigid or
    semi-rigid registration in 3D.
    
    Examples
    --------
    >>> from medicalimageanalysis.utils.image.transform import euler_transform
    >>> matrix = euler_transform(10, 0, 0)
    """
    transform = sitk.Euler3DTransform()

    if angles is not None:
        rotation = [angles[0] * np.pi / 180, angles[1] * np.pi / 180, angles[2] * np.pi / 180]
        transform.SetRotation(rotation[0], rotation[1], rotation[2])

    if matrix is not None:
        transform.SetMatrix(matrix[:3, :3].flatten().astype(np.double))

    if translation is not None:
        transform.SetTranslation(translation)

    if rotation_center is not None:
        transform.SetCenter(rotation_center)

    if zyx:
        transform.SetComputeZYX(True)

    return transform
