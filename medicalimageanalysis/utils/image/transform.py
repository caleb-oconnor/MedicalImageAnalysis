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


def euler_transform(matrix=None, angles=None, translation=None, rotation_center=None):
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

    transform.SetComputeZYX(True)

    return transform
