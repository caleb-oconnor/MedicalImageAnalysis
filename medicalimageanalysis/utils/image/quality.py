"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:
"""

from skimage.filters import sato


class CT(object):
    def __init__(self, array):
        self.array = array

    def sato_filter(self, sigmas):
        return sato(self.array, sigmas=range(1, 3, 5), black_ridges=True, mode='reflect', cval=0)
