"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import numpy as np


class Deformable(object):
    def __init__(self, dvf, origin, spacing, dimensions, roi_names=None, rigid_matrix=None, dvf_matrix=None,
                 registration_name=None, reference_name=None, moving_name=None, reference_sops=None, moving_sops=None):
        self.reference_name = reference_name
        self.reference_sops = reference_sops
        self.moving_name = moving_name
        self.moving_sops = moving_sops
        self.roi_names = roi_names

        self.dvf = dvf
        self.origin = origin
        self.spacing = spacing
        self.dimensions = dimensions

        if rigid_matrix is None:
            self.rigid_matrix = np.identity(4)
        else:
            self.rigid_matrix = rigid_matrix

        if dvf_matrix is None:
            self.dvf_matrix = np.identity(4)
        else:
            self.dvf_matrix = dvf_matrix

        self.deformable_name = self.add_deformable(registration_name)

        # self.display = Display(self)

    def add_deformable(self, deformable_name):
        if deformable_name is None:
            if np.array_equal(self.combo_matrix, np.identity(4)):
                if self.reference_name is None and self.moving_name is None:
                    deformable_name = 'DVF_Unknown'
                else:
                    deformable_name = 'DVF_' + self.reference_name + '_' + self.moving_name

            if deformable_name in Data.deformable_list:
                n = 0
                while n > -1:
                    n += 1
                    new_name = copy.deepcopy(deformable_name + '_' + str(n))
                    if new_name not in Data.deformable_list:
                        n = -100

        Data.deformable[deformable_name] = self
        Data.deformable_list += [deformable_name]

        return deformable_name
