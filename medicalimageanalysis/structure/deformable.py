"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import copy

import numpy as np
import SimpleITK as sitk

import vtk
from vtkmodules.util import numpy_support
from scipy.ndimage import map_coordinates

from ..data import Data
from ..utils.deformable.simpleitk import DeformableITK


class Display(object):
    def __init__(self, deformable):
        self.deformable = deformable

        self.origin = None
        self.spacing = None
        self.array = None
        self.image = None
        self.matrix = np.identity(3)

        self.slice_location = [0, 0, 0]
        self.scroll_max = None
        self.offset = {'Axial': [0, 0], 'Coronal': [0, 0], 'Sagittal': [0, 0]}
        self.misc = {}

        self.compute_scroll_max()

    def compute_array(self, slice_plane):
        array_slice = None
        if slice_plane == 'Axial':
            if 0 <= self.slice_location[0] < self.array.shape[0]:
                array_slice = self.array[self.slice_location[0], :, :].astype(np.double)

        elif slice_plane == 'Coronal':
            if 0 <= self.slice_location[1] < self.array.shape[1]:
                array_slice = self.array[:, self.slice_location[1], :].astype(np.double)

        else:
            if 0 <= self.slice_location[2] < self.array.shape[2]:
                array_slice = self.array[:, :, self.slice_location[2]].astype(np.double)

        return array_slice

    def compute_deformation(self):
        image = self.deformable.create_image()

        self.array = sitk.GetArrayFromImage(image)
        self.spacing = image.GetSpacing()
        self.origin = image.GetOrigin()

    def compute_grid(self, slice_plane='Axial', vector='x'):
        if slice_plane == 'Axial':
            dvf_plane = self.deformable.dvf[self.slice_location[0], :, :, :]
        elif slice_plane == 'Coronal':
            dvf_plane = self.deformable.dvf[:, self.slice_location[1], :, :]
        else:
            dvf_plane = self.deformable.dvf[:, :, self.slice_location[2], :]

        if vector == 'x':
            dvf_vector = dvf_plane[:, :, 0]
        elif vector == 'y':
            dvf_vector = dvf_plane[:, :, 1]
        else:
            dvf_vector = dvf_plane[:, :, 2]

        return dvf_vector.astype(np.float32)

    def compute_matrix_pixel_to_position(self):
        matrix = copy.deepcopy(self.matrix)

        pixel_to_position_matrix = np.identity(4, dtype=np.float32)
        pixel_to_position_matrix[:3, 0] = matrix[0, :] * self.spacing[0]
        pixel_to_position_matrix[:3, 1] = matrix[1, :] * self.spacing[1]
        pixel_to_position_matrix[:3, 2] = matrix[2, :] * self.spacing[2]
        pixel_to_position_matrix[:3, 3] = self.origin

        return pixel_to_position_matrix

    def compute_matrix_position_to_pixel(self):
        matrix = copy.deepcopy(self.matrix)

        hold_matrix = np.identity(3, dtype=np.float32)
        hold_matrix[0, :] = matrix[0, :] / self.spacing[0]
        hold_matrix[1, :] = matrix[1, :] / self.spacing[1]
        hold_matrix[2, :] = matrix[2, :] / self.spacing[2]

        position_to_pixel_matrix = np.identity(4, dtype=np.float32)
        position_to_pixel_matrix[:3, :3] = hold_matrix
        position_to_pixel_matrix[:3, 3] = np.asarray(self.origin).dot(-hold_matrix.T)

        return position_to_pixel_matrix

    def compute_mesh_slice(self, roi_name=None, location=None, slice_plane=None, return_pixel=False):
        if self.deformable.rois[roi_name] is None:
            self.deformable.update_rois(roi_name=roi_name)

    def compute_offset(self):
        if self.deformable.reference_name is not None:
            pos = Data.image[self.deformable.reference_name].origin

            self.offset['Axial'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
            self.offset['Axial'][1] = (self.origin[1] - pos[1]) / self.spacing[1]
            self.offset['Coronal'][0] = (self.origin[0] - pos[0]) / self.spacing[0]
            self.offset['Coronal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]
            self.offset['Sagittal'][0] = (self.origin[1] - pos[1]) / self.spacing[1]
            self.offset['Sagittal'][1] = (self.origin[2] - pos[2]) / self.spacing[2]

    def compute_slice_location(self, position=None):
        if position is None:
            source_location = np.flip(Data.image[self.deformable.reference_name].display.slice_location)
            position = Data.image[self.deformable.reference_name].display.compute_index_positions(source_location)
        self.slice_location = np.flip(np.round((position - self.origin) / self.spacing).astype(np.int32))

    def compute_slice_origin(self, slice_plane):
        slice_origin = None
        if slice_plane == 'Axial' and 0 <= self.slice_location[0] <= self.scroll_max[0]:
            location = np.asarray([0, 0, self.slice_location[0]])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Coronal' and 0 <= self.slice_location[1] <= self.scroll_max[1]:
            location = np.asarray([0, self.slice_location[1], 0])
            slice_origin = self.origin + (location * self.spacing)
        elif slice_plane == 'Sagittal' and 0 <= self.slice_location[2] <= self.scroll_max[2]:
            location = np.asarray([self.slice_location[2], 0, 0])
            slice_origin = self.origin + (location * self.spacing)

        return slice_origin

    def compute_scroll_max(self):
        if self.array is None:
            self.scroll_max = self.deformable.dimensions - 1
        else:
            self.scroll_max = [self.array.shape[0] - 1,
                               self.array.shape[1] - 1,
                               self.array.shape[2] - 1]

    def convert_position_to_pixel(self, position=None):
        position_to_pixel_matrix = self.compute_matrix_position_to_pixel()

        pixel = []
        for ii, pos in enumerate(position):
            p_concat = np.concatenate((pos, np.ones((pos.shape[0], 1))), axis=1)
            pixel_3_axis = p_concat.dot(position_to_pixel_matrix.T)[:, :3]
            pixel += [np.vstack((pixel_3_axis, pixel_3_axis[0, :]))]

        return pixel

    def update_slice_location(self, scroll, slice_plane):
        if slice_plane == 'Axial':
            self.slice_location[0] = scroll
        elif slice_plane == 'Coronal':
            self.slice_location[1] = scroll
        else:
            self.slice_location[2] = scroll


class Deformable(object):
    def __init__(self, dvf=None, origin=None, spacing=None, dimensions=None, roi_names=None, rigid_matrix=None,
                 dvf_matrix=None, registration_name=None, reference_name=None, moving_name=None, reference_sops=None,
                 moving_sops=None):
        self.reference_name = reference_name
        self.reference_sops = reference_sops
        self.moving_name = moving_name
        self.moving_sops = moving_sops
        self.roi_names = roi_names
        self.rigid_rois = dict.fromkeys(Data.roi_list)
        self.deformable_rois = dict.fromkeys(Data.roi_list)

        self.modality = None
        if dvf_matrix is not None:
            if np.allclose(dvf_matrix, np.identity(3), atol=1e-3):
                self.dvf = dvf
                self.origin = origin
                self.spacing = spacing
                self.dimensions = dimensions
            else:
                self.dvf, self.spacing, self.origin, self.dimensions = self.correct_dvf_direction(dvf, spacing, origin,
                                                                                                  dvf_matrix)

        else:
            self.dvf = dvf
            self.origin = origin
            self.spacing = spacing
            self.dimensions = dimensions

        if rigid_matrix is None:
            self.rigid_matrix = np.identity(4)
        else:
            self.rigid_matrix = rigid_matrix

        self.deformable_name = self.add_deformable(registration_name)

        self.rois = {}
        self.display = Display(self)

    def add_deformable(self, deformable_name):
        if deformable_name is None:
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
                        deformable_name = new_name
                        n = -100

        Data.deformable[deformable_name] = self
        Data.deformable_list += [deformable_name]

        return deformable_name

    def compute_aspect(self, slice_plane):
        if slice_plane == 'Axial':
            aspect = np.round(self.spacing[0] / self.spacing[1], 2)
        elif slice_plane == 'Coronal':
            aspect = np.round(self.spacing[0] / self.spacing[2], 2)
        else:
            aspect = np.round(self.spacing[1] / self.spacing[2], 2)

        return aspect

    def compute_biomechanical(self):
        pass

    def compute_bspline(self, modality_gradient=True, sigma=2, control_spacing=None, mesh_size=None, gradient=1e-5,
                        iterations=100, crop=5):
        ref_image = mia.Data.image[self.reference_name].create_sitk_image()
        mov_image = mia.Data.image[self.moving_name].create_sitk_image()

        euler = sitk.Euler3DTransform()
        euler.SetMatrix(self.rigid_matrix[:3, :3].flatten().tolist())
        euler.SetTranslation(self.rigid_matrix[:3, 3])

        mov_resampled = sitk.Resample(mov_image, ref_image, euler, sitk.sitkLinear, 0.0, mov_image.GetPixelID())

        ref_mask = None
        mov_mask = None
        for roi_name in self.roi_names:
            ref_roi = mia.Data.image[self.reference_name].rois[roi_name]
            mov_roi = mia.Data.image[self.moving_name].rois[roi_name]
            if ref_roi.mesh is not None or ref_roi.contour_pixel is not None:
                if mov_roi.mesh is not None or mov_roi.contour_pixel is not None:
                    if ref_mask is None:
                        ref_mask = ref_roi.compute_mask()
                    else:
                        ref_mask = ref_mask + ref_roi.compute_mask()

                        if mov_mask is None:
                            mov_mask = mov_roi.compute_mask()
                        else:
                            mov_mask = mov_mask + mov_roi.compute_mask()

        deform_itk = mia.utils.deformable.simpleitk.DeformableITK(reference_image=ref_image,
                                                                  moving_image=mov_resampled,
                                                                  reference_mask=None,
                                                                  moving_mask=None)

        if mia.Data.image[self.reference_name].modality != mia.Data.image[self.moving_name].modality and modality_gradient:
            deform_itk.cross_modality_correction()

        if ref_mask is not None and mov_mask is not None:
            deform_itk.create_sitk_image(ref_mask, ref.origin, ref.spacing, ref.matrix, mask=True)
            deform_itk.create_sitk_image(mov_mask, mov.origin, mov.spacing, mov.matrix, reference=False, mask=True)

            if sigma is not None:
                deform_itk.blur_mask(sigma=sigma)

        deform_itk.resample()
        dvf_image = deform_itk.bspline(control_spacing=control_spacing, mesh_size=mesh_size, gradient=gradient,
                                       iterations=iterations, crop=crop)

        self.origin = dvf_image.GetOrigin()
        self.spacing = dvf_image.GetSpacing()
        self.dvf = sitk.GetArrayFromImage(dvf_image)

    def compute_demons(self, method=None, modality_gradient=True, sigma=2, smooth=True, std=1, iterations=50,
                       intensity_threshold=0.001, step=2.0, crop=5):
        ref = mia.Data.image[self.reference_name]
        mov = mia.Data.image[self.moving_name]

        deform_itk = mia.utils.deformable.simpleitk.DeformableITK()
        deform_itk.create_sitk_image(ref.array, ref.origin, ref.spacing, ref.matrix)
        deform_itk.create_sitk_image(mov.array, mov.origin, mov.spacing, mov.matrix, reference=False)

        if mia.Data.image[self.reference_name].modality != mia.Data.image[self.moving_name].modality and modality_gradient:
            deform_itk.cross_modality_correction()

        ref_mask = None
        mov_mask = None
        for roi_name in self.roi_names:
            ref_roi = mia.Data.image[self.reference_name].rois[roi_name]
            mov_roi = mia.Data.image[self.moving_name].rois[roi_name]
            if ref_roi.mesh is not None or ref_roi.contour_pixel is not None:
                if mov_roi.mesh is not None or mov_roi.contour_pixel is not None:
                    if ref_mask is None:
                        ref_mask = ref_roi.compute_mask()
                    else:
                        ref_mask = ref_mask + ref_roi.compute_mask()

                        if mov_mask is None:
                            mov_mask = mov_roi.compute_mask()
                        else:
                            mov_mask = mov_mask + mov_roi.compute_mask()

        if ref_mask is not None and mov_mask is not None:
            deform_itk.create_sitk_image(ref_mask, ref.origin, ref.spacing, ref.matrix, mask=True)
            deform_itk.create_sitk_image(mov_mask, mov.origin, mov.spacing, mov.matrix, reference=False, mask=True)

            if sigma is not None:
                deform_itk.blur_mask(sigma=sigma)

        deform_itk.resample()
        if method in ['Demons', 'demons']:
            dvf_image = deform_itk.demons(smooth=smooth, std=std, iterations=iterations,
                                          intensity_threshold=intensity_threshold, crop=crop)

        elif method in ['Diffeomorphic', 'diffeomorphic']:
            dvf_image = deform_itk.diffeomorphic(smooth=smooth, std=std, iterations=iterations,
                                                 intensity_threshold=intensity_threshold, crop=crop)

        else:
            dvf_image = deform_itk.fast_demons(smooth=smooth, std=std, iterations=iterations,
                                               intensity_threshold=intensity_threshold, step=step, crop=crop)

        self.origin = dvf_image.GetOrigin()
        self.spacing = dvf_image.GetSpacing()
        self.dvf = sitk.GetArrayFromImage(dvf_image)

    @staticmethod
    def correct_dvf_direction(dvf, spacing, origin, matrix):
        """
        Rotate a DVF to a new direction, adjusting vectors and origin so that the DVF
        stays aligned in physical space (rotation about the center of the volume).
        """
        D_new = np.identity(3)
        R = D_new @ np.linalg.inv(matrix)

        center_index = (np.flip(dvf.shape)[1:] - 1) / 2.0
        center_phys = origin + matrix @ (center_index * spacing)

        Z, Y, X, _ = dvf.shape
        dvf_flat = dvf.reshape(-1, 3).T
        dvf_rotated_flat = R @ dvf_flat
        dvf_rotated = dvf_rotated_flat.T.reshape(Z, Y, X, 3)

        center_phys_new = center_phys  # center should not move
        origin_new = center_phys_new - D_new @ (center_index * spacing)

        dimensions = dvf_rotated.shape[0:3]

        return dvf_rotated, spacing, origin_new, dimensions

    def create_image(self):
        R_inv = self.rigid_matrix[:3, :3].T
        t_inv = -R_inv @ self.rigid_matrix[:3, 3]

        affine = sitk.AffineTransform(3)
        affine.SetMatrix(R_inv.flatten())
        affine.SetTranslation(t_inv)

        ref_image = Data.image[self.reference_name].create_sitk_image()
        moving_image = Data.image[self.moving_name].create_sitk_image()

        resampled_image = sitk.Resample(
            moving_image,
            size=ref_image.GetSize(),
            transform=affine,
            outputOrigin=ref_image.GetOrigin(),
            outputSpacing=ref_image.GetSpacing(),
            interpolator=sitk.sitkLinear,
            defaultPixelValue=-3001,
            outputPixelType=moving_image.GetPixelID()
        )

        dvf_initial_image = sitk.GetImageFromArray(self.dvf, isVector=True)
        dvf_initial_image.SetSpacing(self.spacing)
        dvf_initial_image.SetOrigin(self.origin)

        cropped_image = sitk.Resample(
            resampled_image,
            dvf_initial_image,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            resampled_image.GetPixelID()
        )

        dvf_transform = sitk.DisplacementFieldTransform(dvf_initial_image)
        dvf_resampled = sitk.ResampleImageFilter()
        dvf_resampled.SetReferenceImage(cropped_image)
        dvf_resampled.SetTransform(dvf_transform)
        dvf_resampled.SetInterpolator(sitk.sitkLinear)
        dvf_resampled.SetDefaultPixelValue(0)
        dvf_resampled.SetOutputDirection(np.identity(3).flatten().tolist())

        return dvf_resampled.Execute(resampled_image)

    def export_image(self, path=None):
        if self.moving_name is not None and path is not None:
            image = self.create_image()

            writer = vtk.vtkMetaImageWriter()
            writer.SetInputData(image)
            writer.SetFileName(path)
            writer.Write()

    def retrieve_array_plane(self, slice_plane, vector=None):
        if vector is None:
            return self.display.compute_array(slice_plane)
        elif vector == 'x':
            return self.display.compute_grid(slice_plane=slice_plane, vector=vector)
        elif vector == 'y':
            return self.display.compute_grid(slice_plane=slice_plane, vector=vector)
        elif vector == 'z':
            return self.display.compute_grid(slice_plane=slice_plane, vector=vector)

    def retrieve_grid(self, slice_plane='Axial', vector='x'):
        return self.display.compute_grid(slice_plane=slice_plane, vector=vector)

    def retrieve_slice_location(self, slice_plane):
        if slice_plane == 'Axial':
            return self.display.slice_location[0]

        elif slice_plane == 'Coronal':
            return self.display.slice_location[1]

        else:
            return self.display.slice_location[2]

    def retrieve_slice_position(self, slice_plane=None):
        pixel_to_position_matrix = self.display.compute_matrix_pixel_to_position()

        if slice_plane is None:
            location = np.asarray([self.display.slice_location[2],
                                   self.display.slice_location[1],
                                   self.display.slice_location[0], 1])
        else:
            if slice_plane == 'Axial':
                location = np.asarray([0, 0, self.display.slice_location[0], 1])
            elif slice_plane == 'Coronal':
                location = np.asarray([0, self.display.slice_location[1], 0, 1])
                print(location)
            else:
                location = np.asarray([self.display.slice_location[2], 0, 0, 1])

        return location.dot(pixel_to_position_matrix.T)[:3]

    def retrieve_scroll_max(self, slice_plane):
        if slice_plane == 'Axial':
            return self.display.scroll_max[0]

        elif slice_plane == 'Coronal':
            return self.display.scroll_max[1]

        else:
            return self.display.scroll_max[2]

    def save_deformable(self, path):
        variable_names = self.__dict__.keys()
        column_names = [name for name in variable_names if name not in ['deformable_rois',
                                                                        'display',
                                                                        'dvf',
                                                                        'rigid_rois']]

        df = pd.DataFrame(index=[0], columns=column_names)
        for name in column_names:
            df.at[0, name] = getattr(self, name)

        df.to_pickle(os.path.join(path, 'info.p'))
        np.save(os.path.join(path, 'dvf.npy'), self.dvf, allow_pickle=True)

    def update_rois(self, roi_name=None):
        for name in list(self.rois.keys()):
            if name not in Data.roi_list:
                del self.rois[name]

        for name in Data.roi_list:
            if name not in list(self.rois.keys()):
                self.rois[name] = None

        for name in Data.roi_list:
            if roi_name is None or name == roi_name:
                roi = Data.image[self.moving_name].rois[name]
                if roi.mesh is not None and roi.visible:
                    self.rigid_rois[name] = roi.mesh.transform(self.rigid_matrix, inplace=False)

                    points = self.rigid_rois[name].points
                    voxel_coords = (points - self.origin) / self.spacing
                    deformed_points = points.copy()
                    for i in range(3):
                        deformed_points[:, i] += map_coordinates(
                            self.dvf[..., i],
                            [voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]],
                            order=1,
                            mode='nearest'
                        )

                    self.deformable_rois[name] = self.rigid_rois[name].copy()
                    self.deformable_rois.points = deformed_points
