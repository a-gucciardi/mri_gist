import ants
import nibabel as nib
import numpy as np

from scipy.linalg import sqrtm, inv
from scipy.ndimage import affine_transform

### Script for geometrical hemisphere separation

input_image = 'mri_register/dhcp/sub-CC00060XX03_ses-12501_T1w_MNI.nii.gz'
flipped_image = 'flipped_image.nii.gz'
flirt_matrix_path = 'flirt_transform.mat'
midway_image_path = 'midway_image.nii.gz'

# - Step 1: Load original image and flip across sagittal plane (left-right flip)
# (assuming RAS : axis 0 is left-right)
img = nib.load(input_image)
data = img.get_fdata()
# Flip
flipped_data = np.flip(data, axis=0)
flipped_img = nib.Nifti1Image(flipped_data, img.affine)

# - Step 2: Run registration
# Using ANTsPy Image format
original_ants_img = ants.from_numpy(data)
flipped_ants_img = ants.from_numpy(flipped_data)
# registration 
registration = ants.registration(fixed=flipped_ants_img, moving=original_ants_img, type_of_transform='Affine')
T = ants.read_transform(registration['fwdtransforms'][0]) #https://github.com/ANTsX/ANTsPy/wiki/ANTs-transform-concepts-and-file-formats
# half-registration
T_half = sqrtm(np.array(T.parameters[:9].reshape(3, 3)))
transformed_data = affine_transform(data, T_half, offset=T.parameters[9:])
transformed_ants = ants.apply_transforms(original_ants_img, original_ants_img, registration['fwdtransforms'][0]) 

# # Save the transformed image
# midway_img = nib.Nifti1Image(transformed_data, img.affine)
# nib.save(midway_img, 'midway_image_ants.nii.gz')

# - Step 3: Separate hemispheres
mid_sagittal_index = data.shape[0] // 2
left_hemisphere_data = transformed_data[:mid_sagittal_index, :, :]  # shape (/2,:,:)
right_hemisphere_data = transformed_data[mid_sagittal_index:, :, :]


# --- Step 4: Save the hemispheres as separate NIfTI images
# Create nibabel NIfTI images for left and right hemispheres, add offset for second hemi
left_hemisphere_img = nib.Nifti1Image(left_hemisphere_data, img.affine)

right_hemisphere_affine = img.affine.copy()
right_hemisphere_affine[0, 3] += data.shape[0] / 2  # Adding the offset along the x-axis
right_hemisphere_img = nib.Nifti1Image(right_hemisphere_data, right_hemisphere_affine)

nib.save(left_hemisphere_img, 'left_hemisphere_d.nii.gz')
nib.save(right_hemisphere_img, 'right_hemisphere_d.nii.gz')

print("Left and right hemispheres saved successfully.")

# Ants version, unsure but seems worse at the moment
# - Step 3: Separate hemispheres
mid_sagittal_index = data.shape[0] // 2
left_hemisphere_data_ants = transformed_ants[:mid_sagittal_index, :, :]  # shape (/2,:,:)
right_hemisphere_data_ants = transformed_ants[mid_sagittal_index:, :, :]


# --- Step 4: Save the hemispheres as separate NIfTI images
# Create nibabel NIfTI images for left and right hemispheres, add offset for second hemi
left_hemisphere_img_ants = nib.Nifti1Image(left_hemisphere_data_ants.numpy(), img.affine)

right_hemisphere_affine_ants = img.affine.copy()
right_hemisphere_affine_ants[0, 3] += data.shape[0] / 2  # Adding the offset along the x-axis
right_hemisphere_img_ants = nib.Nifti1Image(right_hemisphere_data_ants.numpy(), right_hemisphere_affine_ants)

nib.save(left_hemisphere_img_ants, 'left_hemisphere_dants.nii.gz')
nib.save(right_hemisphere_img_ants, 'right_hemisphere_dants.nii.gz')

print("Left and right hemispheres saved successfully.")