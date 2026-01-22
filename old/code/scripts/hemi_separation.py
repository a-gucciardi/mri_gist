import ants
import nibabel as nib
import numpy as np
import subprocess
from scipy.linalg import sqrtm
from scipy.ndimage import affine_transform
from pick import pick

# Hemisphere separation, midway registration method :
# we take the halfway trasnformation between mirror from the front (left-right) axis
# Two methods usable, antspy or fls-flirt (requires fls installed). For each method :
# 1- load and flip the data
# 2- Register, save transform
# 3- Generate hemispheres from the half-transform
# 4- Save hemis separately

# Paths test
input_image = 'mri_register/dhcp/sub-CC00060XX03_ses-12501_T1w_MNI.nii.gz'

# pick method selection
title = "Select registration method (space+enter): \n  ANTsPY (best test results) \t FSL-flirt (slower)"
options = ["ANTsPy", "FSL-flirt"]
method = pick(options, title, multiselect=True, min_selection_count=1)

if 'ANTsPy' in [i[0] for i in method]:
    print("Running ANTsPy")
    # 1
    img = nib.load(input_image)
    data = img.get_fdata()
    flipped_data = np.flip(data, axis=0)

    # 2
    original_ants_img = ants.from_numpy(data)
    flipped_ants_img = ants.from_numpy(flipped_data)
    registration = ants.registration(fixed=flipped_ants_img, moving=original_ants_img, type_of_transform='Affine')
    T = ants.read_transform(registration['fwdtransforms'][0]) #https://github.com/ANTsX/ANTsPy/wiki/ANTs-transform-concepts-and-file-formats

    # 2 - Midway transformation
    T_half = sqrtm(np.array(T.parameters[:9].reshape(3, 3)))
    # transformed_data = affine_transform(data, T_half, offset=T.parameters[9:])
    transformed_ants = ants.apply_transforms(original_ants_img, original_ants_img, registration['fwdtransforms'][0])

    # # 3 - Hemisphere
    mid_sagittal_index = data.shape[0] // 2
    left_hemisphere_data_ants = transformed_ants[:mid_sagittal_index, :, :]  # shape (/2,:,:)
    right_hemisphere_data_ants = transformed_ants[mid_sagittal_index:, :, :]

    left_hemisphere_img_ants = nib.Nifti1Image(left_hemisphere_data_ants.numpy(), img.affine)
    right_hemisphere_affine_ants = img.affine.copy()
    right_hemisphere_affine_ants[0, 3] += data.shape[0] / 2  # Adding the offset along the x-axis
    right_hemisphere_img_ants = nib.Nifti1Image(right_hemisphere_data_ants.numpy(), right_hemisphere_affine_ants)

    # 4
    left_path, right_path = 'left_hemisphere_ants.nii.gz', 'right_hemisphere_ants.nii.gz'
    nib.save(left_hemisphere_img_ants, left_path)
    nib.save(right_hemisphere_img_ants, right_path)

if 'FSL-flirt' in [i[0] for i in method]:
    print("Running FSL-flirt")

    # 1 - flirt requires disk-save
    flipped_image = "flipped_flirt.nii.gz"
    flirt_matrix_path = "flirt_transform.mat"
    img = nib.load(input_image)
    data = img.get_fdata()
    flipped_data = np.flip(data, axis=0)
    nib.save(nib.Nifti1Image(flipped_data, img.affine), flipped_image)

    # 2 - Run FLIRT. load result transofrm
    subprocess.run([
        "flirt",
        "-in", input_image,
        "-ref", flipped_image,
        "-omat", flirt_matrix_path,
        "-dof", "12",
        "-cost", "mutualinfo"
    ])
    T = np.loadtxt(flirt_matrix_path)

    # 2 - Midway transformation
    T_half = sqrtm(T[:3, :3])
    transformed_data = affine_transform(data, T_half, offset=T[:3, 3])

    # 3 - Hemisphere Masks
    mid_sagittal_index = data.shape[0] // 2
    left_mask, right_mask = np.zeros_like(data), np.zeros_like(data)
    left_mask[:mid_sagittal_index, :, :] = 1
    right_mask[mid_sagittal_index:, :, :] = 1
    left_hemisphere_data = transformed_data * left_mask
    right_hemisphere_data = transformed_data * right_mask

    # 4 - Save
    left_path, right_path = 'left_hemisphere_flirt.nii.gz', 'right_hemisphere_flirt.nii.gz'
    nib.save(nib.Nifti1Image(left_hemisphere_data, img.affine), left_path)
    nib.save(nib.Nifti1Image(right_hemisphere_data, img.affine), right_path)

print(f"Left and right hemispheres saved to {left_path} and {right_path}")
