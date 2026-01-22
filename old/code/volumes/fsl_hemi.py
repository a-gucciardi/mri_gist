import nibabel as nib
import numpy as np
import subprocess
from scipy.linalg import sqrtm, inv
from scipy.ndimage import affine_transform
from nibabel.orientations import aff2axcodes

debug = True

# Paths (change these as needed)
input_image = 'mri_register/dhcp/sub-CC00060XX03_ses-12501_T1w_MNI.nii.gz'
flipped_image = 'flipped_image.nii.gz'
flirt_matrix_path = 'flirt_transform.mat'
midway_image_path = 'midway_image.nii.gz'

# --- Step 1: Load original image and flip across sagittal plane (left-right flip)
img = nib.load(input_image)
data = img.get_fdata()


# # Flip along the first axis (assuming RAS : axis 0 is left-right)
flipped_data = np.flip(data, axis=0)
flipped_img = nib.Nifti1Image(flipped_data, img.affine)
# nib.save(flipped_img, flipped_image)
# print("Save flipped image at ", flipped_image)

if debug:
    axcodes = aff2axcodes(img.affine)
    print(img.affine)
    print(axcodes)

# --- Step 2: Run FLIRT registration (original to flipped)
flirt_cmd = [
    'flirt',
    '-in', input_image,
    '-ref', flipped_image,
    '-omat', flirt_matrix_path,
    '-dof', '6'  # rigid-body registration
]

subprocess.run(flirt_cmd, check=True)
print("Saved orig->flipped flirt matrix", flirt_matrix_path)

# --- Step 3: Compute midway transform (matrix square root)
T = np.loadtxt(flirt_matrix_path)  # 4x4 affine matrix from FLIRT
T_half = sqrtm(T)

if debug:
    print(T.astype(int))
    print(T_half)


# Step 4: Mid-sagittal plane in midway space (centered halfway)

T_half_inv = inv(T_half)
mid_x = data.shape[0] // 2 # Midpoint along the left-right axis
y, z = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[2]))
midway_coords = np.vstack([np.full_like(y, mid_x), y.ravel(), z.ravel(), np.ones(y.size)])
# Apply the inverse affine transformation to these coordinates
original_coords = T_half_inv @ midway_coords
original_coords = original_coords[:3, :].T  # Discard the homogeneous coordinate
# Get the x, y, z coordinates back into the original image space
midline_x_original = np.mean(original_coords[:, 0])

# --- Step 5: Now split the image based on this midpoint in the original space

# Left hemisphere: everything before the midline in original space
# Right hemisphere: everything after the midline in original space
left_hemisphere = data[:int(midline_x_original), :, :]
right_hemisphere = data[int(midline_x_original):, :, :]

# Save

nib.save(nib.Nifti1Image(left_hemisphere, img.affine), 'left_hemisphere_c.nii.gz')
nib.save(nib.Nifti1Image(right_hemisphere, img.affine), 'right_hemisphere_c.nii.gz')
print("saved")

# --- Step 4: Apply midway transform to original image
# NOTE: This blurs the image
# Apply to the data using scipy affine_transform (which expects matrix + offset)
# rotation_matrix = T_half[:3, :3]
# translation = T_half[:3, 3]

# transformed_data = affine_transform(
#     data,
#     rotation_matrix,
#     offset=translation,
#     order=1  # linear interpolation
# )

# # # --- Step 5: Save result as a new NIfTI image
# midway_img = nib.Nifti1Image(transformed_data, img.affine)
# nib.save(midway_img, midway_image_path)

# print(f'Midway image saved to {midway_image_path}')
