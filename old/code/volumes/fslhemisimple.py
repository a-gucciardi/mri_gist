import numpy as np
import nibabel as nib
from scipy.linalg import inv

# Load original image
input_image = 'mri_register/dhcp/sub-CC00060XX03_ses-12501_T1w_MNI.nii.gz'
img = nib.load(input_image)
data = img.get_fdata()

# Load the midway transform (T_half from earlier)
T_half = np.loadtxt('flirt_transform.mat')  # You could save this after computing it earlier

# Invert midway transform (go from midway space back to original space)
T_half_inv = inv(T_half)

# Step 1: Mid-sagittal plane in midway space (centered halfway)
mid_x = data.shape[0] // 2
mid_plane_coords = np.array([
    [mid_x, y, z, 1]  # homogeneous coordinates for each voxel along the plane
    for y in range(data.shape[1])
    for z in range(data.shape[2])
])

# Step 2: Map these coordinates back to original space using T_half_inv
original_space_coords = (T_half_inv @ mid_plane_coords.T).T[:, :3]  # drop homogeneous coord

# Step 3: Use mapped plane coordinates to split the original image
# This is a bit tricky — ideally you could fit a plane (Ax + By + Cz + D = 0) to these points
# and split voxels based on which side they fall on.

# In practice, if the transform is mostly rigid, the mid-sagittal plane in original space is roughly a vertical plane.
# So you could just compute the "midline x" in the original space from these points (average x-coord).
midline_x = np.mean(original_space_coords[:, 0])

# Split hemispheres in the original image
left_hemisphere = data[:int(midline_x), :, :]
right_hemisphere = data[int(midline_x):, :, :]

# Save left and right hemispheres (no blurring since no resampling happened!)
nib.save(nib.Nifti1Image(left_hemisphere, img.affine), 'left_hemisphere.nii.gz')
nib.save(nib.Nifti1Image(right_hemisphere, img.affine), 'right_hemisphere.nii.gz')
