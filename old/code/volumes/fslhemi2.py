import nibabel as nib
import numpy as np
from scipy.linalg import sqrtm, inv

input_image = 'mri_register/dhcp/sub-CC00060XX03_ses-12501_T1w_MNI.nii.gz'
# --- Load data and transformation matrix ---
img = nib.load(input_image)
data = img.get_fdata()

T = np.loadtxt('flirt_transform.mat') 
T_half = sqrtm(T)
T_half_inv = inv(T_half)

# --- Find coordinates in midway space ---
mid_x = data.shape[0] // 2  # Middle sagittal slice
y, z = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[2]), indexing='ij')
midway_plane_coords = np.column_stack([
    np.full(y.shape, mid_x).ravel(),
    y.ravel(),
    z.ravel(),
    np.ones(y.size)
])

# --- Map midway plane coords back to original space ---
original_space_coords = (T_half_inv @ midway_plane_coords.T).T[:, :3]  # Drop homogeneous

# --- Fit plane Ax + By + Cz + D = 0 to these points ---
X = original_space_coords[:, 0]
Y = original_space_coords[:, 1]
Z = original_space_coords[:, 2]

# Solve for [A, B, C, D] using homogeneous least squares
# We want to solve: [X Y Z 1] @ [A, B, C, D] = 0
ones = np.ones_like(X)
A = np.column_stack([X, Y, Z, ones])

# Compute null space of A (smallest singular value vector)
_, _, vh = np.linalg.svd(A)
plane_coeffs = vh[-1, :]  # Last row of V^T = right singular vector corresponding to smallest singular value

A, B, C, D = plane_coeffs

# --- Classify voxels into left/right hemispheres based on signed distance to plane ---
# Plane equation gives signed distance
xx, yy, zz = np.meshgrid(np.arange(data.shape[0]),
                         np.arange(data.shape[1]),
                         np.arange(data.shape[2]), indexing='ij')

signed_distance = A*xx + B*yy + C*zz + D

# Left = negative side, Right = positive side
left_hemisphere = np.where(signed_distance <= 0, data, 0)
right_hemisphere = np.where(signed_distance > 0, data, 0)

# --- Save hemispheres ---
nib.save(nib.Nifti1Image(left_hemisphere, img.affine), 'left_hemisphere.nii.gz')
nib.save(nib.Nifti1Image(right_hemisphere, img.affine), 'right_hemisphere.nii.gz')

print('Left and right hemispheres saved!')
