import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib

# Load the data
input_path = "001/mri/wm.nii.gz"
image = nib.load(input_path)
image_data = image.get_fdata()

# Create a figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create coordinates for non-zero voxels
x, y, z = np.where(image_data > 0)  # Get coordinates of non-zero voxels

# Plot the surface
scatter = ax.scatter(x, y, z, c=image_data[x, y, z], 
                    cmap='viridis', 
                    alpha=0.1,  # Make points semi-transparent
                    s=0.1)  # Small point size

# Add a colorbar
plt.colorbar(scatter)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Thalamus Surface Plot')

# Adjust the view
# ax.view_init(elev=20, azim=45)
ax.mouse_init()

plt.show()