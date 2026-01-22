import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import nibabel as nib
from scipy.ndimage import zoom

# nifti to py
input_path = "001/mri/wm.nii.gz"
image = nib.load(input_path)
image_data = image.get_fdata()


scale_factor = 0.05  # This will reduce the size to x% in each dimension
downsampled_data = zoom(image_data, scale_factor)

# isosurface
verts, faces, normals, values = measure.marching_cubes(downsampled_data, level=0.5)

# 3D figure + surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                      triangles=faces,
                      cmap='viridis',
                      alpha=1.0)

plt.colorbar(mesh)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Thalamus Isosurface (Click and drag to rotate)')

# mouse rotation
ax.mouse_init()

plt.show()