import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
r = rng.normal(size=(100,3))
scale = 2
H, edges = np.histogramdd(r, bins=(10, 10, 10))
print(H.shape, edges[0].size, edges[1].size, edges[2].size)

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Get the coordinates of non-empty boxes
x, y, z = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1], indexing='ij')
x, y, z = x.ravel(), y.ravel(), z.ravel()

# Get the values (counts) for each box
values = H.ravel()

# Plot only non-empty boxes
mask = values > 0
ax.scatter(x[mask], y[mask], z[mask], c=values[mask], s=values[mask]*50, cmap='viridis', alpha=0.7)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Histogram Visualization')

# Add a colorbar
cbar = plt.colorbar(ax.scatter(x[mask], y[mask], z[mask], c=values[mask], cmap='viridis'))
cbar.set_label('Count')

plt.tight_layout()
plt.show()

# Print the shape information
print(f"H.shape: {H.shape}")
print(f"edges[0].size: {edges[0].size}")
print(f"edges[1].size: {edges[1].size}")
print(f"edges[2].size: {edges[2].size}")