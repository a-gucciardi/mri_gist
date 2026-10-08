The Analytics Module (
mri_gist/backend/analytics.py
) currently provides the following statistical analyses on the MRI data:

1. Basic Volume Statistics (basic_stats)
Used to get a general overview of the MRI scan's properties.

Intensity Stats: Calculates the mean, standard deviation, min, max, and median intensity values across all voxels in the image.
Volume Calculation:
Total Voxels: Counts the total number of data points.
Voxel Volume: Calculates the physical volume of a single voxel (in mm³) using the image's affine matrix (det(affine)).
Total Brain Volume: Estimates the volume in milliliters (mL) by multiplying voxel count by voxel volume.
2. Tissue Distribution (
tissue_distribution
)
Used to estimate the amount of "tissue" versus "background" in the image.

Automatic Thresholding: Uses Otsu's Method to automatically find the optimal intensity threshold that separates the brain tissue from the dark background.
Classification: every voxel is classified as either background or 
tissue
 based on this threshold.
Metrics: Returns the voxel count, percentage, mean intensity, and standard deviation for both the background and tissue classes.
3. Regional Analysis (
regional
)
Currently a placeholder.

It is designed to eventually support region-specific analysis (e.g., Left vs. Right Hemisphere).
Currently, it returns the whole_brain stats (which runs basic_stats) and marks specific regions as "not_implemented".
How the UI uses this
The Analytics Tab in the frontend calls the 
tissue_distribution
 endpoint. It displays a bar chart comparing the Voxel Count of the Background vs. the Tissue, giving you a visual representation of how much of the volume contains actual data.