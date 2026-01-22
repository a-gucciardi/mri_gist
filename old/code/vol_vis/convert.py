import SimpleITK as sitk

# Load the NIfTI image
nifti_image = sitk.ReadImage('nifti/right.nii.gz')  

# Save as NRRD
sitk.WriteImage(nifti_image, 'nrrd/right.nrrd')  # replace with your desired output file name
