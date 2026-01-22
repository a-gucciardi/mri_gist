import os
import logging
import subprocess
import numpy as np
import nibabel as nib
from scipy.linalg import sqrtm
from scipy.ndimage import affine_transform
from pathlib import Path

# TODO : needs validations, check for flirt port, or antspy light alternatives

try:
    import ants
except ImportError:
    ants = None

logger = logging.getLogger("rich")

def hemisphere_separation(
    input_path: str, 
    left_output: str, 
    right_output: str, 
    method: str = 'antspy'
) -> None:
    """
    Separate brain into left and right hemispheres using midway registration.
    
    Args:
        input_path (str): Path to input NIfTI file
        left_output (str): Path to save left hemisphere
        right_output (str): Path to save right hemisphere
        method (str): 'antspy' or 'flirt' (default: 'antspy')
    """
    input_path = Path(input_path)
    left_output = Path(left_output)
    right_output = Path(right_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    left_output.parent.mkdir(parents=True, exist_ok=True)
    right_output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Separating hemispheres for {input_path} using {method}")

    if method == 'antspy':
        _separate_antspy(input_path, left_output, right_output)
    elif method == 'flirt':
        _separate_flirt(input_path, left_output, right_output)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'antspy' or 'flirt'.")

def _separate_antspy(input_path, left_output, right_output):
    # needs validation and rechecks on the transform / script logic for T_half with ANTs parameters
    if ants is None:
        raise ImportError("ANTsPy is not installed. Please install it with `pip install antspyx`.")

    img = nib.load(str(input_path))
    data = img.get_fdata()
    flipped_data = np.flip(data, axis=0)

    original_ants_img = ants.from_numpy(data)
    flipped_ants_img = ants.from_numpy(flipped_data)

    logger.info("Running ANTs registration...")
    registration = ants.registration(fixed=flipped_ants_img, moving=original_ants_img, type_of_transform='Affine')

    # ANTsPy transforms are stored as files or lists of parameters
    # apply transform to get symmetric brain
    transformed_ants = ants.apply_transforms(original_ants_img, original_ants_img, registration['fwdtransforms'])
    transformed_data = transformed_ants.numpy()

    # split 
    mid_sagittal_index = data.shape[0] // 2
    left_data = transformed_data[:mid_sagittal_index, :, :]
    right_data = transformed_data[mid_sagittal_index:, :, :]

    # save
    left_img = nib.Nifti1Image(left_data, img.affine)

    # Adjust affine for right hemisphere
    right_affine = img.affine.copy()
    right_affine[0, 3] += data.shape[0] / 2
    right_img = nib.Nifti1Image(right_data, right_affine)

    nib.save(left_img, left_output)
    nib.save(right_img, right_output)
    logger.info(f"Saved hemispheres to {left_output} and {right_output}")

def _separate_flirt(input_path, left_output, right_output):
    # note : also needs validation and rechecks, check Note
    if not shutil.which('flirt'):
        raise RuntimeError("FSL 'flirt' command not found.")

    img = nib.load(str(input_path))
    data = img.get_fdata()
    flipped_data = np.flip(data, axis=0)

    # Temporary files
    temp_dir = input_path.parent / "temp_hemi_sep"
    temp_dir.mkdir(exist_ok=True)
    flipped_path = temp_dir / "flipped.nii.gz"
    mat_path = temp_dir / "transform.mat"

    try:
        nib.save(nib.Nifti1Image(flipped_data, img.affine), flipped_path)
        
        logger.info("Running FSL FLIRT...")
        subprocess.run([
            "flirt",
            "-in", str(input_path),
            "-ref", str(flipped_path),
            "-omat", str(mat_path),
            "-dof", "12",
            "-cost", "mutualinfo"
        ], check=True, capture_output=True)

        # Load transform and calculate midway
        T = np.loadtxt(str(mat_path))
        T_half = sqrtm(T[:3, :3])

        # Apply transform
        # Note: affine_transform uses inverse mapping by default, so we might need T_half inverse
        # usingT_half directly, assuming it works for the specific coordinate system
        transformed_data = affine_transform(data, T_half, offset=T[:3, 3])

        # Split
        mid_sagittal_index = data.shape[0] // 2
        left_data = transformed_data[:mid_sagittal_index, :, :]
        right_data = transformed_data[mid_sagittal_index:, :, :]
        
        nib.save(nib.Nifti1Image(left_data, img.affine), left_output)
        nib.save(nib.Nifti1Image(right_data, img.affine), right_output)
        logger.info(f"Saved hemispheres to {left_output} and {right_output}")
        
    finally:
        # Cleanup
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
