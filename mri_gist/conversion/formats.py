import SimpleITK as sitk
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# A Notes
# currently supported extensions to look for
extensions = ['*.nii', '*.nii.gz', '*.nrrd', '*.mha', '*.dcm', '*.ima']
# support file or dir
# tests to do
# verify cleaning

def convert_format(input_path: str, output_path: str, target_format: str, clean_background: bool = False):
    """
    Convert medical image formats using SimpleITK.

    Supports:
    - File or Directory: 1-to-1 or Batch conversion (all supported files in input dir)
    - Directory -> File: Read directory as DICOM series (single volume output)

    Args:
        input_path: Path to input file or directory
        output_path: Path to output file or directory
        target_format: Target format (nrrd, nii, nii.gz)
        clean_background: If True, apply thresholding to remove low-intensity background noise methods
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    
    # handle mesh formats limitation
    if target_format.lower() in ['stl', 'obj', 'vtk']:
        logger.error(f"Conversion to mesh format '{target_format}' is not yet supported.")
        raise NotImplementedError(f"Conversion to {target_format} requires surface extraction.")

    # heuristic: suffix -> file. If not -> directory.
    is_input_dir = input_p.is_dir()
    is_output_file = len(output_p.suffixes) > 0

    if not input_p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Case 1: Batch Conversion (Dir -> Dir)
    if is_input_dir and not is_output_file:
        logger.info(f"Batch converting directory {input_path} to {output_path}")
        output_p.mkdir(parents=True, exist_ok=True)

        # Supported extensions to look for
        extensions = ['*.nii', '*.nii.gz', '*.nrrd', '*.mha', '*.dcm']#, '*.ima']
        files_to_convert = []
        for ext in extensions:
            files_to_convert.extend(input_p.glob(ext))

        if not files_to_convert:
            logger.warning(f"No matching image files found in {input_path}")
            return

        for source in files_to_convert:
            # Construct output filename
            stem = source.name
            # Remove all known extensions
            for ext in ['.nii.gz', '.nii', '.nrrd', '.mha', '.dcm']:
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
                    break

            if not target_format.startswith('.'): # If target_format doesn't start with '.'
                target_ext = f".{target_format}"
            else:
                target_ext = target_format

            dest = output_p / f"{stem}{target_ext}"

            try:
                _convert_single_file(source, dest, clean_background)
            except Exception as e:
                logger.error(f"Failed to convert {source.name}: {e}")

    # Case 2: Series Conversion (Dir -> File)
    elif is_input_dir and is_output_file:
         _convert_dicom_series(input_p, output_p, clean_background)

    # Case 3: Single File Conversion
    else:
        # If output is a directory, construct filename
        if output_p.is_dir() or (not is_output_file and not output_p.exists()):
            # Treat output as directory
            output_p.mkdir(parents=True, exist_ok=True)
            stem = input_p.name
            for ext in ['.nii.gz', '.nii', '.nrrd', '.mha', '.dcm']:
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
                    break
            if not target_format.startswith('.'):
                target_ext = f".{target_format}"
            else:
                target_ext = target_format
            final_output = output_p / f"{stem}{target_ext}"
        else:
            final_output = output_p

        _convert_single_file(input_p, final_output, clean_background)

def _convert_single_file(input_path: Path, output_path: Path, clean: bool = False):
    """Helper to convert a single image file."""
    logger.info(f"Converting: {input_path} -> {output_path}")
    image = sitk.ReadImage(str(input_path))

    # Enforce RAI orientation (mimic c3d -orient RAI)
    try:
        image = sitk.DICOMOrient(image, 'RAI') # requires SimpleITK 2.0+
    except Exception as e:
        logger.warning(f"Could not reorient image to RAI: {e}")

    if clean:
        logger.info(f"Applying background cleaning (Otsu thresholding)")
        masked, _ = _apply_cleaning(image)
        image = masked

    # Ensure parent dir exists and enable compression (c3d -compress)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)

def _convert_dicom_series(input_dir: Path, output_path: Path, clean: bool = False):
    """Helper to read a DICOM series from a directory and write as one volume."""
    logger.info(f"Reading DICOM series from: {input_dir}")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(input_dir))

    if not dicom_names:
        raise ValueError(f"No DICOM series found in {input_dir}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Enforce RAI orientation
    try:
        image = sitk.DICOMOrient(image, 'RAI')
    except Exception as e:
        logger.warning(f"Could not reorient image to RAI: {e}")

    if clean:
        logger.info(f"Applying background cleaning (Otsu thresholding)")
        masked, _ = _apply_cleaning(image)
        image = masked
    
    logger.info(f"Writing volume to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)

# Imported cleaning
def _apply_cleaning(image):
    """Apply standard Otsu thresholding to zero out background noise."""
    # Use Otsu to find the brain/object
    # Since background is low intesity, typically Otsu separates [Background] vs [Brain+Skull]
    
    # Create mask: 0 for background, 1 for foreground
    # OtsuThreshold returns an image where < Threshold is 0, >= Threshold is 1 (default settings?)
    # sitk.OtsuThreshold(image, insideValue, outsideValue, numberOfHistogramBins, maskOutput, maskValue, thresholdOutput)
    # Simplified usage:
    # We want: Mask = 1 where intensity > Threshold.
    
    # Handle float images properly? Otsu works on float but binning applies.
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)   # Foreground? No, Inside is < Threshold usually in ITK nomenclature for "Inside the threshold bounds"? 
                                    # Actually: "Pixels below the threshold are set to InsideValue, pixels above are set to OutsideValue"
                                    # We want Low values (Background) -> 0. High values -> 1.
                                    # So InsideValue = 0, OutsideValue = 1.
    otsu_filter.SetOutsideValue(1)
    
    mask = otsu_filter.Execute(image)
    
    # Apply mask
    # Cast mask to same type as image if needed, or use Mask filter
    # But simpilest is Just multiply? Or sitk.Mask(image, mask)
    
    # Ensure mask is same geometry
    # sitk.Mask expects mask to be integer type usually.
    
    cleaned_image = sitk.Mask(image, mask)
    return cleaned_image, otsu_filter.GetThreshold()
