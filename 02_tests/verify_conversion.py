import os
import shutil
import SimpleITK as sitk
from pathlib import Path
from mri_gist.conversion.formats import convert_format
import logging

logging.basicConfig(level=logging.INFO)

def create_dummy_image(filename):
    """Create a small dummy SimpleITK image (3D)."""
    # Create 3D image (3 slices of 2x2)
    image = sitk.GetImageFromArray([[[0, 100], [200, 255]]] * 3)
    sitk.WriteImage(image, str(filename))
    print(f"Created dummy image: {filename}")

def test_conversion():
    test_dir = Path("02_tests/test_conversion_sandbox")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    input_dir = test_dir / "input"
    input_dir.mkdir()
    output_dir = test_dir / "output"
    
    # 1. Test File -> File
    print("\n--- Testing Single File Conversion ---")
    f1 = input_dir / "test1.nii.gz"
    create_dummy_image(f1)
    
    out1 = test_dir / "out1.nrrd"
    convert_format(str(f1), str(out1), "nrrd")
    
    if out1.exists():
        print("✓ Single file conversion successful")
    else:
        print("✗ Single file conversion failed")

    # 2. Test Batch Dir -> Dir
    print("\n--- Testing Batch Directory Conversion ---")
    f2 = input_dir / "test2.nii"
    f3 = input_dir / "test3.mha"
    create_dummy_image(f2)
    create_dummy_image(f3)
    
    convert_format(str(input_dir), str(output_dir), "nrrd")
    
    expected_out2 = output_dir / "test2.nrrd"
    expected_out3 = output_dir / "test3.nrrd"
    
    if expected_out2.exists() and expected_out3.exists():
        print("✓ Batch folder conversion successful")
        print(f"  Found: {list(output_dir.glob('*'))}")
    else:
        print("✗ Batch folder conversion failed")
        print(f"  Content of output dir: {list(output_dir.glob('*'))}")

    # 3. Test Cleaning (Otsu)
    print("\n--- Testing Cleaning Option ---")
    f4 = input_dir / "noisy.nii"
    # Create image with background noise (10) and object (200)
    # Background: 10, Object: 200
    # Create image with background noise (10) and object (200)
    # Be careful with list multiplication of mutable objects!
    slice_bg = [[10, 10], [10, 10]]
    slice_obj = [[10, 10], [10, 200]] # Object 200 at 1,1
    noisy_arr = [slice_bg, slice_obj, slice_bg] # Middle slice has object
    
    image = sitk.GetImageFromArray(noisy_arr)
    sitk.WriteImage(image, str(f4))
    
    out4 = output_dir / "clean.nrrd"
    convert_format(str(f4), str(out4), "nrrd", clean_background=True)
    
    if out4.exists():
        # Verify background is 0
        res = sitk.ReadImage(str(out4))
        arr = sitk.GetArrayFromImage(res)
        bg_max = arr[0].max() # Slice 0 is all background
        print(f"  Cleaned Background Max: {bg_max}")
        if bg_max == 0:
            print("✓ Cleaning successful (Background is 0)")
        else:
            print(f"✗ Cleaning failed. Background max is {bg_max}")
    else:
        print("✗ Output not found")

    # Cleanup
    # shutil.rmtree(test_dir)

if __name__ == "__main__":
    try:
        test_conversion()
    except Exception as e:
        print(f"Test failed with error: {e}")
