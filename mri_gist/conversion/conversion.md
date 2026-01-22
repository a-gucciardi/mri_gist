# Converter Module

The converter module is made to be robust and feature-rich, addressing both orientation and artifact issues.

## Implementation Details

### 1. Visualization
- **RAI Enforcment**: Resolves the flipped images by ensuring all outputs are in the Right-Anterior-Inferior coordinate system (`sitk.DICOMOrient(image, 'RAI')`).
- **Compression**: Enabled Gzip compression on output files.

### 2. Conversion (`--clean`)
To address the issue where background noise ( values above 0 will render the nrrd) becomes visible as a "squarish box":
- Implemented **Otsu Threshold Masking**.
- When `--clean` is used, the system automatically detects the background/foreground threshold and sets all background pixels to **exactly 0**.

```bash
# Clean conversion command
mri-gist convert 02_tests/test_images -o 02_tests/test_out --format nrrd --clean
```

### 3. Verification tets
Run the verification script to confirm:
```bash
uv run 02_tests/verify_conversion.py
```
Output should be :
```
--- Testing Cleaning Option ---
INFO:mri_gist.conversion.formats:Applying background cleaning (Otsu thresholding)
  Cleaned Background Max: 0
✓ Cleaning successful (Background is 0)
```
