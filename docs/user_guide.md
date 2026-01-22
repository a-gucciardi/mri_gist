# MRI-GIST User Guide

This guide provides detailed instructions for using the MRI-GIST toolkit.

## 1. Registration

The registration module aligns a moving image (e.g., a subject's scan) to a fixed image (e.g., a template).

**Command:** `mri-gist register`

**Options:**
-   `--method`: Transformation type.
    -   `rigid`: Translation and rotation (6 degrees of freedom).
    -   `affine`: Rigid + scaling and shearing (12 degrees of freedom).
    -   `syn`: Symmetric Normalization (non-linear, deformable).
-   `--threads`: Number of CPU threads to use (default: 4).

**Example:**
```bash
mri-gist register subject_T1.nii.gz template_T1.nii.gz -o subject_registered.nii.gz --method syn
```

## 2. Segmentation

The segmentation module uses `mri_synthseg` (from FreeSurfer) to segment brain tissues. It is robust to contrast variations and resolution.

**Command:** `mri-gist segment`

**Options:**
-   `--robust`: Enable robust mode for lower quality scans (recommended).
-   `--parc`: Generate parcellation (cortical regions) in addition to segmentation.
-   `--qc`: Path to save a CSV file with quality control metrics.

**Example:**
```bash
mri-gist segment subject.nii.gz -o seg.nii.gz --robust --qc qc_metrics.csv
```

## 3. Hemisphere Separation

This module separates the brain into left and right hemispheres. It calculates a midway registration to a symmetric space to ensure an unbiased split.

**Command:** `mri-gist separate`

**Options:**
-   `--method`:
    -   `antspy`: Uses ANTsPy for registration (Python-native).
    -   `flirt`: Uses FSL FLIRT (requires FSL installation).

**Example:**
```bash
mri-gist separate subject.nii.gz -l left_hemi.nii.gz -r right_hemi.nii.gz --method antspy
```

## 4. Web Visualization

The Web UI allows for interactive 3D exploration and simple processing triggers.

**Command:** `mri-gist serve`

**Features:**
-   **3D Volume Rendering**: View NIfTI and NRRD files in 3D.
-   **Multi-planar Reconstruction (MPR)**: Axial, Coronal, and Sagittal slice views.
-   **File Browser**: Select files from the data directory.
-   **Processing**: Trigger segmentation tasks directly from the UI.
-   **Statistics**: View tissue volume distributions.

**Usage:**
1.  Run `mri-gist serve`.
2.  Navigate to `http://localhost:8080`.
3.  Use the "File Selection" menu to load a volume.
4.  Use "3Dmodel" and "2D Planes visibility" folders to adjust visualization.
