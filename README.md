# MRI Gist

Comprehensive, open-source MRI processing and visualization toolkit. 
 - Unified CLI and Web UI for brain MRI registration, segmentation, and anomaly detection. 
 - Bridging the gap between advanced processing (ANTs, FreeSurfer) and accessible visualization (Three.js).

## Tasks

- [x] Create project structure and choose functionalities
- [x] Gap analysis for SoftwareX publication requirements:
- [x] Step-by-step roadmap for CLI and UI development

## Phase 1: CLI Foundation & Architecture
- [x] Set up project structure and dependency management (pyproject.toml)
- [x] Implement unified CLI entry point (cli.py)
- [x] Refactor Registration Module
- [x] Refactor Segmentation Module
     - [ ] todo : port synthseg
- [x] Refactor Detection Module
     - [ ] todo : validate ants, flirt ports
- [x] Refactor Conversion Module
     - [x] tests : Nifti to nrrd
     - [ ] todo : other formats

## Phase 2: Web UI Development
- [x] Design and reimplement Web UI
- [x] Implement volvis brain MRI viewer
- [x] Implement file upload and processing, backend integration
- [x] Add visualization and statistics
     - [ ] todo : validate statistics and analysis     
     - [ ] todo : validate segmentation

## Phase 3: Publication Requirements
- [x] Documentation
- [x] Code quality
- [ ] Validation
___

Usage :
```bash
uv run mri_gist <command> <options>
```

For individual command details :
```bash
uv run mri_gist <command> --help
```

Web UI :
```bash
uv run mri_gist serve <anat_path>
```
-> looks into anat_sample by default

Commands :
    - [ ] convert : Convert between medical imaging formats (nii, nrrd, vtk, stl, obj)
     - [ ] fractal : Compute whole-brain and per-label fractal dimensions from segmentation volumes
    - [ ] register : Register an MRI to a template
    - [ ] separate: Separate brain into left and right hemispheres
    - [ ] segment : Segment an MRI 
    - [ ] serve : Launch the web UI

Fractal examples:
```bash
uv run mri_gist fractal compute path/to/seg.nii.gz -o results.csv
uv run mri_gist fractal compute path/to/seg.nii.gz -o results.csv --per-label
uv run mri_gist fractal masks path/to/seg.nii.gz -o mask_dir
uv run mri_gist fractal batch "data/**/*seg.nii.gz" -o batch_results.csv --per-label
```
