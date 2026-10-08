# MRI-GIST Visualization Frontend

A React-based web interface for 3D MRI visualization and analysis, built with Vite.

## Features

- **3D Volume Rendering**: High-performance rendering using Three.js with support for `.nrrd`, `.nii`, and `.nii.gz` formats.
- **Orthographic Views**: Axial, Sagittal, and Coronal slice views.
- **Tabbed Interface**:
    - **Viewer**: Main visualization workspace with controls (Zoom, Rotate, Threshold, Colormap).
    - **Analytics**: Real-time statistical analysis of MRI data (Tissue Distribution, Volume Stats).
- **Backend Integration**:
    - Fully integrated with `mri_gist.backend`.
    - Supports Asynchronous Job Processing for segmentation and analytics.
    - Dynamic file listing from server data directories.

## Architecture

- **React 19**: Modern UI library.
- **Vite**: Fast build tool and dev server.
- **Three.js**: WebGL-based 3D graphics.
- **Chart.js**: Data visualization for analytics.
- **nifti-reader-js**: Client-side NIfTI parsing.

## Development

1. **Start the Unified Backend** (serves frontend at root):
   ```bash
   uv run mri_gist/backend/server.py
   ```
   Access at `http://localhost:8000`.

2. **Standalone Frontend Development** (optional):
   ```bash
   cd mri_gist/visualization/frontend
   npm install
   npm run dev
   ```
   Note: Some API features require the running backend.
