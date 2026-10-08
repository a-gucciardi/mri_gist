import os
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from mri_gist.utils.logging import setup_logger

# FastAPI app server
# supports nifti and nrrd 

logger = logging.getLogger("rich")

app = FastAPI(title="MRI-GIST Visualization Server")

# Models
class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    type: str

class ProcessingRequest(BaseModel):
    input_file: str
    output_file: str
    params: dict = {}

# Global state (simplified )
DATA_DIR = Path.cwd() # Default to current directory, can be configured

@app.on_event("startup")
async def startup_event():
    setup_logger()
    logger.info("Starting MRI-GIST Server...")

# API Endpoints
@app.get("/api/files", response_model=List[FileInfo])
async def list_files(directory: Optional[str] = None):
    """List MRI files in the specified directory or default data directory"""
    target_dir = Path(directory) if directory else DATA_DIR

    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Directory not found")

    files = []
    extensions = {'.nii', '.nii.gz', '.nrrd'}

    for item in target_dir.rglob("*"):
        if item.is_file() and "".join(item.suffixes) in extensions:
            files.append(FileInfo(
                name=item.name,
                path=str(item.absolute()),
                size=item.stat().st_size,
                type="".join(item.suffixes)
            ))
    return files

@app.post("/api/process/segment")
async def trigger_segmentation(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Trigger segmentation task"""
    from mri_gist.segmentation.synthseg import run_synthseg

    input_path = Path(request.input_file)
    output_path = Path(request.output_file)

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Input file not found")

    # Run in background
    background_tasks.add_task(
        run_synthseg, 
        input_path=str(input_path), 
        output_path=str(output_path),
        robust=request.params.get('robust', True),
        parcellation=request.params.get('parcellation', False)
    )

    return {"status": "started", "job_type": "segmentation", "input": request.input_file}



# Mount static files, must be after API routes to avoid masking them
# Mount static files, must be after API routes to avoid masking them
# serve the built frontend
# Assuming server.py is in mri_gist/visualization and build is in mri_gist/visualization/frontend/dist
static_dir = Path(__file__).parent / "frontend" / "dist"

if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="root")
else:
    logger.warning(f"Frontend build not found at {static_dir}. Please run 'npm run build' in frontend directory.")
    # Fallback to old static if needed, or just warn
    original_static = Path(__file__).parent / "static"
    if original_static.exists():
         app.mount("/", StaticFiles(directory=str(original_static), html=True), name="root_legacy")

def start_server(host="localhost", port=8080, data_dir=None):
    """Launch the server programmatically"""
    global DATA_DIR
    if data_dir:
        DATA_DIR = Path(data_dir)

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
