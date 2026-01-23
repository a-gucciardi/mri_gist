"""
MRI-GIST Backend API Server

FastAPI-based backend service for MRI processing, analytics, and model serving.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import uuid

from mri_gist.utils.logging import setup_logger
from mri_gist.backend.analytics import run_analytics_analysis

logger = logging.getLogger("rich")

app = FastAPI(
    title="MRI-GIST Backend API",
    description="API for MRI processing, analytics, and model serving",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Models
class ProcessingRequest(BaseModel):
    """Request model for processing tasks"""
    input_file: str
    output_file: str
    params: Dict[str, Any] = {}
    task_type: str  # 'registration', 'segmentation', 'conversion', etc.

class ProcessingResponse(BaseModel):
    """Response model for processing tasks"""
    job_id: str
    status: str
    task_type: str
    input_file: str
    output_file: str
    timestamp: str
    message: str = ""

class AnalyticsRequest(BaseModel):
    """Request model for analytics tasks"""
    input_file: str
    analysis_type: str  # 'volume_stats', 'tissue_distribution', etc.
    params: Dict[str, Any] = {}

class AnalyticsResponse(BaseModel):
    """Response model for analytics results"""
    job_id: str
    status: str
    analysis_type: str
    results: Dict[str, Any]
    timestamp: str

class ModelPredictionRequest(BaseModel):
    """Request model for ML predictions"""
    input_file: str
    model_name: str
    params: Dict[str, Any] = {}

class ModelPredictionResponse(BaseModel):
    """Response model for ML predictions"""
    job_id: str
    status: str
    model_name: str
    predictions: Dict[str, Any]
    timestamp: str

# Global state
JOB_REGISTRY = {}
DATA_DIR = Path.cwd() / "backend_data"

@app.on_event("startup")
async def startup_event():
    """Initialize backend service"""
    setup_logger()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("MRI-GIST Backend API started")
    logger.info(f"Data directory: {DATA_DIR}")

# Processing Endpoints
@app.post("/api/process", response_model=ProcessingResponse)
async def process_mri(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
) -> ProcessingResponse:
    """
    Process MRI data (registration, segmentation, conversion, etc.)
    
    Args:
        request: ProcessingRequest containing task details
        background_tasks: FastAPI background tasks
        
    Returns:
        ProcessingResponse with job information
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Store job info
    JOB_REGISTRY[job_id] = {
        "status": "queued",
        "task_type": request.task_type,
        "input_file": request.input_file,
        "output_file": request.output_file,
        "params": request.params,
        "timestamp": timestamp
    }
    
    # Add background task
    background_tasks.add_task(
        _execute_processing_task,
        job_id=job_id,
        task_type=request.task_type,
        input_file=request.input_file,
        output_file=request.output_file,
        params=request.params
    )
    
    return ProcessingResponse(
        job_id=job_id,
        status="queued",
        task_type=request.task_type,
        input_file=request.input_file,
        output_file=request.output_file,
        timestamp=timestamp,
        message="Processing task queued successfully"
    )

@app.get("/api/process/{job_id}", response_model=ProcessingResponse)
async def get_processing_status(job_id: str) -> ProcessingResponse:
    """
    Get status of a processing job
    
    Args:
        job_id: Job identifier
        
    Returns:
        ProcessingResponse with current status
    """
    if job_id not in JOB_REGISTRY:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = JOB_REGISTRY[job_id]
    return ProcessingResponse(
        job_id=job_id,
        status=job_data["status"],
        task_type=job_data["task_type"],
        input_file=job_data["input_file"],
        output_file=job_data["output_file"],
        timestamp=job_data["timestamp"],
        message=job_data.get("message", "")
    )

# Analytics Endpoints
@app.post("/api/analytics", response_model=AnalyticsResponse)
async def run_analytics(
    request: AnalyticsRequest,
    background_tasks: BackgroundTasks
) -> AnalyticsResponse:
    """
    Run analytics on MRI data
    
    Args:
        request: AnalyticsRequest containing analysis details
        background_tasks: FastAPI background tasks
        
    Returns:
        AnalyticsResponse with job information
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Store job info
    JOB_REGISTRY[job_id] = {
        "status": "queued",
        "analysis_type": request.analysis_type,
        "input_file": request.input_file,
        "params": request.params,
        "timestamp": timestamp,
        "results": {}
    }
    
    # Add background task
    background_tasks.add_task(
        _execute_analytics_task,
        job_id=job_id,
        analysis_type=request.analysis_type,
        input_file=request.input_file,
        params=request.params
    )
    
    return AnalyticsResponse(
        job_id=job_id,
        status="queued",
        analysis_type=request.analysis_type,
        results={},
        timestamp=timestamp
    )

@app.get("/api/analytics/{job_id}", response_model=AnalyticsResponse)
async def get_analytics_results(job_id: str) -> AnalyticsResponse:
    """
    Get results of an analytics job
    
    Args:
        job_id: Job identifier
        
    Returns:
        AnalyticsResponse with results
    """
    if job_id not in JOB_REGISTRY:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = JOB_REGISTRY[job_id]
    
    if "analysis_type" not in job_data:
        raise HTTPException(status_code=400, detail="Not an analytics job")
    
    return AnalyticsResponse(
        job_id=job_id,
        status=job_data["status"],
        analysis_type=job_data["analysis_type"],
        results=job_data.get("results", {}),
        timestamp=job_data["timestamp"]
    )

# Model Serving Endpoints
@app.post("/api/predict", response_model=ModelPredictionResponse)
async def run_prediction(
    request: ModelPredictionRequest,
    background_tasks: BackgroundTasks
) -> ModelPredictionResponse:
    """
    Run ML model prediction on MRI data
    
    Args:
        request: ModelPredictionRequest containing prediction details
        background_tasks: FastAPI background tasks
        
    Returns:
        ModelPredictionResponse with job information
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Store job info
    JOB_REGISTRY[job_id] = {
        "status": "queued",
        "model_name": request.model_name,
        "input_file": request.input_file,
        "params": request.params,
        "timestamp": timestamp,
        "predictions": {}
    }
    
    # Add background task
    background_tasks.add_task(
        _execute_prediction_task,
        job_id=job_id,
        model_name=request.model_name,
        input_file=request.input_file,
        params=request.params
    )
    
    return ModelPredictionResponse(
        job_id=job_id,
        status="queued",
        model_name=request.model_name,
        predictions={},
        timestamp=timestamp
    )

@app.get("/api/predict/{job_id}", response_model=ModelPredictionResponse)
async def get_prediction_results(job_id: str) -> ModelPredictionResponse:
    """
    Get results of a prediction job
    
    Args:
        job_id: Job identifier
        
    Returns:
        ModelPredictionResponse with prediction results
    """
    if job_id not in JOB_REGISTRY:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = JOB_REGISTRY[job_id]
    
    if "model_name" not in job_data:
        raise HTTPException(status_code=400, detail="Not a prediction job")
    
    return ModelPredictionResponse(
        job_id=job_id,
        status=job_data["status"],
        model_name=job_data["model_name"],
        predictions=job_data.get("predictions", {}),
        timestamp=job_data["timestamp"]
    )

# File Upload Endpoint
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload MRI file to backend storage
    
    Args:
        file: UploadFile containing MRI data
        
    Returns:
        JSON response with file information
    """
    try:
        # Create upload directory
        upload_dir = DATA_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            buffer.write(await file.read())
        
        logger.info(f"File uploaded: {file_path}")
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": file.size
        })
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

# Background Task Execution
def _execute_processing_task(
    job_id: str,
    task_type: str,
    input_file: str,
    output_file: str,
    params: Dict[str, Any]
):
    """Execute processing task in background"""
    try:
        JOB_REGISTRY[job_id]["status"] = "processing"
        logger.info(f"Processing task {job_id}: {task_type}")
        
        # Import and execute appropriate module
        if task_type == "registration":
            from mri_gist.registration.core import register_image
            register_image(
                moving=input_file,
                fixed=params.get("template", input_file),  # Default to self-registration
                output=output_file,
                transform_type=params.get("method", "rigid"),
                num_threads=params.get("threads", 4)
            )
        
        elif task_type == "segmentation":
            from mri_gist.segmentation.synthseg import run_synthseg
            run_synthseg(
                input_path=input_file,
                output_path=output_file,
                robust=params.get("robust", True),
                parcellation=params.get("parcellation", False),
                qc_path=params.get("qc_path")
            )
        
        elif task_type == "conversion":
            from mri_gist.conversion.formats import convert_format
            convert_format(
                input_path=input_file,
                output_path=output_file,
                target_format=params.get("format", "nrrd"),
                clean_background=params.get("clean", False)
            )
        
        elif task_type == "separation":
            from mri_gist.detection.hemisphere import hemisphere_separation
            hemisphere_separation(
                input_path=input_file,
                left_output=params.get("left_output", output_file.replace(".nii.gz", "_left.nii.gz")),
                right_output=params.get("right_output", output_file.replace(".nii.gz", "_right.nii.gz")),
                method=params.get("method", "antspy")
            )
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        JOB_REGISTRY[job_id]["status"] = "completed"
        JOB_REGISTRY[job_id]["message"] = f"{task_type} completed successfully"
        logger.info(f"Task {job_id} completed: {task_type}")
        
    except Exception as e:
        JOB_REGISTRY[job_id]["status"] = "failed"
        JOB_REGISTRY[job_id]["message"] = str(e)
        logger.error(f"Task {job_id} failed: {e}")

# Placeholder functions for analytics and prediction
def _execute_analytics_task(
    job_id: str,
    analysis_type: str,
    input_file: str,
    params: Dict[str, Any]
):
    """Execute analytics task in background"""
    try:
        JOB_REGISTRY[job_id]["status"] = "processing"
        logger.info(f"Analytics task {job_id}: {analysis_type}")
        
        # Run actual analytics using the analytics module
        results = run_analytics_analysis(
            input_file=input_file,
            analysis_type=analysis_type,
            params=params
        )
        
        JOB_REGISTRY[job_id]["status"] = "completed"
        JOB_REGISTRY[job_id]["results"] = results
        logger.info(f"Analytics task {job_id} completed")
        
    except Exception as e:
        JOB_REGISTRY[job_id]["status"] = "failed"
        JOB_REGISTRY[job_id]["results"] = {"error": str(e)}
        logger.error(f"Analytics task {job_id} failed: {e}")

def _execute_prediction_task(
    job_id: str,
    model_name: str,
    input_file: str,
    params: Dict[str, Any]
):
    """Execute prediction task in background"""
    try:
        JOB_REGISTRY[job_id]["status"] = "processing"
        logger.info(f"Prediction task {job_id}: {model_name}")
        
        # Placeholder: Implement actual model prediction
        predictions = {
            "model_name": model_name,
            "input_file": input_file,
            "predictions": {"class": "normal", "confidence": 0.95},
            "message": "Prediction placeholder - implement real model"
        }
        
        JOB_REGISTRY[job_id]["status"] = "completed"
        JOB_REGISTRY[job_id]["predictions"] = predictions
        logger.info(f"Prediction task {job_id} completed")
        
    except Exception as e:
        JOB_REGISTRY[job_id]["status"] = "failed"
        JOB_REGISTRY[job_id]["predictions"] = {"error": str(e)}
        logger.error(f"Prediction task {job_id} failed: {e}")

# Health Check Endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in JOB_REGISTRY.values() if j["status"] in ["queued", "processing"]]),
        "completed_jobs": len([j for j in JOB_REGISTRY.values() if j["status"] == "completed"])
    }

def start_backend_server(
    host: str = "localhost",
    port: int = 8000,
    data_dir: Optional[str] = None
):
    """Launch the backend API server programmatically"""
    global DATA_DIR
    if data_dir:
        DATA_DIR = Path(data_dir)
    
    logger.info(f"Starting MRI-GIST Backend API at http://{host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/api/docs")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_backend_server()
