# MRI-GIST Backend Service

The MRI-GIST Backend Service provides a RESTful API for MRI data processing, analytics, and model serving. It complements the existing CLI and Web UI components by offering programmatic access to MRI-GIST functionality.

## Features

### 1. Processing API
- **Format Conversion**: Convert between NIfTI, NRRD, and other formats
- **Registration**: Align MRI scans using rigid, affine, or SyN transformations
- **Segmentation**: Tissue segmentation using SynthSeg
- **Hemisphere Separation**: Split brain into left and right hemispheres

### 2. Analytics API
- **Basic Statistics**: Mean, std, min, max, volume calculations
- **Tissue Distribution**: Background vs tissue classification
- **Regional Analysis**: Whole brain and hemisphere statistics
- **Custom Analytics**: Extensible for additional analysis types

### 3. Model Serving API (Placeholder)
- **Prediction Endpoints**: Ready for ML model integration
- **Custom Models**: Support for different model types
- **Batch Processing**: Handle multiple predictions efficiently

### 4. File Management
- **File Upload**: Secure file upload with storage management
- **Job Tracking**: Monitor processing jobs with status updates
- **Result Retrieval**: Access processed data and analytics results

## Architecture

```
mri_gist/
└── backend/
    ├── __init__.py          # Module initialization
    ├── server.py           # FastAPI application and endpoints
    ├── analytics.py        # Analytics algorithms and calculations
    └── README.md           # This file
```

## API Endpoints

### Base URL
`http://localhost:8000` (default)

### Processing Endpoints

**POST /api/process**
- Submit MRI processing job
- Request body: `ProcessingRequest`
- Response: `ProcessingResponse` with job ID

**GET /api/process/{job_id}**
- Get processing job status
- Response: `ProcessingResponse` with current status

### Analytics Endpoints

**POST /api/analytics**
- Submit analytics job
- Request body: `AnalyticsRequest`
- Response: `AnalyticsResponse` with job ID

**GET /api/analytics/{job_id}**
- Get analytics results
- Response: `AnalyticsResponse` with results

### Model Serving Endpoints

**POST /api/predict**
- Submit prediction job
- Request body: `ModelPredictionRequest`
- Response: `ModelPredictionResponse` with job ID

**GET /api/predict/{job_id}**
- Get prediction results
- Response: `ModelPredictionResponse` with predictions

### File Management Endpoints

**POST /api/upload**
- Upload MRI file
- Form data: `file` field
- Response: JSON with file information

**GET /api/health**
- Health check endpoint
- Response: Server status and job statistics

## Usage

### Starting the Backend Server

```bash
# Using CLI
python -m mri_gist.cli backend --port 8000 --host localhost

# Direct module execution
python -m mri_gist.backend.server

# With custom data directory
python -m mri_gist.cli backend --port 8000 --data-dir /path/to/data
```

### API Documentation

Once the server is running, access the interactive API documentation:
- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`

### Example Requests

**Format Conversion**
```bash
curl -X POST "http://localhost:8000/api/process" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file": "/path/to/input.nii.gz",
    "output_file": "/path/to/output.nrrd",
    "task_type": "conversion",
    "params": {"format": "nrrd", "clean": true}
  }'
```

**Basic Statistics**
```bash
curl -X POST "http://localhost:8000/api/analytics" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file": "/path/to/input.nii.gz",
    "analysis_type": "basic_stats",
    "params": {}
  }'
```

**File Upload**
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@/path/to/brain_scan.nii.gz"
```

## Analytics Module

The `analytics.py` module provides comprehensive MRI data analysis:

### Available Analysis Types

1. **basic_stats**: Basic statistical measures
   - Mean, std, min, max, median
   - Volume calculations
   - Data shape and type information

2. **tissue_distribution**: Tissue classification
   - Otsu thresholding for segmentation
   - Background vs tissue percentages
   - Tissue intensity statistics

3. **regional**: Regional analysis (placeholder)
   - Whole brain statistics
   - Hemisphere analysis (future)

4. **comprehensive**: All analyses combined

### Example Analytics Usage

```python
from mri_gist.backend.analytics import run_analytics_analysis

# Basic statistics
results = run_analytics_analysis(
    input_file="/path/to/brain.nii.gz",
    analysis_type="basic_stats"
)

# Tissue distribution with custom threshold
results = run_analytics_analysis(
    input_file="/path/to/brain.nii.gz",
    analysis_type="tissue_distribution",
    params={"threshold": 150}
)

# Comprehensive analysis
results = run_analytics_analysis(
    input_file="/path/to/brain.nii.gz",
    analysis_type="comprehensive"
)
```

## Job Management

The backend uses a simple job registry to track processing tasks:

- **Job Status**: queued → processing → completed/failed
- **Job Types**: processing, analytics, prediction
- **Result Storage**: Results stored in job registry
- **Cleanup**: Jobs remain in registry until server restart

## Error Handling

The API provides comprehensive error handling:

- **404 Not Found**: Job not found
- **400 Bad Request**: Invalid request parameters
- **500 Internal Error**: Processing failures
- **Detailed Error Messages**: Included in response bodies

## Future Enhancements

1. **Persistent Job Storage**: Database backend for job tracking
2. **Authentication**: Secure API access
3. **Rate Limiting**: Prevent abuse
4. **Model Integration**: Actual ML model serving
5. **Batch Processing**: Multiple file processing
6. **Webhooks**: Job completion notifications
7. **Advanced Analytics**: More sophisticated analysis algorithms

## Development

### Testing

Run the test script to verify backend functionality:

```bash
python test_backend.py
```

### Examples

See `examples/backend_example.py` for comprehensive usage examples.

### Dependencies

The backend service requires the same dependencies as the main MRI-GIST package, plus:

- `requests` for API testing (optional)
- Standard Python libraries

## Integration

The backend service integrates seamlessly with other MRI-GIST components:

- **CLI**: Can be started via `mri-gist backend` command
- **Web UI**: Can call backend API for processing
- **Processing Modules**: Reuses existing MRI-GIST functionality
- **Analytics**: Extends core capabilities with statistical analysis

## Performance Considerations

- **Background Processing**: All tasks run in background threads
- **Memory Management**: Large MRI files handled efficiently
- **Concurrency**: Multiple jobs can run simultaneously
- **Resource Limits**: Consider adding limits for production use
