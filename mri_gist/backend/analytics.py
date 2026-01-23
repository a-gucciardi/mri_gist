"""
MRI-GIST Backend Analytics Module

Provides statistical analysis and metrics for MRI data.
"""

import logging
import numpy as np
import nibabel as nib
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("rich")

class MRIAnalytics:
    """Class for performing MRI data analytics"""
    
    def __init__(self, input_file: str):
        """Initialize analytics with MRI file"""
        self.input_file = Path(input_file)
        self.data = None
        self.affine = None
        self.header = None
        self._load_data()
    
    def _load_data(self):
        """Load MRI data from file"""
        try:
            img = nib.load(str(self.input_file))
            self.data = img.get_fdata()
            self.affine = img.affine
            self.header = img.header
            logger.info(f"Loaded MRI data: {self.input_file}")
        except Exception as e:
            logger.error(f"Failed to load MRI data: {e}")
            raise
    
    def basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics for MRI volume"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Calculate statistics
        mean_val = np.mean(self.data)
        std_val = np.std(self.data)
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        median_val = np.median(self.data)
        
        # Calculate volume statistics
        voxel_volume = np.abs(np.linalg.det(self.affine[:3, :3]))
        total_voxels = self.data.size
        brain_volume_ml = total_voxels * voxel_volume / 1000  # Convert to mL
        
        return {
            "basic_stats": {
                "mean": float(mean_val),
                "std": float(std_val),
                "min": float(min_val),
                "max": float(max_val),
                "median": float(median_val)
            },
            "volume_stats": {
                "voxel_count": int(total_voxels),
                "voxel_volume_mm3": float(voxel_volume),
                "estimated_brain_volume_ml": float(brain_volume_ml)
            },
            "data_shape": list(self.data.shape),
            "data_type": str(self.data.dtype)
        }
    
    def tissue_distribution(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Estimate tissue distribution based on intensity thresholds"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Simple tissue classification based on intensity
        # This is a basic approach - real applications would use segmentation
        if threshold is None:
            # Use Otsu's method for automatic thresholding
            threshold = self._calculate_otsu_threshold()
        
        # Classify voxels
        background_mask = self.data < threshold
        tissue_mask = self.data >= threshold
        
        background_voxels = np.sum(background_mask)
        tissue_voxels = np.sum(tissue_mask)
        total_voxels = self.data.size
        
        # Calculate tissue statistics
        tissue_mean = float(np.mean(self.data[tissue_mask])) if tissue_voxels > 0 else 0
        tissue_std = float(np.std(self.data[tissue_mask])) if tissue_voxels > 0 else 0
        
        return {
            "threshold": float(threshold),
            "background": {
                "voxel_count": int(background_voxels),
                "percentage": float(background_voxels / total_voxels * 100)
            },
            "tissue": {
                "voxel_count": int(tissue_voxels),
                "percentage": float(tissue_voxels / total_voxels * 100),
                "mean_intensity": tissue_mean,
                "std_intensity": tissue_std
            }
        }
    
    def _calculate_otsu_threshold(self) -> float:
        """Calculate Otsu threshold for binary segmentation"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Flatten data and remove zeros for better thresholding
        flat_data = self.data.flatten()
        flat_data = flat_data[flat_data > 0]  # Remove background zeros
        
        if len(flat_data) == 0:
            return 0
        
        # Calculate histogram
        hist, bin_edges = np.histogram(flat_data, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Otsu's method
        total_weight = np.sum(hist)
        if total_weight == 0:
            return 0
        
        max_variance = 0
        best_threshold = 0
        
        for t in range(1, len(hist)):
            # Weight background
            w0 = np.sum(hist[:t]) / total_weight
            # Weight foreground
            w1 = np.sum(hist[t:]) / total_weight
            
            # Mean background
            if np.sum(hist[:t]) > 0:
                mu0 = np.sum(hist[:t] * bin_centers[:t]) / np.sum(hist[:t])
            else:
                mu0 = 0
            
            # Mean foreground
            if np.sum(hist[t:]) > 0:
                mu1 = np.sum(hist[t:] * bin_centers[t:]) / np.sum(hist[t:])
            else:
                mu1 = 0
            
            # Between-class variance
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                best_threshold = bin_centers[t]
        
        return float(best_threshold)
    
    def regional_analysis(self, regions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform regional analysis (placeholder for future implementation)"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # This would be enhanced with actual region segmentation
        return {
            "message": "Regional analysis placeholder",
            "regions": {
                "whole_brain": self.basic_statistics(),
                "left_hemisphere": {"status": "not_implemented"},
                "right_hemisphere": {"status": "not_implemented"}
            }
        }

def run_analytics_analysis(
    input_file: str,
    analysis_type: str,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run analytics analysis based on type
    
    Args:
        input_file: Path to MRI file
        analysis_type: Type of analysis to perform
        params: Additional parameters
        
    Returns:
        Dictionary with analysis results
    """
    if params is None:
        params = {}
    
    try:
        analytics = MRIAnalytics(input_file)
        
        if analysis_type == "basic_stats":
            return analytics.basic_statistics()
            
        elif analysis_type == "tissue_distribution":
            threshold = params.get("threshold")
            return analytics.tissue_distribution(threshold)
            
        elif analysis_type == "regional":
            return analytics.regional_analysis()
            
        elif analysis_type == "comprehensive":
            return {
                "basic_stats": analytics.basic_statistics(),
                "tissue_distribution": analytics.tissue_distribution(),
                "regional_analysis": analytics.regional_analysis()
            }
        
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        return {"error": str(e), "status": "failed"}
