import subprocess
import logging
import shutil
from pathlib import Path

# TODO : check for a synthseg direct port

logger = logging.getLogger("rich")

def run_synthseg(
    input_path: str, 
    output_path: str, 
    robust: bool = True, 
    parcellation: bool = False, 
    qc_path: str = None, 
    threads: int = 8
) -> None:
    """
    Run SynthSeg segmentation on an input MRI.

    Args:
        input_path (str): Path to input NIfTI file
        output_path (str): Path to save segmentation output
        robust (bool): Use robust mode (default: True)
        parcellation (bool): Generate parcellation (default: False)
        qc_path (str): Path to save QC CSV (optional)
        threads (int): Number of threads (default: 8)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Check if mri_synthseg is available
    if not shutil.which('mri_synthseg'):
        raise RuntimeError("FreeSurfer 'mri_synthseg' command not found. Please ensure FreeSurfer is installed and in your PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ['mri_synthseg', '--i', str(input_path), '--o', str(output_path)]

    if robust:
        cmd.append('--robust')

    if parcellation:
        cmd.append('--parc')

    if qc_path:
        qc_path = Path(qc_path)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(['--qc', str(qc_path)])

        # Also generate volume CSV if QC is requested, as it's often useful
        vol_path = qc_path.parent / f"{input_path.stem}_vol.csv"
        cmd.extend(['--vol', str(vol_path)])

    cmd.extend(['--threads', str(threads)])

    logger.info(f"Running SynthSeg on {input_path}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Segmentation saved to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"SynthSeg failed: {e.stderr}")
        raise RuntimeError(f"SynthSeg failed with error: {e.stderr}")
