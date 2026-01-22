import argparse
import time
import logging
import psutil
import shutil
from pathlib import Path
from mri_gist.utils.logging import setup_logger
from mri_gist.registration.core import register_image
from mri_gist.segmentation.synthseg import run_synthseg
from mri_gist.detection.hemisphere import hemisphere_separation

logger = logging.getLogger("rich")

# debug benchmark

def measure_performance(func, *args, **kwargs):
    """Measure execution time and peak memory usage of a function."""
    process = psutil.Process()
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()

    try:
        result = func(*args, **kwargs)
        success = True
    except Exception as e:
        logger.error(f"Task failed: {e}")
        result = None
        success = False

    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024  # MB

    duration = end_time - start_time
    mem_usage = end_mem - start_mem

    return success, duration, mem_usage

def run_benchmark(data_dir, output_dir):
    """Run benchmark on sample data."""
    data_path = Path(data_dir)
    t1_files = list(data_path.glob("*T1w.nii.gz"))
    t2_files = list(data_path.glob("*T2w.nii.gz"))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not t1_files:
        logger.error("No T1w images found in data directory.")
        return

    input_t1 = t1_files[0]
    logger.info(f"Using input: {input_t1}")

    results = []

    # 1. Registration (Self-registration for demo)
    logger.info("--- Benchmarking Registration ---")
    reg_out = out_path / "registered.nii.gz"
    success, duration, mem = measure_performance(
        register_image,
        moving=str(input_t1),
        fixed=str(input_t1), # self
        output=str(reg_out),
        transform_type='rigid'
    )
    results.append({"Task": "Registration (Rigid)", "Success": success, "Time (s)": duration, "Memory (MB)": mem})

    # 2. Hemisphere Separation
    logger.info("--- Benchmarking Hemisphere Separation ---")
    left_out = out_path / "left.nii.gz"
    right_out = out_path / "right.nii.gz"
    success, duration, mem = measure_performance(
        hemisphere_separation,
        input_path=str(input_t1),
        left_output=str(left_out),
        right_output=str(right_out),
        method='antspy'
    )
    results.append({"Task": "Hemi Separation", "Success": success, "Time (s)": duration, "Memory (MB)": mem})

    # 3. Segmentation
    logger.info("--- Benchmarking Segmentation ---")
    seg_out = out_path / "segmentation.nii.gz"
    success, duration, mem = measure_performance(
        run_synthseg, 
        input_path=str(input_t1), 
        output_path=str(seg_out),
        robust=True,
        parcellation=False
    )
    results.append({"Task": "Segmentation", "Success": success, "Time (s)": duration, "Memory (MB)": mem})

    # Report
    print("\n=== Benchmark Results ===")
    print(f"{'Task':<25} | {'Success':<8} | {'Time (s)':<10} | {'Memory (MB)':<12}")
    print("-" * 65)
    for res in results:
        print(f"{res['Task']:<25} | {str(res['Success']):<8} | {res['Time (s)']:<10.2f} | {res['Memory (MB)']:<12.2f}")

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="anat_sample", help="Path to sample data")
    parser.add_argument("--output", default="benchmark_results", help="Path to output directory")
    args = parser.parse_args()

    run_benchmark(args.data, args.output)
