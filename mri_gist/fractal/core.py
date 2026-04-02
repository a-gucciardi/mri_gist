from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class FractalWindow:
    fd: float
    r2_adjusted: float
    scale_min_mm: float
    scale_max_mm: float
    sample_count: int


@dataclass(frozen=True)
class FractalResult:
    fd: float
    r2_adjusted: float
    min_box_size_mm: float
    max_box_size_mm: float
    voxel_size_mm: tuple[float, float, float]
    nonzero_voxels: int
    windows: tuple[FractalWindow, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["windows"] = [asdict(window) for window in self.windows]
        return payload


@dataclass(frozen=True)
class VolumeData:
    path: Path
    data: np.ndarray
    affine: np.ndarray
    header: Any
    voxel_size_mm: tuple[float, float, float]


def load_volume(volume_path: str | Path) -> VolumeData:
    path = Path(volume_path)
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())
    data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape} for {path}")

    zooms = img.header.get_zooms()[:3]
    voxel_size_mm = tuple(float(value) for value in zooms)

    return VolumeData(
        path=path,
        data=data,
        affine=np.asarray(img.affine),
        header=img.header.copy(),
        voxel_size_mm=voxel_size_mm,
    )


def compute_fractal_dimension(
    volume_path: str | Path,
    *,
    verbose: bool = False,
    n_offsets: int = 20,
    min_window_size: int = 5,
    random_seed: int = 1,
) -> FractalResult:
    volume = load_volume(volume_path)
    return compute_fractal_dimension_from_array(
        volume.data,
        voxel_size_mm=volume.voxel_size_mm,
        verbose=verbose,
        n_offsets=n_offsets,
        min_window_size=min_window_size,
        random_seed=random_seed,
    )


def compute_fractal_dimension_from_array(
    data: np.ndarray,
    *,
    voxel_size_mm: tuple[float, float, float],
    verbose: bool = False,
    n_offsets: int = 20,
    min_window_size: int = 5,
    random_seed: int = 1,
) -> FractalResult:
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")

    coords = np.argwhere(array > 0)
    if coords.size == 0:
        raise ValueError("Fractal dimension requires at least one non-zero voxel")

    scales = _build_scales(array.shape)
    if scales.size < 2:
        raise ValueError("Fractal dimension requires at least two box-counting scales")

    log_scales = np.log2(scales.astype(float))
    counts = _box_counts(coords, array.shape, scales, n_offsets=n_offsets, random_seed=random_seed)
    log_counts = np.log2(counts)

    effective_window_size = max(2, min(min_window_size, scales.size))
    windows = _evaluate_windows(log_scales, log_counts, scales, float(voxel_size_mm[0]), effective_window_size)
    if not windows:
        raise ValueError("Unable to determine a valid fractal scaling window")

    best_window = max(windows, key=lambda window: (window.r2_adjusted, window.sample_count))

    if verbose:
        print(f"FD automatically selected: {best_window.fd:.4f}")

    return FractalResult(
        fd=best_window.fd,
        r2_adjusted=best_window.r2_adjusted,
        min_box_size_mm=best_window.scale_min_mm,
        max_box_size_mm=best_window.scale_max_mm,
        voxel_size_mm=tuple(float(value) for value in voxel_size_mm),
        nonzero_voxels=int(coords.shape[0]),
        windows=tuple(windows),
    )


def _build_scales(shape: tuple[int, int, int]) -> np.ndarray:
    max_extent = max(int(value) for value in shape)
    exponent_limit = math.ceil(math.log2(max_extent))
    return 2 ** np.arange(0, exponent_limit + 1)


def _box_counts(
    coords: np.ndarray,
    shape: tuple[int, int, int],
    scales: np.ndarray,
    *,
    n_offsets: int,
    random_seed: int,
) -> np.ndarray:
    rng = random.Random(random_seed)
    counts: list[float] = []

    for scale in scales:
        scale_value = int(scale)
        offset_counts: list[int] = []
        for _ in range(n_offsets):
            bins = []
            for axis_length in shape:
                start = -rng.randint(0, scale_value)
                stop = int(axis_length) + 1 + scale_value
                bins.append(np.arange(start, stop, scale_value))

            histogram, _ = np.histogramdd(coords, bins=tuple(bins))
            offset_counts.append(int(np.count_nonzero(histogram)))

        counts.append(float(np.mean(offset_counts)))

    return np.asarray(counts, dtype=float)


def _evaluate_windows(
    log_scales: np.ndarray,
    log_counts: np.ndarray,
    scales: np.ndarray,
    min_voxel_size_mm: float,
    min_window_size: int,
) -> list[FractalWindow]:
    windows: list[FractalWindow] = []

    for window_size in range(scales.size, min_window_size - 1, -1):
        for start in range(0, scales.size - window_size + 1):
            end = start + window_size
            x_values = log_scales[start:end]
            y_values = log_counts[start:end]

            slope, intercept = np.polyfit(x_values, y_values, 1)
            predictions = slope * x_values + intercept
            r2 = _r2_score(y_values, predictions)
            adjusted_r2 = _adjusted_r2(r2, sample_count=window_size, predictor_count=1)

            windows.append(
                FractalWindow(
                    fd=round(float(-slope), 4),
                    r2_adjusted=round(float(adjusted_r2), 3),
                    scale_min_mm=round(float(scales[start] * min_voxel_size_mm), 4),
                    scale_max_mm=round(float(scales[end - 1] * min_voxel_size_mm), 4),
                    sample_count=window_size,
                )
            )

    return windows


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual_sum = float(np.sum((y_true - y_pred) ** 2))
    total_sum = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total_sum == 0:
        return 1.0
    return 1.0 - (residual_sum / total_sum)


def _adjusted_r2(r2: float, *, sample_count: int, predictor_count: int) -> float:
    denominator = sample_count - (predictor_count + 1)
    if denominator <= 0:
        return float(r2)
    return 1.0 - (1.0 - r2) * ((sample_count - 1) / denominator)