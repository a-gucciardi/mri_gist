from __future__ import annotations

from glob import glob
from pathlib import Path
from time import perf_counter
from typing import Iterable
import re

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .core import compute_fractal_dimension_from_array, load_volume
from .labels import load_label_lookup, sanitize_label_name


DEFAULT_RESULT_COLUMNS = [
    "source_file",
    "source_name",
    "participant_id",
    "session_id",
    "scope",
    "label",
    "label_name",
    "fd",
    "r2_adjusted",
    "min_box_size_mm",
    "max_box_size_mm",
    "voxel_size_x_mm",
    "voxel_size_y_mm",
    "voxel_size_z_mm",
    "nonzero_voxels",
]
def compute_segmentation_table(
    segmentation_path: str | Path,
    *,
    lut_path: str | Path | None = None,
    per_label: bool = False,
    include_labels: Iterable[int] | None = None,
    verbose: bool = False,
    show_progress: bool = False,
    progress_label: str | None = None,
    progress_leave: bool = True,
) -> pd.DataFrame:
    started_at = perf_counter()
    volume = load_volume(segmentation_path)
    metadata = extract_bids_metadata(volume.path)
    progress_name = progress_label or volume.path.name

    label_data = np.rint(volume.data).astype(np.int64) if per_label else None
    allowed_labels = set(include_labels) if include_labels else None
    labels: list[int] = []
    if per_label and label_data is not None:
        labels = sorted(int(value) for value in np.unique(label_data) if int(value) != 0)
        if allowed_labels is not None:
            labels = [label for label in labels if label in allowed_labels]

    progress_total = 1 + len(labels)
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=progress_total,
            desc=progress_name,
            unit="step",
            leave=progress_leave,
        )

    rows: list[dict[str, object]] = []
    whole_brain_row = _build_result_row(
        volume=volume,
        scope="whole_brain",
        label=None,
        label_name="whole_brain",
        data=(volume.data > 0),
        metadata=metadata,
        verbose=verbose,
    )
    rows.append(whole_brain_row)
    if progress_bar is not None:
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"whole-brain fd={whole_brain_row['fd']:.4f}")

    if per_label:
        lookup = load_label_lookup(lut_path)

        for label in labels:
            label_name = lookup.get(label, f"label_{label}")
            row = _build_result_row(
                volume=volume,
                scope="label",
                label=label,
                label_name=label_name,
                data=(label_data == label),
                metadata=metadata,
                verbose=verbose,
            )
            rows.append(row)

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"last={label_name} fd={row['fd']:.4f}")

    elapsed = perf_counter() - started_at
    if progress_bar is not None:
        progress_bar.set_postfix_str(f"done in {elapsed:.1f}s")
        progress_bar.close()

    dataframe = pd.DataFrame(rows)
    return dataframe[DEFAULT_RESULT_COLUMNS]


def compute_batch_table(
    input_pattern: str,
    *,
    lut_path: str | Path | None = None,
    per_label: bool = False,
    metadata_path: str | Path | None = None,
    merge_keys: Iterable[str] = ("participant_id", "session_id"),
    include_labels: Iterable[int] | None = None,
    verbose: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    started_at = perf_counter()
    matches = sorted(Path(path) for path in glob(input_pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {input_pattern}")

    volume_progress = None
    if show_progress:
        volume_progress = tqdm(matches, desc="Volumes", unit="volume")
        iterator = volume_progress
    else:
        iterator = matches

    tables = []
    for match in iterator:
        tables.append(
            compute_segmentation_table(
                match,
                lut_path=lut_path,
                per_label=per_label,
                include_labels=include_labels,
                verbose=verbose,
                show_progress=show_progress,
                progress_label=match.name,
                progress_leave=False,
            )
        )

    combined = pd.concat(tables, ignore_index=True)
    if metadata_path:
        combined = merge_metadata_table(combined, metadata_path, merge_keys=merge_keys)

    if volume_progress is not None:
        elapsed = perf_counter() - started_at
        volume_progress.set_postfix_str(f"done in {elapsed:.1f}s")
        volume_progress.close()

    return combined


def merge_metadata_table(
    dataframe: pd.DataFrame,
    metadata_path: str | Path,
    *,
    merge_keys: Iterable[str] = ("participant_id", "session_id"),
) -> pd.DataFrame:
    keys = list(merge_keys)
    metadata = pd.read_csv(metadata_path, sep=None, engine="python")

    missing_in_data = [key for key in keys if key not in dataframe.columns]
    missing_in_metadata = [key for key in keys if key not in metadata.columns]

    if missing_in_data:
        raise ValueError(f"Result table is missing merge keys: {', '.join(missing_in_data)}")
    if missing_in_metadata:
        raise ValueError(f"Metadata table is missing merge keys: {', '.join(missing_in_metadata)}")

    return dataframe.merge(metadata, on=keys, how="left")


def write_results_csv(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def extract_bids_metadata(path: str | Path) -> dict[str, str]:
    text = str(path)
    participant_match = re.search(r"(sub-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)", text)
    session_match = re.search(r"(ses-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)", text)

    return {
        "participant_id": participant_match.group(1) if participant_match else "",
        "session_id": session_match.group(1) if session_match else "",
    }


def generate_binary_masks(
    segmentation_path: str | Path,
    output_dir: str | Path,
    *,
    lut_path: str | Path | None = None,
    include_labels: Iterable[int] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    volume = load_volume(segmentation_path)
    label_data = np.rint(volume.data).astype(np.int64)
    lookup = load_label_lookup(lut_path)
    allowed_labels = set(include_labels) if include_labels else None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for label in sorted(int(value) for value in np.unique(label_data) if int(value) != 0):
        if allowed_labels is not None and label not in allowed_labels:
            continue

        label_name = sanitize_label_name(lookup.get(label, f"label_{label}"))
        target = output_path / f"{label:03d}_{label_name}.nii.gz"
        if target.exists() and not overwrite:
            written_files.append(target)
            continue

        mask = (label_data == label).astype(np.uint8)
        image = nib.Nifti1Image(mask, volume.affine)
        nib.save(image, str(target))
        written_files.append(target)

    return written_files


def _build_result_row(
    *,
    volume,
    scope: str,
    label: int | None,
    label_name: str,
    data: np.ndarray,
    metadata: dict[str, str],
    verbose: bool,
) -> dict[str, object]:
    result = compute_fractal_dimension_from_array(
        data,
        voxel_size_mm=volume.voxel_size_mm,
        verbose=verbose,
    )

    voxel_sizes = result.voxel_size_mm
    return {
        "source_file": str(volume.path),
        "source_name": strip_volume_suffix(volume.path.name),
        "participant_id": metadata["participant_id"],
        "session_id": metadata["session_id"],
        "scope": scope,
        "label": label,
        "label_name": label_name,
        "fd": result.fd,
        "r2_adjusted": result.r2_adjusted,
        "min_box_size_mm": result.min_box_size_mm,
        "max_box_size_mm": result.max_box_size_mm,
        "voxel_size_x_mm": voxel_sizes[0],
        "voxel_size_y_mm": voxel_sizes[1],
        "voxel_size_z_mm": voxel_sizes[2],
        "nonzero_voxels": result.nonzero_voxels,
    }


def strip_volume_suffix(filename: str) -> str:
    name = filename
    for suffix in (".nii.gz", ".nii", ".mgz", ".nrrd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem