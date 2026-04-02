from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


def load_label_lookup(lut_path: str | Path) -> dict[int, str]:
    path = Path(lut_path)
    suffixes = path.suffixes
    if ".tsv" in suffixes or path.suffix == ".csv":
        return _load_tabular_lut(path)
    return _load_freesurfer_lut(path)


def sanitize_label_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-") or "label"


def _load_tabular_lut(path: Path) -> dict[int, str]:
    dataframe = pd.read_csv(path, sep=None, engine="python")

    index_column = None
    name_column = None
    for column in dataframe.columns:
        lowered = str(column).strip().lower()
        if lowered in {"index", "label", "id"}:
            index_column = column
        if lowered == "name":
            name_column = column

    if index_column is None or name_column is None:
        raise ValueError(f"Could not find label index/name columns in LUT: {path}")

    lookup: dict[int, str] = {}
    for _, row in dataframe.iterrows():
        lookup[int(row[index_column])] = str(row[name_column]).strip().strip('"')

    return lookup


def _load_freesurfer_lut(path: Path) -> dict[int, str]:
    lookup: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                label = int(parts[0])
            except ValueError:
                continue

            name = " ".join(parts[1:-4]).strip()
            if name:
                lookup[label] = name

    if not lookup:
        raise ValueError(f"No labels were parsed from LUT: {path}")

    return lookup