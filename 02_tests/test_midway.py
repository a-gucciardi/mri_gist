#!/usr/bin/env python3
"""
Test / demo script for 3D Midway Registration & MSP Alignment.

Runs midway_align() + hemisphere_split() + save_diagnostics() on all
three files in anat_sample/ and prints a summary.

Usage:
    python 02_tests/test_midway.py
"""

import sys
import time
from pathlib import Path

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table

from mri_gist.registration.midway import midway_align, hemisphere_split, save_diagnostics

console = Console()

ANAT_DIR = Path(__file__).resolve().parent.parent / "anat_sample"
OUTPUT_DIR = Path(__file__).resolve().parent / "test_out_midway"

# All three sample files
FILES = sorted(ANAT_DIR.glob("*.nii.gz"))

# Use nearest-neighbor interpolation for segmentation label maps
LABEL_KEYWORDS = ("dseg", "seg", "label", "parc")


def is_label_map(path: Path) -> bool:
    name = path.name.lower()
    return any(kw in name for kw in LABEL_KEYWORDS)


def main():
    if not FILES:
        console.print(f"[red]No .nii.gz files found in {ANAT_DIR}[/red]")
        sys.exit(1)

    console.rule("[bold blue]3D Midway Registration – MSP Alignment Test")
    console.print(f"Input directory : {ANAT_DIR}")
    console.print(f"Output directory: {OUTPUT_DIR}")
    console.print(f"Files to process: {len(FILES)}\n")

    summary_table = Table(title="Processing Summary")
    summary_table.add_column("File", style="cyan", no_wrap=True)
    summary_table.add_column("Orientation")
    summary_table.add_column("LR axis", justify="center")
    summary_table.add_column("COM shift LR", justify="right")
    summary_table.add_column("Interp", justify="center")
    summary_table.add_column("Reg time (s)", justify="right")
    summary_table.add_column("T_half valid", justify="center")
    summary_table.add_column("L nonzero", justify="right")
    summary_table.add_column("R nonzero", justify="right")
    summary_table.add_column("L/R ratio", justify="right")
    summary_table.add_column("Files out", justify="right")

    for fpath in FILES:
        console.rule(f"[bold]{fpath.name}")

        interp = 0 if is_label_map(fpath) else 1
        console.print(f"  Interpolation order: {interp} ({'nearest' if interp == 0 else 'linear'})")

        # --- Midway align ---
        t0 = time.time()
        result = midway_align(str(fpath), interp_order=interp)
        total_time = time.time() - t0

        console.print(f"  Orientation   : {result['orientation']}")
        console.print(f"  L-R axis      : {result['lr_axis']}")
        com_s = result['com_shift']
        console.print(f"  COM shift     : [{com_s[0]:.1f}, {com_s[1]:.1f}, {com_s[2]:.1f}] voxels")
        console.print(f"  COM shift LR  : {com_s[result['lr_axis']]:.1f} voxels")
        console.print(f"  Reg time      : {result['elapsed_sec']:.1f}s")
        console.print(f"  Total time    : {total_time:.1f}s")

        # Print transform matrices
        console.print("  T (full):")
        for row in result["T_full"]:
            console.print(f"    [{' '.join(f'{v:8.4f}' for v in row)}]")
        console.print("  T^(1/2):")
        for row in result["T_half"]:
            console.print(f"    [{' '.join(f'{v:8.4f}' for v in row)}]")

        # Validate
        T_recon = result["T_half"] @ result["T_half"]
        rel_err = np.linalg.norm(T_recon - result["T_full"]) / max(np.linalg.norm(result["T_full"]), 1e-12)
        valid = "✓" if rel_err < 1e-3 else f"✗ ({rel_err:.2e})"
        console.print(f"  (T^1/2)^2 ≈ T : {valid}  (rel err = {rel_err:.2e})")

        # --- Hemisphere split ---
        hemi = hemisphere_split(
            result["aligned_data"],
            result["aligned_img"].affine,
            result["lr_axis"],
        )
        ratio = hemi["left_nonzero"] / max(hemi["right_nonzero"], 1)
        console.print(f"  Midpoint      : {hemi['midpoint']}")
        console.print(f"  Left nonzero  : {hemi['left_nonzero']:,}")
        console.print(f"  Right nonzero : {hemi['right_nonzero']:,}")
        console.print(f"  L/R ratio     : {ratio:.4f}")

        # For label maps, check label preservation
        if interp == 0:
            orig_labels = set(np.unique(result["original_img"].get_fdata().astype(int)))
            aligned_labels = set(np.unique(result["aligned_data"].astype(int)))
            extra = aligned_labels - orig_labels
            if extra:
                console.print(f"  [red]WARNING: interpolation created {len(extra)} spurious labels: {sorted(extra)[:10]}[/red]")
            else:
                console.print(f"  [green]Label integrity: OK ({len(orig_labels)} labels preserved)[/green]")

        # --- Save diagnostics ---
        stem = fpath.name.replace(".nii.gz", "")
        written = save_diagnostics(result, hemi, str(OUTPUT_DIR), stem)
        console.print(f"  Outputs       : {len(written)} files → {OUTPUT_DIR}/")

        summary_table.add_row(
            fpath.name[:40],
            str(result["orientation"]),
            str(result["lr_axis"]),
            f"{result['com_shift'][result['lr_axis']]:.1f}",
            str(interp),
            f"{result['elapsed_sec']:.1f}",
            valid,
            f"{hemi['left_nonzero']:,}",
            f"{hemi['right_nonzero']:,}",
            f"{ratio:.3f}",
            str(len(written)),
        )

    console.print()
    console.print(summary_table)
    console.print(f"\n[green]All outputs saved to {OUTPUT_DIR}/[/green]")


if __name__ == "__main__":
    main()
