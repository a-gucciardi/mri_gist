"""
3D Midway Registration and Midsagittal Plane (MSP) Alignment.

Given a 3D brain volume I, this module:
  1. Generates its mirror image I' by flipping along the left-right axis.
  2. Computes the affine transformation T: I -> I' via ANTsPy registration.
  3. Computes the matrix square root T^{1/2} so that applying it to I aligns the
     brain's sagittal plane with the image volume's vertical midplane.
  4. Optionally splits the aligned volume into left/right hemispheres.

tilt and centering the midsagittal plane.
"""

# goal : provide a clean, unbiased baseline for symmetry analysis

import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.linalg import sqrtm, inv, norm
from scipy.ndimage import affine_transform, center_of_mass, shift

try:
    import ants
except ImportError:
    ants = None

logger = logging.getLogger("rich")


# ---------------------------------------------------------------------------
# Core: Midway alignment
# ---------------------------------------------------------------------------

def midway_align(
    input_path: str,
    interp_order: int = 1,
    transform_type: str = "Affine",
    center_to_com: bool = True,
) -> dict:
    """
    Perform 3D midway registration to align the midsagittal plane (MSP)
    with the image grid's vertical midplane.

    Parameters
    ----------
    input_path : str or Path
        Path to a NIfTI (.nii / .nii.gz) brain volume.
    interp_order : int
        Interpolation order for resampling (0 = nearest-neighbor for label
        maps, 1 = linear for anatomical volumes).
    transform_type : str
        ANTsPy registration type ('Affine' or 'Rigid').  Affine recommended.
    center_to_com : bool
        If True (default), shift the volume so that its center of mass
        coincides with the geometric center before flipping.  This removes
        the large translation bias from off-center brains, leaving T to
        capture only rotation / tilt.  The shift is undone when writing
        outputs so world coordinates remain correct.

    Returns
    -------
    dict with keys:
        aligned_data   : np.ndarray  – MSP-aligned volume
        aligned_img    : Nifti1Image – nibabel image with correct affine
        original_img   : Nifti1Image – the loaded input image
        centered_data  : np.ndarray  – volume after COM centering (before registration)
        com_shift      : np.ndarray  (3,) – voxel shift applied for centering (zeros if disabled)
        T_full         : np.ndarray  (4×4) – full affine I_centered → I_centered'
        T_half         : np.ndarray  (4×4) – half-transform T^{1/2}
        lr_axis        : int         – array axis corresponding to L-R
        orientation    : tuple       – axis codes, e.g. ('R','A','S')
        elapsed_sec    : float       – wall-clock time for registration
    """
    if ants is None:
        raise ImportError(
            "ANTsPy is required for midway registration. "
            "Install with: pip install antspyx"
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # --- Load ---
    logger.info(f"Loading {input_path.name}")
    img = nib.load(str(input_path))
    data = img.get_fdata()
    affine = img.affine

    # --- Detect L-R axis from orientation ---
    orientation = aff2axcodes(affine)  # e.g. ('R', 'A', 'S')
    lr_axis = _find_lr_axis(orientation)
    logger.info(f"Orientation: {orientation}, L-R axis = {lr_axis}")

    # --- Step 0 (optional): Center-of-mass pre-centering ---
    # We compute the COM shift but only apply it to a *working copy* used
    # for registration.  The final output is produced by composing the COM
    # shift and T^{1/2} into a single affine applied to the original data,
    # so only ONE interpolation ever touches the output (no double-blur).
    com_shift = np.zeros(3)
    if center_to_com:
        com_shift = _compute_com_shift(data)
        logger.info(
            f"COM centering: shift = [{com_shift[0]:.1f}, {com_shift[1]:.1f}, "
            f"{com_shift[2]:.1f}] voxels (LR = {com_shift[lr_axis]:.1f})"
        )

    # Working copy: apply COM shift for registration only (this gets
    # interpolated once, but is never used as the final output).
    if np.any(com_shift != 0):
        centered_data = shift(
            data, com_shift, order=interp_order, mode="constant", cval=0.0
        )
    else:
        centered_data = data

    # --- Step 1: Generate mirror image I' (from centered data) ---
    flipped_data = np.flip(centered_data, axis=lr_axis).copy()

    # --- Step 2: Register centered → flipped (affine) ---
    logger.info(f"Registering centered → mirror ({transform_type})…")
    t0 = time.time()

    orig_ants = ants.from_numpy(centered_data.astype(np.float32))
    flip_ants = ants.from_numpy(flipped_data.astype(np.float32))

    # Copy spatial metadata from nibabel so ANTs works in the right space
    spacing = tuple(np.abs(np.diag(affine[:3, :3])).tolist())
    origin = tuple(affine[:3, 3].tolist())
    orig_ants.set_spacing(spacing)
    orig_ants.set_origin(origin)
    flip_ants.set_spacing(spacing)
    flip_ants.set_origin(origin)

    reg = ants.registration(
        fixed=flip_ants,
        moving=orig_ants,
        type_of_transform=transform_type,
    )
    elapsed = time.time() - t0
    logger.info(f"Registration completed in {elapsed:.1f}s")

    # --- Step 3: Extract full 4×4 transform matrix T ---
    T_full = _extract_ants_matrix(reg)
    logger.info(f"Full transform T:\n{np.array2string(T_full, precision=4)}")

    # --- Step 4: Compute T^{1/2} (matrix square root of full 4×4) ---
    T_half = _compute_half_transform(T_full)
    logger.info(f"Half transform T^(1/2):\n{np.array2string(T_half, precision=4)}")

    # --- Validate: (T^{1/2})^2 ≈ T ---
    _validate_half_transform(T_full, T_half)

    # --- Step 5: Compose COM-shift + T^{1/2} and apply ONCE ---
    # Forward pipeline: original → shift by com_shift → apply T^{1/2}
    # As a single 4×4: T_combined = T_half @ T_com
    T_com = np.eye(4)
    T_com[:3, 3] = com_shift  # translation-only matrix

    T_combined = T_half @ T_com
    logger.info(
        f"Combined transform (COM + T^1/2) – single interpolation:\n"
        f"{np.array2string(T_combined, precision=4)}"
    )

    aligned_data = _apply_combined_transform(data, T_combined, interp_order)

    # Adjust affine for the effective shift so world coords stay correct
    aligned_affine = affine.copy()
    aligned_affine[:3, 3] -= affine[:3, :3] @ com_shift
    aligned_img = nib.Nifti1Image(aligned_data, aligned_affine)

    return {
        "aligned_data": aligned_data,
        "aligned_img": aligned_img,
        "original_img": img,
        "centered_data": centered_data,
        "com_shift": com_shift,
        "T_full": T_full,
        "T_half": T_half,
        "lr_axis": lr_axis,
        "orientation": orientation,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Hemisphere splitting
# ---------------------------------------------------------------------------

def hemisphere_split(
    aligned_data: np.ndarray,
    affine: np.ndarray,
    lr_axis: int = 0,
) -> dict:
    """
    Split an MSP-aligned volume into left and right hemispheres.

    Parameters
    ----------
    aligned_data : np.ndarray
        The midway-aligned 3-D volume (output of ``midway_align``).
    affine : np.ndarray (4×4)
        NIfTI affine matrix of the aligned volume.
    lr_axis : int
        The array axis corresponding to left-right.

    Returns
    -------
    dict with keys:
        left_data, right_data        : np.ndarray
        left_img, right_img          : Nifti1Image
        midpoint                     : int
        left_nonzero, right_nonzero  : int  (voxel counts)
    """
    midpoint = aligned_data.shape[lr_axis] // 2

    slices_left = [slice(None)] * 3
    slices_right = [slice(None)] * 3
    slices_left[lr_axis] = slice(0, midpoint)
    slices_right[lr_axis] = slice(midpoint, None)

    left_data = aligned_data[tuple(slices_left)]
    right_data = aligned_data[tuple(slices_right)]

    # Adjust affine for right hemisphere (shift origin along L-R axis)
    right_affine = affine.copy()
    right_affine[lr_axis, 3] += midpoint * affine[lr_axis, lr_axis]

    left_img = nib.Nifti1Image(left_data, affine)
    right_img = nib.Nifti1Image(right_data, right_affine)

    left_nz = int(np.count_nonzero(left_data))
    right_nz = int(np.count_nonzero(right_data))

    return {
        "left_data": left_data,
        "right_data": right_data,
        "left_img": left_img,
        "right_img": right_img,
        "midpoint": midpoint,
        "left_nonzero": left_nz,
        "right_nonzero": right_nz,
    }


# ---------------------------------------------------------------------------
# Diagnostics / output saving
# ---------------------------------------------------------------------------

def save_diagnostics(
    result: dict,
    hemi: Optional[dict],
    output_dir: str,
    stem: str,
) -> list[str]:
    """
    Save full diagnostic outputs for a midway-aligned volume.

    Parameters
    ----------
    result : dict
        Output of ``midway_align()``.
    hemi : dict or None
        Output of ``hemisphere_split()``, or None to skip hemisphere files.
    output_dir : str or Path
        Directory to write outputs into (created if needed).
    stem : str
        Filename stem for outputs, e.g. 'sub-01_T1w'.

    Returns
    -------
    list of str – paths of all files written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # --- Aligned volume ---
    p = out / f"{stem}_midway_aligned.nii.gz"
    nib.save(result["aligned_img"], str(p))
    written.append(str(p))
    logger.info(f"Saved aligned volume → {p.name}")

    # --- Transform matrices ---
    for label, key in [("T_full", "T_full"), ("T_half", "T_half")]:
        p = out / f"{stem}_{label}.txt"
        np.savetxt(str(p), result[key], fmt="%.8f")
        written.append(str(p))

    # --- Hemisphere volumes ---
    if hemi is not None:
        for side in ("left", "right"):
            p = out / f"{stem}_{side}.nii.gz"
            nib.save(hemi[f"{side}_img"], str(p))
            written.append(str(p))
            logger.info(f"Saved {side} hemisphere → {p.name}")

    # --- PNG diagnostics ---
    try:
        written += _save_diagnostic_pngs(result, hemi, out, stem)
    except Exception as e:
        logger.warning(f"Could not generate PNG diagnostics: {e}")

    return written


def _save_diagnostic_pngs(result, hemi, out, stem) -> list[str]:
    """Generate sagittal-slice and overlay PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    aligned = result["aligned_data"]
    original = result["original_img"].get_fdata()
    lr = result["lr_axis"]

    mid_idx = aligned.shape[lr] // 2

    view_labels = ["Sagittal", "Coronal", "Axial"]  # along LR, AP, SI

    # --- 3-view comparison: original vs aligned (sagittal, coronal, axial) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, ax_idx in enumerate(range(3)):
        mid = original.shape[ax_idx] // 2
        sl_o = [slice(None)] * 3
        sl_a = [slice(None)] * 3
        sl_o[ax_idx] = mid
        sl_a[ax_idx] = mid
        axes[0, col].imshow(np.rot90(original[tuple(sl_o)]), cmap="gray")
        axes[0, col].set_title(f"Original – {view_labels[col]}")
        axes[0, col].axis("off")
        axes[1, col].imshow(np.rot90(aligned[tuple(sl_a)]), cmap="gray")
        axes[1, col].set_title(f"Aligned – {view_labels[col]}")
        axes[1, col].axis("off")

    com_s = result.get("com_shift", np.zeros(3))
    fig.suptitle(
        f"{stem}  |  {result['orientation']}  |  LR={lr}  "
        f"|  COM shift=[{com_s[0]:.1f}, {com_s[1]:.1f}, {com_s[2]:.1f}]",
        fontsize=11,
    )
    fig.tight_layout()
    p = out / f"{stem}_3view.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    written.append(str(p))

    # --- Mid-sagittal slice (kept for backward compat) ---
    slicer = [slice(None)] * 3
    slicer[lr] = mid_idx
    sag_aligned = aligned[tuple(slicer)]
    sag_original = original[tuple(slicer)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.rot90(sag_original), cmap="gray")
    axes[0].set_title("Original – mid-sagittal")
    axes[0].axis("off")
    axes[1].imshow(np.rot90(sag_aligned), cmap="gray")
    axes[1].set_title("Midway-aligned – mid-sagittal")
    axes[1].axis("off")
    fig.suptitle(f"{stem}  |  orientation {result['orientation']}  |  LR axis = {lr}")
    fig.tight_layout()
    p = out / f"{stem}_midway_slice.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    written.append(str(p))

    # Overlay: original vs. flipped-and-aligned (symmetry check)
    flipped_aligned = np.flip(aligned, axis=lr)
    # Take the central axial slice for the overlay
    ax_axis = 2  # typically S-I
    ax_mid = aligned.shape[ax_axis] // 2
    slicer_ax = [slice(None)] * 3
    slicer_ax[ax_axis] = ax_mid

    slice_orig = aligned[tuple(slicer_ax)]
    slice_flip = flipped_aligned[tuple(slicer_ax)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.rot90(slice_orig), cmap="gray")
    axes[0].set_title("Aligned (axial)")
    axes[0].axis("off")
    axes[1].imshow(np.rot90(slice_flip), cmap="gray")
    axes[1].set_title("Flipped aligned (axial)")
    axes[1].axis("off")

    # Difference map
    diff = np.abs(slice_orig.astype(float) - slice_flip.astype(float))
    axes[2].imshow(np.rot90(diff), cmap="hot")
    axes[2].set_title("| Aligned − Flipped |")
    axes[2].axis("off")

    fig.suptitle(f"{stem} – symmetry overlay")
    fig.tight_layout()
    p = out / f"{stem}_overlay.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    written.append(str(p))

    # If hemispheres exist, show them side-by-side
    if hemi is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        left_slice = hemi["left_data"]
        right_slice = hemi["right_data"]

        # Take central coronal slice from each hemisphere
        cor_axis = 1  # typically A-P
        l_mid = left_slice.shape[cor_axis] // 2
        r_mid = right_slice.shape[cor_axis] // 2

        sl_l = [slice(None)] * 3
        sl_r = [slice(None)] * 3
        sl_l[cor_axis] = l_mid
        sl_r[cor_axis] = r_mid

        axes[0].imshow(np.rot90(left_slice[tuple(sl_l)]), cmap="gray")
        axes[0].set_title(f"Left ({hemi['left_nonzero']:,} nonzero)")
        axes[0].axis("off")
        axes[1].imshow(np.rot90(right_slice[tuple(sl_r)]), cmap="gray")
        axes[1].set_title(f"Right ({hemi['right_nonzero']:,} nonzero)")
        axes[1].axis("off")

        fig.suptitle(f"{stem} – hemisphere split (midpoint = {hemi['midpoint']})")
        fig.tight_layout()
        p = out / f"{stem}_hemispheres.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        written.append(str(p))

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_com_shift(data: np.ndarray) -> np.ndarray:
    """
    Compute the voxel-space shift needed to move the center of mass
    to the geometric center.  Does NOT apply it (no interpolation).
    """
    geometric_center = np.array(data.shape) / 2.0
    com = np.array(center_of_mass(np.abs(data)))  # abs for robustness
    return geometric_center - com


def _find_lr_axis(orientation: tuple) -> int:
    """Return the array axis that corresponds to Left-Right."""
    for i, code in enumerate(orientation):
        if code in ("R", "L"):
            return i
    # Fallback: assume axis 0 and warn
    warnings.warn(
        f"Could not determine L-R axis from orientation {orientation}. "
        "Falling back to axis 0."
    )
    return 0


def _extract_ants_matrix(reg: dict) -> np.ndarray:
    """
    Read the forward affine transform from an ANTsPy registration result
    and return it as a 4×4 homogeneous matrix.
    """
    fwd = reg["fwdtransforms"]
    # Find the affine / linear transform file (*.mat)
    mat_file = None
    for f in fwd:
        if f.endswith(".mat"):
            mat_file = f
            break
    if mat_file is None:
        raise RuntimeError(
            "No affine (.mat) transform found in registration output. "
            f"Forward transforms: {fwd}"
        )

    txf = ants.read_transform(mat_file)
    params = np.array(txf.parameters)

    # ANTsPy stores affine as [R11,R12,R13,R21,...,R33, tx,ty,tz]
    R = params[:9].reshape(3, 3)
    t = params[9:12]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _compute_half_transform(T: np.ndarray) -> np.ndarray:
    """
    Compute the matrix square root T^{1/2} of a 4×4 homogeneous matrix.

    Validates that the imaginary part is negligible and returns .real.
    """
    T_half_complex = sqrtm(T)

    # Check imaginary residual
    max_imag = np.max(np.abs(T_half_complex.imag))
    if max_imag > 1e-6:
        warnings.warn(
            f"Matrix square root has non-negligible imaginary part "
            f"(max |imag| = {max_imag:.2e}). Taking real part only."
        )
    elif max_imag > 0:
        logger.debug(f"sqrtm imaginary residual: {max_imag:.2e} (negligible)")

    return T_half_complex.real


def _validate_half_transform(T_full: np.ndarray, T_half: np.ndarray) -> None:
    """Check that (T^{1/2})^2 ≈ T within tolerance."""
    T_recon = T_half @ T_half
    err = norm(T_recon - T_full, ord="fro")
    rel = err / max(norm(T_full, ord="fro"), 1e-12)

    if rel > 1e-3:
        warnings.warn(
            f"Half-transform validation: (T^(1/2))^2 deviates from T "
            f"(relative Frobenius error = {rel:.4e}). Results may be unreliable."
        )
    else:
        logger.info(f"Half-transform validation passed (rel error = {rel:.2e})")


def _apply_combined_transform(
    data: np.ndarray,
    T_combined: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Apply a composed 4×4 transform (COM-shift + T^{1/2}) to a 3-D volume
    in a **single interpolation step**, avoiding the blur from sequential
    resampling.

    scipy.ndimage.affine_transform uses *inverse* mapping (output→input),
    so we invert T_combined.
    """
    T_inv = inv(T_combined)
    R_inv = T_inv[:3, :3]
    t_inv = T_inv[:3, 3]

    aligned = affine_transform(
        data.astype(np.float64),
        R_inv,
        offset=t_inv,
        order=order,
        mode="constant",
        cval=0.0,
    )
    return aligned
