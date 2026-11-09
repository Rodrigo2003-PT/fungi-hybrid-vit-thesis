#!/usr/bin/env python3
"""
preprocess_external.py

Deterministically preprocess the external new-microscope validation images for
frozen-model inference with the final LSA-ConvTok Hybrid model.

Scientific role
---------------
The external cohort was acquired on a different microscope (ZEISS Axiocam 820
mono, 2x2 reduced to 2256 x 2256 px, 14-bit data in 16-bit TIFF containers).
The model was trained on the original 12-bit domain (Prime 95B, 1200 x 1200 px).
This script implements two modes and nothing else:

    locked_pixel_pipeline  (primary, pre-specified)
        raw TIFF -> verify effective bit depth (expected 14)
                 -> I / (2^14 - 1), clip to [0, 1]
                 -> cv2.resize to 384 x 384 with INTER_LANCZOS4, clip to [0, 1]
                 -> float32 .npy mirroring manifest relative paths

    photometric_mean_std_matched_pipeline  (post hoc diagnostic sensitivity)
        locked_pixel_pipeline output
                 -> per-image affine map to the frozen checkpoint mean/std
                 -> clip to [0, 1]

Both modes are pixel-locked: no physical-scale resampling is performed. The
numerical behaviour of both modes is deliberately frozen; optical geometry is
recorded as descriptive provenance only (``optics_reference``).

Interpolation kernel
--------------------
The resize kernel (cv2.INTER_LANCZOS4 on float32 input) is identical to the
training preprocessing (simpleViT_fungi data/preprocessing.py::
resize_and_preprocess_image, author-verified). The decimation factor differs
(external 2256 -> 384 = 5.875; training 1200 -> 384 = 3.125) and LANCZOS4 is not
antialiased, so aliasing/noise texture can differ between domains. This is a
documented limitation, not a corrected step.

Reproducibility guards
----------------------
* ``preprocessing_fingerprint.json`` hashes every parameter that changes the
  arrays. Existing outputs are reused only when the fingerprint matches;
  otherwise the run aborts unless ``--force`` is given (F-08).
* Results, failures and metadata are written in sorted relative_path order.
* Checkpoint mean/std are resolved with explicit source tracking; a silent
  fallback to built-in defaults is an error in strict runs (F-03).

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tifffile
except ImportError as exc:
    raise ImportError("tifffile is required for preprocess_external.py. Install it with: pip install tifffile") from exc

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required for preprocess_external.py. Install it with: pip install opencv-python") from exc

from load_frozen_hybrid import (
    expected_values_from_config,
    extract_arch_metadata,
    extract_config,
    extract_model_state_dict,
    hash_state_dict_keys,
    resolve_mean_std,
    sha256_file,
    torch_load_checkpoint,
)
from optics_provenance import optics_provenance_record, optics_reference_summary
from pipeline_common import load_optional_yaml, resolve_setting

PREPROCESSING_SCHEMA_VERSION = "2.0"
SUPPORTED_TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}
SUPPORTED_BIT_DEPTHS = {8, 12, 14, 16}
PRIMARY_MODE = "locked_pixel_pipeline"
PHOTOMETRIC_MODE = "photometric_mean_std_matched_pipeline"
SUPPORTED_MODES = (PRIMARY_MODE, PHOTOMETRIC_MODE)
DEFAULT_EXPECTED_EFFECTIVE_BIT_DEPTH = 14
DEFAULT_IMAGE_SIZE = 384
DEFAULT_PHOTOMETRIC_MIN_SOURCE_STD = 1e-6
RESIZE_INTERPOLATION_NAME = "cv2.INTER_LANCZOS4"
TRAINING_RESIZE_KERNEL = (
    "cv2.resize(float32, (384, 384), interpolation=cv2.INTER_LANCZOS4) "
    "[author-verified: simpleViT_fungi data/preprocessing.py::resize_and_preprocess_image]"
)
FINGERPRINT_FILENAME = "preprocessing_fingerprint.json"

FAILURE_COLUMNS = ["relative_path", "image_path", "slide_id", "species_name", "error"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess external-validation TIFF images for deterministic frozen-model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path,
                        help="external_manifest.csv produced by external_manifest.py.")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="final_model_lsa_convtok.pth primary inference checkpoint.")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="external_validation_metadata.json (schema 2.0); used only to record optics provenance.")
    parser.add_argument("--config", type=Path, default=None, help="Optional external_validation_config.yaml.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for preprocessed .npy outputs and metadata.")
    parser.add_argument(
        "--mode",
        type=str,
        default=PRIMARY_MODE,
        choices=list(SUPPORTED_MODES),
        help=(
            "locked_pixel_pipeline is the primary analysis; photometric_mean_std_matched_pipeline "
            "is a post hoc photometric sensitivity analysis."
        ),
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Frozen model input size (384).")
    parser.add_argument(
        "--expected-effective-bit-depth",
        type=int,
        default=None,
        choices=[8, 12, 14, 16],
        help="Expected external effective bit depth. Precedence: flag > config tiff_audit > 14.",
    )
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Parallel workers. Default uses a conservative CPU heuristic.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess images even if outputs with a matching fingerprint exist.")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Abort on any failure or checkpoint/config incompatibility.")
    parser.add_argument("--allow-effective-bit-depth-fallback", action="store_true",
                        help="Diagnostic only: allow per-image inferred bit depth to differ from expected.")
    parser.add_argument("--save-standardized", action="store_true",
                        help="Save arrays after checkpoint mean/std standardization (default: [0,1] arrays).")
    parser.add_argument(
        "--post-resize-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip pre-standardized arrays to [0,1] after interpolation/harmonization (Lanczos overshoot).",
    )
    parser.add_argument("--photometric-min-source-std", type=float, default=DEFAULT_PHOTOMETRIC_MIN_SOURCE_STD,
                        help="Minimum per-image std allowed in the photometric sensitivity pipeline.")
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="Optional dataset root; read raw_root/relative_path instead of manifest image_path.")
    parser.add_argument("--metadata-filename", type=str, default="preprocessing_metadata.json",
                        help="Metadata JSON name inside the output directory.")
    parser.add_argument("--sidecar-output-dir", type=Path, default=None,
                        help="Optional Drive-safe directory receiving copies of metadata, fingerprint and failures CSV.")
    parser.add_argument("--progress-every", type=int, default=250,
                        help="Print progress every N images (0 disables).")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    required = {"image_path", "relative_path", "filename", "slide_id", "species_name", "label", "hyphae_status"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return sorted(rows, key=lambda r: r["relative_path"])


def load_checkpoint_metadata(checkpoint_path: Path, config: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
    """Checkpoint identity fields and resolved mean/std (shared resolver with the loader)."""
    checkpoint, load_mode = torch_load_checkpoint(checkpoint_path)
    state_dict = extract_model_state_dict(checkpoint)
    arch_metadata = extract_arch_metadata(checkpoint)
    ckpt_config, config_source = extract_config(checkpoint, arch_metadata)
    expected = expected_values_from_config(config)
    mean, std, mean_std_info = resolve_mean_std(
        checkpoint,
        expected_mean=expected["checkpoint_mean"],
        expected_std=expected["checkpoint_std"],
        strict=strict,
    )
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    image_size = ckpt_config.get("image_size", arch_metadata.get("image_size"))
    channels = ckpt_config.get("channels", arch_metadata.get("channels"))
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_load_mode": load_mode,
        "config_source": config_source,
        "stored_architecture_hash": arch_metadata.get("arch_hash", checkpoint.get("architecture_hash", metadata.get("architecture_hash"))),
        "image_size": image_size,
        "channels": channels,
        "checkpoint_training_bit_depth": ckpt_config.get("bit_depth", checkpoint.get("bit_depth", metadata.get("bit_depth"))),
        "checkpoint_mean": mean,
        "checkpoint_std": std,
        **mean_std_info,
        "state_dict_n_tensors": len(state_dict),
        "state_dict_tensor_key_hash": hash_state_dict_keys(state_dict),
    }


def validate_checkpoint_against_config(
    ckpt_meta: Dict[str, Any],
    config: Dict[str, Any],
    *,
    image_size: int,
    strict: bool,
) -> List[str]:
    warnings_out: List[str] = []
    expected = expected_values_from_config(config)

    def handle(msg: str) -> None:
        if strict:
            raise RuntimeError(msg)
        warnings_out.append(msg)

    if ckpt_meta["image_size"] is not None and int(ckpt_meta["image_size"]) != int(image_size):
        handle(f"Checkpoint image_size={ckpt_meta['image_size']} but requested image_size={image_size}.")
    if int(expected["image_size"]) != int(image_size):
        handle(f"Config expected image_size={expected['image_size']} but requested image_size={image_size}.")
    if ckpt_meta["channels"] is not None and int(ckpt_meta["channels"]) != 1:
        handle(f"Expected single-channel model, checkpoint channels={ckpt_meta['channels']}.")
    stored = ckpt_meta.get("stored_architecture_hash")
    if stored and str(stored) != str(expected["architecture_hash"]):
        handle(f"Architecture hash mismatch: checkpoint={stored}, config={expected['architecture_hash']}.")
    return warnings_out


def calculate_num_workers(requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, int(requested))
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    if cpu_count <= 4:
        return max(1, cpu_count - 1)
    return max(1, cpu_count - 2)


# -----------------------------------------------------------------------------
# Pixel operations (numerically frozen)
# -----------------------------------------------------------------------------

def get_tiff_page_metadata(image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with tifffile.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]
        image = tif.asarray()
        bits_per_sample = getattr(page, "bitspersample", None)
        if isinstance(bits_per_sample, (tuple, list)):
            container_bit_depth = int(bits_per_sample[0]) if bits_per_sample else None
        else:
            container_bit_depth = int(bits_per_sample) if bits_per_sample is not None else None
        samples_per_pixel = getattr(page, "samplesperpixel", None)
        samples_per_pixel = int(samples_per_pixel) if samples_per_pixel is not None else None
    return image, {
        "container_bit_depth": container_bit_depth,
        "samples_per_pixel": samples_per_pixel,
        "dtype": str(image.dtype),
        "shape": tuple(int(v) for v in image.shape),
    }


def is_single_channel_image(image: np.ndarray, samples_per_pixel: Optional[int]) -> bool:
    if image.ndim == 2:
        return True
    if samples_per_pixel == 1 and image.ndim in (2, 3) and 1 in image.shape:
        return True
    return image.ndim == 3 and 1 in image.shape


def squeeze_singleton_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    squeezed = np.squeeze(image)
    return squeezed if squeezed.ndim == 2 else image


def infer_effective_bit_depth(max_pixel: int, container_bit_depth: Optional[int], expected_effective_bit_depth: int) -> int:
    """Smallest supported depth containing max_pixel, with explicit 14-in-16 handling."""
    thresholds = {8: 255, 12: 4095, 14: 16383, 16: 65535}
    if expected_effective_bit_depth not in thresholds:
        raise ValueError(f"Unsupported expected bit depth: {expected_effective_bit_depth}")
    if container_bit_depth == expected_effective_bit_depth and max_pixel <= thresholds[expected_effective_bit_depth]:
        return expected_effective_bit_depth
    if expected_effective_bit_depth == 14 and container_bit_depth == 16 and max_pixel <= thresholds[14]:
        return 14
    for bit_depth in (8, 12, 14, 16):
        if max_pixel <= thresholds[bit_depth]:
            return bit_depth
    return 16


def normalize_linear_raw(image: np.ndarray, inferred_bit_depth: int) -> Tuple[np.ndarray, int]:
    """I_norm = clip(I_raw / (2^bit_depth - 1), 0, 1) as float32."""
    if inferred_bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(f"Unsupported inferred bit depth: {inferred_bit_depth}")
    denominator = (2 ** int(inferred_bit_depth)) - 1
    x = image.astype(np.float32, copy=False) / float(denominator)
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False), denominator


def clip_unit_interval(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def resize_to_square(image: np.ndarray, image_size: int, *, post_resize_clip: bool = True) -> np.ndarray:
    """Pixel-locked resize with cv2.INTER_LANCZOS4 (same kernel as training)."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image before resize, got {image.shape}")
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Non-square image {image.shape}; refusing aspect-distorting square resize (O-5).")
    resized = cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_LANCZOS4)
    resized = resized.astype(np.float32, copy=False)
    if post_resize_clip:
        resized = clip_unit_interval(resized)
    return resized


def photometric_mean_std_match_to_checkpoint(
    image: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    min_source_std: float = DEFAULT_PHOTOMETRIC_MIN_SOURCE_STD,
    post_harmonization_clip: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Post hoc photometric sensitivity transform:
        y = ((x - mean(x)) / std(x)) * checkpoint_std + checkpoint_mean, clipped to [0, 1].

    Uses unlabeled per-image test statistics. The checkpoint std may be a pooled
    dataset-level statistic (PARTIAL: training statistics code not inspected),
    in which case matching each image to it can over-contrast images relative
    to typical training images. Diagnostic only.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image before photometric matching, got {image.shape}")
    if len(mean) != 1 or len(std) != 1:
        raise ValueError(f"Expected one-channel checkpoint mean/std, got mean={mean}, std={std}")
    target_mean, target_std = float(mean[0]), float(std[0])
    if target_std <= 0 or not np.isfinite(target_std):
        raise ValueError(f"Checkpoint target std must be positive and finite, got {target_std}")
    x = image.astype(np.float32, copy=False)
    source_mean, source_std = float(np.mean(x)), float(np.std(x))
    if source_std < float(min_source_std) or not np.isfinite(source_std):
        raise ValueError(
            f"Per-image source std is too small for stable photometric matching: "
            f"source_std={source_std}, min_source_std={min_source_std}"
        )
    y = (x - source_mean) / source_std * target_std + target_mean
    if post_harmonization_clip:
        y = clip_unit_interval(y)
    return y.astype(np.float32, copy=False), {
        "source_mean_before": source_mean,
        "source_std_before": source_std,
        "output_mean_after": float(np.mean(y)),
        "output_std_after": float(np.std(y)),
    }


def standardize_with_checkpoint_stats(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    if len(mean) != 1 or len(std) != 1:
        raise ValueError(f"This pipeline expects one-channel checkpoint mean/std; got mean={mean}, std={std}")
    if float(std[0]) <= 0:
        raise ValueError(f"Checkpoint std must be positive, got {std[0]}")
    return ((image - float(mean[0])) / float(std[0])).astype(np.float32, copy=False)


def output_path_for_row(row: Dict[str, str], output_dir: Path) -> Path:
    return output_dir / Path(row["relative_path"]).with_suffix(".npy")


def input_path_for_row(row: Dict[str, str], raw_root: Optional[str]) -> Path:
    return Path(raw_root) / Path(row["relative_path"]) if raw_root else Path(row["image_path"])


def atomic_save_npy(array: np.ndarray, target_path: Path) -> None:
    """Write a .npy file atomically to avoid partially written arrays after interruption."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target_path.stem}.", suffix=".tmp.npy", dir=str(target_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(str(tmp_path), array.astype(np.float32, copy=False))
        os.replace(str(tmp_path), str(target_path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Fingerprint (F-08)
# -----------------------------------------------------------------------------

def compute_fingerprint(params: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return {"fingerprint_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "parameters": params}


def check_existing_outputs(output_dir: Path, fingerprint: Dict[str, Any], *, force: bool) -> str:
    """
    Decide whether existing outputs may be reused. Returns "fresh", "reuse" or
    "force". Raises when outputs exist but were produced with different (or
    unrecorded) parameters and --force was not given.
    """
    fp_path = output_dir / FINGERPRINT_FILENAME
    has_arrays = output_dir.exists() and any(output_dir.rglob("*.npy"))
    if force:
        return "force"
    if not has_arrays:
        return "fresh"
    if not fp_path.exists():
        raise RuntimeError(
            f"{output_dir} contains .npy files without {FINGERPRINT_FILENAME}; their parameters are unknown. "
            "Re-run with --force or use a new output directory."
        )
    with fp_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    if existing.get("fingerprint_sha256") != fingerprint["fingerprint_sha256"]:
        raise RuntimeError(
            f"Existing outputs in {output_dir} were produced with different preprocessing parameters "
            f"(stored {existing.get('fingerprint_sha256')}, requested {fingerprint['fingerprint_sha256']}). "
            "Re-run with --force or use a new output directory."
        )
    return "reuse"


# -----------------------------------------------------------------------------
# Per-image worker
# -----------------------------------------------------------------------------

def preprocess_one(task: Dict[str, Any]) -> Dict[str, Any]:
    row = task["row"]
    output_dir = Path(task["output_dir"])
    image_size = int(task["image_size"])
    mode = str(task["mode"])
    image_path = input_path_for_row(row, task.get("raw_root"))
    target_path = output_path_for_row(row, output_dir)
    result_base = {
        "relative_path": row.get("relative_path", ""),
        "image_path": str(image_path),
        "target_path": str(target_path),
        "slide_id": row.get("slide_id", ""),
        "species_name": row.get("species_name", ""),
    }
    try:
        if image_path.suffix not in SUPPORTED_TIFF_EXTENSIONS:
            raise ValueError(f"Unsupported input extension: {image_path.suffix}")

        if target_path.exists() and bool(task["reuse_existing"]):
            arr = np.load(str(target_path), mmap_mode="r")
            if arr.shape != (image_size, image_size) or arr.dtype != np.float32:
                raise ValueError(f"Existing array {target_path} has shape {arr.shape}, dtype {arr.dtype}.")
            return {**result_base, "success": True, "skipped_existing": True}

        image, tiff_meta = get_tiff_page_metadata(image_path)
        if not is_single_channel_image(image, tiff_meta.get("samples_per_pixel")):
            raise ValueError(f"Expected single-channel TIFF, got shape={image.shape}")
        image = squeeze_singleton_channel(image)
        if image.ndim != 2 or image.size == 0:
            raise ValueError(f"Expected non-empty 2D grayscale TIFF after squeeze, got {image.shape}")
        expected_shape = task.get("expected_stored_shape")
        if expected_shape is not None and tuple(image.shape) != tuple(expected_shape):
            raise ValueError(f"O-5: image shape {tuple(image.shape)} != optics stored_image_px {tuple(expected_shape)}")

        min_pixel, max_pixel = int(np.min(image)), int(np.max(image))
        expected_bd = int(task["expected_effective_bit_depth"])
        inferred_bd = infer_effective_bit_depth(max_pixel, tiff_meta.get("container_bit_depth"), expected_bd)
        if inferred_bd != expected_bd and not bool(task["allow_effective_bit_depth_fallback"]):
            raise ValueError(
                f"Inferred effective bit depth {inferred_bd} differs from expected {expected_bd}. "
                "Use --allow-effective-bit-depth-fallback only for diagnostic runs."
            )

        x, denominator = normalize_linear_raw(image, inferred_bd)
        x = resize_to_square(x, image_size, post_resize_clip=bool(task["post_resize_clip"]))
        photometric_metadata = None
        if mode == PHOTOMETRIC_MODE:
            x, photometric_metadata = photometric_mean_std_match_to_checkpoint(
                x,
                task["checkpoint_mean"],
                task["checkpoint_std"],
                min_source_std=float(task["photometric_min_source_std"]),
                post_harmonization_clip=bool(task["post_resize_clip"]),
            )
        elif mode != PRIMARY_MODE:
            raise ValueError(f"Unsupported preprocessing mode: {mode}")

        if bool(task["save_standardized"]):
            x = standardize_with_checkpoint_stats(x, task["checkpoint_mean"], task["checkpoint_std"])
        if not np.all(np.isfinite(x)):
            raise ValueError("Preprocessed image contains non-finite values.")
        atomic_save_npy(x, target_path)
        return {
            **result_base,
            "success": True,
            "skipped_existing": False,
            "inferred_effective_bit_depth": int(inferred_bd),
            "normalization_denominator": int(denominator),
            "container_bit_depth": tiff_meta.get("container_bit_depth"),
            "min_pixel": min_pixel,
            "max_pixel": max_pixel,
            "output_min": float(np.min(x)),
            "output_max": float(np.max(x)),
            "photometric_metadata": photometric_metadata,
        }
    except Exception as exc:
        return {**result_base, "success": False, "error": str(exc)}


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

def write_failures_csv(failures: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for item in failures:
            writer.writerow({k: item.get(k, "") for k in FAILURE_COLUMNS})


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def summarize_results(
    *,
    manifest_rows: List[Dict[str, str]],
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    ckpt_meta: Dict[str, Any],
    checkpoint_warnings: List[str],
    elapsed_seconds: float,
    fingerprint: Dict[str, Any],
    reuse_decision: str,
    optics_record: Optional[Dict[str, Any]],
    settings_resolution: Dict[str, Any],
) -> Dict[str, Any]:
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    processed = [r for r in successes if not r.get("skipped_existing")]
    skipped = [r for r in successes if r.get("skipped_existing")]

    slide_ids = sorted({row["slide_id"] for row in manifest_rows})
    first_row = {sid: next(r for r in manifest_rows if r["slide_id"] == sid) for sid in slide_ids}
    species_slide_counts = Counter(first_row[s]["species_name"] for s in slide_ids)
    hyphae_slide_counts = Counter(first_row[s]["hyphae_status"] for s in slide_ids)

    def counts(key: str) -> Dict[str, int]:
        return dict(sorted(Counter(str(r.get(key)) for r in processed if r.get(key) is not None).items()))

    out_mins = np.array([float(r["output_min"]) for r in processed if r.get("output_min") is not None], dtype=float)
    out_maxs = np.array([float(r["output_max"]) for r in processed if r.get("output_max") is not None], dtype=float)
    phot = [r["photometric_metadata"] for r in processed if r.get("photometric_metadata")]

    def mean_of(key: str) -> Optional[float]:
        vals = [float(m[key]) for m in phot if key in m]
        return float(np.mean(vals)) if vals else None

    optics_reference = optics_reference_summary(optics_record) if optics_record else None
    decimation = None
    if optics_reference and optics_reference.get("external_decimation_factor"):
        decimation = {
            "external": optics_reference["external_decimation_factor"],
            "training": optics_reference.get("training_decimation_factor"),
        }

    is_primary = args.mode == PRIMARY_MODE
    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "script": "preprocess_external.py",
        "created_unix_time": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "preprocessing_mode": args.mode,
        "primary_or_sensitivity": "primary" if is_primary else "sensitivity",
        "analysis_role": "primary_external_validation" if is_primary else "post_hoc_photometric_mean_std_sensitivity_analysis",
        "scientific_role": (
            "Deterministic external preprocessing for frozen-model inference. No training, fine-tuning, "
            "threshold tuning, or preprocessing selection is performed here."
        ),
        "photometric_harmonization": None if is_primary else {
            "scientific_role": "post_hoc_sensitivity_not_primary",
            "method": "per_image_mean_std_match_to_checkpoint_training_statistics",
            "target_mean": float(ckpt_meta["checkpoint_mean"][0]),
            "target_std": float(ckpt_meta["checkpoint_std"][0]),
            "min_source_std": float(args.photometric_min_source_std),
            "n_images_harmonized": len(phot),
            "source_mean_before_mean": mean_of("source_mean_before"),
            "source_std_before_mean": mean_of("source_std_before"),
            "output_mean_after_mean": mean_of("output_mean_after"),
            "output_std_after_mean": mean_of("output_std_after"),
            "post_harmonization_clip": bool(args.post_resize_clip),
            "target_statistic_caveat": (
                "Checkpoint std may be a pooled dataset-level statistic; per-image matching to it can "
                "over-contrast images relative to typical training images (PARTIAL: training statistics "
                "computation not inspected)."
            ),
            "interpretation_guardrail": (
                "Diagnostic post hoc photometric sensitivity only. This transformation uses external "
                "test-image intensity statistics and must not replace the primary locked result."
            ),
        },
        "input_manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "checkpoint": ckpt_meta,
        "checkpoint_validation_warnings": checkpoint_warnings,
        "config_path": str(args.config) if args.config else None,
        "metadata_path": str(args.metadata) if args.metadata else None,
        "settings_resolution": settings_resolution,
        "image_size": int(args.image_size),
        "save_standardized": bool(args.save_standardized),
        "recommended_downstream_standardization": (
            "Apply checkpoint mean/std in the PyTorch Dataset before model inference."
            if not args.save_standardized
            else "Arrays are already checkpoint-standardized; do not standardize again."
        ),
        "raw_bit_depth_policy": {
            "expected_external_effective_bit_depth": int(args.expected_effective_bit_depth),
            "allow_effective_bit_depth_fallback": bool(args.allow_effective_bit_depth_fallback),
            "normalization_rule": "I_norm = clip(I_raw / (2^inferred_effective_bit_depth - 1), 0, 1)",
            "expected_primary_denominator_for_14bit": 16383,
            "checkpoint_training_bit_depth_is_not_used_for_external_raw_scaling": True,
        },
        "spatial_policy": {
            "mode": "pixel_locked_resize_to_model_input",
            "resize_interpolation": RESIZE_INTERPOLATION_NAME,
            "antialiasing": False,
            "training_resize_kernel": TRAINING_RESIZE_KERNEL,
            "interpolation_kernel_matches_training": True,
            "decimation_factor": decimation,
            "post_resize_clip_to_unit_interval": bool(args.post_resize_clip),
            "physical_scale_resampling_applied": False,
            "limitation": (
                "Kernel matches training, but the decimation factor differs (external 5.875 vs training 3.125 "
                "for the real cohorts) and LANCZOS4 is not antialiased, so aliasing/noise texture can differ "
                "between domains. Binning mode of the external camera is unknown (O-6)."
            ),
        },
        "optics_reference": optics_reference,
        "fingerprint": fingerprint,
        "reuse_decision": reuse_decision,
        "counts": {
            "n_manifest_images": len(manifest_rows),
            "n_success": len(successes),
            "n_processed": len(processed),
            "n_skipped_existing": len(skipped),
            "n_failed": len(failures),
            "n_slides": len(slide_ids),
            "species_slide_counts": dict(sorted(species_slide_counts.items())),
            "hyphae_slide_counts": dict(sorted(hyphae_slide_counts.items())),
            "inferred_effective_bit_depth_counts_processed": counts("inferred_effective_bit_depth"),
            "normalization_denominator_counts_processed": counts("normalization_denominator"),
            "container_bit_depth_counts_processed": counts("container_bit_depth"),
        },
        "output_value_summary_processed": {
            "min_of_output_min": float(np.min(out_mins)) if out_mins.size else None,
            "max_of_output_max": float(np.max(out_maxs)) if out_maxs.size else None,
            "expected_range_if_not_standardized": [0.0, 1.0] if not args.save_standardized else None,
        },
        "failures": failures,
        "failure_csv": str(args.output_dir / "preprocessing_failures.csv"),
        "scientific_guardrails": {
            "no_retraining": True,
            "no_fine_tuning": True,
            "no_threshold_tuning": True,
            "external_dataset_not_split": True,
            "primary_statistical_unit_for_downstream_evaluation": "slide",
        },
    }


def run_preprocessing(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], float]:
    manifest_rows = read_manifest(args.manifest)
    output_dir: Path = args.output_dir
    config = load_optional_yaml(args.config)

    settings_resolution: Dict[str, Any] = {}
    args.expected_effective_bit_depth = int(resolve_setting(
        "expected_effective_bit_depth", args.expected_effective_bit_depth, config,
        ["tiff_audit", "expected_external_effective_bit_depth"], DEFAULT_EXPECTED_EFFECTIVE_BIT_DEPTH,
        settings_resolution,
    ))

    ckpt_meta = load_checkpoint_metadata(args.checkpoint, config, strict=args.strict)
    checkpoint_warnings = validate_checkpoint_against_config(ckpt_meta, config, image_size=args.image_size, strict=args.strict)

    optics_record = None
    expected_stored_shape = None
    if args.metadata is not None:
        with args.metadata.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if isinstance(metadata.get("optics"), dict):
            optics_record = optics_provenance_record(metadata["optics"])
            stored = (metadata["optics"].get("external") or {}).get("stored_image_px")
            if isinstance(stored, list) and len(stored) == 2:
                expected_stored_shape = (int(stored[0]), int(stored[1]))
        elif args.strict:
            raise ValueError(f"Metadata {args.metadata} has no 'optics' block (schema 2.0 required).")

    fingerprint = compute_fingerprint({
        "mode": args.mode,
        "image_size": int(args.image_size),
        "expected_effective_bit_depth": int(args.expected_effective_bit_depth),
        "allow_effective_bit_depth_fallback": bool(args.allow_effective_bit_depth_fallback),
        "post_resize_clip": bool(args.post_resize_clip),
        "save_standardized": bool(args.save_standardized),
        "photometric_min_source_std": float(args.photometric_min_source_std),
        "checkpoint_mean": ckpt_meta["checkpoint_mean"],
        "checkpoint_std": ckpt_meta["checkpoint_std"],
        "resize_interpolation": RESIZE_INTERPOLATION_NAME,
        "script_sha256": sha256_file(Path(__file__)),
        "optics_sha256": (optics_record or {}).get("optics_sha256"),
    })
    reuse_decision = check_existing_outputs(output_dir, fingerprint, force=bool(args.force))
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "row": row,
            "output_dir": str(output_dir),
            "image_size": int(args.image_size),
            "mode": args.mode,
            "expected_effective_bit_depth": int(args.expected_effective_bit_depth),
            "allow_effective_bit_depth_fallback": bool(args.allow_effective_bit_depth_fallback),
            "save_standardized": bool(args.save_standardized),
            "checkpoint_mean": ckpt_meta["checkpoint_mean"],
            "checkpoint_std": ckpt_meta["checkpoint_std"],
            "reuse_existing": reuse_decision == "reuse",
            "post_resize_clip": bool(args.post_resize_clip),
            "photometric_min_source_std": float(args.photometric_min_source_std),
            "raw_root": str(args.raw_root) if args.raw_root else None,
            "expected_stored_shape": expected_stored_shape,
        }
        for row in manifest_rows
    ]

    num_workers = calculate_num_workers(args.num_workers)
    start = time.time()
    results: List[Dict[str, Any]] = []
    if num_workers == 1:
        for i, task in enumerate(tasks, start=1):
            results.append(preprocess_one(task))
            if args.progress_every and i % int(args.progress_every) == 0:
                print(f"Preprocessed {i}/{len(tasks)} images...")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(preprocess_one, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                if args.progress_every and i % int(args.progress_every) == 0:
                    print(f"Preprocessed {i}/{len(tasks)} images...")
    results.sort(key=lambda r: r.get("relative_path", ""))
    elapsed = time.time() - start

    failures = [r for r in results if not r.get("success")]
    failures_path = output_dir / "preprocessing_failures.csv"
    write_failures_csv(failures, failures_path)
    fingerprint_path = output_dir / FINGERPRINT_FILENAME
    if not failures:
        write_json(fingerprint, fingerprint_path)

    metadata_out = summarize_results(
        manifest_rows=manifest_rows,
        results=results,
        args=args,
        ckpt_meta=ckpt_meta,
        checkpoint_warnings=checkpoint_warnings,
        elapsed_seconds=elapsed,
        fingerprint=fingerprint,
        reuse_decision=reuse_decision,
        optics_record=optics_record,
        settings_resolution=settings_resolution,
    )
    metadata_path = output_dir / args.metadata_filename
    write_json(metadata_out, metadata_path)

    if args.sidecar_output_dir is not None:
        args.sidecar_output_dir.mkdir(parents=True, exist_ok=True)
        for path in (metadata_path, failures_path, fingerprint_path):
            if path.exists():
                shutil.copy2(path, args.sidecar_output_dir / path.name)

    if failures and args.strict:
        raise RuntimeError(f"{len(failures)} images failed preprocessing. See {failures_path}")
    return results, elapsed


def print_summary(args: argparse.Namespace, results: List[Dict[str, Any]], elapsed: float) -> None:
    successes = [r for r in results if r.get("success")]
    print("\n" + "=" * 80)
    print("EXTERNAL PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Mode              : {args.mode}")
    print(f"Output directory  : {args.output_dir}")
    print(f"Image size        : {args.image_size}")
    print(f"Expected bit depth: {args.expected_effective_bit_depth}")
    print(f"Resize kernel     : {RESIZE_INTERPOLATION_NAME} (matches training)")
    print(f"Processed         : {sum(1 for r in successes if not r.get('skipped_existing'))}")
    print(f"Reused (fingerprint match): {sum(1 for r in successes if r.get('skipped_existing'))}")
    print(f"Failed            : {len(results) - len(successes)}")
    print(f"Elapsed seconds   : {elapsed:.2f}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()
    results, elapsed = run_preprocessing(args)
    print_summary(args, results, elapsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
