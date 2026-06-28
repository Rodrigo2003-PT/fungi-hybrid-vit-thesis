#!/usr/bin/env python3
"""
preprocess_external.py

Deterministically preprocess the external new-microscope validation images for
frozen-model inference with the final LSA-ConvTok Hybrid model.

Scientific role
---------------
The external validation cohort was acquired on a different microscope system
than the training data. The final Hybrid model was trained on the original
12-bit microscope domain, while the external cohort is expected to contain
14-bit brightfield monochrome TIFFs, commonly stored in a 16-bit TIFF container.

This script implements the primary locked external preprocessing mode:

    locked_pixel_pipeline:
        Raw external TIFF
          -> infer/verify effective external bit depth, expected 14-bit
          -> linear normalization to [0, 1] using the inferred raw bit depth
          -> resize to the frozen model input size, expected 384 x 384
          -> save float32 .npy files mirroring manifest relative paths

Important:
    - This script does not train, fine-tune, or tune anything.
    - It does not fit BaSiCPy in the primary mode.
    - It does not recompute model mean/std on the external cohort.
    - The checkpoint mean/std are recorded for downstream dataset
      standardization, but this script saves pre-standardization float32 arrays
      in [0, 1] by default.

Why not normalize external data as 12-bit?
------------------------------------------
The checkpoint bit_depth=12 describes the training-domain acquisition. It must
not force the external raw normalization denominator. If external files are true
14-bit images, using 4095 would clip or distort intensity values. The expected
primary raw normalization denominator is therefore:

    2^14 - 1 = 16383

when TIFF audit and pixel values support effective 14-bit data.

Usage
-----
    python preprocess_external.py \\
        --manifest outputs_external/manifest/external_manifest.csv \\
        --checkpoint final_model_lsa_convtok.pth \\
        --config configs/external_validation_config.yaml \\
        --output-dir outputs_external/preprocessed/locked_pixel_pipeline \\
        --mode locked_pixel_pipeline \\
        --image-size 384

Outputs
-------
    outputs_external/preprocessed/locked_pixel_pipeline/
        Candida albicans/
            20260508_alb_C_1.npy
            ...
        Candida glabrata/
            20260507_gla_A_1.npy
            ...
        preprocessing_metadata.json
        preprocessing_failures.csv

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tifffile
except ImportError as exc:
    raise ImportError(
        "tifffile is required for preprocess_external.py. "
        "Install it with: pip install tifffile"
    ) from exc

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "opencv-python is required for preprocess_external.py. "
        "Install it with: pip install opencv-python"
    ) from exc

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "torch is required to inspect the model checkpoint metadata. "
        "Install PyTorch before running this script."
    ) from exc


SUPPORTED_TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}
SUPPORTED_BIT_DEPTHS = {8, 12, 14, 16}
DEFAULT_MODE = "locked_pixel_pipeline"
DEFAULT_EXPECTED_EFFECTIVE_BIT_DEPTH = 14
DEFAULT_IMAGE_SIZE = 384
DEFAULT_CHECKPOINT_MEAN = [0.23691008985042572]
DEFAULT_CHECKPOINT_STD = [0.007409718818962574]


FAILURE_COLUMNS = [
    "relative_path",
    "image_path",
    "slide_id",
    "species_name",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess external-validation TIFF images for deterministic "
            "frozen-model inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to external_manifest.csv produced by external_manifest.py.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to final_model_lsa_convtok.pth primary inference checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional external_validation_config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for preprocessed .npy outputs and metadata.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["locked_pixel_pipeline", "scale_matched_physical_pipeline", "photometric_mean_std_matched_pipeline"],
        help=(
            "Preprocessing mode. locked_pixel_pipeline is the primary analysis; "
            "scale_matched_physical_pipeline is a post hoc physical-scale sensitivity analysis; photometric_mean_std_matched_pipeline is a post hoc photometric sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Frozen model input image size. Expected 384.",
    )
    parser.add_argument(
        "--expected-effective-bit-depth",
        type=int,
        default=DEFAULT_EXPECTED_EFFECTIVE_BIT_DEPTH,
        choices=[8, 12, 14, 16],
        help="Expected external effective bit depth.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default uses a conservative CPU heuristic.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess images even if target .npy files already exist.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, abort if any image fails preprocessing or if severe "
            "checkpoint/config incompatibility is detected."
        ),
    )
    parser.add_argument(
        "--allow-effective-bit-depth-fallback",
        action="store_true",
        help=(
            "Allow per-image inferred bit depth to differ from the expected "
            "external bit depth. This is usually only for diagnostic runs."
        ),
    )
    parser.add_argument(
        "--save-standardized",
        action="store_true",
        help=(
            "If set, save arrays after checkpoint mean/std standardization "
            "instead of [0,1] arrays. The recommended primary pipeline is false; "
            "standardization should usually occur in the PyTorch Dataset."
        ),
    )
    parser.add_argument(
        "--post-resize-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clip pre-standardized arrays to [0,1] after interpolation/resizing/harmonization. "
            "This is scientifically required for normalized raw intensities because Lanczos "
            "interpolation can overshoot slightly outside the physical range. Use --no-post-resize-clip "
            "only for diagnostic replication of older runs."
        ),
    )
    parser.add_argument(
        "--photometric-min-source-std",
        type=float,
        default=1e-6,
        help=(
            "Minimum per-image source standard deviation allowed in the photometric mean/std "
            "matching sensitivity pipeline. Images below this threshold are rejected in strict runs "
            "because mean/std matching would be numerically unstable."
        ),
    )

    parser.add_argument(
        "--training-pixel-spacing-um",
        type=float,
        default=None,
        help=(
            "Training-domain pixel spacing in µm/px. Defaults to config "
            "training_domain_reference.pixel_spacing_um.x when available."
        ),
    )
    parser.add_argument(
        "--external-pixel-spacing-um",
        type=float,
        default=None,
        help=(
            "External-domain pixel spacing in µm/px. Defaults to config "
            "external_acquisition_metadata.pixel_spacing_um.x or "
            "external_domain.pixel_spacing_um.x when available."
        ),
    )
    parser.add_argument(
        "--scale-match-padding",
        type=str,
        default="median",
        choices=["median", "edge", "reflect", "zero"],
        help=(
            "Padding policy used after physical scale matching. Median is the "
            "recommended default for sparse brightfield images because it approximates background."
        ),
    )
    parser.add_argument(
        "--scale-match-target-canvas-height-px",
        type=int,
        default=None,
        help=(
            "Target raw canvas height after scale matching. For the final thesis sensitivity "
            "analysis use the training raw height, 1200 px. If omitted, the external image height "
            "is used and the metadata will mark the analysis as less interpretable."
        ),
    )
    parser.add_argument(
        "--scale-match-target-canvas-width-px",
        type=int,
        default=None,
        help=(
            "Target raw canvas width after scale matching. For the final thesis sensitivity "
            "analysis use the training raw width, 1200 px. If omitted, the external image width "
            "is used and the metadata will mark the analysis as less interpretable."
        ),
    )

    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="preprocessing_metadata.json",
        help="Name of metadata JSON written inside output directory.",
    )
    parser.add_argument(
        "--sidecar-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional Drive-safe directory where only lightweight preprocessing "
            "sidecar files are copied: metadata JSON and failures CSV. The .npy "
            "arrays remain only in --output-dir. In Colab, set --output-dir to "
            "/content/... and --sidecar-output-dir to your Drive outputs directory."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print preprocessing progress every N images. Set 0 to disable progress messages.",
    )
    return parser.parse_args()


def load_optional_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")

    try:
        import yaml
    except ImportError:
        print(
            "WARNING: PyYAML not installed; config YAML will not be parsed. "
            "Install with: pip install pyyaml",
            file=sys.stderr,
        )
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def get_nested(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    required = {
        "image_path",
        "relative_path",
        "filename",
        "slide_id",
        "species_name",
        "label",
        "hyphae_status",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    return rows


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dictionary: {checkpoint_path}")

    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None or not isinstance(model_state_dict, dict):
        raise ValueError(
            f"Checkpoint does not contain a valid model_state_dict: {checkpoint_path}"
        )

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    arch_metadata = checkpoint.get("arch_metadata")
    if arch_metadata is None and isinstance(metadata, dict):
        arch_metadata = metadata.get("arch_metadata")
    if not isinstance(arch_metadata, dict):
        arch_metadata = {}

    checkpoint_mean = (
        checkpoint.get("mean")
        or metadata.get("mean")
        or config.get("mean")
        or DEFAULT_CHECKPOINT_MEAN
    )
    checkpoint_std = (
        checkpoint.get("std")
        or metadata.get("std")
        or config.get("std")
        or DEFAULT_CHECKPOINT_STD
    )

    if isinstance(checkpoint_mean, np.ndarray):
        checkpoint_mean = checkpoint_mean.tolist()
    if isinstance(checkpoint_std, np.ndarray):
        checkpoint_std = checkpoint_std.tolist()

    if not isinstance(checkpoint_mean, list):
        checkpoint_mean = [float(checkpoint_mean)]
    if not isinstance(checkpoint_std, list):
        checkpoint_std = [float(checkpoint_std)]

    out = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "model_variant": (
            checkpoint.get("model_variant")
            or metadata.get("model_variant")
            or arch_metadata.get("model_class")
            or "unknown"
        ),
        "experiment_name": (
            checkpoint.get("experiment_name")
            or metadata.get("experiment_name")
            or arch_metadata.get("experiment_name")
            or config.get("experiment_name")
            or "unknown"
        ),
        "architecture_hash": (
            checkpoint.get("architecture_hash")
            or metadata.get("architecture_hash")
            or arch_metadata.get("arch_hash")
            or "unknown"
        ),
        "image_size": (
            checkpoint.get("image_size")
            or metadata.get("image_size")
            or arch_metadata.get("image_size")
            or config.get("image_size")
        ),
        "channels": (
            checkpoint.get("channels")
            or metadata.get("channels")
            or arch_metadata.get("channels")
            or config.get("channels")
        ),
        "checkpoint_training_bit_depth": (
            checkpoint.get("bit_depth")
            or metadata.get("bit_depth")
            or config.get("bit_depth")
        ),
        "checkpoint_mean": [float(v) for v in checkpoint_mean],
        "checkpoint_std": [float(v) for v in checkpoint_std],
        "state_dict_n_tensors": len(model_state_dict),
        "state_dict_tensor_key_hash": hash_state_dict_keys(model_state_dict),
    }

    return out


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_state_dict_keys(state_dict: Dict[str, Any]) -> str:
    keys = sorted(str(k) for k in state_dict.keys())
    text = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def validate_checkpoint_against_config(
    ckpt_meta: Dict[str, Any],
    config: Dict[str, Any],
    *,
    image_size: int,
    strict: bool,
) -> List[str]:
    warnings_out: List[str] = []

    expected_model = get_nested(config, ["checkpoints", "expected_model"], {})
    expected_hash = expected_model.get("architecture_hash") if isinstance(expected_model, dict) else None
    expected_channels = expected_model.get("channels") if isinstance(expected_model, dict) else None
    expected_image_size = expected_model.get("image_size") if isinstance(expected_model, dict) else None

    if expected_hash and ckpt_meta["architecture_hash"] != "unknown":
        if ckpt_meta["architecture_hash"] != expected_hash:
            msg = (
                f"Architecture hash mismatch: checkpoint={ckpt_meta['architecture_hash']}, "
                f"config={expected_hash}."
            )
            if strict:
                raise RuntimeError(msg)
            warnings_out.append(msg)

    if ckpt_meta["image_size"] is not None:
        if int(ckpt_meta["image_size"]) != int(image_size):
            msg = (
                f"Checkpoint image_size={ckpt_meta['image_size']} but requested "
                f"image_size={image_size}."
            )
            if strict:
                raise RuntimeError(msg)
            warnings_out.append(msg)

    if expected_image_size is not None and int(expected_image_size) != int(image_size):
        msg = (
            f"Config expected image_size={expected_image_size} but requested "
            f"image_size={image_size}."
        )
        if strict:
            raise RuntimeError(msg)
        warnings_out.append(msg)

    if ckpt_meta["channels"] is not None and int(ckpt_meta["channels"]) != 1:
        msg = f"Expected single-channel model, checkpoint channels={ckpt_meta['channels']}."
        if strict:
            raise RuntimeError(msg)
        warnings_out.append(msg)

    if expected_channels is not None and int(expected_channels) != 1:
        msg = f"Expected single-channel config, got channels={expected_channels}."
        if strict:
            raise RuntimeError(msg)
        warnings_out.append(msg)

    for std in ckpt_meta["checkpoint_std"]:
        if std <= 0:
            msg = f"Invalid checkpoint std value: {std}"
            if strict:
                raise RuntimeError(msg)
            warnings_out.append(msg)

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


def get_tiff_page_metadata(image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with tifffile.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]
        image = tif.asarray()

        bits_per_sample = getattr(page, "bitspersample", None)
        if isinstance(bits_per_sample, (tuple, list)):
            container_bit_depth = int(bits_per_sample[0]) if bits_per_sample else None
        else:
            container_bit_depth = int(bits_per_sample) if bits_per_sample is not None else None

        photometric = str(getattr(page, "photometric", "unknown"))
        samples_per_pixel = getattr(page, "samplesperpixel", None)
        samples_per_pixel = int(samples_per_pixel) if samples_per_pixel is not None else None

    return image, {
        "container_bit_depth": container_bit_depth,
        "photometric": photometric,
        "samples_per_pixel": samples_per_pixel,
        "dtype": str(image.dtype),
        "shape": tuple(int(v) for v in image.shape),
    }


def is_single_channel_image(image: np.ndarray, samples_per_pixel: Optional[int]) -> bool:
    if image.ndim == 2:
        return True
    if samples_per_pixel == 1 and image.ndim in (2, 3) and 1 in image.shape:
        return True
    if image.ndim == 3 and 1 in image.shape:
        return True
    return False


def squeeze_singleton_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    squeezed = np.squeeze(image)
    return squeezed if squeezed.ndim == 2 else image


def infer_effective_bit_depth(
    max_pixel: int,
    container_bit_depth: Optional[int],
    expected_effective_bit_depth: int,
) -> int:
    thresholds = {8: 255, 12: 4095, 14: 16383, 16: 65535}

    if expected_effective_bit_depth not in thresholds:
        raise ValueError(f"Unsupported expected bit depth: {expected_effective_bit_depth}")

    expected_max = thresholds[expected_effective_bit_depth]

    if container_bit_depth == expected_effective_bit_depth and max_pixel <= expected_max:
        return expected_effective_bit_depth

    # Critical external-validation case: 14-bit data in 16-bit TIFF container.
    if (
        expected_effective_bit_depth == 14
        and container_bit_depth == 16
        and max_pixel <= thresholds[14]
    ):
        return 14

    for bit_depth in (8, 12, 14, 16):
        if max_pixel <= thresholds[bit_depth]:
            return bit_depth

    return 16


def normalize_linear_raw(
    image: np.ndarray,
    inferred_bit_depth: int,
) -> Tuple[np.ndarray, int]:
    if inferred_bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(f"Unsupported inferred bit depth: {inferred_bit_depth}")

    denominator = (2**int(inferred_bit_depth)) - 1
    if denominator <= 0:
        raise ValueError(f"Invalid normalization denominator: {denominator}")

    x = image.astype(np.float32, copy=False) / float(denominator)

    # Values outside [0,1] are scientifically meaningful warnings in audit, but
    # preprocessing must produce bounded model inputs. Clip only after the
    # denominator has been recorded in metadata.
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32, copy=False), denominator


def clip_unit_interval(image: np.ndarray) -> np.ndarray:
    """Clip normalized pre-standardization intensities to their physical [0,1] range."""
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def resize_to_square(
    image: np.ndarray,
    image_size: int,
    *,
    post_resize_clip: bool = True,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image before resize, got {image.shape}")

    resized = cv2.resize(
        image,
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_LANCZOS4,
    ).astype(np.float32, copy=False)

    # Lanczos interpolation may overshoot slightly outside [0,1]. For raw-intensity
    # microscopy arrays saved before checkpoint standardization, values above 1 or
    # below 0 are non-physical and can create extreme standardized inputs.
    if post_resize_clip:
        resized = clip_unit_interval(resized)
    return resized.astype(np.float32, copy=False)


def center_crop_or_pad_to_canvas(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    *,
    padding: str = "median",
) -> np.ndarray:
    """
    Center-crop or center-pad a 2D image to a target raw canvas.

    Scientific role for scale matching
    ----------------------------------
    After external content is downsampled from 0.043 µm/px toward the training
    spacing of 0.175 µm/px, it should be embedded in the *training raw canvas*
    when that canvas is known. This preserves the intended physical-scale change
    through the final resize to the model's fixed 384 x 384 input. Padding is a
    known post hoc sensitivity-analysis confound and is therefore quantified in
    preprocessing_metadata.json; it must not be interpreted as a corrected primary
    deployment pipeline.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image before canvas operation, got {image.shape}")
    target_h = int(target_h)
    target_w = int(target_w)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Target canvas must be positive, got {(target_h, target_w)}")

    h, w = image.shape

    # First center-crop if resampled content is larger than the target canvas.
    if h > target_h:
        top = (h - target_h) // 2
        image = image[top:top + target_h, :]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        image = image[:, left:left + target_w]
        w = target_w

    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)

    if pad_top == pad_bottom == pad_left == pad_right == 0:
        return image.astype(np.float32, copy=False)

    if padding == "median":
        pad_value = float(np.median(image)) if image.size else 0.0
        out = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
    elif padding == "zero":
        out = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
    elif padding == "edge":
        out = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    elif padding == "reflect":
        # Reflect requires dimensions > 1. Fall back to edge for degenerate images.
        mode = "reflect" if image.shape[0] > 1 and image.shape[1] > 1 else "edge"
        out = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    else:
        raise ValueError(f"Unsupported scale-match padding policy: {padding}")

    return out.astype(np.float32, copy=False)


def scale_match_to_training_physical_canvas(
    image: np.ndarray,
    *,
    training_pixel_spacing_um: float,
    external_pixel_spacing_um: float,
    target_canvas_height_px: Optional[int],
    target_canvas_width_px: Optional[int],
    padding: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resample external image content toward the training physical pixel spacing.

    The external acquisition has finer sampling (0.043 µm/px) than the training
    acquisition (0.175 µm/px). To represent the same biological length with a
    training-like number of pixels, external content is downsampled by:

        external_pixel_spacing_um / training_pixel_spacing_um.

    If the raw training canvas is known, the resampled content is embedded into
    that canvas before the final resize to 384 x 384. For this project the raw
    training canvas is 1200 x 1200 px. This is a post hoc sensitivity analysis,
    not the primary external-validation preprocessing.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image before scale matching, got {image.shape}")
    training_px = float(training_pixel_spacing_um)
    external_px = float(external_pixel_spacing_um)
    if training_px <= 0 or external_px <= 0:
        raise ValueError(
            f"Pixel spacings must be positive, got training={training_px}, external={external_px}"
        )

    content_scale = external_px / training_px
    if not np.isfinite(content_scale) or content_scale <= 0:
        raise ValueError(f"Invalid scale-matching content scale: {content_scale}")

    original_h, original_w = image.shape
    scaled_h = max(1, int(round(original_h * content_scale)))
    scaled_w = max(1, int(round(original_w * content_scale)))

    interpolation = cv2.INTER_AREA if content_scale < 1.0 else cv2.INTER_LANCZOS4
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=interpolation).astype(np.float32, copy=False)

    canvas_h = int(target_canvas_height_px) if target_canvas_height_px is not None else int(original_h)
    canvas_w = int(target_canvas_width_px) if target_canvas_width_px is not None else int(original_w)
    target_policy = "training_raw_canvas" if (target_canvas_height_px and target_canvas_width_px) else "external_raw_canvas_fallback"

    canvas = center_crop_or_pad_to_canvas(
        scaled,
        canvas_h,
        canvas_w,
        padding=padding,
    )

    effective_content_fraction = min(scaled_h, canvas_h) * min(scaled_w, canvas_w) / float(canvas_h * canvas_w)
    meta = {
        "original_height_px": int(original_h),
        "original_width_px": int(original_w),
        "training_pixel_spacing_um": training_px,
        "external_pixel_spacing_um": external_px,
        "external_over_training_resampling_factor": float(content_scale),
        "training_over_external_linear_sampling_ratio": float(training_px / external_px),
        "resampled_content_height_px": int(scaled_h),
        "resampled_content_width_px": int(scaled_w),
        "target_canvas_height_px": int(canvas_h),
        "target_canvas_width_px": int(canvas_w),
        "target_canvas_policy": target_policy,
        "padding_policy": str(padding),
        "content_fraction_before_final_resize": float(effective_content_fraction),
        "padding_fraction_before_final_resize": float(max(0.0, 1.0 - effective_content_fraction)),
    }
    return canvas.astype(np.float32, copy=False), meta


def photometric_mean_std_match_to_checkpoint(
    image: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    min_source_std: float = 1e-6,
    post_harmonization_clip: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Post hoc photometric sensitivity transform.

    Each external image is linearly transformed so that its pre-standardization
    mean and standard deviation match the frozen training-domain mean/std stored
    in the checkpoint. This tests whether global photometric mismatch plausibly
    contributes to external-domain failure. It must not be reported as the primary
    external-validation result because it uses test-image intensity statistics at
    inference time.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image before photometric matching, got {image.shape}")
    if len(mean) != 1 or len(std) != 1:
        raise ValueError(f"Expected one-channel checkpoint mean/std, got mean={mean}, std={std}")

    target_mean = float(mean[0])
    target_std = float(std[0])
    if target_std <= 0 or not np.isfinite(target_std):
        raise ValueError(f"Checkpoint target std must be positive and finite, got {target_std}")

    x = image.astype(np.float32, copy=False)
    source_mean = float(np.mean(x))
    source_std = float(np.std(x))
    min_source_std = float(min_source_std)
    if source_std < min_source_std or not np.isfinite(source_std):
        raise ValueError(
            f"Per-image source std is too small for stable photometric matching: "
            f"source_std={source_std}, min_source_std={min_source_std}"
        )

    y = (x - source_mean) / source_std
    y = y * target_std + target_mean
    if post_harmonization_clip:
        y = clip_unit_interval(y)

    meta = {
        "method": "per_image_mean_std_match_to_checkpoint_training_statistics",
        "source_mean_before": source_mean,
        "source_std_before": source_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "min_source_std": min_source_std,
        "output_mean_after": float(np.mean(y)),
        "output_std_after": float(np.std(y)),
        "post_harmonization_clip": bool(post_harmonization_clip),
        "interpretation_guardrail": (
            "Post hoc diagnostic photometric sensitivity only. This transformation uses "
            "external test-image intensity statistics and must not replace the primary "
            "locked external-validation result."
        ),
    }
    return y.astype(np.float32, copy=False), meta


def standardize_with_checkpoint_stats(
    image: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    if len(mean) != 1 or len(std) != 1:
        raise ValueError(
            "This pipeline expects one-channel checkpoint mean/std; got "
            f"mean={mean}, std={std}"
        )
    if float(std[0]) <= 0:
        raise ValueError(f"Checkpoint std must be positive, got {std[0]}")
    return ((image - float(mean[0])) / float(std[0])).astype(np.float32, copy=False)


def output_path_for_row(row: Dict[str, str], output_dir: Path) -> Path:
    rel_path = Path(row["relative_path"])
    return output_dir / rel_path.with_suffix(".npy")


def atomic_save_npy(array: np.ndarray, target_path: Path) -> None:
    """Write a .npy file atomically to avoid partially written arrays after interruption."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.",
        suffix=".tmp.npy",
        dir=str(target_path.parent),
    )
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


def preprocess_one(task: Dict[str, Any]) -> Dict[str, Any]:
    row = task["row"]
    output_dir = Path(task["output_dir"])
    image_size = int(task["image_size"])
    expected_effective_bit_depth = int(task["expected_effective_bit_depth"])
    allow_effective_bit_depth_fallback = bool(task["allow_effective_bit_depth_fallback"])
    save_standardized = bool(task["save_standardized"])
    checkpoint_mean = task["checkpoint_mean"]
    checkpoint_std = task["checkpoint_std"]
    force = bool(task["force"])
    mode = str(task.get("mode", "locked_pixel_pipeline"))
    training_pixel_spacing_um = task.get("training_pixel_spacing_um")
    external_pixel_spacing_um = task.get("external_pixel_spacing_um")
    scale_match_padding = str(task.get("scale_match_padding", "median"))
    scale_match_target_canvas_height_px = task.get("scale_match_target_canvas_height_px")
    scale_match_target_canvas_width_px = task.get("scale_match_target_canvas_width_px")
    post_resize_clip = bool(task.get("post_resize_clip", True))
    photometric_min_source_std = float(task.get("photometric_min_source_std", 1e-6))

    image_path = Path(row["image_path"])
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

        if target_path.exists() and not force:
            return {
                **result_base,
                "success": True,
                "skipped_existing": True,
                "inferred_effective_bit_depth": None,
                "normalization_denominator": None,
                "container_bit_depth": None,
                "min_pixel": None,
                "max_pixel": None,
                "output_min": None,
                "output_max": None,
                "scale_matching_metadata": None,
                "photometric_metadata": None,
            }

        image, tiff_meta = get_tiff_page_metadata(image_path)
        samples_per_pixel = tiff_meta.get("samples_per_pixel")
        container_bit_depth = tiff_meta.get("container_bit_depth")

        if not is_single_channel_image(image, samples_per_pixel):
            raise ValueError(
                f"Expected single-channel TIFF, got shape={image.shape}, "
                f"samples_per_pixel={samples_per_pixel}"
            )

        image = squeeze_singleton_channel(image)
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale TIFF after squeeze, got {image.shape}")
        if image.size == 0:
            raise ValueError("Image array is empty.")

        min_pixel = int(np.min(image))
        max_pixel = int(np.max(image))
        inferred_bd = infer_effective_bit_depth(
            max_pixel=max_pixel,
            container_bit_depth=container_bit_depth,
            expected_effective_bit_depth=expected_effective_bit_depth,
        )

        if inferred_bd != expected_effective_bit_depth and not allow_effective_bit_depth_fallback:
            raise ValueError(
                f"Inferred effective bit depth {inferred_bd} differs from expected "
                f"{expected_effective_bit_depth}. Use --allow-effective-bit-depth-fallback "
                "only for diagnostic runs after auditing this discrepancy."
            )

        x, denominator = normalize_linear_raw(image, inferred_bd)
        scale_matching_metadata = None
        photometric_metadata = None

        if mode == "locked_pixel_pipeline":
            x = resize_to_square(x, image_size, post_resize_clip=post_resize_clip)
        elif mode == "photometric_mean_std_matched_pipeline":
            x = resize_to_square(x, image_size, post_resize_clip=post_resize_clip)
            x, photometric_metadata = photometric_mean_std_match_to_checkpoint(
                x,
                checkpoint_mean,
                checkpoint_std,
                min_source_std=photometric_min_source_std,
                post_harmonization_clip=post_resize_clip,
            )
        elif mode == "scale_matched_physical_pipeline":
            if training_pixel_spacing_um is None or external_pixel_spacing_um is None:
                raise ValueError(
                    "scale_matched_physical_pipeline requires training and external pixel spacings. "
                    "Provide --training-pixel-spacing-um and --external-pixel-spacing-um or config values."
                )
            x, scale_matching_metadata = scale_match_to_training_physical_canvas(
                x,
                training_pixel_spacing_um=float(training_pixel_spacing_um),
                external_pixel_spacing_um=float(external_pixel_spacing_um),
                target_canvas_height_px=scale_match_target_canvas_height_px,
                target_canvas_width_px=scale_match_target_canvas_width_px,
                padding=scale_match_padding,
            )
            x = resize_to_square(x, image_size, post_resize_clip=post_resize_clip)
        else:
            raise ValueError(f"Unsupported preprocessing mode: {mode}")

        if save_standardized:
            x = standardize_with_checkpoint_stats(x, checkpoint_mean, checkpoint_std)

        if not np.all(np.isfinite(x)):
            raise ValueError("Preprocessed image contains non-finite values.")

        atomic_save_npy(x, target_path)

        return {
            **result_base,
            "success": True,
            "skipped_existing": False,
            "inferred_effective_bit_depth": int(inferred_bd),
            "normalization_denominator": int(denominator),
            "container_bit_depth": container_bit_depth,
            "min_pixel": min_pixel,
            "max_pixel": max_pixel,
            "output_min": float(np.min(x)),
            "output_max": float(np.max(x)),
            "scale_matching_metadata": scale_matching_metadata,
            "photometric_metadata": photometric_metadata,
        }

    except Exception as exc:
        return {
            **result_base,
            "success": False,
            "error": str(exc),
        }


def write_failures_csv(failures: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for item in failures:
            writer.writerow(
                {
                    "relative_path": item.get("relative_path", ""),
                    "image_path": item.get("image_path", ""),
                    "slide_id": item.get("slide_id", ""),
                    "species_name": item.get("species_name", ""),
                    "error": item.get("error", ""),
                }
            )


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def summarize_results(
    *,
    manifest_rows: List[Dict[str, str]],
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    config: Dict[str, Any],
    ckpt_meta: Dict[str, Any],
    checkpoint_warnings: List[str],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    processed = [r for r in successes if not r.get("skipped_existing")]
    skipped = [r for r in successes if r.get("skipped_existing")]

    species_image_counts = Counter(row["species_name"] for row in manifest_rows)
    slide_ids = sorted({row["slide_id"] for row in manifest_rows})
    species_slide_counts = Counter()
    hyphae_slide_counts = Counter()
    for slide_id in slide_ids:
        slide_rows = [r for r in manifest_rows if r["slide_id"] == slide_id]
        species_slide_counts[slide_rows[0]["species_name"]] += 1
        hyphae_slide_counts[slide_rows[0]["hyphae_status"]] += 1

    inferred_counts = Counter(
        str(r.get("inferred_effective_bit_depth"))
        for r in processed
        if r.get("inferred_effective_bit_depth") is not None
    )
    denominator_counts = Counter(
        str(r.get("normalization_denominator"))
        for r in processed
        if r.get("normalization_denominator") is not None
    )
    container_counts = Counter(
        str(r.get("container_bit_depth"))
        for r in processed
        if r.get("container_bit_depth") is not None
    )

    output_mins = np.array(
        [float(r["output_min"]) for r in processed if r.get("output_min") is not None],
        dtype=float,
    )
    output_maxs = np.array(
        [float(r["output_max"]) for r in processed if r.get("output_max") is not None],
        dtype=float,
    )

    training_px = getattr(args, "training_pixel_spacing_um", None) or get_nested(config, ["training_domain_reference", "pixel_spacing_um", "x"])
    external_px = (
        getattr(args, "external_pixel_spacing_um", None)
        or get_nested(config, ["external_acquisition_metadata", "pixel_spacing_um", "x"])
        or get_nested(config, ["external_domain", "pixel_spacing_um", "x"])
    )
    linear_ratio_training_over_external = None
    content_scale_external_over_training = None
    if training_px is not None and external_px is not None:
        try:
            linear_ratio_training_over_external = float(training_px) / float(external_px)
            content_scale_external_over_training = float(external_px) / float(training_px)
        except Exception:
            linear_ratio_training_over_external = None
            content_scale_external_over_training = None

    scale_metas = [r.get("scale_matching_metadata") for r in processed if r.get("scale_matching_metadata")]
    content_fractions = np.array(
        [float(m["content_fraction_before_final_resize"]) for m in scale_metas if "content_fraction_before_final_resize" in m],
        dtype=float,
    )
    padding_fractions = np.array(
        [float(m["padding_fraction_before_final_resize"]) for m in scale_metas if "padding_fraction_before_final_resize" in m],
        dtype=float,
    )
    resampled_h = np.array([float(m["resampled_content_height_px"]) for m in scale_metas if "resampled_content_height_px" in m], dtype=float)
    resampled_w = np.array([float(m["resampled_content_width_px"]) for m in scale_metas if "resampled_content_width_px" in m], dtype=float)
    canvas_h = np.array([float(m["target_canvas_height_px"]) for m in scale_metas if "target_canvas_height_px" in m], dtype=float)
    canvas_w = np.array([float(m["target_canvas_width_px"]) for m in scale_metas if "target_canvas_width_px" in m], dtype=float)
    photometric_metas = [r.get("photometric_metadata") for r in processed if r.get("photometric_metadata")]
    phot_src_mean = np.array([float(m["source_mean_before"]) for m in photometric_metas if "source_mean_before" in m], dtype=float)
    phot_src_std = np.array([float(m["source_std_before"]) for m in photometric_metas if "source_std_before" in m], dtype=float)
    phot_out_mean = np.array([float(m["output_mean_after"]) for m in photometric_metas if "output_mean_after" in m], dtype=float)
    phot_out_std = np.array([float(m["output_std_after"]) for m in photometric_metas if "output_std_after" in m], dtype=float)

    if args.mode == "locked_pixel_pipeline":
        analysis_role = "primary_external_validation"
    elif args.mode == "scale_matched_physical_pipeline":
        analysis_role = "post_hoc_scale_matching_sensitivity_analysis"
    elif args.mode == "photometric_mean_std_matched_pipeline":
        analysis_role = "post_hoc_photometric_mean_std_sensitivity_analysis"
    else:
        analysis_role = "post_hoc_sensitivity_analysis"

    metadata = {
        "schema_version": "1.0",
        "script": "preprocess_external.py",
        "created_unix_time": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "preprocessing_mode": args.mode,
        "primary_or_sensitivity": "primary" if args.mode == "locked_pixel_pipeline" else "sensitivity",
        "analysis_role": analysis_role,
        "scale_matching": (
            None if args.mode != "scale_matched_physical_pipeline" else {
                "scientific_role": "post_hoc_sensitivity_not_primary",
                "training_pixel_spacing_um": float(training_px) if training_px is not None else None,
                "external_pixel_spacing_um": float(external_px) if external_px is not None else None,
                "external_over_training_resampling_factor": content_scale_external_over_training,
                "training_over_external_linear_sampling_ratio": linear_ratio_training_over_external,
                "target_canvas_height_px": int(getattr(args, "scale_match_target_canvas_height_px", 0) or 0) or None,
                "target_canvas_width_px": int(getattr(args, "scale_match_target_canvas_width_px", 0) or 0) or None,
                "target_canvas_policy": (scale_metas[0].get("target_canvas_policy") if scale_metas else None),
                "padding_policy": getattr(args, "scale_match_padding", None),
                "mean_resampled_content_height_px": float(np.mean(resampled_h)) if resampled_h.size else None,
                "mean_resampled_content_width_px": float(np.mean(resampled_w)) if resampled_w.size else None,
                "mean_approx_content_fraction_in_model_input": float(np.mean(content_fractions)) if content_fractions.size else None,
                "mean_approx_padding_fraction_in_model_input": float(np.mean(padding_fractions)) if padding_fractions.size else None,
                "mean_effective_content_height_px_in_384": (
                    float(np.mean(resampled_h / canvas_h * int(args.image_size)))
                    if resampled_h.size and canvas_h.size and np.all(canvas_h > 0) else None
                ),
                "mean_effective_content_width_px_in_384": (
                    float(np.mean(resampled_w / canvas_w * int(args.image_size)))
                    if resampled_w.size and canvas_w.size and np.all(canvas_w > 0) else None
                ),
                "interpretation_guardrail": (
                    "Diagnostic post hoc sensitivity analysis only. It tests whether physical scale mismatch "
                    "may contribute to external-domain failure, but padding/background fraction and other "
                    "domain shifts remain confounded. Do not report this as the primary external-validation result."
                ),
            }
        ),
        "photometric_harmonization": (
            None if args.mode != "photometric_mean_std_matched_pipeline" else {
                "scientific_role": "post_hoc_sensitivity_not_primary",
                "method": "per_image_mean_std_match_to_checkpoint_training_statistics",
                "target_mean": float(ckpt_meta["checkpoint_mean"][0]) if ckpt_meta.get("checkpoint_mean") else None,
                "target_std": float(ckpt_meta["checkpoint_std"][0]) if ckpt_meta.get("checkpoint_std") else None,
                "min_source_std": float(getattr(args, "photometric_min_source_std", 1e-6)),
                "n_images_harmonized": len(photometric_metas),
                "source_mean_before_mean": float(np.mean(phot_src_mean)) if phot_src_mean.size else None,
                "source_std_before_mean": float(np.mean(phot_src_std)) if phot_src_std.size else None,
                "output_mean_after_mean": float(np.mean(phot_out_mean)) if phot_out_mean.size else None,
                "output_std_after_mean": float(np.mean(phot_out_std)) if phot_out_std.size else None,
                "post_harmonization_clip": bool(getattr(args, "post_resize_clip", True)),
                "interpretation_guardrail": (
                    "Diagnostic post hoc photometric sensitivity analysis only. It tests whether "
                    "global intensity/contrast mismatch plausibly contributed to external-domain "
                    "failure. Because it uses external test-image statistics, do not report it as "
                    "the primary external-validation result."
                ),
            }
        ),
        "scientific_role": (
            "Deterministic external preprocessing for frozen-model inference. "
            "No training, fine-tuning, threshold tuning, or preprocessing selection "
            "is performed here."
        ),
        "input_manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "checkpoint": ckpt_meta,
        "checkpoint_validation_warnings": checkpoint_warnings,
        "config_path": str(args.config) if args.config else None,
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
            "normalization_rule": "I_norm = I_raw / (2^inferred_effective_bit_depth - 1)",
            "expected_primary_denominator_for_14bit": 16383,
            "checkpoint_training_bit_depth_is_not_used_for_external_raw_scaling": True,
        },
        "spatial_policy": {
            "mode": (
                "pixel_locked_resize_to_model_input"
                if args.mode == "locked_pixel_pipeline"
                else (
                    "post_hoc_physical_scale_matched_to_training_canvas"
                    if args.mode == "scale_matched_physical_pipeline"
                    else "pixel_locked_resize_with_post_hoc_photometric_mean_std_matching"
                )
            ),
            "resize_interpolation": "cv2.INTER_LANCZOS4",
            "post_resize_clip_to_unit_interval": bool(getattr(args, "post_resize_clip", True)),
            "scale_matching_content_interpolation": (
                "cv2.INTER_AREA when downsampling, cv2.INTER_LANCZOS4 when upsampling"
                if args.mode == "scale_matched_physical_pipeline"
                else None
            ),
            "training_pixel_spacing_um": float(training_px) if training_px is not None else None,
            "external_pixel_spacing_um": float(external_px) if external_px is not None else None,
            "training_over_external_linear_sampling_ratio": linear_ratio_training_over_external,
            "external_over_training_resampling_factor": content_scale_external_over_training,
            "scale_matching_applied": args.mode == "scale_matched_physical_pipeline",
            "scale_match_padding": getattr(args, "scale_match_padding", None),
            "target_canvas_height_px_requested": getattr(args, "scale_match_target_canvas_height_px", None),
            "target_canvas_width_px_requested": getattr(args, "scale_match_target_canvas_width_px", None),
            "target_canvas_policy": (
                scale_metas[0].get("target_canvas_policy") if scale_metas else (
                    "not_applicable" if args.mode != "scale_matched_physical_pipeline" else "unknown_no_images_processed"
                )
            ),
            "target_canvas_height_px_observed": float(np.mean(canvas_h)) if canvas_h.size else None,
            "target_canvas_width_px_observed": float(np.mean(canvas_w)) if canvas_w.size else None,
            "mean_resampled_content_height_px": float(np.mean(resampled_h)) if resampled_h.size else None,
            "mean_resampled_content_width_px": float(np.mean(resampled_w)) if resampled_w.size else None,
            "mean_approx_content_fraction_before_final_resize": float(np.mean(content_fractions)) if content_fractions.size else None,
            "mean_approx_padding_fraction_before_final_resize": float(np.mean(padding_fractions)) if padding_fractions.size else None,
            "mean_effective_content_height_px_in_384": (
                float(np.mean(resampled_h / canvas_h * int(args.image_size)))
                if resampled_h.size and canvas_h.size and np.all(canvas_h > 0) else None
            ),
            "mean_effective_content_width_px_in_384": (
                float(np.mean(resampled_w / canvas_w * int(args.image_size)))
                if resampled_w.size and canvas_w.size and np.all(canvas_w > 0) else None
            ),
            "note": (
                "The primary locked_pixel_pipeline preserves the computational input size used by the model "
                "but does not attempt physical-scale harmonization."
                if args.mode == "locked_pixel_pipeline"
                else (
                    "Post hoc diagnostic sensitivity analysis: external content is resampled toward training pixel spacing and embedded into the specified raw canvas before final resize. This must not replace the primary external-validation result."
                    if args.mode == "scale_matched_physical_pipeline"
                    else "Post hoc diagnostic photometric sensitivity analysis: resized external inputs are linearly transformed to the frozen training-domain mean/std before checkpoint standardization. This must not replace the primary external-validation result."
                )
            ),
            "interpretation_guardrail": (
                None
                if args.mode == "locked_pixel_pipeline"
                else (
                    "Performance under scale matching is diagnostic of scale sensitivity only. Padding/background fraction and remaining camera, illumination, exposure, FOV, contrast, and preparation shifts remain confounded."
                    if args.mode == "scale_matched_physical_pipeline"
                    else "Performance under photometric mean/std matching is diagnostic of photometric sensitivity only. It uses external input statistics and remains confounded by scale, FOV, camera response, focus, texture, and preparation shifts."
                )
            ),
        },
        "counts": {
            "n_manifest_images": len(manifest_rows),
            "n_success": len(successes),
            "n_processed": len(processed),
            "n_skipped_existing": len(skipped),
            "n_failed": len(failures),
            "n_slides": len(slide_ids),
            "species_image_counts": dict(sorted(species_image_counts.items())),
            "species_slide_counts": dict(sorted(species_slide_counts.items())),
            "hyphae_slide_counts": dict(sorted(hyphae_slide_counts.items())),
            "inferred_effective_bit_depth_counts_processed": dict(sorted(inferred_counts.items())),
            "normalization_denominator_counts_processed": dict(sorted(denominator_counts.items())),
            "container_bit_depth_counts_processed": dict(sorted(container_counts.items())),
        },
        "output_value_summary_processed": {
            "min_of_output_min": float(np.min(output_mins)) if output_mins.size else None,
            "max_of_output_max": float(np.max(output_maxs)) if output_maxs.size else None,
            "mean_output_min": float(np.mean(output_mins)) if output_mins.size else None,
            "mean_output_max": float(np.mean(output_maxs)) if output_maxs.size else None,
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

    return metadata


def run_preprocessing(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], float]:
    manifest_rows = read_manifest(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_meta = load_checkpoint_metadata(args.checkpoint)
    config = load_optional_yaml_config(args.config)
    checkpoint_warnings = validate_checkpoint_against_config(
        ckpt_meta,
        config,
        image_size=args.image_size,
        strict=args.strict,
    )

    # Use config value when provided and CLI left at default. Explicit CLI value
    # remains authoritative.
    config_expected_bd = get_nested(
        config,
        ["tiff_audit", "expected_external_effective_bit_depth"],
        None,
    )
    if config_expected_bd is not None and args.expected_effective_bit_depth == DEFAULT_EXPECTED_EFFECTIVE_BIT_DEPTH:
        args.expected_effective_bit_depth = int(config_expected_bd)

    # Resolve pixel spacing and training-canvas metadata for the scale-matched sensitivity mode.
    if args.mode == "scale_matched_physical_pipeline":
        if args.training_pixel_spacing_um is None:
            cfg_training_px = get_nested(config, ["training_domain_reference", "pixel_spacing_um", "x"])
            if cfg_training_px is not None:
                args.training_pixel_spacing_um = float(cfg_training_px)
        if args.external_pixel_spacing_um is None:
            cfg_external_px = (
                get_nested(config, ["external_acquisition_metadata", "pixel_spacing_um", "x"])
                or get_nested(config, ["external_domain", "pixel_spacing_um", "x"])
            )
            if cfg_external_px is not None:
                args.external_pixel_spacing_um = float(cfg_external_px)
        if args.scale_match_target_canvas_height_px is None:
            cfg_h = get_nested(config, ["scale_matching_sensitivity", "training_raw_image_size_px", "height"])
            if cfg_h is not None:
                args.scale_match_target_canvas_height_px = int(cfg_h)
        if args.scale_match_target_canvas_width_px is None:
            cfg_w = get_nested(config, ["scale_matching_sensitivity", "training_raw_image_size_px", "width"])
            if cfg_w is not None:
                args.scale_match_target_canvas_width_px = int(cfg_w)

        if args.training_pixel_spacing_um is None or args.external_pixel_spacing_um is None:
            raise ValueError(
                "scale_matched_physical_pipeline requires training/external pixel spacings. "
                "Pass --training-pixel-spacing-um and --external-pixel-spacing-um or add them to config."
            )

        if args.scale_match_target_canvas_height_px is None or args.scale_match_target_canvas_width_px is None:
            print(
                "WARNING: scale-matched sensitivity is falling back to the external raw canvas because "
                "training raw canvas size was not provided. For this thesis dataset, use 1200 x 1200.",
                file=sys.stderr,
            )

    num_workers = calculate_num_workers(args.num_workers)

    tasks = []
    for row in manifest_rows:
        tasks.append(
            {
                "row": row,
                "output_dir": str(output_dir),
                "image_size": int(args.image_size),
                "expected_effective_bit_depth": int(args.expected_effective_bit_depth),
                "allow_effective_bit_depth_fallback": bool(args.allow_effective_bit_depth_fallback),
                "save_standardized": bool(args.save_standardized),
                "checkpoint_mean": ckpt_meta["checkpoint_mean"],
                "checkpoint_std": ckpt_meta["checkpoint_std"],
                "force": bool(args.force),
                "mode": args.mode,
                "training_pixel_spacing_um": args.training_pixel_spacing_um,
                "external_pixel_spacing_um": args.external_pixel_spacing_um,
                "scale_match_padding": args.scale_match_padding,
                "scale_match_target_canvas_height_px": args.scale_match_target_canvas_height_px,
                "scale_match_target_canvas_width_px": args.scale_match_target_canvas_width_px,
                "post_resize_clip": args.post_resize_clip,
                "photometric_min_source_std": args.photometric_min_source_std,
            }
        )

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

    elapsed = time.time() - start

    failures = [r for r in results if not r.get("success")]
    failures_path = output_dir / "preprocessing_failures.csv"
    write_failures_csv(failures, failures_path)

    metadata = summarize_results(
        manifest_rows=manifest_rows,
        results=results,
        args=args,
        config=config,
        ckpt_meta=ckpt_meta,
        checkpoint_warnings=checkpoint_warnings,
        elapsed_seconds=elapsed,
    )
    metadata_path = output_dir / args.metadata_filename
    write_json(metadata, metadata_path)

    if args.sidecar_output_dir is not None:
        args.sidecar_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_path, args.sidecar_output_dir / args.metadata_filename)
        if failures_path.exists():
            shutil.copy2(failures_path, args.sidecar_output_dir / "preprocessing_failures.csv")

    if failures and args.strict:
        # Metadata and failure CSV have already been saved for diagnosis.
        raise RuntimeError(
            f"{len(failures)} images failed preprocessing. See "
            f"{output_dir / 'preprocessing_failures.csv'}"
        )

    return results, elapsed


def print_summary(args: argparse.Namespace, results: List[Dict[str, Any]], elapsed: float) -> None:
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    skipped = [r for r in successes if r.get("skipped_existing")]
    processed = [r for r in successes if not r.get("skipped_existing")]

    print("\n" + "=" * 80)
    print("EXTERNAL PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Mode             : {args.mode}")
    print(f"Output directory : {args.output_dir}")
    print(f"Image size       : {args.image_size}")
    print(f"Expected bit depth: {args.expected_effective_bit_depth}")
    print(f"Post-resize clip: {args.post_resize_clip}")
    if args.mode == "scale_matched_physical_pipeline":
        print(f"Training spacing: {args.training_pixel_spacing_um} µm/px")
        print(f"External spacing: {args.external_pixel_spacing_um} µm/px")
        print(f"Target canvas   : {args.scale_match_target_canvas_height_px} x {args.scale_match_target_canvas_width_px}")
    if args.mode == "photometric_mean_std_matched_pipeline":
        print(f"Photometric mode: per-image mean/std -> checkpoint mean/std")
        print(f"Min source std  : {args.photometric_min_source_std}")
    print(f"Processed        : {len(processed)}")
    print(f"Skipped existing : {len(skipped)}")
    print(f"Failed           : {len(failures)}")
    print(f"Elapsed seconds  : {elapsed:.2f}")
    print(f"Metadata         : {args.output_dir / args.metadata_filename}")
    print(f"Failures CSV     : {args.output_dir / 'preprocessing_failures.csv'}")
    if args.sidecar_output_dir is not None:
        print(f"Sidecar directory: {args.sidecar_output_dir}")
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
