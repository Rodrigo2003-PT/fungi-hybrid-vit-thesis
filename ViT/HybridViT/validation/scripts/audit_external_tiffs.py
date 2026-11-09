#!/usr/bin/env python3
"""
audit_external_tiffs.py

Audit raw TIFF files from the external new-microscope validation cohort before
preprocessing or model inference.

Scientific role
---------------
The final Hybrid LSA-ConvTok model was trained on the original 12-bit microscope
domain. The external validation cohort was acquired on a different microscope
system and is expected to contain 14-bit brightfield monochrome TIFF images,
often stored in a 16-bit TIFF container. This script verifies that the raw files
are compatible with the pre-specified external-validation protocol and quantifies
photometric properties before any preprocessing is applied.

This audit is required because incorrect bit-depth handling can produce invalid
model inputs:
    - treating true 14-bit data as 12-bit can clip or distort intensities;
    - treating true 14-bit data as full 16-bit can compress intensities;
    - RGB/multichannel, saturated, corrupt, or unexpectedly scaled files would
      compromise the external-validation claim.

Optical calibration (O-4 / O-5)
-------------------------------
Each image's header calibration is read from the already-open TIFF (no extra
file pass) with priority ImageJ ``unit`` + XResolution/YResolution, then
OME-XML PhysicalSize, then TIFF ResolutionUnit. It is compared with the pixel
spacing derived from the metadata ``optics`` primitives (optics_provenance.py).
A disagreement beyond the tolerance or a shape different from
``optics.external.stored_image_px`` is an audit error. A missing calibration
tag is a warning and is never counted as a pass.

Status semantics (F-05)
-----------------------
``audit_info`` records expected conditions (for example values above 4095 in
true 14-bit data). ``audit_warnings`` is reserved for anomalies, so the
``warning`` status remains informative.

Inputs
------
    1. external_manifest.csv produced by external_manifest.py.
    2. external_validation_metadata.json (schema 2.0, ``optics`` block).
    3. Optional external_validation_config.yaml for expected bit-depth policy.

Outputs
-------
    tiff_audit_per_image.csv
        One row per image with TIFF metadata and intensity statistics.

    tiff_audit_per_slide.csv
        Aggregated statistics per slide.

    tiff_audit_summary.json
        Dataset-level summary, warnings, errors, and validation status.

    intensity_histograms_raw.png
        Raw-pixel intensity histograms by species.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tifffile
except ImportError as exc:
    raise ImportError(
        "tifffile is required for audit_external_tiffs.py. "
        "Install it with: pip install tifffile"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for histogram output. "
        "Install it with: pip install matplotlib"
    ) from exc

from optics_provenance import (
    derive_pixel_spacing_um,
    optics_provenance_record,
    read_tiff_calibration,
    relative_error,
    validate_optics,
)
from pipeline_common import resolve_setting


AUDIT_SCHEMA_VERSION = "2.0"
SUPPORTED_TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}
DEFAULT_EXPECTED_EXTERNAL_BIT_DEPTH = 14
DEFAULT_EXPECTED_CONTAINER_BIT_DEPTHS = {14, 16}

PER_IMAGE_COLUMNS = [
    "image_path",
    "relative_path",
    "filename",
    "slide_id",
    "species_name",
    "species_code",
    "label",
    "hyphae_status",
    "shape",
    "ndim",
    "dtype",
    "container_bit_depth",
    "photometric",
    "samples_per_pixel",
    "is_single_channel",
    "min_pixel",
    "max_pixel",
    "mean_pixel",
    "std_pixel",
    "p001",
    "p01",
    "p1",
    "p50",
    "p99",
    "p999",
    "zero_fraction",
    "values_above_4095_fraction",
    "values_above_16383_fraction",
    "saturation_fraction_12bit",
    "saturation_fraction_14bit",
    "saturation_fraction_16bit",
    "inferred_effective_bit_depth",
    "normalization_denominator_if_inferred",
    "dynamic_range_fraction_12bit",
    "dynamic_range_fraction_14bit",
    "dynamic_range_fraction_16bit",
    "header_spacing_x_um",
    "header_spacing_y_um",
    "header_spacing_source",
    "spacing_rel_error",
    "shape_matches_optics",
    "audit_status",
    "audit_info",
    "audit_warnings",
    "audit_errors",
]

PER_SLIDE_COLUMNS = [
    "slide_id",
    "species_name",
    "label",
    "hyphae_status",
    "n_images",
    "n_failed_images",
    "n_warning_images",
    "n_single_channel_images",
    "container_bit_depths",
    "inferred_effective_bit_depths",
    "min_pixel_min",
    "max_pixel_max",
    "mean_pixel_mean",
    "std_pixel_mean",
    "p001_median",
    "p01_median",
    "p1_median",
    "p50_median",
    "p99_median",
    "p999_median",
    "zero_fraction_mean",
    "values_above_4095_fraction_mean",
    "values_above_16383_fraction_mean",
    "saturation_fraction_14bit_mean",
    "audit_status",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit external-validation TIFF files before preprocessing and "
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
        "--metadata",
        required=True,
        type=Path,
        help="external_validation_metadata.json (schema 2.0) providing the optics block.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional external_validation_config.yaml (expected bit-depth policy).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help=(
            "Optional dataset root. When given, images are read from raw_root/relative_path "
            "instead of the absolute manifest image_path (portable re-runs)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where audit CSV/JSON/figure outputs will be written.",
    )
    parser.add_argument(
        "--expected-effective-bit-depth",
        type=int,
        default=None,
        choices=[8, 12, 14, 16],
        help=(
            "Expected effective bit depth of external raw acquisition. Precedence: "
            "this flag > config tiff_audit.expected_external_effective_bit_depth > 14."
        ),
    )
    parser.add_argument(
        "--histogram-sample-pixels-per-image",
        type=int,
        default=5000,
        help=(
            "Maximum number of pixels sampled per image for histogram plotting. "
            "Set to 0 to use all pixels, which may be memory intensive."
        ),
    )
    parser.add_argument(
        "--histogram-max-images-per-species",
        type=int,
        default=200,
        help="Maximum number of images sampled per species for raw histograms.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for histogram pixel/image sampling.",
    )
    parser.add_argument(
        "--high-saturation-warning-threshold",
        type=float,
        default=0.001,
        help=(
            "Warn if fraction of pixels saturated at the expected effective "
            "bit depth is greater than this value."
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, return a non-zero exit code when severe audit errors are "
            "detected."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def load_optional_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read the config. Install with: pip install pyyaml") from exc

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
        "species_code",
        "label",
        "hyphae_status",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    return rows


def is_single_channel_image(image: np.ndarray, samples_per_pixel: Optional[int]) -> bool:
    """
    Determine whether image is single-channel.

    Valid expected cases:
        - 2D grayscale image: (H, W)
        - 3D singleton-channel image: (H, W, 1) or (1, H, W)
          This is accepted but flagged through shape/ndim.
    """
    if image.ndim == 2:
        return True

    if samples_per_pixel == 1 and image.ndim in (2, 3):
        # Some TIFF readers expose singleton sample/channel dimensions.
        if 1 in image.shape:
            return True

    if image.ndim == 3 and 1 in image.shape:
        return True

    return False


def squeeze_singleton_channel(image: np.ndarray) -> np.ndarray:
    """
    Squeeze singleton channel dimensions while preserving grayscale arrays.
    """
    if image.ndim == 2:
        return image

    squeezed = np.squeeze(image)
    if squeezed.ndim == 2:
        return squeezed

    return image


def safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def infer_effective_bit_depth_from_values(
    max_pixel: int,
    container_bit_depth: Optional[int],
    expected_effective_bit_depth: int,
) -> int:
    """
    Infer effective bit depth from observed intensity range and TIFF metadata.

    The external domain is expected to be 14-bit acquisition, often stored in a
    uint16/16-bit TIFF container. If the expected bit depth is 14 and the maximum
    value is <= 16383, the file is treated as effectively 14-bit even if the
    TIFF container reports 16 bits per sample.
    """
    thresholds = {
        8: 2**8 - 1,
        12: 2**12 - 1,
        14: 2**14 - 1,
        16: 2**16 - 1,
    }

    if expected_effective_bit_depth not in thresholds:
        raise ValueError(
            f"Unsupported expected bit depth: {expected_effective_bit_depth}"
        )

    expected_max = thresholds[expected_effective_bit_depth]

    # Respect exact container/effective match when plausible.
    if container_bit_depth == expected_effective_bit_depth and max_pixel <= expected_max:
        return expected_effective_bit_depth

    # Explicitly support 14-bit data stored inside a 16-bit TIFF container.
    if (
        expected_effective_bit_depth == 14
        and container_bit_depth == 16
        and max_pixel <= thresholds[14]
    ):
        return 14

    # General fallback: choose the smallest supported bit depth that can contain
    # the observed maximum. This is descriptive and does not override the audit
    # warning logic.
    for bd in (8, 12, 14, 16):
        if max_pixel <= thresholds[bd]:
            return bd

    return 16


def get_tiff_page_metadata(image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with tifffile.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]
        image = tif.asarray()

        # TIFF tags may be scalar or tuple depending on file.
        bits_per_sample = getattr(page, "bitspersample", None)
        if isinstance(bits_per_sample, (tuple, list)):
            # For grayscale this is usually a single value. For RGB it may be
            # (8, 8, 8); keep the first for container_bit_depth and use
            # samples_per_pixel to flag multichannel.
            container_bit_depth = int(bits_per_sample[0]) if bits_per_sample else None
        else:
            container_bit_depth = int(bits_per_sample) if bits_per_sample is not None else None

        photometric = str(getattr(page, "photometric", "unknown"))
        samples_per_pixel = getattr(page, "samplesperpixel", None)
        samples_per_pixel = int(samples_per_pixel) if samples_per_pixel is not None else None

        metadata = {
            "container_bit_depth": container_bit_depth,
            "photometric": photometric,
            "samples_per_pixel": samples_per_pixel,
            "calibration": read_tiff_calibration(tif),
        }

    return image, metadata


def compute_image_statistics(
    manifest_row: Dict[str, str],
    *,
    expected_effective_bit_depth: int,
    expected_container_bit_depths: Sequence[int],
    high_saturation_warning_threshold: float,
    optics_expectation: Optional[Dict[str, Any]] = None,
    raw_root: Optional[Path] = None,
) -> Dict[str, Any]:
    image_path = resolve_image_path(manifest_row, raw_root)

    audit_info: List[str] = []
    audit_warnings: List[str] = []
    audit_errors: List[str] = []

    base_row: Dict[str, Any] = {
        "image_path": str(image_path),
        "relative_path": manifest_row.get("relative_path", ""),
        "filename": manifest_row.get("filename", image_path.name),
        "slide_id": manifest_row.get("slide_id", ""),
        "species_name": manifest_row.get("species_name", ""),
        "species_code": manifest_row.get("species_code", ""),
        "label": manifest_row.get("label", ""),
        "hyphae_status": manifest_row.get("hyphae_status", ""),
    }

    if image_path.suffix not in SUPPORTED_TIFF_EXTENSIONS:
        audit_errors.append(f"Unsupported extension: {image_path.suffix}")

    if not image_path.exists():
        audit_errors.append("File does not exist.")
        return finalize_failed_row(base_row, audit_warnings, audit_errors)

    try:
        image, tiff_meta = get_tiff_page_metadata(image_path)
    except Exception as exc:
        audit_errors.append(f"Failed to read TIFF: {exc}")
        return finalize_failed_row(base_row, audit_warnings, audit_errors)

    optics_fields = check_optics_against_image(
        image_shape=tuple(int(v) for v in squeeze_singleton_channel(image).shape),
        calibration=tiff_meta.get("calibration", {}),
        optics_expectation=optics_expectation,
        audit_warnings=audit_warnings,
        audit_errors=audit_errors,
    )
    base_row.update(optics_fields)

    container_bit_depth = tiff_meta.get("container_bit_depth")
    photometric = tiff_meta.get("photometric")
    samples_per_pixel = tiff_meta.get("samples_per_pixel")

    shape = tuple(int(v) for v in image.shape)
    is_single = is_single_channel_image(image, samples_per_pixel)
    if not is_single:
        audit_errors.append(
            f"Expected single-channel image, got shape={shape}, "
            f"samples_per_pixel={samples_per_pixel}."
        )

    image_for_stats = squeeze_singleton_channel(image)
    if image_for_stats.ndim != 2:
        audit_errors.append(
            f"Could not reduce image to 2D grayscale array for statistics; "
            f"shape after squeeze={image_for_stats.shape}."
        )
        return finalize_failed_row(
            {
                **base_row,
                "shape": str(shape),
                "ndim": int(image.ndim),
                "dtype": str(image.dtype),
                "container_bit_depth": container_bit_depth,
                "photometric": photometric,
                "samples_per_pixel": samples_per_pixel,
                "is_single_channel": bool(is_single),
            },
            audit_warnings,
            audit_errors,
        )

    if image_for_stats.size == 0:
        audit_errors.append("Image array is empty.")
        return finalize_failed_row(base_row, audit_warnings, audit_errors)

    if not np.issubdtype(image_for_stats.dtype, np.integer):
        audit_warnings.append(
            f"Expected integer raw TIFF data; got dtype={image_for_stats.dtype}."
        )

    arr = image_for_stats.astype(np.float64, copy=False)

    min_pixel = float(np.min(arr))
    max_pixel = float(np.max(arr))
    mean_pixel = float(np.mean(arr))
    std_pixel = float(np.std(arr))

    p001, p01, p1, p50, p99, p999 = [
        float(v) for v in np.percentile(arr, [0.01, 0.1, 1.0, 50.0, 99.0, 99.9])
    ]
    # Percentile naming convention:
    #   p001 = 0.01th percentile
    #   p01  = 0.1th percentile
    #   p1   = 1st percentile
    #   p50  = 50th percentile
    #   p99  = 99th percentile
    #   p999 = 99.9th percentile
    # This avoids duplicate lower-tail statistics and keeps the exported schema stable.

    zero_fraction = float(np.mean(arr == 0))
    values_above_4095_fraction = float(np.mean(arr > (2**12 - 1)))
    values_above_16383_fraction = float(np.mean(arr > (2**14 - 1)))

    saturation_fraction_12bit = float(np.mean(arr >= (2**12 - 1)))
    saturation_fraction_14bit = float(np.mean(arr >= (2**14 - 1)))
    saturation_fraction_16bit = float(np.mean(arr >= (2**16 - 1)))

    inferred_bd = infer_effective_bit_depth_from_values(
        max_pixel=int(max_pixel),
        container_bit_depth=container_bit_depth,
        expected_effective_bit_depth=expected_effective_bit_depth,
    )
    inferred_denom = (2**inferred_bd) - 1

    dynamic_range_fraction_12bit = float(max_pixel / (2**12 - 1))
    dynamic_range_fraction_14bit = float(max_pixel / (2**14 - 1))
    dynamic_range_fraction_16bit = float(max_pixel / (2**16 - 1))

    if container_bit_depth is not None and int(container_bit_depth) not in set(expected_container_bit_depths):
        audit_warnings.append(
            f"Unexpected TIFF container bit depth {container_bit_depth}; "
            f"expected one of {sorted(expected_container_bit_depths)}."
        )

    if expected_effective_bit_depth == 14 and values_above_4095_fraction > 0:
        # Expected condition for true 14-bit data: informational, not a warning (F-05).
        audit_info.append(
            "Pixel values exceed 4095; this is expected for true 14-bit data "
            "and confirms that 12-bit normalization would be inappropriate."
        )

    if values_above_16383_fraction > 0:
        audit_errors.append(
            "Pixel values exceed 16383, which is unexpected for true 14-bit "
            "acquisition. Investigate export scaling or bit-depth metadata."
        )

    if saturation_fraction_14bit > high_saturation_warning_threshold:
        audit_warnings.append(
            f"High 14-bit saturation fraction: {saturation_fraction_14bit:.6f}."
        )

    if inferred_bd != expected_effective_bit_depth:
        audit_warnings.append(
            f"Inferred effective bit depth is {inferred_bd}, but expected "
            f"{expected_effective_bit_depth}. This may indicate under-used "
            "dynamic range or export scaling."
        )

    if max_pixel <= (2**12 - 1) and expected_effective_bit_depth == 14:
        audit_warnings.append(
            "Maximum pixel value does not exceed 4095. The file may be 14-bit "
            "by metadata but effectively uses only a 12-bit intensity range."
        )

    status = "failed" if audit_errors else ("warning" if audit_warnings else "passed")

    return {
        **base_row,
        "shape": str(shape),
        "ndim": int(image.ndim),
        "dtype": str(image.dtype),
        "container_bit_depth": container_bit_depth,
        "photometric": photometric,
        "samples_per_pixel": samples_per_pixel,
        "is_single_channel": bool(is_single),
        "min_pixel": min_pixel,
        "max_pixel": max_pixel,
        "mean_pixel": mean_pixel,
        "std_pixel": std_pixel,
        "p001": p001,
        "p01": p01,
        "p1": p1,
        "p50": p50,
        "p99": p99,
        "p999": p999,
        "zero_fraction": zero_fraction,
        "values_above_4095_fraction": values_above_4095_fraction,
        "values_above_16383_fraction": values_above_16383_fraction,
        "saturation_fraction_12bit": saturation_fraction_12bit,
        "saturation_fraction_14bit": saturation_fraction_14bit,
        "saturation_fraction_16bit": saturation_fraction_16bit,
        "inferred_effective_bit_depth": int(inferred_bd),
        "normalization_denominator_if_inferred": int(inferred_denom),
        "dynamic_range_fraction_12bit": dynamic_range_fraction_12bit,
        "dynamic_range_fraction_14bit": dynamic_range_fraction_14bit,
        "dynamic_range_fraction_16bit": dynamic_range_fraction_16bit,
        "audit_status": status,
        "audit_info": " | ".join(audit_info),
        "audit_warnings": " | ".join(audit_warnings),
        "audit_errors": " | ".join(audit_errors),
    }


def resolve_image_path(manifest_row: Dict[str, str], raw_root: Optional[Path]) -> Path:
    """raw_root/relative_path when --raw-root is given, else the manifest image_path."""
    if raw_root is not None:
        return Path(raw_root) / Path(manifest_row["relative_path"])
    return Path(manifest_row["image_path"])


def build_optics_expectation(optics: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Derive the per-image expectation (spacing, stored shape, tolerance) from the
    metadata optics block. Returns (expectation_or_None, provenance_record).
    """
    record = optics_provenance_record(optics)
    validation, domains = validate_optics(optics)
    ext = domains.get("external")
    if ext is None:
        return None, record
    return {
        "pixel_spacing_um": derive_pixel_spacing_um(ext),
        "stored_image_px": tuple(ext.stored_image_px),
        "spacing_rel_tolerance": float(validation.checks.get("spacing_rel_tolerance")),
    }, record


def check_optics_against_image(
    *,
    image_shape: Tuple[int, ...],
    calibration: Dict[str, Any],
    optics_expectation: Optional[Dict[str, Any]],
    audit_warnings: List[str],
    audit_errors: List[str],
) -> Dict[str, Any]:
    """Apply O-4 (header calibration) and O-5 (shape) to one image."""
    out: Dict[str, Any] = {
        "header_spacing_x_um": calibration.get("x_um"),
        "header_spacing_y_um": calibration.get("y_um"),
        "header_spacing_source": calibration.get("source", "calibration_tag_absent"),
        "spacing_rel_error": "",
        "shape_matches_optics": "",
    }
    if optics_expectation is None:
        audit_warnings.append("Optics expectation unavailable; O-4/O-5 not evaluated.")
        return out

    expected_shape = tuple(optics_expectation["stored_image_px"])
    shape_ok = tuple(image_shape) == expected_shape
    out["shape_matches_optics"] = bool(shape_ok)
    if not shape_ok:
        audit_errors.append(f"O-5: image shape {tuple(image_shape)} != optics stored_image_px {expected_shape}.")

    x_um, y_um = calibration.get("x_um"), calibration.get("y_um")
    if x_um is None or y_um is None:
        audit_warnings.append("O-4: calibration_tag_absent (no ImageJ/OME/TIFF pixel calibration found).")
        return out

    expected = float(optics_expectation["pixel_spacing_um"])
    rel = max(relative_error(x_um, expected), relative_error(y_um, expected))
    out["spacing_rel_error"] = rel
    if rel > float(optics_expectation["spacing_rel_tolerance"]):
        audit_errors.append(
            f"O-4: header spacing ({x_um:.8g}, {y_um:.8g}) µm/px from {calibration.get('source')} "
            f"disagrees with derived {expected:.8g} µm/px (relative error {rel:.4%})."
        )
    if relative_error(x_um, y_um) > float(optics_expectation["spacing_rel_tolerance"]):
        audit_warnings.append(f"O-4: anisotropic calibration x={x_um:.8g}, y={y_um:.8g} µm/px.")
    return out


def finalize_failed_row(
    base_row: Dict[str, Any],
    audit_warnings: List[str],
    audit_errors: List[str],
) -> Dict[str, Any]:
    row = {col: "" for col in PER_IMAGE_COLUMNS}
    row.update(base_row)
    row["audit_status"] = "failed"
    row["audit_warnings"] = " | ".join(audit_warnings)
    row["audit_errors"] = " | ".join(audit_errors)
    return row


def write_csv(rows: List[Dict[str, Any]], output_path: Path, columns: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_float_or_nan(value: Any) -> float:
    try:
        if value == "" or value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def aggregate_per_slide(per_image_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slide_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        slide_to_rows[str(row.get("slide_id", ""))].append(row)

    out: List[Dict[str, Any]] = []

    numeric_fields = [
        "min_pixel",
        "max_pixel",
        "mean_pixel",
        "std_pixel",
        "p001",
        "p01",
        "p1",
        "p50",
        "p99",
        "p999",
        "zero_fraction",
        "values_above_4095_fraction",
        "values_above_16383_fraction",
        "saturation_fraction_14bit",
    ]

    for slide_id, rows in sorted(slide_to_rows.items()):
        species = sorted({str(r.get("species_name", "")) for r in rows})
        labels = sorted({str(r.get("label", "")) for r in rows})
        hyphae = sorted({str(r.get("hyphae_status", "")) for r in rows})

        status_counts = Counter(str(r.get("audit_status", "")) for r in rows)
        n_failed = status_counts.get("failed", 0)
        n_warning = status_counts.get("warning", 0)

        row_out: Dict[str, Any] = {
            "slide_id": slide_id,
            "species_name": species[0] if len(species) == 1 else "|".join(species),
            "label": labels[0] if len(labels) == 1 else "|".join(labels),
            "hyphae_status": hyphae[0] if len(hyphae) == 1 else "|".join(hyphae),
            "n_images": len(rows),
            "n_failed_images": n_failed,
            "n_warning_images": n_warning,
            "n_single_channel_images": sum(str(r.get("is_single_channel")) == "True" or r.get("is_single_channel") is True for r in rows),
            "container_bit_depths": "|".join(sorted({str(r.get("container_bit_depth", "")) for r in rows if r.get("container_bit_depth", "") != ""})),
            "inferred_effective_bit_depths": "|".join(sorted({str(r.get("inferred_effective_bit_depth", "")) for r in rows if r.get("inferred_effective_bit_depth", "") != ""})),
            "audit_status": "failed" if n_failed else ("warning" if n_warning else "passed"),
        }

        values: Dict[str, np.ndarray] = {}
        for field in numeric_fields:
            arr = np.array([to_float_or_nan(r.get(field)) for r in rows], dtype=float)
            arr = arr[np.isfinite(arr)]
            values[field] = arr

        row_out["min_pixel_min"] = float(np.min(values["min_pixel"])) if values["min_pixel"].size else ""
        row_out["max_pixel_max"] = float(np.max(values["max_pixel"])) if values["max_pixel"].size else ""
        row_out["mean_pixel_mean"] = float(np.mean(values["mean_pixel"])) if values["mean_pixel"].size else ""
        row_out["std_pixel_mean"] = float(np.mean(values["std_pixel"])) if values["std_pixel"].size else ""

        for field in ("p001", "p01", "p1", "p50", "p99", "p999"):
            row_out[f"{field}_median"] = float(np.median(values[field])) if values[field].size else ""

        row_out["zero_fraction_mean"] = float(np.mean(values["zero_fraction"])) if values["zero_fraction"].size else ""
        row_out["values_above_4095_fraction_mean"] = (
            float(np.mean(values["values_above_4095_fraction"]))
            if values["values_above_4095_fraction"].size else ""
        )
        row_out["values_above_16383_fraction_mean"] = (
            float(np.mean(values["values_above_16383_fraction"]))
            if values["values_above_16383_fraction"].size else ""
        )
        row_out["saturation_fraction_14bit_mean"] = (
            float(np.mean(values["saturation_fraction_14bit"]))
            if values["saturation_fraction_14bit"].size else ""
        )

        out.append(row_out)

    return out


def summarize_audit(
    per_image_rows: List[Dict[str, Any]],
    per_slide_rows: List[Dict[str, Any]],
    *,
    manifest_path: Path,
    config_path: Optional[Path],
    expected_effective_bit_depth: int,
    optics_record: Optional[Dict[str, Any]] = None,
    settings_resolution: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    image_status_counts = Counter(str(r.get("audit_status", "")) for r in per_image_rows)
    slide_status_counts = Counter(str(r.get("audit_status", "")) for r in per_slide_rows)
    species_counts_images = Counter(str(r.get("species_name", "")) for r in per_image_rows)
    species_counts_slides = Counter(str(r.get("species_name", "")) for r in per_slide_rows)
    hyphae_counts_slides = Counter(str(r.get("hyphae_status", "")) for r in per_slide_rows)

    errors = []
    warnings_list = []

    failed_images = [
        {
            "relative_path": r.get("relative_path", ""),
            "slide_id": r.get("slide_id", ""),
            "errors": r.get("audit_errors", ""),
        }
        for r in per_image_rows
        if r.get("audit_status") == "failed"
    ]

    warning_images = [
        {
            "relative_path": r.get("relative_path", ""),
            "slide_id": r.get("slide_id", ""),
            "warnings": r.get("audit_warnings", ""),
        }
        for r in per_image_rows
        if r.get("audit_status") == "warning"
    ]

    if failed_images:
        errors.append(f"{len(failed_images)} images failed TIFF audit.")

    def _is_true(v: Any) -> bool:
        return v is True or str(v) == "True"

    optics_checks = {
        "n_shape_matches_optics": sum(1 for r in per_image_rows if _is_true(r.get("shape_matches_optics"))),
        "n_shape_mismatch": sum(1 for r in per_image_rows if r.get("shape_matches_optics") is False or str(r.get("shape_matches_optics")) == "False"),
        "header_spacing_source_counts": dict(sorted(Counter(str(r.get("header_spacing_source", "")) for r in per_image_rows).items())),
        "n_calibration_tag_absent": sum(1 for r in per_image_rows if str(r.get("header_spacing_source", "")) == "calibration_tag_absent"),
        "max_spacing_rel_error": max(
            [float(r["spacing_rel_error"]) for r in per_image_rows if r.get("spacing_rel_error") not in ("", None)],
            default=None,
        ),
    }
    if optics_checks["n_calibration_tag_absent"]:
        warnings_list.append(f"{optics_checks['n_calibration_tag_absent']} images have no pixel calibration tag (O-4).")
    info_counts = Counter()
    for r in per_image_rows:
        for msg in [m for m in str(r.get("audit_info", "")).split(" | ") if m]:
            info_counts[msg] += 1

    bit_depth_counts = Counter(
        str(r.get("inferred_effective_bit_depth", ""))
        for r in per_image_rows
        if r.get("inferred_effective_bit_depth", "") != ""
    )
    container_counts = Counter(
        str(r.get("container_bit_depth", ""))
        for r in per_image_rows
        if r.get("container_bit_depth", "") != ""
    )

    if str(expected_effective_bit_depth) not in bit_depth_counts:
        warnings_list.append(
            f"No image was inferred as expected {expected_effective_bit_depth}-bit."
        )

    values_above_4095 = np.array(
        [to_float_or_nan(r.get("values_above_4095_fraction")) for r in per_image_rows],
        dtype=float,
    )
    values_above_4095 = values_above_4095[np.isfinite(values_above_4095)]

    values_above_16383 = np.array(
        [to_float_or_nan(r.get("values_above_16383_fraction")) for r in per_image_rows],
        dtype=float,
    )
    values_above_16383 = values_above_16383[np.isfinite(values_above_16383)]

    sat14 = np.array(
        [to_float_or_nan(r.get("saturation_fraction_14bit")) for r in per_image_rows],
        dtype=float,
    )
    sat14 = sat14[np.isfinite(sat14)]

    mean_values_above_4095 = float(np.mean(values_above_4095)) if values_above_4095.size else None
    mean_values_above_16383 = float(np.mean(values_above_16383)) if values_above_16383.size else None
    mean_sat14 = float(np.mean(sat14)) if sat14.size else None

    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "validation_status": "failed" if errors else "passed_with_warnings" if warning_images or warnings_list else "passed",
        "manifest_path": str(manifest_path),
        "config_path": str(config_path) if config_path else None,
        "expected_effective_bit_depth": expected_effective_bit_depth,
        "n_images": len(per_image_rows),
        "n_slides": len(per_slide_rows),
        "image_audit_status_counts": dict(sorted(image_status_counts.items())),
        "slide_audit_status_counts": dict(sorted(slide_status_counts.items())),
        "species_image_counts": dict(sorted(species_counts_images.items())),
        "species_slide_counts": dict(sorted(species_counts_slides.items())),
        "hyphae_slide_counts": dict(sorted(hyphae_counts_slides.items())),
        "container_bit_depth_counts": dict(sorted(container_counts.items())),
        "inferred_effective_bit_depth_counts": dict(sorted(bit_depth_counts.items())),
        "mean_values_above_4095_fraction": mean_values_above_4095,
        "mean_values_above_16383_fraction": mean_values_above_16383,
        "mean_saturation_fraction_14bit": mean_sat14,
        "optics_checks": optics_checks,
        "optics_sha256": (optics_record or {}).get("optics_sha256"),
        "optics_provenance_gaps": (optics_record or {}).get("provenance_gaps", []),
        "info_counts": dict(sorted(info_counts.items())),
        "settings_resolution": settings_resolution or {},
        "errors": errors,
        "warnings": warnings_list,
        "failed_images": failed_images,
        "warning_images_count": len(warning_images),
        "warning_images_first_50": warning_images[:50],
        "scientific_interpretation": {
            "values_above_4095": (
                "For a 14-bit external acquisition, non-zero values above 4095 "
                "support the conclusion that 12-bit normalization would be "
                "incorrect."
            ),
            "values_above_16383": (
                "Values above 16383 are unexpected for true 14-bit raw data and "
                "should be investigated before final preprocessing."
            ),
            "audit_info": (
                "Expected conditions are recorded in audit_info and do not change the "
                "image status; audit_warnings is reserved for anomalies."
            ),
            "slide_level_unit": (
                "This TIFF audit is image-level for quality control, but model "
                "performance statistics should use slide-level aggregation."
            ),
        },
    }

    return summary


def sample_pixels_for_histograms(
    per_image_rows: List[Dict[str, Any]],
    *,
    max_images_per_species: int,
    sample_pixels_per_image: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    species_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for row in per_image_rows:
        if row.get("audit_status") == "failed":
            continue
        species_to_paths[str(row.get("species_name", ""))].append(Path(str(row.get("image_path"))))

    species_to_pixels: Dict[str, List[np.ndarray]] = defaultdict(list)

    for species, paths in species_to_paths.items():
        paths_sorted = sorted(paths, key=lambda p: p.as_posix())
        if max_images_per_species > 0 and len(paths_sorted) > max_images_per_species:
            selected_idx = rng.choice(len(paths_sorted), size=max_images_per_species, replace=False)
            selected_paths = [paths_sorted[int(i)] for i in selected_idx]
        else:
            selected_paths = paths_sorted

        for path in selected_paths:
            try:
                image, _ = get_tiff_page_metadata(path)
                image = squeeze_singleton_channel(image)
                if image.ndim != 2:
                    continue
                flat = image.reshape(-1)
                if sample_pixels_per_image > 0 and flat.size > sample_pixels_per_image:
                    idx = rng.choice(flat.size, size=sample_pixels_per_image, replace=False)
                    flat = flat[idx]
                species_to_pixels[species].append(flat.astype(np.float64, copy=False))
            except Exception:
                continue

    return {
        species: np.concatenate(chunks) if chunks else np.array([], dtype=float)
        for species, chunks in species_to_pixels.items()
    }


def plot_raw_intensity_histograms(
    species_to_pixels: Dict[str, np.ndarray],
    output_path: Path,
    *,
    expected_effective_bit_depth: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_expected = (2**expected_effective_bit_depth) - 1
    bins = np.linspace(0, max_expected, 256)

    plt.figure(figsize=(10, 6))
    any_data = False

    for species, pixels in sorted(species_to_pixels.items()):
        if pixels.size == 0:
            continue
        any_data = True
        clipped = pixels[(pixels >= 0) & (pixels <= max_expected)]
        if clipped.size == 0:
            continue
        plt.hist(
            clipped,
            bins=bins,
            density=True,
            alpha=0.45,
            label=f"{species} (n={clipped.size:,} pixels)",
        )

    plt.axvline(2**12 - 1, linestyle="--", linewidth=1.2, label="12-bit max (4095)")
    plt.axvline(2**14 - 1, linestyle=":", linewidth=1.2, label="14-bit max (16383)")

    plt.xlabel("Raw pixel intensity")
    plt.ylabel("Density")
    plt.title("Raw external TIFF intensity distributions by species")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if any_data:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        # Still create an empty diagnostic figure.
        plt.text(0.5, 0.5, "No valid pixel data available", ha="center", va="center")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def print_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("EXTERNAL TIFF AUDIT COMPLETE")
    print("=" * 80)
    print(f"Output directory : {output_dir}")
    print(f"Status           : {summary['validation_status']}")
    print(f"Images           : {summary['n_images']}")
    print(f"Slides           : {summary['n_slides']}")
    print("\nImage audit status:")
    for k, v in summary["image_audit_status_counts"].items():
        print(f"  - {k}: {v}")
    print("\nInferred effective bit depths:")
    for k, v in summary["inferred_effective_bit_depth_counts"].items():
        print(f"  - {k}: {v}")
    print("\nContainer bit depths:")
    for k, v in summary["container_bit_depth_counts"].items():
        print(f"  - {k}: {v}")
    print("\nMean fractions:")
    print(f"  - values > 4095   : {summary['mean_values_above_4095_fraction']}")
    print(f"  - values > 16383  : {summary['mean_values_above_16383_fraction']}")
    print(f"  - 14-bit saturation: {summary['mean_saturation_fraction_14bit']}")
    print("\nOptics checks (O-4/O-5):")
    for k, v in summary.get("optics_checks", {}).items():
        print(f"  - {k}: {v}")
    if summary["errors"]:
        print("\nErrors:")
        for e in summary["errors"]:
            print(f"  - {e}")
    if summary["warnings"]:
        print("\nWarnings:")
        for w in summary["warnings"]:
            print(f"  - {w}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()

    config = load_optional_yaml_config(args.config)
    settings_resolution: Dict[str, Any] = {}
    expected_effective_bit_depth = int(
        resolve_setting(
            "expected_effective_bit_depth",
            args.expected_effective_bit_depth,
            config,
            ["tiff_audit", "expected_external_effective_bit_depth"],
            DEFAULT_EXPECTED_EXTERNAL_BIT_DEPTH,
            settings_resolution,
        )
    )
    expected_container_bit_depths = [
        int(v)
        for v in resolve_setting(
            "expected_container_bit_depths",
            None,
            config,
            ["tiff_audit", "expected_container_bit_depths"],
            sorted(DEFAULT_EXPECTED_CONTAINER_BIT_DEPTHS),
            settings_resolution,
        )
    ]

    with args.metadata.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    optics = metadata.get("optics") if isinstance(metadata, dict) else None
    if not isinstance(optics, dict):
        raise ValueError(f"Metadata {args.metadata} has no 'optics' block (schema 2.0 required).")
    optics_expectation, optics_record = build_optics_expectation(optics)
    if optics_expectation is None and args.strict:
        raise ValueError(
            "Optics block could not be parsed; O-4/O-5 cannot be evaluated: "
            f"{optics_record['validation']['errors']}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(args.manifest)

    per_image_rows: List[Dict[str, Any]] = []
    for i, manifest_row in enumerate(manifest_rows, start=1):
        row = compute_image_statistics(
            manifest_row,
            expected_effective_bit_depth=expected_effective_bit_depth,
            expected_container_bit_depths=expected_container_bit_depths,
            high_saturation_warning_threshold=args.high_saturation_warning_threshold,
            optics_expectation=optics_expectation,
            raw_root=args.raw_root,
        )
        per_image_rows.append(row)

        if i % 250 == 0:
            print(f"Audited {i}/{len(manifest_rows)} images...")

    per_slide_rows = aggregate_per_slide(per_image_rows)

    per_image_csv = output_dir / "tiff_audit_per_image.csv"
    per_slide_csv = output_dir / "tiff_audit_per_slide.csv"
    summary_json = output_dir / "tiff_audit_summary.json"
    histogram_path = output_dir / "intensity_histograms_raw.png"

    write_csv(per_image_rows, per_image_csv, PER_IMAGE_COLUMNS)
    write_csv(per_slide_rows, per_slide_csv, PER_SLIDE_COLUMNS)

    summary = summarize_audit(
        per_image_rows,
        per_slide_rows,
        manifest_path=args.manifest,
        config_path=args.config,
        expected_effective_bit_depth=expected_effective_bit_depth,
        optics_record=optics_record,
        settings_resolution=settings_resolution,
    )
    summary["outputs"] = {
        "per_image_csv": str(per_image_csv),
        "per_slide_csv": str(per_slide_csv),
        "summary_json": str(summary_json),
        "raw_intensity_histograms": str(histogram_path),
    }

    species_pixels = sample_pixels_for_histograms(
        per_image_rows,
        max_images_per_species=args.histogram_max_images_per_species,
        sample_pixels_per_image=args.histogram_sample_pixels_per_image,
        seed=args.random_seed,
    )
    plot_raw_intensity_histograms(
        species_pixels,
        histogram_path,
        expected_effective_bit_depth=expected_effective_bit_depth,
    )

    write_json(summary, summary_json)
    print_summary(summary, output_dir)

    if args.strict and summary["validation_status"] == "failed":
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
