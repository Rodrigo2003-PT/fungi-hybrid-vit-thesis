#!/usr/bin/env python3
"""
external_statistics.py

Compute final statistical summaries for external validation of the frozen
LSA-ConvTok Hybrid model under microscope-domain shift.

Scientific role
---------------
This script operates AFTER frozen-model inference has already been completed by
evaluate_external_generalization.py. It does not train, fine-tune, tune
thresholds, or select models. Its role is to quantify:

    1. Primary slide-level external-validation performance.
    2. Slide-level bootstrap confidence intervals.
    3. Secondary descriptive image-level performance.
    4. Subgroup/error-regime statistics, especially:
         - C. albicans with hyphae
         - C. albicans without hyphae
         - C. glabrata
    5. Calibration/confidence summaries.
    6. Optional raw TIFF/domain-shift summaries when audit files are provided.

Why slide-level bootstrap?
--------------------------
Images from the same slide are correlated and must not be treated as independent
biological observations for primary uncertainty estimation. The slide is the
independent biological/acquisition unit. Therefore, confidence intervals are
computed by resampling slides, not images.

Inputs
------
    - image_predictions.csv
    - slide_predictions.csv
    - external_manifest.csv
    - optional external_validation_config.yaml
    - optional tiff_audit_per_image.csv / tiff_audit_per_slide.csv

Outputs
-------
    output_dir/
      external_statistics_summary.json
      slide_metrics_with_ci.json
      image_metrics_descriptive.json
      bootstrap_distributions.csv
      subgroup_statistics.csv
      calibration_statistics.json
      confidence_by_slide.csv
      error_analysis_slide_level.csv
      optional_domain_shift_statistics.json

Usage
-----
    python external_statistics.py \\
        --image-predictions outputs_external/evaluation/locked_pixel_pipeline/image_predictions.csv \\
        --slide-predictions outputs_external/evaluation/locked_pixel_pipeline/slide_predictions.csv \\
        --manifest outputs_external/manifest/external_manifest.csv \\
        --config configs/external_validation_config.yaml \\
        --output-dir outputs_external/statistics/locked_pixel_pipeline \\
        --n-bootstrap 10000

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


SLIDE_METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "matthews_corrcoef",
    "weighted_f1",
    "auroc",
    "sensitivity_recall_Candida_albicans",
    "sensitivity_recall_Candida_glabrata",
    "specificity_Candida_albicans",
    "specificity_Candida_glabrata",
]

SUBGROUP_COLUMNS = [
    "subgroup",
    "n_slides",
    "n_images",
    "slide_accuracy",
    "image_accuracy_descriptive_only",
    "mean_true_class_probability_slide",
    "mean_confidence_slide",
    "mean_true_class_probability_image",
    "mean_confidence_image",
    "n_misclassified_slides",
    "misclassified_slide_ids",
]

CONFIDENCE_BY_SLIDE_COLUMNS = [
    "slide_id",
    "species_name",
    "true_label",
    "pred_label",
    "n_images",
    "confidence",
    "correct",
    "hyphae_status",
    "mean_prob_albicans",
    "mean_prob_glabrata",
    "true_class_probability",
    "margin_abs",
]

ERROR_ANALYSIS_COLUMNS = [
    "slide_id",
    "species_name",
    "true_label",
    "pred_label",
    "pred_species",
    "correct",
    "n_images",
    "confidence",
    "hyphae_status",
    "mean_prob_albicans",
    "mean_prob_glabrata",
    "median_prob_albicans",
    "median_prob_glabrata",
    "std_prob_albicans",
    "std_prob_glabrata",
    "error_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute slide-level confidence intervals, subgroup/error analysis, "
            "and descriptive statistics for external validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-predictions",
        required=True,
        type=Path,
        help="Path to image_predictions.csv from evaluate_external_generalization.py.",
    )
    parser.add_argument(
        "--slide-predictions",
        required=True,
        type=Path,
        help="Path to slide_predictions.csv from evaluate_external_generalization.py.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to external_manifest.csv.",
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
        help="Directory where statistics outputs will be written.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of slide-level bootstrap resamples for final CIs.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence interval level for percentile bootstrap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--stratified-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample slides with replacement within each true class.",
    )
    parser.add_argument(
        "--tiff-audit-per-image",
        type=Path,
        default=None,
        help="Optional tiff_audit_per_image.csv for domain-shift summaries.",
    )
    parser.add_argument(
        "--tiff-audit-per-slide",
        type=Path,
        default=None,
        help="Optional tiff_audit_per_slide.csv for domain-shift summaries.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort on severe consistency errors.",
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
            "WARNING: PyYAML not installed; continuing without parsing YAML config.",
            file=sys.stderr,
        )
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV file is empty: {path}")
    return rows


def write_csv_dicts(rows: List[Dict[str, Any]], path: Path, columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        if rows:
            columns = list(rows[0].keys())
        else:
            columns = []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def to_int(value: Any) -> int:
    return int(float(value))


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def validate_prediction_files(
    image_rows: List[Dict[str, str]],
    slide_rows: List[Dict[str, str]],
    manifest_rows: List[Dict[str, str]],
    *,
    strict: bool,
) -> List[str]:
    warnings_out: List[str] = []
    errors: List[str] = []

    required_image_cols = {
        "relative_path",
        "slide_id",
        "species_name",
        "true_label",
        "pred_label",
        "prob_albicans",
        "prob_glabrata",
        "confidence",
        "correct",
        "hyphae_status",
    }
    required_slide_cols = {
        "slide_id",
        "species_name",
        "true_label",
        "pred_label",
        "n_images",
        "mean_prob_albicans",
        "mean_prob_glabrata",
        "confidence",
        "correct",
        "hyphae_status",
    }

    missing_image = required_image_cols - set(image_rows[0].keys())
    missing_slide = required_slide_cols - set(slide_rows[0].keys())
    if missing_image:
        errors.append(f"image_predictions.csv missing columns: {sorted(missing_image)}")
    if missing_slide:
        errors.append(f"slide_predictions.csv missing columns: {sorted(missing_slide)}")

    manifest_image_paths = {row["relative_path"] for row in manifest_rows}
    pred_image_paths = {row["relative_path"] for row in image_rows}
    if manifest_image_paths != pred_image_paths:
        missing_pred = sorted(manifest_image_paths - pred_image_paths)
        extra_pred = sorted(pred_image_paths - manifest_image_paths)
        errors.append(
            f"Image prediction rows do not match manifest. "
            f"Missing={len(missing_pred)}, extra={len(extra_pred)}."
        )
        if missing_pred:
            warnings_out.append(f"First missing predictions: {missing_pred[:10]}")
        if extra_pred:
            warnings_out.append(f"First extra predictions: {extra_pred[:10]}")

    manifest_slides = {row["slide_id"] for row in manifest_rows}
    pred_slides = {row["slide_id"] for row in slide_rows}
    if manifest_slides != pred_slides:
        missing_slide_pred = sorted(manifest_slides - pred_slides)
        extra_slide_pred = sorted(pred_slides - manifest_slides)
        errors.append(
            f"Slide prediction rows do not match manifest. "
            f"Missing={len(missing_slide_pred)}, extra={len(extra_slide_pred)}."
        )

    # Validate one true label/species/hyphae status per slide.
    for sid in sorted(pred_slides):
        srows = [r for r in slide_rows if r["slide_id"] == sid]
        if len(srows) != 1:
            errors.append(f"Expected one slide prediction row for {sid}, got {len(srows)}.")

    if len(pred_slides) != 24:
        msg = f"Expected 24 external slides, observed {len(pred_slides)}."
        if strict:
            errors.append(msg)
        else:
            warnings_out.append(msg)

    if errors and strict:
        raise RuntimeError("Prediction validation failed:\n  - " + "\n  - ".join(errors))

    warnings_out.extend(errors)
    return warnings_out


def arrays_from_rows(rows: List[Dict[str, str]], *, probability_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([to_int(r["true_label"]) for r in rows], dtype=int)
    y_pred = np.array([to_int(r["pred_label"]) for r in rows], dtype=int)
    p_pos = np.array([to_float(r[probability_column]) for r in rows], dtype=float)
    return y_true, y_pred, p_pos


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    p_pos: Sequence[float],
    *,
    unit: str,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)

    out: Dict[str, Any] = {
        "unit": unit,
        "n": int(len(y_true)),
        "class_mapping": {
            "0": "Candida albicans",
            "1": "Candida glabrata",
            "positive_class_for_auc": "Candida glabrata",
        },
    }

    if len(y_true) == 0:
        return {**out, "error": "No observations."}

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    recalls = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    out["sensitivity_recall_per_class"] = {
        "Candida albicans": float(recalls[0]),
        "Candida glabrata": float(recalls[1]),
    }
    out["sensitivity_recall_Candida_albicans"] = float(recalls[0])
    out["sensitivity_recall_Candida_glabrata"] = float(recalls[1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix_labels"] = ["Candida albicans", "Candida glabrata"]
    out["confusion_matrix"] = cm.astype(int).tolist()

    specificities: Dict[str, float] = {}
    flat_specificities: Dict[str, float] = {}
    for cls, cls_name in [(0, "Candida albicans"), (1, "Candida glabrata")]:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        denom = tn + fp
        spec = float(tn / denom) if denom > 0 else float("nan")
        specificities[cls_name] = spec
        flat_specificities[f"specificity_{cls_name.replace(' ', '_')}"] = spec
    out["specificity_per_class"] = specificities
    out.update(flat_specificities)

    try:
        if len(np.unique(y_true)) == 2 and np.all(np.isfinite(p_pos)):
            out["auroc"] = float(roc_auc_score(y_true, p_pos))
        else:
            out["auroc"] = None
            out["auroc_note"] = "AUROC undefined because only one class is present or probabilities are non-finite."
    except Exception as exc:
        out["auroc"] = None
        out["auroc_note"] = f"AUROC could not be computed: {exc}"

    return out


def metric_value_for_bootstrap(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, p_pos: np.ndarray) -> float:
    m = compute_binary_metrics(y_true, y_pred, p_pos, unit="bootstrap_slide")
    value = m.get(metric_name)
    if value is None:
        return float("nan")
    return float(value)


def slide_level_bootstrap_ci(
    slide_rows: List[Dict[str, str]],
    *,
    n_bootstrap: int,
    ci: float,
    seed: int,
    stratified: bool,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[Dict[str, Any]]]:
    """
    Bootstrap confidence intervals by resampling slides, not images.

    Parameters
    ----------
    slide_rows:
        One row per slide.
    stratified:
        If true, sample with replacement within each true class, preserving
        external cohort class counts in every bootstrap replicate.
    """
    rng = np.random.default_rng(seed)

    y_true_all, y_pred_all, p_pos_all = arrays_from_rows(
        slide_rows,
        probability_column="mean_prob_glabrata",
    )

    slide_indices_by_class: Dict[int, np.ndarray] = {}
    for cls in sorted(np.unique(y_true_all)):
        slide_indices_by_class[int(cls)] = np.where(y_true_all == int(cls))[0]

    alpha = 1.0 - float(ci)
    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    distributions: Dict[str, List[float]] = {name: [] for name in SLIDE_METRIC_NAMES}
    rows_out: List[Dict[str, Any]] = []

    all_indices = np.arange(len(slide_rows))

    for b in range(int(n_bootstrap)):
        if stratified:
            sampled_parts = []
            for cls, idxs in slide_indices_by_class.items():
                sampled = rng.choice(idxs, size=len(idxs), replace=True)
                sampled_parts.append(sampled)
            boot_idx = np.concatenate(sampled_parts)
        else:
            boot_idx = rng.choice(all_indices, size=len(all_indices), replace=True)

        y_true = y_true_all[boot_idx]
        y_pred = y_pred_all[boot_idx]
        p_pos = p_pos_all[boot_idx]

        metrics = compute_binary_metrics(y_true, y_pred, p_pos, unit="bootstrap_slide")
        boot_row: Dict[str, Any] = {"bootstrap_index": b}

        for name in SLIDE_METRIC_NAMES:
            value = metrics.get(name)
            if value is None:
                value = float("nan")
            value = float(value)
            distributions[name].append(value)
            boot_row[name] = value

        rows_out.append(boot_row)

    ci_out: Dict[str, Dict[str, Optional[float]]] = {}
    point_metrics = compute_binary_metrics(y_true_all, y_pred_all, p_pos_all, unit="slide")

    for name, values in distributions.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        point = point_metrics.get(name)
        point_float = None if point is None else float(point)

        if arr.size == 0:
            ci_out[name] = {
                "point_estimate": point_float,
                "ci_lower": None,
                "ci_upper": None,
                "bootstrap_mean": None,
                "n_valid_bootstrap": 0,
            }
        else:
            ci_out[name] = {
                "point_estimate": point_float,
                "ci_lower": float(np.percentile(arr, lo_q)),
                "ci_upper": float(np.percentile(arr, hi_q)),
                "bootstrap_mean": float(np.mean(arr)),
                "n_valid_bootstrap": int(arr.size),
            }

    return ci_out, rows_out


def true_class_probability(row: Dict[str, Any], *, level: str) -> float:
    label = to_int(row["true_label"])
    if level == "slide":
        return to_float(row["mean_prob_albicans"]) if label == 0 else to_float(row["mean_prob_glabrata"])
    if level == "image":
        return to_float(row["prob_albicans"]) if label == 0 else to_float(row["prob_glabrata"])
    raise ValueError(f"Unknown level: {level}")


def confidence_by_slide(slide_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in slide_rows:
        p0 = to_float(row["mean_prob_albicans"])
        p1 = to_float(row["mean_prob_glabrata"])
        true_prob = true_class_probability(row, level="slide")
        out.append(
            {
                "slide_id": row["slide_id"],
                "species_name": row["species_name"],
                "true_label": to_int(row["true_label"]),
                "pred_label": to_int(row["pred_label"]),
                "n_images": to_int(row["n_images"]),
                "confidence": to_float(row["confidence"]),
                "correct": to_int(row["correct"]),
                "hyphae_status": row["hyphae_status"],
                "mean_prob_albicans": p0,
                "mean_prob_glabrata": p1,
                "true_class_probability": true_prob,
                "margin_abs": float(abs(p1 - p0)),
            }
        )
    return out


def error_analysis_slide_level(slide_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in slide_rows:
        true_label = to_int(row["true_label"])
        pred_label = to_int(row["pred_label"])
        correct = int(true_label == pred_label)

        if correct:
            error_type = "correct"
        elif true_label == 0 and pred_label == 1:
            error_type = "albicans_misclassified_as_glabrata"
        elif true_label == 1 and pred_label == 0:
            error_type = "glabrata_misclassified_as_albicans"
        else:
            error_type = "unknown_error"

        out.append(
            {
                "slide_id": row["slide_id"],
                "species_name": row["species_name"],
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_species": row.get("pred_species", ""),
                "correct": correct,
                "n_images": to_int(row["n_images"]),
                "confidence": to_float(row["confidence"]),
                "hyphae_status": row["hyphae_status"],
                "mean_prob_albicans": to_float(row["mean_prob_albicans"]),
                "mean_prob_glabrata": to_float(row["mean_prob_glabrata"]),
                "median_prob_albicans": to_float(row.get("median_prob_albicans", "")),
                "median_prob_glabrata": to_float(row.get("median_prob_glabrata", "")),
                "std_prob_albicans": to_float(row.get("std_prob_albicans", "")),
                "std_prob_glabrata": to_float(row.get("std_prob_glabrata", "")),
                "error_type": error_type,
            }
        )
    return out


def subgroup_statistics(
    image_rows: List[Dict[str, str]],
    slide_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    subgroup_defs = {
        "Candida albicans with hyphae": lambda r: r["species_name"] == "Candida albicans" and r["hyphae_status"] == "hyphae",
        "Candida albicans without hyphae": lambda r: r["species_name"] == "Candida albicans" and r["hyphae_status"] == "no_hyphae",
        "Candida glabrata": lambda r: r["species_name"] == "Candida glabrata",
    }

    out: List[Dict[str, Any]] = []

    for subgroup, predicate in subgroup_defs.items():
        srows = [r for r in slide_rows if predicate(r)]
        irows = [r for r in image_rows if predicate(r)]

        n_slides = len(srows)
        n_images = len(irows)

        if n_slides:
            slide_correct = np.array([to_int(r["correct"]) for r in srows], dtype=float)
            slide_acc = float(slide_correct.mean())
            slide_true_probs = np.array([true_class_probability(r, level="slide") for r in srows], dtype=float)
            slide_conf = np.array([to_float(r["confidence"]) for r in srows], dtype=float)
            misclassified = [r["slide_id"] for r in srows if to_int(r["correct"]) == 0]
        else:
            slide_acc = float("nan")
            slide_true_probs = np.array([], dtype=float)
            slide_conf = np.array([], dtype=float)
            misclassified = []

        if n_images:
            image_correct = np.array([to_int(r["correct"]) for r in irows], dtype=float)
            image_acc = float(image_correct.mean())
            image_true_probs = np.array([true_class_probability(r, level="image") for r in irows], dtype=float)
            image_conf = np.array([to_float(r["confidence"]) for r in irows], dtype=float)
        else:
            image_acc = float("nan")
            image_true_probs = np.array([], dtype=float)
            image_conf = np.array([], dtype=float)

        out.append(
            {
                "subgroup": subgroup,
                "n_slides": n_slides,
                "n_images": n_images,
                "slide_accuracy": slide_acc,
                "image_accuracy_descriptive_only": image_acc,
                "mean_true_class_probability_slide": float(np.nanmean(slide_true_probs)) if slide_true_probs.size else float("nan"),
                "mean_confidence_slide": float(np.nanmean(slide_conf)) if slide_conf.size else float("nan"),
                "mean_true_class_probability_image": float(np.nanmean(image_true_probs)) if image_true_probs.size else float("nan"),
                "mean_confidence_image": float(np.nanmean(image_conf)) if image_conf.size else float("nan"),
                "n_misclassified_slides": len(misclassified),
                "misclassified_slide_ids": "|".join(misclassified),
            }
        )

    return out


def expected_calibration_error(
    y_true_correct: np.ndarray,
    confidence: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute ECE using confidence and correctness.

    This is descriptive only. No calibration model is fitted on the external
    validation labels.
    """
    y_true_correct = np.asarray(y_true_correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)

    finite = np.isfinite(confidence) & np.isfinite(y_true_correct)
    y_true_correct = y_true_correct[finite]
    confidence = confidence[finite]

    if confidence.size == 0:
        return {"ece": None, "bins": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n == 0:
            bin_rows.append(
                {
                    "bin_index": i,
                    "bin_lower": float(lo),
                    "bin_upper": float(hi),
                    "n": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "abs_gap": None,
                }
            )
            continue

        mean_conf = float(confidence[mask].mean())
        acc = float(y_true_correct[mask].mean())
        gap = abs(acc - mean_conf)
        ece += (n / confidence.size) * gap
        bin_rows.append(
            {
                "bin_index": i,
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "n": n,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "abs_gap": float(gap),
            }
        )

    return {"ece": float(ece), "bins": bin_rows}


def calibration_statistics(
    image_rows: List[Dict[str, str]],
    slide_rows: List[Dict[str, str]],
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    y_true_img = np.array([to_int(r["true_label"]) for r in image_rows], dtype=int)
    p_gla_img = np.array([to_float(r["prob_glabrata"]) for r in image_rows], dtype=float)
    conf_img = np.array([to_float(r["confidence"]) for r in image_rows], dtype=float)
    correct_img = np.array([to_int(r["correct"]) for r in image_rows], dtype=int)

    y_true_slide = np.array([to_int(r["true_label"]) for r in slide_rows], dtype=int)
    p_gla_slide = np.array([to_float(r["mean_prob_glabrata"]) for r in slide_rows], dtype=float)
    conf_slide = np.array([to_float(r["confidence"]) for r in slide_rows], dtype=float)
    correct_slide = np.array([to_int(r["correct"]) for r in slide_rows], dtype=int)

    out: Dict[str, Any] = {
        "interpretation": (
            "Calibration metrics are descriptive. No recalibration model or "
            "temperature scaling was fitted on the external validation labels."
        )
    }

    try:
        out["brier_score_image_positive_glabrata"] = float(brier_score_loss(y_true_img, p_gla_img))
    except Exception as exc:
        out["brier_score_image_positive_glabrata"] = None
        out["brier_score_image_note"] = str(exc)

    try:
        out["brier_score_slide_positive_glabrata"] = float(brier_score_loss(y_true_slide, p_gla_slide))
    except Exception as exc:
        out["brier_score_slide_positive_glabrata"] = None
        out["brier_score_slide_note"] = str(exc)

    out["ece_image"] = expected_calibration_error(correct_img, conf_img, n_bins=n_bins)
    out["ece_slide"] = expected_calibration_error(correct_slide, conf_slide, n_bins=n_bins)

    for level, conf, corr in [("image", conf_img, correct_img), ("slide", conf_slide, correct_slide)]:
        correct_conf = conf[corr == 1]
        incorrect_conf = conf[corr == 0]
        out[f"mean_confidence_{level}_correct"] = float(np.mean(correct_conf)) if correct_conf.size else None
        out[f"mean_confidence_{level}_incorrect"] = float(np.mean(incorrect_conf)) if incorrect_conf.size else None
        out[f"n_{level}_correct"] = int(correct_conf.size)
        out[f"n_{level}_incorrect"] = int(incorrect_conf.size)

    return out


def summarize_numeric(values: List[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "p1": None, "p50": None, "p99": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def optional_domain_shift_statistics(
    tiff_audit_per_image: Optional[Path],
    tiff_audit_per_slide: Optional[Path],
) -> Dict[str, Any]:
    """
    Summarize external raw TIFF audit outputs when available.

    This function does not compare against original-domain data unless such data
    are supplied in future extensions. It provides external-domain descriptive
    photometric statistics for reporting and downstream comparison.
    """
    out: Dict[str, Any] = {
        "available": False,
        "interpretation": (
            "These are descriptive external-domain TIFF statistics. They should "
            "be compared against original-domain train/test audit statistics when "
            "available; do not use arbitrary hard thresholds as proof of domain shift."
        ),
    }

    if tiff_audit_per_image is None or not tiff_audit_per_image.exists():
        out["note"] = "No per-image TIFF audit CSV provided."
        return out

    rows = read_csv_dicts(tiff_audit_per_image)
    out["available"] = True
    out["per_image_audit_path"] = str(tiff_audit_per_image)
    out["n_images"] = len(rows)

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
        "dynamic_range_fraction_14bit",
    ]

    out["numeric_summaries"] = {}
    for field in numeric_fields:
        out["numeric_summaries"][field] = summarize_numeric([safe_float(r.get(field)) for r in rows])

    out["audit_status_counts"] = dict(Counter(r.get("audit_status", "") for r in rows))
    out["container_bit_depth_counts"] = dict(Counter(r.get("container_bit_depth", "") for r in rows))
    out["inferred_effective_bit_depth_counts"] = dict(Counter(r.get("inferred_effective_bit_depth", "") for r in rows))

    # Per-species summaries for the most relevant fields.
    species_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        species_groups[r.get("species_name", "")].append(r)

    out["species_summaries"] = {}
    for species, group in species_groups.items():
        out["species_summaries"][species] = {
            field: summarize_numeric([safe_float(r.get(field)) for r in group])
            for field in ["mean_pixel", "std_pixel", "p50", "p99", "values_above_4095_fraction", "saturation_fraction_14bit"]
        }

    if tiff_audit_per_slide is not None and tiff_audit_per_slide.exists():
        slide_rows = read_csv_dicts(tiff_audit_per_slide)
        out["per_slide_audit_path"] = str(tiff_audit_per_slide)
        out["n_audited_slides"] = len(slide_rows)
        out["slide_audit_status_counts"] = dict(Counter(r.get("audit_status", "") for r in slide_rows))

    return out


def create_summary(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    slide_metrics_ci: Dict[str, Dict[str, Optional[float]]],
    image_metrics: Dict[str, Any],
    subgroup_rows: List[Dict[str, Any]],
    calibration: Dict[str, Any],
    domain_shift: Dict[str, Any],
    validation_warnings: List[str],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "script": "external_statistics.py",
        "created_unix_time": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "scientific_role": (
            "Post-inference statistical analysis for external validation. Primary "
            "uncertainty estimation uses slide-level bootstrap resampling."
        ),
        "inputs": {
            "image_predictions": str(args.image_predictions),
            "slide_predictions": str(args.slide_predictions),
            "manifest": str(args.manifest),
            "config": str(args.config) if args.config else None,
            "tiff_audit_per_image": str(args.tiff_audit_per_image) if args.tiff_audit_per_image else None,
            "tiff_audit_per_slide": str(args.tiff_audit_per_slide) if args.tiff_audit_per_slide else None,
        },
        "bootstrap": {
            "resampling_unit": "slide",
            "n_bootstrap": int(args.n_bootstrap),
            "confidence_level": float(args.ci),
            "seed": int(args.seed),
            "stratified_by_true_class": bool(args.stratified_bootstrap),
            "images_bootstrapped_independently": False,
        },
        "primary_slide_metrics_with_ci": slide_metrics_ci,
        "secondary_image_metrics_descriptive": image_metrics,
        "subgroup_statistics": subgroup_rows,
        "calibration_statistics": calibration,
        "domain_shift_statistics": domain_shift,
        "validation_warnings": validation_warnings,
        "scientific_guardrails": {
            "no_training_performed": True,
            "no_fine_tuning_performed": True,
            "no_threshold_tuning_performed": True,
            "slide_level_metrics_primary": True,
            "image_level_metrics_secondary_only": True,
            "hyphae_status_used_only_for_subgroup_analysis": True,
        },
        "outputs": {
            "external_statistics_summary_json": str(args.output_dir / "external_statistics_summary.json"),
            "slide_metrics_with_ci_json": str(args.output_dir / "slide_metrics_with_ci.json"),
            "image_metrics_descriptive_json": str(args.output_dir / "image_metrics_descriptive.json"),
            "bootstrap_distributions_csv": str(args.output_dir / "bootstrap_distributions.csv"),
            "subgroup_statistics_csv": str(args.output_dir / "subgroup_statistics.csv"),
            "calibration_statistics_json": str(args.output_dir / "calibration_statistics.json"),
            "confidence_by_slide_csv": str(args.output_dir / "confidence_by_slide.csv"),
            "error_analysis_slide_level_csv": str(args.output_dir / "error_analysis_slide_level.csv"),
            "optional_domain_shift_statistics_json": str(args.output_dir / "optional_domain_shift_statistics.json"),
        },
    }


def print_summary(summary: Dict[str, Any]) -> None:
    slide = summary["primary_slide_metrics_with_ci"]
    img = summary["secondary_image_metrics_descriptive"]

    print("\n" + "=" * 80)
    print("EXTERNAL VALIDATION STATISTICS COMPLETE")
    print("=" * 80)
    print(f"Output directory : {Path(summary['outputs']['external_statistics_summary_json']).parent}")
    print(f"Bootstrap        : {summary['bootstrap']['n_bootstrap']} slide-level resamples")
    print("\nPrimary slide-level metrics with CI:")
    for name in ["accuracy", "balanced_accuracy", "matthews_corrcoef", "weighted_f1", "auroc"]:
        ci = slide.get(name, {})
        print(
            f"  - {name}: {ci.get('point_estimate')} "
            f"[{ci.get('ci_lower')}, {ci.get('ci_upper')}]"
        )
    print("\nSecondary image-level descriptive metrics:")
    for name in ["accuracy", "balanced_accuracy", "matthews_corrcoef", "weighted_f1", "auroc"]:
        print(f"  - {name}: {img.get(name)}")
    print("\nSubgroups:")
    for row in summary["subgroup_statistics"]:
        print(
            f"  - {row['subgroup']}: n_slides={row['n_slides']}, "
            f"slide_acc={row['slide_accuracy']}, misclassified={row['n_misclassified_slides']}"
        )
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_optional_yaml_config(args.config)

    image_rows = read_csv_dicts(args.image_predictions)
    slide_rows = read_csv_dicts(args.slide_predictions)
    manifest_rows = read_csv_dicts(args.manifest)

    validation_warnings = validate_prediction_files(
        image_rows,
        slide_rows,
        manifest_rows,
        strict=args.strict,
    )

    # Metrics.
    y_true_img, y_pred_img, p_gla_img = arrays_from_rows(
        image_rows,
        probability_column="prob_glabrata",
    )
    image_metrics = compute_binary_metrics(
        y_true_img,
        y_pred_img,
        p_gla_img,
        unit="image",
    )
    image_metrics["interpretation"] = (
        "Image-level metrics are descriptive only because images from the same "
        "slide are correlated."
    )

    slide_metrics_ci, bootstrap_rows = slide_level_bootstrap_ci(
        slide_rows,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
        stratified=args.stratified_bootstrap,
    )

    subgroup_rows = subgroup_statistics(image_rows, slide_rows)
    calib = calibration_statistics(image_rows, slide_rows, n_bins=10)
    confidence_rows = confidence_by_slide(slide_rows)
    error_rows = error_analysis_slide_level(slide_rows)
    domain_shift = optional_domain_shift_statistics(
        args.tiff_audit_per_image,
        args.tiff_audit_per_slide,
    )

    elapsed = time.time() - start

    summary = create_summary(
        args=args,
        config=config,
        slide_metrics_ci=slide_metrics_ci,
        image_metrics=image_metrics,
        subgroup_rows=subgroup_rows,
        calibration=calib,
        domain_shift=domain_shift,
        validation_warnings=validation_warnings,
        elapsed_seconds=elapsed,
    )

    # Write outputs.
    write_json(summary, args.output_dir / "external_statistics_summary.json")
    write_json(slide_metrics_ci, args.output_dir / "slide_metrics_with_ci.json")
    write_json(image_metrics, args.output_dir / "image_metrics_descriptive.json")
    write_csv_dicts(bootstrap_rows, args.output_dir / "bootstrap_distributions.csv")
    write_csv_dicts(subgroup_rows, args.output_dir / "subgroup_statistics.csv", SUBGROUP_COLUMNS)
    write_json(calib, args.output_dir / "calibration_statistics.json")
    write_csv_dicts(confidence_rows, args.output_dir / "confidence_by_slide.csv", CONFIDENCE_BY_SLIDE_COLUMNS)
    write_csv_dicts(error_rows, args.output_dir / "error_analysis_slide_level.csv", ERROR_ANALYSIS_COLUMNS)
    write_json(domain_shift, args.output_dir / "optional_domain_shift_statistics.json")

    print_summary(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
