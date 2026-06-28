#!/usr/bin/env python3
"""
evaluate_external_generalization.py

Evaluate the frozen LSA-ConvTok Hybrid model on the independent external
new-microscope validation cohort.

Scientific role
---------------
This script performs frozen-model inference only. It does not train, fine-tune,
select models, tune thresholds, or select preprocessing based on performance.

The external cohort is treated as an acquisition-domain validation set. The
primary statistical unit is the slide. Image-level predictions are generated
first, then aggregated to slide-level predictions using a pre-specified mean
probability rule:

    p_slide,c = mean_i(p_image_i,c), for all images i in slide s
    y_slide   = argmax_c(p_slide,c)

Primary metrics are slide-level. Image-level metrics are reported only as
secondary/descriptive because images from the same slide are correlated.

Inputs
------
    - final_model_lsa_convtok.pth
    - optional final_model_checkpoint.pth for provenance equivalence check
    - external_manifest.csv
    - preprocessed external .npy directory from preprocess_external.py
    - optional external_validation_config.yaml

Outputs
-------
    output_dir/
      image_predictions.csv
      slide_predictions.csv
      metrics_image_level.json
      metrics_slide_level.json
      confusion_matrix_image.csv
      confusion_matrix_slide.csv
      external_validation_metadata.json

Usage
-----
    python evaluate_external_generalization.py \\
        --checkpoint final_model_lsa_convtok.pth \\
        --provenance-checkpoint final_model_checkpoint.pth \\
        --manifest outputs_external/manifest/external_manifest.csv \\
        --preprocessed-dir outputs_external/preprocessed/locked_pixel_pipeline \\
        --config configs/external_validation_config.yaml \\
        --output-dir outputs_external/evaluation/locked_pixel_pipeline \\
        --batch-size 64

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

try:
    from load_frozen_hybrid import load_frozen_hybrid, save_loader_metadata
except Exception as exc:
    raise ImportError(
        "Could not import load_frozen_hybrid.py. Ensure it is in the same "
        "directory or on PYTHONPATH."
    ) from exc


IMAGE_PREDICTION_COLUMNS = [
    "relative_path",
    "filename",
    "slide_id",
    "species_name",
    "true_label",
    "pred_label",
    "pred_species",
    "prob_albicans",
    "prob_glabrata",
    "confidence",
    "correct",
    "hyphae_status",
    "preprocessing_mode",
]

SLIDE_PREDICTION_COLUMNS = [
    "slide_id",
    "species_name",
    "true_label",
    "pred_label",
    "pred_species",
    "n_images",
    "mean_prob_albicans",
    "mean_prob_glabrata",
    "median_prob_albicans",
    "median_prob_glabrata",
    "std_prob_albicans",
    "std_prob_glabrata",
    "confidence",
    "correct",
    "hyphae_status",
    "preprocessing_mode",
]

CONFUSION_COLUMNS = ["true_label", "pred_label", "count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen-model external validation on preprocessed external "
            "microscopy images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Primary frozen inference checkpoint: final_model_lsa_convtok.pth.",
    )
    parser.add_argument(
        "--provenance-checkpoint",
        type=Path,
        default=None,
        help="Optional final_model_checkpoint.pth for provenance equivalence check.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to external_manifest.csv.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        type=Path,
        help="Directory containing preprocessed .npy files mirroring manifest paths.",
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
        help="Directory where predictions, metrics, and metadata will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num-workers > 0.",
    )
    parser.add_argument(
        "--input-distribution-max-images",
        type=int,
        default=256,
        help=(
            "Maximum number of preprocessed arrays to re-read for the metadata input-distribution "
            "summary. This avoids a full second pass over all .npy files in Colab. Set 0 to skip; "
            "set -1 to summarize all images as in the original implementation."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', 'cuda', or a valid torch device string.",
    )
    parser.add_argument(
        "--preprocessing-mode",
        type=str,
        default="locked_pixel_pipeline",
        help="Name of preprocessing mode used to produce the input .npy files.",
    )
    parser.add_argument(
        "--arrays-already-standardized",
        action="store_true",
        help=(
            "Set only if preprocess_external.py was run with --save-standardized. "
            "Default false: this script applies checkpoint mean/std to [0,1] arrays."
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort on missing files or severe inconsistencies.",
    )
    parser.add_argument(
        "--smoke-test-loader",
        action="store_true",
        help="Run model loader dummy forward smoke test.",
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


def npy_path_for_row(row: Dict[str, str], preprocessed_dir: Path) -> Path:
    return preprocessed_dir / Path(row["relative_path"]).with_suffix(".npy")


class ExternalNPYDataset(Dataset):
    """
    Deterministic dataset for external preprocessed .npy images.

    The recommended primary pipeline saves [0,1] float32 arrays in
    preprocess_external.py, then applies the training-domain checkpoint mean/std
    here immediately before inference. This is intentional: external validation
    should test the frozen model under the same input standardization used during
    training, rather than recomputing external-domain statistics that would
    partially adapt the inputs to the new microscope domain.

    Scientific interpretation
    -------------------------
    - Applying checkpoint mean/std is part of the frozen inference pipeline.
    - Recomputing mean/std on the external cohort would be a domain-specific
      preprocessing adaptation and should not be used for the primary result.
    - Any alternative external-domain normalization should be reported only as a
      separate sensitivity analysis, not as the primary external validation.
    """

    def __init__(
        self,
        manifest_rows: List[Dict[str, str]],
        preprocessed_dir: Path,
        mean: Sequence[float],
        std: Sequence[float],
        *,
        arrays_already_standardized: bool = False,
        strict: bool = True,
    ) -> None:
        self.rows = manifest_rows
        self.preprocessed_dir = preprocessed_dir
        self.mean = [float(v) for v in mean]
        self.std = [float(v) for v in std]
        self.arrays_already_standardized = bool(arrays_already_standardized)
        self.strict = bool(strict)

        if len(self.mean) != 1 or len(self.std) != 1:
            raise ValueError(
                "ExternalNPYDataset currently expects one-channel mean/std, got "
                f"mean={self.mean}, std={self.std}"
            )
        if self.std[0] <= 0:
            raise ValueError(f"Checkpoint std must be positive, got {self.std[0]}")

        missing = []
        for row in self.rows:
            p = npy_path_for_row(row, self.preprocessed_dir)
            if not p.exists():
                missing.append(str(p))
        if missing and strict:
            preview = "\n".join(missing[:20])
            raise FileNotFoundError(
                f"{len(missing)} preprocessed .npy files are missing. "
                f"First missing files:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        npy_path = npy_path_for_row(row, self.preprocessed_dir)
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing preprocessed file: {npy_path}")

        arr = np.load(str(npy_path), mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D preprocessed array, got {arr.shape}: {npy_path}")
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32, copy=False)

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in preprocessed array: {npy_path}")

        if not self.arrays_already_standardized:
            arr = (arr - self.mean[0]) / self.std[0]

        # Model expects BCHW; sample returns CHW.
        tensor = torch.from_numpy(arr[None, :, :].astype(np.float32, copy=False))

        return {
            "image": tensor,
            "label": int(row["label"]),
            "relative_path": row["relative_path"],
            "filename": row.get("filename", Path(row["relative_path"]).name),
            "slide_id": row["slide_id"],
            "species_name": row["species_name"],
            "hyphae_status": row["hyphae_status"],
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    return {
        "image": images,
        "label": labels,
        "relative_path": [b["relative_path"] for b in batch],
        "filename": [b["filename"] for b in batch],
        "slide_id": [b["slide_id"] for b in batch],
        "species_name": [b["species_name"] for b in batch],
        "hyphae_status": [b["hyphae_status"] for b in batch],
    }


def run_inference(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
) -> List[Dict[str, Any]]:
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_batch,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(dataset, **loader_kwargs)

    rows: List[Dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(
                    f"Expected logits shape [B,2], got {tuple(logits.shape)}"
                )
            probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = batch["label"].cpu().numpy()

            for i in range(len(labels)):
                prob_alb = float(probs[i, 0])
                prob_gla = float(probs[i, 1])
                pred = int(preds[i])
                true = int(labels[i])
                rows.append(
                    {
                        "relative_path": batch["relative_path"][i],
                        "filename": batch["filename"][i],
                        "slide_id": batch["slide_id"][i],
                        "species_name": batch["species_name"][i],
                        "true_label": true,
                        "pred_label": pred,
                        "pred_species": "Candida albicans" if pred == 0 else "Candida glabrata",
                        "prob_albicans": prob_alb,
                        "prob_glabrata": prob_gla,
                        "confidence": float(max(prob_alb, prob_gla)),
                        "correct": int(pred == true),
                        "hyphae_status": batch["hyphae_status"][i],
                    }
                )

            if batch_idx % 25 == 0:
                print(f"Inference batches completed: {batch_idx}")

    return rows


def aggregate_to_slides(
    image_rows: List[Dict[str, Any]],
    *,
    preprocessing_mode: str,
) -> List[Dict[str, Any]]:
    slide_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        slide_to_rows[row["slide_id"]].append(row)

    slide_rows: List[Dict[str, Any]] = []

    for slide_id, rows in sorted(slide_to_rows.items()):
        true_labels = sorted({int(r["true_label"]) for r in rows})
        species_values = sorted({str(r["species_name"]) for r in rows})
        hyphae_values = sorted({str(r["hyphae_status"]) for r in rows})

        if len(true_labels) != 1:
            raise RuntimeError(f"Mixed true labels within slide {slide_id}: {true_labels}")
        if len(species_values) != 1:
            raise RuntimeError(f"Mixed species within slide {slide_id}: {species_values}")
        if len(hyphae_values) != 1:
            raise RuntimeError(f"Mixed hyphae status within slide {slide_id}: {hyphae_values}")

        probs_alb = np.array([float(r["prob_albicans"]) for r in rows], dtype=float)
        probs_gla = np.array([float(r["prob_glabrata"]) for r in rows], dtype=float)

        mean_probs = np.array([probs_alb.mean(), probs_gla.mean()], dtype=float)
        pred_label = int(np.argmax(mean_probs))
        true_label = int(true_labels[0])

        slide_rows.append(
            {
                "slide_id": slide_id,
                "species_name": species_values[0],
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_species": "Candida albicans" if pred_label == 0 else "Candida glabrata",
                "n_images": len(rows),
                "mean_prob_albicans": float(probs_alb.mean()),
                "mean_prob_glabrata": float(probs_gla.mean()),
                "median_prob_albicans": float(np.median(probs_alb)),
                "median_prob_glabrata": float(np.median(probs_gla)),
                "std_prob_albicans": float(np.std(probs_alb)),
                "std_prob_glabrata": float(np.std(probs_gla)),
                "confidence": float(np.max(mean_probs)),
                "correct": int(pred_label == true_label),
                "hyphae_status": hyphae_values[0],
                "preprocessing_mode": preprocessing_mode,
            }
        )

    return slide_rows


def compute_binary_metrics(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    prob_positive: Sequence[float],
    *,
    unit_name: str,
) -> Dict[str, Any]:
    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(pred_labels, dtype=int)
    p_pos = np.asarray(prob_positive, dtype=float)

    metrics: Dict[str, Any] = {
        "unit": unit_name,
        "n": int(len(y_true)),
        "class_mapping": {
            "0": "Candida albicans",
            "1": "Candida glabrata",
            "positive_class_for_auc": "Candida glabrata",
        },
    }

    if len(y_true) == 0:
        raise ValueError(f"No samples available for {unit_name} metrics.")

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class recall/sensitivity.
    recalls = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    metrics["sensitivity_recall_per_class"] = {
        "Candida albicans": float(recalls[0]),
        "Candida glabrata": float(recalls[1]),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix_labels"] = ["Candida albicans", "Candida glabrata"]
    metrics["confusion_matrix"] = cm.astype(int).tolist()

    # Specificity per class: treat each class as one-vs-rest.
    specificities: Dict[str, float] = {}
    for cls, cls_name in [(0, "Candida albicans"), (1, "Candida glabrata")]:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        denom = tn + fp
        specificities[cls_name] = float(tn / denom) if denom > 0 else float("nan")
    metrics["specificity_per_class"] = specificities

    try:
        if len(np.unique(y_true)) == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, p_pos))
        else:
            metrics["auroc"] = None
            metrics["auroc_note"] = "Only one class present; AUROC undefined."
    except Exception as exc:
        metrics["auroc"] = None
        metrics["auroc_note"] = f"AUROC could not be computed: {exc}"

    return metrics


def write_csv(rows: List[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def confusion_matrix_long_rows(true_labels: Sequence[int], pred_labels: Sequence[int]) -> List[Dict[str, Any]]:
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    rows = []
    for t in [0, 1]:
        for p in [0, 1]:
            rows.append({"true_label": t, "pred_label": p, "count": int(cm[t, p])})
    return rows


def add_preprocessing_mode_to_image_rows(rows: List[Dict[str, Any]], mode: str) -> None:
    for row in rows:
        row["preprocessing_mode"] = mode


def validate_manifest_cohort(rows: List[Dict[str, str]], strict: bool = True) -> List[str]:
    warnings_out: List[str] = []
    slide_to_labels: Dict[str, set] = defaultdict(set)
    slide_to_species: Dict[str, set] = defaultdict(set)
    slide_to_hyphae: Dict[str, set] = defaultdict(set)

    for row in rows:
        sid = row["slide_id"]
        slide_to_labels[sid].add(row["label"])
        slide_to_species[sid].add(row["species_name"])
        slide_to_hyphae[sid].add(row["hyphae_status"])

    errors = []
    for sid in sorted(slide_to_labels):
        if len(slide_to_labels[sid]) != 1:
            errors.append(f"Mixed labels in slide {sid}: {sorted(slide_to_labels[sid])}")
        if len(slide_to_species[sid]) != 1:
            errors.append(f"Mixed species in slide {sid}: {sorted(slide_to_species[sid])}")
        if len(slide_to_hyphae[sid]) != 1:
            errors.append(f"Mixed hyphae status in slide {sid}: {sorted(slide_to_hyphae[sid])}")

    n_slides = len(slide_to_labels)
    if n_slides != 24:
        msg = f"Expected 24 external slides, observed {n_slides}."
        if strict:
            errors.append(msg)
        else:
            warnings_out.append(msg)

    species_slide_counts = Counter(next(iter(v)) for v in slide_to_species.values())
    if species_slide_counts.get("Candida albicans", 0) != 12:
        msg = f"Expected 12 Candida albicans slides, observed {species_slide_counts.get('Candida albicans', 0)}."
        if strict:
            errors.append(msg)
        else:
            warnings_out.append(msg)
    if species_slide_counts.get("Candida glabrata", 0) != 12:
        msg = f"Expected 12 Candida glabrata slides, observed {species_slide_counts.get('Candida glabrata', 0)}."
        if strict:
            errors.append(msg)
        else:
            warnings_out.append(msg)

    if errors:
        raise RuntimeError("Manifest cohort validation failed:\n  - " + "\n  - ".join(errors))

    return warnings_out



def summarize_preprocessed_input_distribution(
    manifest_rows: List[Dict[str, str]],
    preprocessed_dir: Path,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    arrays_already_standardized: bool,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Summarize the numeric distribution of the model inputs used for inference.

    This is a diagnostic/provenance summary only. It must not be used to tune
    preprocessing, thresholds, or model selection. Its purpose is to document
    whether external inputs are shifted after the frozen checkpoint
    standardization.
    """
    raw_min_values: List[float] = []
    raw_max_values: List[float] = []
    raw_mean_values: List[float] = []
    raw_std_values: List[float] = []
    standardized_min_values: List[float] = []
    standardized_max_values: List[float] = []
    standardized_mean_values: List[float] = []
    standardized_std_values: List[float] = []

    rows = manifest_rows if max_images is None else manifest_rows[: max(0, int(max_images))]
    missing = 0

    for row in rows:
        npy_path = npy_path_for_row(row, preprocessed_dir)
        if not npy_path.exists():
            missing += 1
            continue

        arr = np.load(str(npy_path), mmap_mode="r").astype(np.float32, copy=False)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            continue

        raw_min_values.append(float(np.min(arr)))
        raw_max_values.append(float(np.max(arr)))
        raw_mean_values.append(float(np.mean(arr)))
        raw_std_values.append(float(np.std(arr)))

        if arrays_already_standardized:
            arr_std = arr
        else:
            arr_std = (arr - float(mean[0])) / float(std[0])

        standardized_min_values.append(float(np.min(arr_std)))
        standardized_max_values.append(float(np.max(arr_std)))
        standardized_mean_values.append(float(np.mean(arr_std)))
        standardized_std_values.append(float(np.std(arr_std)))

    def describe(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"min": None, "median": None, "max": None, "mean": None}
        a = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(a)),
            "median": float(np.median(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
        }

    theoretical_standardized_range_for_unit_interval = None
    if not arrays_already_standardized:
        theoretical_standardized_range_for_unit_interval = {
            "input_0_maps_to": float((0.0 - float(mean[0])) / float(std[0])),
            "input_1_maps_to": float((1.0 - float(mean[0])) / float(std[0])),
        }

    return {
        "scientific_role": (
            "Diagnostic provenance summary of preprocessed [0,1] arrays and "
            "their values after frozen checkpoint standardization. This summary "
            "is not used for preprocessing selection, threshold tuning, or model "
            "selection."
        ),
        "n_manifest_images": len(manifest_rows),
        "n_images_summarized": len(raw_mean_values),
        "n_missing_npy_files": int(missing),
        "max_images_requested": max_images,
        "arrays_already_standardized": bool(arrays_already_standardized),
        "checkpoint_mean": [float(v) for v in mean],
        "checkpoint_std": [float(v) for v in std],
        "theoretical_standardized_range_for_unit_interval": theoretical_standardized_range_for_unit_interval,
        "pre_standardization_array_min": describe(raw_min_values),
        "pre_standardization_array_max": describe(raw_max_values),
        "pre_standardization_array_mean": describe(raw_mean_values),
        "pre_standardization_array_std": describe(raw_std_values),
        "model_input_standardized_min": describe(standardized_min_values),
        "model_input_standardized_max": describe(standardized_max_values),
        "model_input_standardized_mean": describe(standardized_mean_values),
        "model_input_standardized_std": describe(standardized_std_values),
    }


def create_metadata(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    loader_metadata: Dict[str, Any],
    manifest_rows: List[Dict[str, str]],
    image_rows: List[Dict[str, Any]],
    slide_rows: List[Dict[str, Any]],
    elapsed_seconds: float,
    manifest_warnings: List[str],
    input_distribution_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    species_image_counts = Counter(row["species_name"] for row in manifest_rows)
    slide_ids = sorted({row["slide_id"] for row in manifest_rows})
    species_slide_counts = Counter()
    hyphae_slide_counts = Counter()

    for sid in slide_ids:
        r = next(row for row in manifest_rows if row["slide_id"] == sid)
        species_slide_counts[r["species_name"]] += 1
        hyphae_slide_counts[r["hyphae_status"]] += 1

    return {
        "schema_version": "1.0",
        "script": "evaluate_external_generalization.py",
        "created_unix_time": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "scientific_role": (
            "Frozen external validation under microscope-domain shift. No "
            "training, fine-tuning, threshold tuning, or model selection was "
            "performed on the external cohort."
        ),
        "inputs": {
            "checkpoint": str(args.checkpoint),
            "provenance_checkpoint": str(args.provenance_checkpoint) if args.provenance_checkpoint else None,
            "manifest": str(args.manifest),
            "preprocessed_dir": str(args.preprocessed_dir),
            "config": str(args.config) if args.config else None,
        },
        "preprocessing": {
            "mode": args.preprocessing_mode,
            "arrays_already_standardized": bool(args.arrays_already_standardized),
            "checkpoint_mean_std_applied_in_dataset": not bool(args.arrays_already_standardized),
            "standardization_policy": {
                "mean_std_source": (
                    "frozen training-domain checkpoint"
                    if not bool(args.arrays_already_standardized)
                    else "preprocessed arrays already checkpoint-standardized"
                ),
                "rationale": (
                    "The primary external-validation analysis applies the same "
                    "training-domain checkpoint mean/std used by the frozen model. "
                    "External-domain mean/std are not recomputed because that would "
                    "constitute a domain-specific input adaptation and would no "
                    "longer measure raw frozen-model generalization."
                ),
                "external_domain_statistics_used_for_standardization": False,
                "sensitivity_analysis_required_for_alternative_standardization": True,
            },
            "input_distribution_summary": input_distribution_summary,
        },
        "model_loader_metadata": loader_metadata,
        "cohort": {
            "n_images": len(manifest_rows),
            "n_slides": len(slide_ids),
            "species_image_counts": dict(sorted(species_image_counts.items())),
            "species_slide_counts": dict(sorted(species_slide_counts.items())),
            "hyphae_slide_counts": dict(sorted(hyphae_slide_counts.items())),
            "manifest_validation_warnings": manifest_warnings,
        },
        "evaluation_policy": {
            "primary_unit": "slide",
            "secondary_unit": "image",
            "slide_aggregation_rule": "mean_probability",
            "threshold_tuning_performed": False,
            "model_selection_performed": False,
            "external_dataset_split_for_training": False,
        },
        "outputs": {
            "image_predictions_csv": str(args.output_dir / "image_predictions.csv"),
            "slide_predictions_csv": str(args.output_dir / "slide_predictions.csv"),
            "metrics_image_level_json": str(args.output_dir / "metrics_image_level.json"),
            "metrics_slide_level_json": str(args.output_dir / "metrics_slide_level.json"),
            "confusion_matrix_image_csv": str(args.output_dir / "confusion_matrix_image.csv"),
            "confusion_matrix_slide_csv": str(args.output_dir / "confusion_matrix_slide.csv"),
            "metadata_json": str(args.output_dir / "external_validation_metadata.json"),
        },
        "scientific_guardrails": {
            "no_retraining": True,
            "no_fine_tuning": True,
            "no_threshold_tuning": True,
            "slide_level_primary": True,
            "image_level_descriptive_only": True,
            "hyphae_status_used_only_for_subgroup_analysis": True,
        },
    }


def print_summary(
    output_dir: Path,
    image_metrics: Dict[str, Any],
    slide_metrics: Dict[str, Any],
    n_images: int,
    n_slides: int,
    elapsed: float,
) -> None:
    print("\n" + "=" * 80)
    print("EXTERNAL GENERALIZATION EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Output directory : {output_dir}")
    print(f"Images evaluated : {n_images}")
    print(f"Slides evaluated : {n_slides}")
    print(f"Elapsed seconds  : {elapsed:.2f}")
    print("\nPrimary slide-level metrics:")
    print(f"  - Accuracy          : {slide_metrics.get('accuracy')}")
    print(f"  - Balanced accuracy : {slide_metrics.get('balanced_accuracy')}")
    print(f"  - MCC               : {slide_metrics.get('matthews_corrcoef')}")
    print(f"  - Weighted F1       : {slide_metrics.get('weighted_f1')}")
    print(f"  - AUROC             : {slide_metrics.get('auroc')}")
    print("\nSecondary image-level metrics:")
    print(f"  - Accuracy          : {image_metrics.get('accuracy')}")
    print(f"  - Balanced accuracy : {image_metrics.get('balanced_accuracy')}")
    print(f"  - MCC               : {image_metrics.get('matthews_corrcoef')}")
    print(f"  - Weighted F1       : {image_metrics.get('weighted_f1')}")
    print(f"  - AUROC             : {image_metrics.get('auroc')}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()
    start = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_optional_yaml_config(args.config)
    manifest_rows = read_manifest(args.manifest)
    manifest_warnings = validate_manifest_cohort(manifest_rows, strict=args.strict)

    bundle = load_frozen_hybrid(
        checkpoint_path=args.checkpoint,
        provenance_checkpoint_path=args.provenance_checkpoint,
        config_path=args.config,
        device=args.device,
        strict=args.strict,
        run_dummy_forward=args.smoke_test_loader,
    )

    model = bundle["model"]
    device = bundle["device"]
    mean = bundle["mean"]
    std = bundle["std"]

    dataset = ExternalNPYDataset(
        manifest_rows=manifest_rows,
        preprocessed_dir=args.preprocessed_dir,
        mean=mean,
        std=std,
        arrays_already_standardized=args.arrays_already_standardized,
        strict=args.strict,
    )

    if args.input_distribution_max_images == 0:
        input_distribution_summary = {
            "skipped": True,
            "reason": "Skipped by --input-distribution-max-images 0 to avoid an additional pre-inference pass over .npy files.",
        }
    else:
        max_images_for_summary = None if args.input_distribution_max_images < 0 else int(args.input_distribution_max_images)
        input_distribution_summary = summarize_preprocessed_input_distribution(
            manifest_rows=manifest_rows,
            preprocessed_dir=args.preprocessed_dir,
            mean=mean,
            std=std,
            arrays_already_standardized=args.arrays_already_standardized,
            max_images=max_images_for_summary,
        )

    image_rows = run_inference(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    add_preprocessing_mode_to_image_rows(image_rows, args.preprocessing_mode)

    slide_rows = aggregate_to_slides(
        image_rows,
        preprocessing_mode=args.preprocessing_mode,
    )

    # Metrics.
    y_true_img = [int(r["true_label"]) for r in image_rows]
    y_pred_img = [int(r["pred_label"]) for r in image_rows]
    p_gla_img = [float(r["prob_glabrata"]) for r in image_rows]

    y_true_slide = [int(r["true_label"]) for r in slide_rows]
    y_pred_slide = [int(r["pred_label"]) for r in slide_rows]
    p_gla_slide = [float(r["mean_prob_glabrata"]) for r in slide_rows]

    image_metrics = compute_binary_metrics(
        y_true_img,
        y_pred_img,
        p_gla_img,
        unit_name="image",
    )
    image_metrics["interpretation"] = (
        "Image-level metrics are secondary and descriptive because images from "
        "the same slide are correlated."
    )

    slide_metrics = compute_binary_metrics(
        y_true_slide,
        y_pred_slide,
        p_gla_slide,
        unit_name="slide",
    )
    slide_metrics["interpretation"] = (
        "Slide-level metrics are the primary external-validation result because "
        "the slide is the independent biological/acquisition unit."
    )
    slide_metrics["slide_aggregation_rule"] = "mean_probability"

    # Outputs.
    image_predictions_csv = args.output_dir / "image_predictions.csv"
    slide_predictions_csv = args.output_dir / "slide_predictions.csv"
    metrics_image_json = args.output_dir / "metrics_image_level.json"
    metrics_slide_json = args.output_dir / "metrics_slide_level.json"
    cm_image_csv = args.output_dir / "confusion_matrix_image.csv"
    cm_slide_csv = args.output_dir / "confusion_matrix_slide.csv"
    metadata_json = args.output_dir / "external_validation_metadata.json"

    write_csv(image_rows, image_predictions_csv, IMAGE_PREDICTION_COLUMNS)
    write_csv(slide_rows, slide_predictions_csv, SLIDE_PREDICTION_COLUMNS)
    write_json(image_metrics, metrics_image_json)
    write_json(slide_metrics, metrics_slide_json)
    write_csv(confusion_matrix_long_rows(y_true_img, y_pred_img), cm_image_csv, CONFUSION_COLUMNS)
    write_csv(confusion_matrix_long_rows(y_true_slide, y_pred_slide), cm_slide_csv, CONFUSION_COLUMNS)

    elapsed = time.time() - start
    metadata = create_metadata(
        args=args,
        config=config,
        loader_metadata=bundle["metadata"],
        manifest_rows=manifest_rows,
        image_rows=image_rows,
        slide_rows=slide_rows,
        elapsed_seconds=elapsed,
        manifest_warnings=manifest_warnings,
        input_distribution_summary=input_distribution_summary,
    )
    write_json(metadata, metadata_json)

    print_summary(
        args.output_dir,
        image_metrics=image_metrics,
        slide_metrics=slide_metrics,
        n_images=len(image_rows),
        n_slides=len(slide_rows),
        elapsed=elapsed,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
