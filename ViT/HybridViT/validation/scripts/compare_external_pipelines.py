#!/usr/bin/env python3
"""
compare_external_pipelines_refactored.py

General paired slide-level comparison for the locked primary external-validation
pipeline versus one or more post hoc sensitivity-analysis pipelines.

Scientific role
---------------
This script does not train, fine-tune, tune thresholds, or select a model. It
compares already-computed frozen-model external evaluations on the same slides.
The comparison is paired at slide level because the slide is the independent
biological/acquisition unit.

Primary interpretation
----------------------
- ``locked_pixel_pipeline`` remains the primary external-validation result.
- Additional pipelines such as ``scale_matched_physical_pipeline`` and
  ``photometric_mean_std_matched_pipeline`` are post hoc diagnostic
  sensitivity analyses.
- Sensitivity pipelines must not replace the primary external-validation
  estimate and must not be used for model selection or threshold tuning.
- Pairwise deltas help assess whether a specific hypothesized mechanism
  (for example geometric-scale mismatch or photometric mismatch) plausibly
  contributed to external-domain failure.

Outputs
-------
output_dir/
  multi_pipeline_slide_comparison.csv
  multi_pipeline_summary.json
  pairwise_comparison__<comparison_label>.csv
  pairwise_summary__<comparison_label>.json

Optional backward-compatible two-pipeline mode is supported via the legacy
``--sensitivity-slide-predictions`` / ``--sensitivity-label`` arguments.

Author: Rodrigo Sá / refactored implementation
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

PRIMARY_DEFAULT_LABEL = "locked_pixel_pipeline"
LEGACY_DEFAULT_SENSITIVITY_LABEL = "scale_matched_physical_pipeline"

REQUIRED_SLIDE_COLUMNS = {
    "slide_id",
    "species_name",
    "true_label",
    "pred_label",
    "confidence",
    "correct",
    "mean_prob_albicans",
    "mean_prob_glabrata",
    "hyphae_status",
}

BASE_MULTI_COLUMNS = [
    "slide_id",
    "species_name",
    "hyphae_status",
    "true_label",
]

PAIRWISE_BASE_COLUMNS = [
    "slide_id",
    "species_name",
    "hyphae_status",
    "true_label",
    "primary_pred_label",
    "comparison_pred_label",
    "primary_correct",
    "comparison_correct",
    "correctness_change",
    "primary_prob_albicans",
    "primary_prob_glabrata",
    "comparison_prob_albicans",
    "comparison_prob_glabrata",
    "primary_true_class_probability",
    "comparison_true_class_probability",
    "delta_true_class_probability",
    "primary_confidence",
    "comparison_confidence",
    "delta_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired slide-level comparison between the primary locked external-"
            "validation pipeline and one or more post hoc sensitivity-analysis "
            "pipelines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--primary-slide-predictions", required=True, type=Path)
    parser.add_argument("--primary-label", default=PRIMARY_DEFAULT_LABEL)

    # Preferred generalized interface.
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help=(
            "Comparison pipeline in the form LABEL=PATH_TO_SLIDE_PREDICTIONS_CSV. "
            "May be supplied multiple times. Example: "
            "--comparison scale_matched_physical_pipeline=outputs/.../slide_predictions.csv"
        ),
    )

    # Backward-compatible legacy interface.
    parser.add_argument("--sensitivity-slide-predictions", type=Path, default=None)
    parser.add_argument("--sensitivity-label", default=LEGACY_DEFAULT_SENSITIVITY_LABEL)

    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    out = out.strip("._-")
    return out or "pipeline"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def to_int(x: Any) -> int:
    return int(float(x))


def to_float(x: Any) -> float:
    return float(x)


def safe_auc(y_true: Sequence[int], prob_pos: Sequence[float]) -> Optional[float]:
    try:
        if len(set(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, prob_pos))
    except Exception:
        return None


def percentile_ci(values: np.ndarray, ci: float = 0.95) -> Tuple[float, float]:
    alpha = 1.0 - ci
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def exact_sign_flip_pvalue(values: Sequence[float]) -> Optional[float]:
    """Two-sided exact sign-flip p-value for paired mean difference.

    Intended as descriptive only for small n (the expected use here is 24 slides).
    Zero differences are removed.
    """
    vals = np.asarray([float(v) for v in values if abs(float(v)) > 1e-12], dtype=float)
    n = len(vals)
    if n == 0:
        return None
    if n > 24:
        return None
    observed = abs(float(np.mean(vals)))
    count = 0
    total = 2 ** n
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        stat = abs(float(np.mean(vals * np.asarray(signs))))
        if stat >= observed - 1e-15:
            count += 1
    return float(count / total)


def exact_mcnemar_pvalue_from_changes(changes: Sequence[str]) -> Optional[float]:
    improved = sum(1 for c in changes if c == "improved")
    worsened = sum(1 for c in changes if c == "worsened")
    n = improved + worsened
    if n == 0:
        return None
    k = min(improved, worsened)
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return float(min(1.0, 2.0 * prob))


def bootstrap_delta_metric(
    primary_correct: Sequence[int],
    comparison_correct: Sequence[int],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    p = np.asarray(primary_correct, dtype=float)
    c = np.asarray(comparison_correct, dtype=float)
    n = len(p)
    deltas = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(c[idx]) - np.mean(p[idx]))
    lo, hi = percentile_ci(deltas)
    return {
        "delta_accuracy_comparison_minus_primary_mean_bootstrap": float(np.mean(deltas)),
        "delta_accuracy_95ci_percentile": [lo, hi],
        "n_bootstrap": int(n_bootstrap),
    }


def validate_rows(name: str, rows: List[Dict[str, str]]) -> None:
    missing = REQUIRED_SLIDE_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"{name} slide_predictions missing columns: {sorted(missing)}")


def build_slide_map(name: str, rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    validate_rows(name, rows)
    by_slide: Dict[str, Dict[str, str]] = {}
    for row in rows:
        slide_id = row["slide_id"]
        if slide_id in by_slide:
            raise ValueError(f"{name} contains duplicate slide_id: {slide_id}")
        by_slide[slide_id] = row
    return by_slide


def parse_comparisons(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    comparisons: List[Tuple[str, Path]] = []
    seen_labels = set()

    for spec in args.comparison:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --comparison value: {spec!r}. Expected LABEL=PATH_TO_SLIDE_PREDICTIONS_CSV"
            )
        label, path_str = spec.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
        if not label:
            raise ValueError(f"Invalid --comparison label in spec: {spec!r}")
        if label == args.primary_label:
            raise ValueError("Comparison label must differ from --primary-label")
        if label in seen_labels:
            raise ValueError(f"Duplicate comparison label: {label}")
        seen_labels.add(label)
        comparisons.append((label, path))

    if args.sensitivity_slide_predictions is not None:
        legacy_label = args.sensitivity_label
        if legacy_label == args.primary_label:
            raise ValueError("--sensitivity-label must differ from --primary-label")
        if legacy_label in seen_labels:
            raise ValueError(
                f"Legacy sensitivity label {legacy_label!r} duplicates a generalized --comparison label"
            )
        comparisons.append((legacy_label, args.sensitivity_slide_predictions))
        seen_labels.add(legacy_label)

    if not comparisons:
        raise ValueError(
            "At least one comparison pipeline is required. Use one or more --comparison LABEL=PATH "
            "arguments, or the legacy --sensitivity-slide-predictions argument."
        )

    return comparisons


def build_primary_reference(primary_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    p_map = build_slide_map("primary", primary_rows)
    ref: Dict[str, Dict[str, Any]] = {}
    for slide_id, row in p_map.items():
        ref[slide_id] = {
            "slide_id": slide_id,
            "species_name": row["species_name"],
            "hyphae_status": row.get("hyphae_status", ""),
            "true_label": to_int(row["true_label"]),
            "primary_pred_label": to_int(row["pred_label"]),
            "primary_correct": to_int(row["correct"]),
            "primary_prob_albicans": to_float(row["mean_prob_albicans"]),
            "primary_prob_glabrata": to_float(row["mean_prob_glabrata"]),
            "primary_confidence": to_float(row["confidence"]),
        }
        true_label = ref[slide_id]["true_label"]
        ref[slide_id]["primary_true_class_probability"] = (
            ref[slide_id]["primary_prob_albicans"] if true_label == 0 else ref[slide_id]["primary_prob_glabrata"]
        )
    return ref


def ensure_same_slide_set(
    primary_ref: Mapping[str, Dict[str, Any]],
    comparison_rows: List[Dict[str, str]],
    comparison_name: str,
) -> Dict[str, Dict[str, str]]:
    c_map = build_slide_map(comparison_name, comparison_rows)
    p_slides = set(primary_ref.keys())
    c_slides = set(c_map.keys())
    if p_slides != c_slides:
        raise ValueError(
            f"Primary and {comparison_name} slide sets differ: "
            f"missing_in_comparison={sorted(p_slides - c_slides)[:10]}, "
            f"extra_in_comparison={sorted(c_slides - p_slides)[:10]}"
        )

    for slide_id, c_row in c_map.items():
        p = primary_ref[slide_id]
        if to_int(c_row["true_label"]) != p["true_label"]:
            raise ValueError(f"True-label mismatch for slide {slide_id} in {comparison_name}")
        if c_row["species_name"] != p["species_name"]:
            raise ValueError(f"Species mismatch for slide {slide_id} in {comparison_name}")
    return c_map


def compute_metrics_from_rows(rows: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    y = [to_int(r["true_label"]) for r in rows]
    pred = [to_int(r[f"{prefix}_pred_label"]) for r in rows]
    prob_g = [to_float(r[f"{prefix}_prob_glabrata"]) for r in rows]
    return {
        "n_slides": len(rows),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "recall_Candida_albicans_label0": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "recall_Candida_glabrata_label1": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "auroc_glabrata_positive": safe_auc(y, prob_g),
    }


def build_pairwise_rows(
    primary_ref: Mapping[str, Dict[str, Any]],
    comparison_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    c_map = ensure_same_slide_set(primary_ref, comparison_rows, "comparison")
    paired: List[Dict[str, Any]] = []
    for slide_id in sorted(primary_ref.keys()):
        p = primary_ref[slide_id]
        c = c_map[slide_id]
        comparison_pred_label = to_int(c["pred_label"])
        comparison_correct = to_int(c["correct"])
        comparison_prob_a = to_float(c["mean_prob_albicans"])
        comparison_prob_g = to_float(c["mean_prob_glabrata"])
        comparison_confidence = to_float(c["confidence"])
        comparison_true_prob = comparison_prob_a if p["true_label"] == 0 else comparison_prob_g

        if p["primary_correct"] == 0 and comparison_correct == 1:
            change = "improved"
        elif p["primary_correct"] == 1 and comparison_correct == 0:
            change = "worsened"
        elif p["primary_correct"] == 1 and comparison_correct == 1:
            change = "both_correct"
        else:
            change = "both_wrong"

        paired.append({
            "slide_id": slide_id,
            "species_name": p["species_name"],
            "hyphae_status": p.get("hyphae_status", ""),
            "true_label": p["true_label"],
            "primary_pred_label": p["primary_pred_label"],
            "comparison_pred_label": comparison_pred_label,
            "primary_correct": p["primary_correct"],
            "comparison_correct": comparison_correct,
            "correctness_change": change,
            "primary_prob_albicans": p["primary_prob_albicans"],
            "primary_prob_glabrata": p["primary_prob_glabrata"],
            "comparison_prob_albicans": comparison_prob_a,
            "comparison_prob_glabrata": comparison_prob_g,
            "primary_true_class_probability": p["primary_true_class_probability"],
            "comparison_true_class_probability": comparison_true_prob,
            "delta_true_class_probability": comparison_true_prob - p["primary_true_class_probability"],
            "primary_confidence": p["primary_confidence"],
            "comparison_confidence": comparison_confidence,
            "delta_confidence": comparison_confidence - p["primary_confidence"],
        })
    return paired


def subgroup_summary_pairwise(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["species_name"] == "Candida albicans":
            key = f"Candida albicans/{r.get('hyphae_status', '')}"
        else:
            key = "Candida glabrata"
        groups[key].append(r)

    out: List[Dict[str, Any]] = []
    for key, g_rows in sorted(groups.items()):
        p_acc = float(np.mean([to_int(r["primary_correct"]) for r in g_rows]))
        c_acc = float(np.mean([to_int(r["comparison_correct"]) for r in g_rows]))
        out.append({
            "subgroup": key,
            "n_slides": len(g_rows),
            "primary_accuracy": p_acc,
            "comparison_accuracy": c_acc,
            "delta_accuracy": c_acc - p_acc,
            "mean_delta_true_class_probability": float(np.mean([to_float(r["delta_true_class_probability"]) for r in g_rows])),
            "improved_slides": [r["slide_id"] for r in g_rows if r["correctness_change"] == "improved"],
            "worsened_slides": [r["slide_id"] for r in g_rows if r["correctness_change"] == "worsened"],
        })
    return out


def build_multi_table(
    primary_ref: Mapping[str, Dict[str, Any]],
    comparison_maps: Mapping[str, Dict[str, Dict[str, str]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    columns = list(BASE_MULTI_COLUMNS)
    columns.extend([
        "primary_pred_label",
        "primary_correct",
        "primary_prob_albicans",
        "primary_prob_glabrata",
        "primary_true_class_probability",
        "primary_confidence",
    ])
    for label in comparison_maps.keys():
        slug = slugify(label)
        columns.extend([
            f"{slug}_pred_label",
            f"{slug}_correct",
            f"{slug}_prob_albicans",
            f"{slug}_prob_glabrata",
            f"{slug}_true_class_probability",
            f"{slug}_confidence",
            f"delta_true_class_probability__{slug}_minus_primary",
            f"delta_confidence__{slug}_minus_primary",
            f"correctness_change__{slug}_vs_primary",
        ])

    rows_out: List[Dict[str, Any]] = []
    for slide_id in sorted(primary_ref.keys()):
        p = primary_ref[slide_id]
        row: Dict[str, Any] = {
            "slide_id": slide_id,
            "species_name": p["species_name"],
            "hyphae_status": p.get("hyphae_status", ""),
            "true_label": p["true_label"],
            "primary_pred_label": p["primary_pred_label"],
            "primary_correct": p["primary_correct"],
            "primary_prob_albicans": p["primary_prob_albicans"],
            "primary_prob_glabrata": p["primary_prob_glabrata"],
            "primary_true_class_probability": p["primary_true_class_probability"],
            "primary_confidence": p["primary_confidence"],
        }
        for label, c_map in comparison_maps.items():
            slug = slugify(label)
            c = c_map[slide_id]
            c_pred = to_int(c["pred_label"])
            c_correct = to_int(c["correct"])
            c_prob_a = to_float(c["mean_prob_albicans"])
            c_prob_g = to_float(c["mean_prob_glabrata"])
            c_conf = to_float(c["confidence"])
            c_true_prob = c_prob_a if p["true_label"] == 0 else c_prob_g
            if p["primary_correct"] == 0 and c_correct == 1:
                change = "improved"
            elif p["primary_correct"] == 1 and c_correct == 0:
                change = "worsened"
            elif p["primary_correct"] == 1 and c_correct == 1:
                change = "both_correct"
            else:
                change = "both_wrong"
            row.update({
                f"{slug}_pred_label": c_pred,
                f"{slug}_correct": c_correct,
                f"{slug}_prob_albicans": c_prob_a,
                f"{slug}_prob_glabrata": c_prob_g,
                f"{slug}_true_class_probability": c_true_prob,
                f"{slug}_confidence": c_conf,
                f"delta_true_class_probability__{slug}_minus_primary": c_true_prob - p["primary_true_class_probability"],
                f"delta_confidence__{slug}_minus_primary": c_conf - p["primary_confidence"],
                f"correctness_change__{slug}_vs_primary": change,
            })
        rows_out.append(row)
    return rows_out, columns


def summarize_pairwise(
    paired_rows: List[Dict[str, Any]],
    *,
    primary_label: str,
    comparison_label: str,
    primary_slide_predictions_path: Path,
    comparison_slide_predictions_path: Path,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    primary_metrics = compute_metrics_from_rows(paired_rows, "primary")
    comparison_metrics = compute_metrics_from_rows(paired_rows, "comparison")
    deltas = {
        key: (
            None if primary_metrics.get(key) is None or comparison_metrics.get(key) is None
            else float(comparison_metrics[key] - primary_metrics[key])
        )
        for key in primary_metrics.keys()
        if key != "n_slides"
    }
    delta_true_probs = [to_float(r["delta_true_class_probability"]) for r in paired_rows]
    changes = [str(r["correctness_change"]) for r in paired_rows]
    primary_correct = [to_int(r["primary_correct"]) for r in paired_rows]
    comparison_correct = [to_int(r["comparison_correct"]) for r in paired_rows]

    return {
        "schema_version": "2.0",
        "scientific_role": "post_hoc_paired_slide_level_sensitivity_comparison",
        "primary_label": primary_label,
        "comparison_label": comparison_label,
        "primary_slide_predictions": str(primary_slide_predictions_path),
        "comparison_slide_predictions": str(comparison_slide_predictions_path),
        "interpretation_guardrail": (
            f"{primary_label} remains the primary external-validation result. "
            f"{comparison_label} is a post hoc diagnostic sensitivity analysis and must not be used "
            "for model selection, threshold tuning, or as the headline external-validation estimate."
        ),
        "n_paired_slides": len(paired_rows),
        "primary_metrics": primary_metrics,
        "comparison_metrics": comparison_metrics,
        "metric_deltas_comparison_minus_primary": deltas,
        "correctness_change_counts": dict(sorted(Counter(changes).items())),
        "mcnemar_exact_pvalue_descriptive": exact_mcnemar_pvalue_from_changes(changes),
        "mean_delta_true_class_probability": float(np.mean(delta_true_probs)),
        "median_delta_true_class_probability": float(np.median(delta_true_probs)),
        "sign_flip_exact_pvalue_for_mean_delta_true_class_probability_descriptive": exact_sign_flip_pvalue(delta_true_probs),
        "bootstrap_delta_accuracy": bootstrap_delta_metric(
            primary_correct,
            comparison_correct,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "subgroup_summary": subgroup_summary_pairwise(paired_rows),
        "recommended_thesis_language": {
            "if_performance_improves": (
                f"The {comparison_label} sensitivity analysis improved slide-level recognition "
                "relative to the locked primary pipeline, suggesting that the hypothesized "
                "mechanism motivating this sensitivity analysis plausibly contributed to the "
                "external-domain failure. Because the analysis is post hoc, its performance "
                "must be interpreted diagnostically rather than as a corrected primary "
                "external-validation score."
            ),
            "if_performance_does_not_improve": (
                f"The {comparison_label} sensitivity analysis did not recover performance "
                "relative to the locked primary pipeline, suggesting that the hypothesized "
                "mechanism motivating this sensitivity analysis is insufficient on its own "
                "to explain the external-domain failure. Other acquisition-domain factors "
                "likely contributed."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    comparisons = parse_comparisons(args)

    primary_rows = read_csv(args.primary_slide_predictions)
    primary_ref = build_primary_reference(primary_rows)

    comparison_maps: Dict[str, Dict[str, Dict[str, str]]] = {}
    pairwise_rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    pairwise_summaries: Dict[str, Dict[str, Any]] = {}
    primary_metrics_for_multi = compute_metrics_from_rows(list(primary_ref.values()), "primary")

    for label, path in comparisons:
        c_rows = read_csv(path)
        c_map = ensure_same_slide_set(primary_ref, c_rows, label)
        comparison_maps[label] = c_map
        pairwise_rows = build_pairwise_rows(primary_ref, c_rows)
        pairwise_rows_by_label[label] = pairwise_rows
        pairwise_summaries[label] = summarize_pairwise(
            pairwise_rows,
            primary_label=args.primary_label,
            comparison_label=label,
            primary_slide_predictions_path=args.primary_slide_predictions,
            comparison_slide_predictions_path=path,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Combined slide table.
    multi_rows, multi_columns = build_multi_table(primary_ref, comparison_maps)
    write_csv(multi_rows, args.output_dir / "multi_pipeline_slide_comparison.csv", multi_columns)

    # Pairwise outputs.
    pairwise_manifest: List[Dict[str, Any]] = []
    for label, path in comparisons:
        label_slug = slugify(label)
        pair_csv = args.output_dir / f"pairwise_comparison__{label_slug}.csv"
        pair_json = args.output_dir / f"pairwise_summary__{label_slug}.json"
        write_csv(pairwise_rows_by_label[label], pair_csv, PAIRWISE_BASE_COLUMNS)
        write_json(pairwise_summaries[label], pair_json)
        pairwise_manifest.append({
            "comparison_label": label,
            "comparison_label_slug": label_slug,
            "comparison_slide_predictions": str(path),
            "pairwise_comparison_csv": str(pair_csv),
            "pairwise_summary_json": str(pair_json),
        })

    # Multi-summary.
    comparison_metrics_summary: Dict[str, Any] = {}
    pairwise_delta_summary: Dict[str, Any] = {}
    for label in comparison_maps.keys():
        summary = pairwise_summaries[label]
        comparison_metrics_summary[label] = summary["comparison_metrics"]
        pairwise_delta_summary[label] = {
            "metric_deltas_comparison_minus_primary": summary["metric_deltas_comparison_minus_primary"],
            "correctness_change_counts": summary["correctness_change_counts"],
            "bootstrap_delta_accuracy": summary["bootstrap_delta_accuracy"],
            "mcnemar_exact_pvalue_descriptive": summary["mcnemar_exact_pvalue_descriptive"],
            "mean_delta_true_class_probability": summary["mean_delta_true_class_probability"],
            "median_delta_true_class_probability": summary["median_delta_true_class_probability"],
            "sign_flip_exact_pvalue_for_mean_delta_true_class_probability_descriptive": summary[
                "sign_flip_exact_pvalue_for_mean_delta_true_class_probability_descriptive"
            ],
        }

    multi_summary = {
        "schema_version": "2.0",
        "scientific_role": "paired_slide_level_multi_pipeline_sensitivity_comparison",
        "primary_label": args.primary_label,
        "primary_slide_predictions": str(args.primary_slide_predictions),
        "comparison_pipelines": [
            {"label": label, "slide_predictions": str(path)} for label, path in comparisons
        ],
        "interpretation_guardrail": (
            f"{args.primary_label} remains the primary external-validation result. "
            "All other pipelines summarized here are post hoc sensitivity analyses and must "
            "not replace the primary external-validation estimate or be used for model or "
            "threshold selection."
        ),
        "n_paired_slides": len(primary_ref),
        "outputs": {
            "multi_pipeline_slide_comparison_csv": str(args.output_dir / "multi_pipeline_slide_comparison.csv"),
            "pairwise_outputs": pairwise_manifest,
        },
        "primary_metrics": primary_metrics_for_multi,
        "comparison_metrics": comparison_metrics_summary,
        "pairwise_deltas_vs_primary": pairwise_delta_summary,
        "recommended_thesis_language": {
            "general": (
                "Sensitivity analyses should be reported as diagnostic follow-up analyses "
                "performed after observing the primary external-validation failure. They "
                "clarify plausible mechanisms of failure but do not redefine the primary "
                "external-validation performance."
            ),
            "scale_matched_physical_pipeline": (
                "If scale matching improves performance, geometric scale mismatch plausibly "
                "contributed to failure. If it does not improve performance, physical scale "
                "alone is unlikely to explain the external-domain collapse."
            ),
            "photometric_mean_std_matched_pipeline": (
                "If photometric mean/std matching improves performance, photometric mismatch "
                "plausibly contributed to failure. If it does not improve performance, global "
                "photometric mismatch alone is unlikely to explain the external-domain collapse."
            ),
        },
    }
    write_json(multi_summary, args.output_dir / "multi_pipeline_summary.json")

    print("\nMulti-pipeline comparison complete")
    print(f"Primary label: {args.primary_label}")
    print(f"Paired slides: {len(primary_ref)}")
    print(f"Primary accuracy: {primary_metrics_for_multi['accuracy']:.4f}")
    for label in comparison_maps.keys():
        comp_acc = pairwise_summaries[label]["comparison_metrics"]["accuracy"]
        delta_acc = pairwise_summaries[label]["metric_deltas_comparison_minus_primary"]["accuracy"]
        print(f"Comparison {label}: accuracy={comp_acc:.4f}, delta_vs_primary={delta_acc:.4f}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
