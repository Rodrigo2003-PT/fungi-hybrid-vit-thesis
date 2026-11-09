#!/usr/bin/env python3
"""
compare_external_pipelines.py

Paired slide-level comparison of the locked primary external-validation
pipeline against one or more post hoc sensitivity-analysis pipelines.

Scientific role
---------------
This script does not train, fine-tune, tune thresholds or select a model. It
compares already-computed frozen-model external evaluations on the same
slides. The comparison is paired at slide level because the slide is the
independent biological/acquisition unit.

Primary interpretation
----------------------
- ``locked_pixel_pipeline`` remains the primary external-validation result.
- Comparison pipelines (currently ``photometric_mean_std_matched_pipeline``)
  are post hoc diagnostic sensitivity analyses. They must not replace the
  primary estimate and must not be used for model selection or threshold
  tuning.
- Pairwise deltas indicate whether a hypothesised mechanism (for example a
  global photometric mismatch) plausibly contributed to external failure.
  Acquisition date is confounded with species in this cohort, so any
  mechanism statement remains conditional on that confound.

Statistics
----------
- exact McNemar test on discordant slide pairs;
- paired sign-flip test on the per-slide change in true-class probability:
  exact vectorised enumeration for up to 24 non-zero differences, seeded Monte
  Carlo above that (``validation_stats.sign_flip_pvalue``);
- class-stratified paired bootstrap of the accuracy difference, using the same
  resampling scheme as external_statistics.py.

Outputs
-------
output_dir/
  multi_pipeline_slide_comparison.csv
  multi_pipeline_summary.json                 (schema 3.0)
  pairwise_comparison__<comparison_label>.csv
  pairwise_summary__<comparison_label>.json

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

from validation_stats import (
    exact_mcnemar_pvalue,
    percentile_interval,
    sign_flip_pvalue,
    stratified_bootstrap_indices,
)

COMPARISON_SCHEMA_VERSION = "3.0"
PRIMARY_DEFAULT_LABEL = "locked_pixel_pipeline"

MECHANISM_DESCRIPTIONS = {
    "photometric_mean_std_matched_pipeline": "global photometric (image mean/std) mismatch",
}

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

BASE_MULTI_COLUMNS = ["slide_id", "species_name", "hyphae_status", "true_label"]

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired slide-level comparison between the primary locked external-validation "
            "pipeline and one or more post hoc sensitivity-analysis pipelines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--primary-slide-predictions", required=True, type=Path)
    parser.add_argument("--primary-label", default=PRIMARY_DEFAULT_LABEL)
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help=(
            "Comparison pipeline as LABEL=PATH_TO_SLIDE_PREDICTIONS_CSV. May be repeated. Example: "
            "--comparison photometric_mean_std_matched_pipeline=outputs/.../slide_predictions.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
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
    if len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, prob_pos))
    except ValueError:
        return None


def classify_change(primary_correct: int, comparison_correct: int) -> str:
    if primary_correct == 0 and comparison_correct == 1:
        return "improved"
    if primary_correct == 1 and comparison_correct == 0:
        return "worsened"
    return "both_correct" if primary_correct == 1 else "both_wrong"


def bootstrap_delta_accuracy(
    true_labels: Sequence[int],
    primary_correct: Sequence[int],
    comparison_correct: Sequence[int],
    *,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> Dict[str, Any]:
    """Class-stratified paired bootstrap of accuracy(comparison) - accuracy(primary)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(true_labels, dtype=int)
    p = np.asarray(primary_correct, dtype=float)
    c = np.asarray(comparison_correct, dtype=float)
    deltas = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = stratified_bootstrap_indices(y, rng)
        deltas[i] = float(np.mean(c[idx]) - np.mean(p[idx]))
    interval = percentile_interval(deltas, ci)
    return {
        "method": "percentile_bootstrap_class_stratified_paired",
        "point_estimate": float(np.mean(c) - np.mean(p)),
        "delta_accuracy_95ci_percentile": [interval["ci_lower"], interval["ci_upper"]],
        **interval,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def build_slide_map(name: str, rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    missing = REQUIRED_SLIDE_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"{name} slide_predictions missing columns: {sorted(missing)}")
    by_slide: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row["slide_id"] in by_slide:
            raise ValueError(f"{name} contains duplicate slide_id: {row['slide_id']}")
        by_slide[row["slide_id"]] = row
    return by_slide


def parse_comparisons(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    comparisons: List[Tuple[str, Path]] = []
    seen = set()
    for spec in args.comparison:
        if "=" not in spec:
            raise ValueError(f"Invalid --comparison value: {spec!r}. Expected LABEL=PATH_TO_SLIDE_PREDICTIONS_CSV")
        label, path_str = (part.strip() for part in spec.split("=", 1))
        if not label:
            raise ValueError(f"Invalid --comparison label in spec: {spec!r}")
        if label == args.primary_label:
            raise ValueError("Comparison label must differ from --primary-label")
        if label in seen:
            raise ValueError(f"Duplicate comparison label: {label}")
        seen.add(label)
        comparisons.append((label, Path(path_str)))
    if not comparisons:
        raise ValueError("At least one --comparison LABEL=PATH argument is required.")
    return comparisons


def build_primary_reference(primary_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    ref: Dict[str, Dict[str, Any]] = {}
    for slide_id, row in build_slide_map("primary", primary_rows).items():
        true_label = to_int(row["true_label"])
        pa, pg = to_float(row["mean_prob_albicans"]), to_float(row["mean_prob_glabrata"])
        ref[slide_id] = {
            "slide_id": slide_id,
            "species_name": row["species_name"],
            "hyphae_status": row.get("hyphae_status", ""),
            "true_label": true_label,
            "primary_pred_label": to_int(row["pred_label"]),
            "primary_correct": to_int(row["correct"]),
            "primary_prob_albicans": pa,
            "primary_prob_glabrata": pg,
            "primary_confidence": to_float(row["confidence"]),
            "primary_true_class_probability": pa if true_label == 0 else pg,
        }
    return ref


def ensure_same_slide_set(
    primary_ref: Mapping[str, Dict[str, Any]],
    comparison_rows: List[Dict[str, str]],
    comparison_name: str,
) -> Dict[str, Dict[str, str]]:
    c_map = build_slide_map(comparison_name, comparison_rows)
    p_slides, c_slides = set(primary_ref.keys()), set(c_map.keys())
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
        if c_row.get("hyphae_status", "") != p["hyphae_status"]:
            raise ValueError(f"Hyphae-status mismatch for slide {slide_id} in {comparison_name}")
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
    c_map: Mapping[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    paired: List[Dict[str, Any]] = []
    for slide_id in sorted(primary_ref.keys()):
        p, c = primary_ref[slide_id], c_map[slide_id]
        c_pred, c_correct = to_int(c["pred_label"]), to_int(c["correct"])
        c_pa, c_pg, c_conf = to_float(c["mean_prob_albicans"]), to_float(c["mean_prob_glabrata"]), to_float(c["confidence"])
        c_true = c_pa if p["true_label"] == 0 else c_pg
        paired.append({
            "slide_id": slide_id,
            "species_name": p["species_name"],
            "hyphae_status": p["hyphae_status"],
            "true_label": p["true_label"],
            "primary_pred_label": p["primary_pred_label"],
            "comparison_pred_label": c_pred,
            "primary_correct": p["primary_correct"],
            "comparison_correct": c_correct,
            "correctness_change": classify_change(p["primary_correct"], c_correct),
            "primary_prob_albicans": p["primary_prob_albicans"],
            "primary_prob_glabrata": p["primary_prob_glabrata"],
            "comparison_prob_albicans": c_pa,
            "comparison_prob_glabrata": c_pg,
            "primary_true_class_probability": p["primary_true_class_probability"],
            "comparison_true_class_probability": c_true,
            "delta_true_class_probability": c_true - p["primary_true_class_probability"],
            "primary_confidence": p["primary_confidence"],
            "comparison_confidence": c_conf,
            "delta_confidence": c_conf - p["primary_confidence"],
        })
    return paired


def subgroup_summary_pairwise(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        key = f"Candida albicans/{r.get('hyphae_status', '')}" if r["species_name"] == "Candida albicans" else "Candida glabrata"
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
    pairwise_by_label: Mapping[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    columns = list(BASE_MULTI_COLUMNS) + [
        "primary_pred_label", "primary_correct", "primary_prob_albicans", "primary_prob_glabrata",
        "primary_true_class_probability", "primary_confidence",
    ]
    for label in pairwise_by_label:
        slug = slugify(label)
        columns += [
            f"{slug}_pred_label", f"{slug}_correct", f"{slug}_prob_albicans", f"{slug}_prob_glabrata",
            f"{slug}_true_class_probability", f"{slug}_confidence",
            f"delta_true_class_probability__{slug}_minus_primary",
            f"delta_confidence__{slug}_minus_primary",
            f"correctness_change__{slug}_vs_primary",
        ]
    by_label_slide = {label: {r["slide_id"]: r for r in rows} for label, rows in pairwise_by_label.items()}
    rows_out: List[Dict[str, Any]] = []
    for slide_id in sorted(primary_ref.keys()):
        p = primary_ref[slide_id]
        row: Dict[str, Any] = {k: p[k] for k in (
            "slide_id", "species_name", "hyphae_status", "true_label", "primary_pred_label", "primary_correct",
            "primary_prob_albicans", "primary_prob_glabrata", "primary_true_class_probability", "primary_confidence",
        )}
        for label in pairwise_by_label:
            slug, r = slugify(label), by_label_slide[label][slide_id]
            row.update({
                f"{slug}_pred_label": r["comparison_pred_label"],
                f"{slug}_correct": r["comparison_correct"],
                f"{slug}_prob_albicans": r["comparison_prob_albicans"],
                f"{slug}_prob_glabrata": r["comparison_prob_glabrata"],
                f"{slug}_true_class_probability": r["comparison_true_class_probability"],
                f"{slug}_confidence": r["comparison_confidence"],
                f"delta_true_class_probability__{slug}_minus_primary": r["delta_true_class_probability"],
                f"delta_confidence__{slug}_minus_primary": r["delta_confidence"],
                f"correctness_change__{slug}_vs_primary": r["correctness_change"],
            })
        rows_out.append(row)
    return rows_out, columns


def thesis_language(label: str) -> Dict[str, str]:
    mechanism = MECHANISM_DESCRIPTIONS.get(label, "the hypothesised mechanism motivating this sensitivity analysis")
    return {
        "if_performance_improves": (
            f"The {label} sensitivity analysis improved slide-level recognition relative to the locked primary "
            f"pipeline, suggesting that {mechanism} plausibly contributed to the external-domain failure. Because "
            "the analysis is post hoc, and acquisition date is confounded with species in this cohort, it is "
            "interpreted diagnostically rather than as a corrected primary external-validation score."
        ),
        "if_performance_does_not_improve": (
            f"The {label} sensitivity analysis did not recover performance relative to the locked primary "
            f"pipeline, suggesting that {mechanism} alone is insufficient to explain the external-domain failure."
        ),
    }


def summarize_pairwise(
    paired_rows: List[Dict[str, Any]],
    *,
    primary_label: str,
    comparison_label: str,
    primary_path: Path,
    comparison_path: Path,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> Dict[str, Any]:
    primary_metrics = compute_metrics_from_rows(paired_rows, "primary")
    comparison_metrics = compute_metrics_from_rows(paired_rows, "comparison")
    deltas = {
        key: None if primary_metrics.get(key) is None or comparison_metrics.get(key) is None
        else float(comparison_metrics[key] - primary_metrics[key])
        for key in primary_metrics if key != "n_slides"
    }
    delta_true = [to_float(r["delta_true_class_probability"]) for r in paired_rows]
    changes = Counter(str(r["correctness_change"]) for r in paired_rows)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "scientific_role": "post_hoc_paired_slide_level_sensitivity_comparison",
        "primary_label": primary_label,
        "comparison_label": comparison_label,
        "primary_slide_predictions": str(primary_path),
        "comparison_slide_predictions": str(comparison_path),
        "interpretation_guardrail": (
            f"{primary_label} remains the primary external-validation result. {comparison_label} is a post hoc "
            "diagnostic sensitivity analysis and must not be used for model selection, threshold tuning, or as "
            "the headline external-validation estimate."
        ),
        "n_paired_slides": len(paired_rows),
        "primary_metrics": primary_metrics,
        "comparison_metrics": comparison_metrics,
        "metric_deltas_comparison_minus_primary": deltas,
        "correctness_change_counts": dict(sorted(changes.items())),
        "mcnemar_exact_pvalue_descriptive": exact_mcnemar_pvalue(changes.get("improved", 0), changes.get("worsened", 0)),
        "mean_delta_true_class_probability": float(np.mean(delta_true)),
        "median_delta_true_class_probability": float(np.median(delta_true)),
        "sign_flip_test_mean_delta_true_class_probability_descriptive": sign_flip_pvalue(delta_true, seed=seed),
        "bootstrap_delta_accuracy": bootstrap_delta_accuracy(
            [r["true_label"] for r in paired_rows],
            [to_int(r["primary_correct"]) for r in paired_rows],
            [to_int(r["comparison_correct"]) for r in paired_rows],
            n_bootstrap=n_bootstrap, seed=seed, ci=ci,
        ),
        "subgroup_summary": subgroup_summary_pairwise(paired_rows),
        "recommended_thesis_language": thesis_language(comparison_label),
    }


def main() -> None:
    args = parse_args()
    comparisons = parse_comparisons(args)
    primary_ref = build_primary_reference(read_csv(args.primary_slide_predictions))

    pairwise_by_label: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for label, path in comparisons:
        c_map = ensure_same_slide_set(primary_ref, read_csv(path), label)
        pairwise_by_label[label] = build_pairwise_rows(primary_ref, c_map)
        summaries[label] = summarize_pairwise(
            pairwise_by_label[label],
            primary_label=args.primary_label,
            comparison_label=label,
            primary_path=args.primary_slide_predictions,
            comparison_path=path,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            ci=args.ci,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    multi_rows, multi_columns = build_multi_table(primary_ref, pairwise_by_label)
    write_csv(multi_rows, args.output_dir / "multi_pipeline_slide_comparison.csv", multi_columns)

    manifest_entries: List[Dict[str, Any]] = []
    for label, path in comparisons:
        slug = slugify(label)
        pair_csv = args.output_dir / f"pairwise_comparison__{slug}.csv"
        pair_json = args.output_dir / f"pairwise_summary__{slug}.json"
        write_csv(pairwise_by_label[label], pair_csv, PAIRWISE_BASE_COLUMNS)
        write_json(summaries[label], pair_json)
        manifest_entries.append({
            "comparison_label": label,
            "comparison_label_slug": slug,
            "comparison_slide_predictions": str(path),
            "pairwise_comparison_csv": str(pair_csv),
            "pairwise_summary_json": str(pair_json),
        })

    primary_metrics = compute_metrics_from_rows(list(primary_ref.values()), "primary")
    multi_summary = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "scientific_role": "paired_slide_level_multi_pipeline_sensitivity_comparison",
        "primary_label": args.primary_label,
        "primary_slide_predictions": str(args.primary_slide_predictions),
        "comparison_pipelines": [{"label": label, "slide_predictions": str(path)} for label, path in comparisons],
        "interpretation_guardrail": (
            f"{args.primary_label} remains the primary external-validation result. All other pipelines "
            "summarised here are post hoc sensitivity analyses and must not replace the primary estimate or be "
            "used for model or threshold selection."
        ),
        "n_paired_slides": len(primary_ref),
        "outputs": {
            "multi_pipeline_slide_comparison_csv": str(args.output_dir / "multi_pipeline_slide_comparison.csv"),
            "pairwise_outputs": manifest_entries,
        },
        "primary_metrics": primary_metrics,
        "comparison_metrics": {label: s["comparison_metrics"] for label, s in summaries.items()},
        "pairwise_deltas_vs_primary": {
            label: {
                key: s[key] for key in (
                    "metric_deltas_comparison_minus_primary",
                    "correctness_change_counts",
                    "bootstrap_delta_accuracy",
                    "mcnemar_exact_pvalue_descriptive",
                    "mean_delta_true_class_probability",
                    "median_delta_true_class_probability",
                    "sign_flip_test_mean_delta_true_class_probability_descriptive",
                )
            }
            for label, s in summaries.items()
        },
        "recommended_thesis_language": {
            "general": (
                "Sensitivity analyses are reported as diagnostic follow-up analyses performed after observing "
                "the primary external-validation failure. They clarify plausible mechanisms of failure but do not "
                "redefine the primary external-validation performance."
            ),
            **{label: thesis_language(label) for label, _ in comparisons},
        },
    }
    write_json(multi_summary, args.output_dir / "multi_pipeline_summary.json")

    print("\nMulti-pipeline comparison complete")
    print(f"Primary label: {args.primary_label}")
    print(f"Paired slides: {len(primary_ref)}")
    print(f"Primary accuracy: {primary_metrics['accuracy']:.4f}")
    for label, s in summaries.items():
        print(
            f"Comparison {label}: accuracy={s['comparison_metrics']['accuracy']:.4f}, "
            f"delta_vs_primary={s['metric_deltas_comparison_minus_primary']['accuracy']:.4f}"
        )
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
