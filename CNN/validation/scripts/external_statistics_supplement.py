#!/usr/bin/env python3
"""
external_statistics_supplement.py

Outputs (in --output-dir)
-------------------------
supplement_summary.json
    - slide-level point metrics incl. Brier and ECE (10 bins),
    - class-stratified slide bootstrap CIs (B, seed identical to legacy). The
      ``legacy_crosscheck`` block compares them with slide_metrics_with_ci.json
      from the legacy script; any difference is an error,
    - exact Clopper-Pearson intervals for both per-class recalls,
    - image-level descriptive metrics with slide-cluster bootstrap CIs
      (slides resampled within class with the SAME draws as the slide-level
      bootstrap; all images of a drawn slide kept),
    - within-acquisition-date descriptive contrast,
    - slide-level error list.
latex_row.tex
    One row for the external results table (BA, MCC, AUROC with bootstrap
    CIs; recalls with Clopper-Pearson CIs).

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ext_common import ANALYSED_PIPELINES, read_csv, read_json, slide_date, write_json
from ext_stats import (
    binary_metrics,
    clopper_pearson,
    fmt_ci_latex,
    marginal_bootstrap,
    percentile_interval,
    stratified_bootstrap_indices,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplementary external statistics (CP recalls, cluster bootstrap).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image-predictions", required=True, type=Path)
    p.add_argument("--slide-predictions", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model-label", required=True,
                   help="Must equal the CSV's model_family column when that column exists.")
    p.add_argument("--pipeline-label", required=True, choices=list(ANALYSED_PIPELINES))
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--legacy-ci-json", type=Path, default=None,
                   help="slide_metrics_with_ci.json from external_statistics.py for cross-check")
    p.add_argument("--require-legacy-crosscheck", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--expected-slides", type=int, default=24)
    return p.parse_args(argv)


def validate_slide_rows(rows: Sequence[Dict[str, str]], pipeline: str, model_label: str) -> None:
    """Every row must come from ``pipeline``; model_family (if present) must equal ``model_label``."""
    modes = sorted({r.get("preprocessing_mode", "") for r in rows})
    if modes != [pipeline]:
        raise RuntimeError(f"preprocessing_mode {modes} does not match --pipeline-label {pipeline!r}")
    if "model_family" in rows[0]:
        fams = sorted({r.get("model_family", "") for r in rows})
        if fams != [model_label]:
            raise RuntimeError(f"model_family {fams} does not match --model-label {model_label!r}")


def slide_arrays(rows: List[Dict[str, str]]):
    rows = sorted(rows, key=lambda r: r["slide_id"])
    y = np.array([int(float(r["true_label"])) for r in rows])
    yp = np.array([int(float(r["pred_label"])) for r in rows])
    p = np.array([float(r["mean_prob_glabrata"]) for r in rows])
    c = np.array([float(r["confidence"]) for r in rows])
    ids = [r["slide_id"] for r in rows]
    return rows, ids, y, yp, p, c


def cluster_bootstrap_draws(image_rows: List[Dict[str, str]], slide_ids: List[str],
                            slide_y: np.ndarray, *, n_bootstrap: int,
                            seed: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (slide indices drawn, image indices of those slides) per replicate.
    Slide draws are exactly ext_stats.stratified_bootstrap_indices(slide_y, B, seed)."""
    by_slide: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(image_rows):
        by_slide[r["slide_id"]].append(i)
    missing = [s for s in slide_ids if s not in by_slide]
    if missing:
        raise RuntimeError(f"Slides without images: {missing[:5]}")
    for picked in stratified_bootstrap_indices(slide_y, n_bootstrap, seed):
        img_idx = np.concatenate([np.asarray(by_slide[slide_ids[j]], dtype=int) for j in picked])
        yield picked, img_idx


def image_cluster_bootstrap(image_rows: List[Dict[str, str]], slide_ids: List[str],
                            slide_y: np.ndarray, *, n_bootstrap: int, seed: int,
                            ci: float) -> Dict[str, Any]:
    y = np.array([int(float(r["true_label"])) for r in image_rows])
    yp = np.array([int(float(r["pred_label"])) for r in image_rows])
    p = np.array([float(r["prob_glabrata"]) for r in image_rows])
    point = {"balanced_accuracy": float(balanced_accuracy_score(y, yp)),
             "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    ba, au = [], []
    for _, img_idx in cluster_bootstrap_draws(image_rows, slide_ids, slide_y,
                                              n_bootstrap=n_bootstrap, seed=seed):
        yb, ypb, pb = y[img_idx], yp[img_idx], p[img_idx]
        ba.append(float(balanced_accuracy_score(yb, ypb)))
        au.append(float(roc_auc_score(yb, pb)) if len(np.unique(yb)) == 2 else float("nan"))
    lo_b, hi_b, n_b = percentile_interval(ba, ci)
    lo_a, hi_a, n_a = percentile_interval(au, ci)
    return {
        "n_images": int(len(y)),
        "balanced_accuracy": {"point_estimate": point["balanced_accuracy"], "ci_lower": lo_b,
                              "ci_upper": hi_b, "n_valid_bootstrap": n_b},
        "auroc": {"point_estimate": point["auroc"], "ci_lower": lo_a, "ci_upper": hi_a,
                  "n_valid_bootstrap": n_a},
        "resampling": "slides within class (same draws as the slide bootstrap), all images of each drawn slide",
        "interpretation": "Descriptive only; images within a slide are correlated.",
    }


def legacy_crosscheck(legacy_json: Optional[Path], boot: Dict[str, Dict[str, Any]],
                      required: bool) -> Optional[Dict[str, Any]]:
    """Exact equality of the supplement and legacy bootstrap intervals; any difference raises."""
    if legacy_json is None or not legacy_json.exists():
        if required:
            raise RuntimeError(f"Legacy CI JSON required for the cross-check but not found: {legacy_json}")
        return None
    leg = read_json(legacy_json)
    out: Dict[str, Any] = {}
    for key in ("balanced_accuracy", "matthews_corrcoef", "auroc"):
        if key not in leg:
            raise RuntimeError(f"Legacy CI JSON lacks {key!r}")
        same = (leg[key].get("ci_lower") == boot[key]["ci_lower"]
                and leg[key].get("ci_upper") == boot[key]["ci_upper"])
        out[key] = {"legacy": [leg[key].get("ci_lower"), leg[key].get("ci_upper")],
                    "supplement": [boot[key]["ci_lower"], boot[key]["ci_upper"]],
                    "identical": bool(same)}
    bad = [k for k, v in out.items() if not v["identical"]]
    if bad:
        raise RuntimeError(f"Bootstrap intervals differ from the legacy script for {bad}: "
                           f"{ {k: out[k] for k in bad} }")
    return out


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slide_rows_raw = read_csv(args.slide_predictions)
    image_rows = read_csv(args.image_predictions)
    validate_slide_rows(slide_rows_raw, args.pipeline_label, args.model_label)
    rows, ids, y, yp, p, c = slide_arrays(slide_rows_raw)
    if len(ids) != args.expected_slides:
        raise RuntimeError(f"Expected {args.expected_slides} slides, got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate slide_id rows.")

    point = binary_metrics(y, yp, p, c)
    boot = marginal_bootstrap(y, yp, p, c, n_bootstrap=args.n_bootstrap, seed=args.seed, ci=args.ci)
    cp_alb = clopper_pearson(point["k_correct_albicans"], point["n_albicans"], args.ci)
    cp_gla = clopper_pearson(point["k_correct_glabrata"], point["n_glabrata"], args.ci)
    legacy_cross = legacy_crosscheck(args.legacy_ci_json, boot, args.require_legacy_crosscheck)

    by_date: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = slide_date(r["slide_id"])
        g = by_date.setdefault(d, {"n_slides": 0, "n_albicans": 0, "n_glabrata": 0,
                                   "n_correct": 0, "slides": []})
        g["n_slides"] += 1
        g["n_albicans"] += int(int(float(r["true_label"])) == 0)
        g["n_glabrata"] += int(int(float(r["true_label"])) == 1)
        g["n_correct"] += int(float(r["correct"]))
        g["slides"].append({"slide_id": r["slide_id"], "true_label": int(float(r["true_label"])),
                            "pred_label": int(float(r["pred_label"])),
                            "mean_prob_glabrata": float(r["mean_prob_glabrata"])})

    errors = [{"slide_id": r["slide_id"], "species_name": r["species_name"],
               "hyphae_status": r["hyphae_status"], "pred_label": int(float(r["pred_label"])),
               "confidence": float(r["confidence"])}
              for r in rows if int(float(r["correct"])) == 0]

    img = image_cluster_bootstrap(image_rows, ids, y, n_bootstrap=args.n_bootstrap,
                                  seed=args.seed, ci=args.ci)

    summary = {
        "schema_version": "1.1",
        "script": "external_statistics_supplement.py",
        "model_label": args.model_label,
        "pipeline_label": args.pipeline_label,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "ci_level": float(args.ci),
        "slide_point_metrics": point,
        "slide_bootstrap_ci": boot,
        "clopper_pearson_recall": {"Candida albicans": cp_alb, "Candida glabrata": cp_gla},
        "legacy_crosscheck": legacy_cross,
        "image_level_descriptive": img,
        "by_acquisition_date": by_date,
        "misclassified_slides": errors,
        "elapsed_seconds": time.time() - t0,
    }
    write_json(summary, args.output_dir / "supplement_summary.json")

    b = boot
    latex = (
        f"% {args.model_label} | {args.pipeline_label} | slide-level, B={args.n_bootstrap}, seed={args.seed}\n"
        f"{args.model_label} & "
        f"{fmt_ci_latex(b['balanced_accuracy']['point_estimate'], b['balanced_accuracy']['ci_lower'], b['balanced_accuracy']['ci_upper'])} & "
        f"{fmt_ci_latex(b['matthews_corrcoef']['point_estimate'], b['matthews_corrcoef']['ci_lower'], b['matthews_corrcoef']['ci_upper'])} & "
        f"{fmt_ci_latex(b['auroc']['point_estimate'], b['auroc']['ci_lower'], b['auroc']['ci_upper'])} & "
        f"{fmt_ci_latex(cp_alb['estimate'], cp_alb['ci_lower'], cp_alb['ci_upper'])} & "
        f"{fmt_ci_latex(cp_gla['estimate'], cp_gla['ci_lower'], cp_gla['ci_upper'])} \\\\\n"
    )
    (args.output_dir / "latex_row.tex").write_text(latex, encoding="utf-8")

    print(f"SUPPLEMENT_DONE {args.model_label} {args.pipeline_label}: "
          f"BA={point['balanced_accuracy']:.4f} [{b['balanced_accuracy']['ci_lower']}, {b['balanced_accuracy']['ci_upper']}] "
          f"recall_alb={cp_alb['k']}/{cp_alb['n']} [{cp_alb['ci_lower']:.4f}, {cp_alb['ci_upper']:.4f}] "
          f"recall_gla={cp_gla['k']}/{cp_gla['n']} [{cp_gla['ci_lower']:.4f}, {cp_gla['ci_upper']:.4f}] "
          f"Brier={point['brier']:.4f} ECE={point['ece']:.4f} "
          f"legacy_crosscheck={'identical' if legacy_cross else 'not run'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
