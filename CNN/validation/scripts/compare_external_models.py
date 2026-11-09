#!/usr/bin/env python3
"""
compare_external_models.py

Paired slide-level comparison of two frozen models evaluated on the SAME
external slides under the SAME preprocessing pipeline (archived Hybrid_43 vs
DenseNet-121). Reads slide_predictions.csv files only.

Statistics (delta = model_b minus model_a for every metric)
-----------------------------------------------------------
- Per-model point metrics: BA, accuracy, MCC (+degeneracy flag), weighted F1,
  AUROC, per-class recall with exact Clopper-Pearson CIs, Brier, ECE.
- Paired class-stratified slide bootstrap (B = 10,000, seed 42): ONE index
  vector per replicate applied to both models (identical to the marginal
  resampling of external_statistics.py), giving percentile CIs for
  dBA, dMCC, dAUROC, dBrier, daccuracy, drecall_albicans, drecall_glabrata.
- Exact McNemar test on slide correctness and fast DeLong test of correlated
  AUROCs. Both are reported UNADJUSTED here; the Holm-adjusted values across
  the declared family are in outputs/inferential_summary.json.

Inferential role: delta balanced accuracy on locked_pixel_pipeline is the
pre-declared primary contrast (interval estimate, no p-value).

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ext_common import ANALYSED_PIPELINES, PIPELINE_LOCKED, read_csv, write_csv, write_json
from ext_stats import (
    binary_metrics,
    clopper_pearson,
    delong_paired,
    fmt,
    mcnemar_exact,
    percentile_interval,
    stratified_bootstrap_indices,
)

DELTA_KEYS = ("balanced_accuracy", "matthews_corrcoef", "auroc", "brier", "accuracy",
              "recall_albicans", "recall_glabrata")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired slide-level comparison of two models.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-a", required=True, help="LABEL=path/to/slide_predictions.csv (reference)")
    p.add_argument("--model-b", required=True, help="LABEL=path/to/slide_predictions.csv (comparator)")
    p.add_argument("--pipeline-label", required=True, choices=list(ANALYSED_PIPELINES))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    return p.parse_args(argv)


def parse_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def load_slides(path: Path, expected_pipeline: str) -> Dict[str, Dict[str, Any]]:
    rows = read_csv(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = r["slide_id"]
        if sid in out:
            raise ValueError(f"Duplicate slide {sid} in {path}")
        mode = r.get("preprocessing_mode", "")
        if mode != expected_pipeline:
            raise ValueError(f"{path}: slide {sid} has preprocessing_mode={mode!r}, "
                             f"expected {expected_pipeline!r}")
        out[sid] = {
            "slide_id": sid,
            "species_name": r["species_name"],
            "hyphae_status": r.get("hyphae_status", ""),
            "true_label": int(float(r["true_label"])),
            "pred_label": int(float(r["pred_label"])),
            "p_glabrata": float(r["mean_prob_glabrata"]),
            "p_albicans": float(r["mean_prob_albicans"]),
            "confidence": float(r["confidence"]),
            "correct": int(float(r["correct"])),
        }
    return out


def model_arrays(M: Dict[str, Dict[str, Any]], ids: List[str], y: np.ndarray, label: str) -> Dict[str, np.ndarray]:
    arr = {
        "pred": np.array([M[s]["pred_label"] for s in ids]),
        "p": np.array([M[s]["p_glabrata"] for s in ids]),
        "c": np.array([M[s]["confidence"] for s in ids]),
        "correct": np.array([M[s]["correct"] for s in ids]),
    }
    if not np.array_equal((arr["pred"] == y).astype(int), arr["correct"]):
        raise ValueError(f"'correct' column inconsistent with labels for {label}")
    return arr


def paired_contrast(y: np.ndarray, arr_a: Dict[str, np.ndarray], arr_b: Dict[str, np.ndarray], *,
                    n_bootstrap: int, seed: int, ci: float = 0.95) -> Dict[str, Any]:
    """Pure statistical core: point metrics, CP recalls, paired bootstrap deltas (b - a), McNemar, DeLong."""
    y = np.asarray(y, dtype=int)
    point = {k: binary_metrics(y, arr["pred"], arr["p"], arr["c"]) for k, arr in (("a", arr_a), ("b", arr_b))}
    cp = {k: {"Candida albicans": clopper_pearson(point[k]["k_correct_albicans"], point[k]["n_albicans"], ci),
              "Candida glabrata": clopper_pearson(point[k]["k_correct_glabrata"], point[k]["n_glabrata"], ci)}
          for k in ("a", "b")}

    deltas: Dict[str, List[float]] = {k: [] for k in DELTA_KEYS}
    n_degenerate = {"a": 0, "b": 0}
    for idx in stratified_bootstrap_indices(y, n_bootstrap, seed):
        ma = binary_metrics(y[idx], arr_a["pred"][idx], arr_a["p"][idx], arr_a["c"][idx])
        mb = binary_metrics(y[idx], arr_b["pred"][idx], arr_b["p"][idx], arr_b["c"][idx])
        n_degenerate["a"] += int(ma["mcc_denominator_zero"])
        n_degenerate["b"] += int(mb["mcc_denominator_zero"])
        for k in DELTA_KEYS:
            va, vb = ma.get(k), mb.get(k)
            deltas[k].append(float("nan") if va is None or vb is None else float(vb) - float(va))

    summary: Dict[str, Any] = {}
    for k in DELTA_KEYS:
        lo, hi, nv = percentile_interval(deltas[k], ci)
        darr = np.asarray(deltas[k], dtype=float)
        darr = darr[np.isfinite(darr)]
        pa_v, pb_v = point["a"].get(k), point["b"].get(k)
        summary[k] = {
            "point_delta_b_minus_a": None if pa_v is None or pb_v is None else float(pb_v) - float(pa_v),
            "ci_lower": lo, "ci_upper": hi, "n_valid_bootstrap": nv,
            "fraction_delta_gt_0": float((darr > 0).mean()) if darr.size else None,
            "fraction_delta_lt_0": float((darr < 0).mean()) if darr.size else None,
            "fraction_delta_eq_0": float((darr == 0).mean()) if darr.size else None,
        }
    return {
        "point": point,
        "clopper_pearson": cp,
        "deltas": summary,
        "replicate_deltas": deltas,
        "mcc_degenerate_replicates": n_degenerate,
        "mcnemar": mcnemar_exact(arr_a["correct"], arr_b["correct"]),
        "delong": delong_paired(y, arr_a["p"], arr_b["p"]),
        "prediction_agreement_rate": float(np.mean(arr_a["pred"] == arr_b["pred"])),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    la, pa = parse_spec(args.model_a)
    lb, pb = parse_spec(args.model_b)
    if la == lb:
        raise ValueError("Model labels must differ.")
    A = load_slides(pa, args.pipeline_label)
    B = load_slides(pb, args.pipeline_label)
    if set(A) != set(B):
        raise ValueError(f"Slide sets differ: only_a={sorted(set(A) - set(B))}, "
                         f"only_b={sorted(set(B) - set(A))}")
    ids = sorted(A)
    for s in ids:
        if A[s]["true_label"] != B[s]["true_label"] or A[s]["species_name"] != B[s]["species_name"]:
            raise ValueError(f"Label/species mismatch for slide {s}")

    y = np.array([A[s]["true_label"] for s in ids])
    arr_a = model_arrays(A, ids, y, la)
    arr_b = model_arrays(B, ids, y, lb)
    res = paired_contrast(y, arr_a, arr_b, n_bootstrap=args.n_bootstrap, seed=args.seed, ci=args.ci)

    paired_rows = []
    for i, s in enumerate(ids):
        paired_rows.append({
            "slide_id": s, "species_name": A[s]["species_name"], "hyphae_status": A[s]["hyphae_status"],
            "true_label": int(y[i]),
            f"{la}_pred": int(arr_a["pred"][i]), f"{la}_p_glabrata": float(arr_a["p"][i]),
            f"{la}_correct": int(arr_a["correct"][i]),
            f"{lb}_pred": int(arr_b["pred"][i]), f"{lb}_p_glabrata": float(arr_b["p"][i]),
            f"{lb}_correct": int(arr_b["correct"][i]),
            "agreement": int(arr_a["pred"][i] == arr_b["pred"][i]),
        })
    cols = ["slide_id", "species_name", "hyphae_status", "true_label",
            f"{la}_pred", f"{la}_p_glabrata", f"{la}_correct",
            f"{lb}_pred", f"{lb}_p_glabrata", f"{lb}_correct", "agreement"]
    write_csv(paired_rows, args.output_dir / "paired_slide_table.csv", cols)
    boot_rows = [{"bootstrap_index": i, **{f"delta_{k}": res["replicate_deltas"][k][i] for k in DELTA_KEYS}}
                 for i in range(int(args.n_bootstrap))]
    write_csv(boot_rows, args.output_dir / "bootstrap_deltas.csv",
              ["bootstrap_index"] + [f"delta_{k}" for k in DELTA_KEYS])

    primary = args.pipeline_label == PIPELINE_LOCKED
    summary = {
        "schema_version": "2.0",
        "script": "compare_external_models.py",
        "pipeline_label": args.pipeline_label,
        "model_a": {"label": la, "slide_predictions": str(pa)},
        "model_b": {"label": lb, "slide_predictions": str(pb)},
        "delta_convention": "model_b minus model_a",
        "n_slides": len(ids),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "resampling": "class-stratified slide bootstrap; identical indices for both models (paired)",
        "inferential_role": {
            "delta_balanced_accuracy": "primary (interval estimate)" if primary else "secondary, descriptive",
            "other_deltas": "descriptive",
            "mcnemar_exact": "secondary (Holm-adjusted in inferential_summary.json)",
            "delong_paired_auroc": "secondary cross-check (Holm-adjusted in inferential_summary.json)",
        },
        "p_values_unadjusted": True,
        "fraction_delta_gt_0_is_not_a_p_value": True,
        "point_metrics": {la: res["point"]["a"], lb: res["point"]["b"]},
        "clopper_pearson_recall": {la: res["clopper_pearson"]["a"], lb: res["clopper_pearson"]["b"]},
        "paired_bootstrap_deltas": res["deltas"],
        "mcc_degenerate_replicates": {la: res["mcc_degenerate_replicates"]["a"],
                                      lb: res["mcc_degenerate_replicates"]["b"]},
        "mcnemar_exact": res["mcnemar"],
        "delong_paired_auroc": res["delong"],
        "prediction_agreement_rate": res["prediction_agreement_rate"],
        "interpretation_guardrails": [
            "n = 24 slides; intervals are wide and saturate at metric bounds.",
            "Hybrid_43 was selected with held-out information; DenseNet-121 was not.",
            "Archived Hybrid_43 predictions were produced in the published software environment; "
            "DenseNet-121 was scored in the environment recorded in provenance/env_snapshot.json. "
            "Preprocessing parity rests on the pinned legacy scripts.",
            "Species is confounded with strain and acquisition date in the external cohort.",
        ],
        "elapsed_seconds": time.time() - t0,
    }
    write_json(summary, args.output_dir / "model_comparison_summary.json")

    d = res["deltas"]
    mc = res["mcnemar"]
    tex = [f"% Paired contrast {lb} minus {la}, {args.pipeline_label}, B={args.n_bootstrap}, seed={args.seed}"]
    for k, name in (("balanced_accuracy", "Balanced accuracy"), ("matthews_corrcoef", "MCC"),
                    ("auroc", "AUROC"), ("brier", "Brier score")):
        tex.append(f"$\\Delta$ {name} & {fmt(d[k]['point_delta_b_minus_a'])} & "
                   f"[{fmt(d[k]['ci_lower'])}, {fmt(d[k]['ci_upper'])}] \\\\")
    tex.append(f"Discordant slides & \\multicolumn{{2}}{{c}}{{$b={mc['table']['a_correct_b_wrong']}$, "
               f"$c={mc['table']['a_wrong_b_correct']}$}} \\\\")
    (args.output_dir / "latex_comparison_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"MODEL_COMPARISON_DONE {args.pipeline_label}: {lb} - {la}")
    print(f"  BA: {fmt(res['point']['a']['balanced_accuracy'])} vs {fmt(res['point']['b']['balanced_accuracy'])} | "
          f"dBA={fmt(d['balanced_accuracy']['point_delta_b_minus_a'])} "
          f"[{fmt(d['balanced_accuracy']['ci_lower'])}, {fmt(d['balanced_accuracy']['ci_upper'])}]")
    print(f"  McNemar b={mc['table']['a_correct_b_wrong']} c={mc['table']['a_wrong_b_correct']} "
          f"(unadjusted p={mc['p_value_two_sided_exact']:.4f})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
