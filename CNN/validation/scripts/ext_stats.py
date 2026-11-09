#!/usr/bin/env python3
"""
ext_stats.py

Slide-level statistics for the DenseNet-121 external validation.

All resampling is at the SLIDE level.
The class-stratified bootstrap reproduces, draw for draw, the resampling of
``external_statistics.py::slide_level_bootstrap_ci`` (same generator, same class
order, same ``rng.choice`` calls). Consequently:

  * marginal per-model intervals computed here equal the legacy intervals;
  * two models evaluated on the same 24 slides (same sorted slide order, same
    labels, same seed) receive IDENTICAL resample indices, so replicate-wise
    differences form a valid paired bootstrap of the model contrast.

Contents
--------
- binary_metrics: BA, accuracy, MCC (+degeneracy flag), weighted F1,
  per-class recall, AUROC, Brier score, ECE.
- expected_calibration_error: 10 equal-width bins on max-class confidence,
  last bin closed (identical to the legacy implementation).
- stratified_bootstrap_indices: legacy-identical resampling generator.
- clopper_pearson: exact binomial interval.
- mcnemar_exact: two-sided exact McNemar test on discordant pairs.
- delong_paired: fast DeLong for two correlated AUROCs.
- holm_adjust: Holm step-down adjustment within a declared family.
- validate_reference_slide_csv: fail-closed gate for the archived Hybrid_43
  slide predictions (pipeline, slide set, labels, SHA-256 pin, Table 10).

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS: Tuple[str, ...] = (
    "balanced_accuracy",
    "accuracy",
    "matthews_corrcoef",
    "weighted_f1",
    "auroc",
    "recall_albicans",
    "recall_glabrata",
    "brier",
    "ece",
)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def expected_calibration_error(
    correct: Sequence[float],
    confidence: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    ECE = sum_b (n_b / n) * |acc_b - conf_b| over equal-width bins of the
    maximum class probability. Bins are [lo, hi) except the last, [0.9, 1.0].
    Identical to external_statistics.py::expected_calibration_error.
    """
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    finite = np.isfinite(correct) & np.isfinite(confidence)
    correct = correct[finite]
    confidence = confidence[finite]
    if confidence.size == 0:
        return {"ece": None, "bins": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: List[Dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin_index": i, "bin_lower": float(lo), "bin_upper": float(hi),
                         "n": 0, "mean_confidence": None, "accuracy": None, "abs_gap": None})
            continue
        conf_b = float(confidence[mask].mean())
        acc_b = float(correct[mask].mean())
        gap = abs(acc_b - conf_b)
        ece += (n / confidence.size) * gap
        bins.append({"bin_index": i, "bin_lower": float(lo), "bin_upper": float(hi),
                     "n": n, "mean_confidence": conf_b, "accuracy": acc_b, "abs_gap": gap})
    return {"ece": float(ece), "bins": bins}


# -----------------------------------------------------------------------------
# Point metrics
# -----------------------------------------------------------------------------

def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    p_glabrata: Sequence[float],
    confidence: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Metrics with C. glabrata (label 1) as the positive class.

    ``confidence`` defaults to max(p, 1 - p); pass the CSV column to use the
    exact stored values.
    """
    y = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    p = np.asarray(p_glabrata, dtype=float)
    if y.size == 0:
        raise ValueError("binary_metrics received zero samples.")
    conf = (np.maximum(p, 1.0 - p) if confidence is None
            else np.asarray(confidence, dtype=float))
    correct = (y == yp).astype(float)

    recalls = recall_score(y, yp, labels=[0, 1], average=None, zero_division=0)
    both_classes = len(np.unique(y)) == 2
    mcc_degenerate = len(np.unique(yp)) < 2 or not both_classes

    out: Dict[str, Any] = {
        "n": int(y.size),
        "n_albicans": int((y == 0).sum()),
        "n_glabrata": int((y == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "accuracy": float(accuracy_score(y, yp)),
        "matthews_corrcoef": float(matthews_corrcoef(y, yp)),
        "mcc_denominator_zero": bool(mcc_degenerate),
        "weighted_f1": float(f1_score(y, yp, average="weighted", zero_division=0)),
        "recall_albicans": float(recalls[0]),
        "recall_glabrata": float(recalls[1]),
        "k_correct_albicans": int(((y == 0) & (yp == 0)).sum()),
        "k_correct_glabrata": int(((y == 1) & (yp == 1)).sum()),
        "auroc": float(roc_auc_score(y, p)) if both_classes else None,
        "brier": float(brier_score_loss(y, p)) if both_classes else float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(correct, conf)["ece"],
        "mean_confidence_correct": float(conf[correct == 1].mean()) if (correct == 1).any() else None,
        "mean_confidence_incorrect": float(conf[correct == 0].mean()) if (correct == 0).any() else None,
        "n_correct": int(correct.sum()),
        "n_incorrect": int((1 - correct).sum()),
    }
    return out


# -----------------------------------------------------------------------------
# Resampling
# -----------------------------------------------------------------------------

def stratified_bootstrap_indices(
    y_true: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> Iterator[np.ndarray]:
    """
    Class-stratified slide bootstrap, draw-for-draw identical to
    external_statistics.py::slide_level_bootstrap_ci(stratified=True):

        rng = default_rng(seed)
        for b: concat([rng.choice(idx_c, len(idx_c), replace=True) for c in sorted classes])
    """
    y = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    by_class: Dict[int, np.ndarray] = {}
    for cls in sorted(np.unique(y)):
        by_class[int(cls)] = np.where(y == int(cls))[0]
    for _ in range(int(n_bootstrap)):
        parts = [rng.choice(idx, size=len(idx), replace=True) for idx in by_class.values()]
        yield np.concatenate(parts)


def percentile_interval(values: Sequence[float], ci: float = 0.95) -> Tuple[Optional[float], Optional[float], int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, 0
    alpha = 1.0 - ci
    return (float(np.percentile(arr, 100 * alpha / 2)),
            float(np.percentile(arr, 100 * (1 - alpha / 2))),
            int(arr.size))


def marginal_bootstrap(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    p_glabrata: Sequence[float],
    confidence: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
    ci: float = 0.95,
    keys: Sequence[str] = ("balanced_accuracy", "matthews_corrcoef", "auroc", "weighted_f1", "brier"),
) -> Dict[str, Dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    p = np.asarray(p_glabrata, dtype=float)
    c = np.asarray(confidence, dtype=float)
    point = binary_metrics(y, yp, p, c)
    dist: Dict[str, List[float]] = {k: [] for k in keys}
    for idx in stratified_bootstrap_indices(y, n_bootstrap, seed):
        m = binary_metrics(y[idx], yp[idx], p[idx], c[idx])
        for k in keys:
            v = m.get(k)
            dist[k].append(float("nan") if v is None else float(v))
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        lo, hi, n_valid = percentile_interval(dist[k], ci)
        out[k] = {"point_estimate": point.get(k), "ci_lower": lo, "ci_upper": hi,
                  "n_valid_bootstrap": n_valid}
    return out


# -----------------------------------------------------------------------------
# Exact tests, intervals, multiplicity
# -----------------------------------------------------------------------------

def clopper_pearson(k: int, n: int, ci: float = 0.95) -> Dict[str, Any]:
    """Exact (Clopper–Pearson) interval for a binomial proportion k/n."""
    k = int(k)
    n = int(n)
    if n <= 0:
        return {"k": k, "n": n, "estimate": None, "ci_lower": None, "ci_upper": None}
    alpha = 1.0 - ci
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {"k": k, "n": n, "estimate": k / n, "ci_lower": lo, "ci_upper": hi,
            "method": "Clopper-Pearson exact (beta quantiles)"}


def mcnemar_exact(correct_a: Sequence[int], correct_b: Sequence[int]) -> Dict[str, Any]:
    """
    Two-sided exact McNemar test. b = A correct & B wrong, c = A wrong & B
    correct. Under H0 (equal marginal accuracy), b | b + c ~ Binomial(b + c, 1/2);
    p = min(1, 2 * P(X <= min(b, c))).
    """
    a = np.asarray(correct_a, dtype=int)
    bvec = np.asarray(correct_b, dtype=int)
    if a.shape != bvec.shape:
        raise ValueError("McNemar inputs must have the same length.")
    both = int(((a == 1) & (bvec == 1)).sum())
    neither = int(((a == 0) & (bvec == 0)).sum())
    b = int(((a == 1) & (bvec == 0)).sum())
    c = int(((a == 0) & (bvec == 1)).sum())
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        p = min(1.0, 2.0 * float(sps.binom.cdf(min(b, c), n_disc, 0.5)))
    return {
        "table": {"both_correct": both, "a_correct_b_wrong": b,
                  "a_wrong_b_correct": c, "both_wrong": neither},
        "n_discordant": n_disc,
        "p_value_two_sided_exact": p,
    }


def holm_adjust(pvals: Mapping[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """
    Holm step-down adjustment over the non-missing p-values of one declared family:
    order p_(1) <= ... <= p_(m); adjusted p_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)).
    Missing (None / non-finite) values pass through as None and do not count in m.
    Ties are broken by key for determinism.
    """
    items = [(k, float(v)) for k, v in pvals.items()
             if v is not None and math.isfinite(float(v))]
    m = len(items)
    out: Dict[str, Optional[float]] = {k: None for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(sorted(items, key=lambda kv: (kv[1], kv[0]))):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks_sorted = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks_sorted
    return out


def delong_paired(
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
) -> Dict[str, Any]:
    """
    Fast DeLong test for two correlated AUROCs on the same samples
    (Sun & Xu, IEEE SPL 2014). Positive class = label 1.
    Reported as a cross-check only: with 12 + 12 slides the normal
    approximation is coarse, and the paired bootstrap is primary.
    If the variance of the difference is zero, p = 1 when the AUROCs are equal
    and None otherwise (the test is undefined).
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.vstack([np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)])
    pos = y == 1
    neg = y == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        return {"auc_a": None, "auc_b": None, "p_value_two_sided": None,
                "note": "one class absent"}
    k = scores.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        xp = scores[r, pos]
        yn = scores[r, neg]
        tx[r] = _midrank(xp)
        ty[r] = _midrank(yn)
        tz[r] = _midrank(np.concatenate([xp, yn]))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n          # structural components of positives
    v10 = 1.0 - (tz[:, m:] - ty) / m    # structural components of negatives
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    s = sx / m + sy / n
    diff = float(aucs[1] - aucs[0])
    var = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    if var <= 0 or not math.isfinite(var):
        z = None
        p = 1.0 if abs(diff) < 1e-15 else None
    else:
        z = diff / math.sqrt(var)
        p = float(2.0 * sps.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc_b_minus_a": diff,
        "var_delta": var,
        "z": z,
        "p_value_two_sided": p,
        "covariance_matrix": s.tolist(),
        "n_positive": m,
        "n_negative": n,
        "note": "Fast DeLong (Sun & Xu 2014); cross-check only at n = 24 slides.",
    }


# -----------------------------------------------------------------------------
# Archived Hybrid_43 reference gate
# -----------------------------------------------------------------------------

def _r4(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


def validate_reference_slide_csv(
    path: Path,
    pipeline: str,
    expected_slide_labels: Mapping[str, int],
    expected_sha256: Optional[str],
    table10: Mapping[str, Any],
    *,
    n_bootstrap: int = 10000,
    seed: int = 42,
    require_sha256: bool = True,
) -> Dict[str, Any]:
    """
    Fail-closed gate for an archived Hybrid_43 slide_predictions.csv.

    Checks, in order: (1) the slide set and labels equal the pinned cohort metadata;
    (2) every row's preprocessing_mode equals ``pipeline``; (3) row-wise
    consistency (correct == [pred == true], probabilities in [0, 1], p_a + p_g = 1);
    (4) the file SHA-256 equals the pin (skipped only when require_sha256 is False,
    i.e. while creating the pin); (5) point metrics, class-stratified slide bootstrap
    intervals (B, seed as published) and Clopper-Pearson recall intervals all equal
    main.tex Table 10 to 4 decimals. Any failure raises ValueError naming the check.
    """
    from ext_common import read_csv, sha256_file  # local import keeps ext_stats free of IO deps

    path = Path(path)
    rows = sorted(read_csv(path), key=lambda r: r["slide_id"])
    ids = [r["slide_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"[slide_set] duplicate slide_id in {path}")
    if set(ids) != set(expected_slide_labels):
        raise ValueError(f"[slide_set] {path} slides differ from the cohort metadata: "
                         f"missing={sorted(set(expected_slide_labels) - set(ids))}, "
                         f"extra={sorted(set(ids) - set(expected_slide_labels))}")
    modes = sorted({r.get("preprocessing_mode", "") for r in rows})
    if modes != [pipeline]:
        raise ValueError(f"[pipeline] {path} preprocessing_mode={modes}, expected [{pipeline!r}]")
    y = np.array([int(float(r["true_label"])) for r in rows])
    yp = np.array([int(float(r["pred_label"])) for r in rows])
    pa = np.array([float(r["mean_prob_albicans"]) for r in rows])
    pg = np.array([float(r["mean_prob_glabrata"]) for r in rows])
    conf = np.array([float(r["confidence"]) for r in rows])
    corr = np.array([int(float(r["correct"])) for r in rows])
    exp_y = np.array([int(expected_slide_labels[s]) for s in ids])
    if not np.array_equal(y, exp_y):
        raise ValueError(f"[labels] true_label in {path} disagrees with the cohort metadata")
    if not np.array_equal(corr, (yp == y).astype(int)):
        raise ValueError(f"[consistency] 'correct' column inconsistent with labels in {path}")
    if np.any((pg < 0) | (pg > 1) | (pa < 0) | (pa > 1)) or np.max(np.abs(pa + pg - 1.0)) > 1e-6:
        raise ValueError(f"[consistency] slide probabilities out of range or not summing to 1 in {path}")

    sha = sha256_file(path)
    if require_sha256:
        if expected_sha256 is None:
            raise ValueError(f"[sha256] no pin registered for {pipeline}; run once with --pin-reference")
        if sha != expected_sha256:
            raise ValueError(f"[sha256] {path} sha256={sha} differs from the pin {expected_sha256}")

    point = binary_metrics(y, yp, pg, conf)
    boot = marginal_bootstrap(y, yp, pg, conf, n_bootstrap=n_bootstrap, seed=seed,
                              keys=("balanced_accuracy", "matthews_corrcoef", "auroc"))
    cp = {"recall_albicans": clopper_pearson(point["k_correct_albicans"], point["n_albicans"]),
          "recall_glabrata": clopper_pearson(point["k_correct_glabrata"], point["n_glabrata"])}
    observed = {
        "point": {k: _r4(point[k]) for k in table10["point"]},
        "bootstrap_ci": {k: (_r4(boot[k]["ci_lower"]), _r4(boot[k]["ci_upper"]))
                         for k in table10["bootstrap_ci"]},
        "clopper_pearson": {k: (_r4(cp[k]["ci_lower"]), _r4(cp[k]["ci_upper"]))
                            for k in table10["clopper_pearson"]},
    }
    mismatches = []
    for block in ("point", "bootstrap_ci", "clopper_pearson"):
        for k, exp in table10[block].items():
            got = observed[block][k]
            exp_r = tuple(_r4(e) for e in exp) if isinstance(exp, (tuple, list)) else _r4(exp)
            if got != exp_r:
                mismatches.append(f"{block}.{k}: observed {got}, Table 10 {exp_r}")
    if mismatches:
        raise ValueError(f"[table10] {path} does not reproduce main.tex Table 10: {mismatches}")
    return {"path": str(path), "pipeline": pipeline, "sha256": sha, "sha256_pinned": expected_sha256,
            "n_slides": len(ids), "table10_reproduced": True, "observed": observed,
            "n_bootstrap": int(n_bootstrap), "seed": int(seed)}


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------

def fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:.{digits}f}"


def fmt_ci_latex(point: Optional[float], lo: Optional[float], hi: Optional[float], digits: int = 4) -> str:
    return (r"\begin{tabular}[c]{@{}c@{}}" + fmt(point, digits) +
            r"\\ {[" + fmt(lo, digits) + ", " + fmt(hi, digits) + r"]}\end{tabular}")
