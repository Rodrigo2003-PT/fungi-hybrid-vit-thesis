#!/usr/bin/env python3
"""
validation_stats.py

Small-sample statistical utilities for slide-level external validation
(n = 24 independent slides). Standard library + NumPy only.

Contents
--------
clopper_pearson_ci        exact binomial CI for accuracy / per-class recall
delong_auc_ci             DeLong (1988) variance and Wald CI for AUROC
exact_mcnemar_pvalue      two-sided exact McNemar test from discordant counts
sign_flip_pvalue          exact (chunked, vectorised) or Monte Carlo paired
                          sign-flip test for a mean difference
stratified_bootstrap_indices
                          class-stratified resampling indices with a fixed,
                          documented RNG call order (classes in ascending order)
percentile_interval       percentile CI with an explicit degeneracy flag

Design notes
------------
* Percentile bootstrap intervals with n = 24 under-cover and become degenerate
  (zero width) when predictions are constant within a class. Exact binomial
  intervals are therefore reported as primary for proportions, and every
  bootstrap interval carries ``ci_degenerate`` and ``bootstrap_sd``.
* All results are descriptive. Nothing here is used for model selection,
  threshold tuning or preprocessing selection.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, Optional, Sequence

import numpy as np

EXACT_SIGN_FLIP_MAX_N = 24
SIGN_FLIP_CHUNK_BITS = 16
DEFAULT_MONTE_CARLO_DRAWS = 200_000


# -----------------------------------------------------------------------------
# Exact binomial interval
# -----------------------------------------------------------------------------

def _binom_logpmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log1p(-p)
    )


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(min(1.0, sum(math.exp(_binom_logpmf(i, n, p)) for i in range(0, k + 1))))


def _bisect(fn, lo: float, hi: float, iters: int = 200) -> float:
    """Root of a monotone function on [lo, hi] by bisection (fn(lo) and fn(hi) differ in sign)."""
    f_lo = fn(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid)
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Dict[str, Any]:
    """
    Exact Clopper-Pearson interval for a binomial proportion k/n.

    lower solves P(X >= k | p) = alpha/2 (0 if k == 0)
    upper solves P(X <= k | p) = alpha/2 (1 if k == n)
    """
    k, n = int(k), int(n)
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial counts k={k}, n={n}")
    alpha = 1.0 - float(confidence)
    if k == 0:
        lower = 0.0
    else:
        lower = _bisect(lambda p: (1.0 - binom_cdf(k - 1, n, p)) - alpha / 2.0, 0.0, 1.0)
    if k == n:
        upper = 1.0
    else:
        upper = _bisect(lambda p: binom_cdf(k, n, p) - alpha / 2.0, 0.0, 1.0)
    return {
        "method": "clopper_pearson_exact",
        "k": k,
        "n": n,
        "point_estimate": k / n,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": float(confidence),
    }


# -----------------------------------------------------------------------------
# DeLong AUROC interval
# -----------------------------------------------------------------------------

def delong_auc_ci(
    y_true: Sequence[int],
    scores: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    AUROC with DeLong structural-component variance and a Wald interval
    clipped to [0, 1]. Positive class = label 1. Ties count 1/2.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    m, n = pos.size, neg.size
    out: Dict[str, Any] = {"method": "delong_wald", "n_positive": int(m), "n_negative": int(n)}
    if m == 0 or n == 0 or not np.all(np.isfinite(s)):
        out.update(auc=None, variance=None, ci_lower=None, ci_upper=None, ci_degenerate=True,
                   note="AUROC undefined: one class absent or non-finite scores.")
        return out

    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    auc = float(psi.mean())
    var10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    var01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    variance = var10 / m + var01 / n
    z = NormalDist().inv_cdf(1.0 - (1.0 - float(confidence)) / 2.0)
    half = z * math.sqrt(max(variance, 0.0))
    degenerate = variance <= 0.0
    out.update(
        auc=auc,
        variance=variance,
        standard_error=math.sqrt(max(variance, 0.0)),
        ci_lower=None if degenerate else max(0.0, auc - half),
        ci_upper=None if degenerate else min(1.0, auc + half),
        ci_degenerate=bool(degenerate),
        confidence_level=float(confidence),
    )
    if degenerate:
        out["note"] = "DeLong variance is zero (perfect or constant separation); interval not reported."
    return out


# -----------------------------------------------------------------------------
# Paired tests
# -----------------------------------------------------------------------------

def exact_mcnemar_pvalue(improved: int, worsened: int) -> Optional[float]:
    """Two-sided exact McNemar p-value from the discordant-pair counts."""
    b, c = int(improved), int(worsened)
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def sign_flip_pvalue(
    values: Sequence[float],
    *,
    max_exact_n: int = EXACT_SIGN_FLIP_MAX_N,
    n_monte_carlo: int = DEFAULT_MONTE_CARLO_DRAWS,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Two-sided paired sign-flip test for H0: mean difference = 0.

    Zero differences (|d| <= 1e-12) are removed. For n <= max_exact_n all 2^n
    sign patterns are enumerated in vectorised chunks of 2^16; otherwise a
    seeded Monte Carlo with the observed pattern included (add-one) is used.
    """
    vals = np.asarray([float(v) for v in values if abs(float(v)) > 1e-12], dtype=float)
    n = int(vals.size)
    if n == 0:
        return {"p_value": None, "method": "none_all_differences_zero", "n_nonzero": 0}
    observed = abs(float(vals.mean()))
    tol = 1e-12 * max(1.0, observed)

    if n <= max_exact_n:
        total = 1 << n
        chunk = 1 << min(SIGN_FLIP_CHUNK_BITS, n)
        bit_positions = np.arange(n, dtype=np.int64)
        count = 0
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
            bits = (idx[:, None] >> bit_positions[None, :]) & 1
            signs = 1.0 - 2.0 * bits
            stats = np.abs(signs @ vals) / n
            count += int(np.count_nonzero(stats >= observed - tol))
        return {"p_value": float(count / total), "method": "exact", "n_nonzero": n, "n_patterns": int(total)}

    rng = np.random.default_rng(seed)
    count = 0
    remaining = int(n_monte_carlo)
    while remaining > 0:
        size = min(remaining, 1 << 16)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n))
        stats = np.abs(signs @ vals) / n
        count += int(np.count_nonzero(stats >= observed - tol))
        remaining -= size
    return {
        "p_value": float((count + 1) / (int(n_monte_carlo) + 1)),
        "method": "monte_carlo",
        "n_nonzero": n,
        "n_draws": int(n_monte_carlo),
        "seed": int(seed),
    }


# -----------------------------------------------------------------------------
# Bootstrap helpers
# -----------------------------------------------------------------------------

def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    One class-stratified bootstrap replicate. RNG call order is fixed and
    documented: one ``rng.choice(idx, size=len(idx), replace=True)`` per class,
    classes in ascending label order, indices concatenated in that order.
    """
    y = np.asarray(y_true)
    parts = []
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"ci_lower": None, "ci_upper": None, "bootstrap_mean": None, "bootstrap_sd": None,
                "n_valid_bootstrap": 0, "ci_degenerate": True}
    alpha = 1.0 - float(confidence)
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    degenerate = sd == 0.0
    return {
        "ci_lower": None if degenerate else float(np.percentile(arr, 100 * alpha / 2)),
        "ci_upper": None if degenerate else float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "bootstrap_mean": float(np.mean(arr)),
        "bootstrap_sd": sd,
        "n_valid_bootstrap": int(arr.size),
        "ci_degenerate": bool(degenerate),
        **({"degenerate_reason": "zero bootstrap variance (statistic constant under class-stratified resampling)"}
           if degenerate else {}),
    }
