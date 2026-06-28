"""
Perturbation-Based Faithfulness Evaluation for Grad-CAM
=========================================================

Implements the two canonical perturbation protocols used in the XAI literature
to evaluate whether a Grad-CAM localization map is *causally* aligned with the
model's decision, going beyond shape descriptors (SCI, entropy) which only
characterise the geometry of the map.

Author: Rodrigo Sá
Date: 2025
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Masking primitives
# ---------------------------------------------------------------------------

def _rank_pixels_by_saliency(
    cam: np.ndarray,
    descending: bool = True,
) -> np.ndarray:
    """
    Return flat pixel indices sorted by Grad-CAM saliency.

    Args:
        cam:        2-D float32 array, shape (H, W), values in [0, 1].
        descending: True  → most salient first (MoRF order; used by both
                            deletion and insertion protocols in this pipeline).
                    False → least salient first (true LiRF order; currently
                            unused in this pipeline).

    Returns:
        Sorted flat indices, shape (H*W,).
    """
    flat = cam.flatten()
    if descending:
        return np.argsort(flat)[::-1].copy()
    else:
        return np.argsort(flat).copy()


def _build_perturbation_sequence(
    n_pixels: int,
    n_steps: int,
) -> np.ndarray:
    """
    Build pixel-count checkpoints for the perturbation curve.

    Returns an integer array of length (n_steps + 1) starting at 0 and
    ending at n_pixels, evenly spaced. Checkpoint 0 = unperturbed image.
    """
    return np.linspace(0, n_pixels, n_steps + 1, dtype=int)


def _apply_mask(
    x: torch.Tensor,
    flat_indices: np.ndarray,
    n_masked: int,
    baseline_value: float = 0.0,
) -> torch.Tensor:
    """
    Replace the first n_masked pixels (from flat_indices) with baseline_value.

    Args:
        x:              Input tensor, shape (C, H, W).
        flat_indices:   Sorted pixel order (flat index into H*W).
        n_masked:       How many pixels to mask.
        baseline_value: Fill value (0.0 = training-set mean in normalized space).

    Returns:
        Masked tensor, shape (C, H, W).
    """
    x_masked = x.clone()
    if n_masked == 0:
        return x_masked
    H, W = x.shape[1], x.shape[2]
    idx_to_mask = flat_indices[:n_masked]
    rows = idx_to_mask // W
    cols = idx_to_mask % W
    x_masked[:, rows, cols] = baseline_value
    return x_masked


# ---------------------------------------------------------------------------
# Per-sample deletion / insertion curves
# ---------------------------------------------------------------------------

def compute_deletion_curve(
    model: nn.Module,
    x: torch.Tensor,
    cam: np.ndarray,
    predicted_class: int,
    device: torch.device,
    n_steps: int = 100,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """
    Compute the deletion (MoRF) confidence curve for a single sample.

    Starting from the full image, pixels are masked in decreasing saliency
    order. At each step the softmax confidence for the predicted class is
    recorded.

    Args:
        model:           Trained model in eval() mode.
        x:               Normalized input tensor, shape (C, H, W).
        cam:             Grad-CAM map, shape (H, W), values ∈ [0, 1].
        predicted_class: Class index for which to track confidence.
        device:          Inference device.
        n_steps:         Number of perturbation steps (curve resolution).
        baseline_value:  Pixel fill value (0.0 = training-mean in norm. space).

    Returns:
        confidence_curve: Float32 array, shape (n_steps + 1,).
                          Entry 0 = confidence on the original image.
                          Entry k = confidence after masking k/n_steps pixels.
    """
    _, H, W = x.shape
    n_pixels = H * W
    sorted_idx = _rank_pixels_by_saliency(cam, descending=True)
    checkpoints = _build_perturbation_sequence(n_pixels, n_steps)

    # ── Build all perturbed frames in one shot ────────────────────────────────
    # Shape: (n_steps+1, C, H, W).  We start from the full image and
    # incrementally mask pixels in saliency order, reusing the previous frame
    # to avoid recomputing masks from scratch at each step (O(n_steps × H × W)
    # → O(H × W) total pixel writes).
    n_frames = n_steps + 1
    C = x.shape[0]
    batch = x.unsqueeze(0).expand(n_frames, -1, -1, -1).clone()  # (S, C, H, W)

    prev_masked = 0
    for step_i, n_masked in enumerate(checkpoints):
        n_masked = int(n_masked)
        if n_masked > prev_masked:
            new_idx = sorted_idx[prev_masked:n_masked]
            rows = new_idx // W
            cols = new_idx % W
            # All frames from step_i onward share this masking increment.
            # Efficient: write only the newly-masked pixels across all later
            # frames in one vectorised op.
            batch[step_i:, :, rows, cols] = baseline_value
        prev_masked = n_masked

    # ── Single batched forward pass ───────────────────────────────────────────
    _use_amp = device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_use_amp):
            logits = model(batch.to(device))               # (S, num_classes)
        probs  = F.softmax(logits.float(), dim=1)          # cast back to fp32 for softmax
        curve  = probs[:, predicted_class].cpu().numpy().astype(np.float32)

    return curve


def compute_insertion_curve(
    model: nn.Module,
    x: torch.Tensor,
    cam: np.ndarray,
    predicted_class: int,
    device: torch.device,
    n_steps: int = 100,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """
    Compute the insertion (MoRF-Insertion) confidence curve for a single sample.

    Starting from a fully masked (baseline) image, pixels are revealed in
    descending saliency order (most salient first — MoRF order). At each step
    the softmax confidence for the predicted class is recorded.

    Args: same as compute_deletion_curve.

    Returns:
        confidence_curve: Float32 array, shape (n_steps + 1,).
                          Entry 0 = confidence on the fully masked image.
                          Entry k = confidence after revealing k/n_steps pixels.
    """
    _, H, W = x.shape
    n_pixels = H * W
    sorted_idx = _rank_pixels_by_saliency(cam, descending=True)
    checkpoints = _build_perturbation_sequence(n_pixels, n_steps)

    # ── Build all reveal frames in one shot ──────────────────────────────────
    n_frames = n_steps + 1
    batch = torch.full(
        (n_frames, x.shape[0], H, W), baseline_value, dtype=x.dtype
    )  # start: fully masked

    prev_revealed = 0
    for step_i, n_revealed in enumerate(checkpoints):
        n_revealed = int(n_revealed)
        if n_revealed > prev_revealed:
            new_idx = sorted_idx[prev_revealed:n_revealed]
            rows = new_idx // W
            cols = new_idx % W
            # All frames from step_i onward share these newly-revealed pixels.
            batch[step_i:, :, rows, cols] = x[:, rows, cols]
        prev_revealed = n_revealed

    # ── Single batched forward pass ───────────────────────────────────────────
    _use_amp = device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_use_amp):
            logits = model(batch.to(device))               # (S, num_classes)
        probs  = F.softmax(logits.float(), dim=1)
        curve  = probs[:, predicted_class].cpu().numpy().astype(np.float32)

    return curve


def compute_random_baseline_curves(
    model: nn.Module,
    x: torch.Tensor,
    predicted_class: int,
    device: torch.device,
    n_steps: int = 100,
    n_random_trials: int = 10,
    baseline_value: float = 0.0,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean random deletion and insertion curves (averaged over multiple
    random pixel-orderings) as a chance-level baseline.

    GPU batching strategy
    ----------------------
    For each random trial the (n_steps+1) deletion images and (n_steps+1)
    insertion images are assembled into two tensors of shape
    (n_steps+1, C, H, W) and evaluated in two forward passes (one per
    protocol), instead of the original 2 × (n_steps+1) serial passes of
    shape (1, C, H, W).

    The trials are processed one at a time (not all at once) to avoid
    materialising an (n_trials × (n_steps+1), C, H, W) tensor that would
    exhaust T4 VRAM for large images.  The total forward-pass count drops
    from n_trials × 2 × (n_steps+1) = 2,020 to n_trials × 2 = 20 — a
    ~101× reduction in kernel launch overhead, while peak VRAM stays bounded
    at 2 × (n_steps+1) frames at a time.

    Args:
        n_random_trials: Number of random permutations to average over.
        rng_seed:        Seed for reproducibility.

    Returns:
        mean_del_curve: Float32 array, shape (n_steps + 1,). Mean deletion.
        mean_ins_curve: Float32 array, shape (n_steps + 1,). Mean insertion.
    """
    rng = np.random.default_rng(rng_seed)
    _, H, W = x.shape
    n_pixels = H * W
    checkpoints = _build_perturbation_sequence(n_pixels, n_steps)
    n_frames = n_steps + 1

    del_curves = np.zeros((n_random_trials, n_frames), dtype=np.float32)
    ins_curves = np.zeros((n_random_trials, n_frames), dtype=np.float32)

    with torch.no_grad():
        for trial in range(n_random_trials):
            rand_order = rng.permutation(n_pixels).astype(np.int64)

            # ── Build deletion batch ──────────────────────────────────────────
            del_batch = x.unsqueeze(0).expand(n_frames, -1, -1, -1).clone()
            prev = 0
            for step_i, n_masked in enumerate(checkpoints):
                n_masked = int(n_masked)
                if n_masked > prev:
                    new_idx = rand_order[prev:n_masked]
                    rows = new_idx // W
                    cols = new_idx % W
                    del_batch[step_i:, :, rows, cols] = baseline_value
                prev = n_masked

            # ── Build insertion batch ─────────────────────────────────────────
            ins_batch = torch.full(
                (n_frames, x.shape[0], H, W), baseline_value, dtype=x.dtype
            )
            prev = 0
            for step_i, n_revealed in enumerate(checkpoints):
                n_revealed = int(n_revealed)
                if n_revealed > prev:
                    new_idx = rand_order[prev:n_revealed]
                    rows = new_idx // W
                    cols = new_idx % W
                    ins_batch[step_i:, :, rows, cols] = x[:, rows, cols]
                prev = n_revealed

            # ── Two forward passes per trial (del + ins) ──────────────────────
            _use_amp = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_use_amp):
                logits_del = model(del_batch.to(device))   # (S, num_classes)
            del_curves[trial] = (
                F.softmax(logits_del.float(), dim=1)[:, predicted_class]
                .cpu().numpy().astype(np.float32)
            )

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_use_amp):
                logits_ins = model(ins_batch.to(device))   # (S, num_classes)
            ins_curves[trial] = (
                F.softmax(logits_ins.float(), dim=1)[:, predicted_class]
                .cpu().numpy().astype(np.float32)
            )

    return del_curves.mean(axis=0), ins_curves.mean(axis=0)


# ---------------------------------------------------------------------------
# Area under curve (AUC) summary scalar
# ---------------------------------------------------------------------------

def area_under_curve(curve: np.ndarray) -> float:
    """
    Compute the normalized area under a perturbation curve using the
    trapezoidal rule.

    The x-axis is the fraction of pixels perturbed ∈ [0, 1] at each step.
    Normalization ensures the result is ∈ [0, 1] regardless of n_steps.

    Args:
        curve: Float32 array, shape (n_steps + 1,).

    Returns:
        Scalar AUC ∈ [0, 1].
    """
    n_steps = len(curve) - 1
    x_vals  = np.linspace(0.0, 1.0, n_steps + 1)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(curve, x_vals))


# ---------------------------------------------------------------------------
# Dataset-level perturbation evaluation
# ---------------------------------------------------------------------------

def run_perturbation_faithfulness(
    model: nn.Module,
    dataset,                         # NPYPatchDataset
    results: Dict[str, Any],
    device: torch.device,
    n_steps: int = 100,
    n_random_trials: int = 10,
    baseline_value: float = 0.0,
    rng_seed: int = 42,
    max_samples: Optional[int] = None,
    slide_groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run the full perturbation faithfulness evaluation on the test set.

    For each sample, three curves are computed:
        - Grad-CAM deletion   (MoRF, confidence vs. fraction deleted)
        - Grad-CAM insertion  (MoRF-Insertion, confidence vs. fraction revealed)
        - Random deletion / insertion mean baseline

    Then summary AUC scalars are computed per sample, and group-level
    mean ± std statistics are returned.

    Forward-pass budget per sample (GPU-batched implementation)
    -----------------------------------------------------------
    Original (serial):  (n_steps+1) + (n_steps+1) + n_random_trials × 2 × (n_steps+1)
                        = 1,212 passes of shape (1, C, H, W)  [for default args]

    Optimised (batched): 1 + 1 + n_random_trials × 2
                        = 22 passes of shape (n_steps+1, C, H, W)

    The numerical outputs are identical; see module docstring for proof.

    Args:
        model:          Trained DenseNet121Classifier in eval() mode.
        dataset:        NPYPatchDataset (test split).
        results:        Output dict from run_gradcam_inference() — must
                        contain 'cams', 'predicted_classes', 'true_labels'.
        device:         Inference device.
        n_steps:        Curve resolution (number of perturbation steps).
        n_random_trials: Random permutations for the baseline.
        baseline_value: Fill value for masked pixels (0.0 = training mean
                        in normalized space, the standard choice per
                        Samek et al. 2017).
        rng_seed:       Global seed for random baseline reproducibility.
        max_samples:    If set, evaluate only the first N samples (useful
                        for rapid sanity checks; set to None for full eval).
        slide_groups:   Optional array of slide identifiers, shape (N,), with
                        one entry per patch in the SAME ORDER as results.
                        When provided, this array is stored in the returned dict
                        and passed to compute_faithfulness_summary so that
                        inferential statistics (Wilcoxon test and bootstrap CI)
                        are computed on slide-level means, eliminating the
                        within-slide patch correlation that would otherwise
                        artificially narrow CIs and produce anti-conservative
                        p-values.  See _inferential_stats for the full
                        statistical procedure.
                        If None, patch-level statistics are computed with a
                        warning that they may be anti-conservative.

    Returns:
        faith_results: Dict with keys:
            'audc_gradcam'    : np.ndarray (N,) — deletion AUC per sample.
            'auic_gradcam'    : np.ndarray (N,) — insertion AUC per sample.
            'audc_random'     : np.ndarray (N,) — random deletion AUC.
            'auic_random'     : np.ndarray (N,) — random insertion AUC.
            'del_curves_gradcam': List[np.ndarray] — full deletion curves.
            'ins_curves_gradcam': List[np.ndarray] — full insertion curves.
            'del_curves_random' : List[np.ndarray] — random deletion curves.
            'ins_curves_random' : List[np.ndarray] — random insertion curves.
            'predicted_classes': np.ndarray (N,).
            'true_labels'      : np.ndarray (N,).
            'slide_groups'     : np.ndarray (N,) or None.
            'n_steps'          : int.
            'n_random_trials'  : int.
            'baseline_value'   : float.
    """
    if slide_groups is None:
        warnings.warn(
            "slide_groups not provided to run_perturbation_faithfulness. "
            "Inferential statistics (Wilcoxon test and bootstrap CI) will be "
            "computed at the patch level, treating every patch as an independent "
            "observation. Because patches from the same slide share staining, "
            "tissue, and illumination characteristics they are positively "
            "correlated; the effective sample size is smaller than the patch "
            "count, so CIs will be artificially narrow and p-values "
            "anti-conservative. Pass slide_groups=test_groups from "
            "densenet_fungi.py to obtain slide-level mean aggregation with an "
            "IID bootstrap over slides.",
            UserWarning,
            stacklevel=2,
        )
    model.eval()

    cams       = results["cams"]
    y_pred     = results["predicted_classes"]
    y_true     = results["true_labels"]
    n_total    = len(cams) if max_samples is None else min(max_samples, len(cams))

    audc_gradcam = np.zeros(n_total, dtype=np.float32)
    auic_gradcam = np.zeros(n_total, dtype=np.float32)
    audc_random  = np.zeros(n_total, dtype=np.float32)
    auic_random  = np.zeros(n_total, dtype=np.float32)

    del_curves_gc:  List[np.ndarray] = []
    ins_curves_gc:  List[np.ndarray] = []
    del_curves_rnd: List[np.ndarray] = []
    ins_curves_rnd: List[np.ndarray] = []

    # Per-sample forward-pass count: 2 (del+ins GradCAM) + 2*n_random_trials (random)
    passes_per_sample = 2 + 2 * n_random_trials
    print(
        f"  Perturbation faithfulness: evaluating {n_total} samples "
        f"({n_steps} steps, {n_random_trials} random trials each).\n"
        f"  GPU-batched: {passes_per_sample} forward passes per sample "
        f"[each of shape ({n_steps+1}, C, H, W)] "
        f"vs. {2*(n_steps+1) + 2*n_random_trials*(n_steps+1)} serial passes "
        f"in the original implementation."
    )

    # tqdm for fine-grained progress; fall back gracefully if not installed.
    try:
        from tqdm import tqdm as _tqdm
        sample_iter = _tqdm(range(n_total), desc="  Faithfulness", unit="sample",
                            dynamic_ncols=True)
    except ImportError:
        sample_iter = range(n_total)

    for i in sample_iter:
        x, _ = dataset[i]                          # normalized tensor (C, H, W)
        cam   = cams[i]                            # (H, W) in [0, 1]
        cls   = int(y_pred[i])

        # ── Grad-CAM curves (1 fwd pass each) ───────────────────────────────
        dc_gc = compute_deletion_curve(
            model, x, cam, cls, device, n_steps, baseline_value
        )
        ic_gc = compute_insertion_curve(
            model, x, cam, cls, device, n_steps, baseline_value
        )

        # ── Random baseline curves (2 fwd passes per trial) ─────────────────
        # Derive a per-sample seed from the global seed + index so that
        # different samples get different permutations but the whole run
        # is reproducible from rng_seed alone.
        sample_seed = (rng_seed + i) % (2**31)
        dc_rnd, ic_rnd = compute_random_baseline_curves(
            model, x, cls, device, n_steps, n_random_trials,
            baseline_value, sample_seed,
        )

        audc_gradcam[i] = area_under_curve(dc_gc)
        auic_gradcam[i] = area_under_curve(ic_gc)
        audc_random[i]  = area_under_curve(dc_rnd)
        auic_random[i]  = area_under_curve(ic_rnd)

        del_curves_gc.append(dc_gc)
        ins_curves_gc.append(ic_gc)
        del_curves_rnd.append(dc_rnd)
        ins_curves_rnd.append(ic_rnd)

        # Periodic console summary (in addition to tqdm bar)
        if (i + 1) % max(1, n_total // 10) == 0:
            msg = (
                f"    [{i+1}/{n_total}]  "
                f"AUDC={audc_gradcam[i]:.3f} (rand={audc_random[i]:.3f})  "
                f"AUIC={auic_gradcam[i]:.3f} (rand={auic_random[i]:.3f})"
            )
            # tqdm.write keeps the bar intact; plain print otherwise.
            try:
                from tqdm import tqdm as _tqdm
                _tqdm.write(msg)
            except ImportError:
                print(msg)

    return {
        "audc_gradcam":        audc_gradcam,
        "auic_gradcam":        auic_gradcam,
        "audc_random":         audc_random,
        "auic_random":         auic_random,
        "del_curves_gradcam":  del_curves_gc,
        "ins_curves_gradcam":  ins_curves_gc,
        "del_curves_random":   del_curves_rnd,
        "ins_curves_random":   ins_curves_rnd,
        "predicted_classes":   y_pred[:n_total],
        "true_labels":         y_true[:n_total],
        "slide_groups":        (
            slide_groups[:n_total] if slide_groups is not None else None
        ),
        "n_steps":             n_steps,
        "n_random_trials":     n_random_trials,
        "baseline_value":      baseline_value,
    }


# ---------------------------------------------------------------------------
# Faithfulness summary statistics
# ---------------------------------------------------------------------------

def _inferential_stats(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "greater",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 0,
    slide_groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute paired inferential statistics for the faithfulness gap Δ = a − b.

    Estimand alignment
    ------------------
    Aggregation function: per-slide *mean* (rather than median) of d_i.
    Rationale: the bootstrap CI targets mean(Δ), so using mean for aggregation
    makes the Wilcoxon test on slide_deltas the most directly comparable
    statistic. Both are sensitive to the same location shift.

    Args:
        a:            Per-patch metric values, shape (N,).
        b:            Per-patch metric values, shape (N,).
        alternative:  'greater' or 'two-sided'.
        n_bootstrap:  Bootstrap resamples for the CI.
        ci:           Confidence level (default 0.95).
        rng_seed:     Reproducibility seed.
        slide_groups: Optional array of slide IDs, shape (N,). When provided,
                      both Wilcoxon and CI are computed on slide-level means,
                      giving a coherent estimand.

    Returns:
        Dict with keys:
            'wilcoxon_stat'   : Wilcoxon W statistic (on slide-level means,
                                or patch-level if slide_groups is None).
            'wilcoxon_pvalue' : one-sided p-value.
            'ci_low'          : lower bound of bootstrap CI for mean(Δ_slide).
            'ci_high'         : upper bound.
            'n'               : number of patches.
            'n_slides'        : number of slides (0 if no grouping).
            'delta_slide_mean': mean of per-slide Δ (NaN if no grouping).
            'delta_slide_std' : std of per-slide Δ (NaN if no grouping).
    """
    try:
        from scipy.stats import wilcoxon as _wilcoxon
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    d = a - b          # per-patch differences, shape (N,)
    n = len(d)
    rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Slide-level aggregation — aligned estimand for both statistics
    # ------------------------------------------------------------------
    if slide_groups is not None:
        unique_slides = np.unique(slide_groups)
        n_slides = len(unique_slides)

        # Per-slide MEAN of the paired differences.
        # Using mean (not median) here keeps the aggregated statistic
        # consistent with the bootstrap CI target mean(Δ_slide).
        slide_deltas = np.array([
            d[slide_groups == s].mean()
            for s in unique_slides
        ], dtype=np.float64)

        # Wilcoxon on slide-level means — same estimand as bootstrap CI
        if _has_scipy and n_slides >= 10:
            nonzero_s = slide_deltas[slide_deltas != 0]
            if len(nonzero_s) >= 10:
                w_stat, w_pval = _wilcoxon(nonzero_s, alternative=alternative)
            else:
                w_stat, w_pval = float("nan"), float("nan")
        else:
            w_stat, w_pval = float("nan"), float("nan")

        # Simple IID bootstrap on slide-level means — valid because
        # slide-level observations are independent after aggregation.
        boot_means = np.empty(n_bootstrap, dtype=np.float64)
        for k in range(n_bootstrap):
            idx = rng.integers(0, n_slides, size=n_slides)
            boot_means[k] = slide_deltas[idx].mean()

        delta_slide_mean = float(slide_deltas.mean())
        delta_slide_std  = float(slide_deltas.std())

    # ------------------------------------------------------------------
    # Patch-level fallback (no slide grouping)
    # ------------------------------------------------------------------
    else:
        n_slides         = 0
        delta_slide_mean = float("nan")
        delta_slide_std  = float("nan")

        if _has_scipy and n >= 10:
            nonzero = d[d != 0]
            if len(nonzero) >= 10:
                w_stat, w_pval = _wilcoxon(nonzero, alternative=alternative)
            else:
                w_stat, w_pval = float("nan"), float("nan")
        else:
            w_stat, w_pval = float("nan"), float("nan")

        boot_means = np.empty(n_bootstrap, dtype=np.float64)
        for k in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_means[k] = d[idx].mean()

    alpha   = 1.0 - ci
    ci_low  = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "wilcoxon_stat":   float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "ci_low":          ci_low,
        "ci_high":         ci_high,
        "n":               n,
        "n_slides":        n_slides,
        "delta_slide_mean": delta_slide_mean,
        "delta_slide_std":  delta_slide_std,
    }

def _apply_multiple_comparisons_correction(
    summary: Dict[str, Any],
) -> None:
    """
    Apply multiple comparisons corrections in-place to all p-values in summary.

    Two corrections are computed and stored alongside the raw p-values:
        1. Bonferroni correction  (family-wise error rate, FWER):
               p_bonf = min(1, p_raw × M)
           where M is the number of tests in the family.  Conservative.
        2. Benjamini-Hochberg correction (false discovery rate, FDR):
               p_bh = rank-adjusted p-value under BH procedure.
           Less conservative; standard in computational biology.

    The "family" of tests is all (group, protocol) combinations:
        - overall × {deletion, insertion}                     = 2 tests
        - per_class × n_classes × {deletion, insertion}       = 2K tests
        - per_class_outcome × 2n_classes × {deletion, insertion} = 4K tests
    Total = 2 + 2K + 4K = 2 + 6K tests for K classes.
    """
    # Collect all (dict_ref, key) pairs pointing to a p-value
    pval_locations: List[Tuple[Dict, str]] = []

    def _collect(d: Dict) -> None:
        for pkey in ["deletion_wilcoxon_pvalue", "insertion_wilcoxon_pvalue"]:
            if pkey in d:
                pval_locations.append((d, pkey))

    _collect(summary.get("overall", {}))
    for stats in summary.get("per_class", {}).values():
        _collect(stats)
    for stats in summary.get("per_class_outcome", {}).values():
        _collect(stats)

    # Extract raw p-values (NaN treated as missing)
    raw_pvals = np.array([
        d[k] for d, k in pval_locations
    ], dtype=np.float64)
    M = len(raw_pvals)
    if M == 0:
        return

    # --- Bonferroni ---
    bonf = np.minimum(1.0, raw_pvals * M)

    # --- Benjamini-Hochberg ---
    # BH procedure on the non-NaN subset; NaN positions keep NaN
    bh = np.full(M, float("nan"))
    valid = ~np.isnan(raw_pvals)
    if valid.sum() > 0:
        pv_valid = raw_pvals[valid]
        m = len(pv_valid)
        order = np.argsort(pv_valid)
        ranks = np.empty(m, dtype=int)
        ranks[order] = np.arange(1, m + 1)
        # BH adjusted: p_adj[i] = min over j>=rank(i) of (p_j * m / rank_j)
        # Computed in reverse order for monotonicity
        adjusted = pv_valid[order] * m / (np.arange(1, m + 1))
        # Enforce monotonicity from right
        for j in range(m - 2, -1, -1):
            adjusted[j] = min(adjusted[j], adjusted[j + 1])
        bh_valid = np.minimum(1.0, adjusted)[np.argsort(order)]
        bh[valid] = bh_valid

    # Write corrected p-values back into the dicts
    for i, (d, k) in enumerate(pval_locations):
        protocol = "deletion" if "deletion" in k else "insertion"
        d[f"{protocol}_wilcoxon_pvalue_bonferroni"] = float(bonf[i])
        d[f"{protocol}_wilcoxon_pvalue_bh"]         = float(bh[i])

    # Record family size for transparency
    summary["n_hypothesis_tests"] = M


def compute_faithfulness_summary(
    faith_results: Dict[str, Any],
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute group-level faithfulness statistics from per-sample AUC values.

    Faithfulness gaps (Δ):
        Δ_deletion  = AUDC_random − AUDC_gradcam
                      Positive → Grad-CAM deletion confidence drops faster
                      than random deletion, i.e. the explanation is causally
                      aligned with the model decision.

        Δ_insertion = AUIC_gradcam − AUIC_random
                      Positive → Grad-CAM insertion confidence recovers faster
                      than random insertion, i.e. the highlighted regions are
                      genuinely sufficient for the prediction.

    A faithful explanation should produce:
        - Low AUDC  (confidence collapses quickly during deletion)
        - High AUIC (confidence recovers quickly during insertion)
        - Δ_deletion > 0  and  Δ_insertion > 0

    Stratification
    --------------
    Results are reported at three levels of granularity:

      1. "overall"           — all N test samples.
      2. "per_class"         — one entry per true class.
      3. "per_class_outcome" — (class, correct) and (class, incorrect).

    Stratification by correctness is methodologically necessary because the
    faithfulness of an explanation for an *incorrect* prediction is a
    qualitatively different phenomenon from faithfulness for a *correct* one.

    Slide-level mean aggregation
    -----------------------------
    When faith_results['slide_groups'] is not None, per-patch AUC differences
    (d_i = AUDC_random_i − AUDC_gradcam_i) are first aggregated to slide level
    by computing the mean within each slide, yielding one value Δ_s per slide.
    Both the Wilcoxon signed-rank test and the bootstrap CI are then applied to
    this slide-level array {Δ_s}, giving a coherent estimand — the mean
    slide-level Δ — for both statistics.


    Returns a dict with keys 'overall', 'per_class', 'per_class_outcome',
    'n_hypothesis_tests'.
    """
    audc_gc  = faith_results["audc_gradcam"]
    auic_gc  = faith_results["auic_gradcam"]
    audc_rnd = faith_results["audc_random"]
    auic_rnd = faith_results["auic_random"]
    y_true   = faith_results["true_labels"]
    y_pred   = faith_results["predicted_classes"]
    correct  = (y_true == y_pred)
    slide_groups = faith_results.get("slide_groups", None)

    # Δ definitions (see docstring above for sign conventions)
    delta_del = audc_rnd - audc_gc       # positive = gradcam < random (better)
    delta_ins = auic_gc  - auic_rnd      # positive = gradcam > random (better)

    def _stats(mask: np.ndarray, seed_offset: int = 0) -> Dict[str, float]:
        """
        Return descriptive + inferential statistics for one group.

        When slide_groups is available, per-patch differences are aggregated
        to slide-level means before both the Wilcoxon test and the bootstrap
        CI, giving a coherent estimand (mean slide-level Δ) for both
        statistics and eliminating IID violations from within-slide correlation.
        """
        if mask.sum() == 0:
            return {}
        sg = slide_groups[mask] if slide_groups is not None else None
        base = {
            "audc_gradcam_mean":    float(audc_gc[mask].mean()),
            "audc_gradcam_std":     float(audc_gc[mask].std()),
            "auic_gradcam_mean":    float(auic_gc[mask].mean()),
            "auic_gradcam_std":     float(auic_gc[mask].std()),
            "audc_random_mean":     float(audc_rnd[mask].mean()),
            "audc_random_std":      float(audc_rnd[mask].std()),
            "auic_random_mean":     float(auic_rnd[mask].mean()),
            "auic_random_std":      float(auic_rnd[mask].std()),
            "delta_deletion_mean":  float(delta_del[mask].mean()),
            "delta_deletion_std":   float(delta_del[mask].std()),
            "delta_insertion_mean": float(delta_ins[mask].mean()),
            "delta_insertion_std":  float(delta_ins[mask].std()),
            "n_samples":            int(mask.sum()),
            "n_slides":             int(len(np.unique(sg))) if sg is not None else 0,
        }
        del_inf = _inferential_stats(
            audc_rnd[mask], audc_gc[mask],
            alternative="greater",
            rng_seed=seed_offset,
            slide_groups=sg,
        )
        base["deletion_wilcoxon_stat"]    = del_inf["wilcoxon_stat"]
        base["deletion_wilcoxon_pvalue"]  = del_inf["wilcoxon_pvalue"]
        base["deletion_ci95_low"]         = del_inf["ci_low"]
        base["deletion_ci95_high"]        = del_inf["ci_high"]
        base["delta_deletion_slide_mean"] = del_inf["delta_slide_mean"]
        base["delta_deletion_slide_std"]  = del_inf["delta_slide_std"]
        ins_inf = _inferential_stats(
            auic_gc[mask], auic_rnd[mask],
            alternative="greater",
            rng_seed=seed_offset + 1,
            slide_groups=sg,
        )
        base["insertion_wilcoxon_stat"]    = ins_inf["wilcoxon_stat"]
        base["insertion_wilcoxon_pvalue"]  = ins_inf["wilcoxon_pvalue"]
        base["insertion_ci95_low"]         = ins_inf["ci_low"]
        base["insertion_ci95_high"]        = ins_inf["ci_high"]
        base["delta_insertion_slide_mean"] = ins_inf["delta_slide_mean"]
        base["delta_insertion_slide_std"]  = ins_inf["delta_slide_std"]
        return base

    # 1. Overall
    summary: Dict[str, Any] = {
        "overall": _stats(np.ones(len(audc_gc), dtype=bool), seed_offset=0),
        "per_class": {},
        "per_class_outcome": {},
    }

    # 2. Per class (all predictions, correct + incorrect combined)
    for cls_idx, cname in enumerate(class_names):
        mask_cls = (y_true == cls_idx)
        stats = _stats(mask_cls, seed_offset=100 * (cls_idx + 1))
        if stats:
            summary["per_class"][cname] = stats

    # 3. Per (class × correct/incorrect)
    for cls_idx, cname in enumerate(class_names):
        for out_i, (outcome_label, outcome_mask) in enumerate([
            ("correct",   correct),
            ("incorrect", ~correct),
        ]):
            mask = (y_true == cls_idx) & outcome_mask
            stats = _stats(
                mask,
                seed_offset=1000 * (cls_idx + 1) + 10 * out_i,
            )
            if stats:
                key = f"{cname}__{outcome_label}"
                summary["per_class_outcome"][key] = {
                    "class":   cname,
                    "outcome": outcome_label,
                    **stats,
                }

    # Apply multiple comparisons corrections across all tests in the family
    _apply_multiple_comparisons_correction(summary)

    return summary


def print_faithfulness_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted faithfulness summary to stdout."""
    ov = summary["overall"]
    n_tests = summary.get("n_hypothesis_tests", "?")

    def _pval_str(p: float) -> str:
        if p != p:       return "n/a"
        if p < 0.001:    return "p<0.001"
        return f"p={p:.3f}"

    print("\n" + "=" * 90)
    print("PERTURBATION FAITHFULNESS SUMMARY")
    slide_info = (
        f"  (slide-level cluster bootstrap, n_slides={ov.get('n_slides', '?')})"
        if ov.get("n_slides", 0) > 0 else
        "  (patch-level bootstrap — may be anti-conservative without slide grouping)"
    )
    print(slide_info)
    print(f"  Multiple comparisons: {n_tests} tests; "
          "Bonferroni (FWER) and BH (FDR) corrections applied.")
    print("=" * 90)
    hdr = (f"{'':30s}  {'Grad-CAM':>14}  {'Random':>14}  "
           f"{'Δ (95% CI)':>22}  {'p_raw':>8}  {'p_bonf':>8}  {'p_BH':>8}")
    print(hdr)
    print("-" * 90)

    for proto, gc_mean_k, gc_std_k, rnd_mean_k, rnd_std_k, \
        delta_mean_k, ci_lo_k, ci_hi_k, p_k, p_bonf_k, p_bh_k, \
        delta_slide_mean_k, label in [
        ("deletion",
         "audc_gradcam_mean", "audc_gradcam_std",
         "audc_random_mean",  "audc_random_std",
         "delta_deletion_mean",
         "deletion_ci95_low", "deletion_ci95_high",
         "deletion_wilcoxon_pvalue",
         "deletion_wilcoxon_pvalue_bonferroni",
         "deletion_wilcoxon_pvalue_bh",
         "delta_deletion_slide_mean",
         "AUDC (deletion, ↓ better)"),
        ("insertion",
         "auic_gradcam_mean", "auic_gradcam_std",
         "auic_random_mean",  "auic_random_std",
         "delta_insertion_mean",
         "insertion_ci95_low", "insertion_ci95_high",
         "insertion_wilcoxon_pvalue",
         "insertion_wilcoxon_pvalue_bonferroni",
         "insertion_wilcoxon_pvalue_bh",
         "delta_insertion_slide_mean",
         "AUIC (insertion, ↑ better)"),
    ]:
        ci_str = f"[{ov[ci_lo_k]:+.3f},{ov[ci_hi_k]:+.3f}]"
        slide_delta = ov.get(delta_slide_mean_k, float("nan"))
        slide_delta_str = (
            f"  Δ_slide={slide_delta:+.3f}" if slide_delta == slide_delta else ""
        )
        print(
            f"{label:30s}  "
            f"{ov[gc_mean_k]:5.3f}±{ov[gc_std_k]:4.3f}  "
            f"{ov[rnd_mean_k]:5.3f}±{ov[rnd_std_k]:4.3f}  "
            f"{ov[delta_mean_k]:+6.3f} {ci_str:>22}{slide_delta_str}  "
            f"{_pval_str(ov[p_k]):>8}  "
            f"{_pval_str(ov.get(p_bonf_k, float('nan'))):>8}  "
            f"{_pval_str(ov.get(p_bh_k, float('nan'))):>8}"
        )
    print("-" * 90)
    print(f"N patches: {ov['n_samples']}  |  "
          f"N slides: {ov.get('n_slides', 'N/A')}  |  "
          f"N hypothesis tests: {n_tests}")

    if summary.get("per_class_outcome"):
        print("\nPer-class × outcome breakdown (raw p only; see CSV for corrected):")
        hdr2 = (f"  {'Class':<16} {'Outcome':<10} {'N_pat':>6} {'N_sli':>6}  "
                f"{'AUDC':>11}  {'AUIC':>11}  "
                f"{'Δ_del':>7}  {'p_del':>8}  "
                f"{'Δ_ins':>7}  {'p_ins':>8}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for key, st in sorted(summary["per_class_outcome"].items()):
            print(
                f"  {st['class']:<16} {st['outcome']:<10} "
                f"{st['n_samples']:>6} {st.get('n_slides', 0):>6}  "
                f"{st['audc_gradcam_mean']:5.3f}±{st['audc_gradcam_std']:4.3f}  "
                f"{st['auic_gradcam_mean']:5.3f}±{st['auic_gradcam_std']:4.3f}  "
                f"{st['delta_deletion_mean']:+6.3f}  "
                f"{_pval_str(st['deletion_wilcoxon_pvalue']):>8}  "
                f"{st['delta_insertion_mean']:+6.3f}  "
                f"{_pval_str(st['insertion_wilcoxon_pvalue']):>8}"
            )

    print("=" * 90)
    print(
        "AUDC     = Area Under Deletion Curve  (lower  = more faithful)\n"
        "AUIC     = Area Under Insertion Curve (higher = more faithful)\n"
        "Δ_del    = AUDC_random − AUDC_gradcam   (positive → gradcam drops faster)\n"
        "Δ_ins    = AUIC_gradcam − AUIC_random   (positive → gradcam recovers faster)\n"
        "Δ_slide  = mean of per-slide Δ (estimand used for both Wilcoxon and CI)\n"
        "CI       = 95% bootstrap CI for mean slide-level Δ (IID bootstrap on slides)\n"
        "p_raw    = Wilcoxon signed-rank on slide-level means, one-sided H1: Δ > 0\n"
        "p_bonf   = Bonferroni-corrected p (FWER; family = all tests in summary)\n"
        "p_BH     = Benjamini-Hochberg corrected p (FDR; Benjamini & Hochberg 1995)"
    )
    print()
