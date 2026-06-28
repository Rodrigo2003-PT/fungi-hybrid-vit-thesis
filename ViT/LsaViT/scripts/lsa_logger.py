"""
LSA Training Logger

Provides per-epoch monitoring:
    1. Per-layer temperature values exp(τ_l) at each epoch.
    2. Runtime warning when any τ_l > 3.0 (approaching AMP saturation ceiling).
    3. Per-layer attention entropy H̄ (mean spatial entropy over the validation
       set) at each epoch, linking to the WCCI Gini/entropy XAI analysis.
    4. Per-fold final temperature values at the selected checkpoint epoch.

Author: Rodrigo Sá
Date: 2025
"""

import math
import warnings
from typing import Dict, List

import torch
import torch.nn as nn

from LSA_ViT import LSAAttention, SimpleViT_LSA


# ---------------------------------------------------------------------------
# Entropy utilities
# ---------------------------------------------------------------------------

def _gpu_batch_attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    Compute mean spatial entropy H̄ for one batch of attention weights,
    entirely on the originating device (GPU).

    Args:
        attn_weights: shape [B, H, N, N], post-softmax, any dtype, any device.

    Returns:
        Scalar mean entropy (Python float) averaged over B, H, and query
        positions N.  H = -sum_j a_{ij} * log(a_{ij} + eps)
    """
    with torch.no_grad():
        w = attn_weights.float()              # fp16 → fp32, stays on GPU
        eps = 1e-9
        H = -(w * (w + eps).log()).sum(dim=-1).mean()
    return H.item()                           # 4-byte scalar D2H transfer


# ---------------------------------------------------------------------------
# LSALogger
# ---------------------------------------------------------------------------

class LSALogger:
    """
    Collects and logs LSA-specific training diagnostics per fold.

    Parameters
    ----------
    model        : The SimpleViT_LSA instance being trained.
    fold_results : The mutable fold_results dict that is persisted by
                   CheckpointManager.  LSA metrics are appended in-place.
    iter_idx     : Iteration index (for log messages).
    fold_idx     : Fold index (for log messages).
    tau_warn_threshold : Raw temperature threshold above which a warning
                         is emitted (default 3.0 per workplan §4.3).
    """

    _HISTORY_KEYS = [
        "lsa_temperatures",     # List[List[float]]: per-epoch, per-layer exp(τ)
        "lsa_tau_raw",          # List[List[float]]: per-epoch, per-layer raw τ
        "lsa_attn_entropy",     # List[List[float]]: per-epoch, per-layer H̄
    ]

    def __init__(
        self,
        model: SimpleViT_LSA,
        fold_results: Dict,
        iter_idx: int,
        fold_idx: int,
        tau_warn_threshold: float = 3.0,
    ):
        self.model = model
        self.fold_results = fold_results
        self.iter_idx = iter_idx
        self.fold_idx = fold_idx
        self.tau_warn_threshold = tau_warn_threshold

        self._attn_layers: List[LSAAttention] = [
            m for m in model.modules() if isinstance(m, LSAAttention)
        ]
        self._n_layers = len(self._attn_layers)

        if self._n_layers == 0:
            raise RuntimeError(
                "LSALogger: model contains no LSAAttention layers. "
                "Did you pass a SimpleViT instead of SimpleViT_LSA?"
            )

        # Initialise history lists inside fold_results
        for key in self._HISTORY_KEYS:
            if key not in fold_results:
                fold_results[key] = []

        # Online entropy accumulators: one (sum, count) pair per layer.
        # Reset at start_validation_epoch, read at end_validation_epoch.
        # Total memory: O(depth) scalars regardless of validation set size.
        self._entropy_sum:   List[float] = [0.0] * self._n_layers
        self._entropy_count: List[int]   = [0]   * self._n_layers

        self._hooks: list = []

    # ------------------------------------------------------------------
    # Temperature logging
    # ------------------------------------------------------------------

    def _read_temperatures(self) -> Dict[str, List[float]]:
        """Read current temperature values from all LSA layers."""
        tau_raw = []
        tau_exp = []
        warnings_raised = []

        for i, layer in enumerate(self._attn_layers):
            tau = layer.temperature.item()
            tau_raw.append(tau)
            tau_exp.append(math.exp(min(tau, 4.0)))

            if tau > self.tau_warn_threshold:
                warnings_raised.append((i, tau))

        if warnings_raised:
            for layer_i, tau_val in warnings_raised:
                warnings.warn(
                    f"[LSA WARNING] Iter {self.iter_idx+1} Fold {self.fold_idx+1}: "
                    f"Layer {layer_i} temperature τ={tau_val:.4f} > {self.tau_warn_threshold}. "
                    f"exp(τ)={math.exp(tau_val):.2f}. "
                    "Model is aggressively sharpening attention — approaching AMP "
                    "saturation ceiling of 4.0. Review learning rate or weight decay.",
                    RuntimeWarning,
                    stacklevel=3,
                )

        return {"tau_raw": tau_raw, "tau_exp": tau_exp}

    # ------------------------------------------------------------------
    # Attention entropy logging
    # ------------------------------------------------------------------

    def _accumulate_entropy(self, layer_idx: int, attn_out: torch.Tensor) -> None:
        """
        Forward-hook callback: compute batch entropy on the GPU immediately
        and accumulate the scalar result for layer_idx.

        The attention tensor never leaves the GPU.  Only a 4-byte float is
        transferred to the CPU host via .item(), compared to the previous
        243 MB (fp16) D2H transfer that stalled the GPU pipeline for ~158 ms
        per hook call.

        Args:
            layer_idx : Index of the LSAAttention layer that fired.
            attn_out  : Softmax output, shape [B, H, N, N], on GPU.
        """
        batch_entropy = _gpu_batch_attention_entropy(attn_out)
        if not math.isnan(batch_entropy):
            self._entropy_sum[layer_idx]   += batch_entropy
            self._entropy_count[layer_idx] += 1

    def _register_entropy_hooks(self) -> None:
        """
        Register forward hooks on all LSAAttention layers.

        Each hook uses a default-argument capture (layer_i=i) to bind the
        loop variable by value at registration time, avoiding the classic
        Python closure-over-loop-variable pitfall.
        """
        self._hooks.clear()

        for i, layer in enumerate(self._attn_layers):
            hook = layer.attend.register_forward_hook(
                lambda module, inp, out, layer_i=i: self._accumulate_entropy(layer_i, out)
            )
            self._hooks.append(hook)

    def _remove_entropy_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _finalise_entropy(self) -> List[float]:
        """
        Return the mean entropy per layer from the online accumulators
        and reset the accumulators for the next epoch.
        """
        result = []
        for layer_i in range(self._n_layers):
            count = self._entropy_count[layer_i]
            if count > 0:
                result.append(self._entropy_sum[layer_i] / count)
            else:
                result.append(float("nan"))

        # Reset for the next validation epoch
        self._entropy_sum   = [0.0] * self._n_layers
        self._entropy_count = [0]   * self._n_layers

        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start_validation_epoch(self) -> None:
        """
        Call BEFORE iterating over the validation DataLoader.
        Resets online entropy accumulators and registers forward hooks.
        """
        # Reset accumulators first so a re-used logger starts clean
        self._entropy_sum   = [0.0] * self._n_layers
        self._entropy_count = [0]   * self._n_layers
        self._register_entropy_hooks()

    def end_validation_epoch(self, epoch: int, verbose: bool = True) -> Dict:
        """
        Call AFTER the validation DataLoader is exhausted.
        Computes and stores all LSA metrics for the current epoch.

        Args:
            epoch   : Current epoch index (0-based).
            verbose : Print per-layer temperature table.

        Returns:
            dict with keys 'tau_raw', 'tau_exp', 'entropy' (lists over layers).
        """
        self._remove_entropy_hooks()

        temps   = self._read_temperatures()
        entropy = self._finalise_entropy()

        # Append to fold_results history
        self.fold_results["lsa_temperatures"].append(temps["tau_exp"])
        self.fold_results["lsa_tau_raw"].append(temps["tau_raw"])
        self.fold_results["lsa_attn_entropy"].append(entropy)

        if verbose:
            print(f"\n  [LSA] Epoch {epoch+1} temperature & entropy summary:")
            print(f"  {'Layer':>6}  {'τ (raw)':>10}  {'exp(τ)':>10}  {'H̄ (entropy)':>14}")
            print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*14}")
            for i in range(self._n_layers):
                flag = " ⚠" if temps["tau_raw"][i] > self.tau_warn_threshold else ""
                print(
                    f"  {i:>6}  {temps['tau_raw'][i]:>10.4f}  "
                    f"{temps['tau_exp'][i]:>10.4f}  {entropy[i]:>14.4f}{flag}"
                )

        return {
            "tau_raw": temps["tau_raw"],
            "tau_exp": temps["tau_exp"],
            "entropy": entropy,
        }

    def log_fold_summary(self, best_epoch: int) -> Dict:
        """
        Extracts and prints the per-layer temperature values at the best epoch.

        Returns:
            dict with 'best_epoch_temperatures' and 'best_epoch_entropy'.
        """
        temps_history   = self.fold_results.get("lsa_temperatures", [])
        entropy_history = self.fold_results.get("lsa_attn_entropy", [])

        if not temps_history or best_epoch >= len(temps_history):
            print(
                f"  [LSA] Warning: cannot retrieve temperatures for best_epoch={best_epoch+1} "
                f"(only {len(temps_history)} epochs recorded)."
            )
            return {}

        best_temps   = temps_history[best_epoch]
        best_entropy = entropy_history[best_epoch] if entropy_history else []

        print(f"\n  [LSA] Fold {self.fold_idx+1} — Temperature at best epoch ({best_epoch+1}):")
        for i, (t, e) in enumerate(zip(best_temps, best_entropy)):
            print(f"    Layer {i}: exp(τ)={t:.4f}, H̄={e:.4f}")

        # Persist in fold_results for JSON serialisation
        self.fold_results["lsa_best_epoch_temperatures"] = best_temps
        self.fold_results["lsa_best_epoch_entropy"]       = best_entropy

        return {
            "best_epoch_temperatures": best_temps,
            "best_epoch_entropy":      best_entropy,
        }

    @staticmethod
    def is_lsa_model(model: nn.Module) -> bool:
        """Return True if the model contains at least one LSAAttention layer."""
        return any(isinstance(m, LSAAttention) for m in model.modules())
