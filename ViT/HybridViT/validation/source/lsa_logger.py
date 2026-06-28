"""
LSA Training Logger
============================================================
Provides per-epoch monitoring for any model whose transformer layers use
``LSAAttention``:

    1. Per-layer temperature values ``exp(τ_l)`` at each epoch.
    2. Runtime warning when any ``τ_l > tau_warn_threshold`` (default 3.0),
       signalling aggressive attention sharpening approaching the AMP
       saturation ceiling of 4.0.
    3. Per-layer attention entropy ``H̄`` (mean spatial entropy over the
       validation set) at each epoch, linking to the WCCI Gini/entropy
       XAI analysis.
    4. Per-fold best-epoch temperature and entropy values persisted to
       ``fold_results`` for downstream ablation analysis.

Author: Rodrigo Sá
Date: 2026
"""

import math
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from lsa_attention import LSAAttention


# ---------------------------------------------------------------------------
# Entropy utilities
# ---------------------------------------------------------------------------

def _gpu_batch_attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    Compute mean spatial entropy ``H̄`` for one batch of attention weights,
    entirely on the originating device (GPU).

    ``H = -Σ_j a_{ij} * log(a_{ij} + ε)``  averaged over B, H, and N.

    Parameters
    ----------
    attn_weights : (B, H, N, N)  post-softmax, any dtype, on any device.

    Returns
    -------
    float  — scalar mean entropy; only a 4-byte D2H transfer occurs.
    """
    with torch.no_grad():
        w   = attn_weights.float()   # fp16 → fp32, stays on GPU
        eps = 1e-9
        H   = -(w * (w + eps).log()).sum(dim=-1).mean()
    return H.item()


# ---------------------------------------------------------------------------
# LSALogger
# ---------------------------------------------------------------------------

class LSALogger:
    """
    Collects and logs LSA-specific training diagnostics per fold.

    Parameters
    ----------
    model : nn.Module
        The model being trained.  Must contain at least one ``LSAAttention``
        layer; raises ``RuntimeError`` otherwise.
    fold_results : dict
        The mutable ``fold_results`` dict persisted by ``CheckpointManager``.
        LSA metric lists are initialised in-place if not already present.
    iter_idx : int
        Iteration index (0-based), used in log messages.
    fold_idx : int
        Fold index (0-based), used in log messages.
    tau_warn_threshold : float
        Raw log-temperature threshold above which a ``RuntimeWarning`` is
        emitted (default 3.0; ``exp(3) ≈ 20``).  The hard AMP ceiling is
        4.0, enforced by the clamp in ``LSAAttention.forward``.
    """

    _HISTORY_KEYS: List[str] = [
        "lsa_temperatures",   # List[List[float]]: per-epoch, per-layer exp(τ)
        "lsa_tau_raw",        # List[List[float]]: per-epoch, per-layer raw τ
        "lsa_attn_entropy",   # List[List[float]]: per-epoch, per-layer H̄
    ]

    def __init__(
        self,
        model: nn.Module,
        fold_results: Dict,
        iter_idx: int,
        fold_idx: int,
        tau_warn_threshold: float = 3.0,
    ):
        self.model              = model
        self.fold_results       = fold_results
        self.iter_idx           = iter_idx
        self.fold_idx           = fold_idx
        self.tau_warn_threshold = tau_warn_threshold

        # Collect all LSAAttention layers in forward order.
        self._attn_layers: List[LSAAttention] = [
            m for m in model.modules() if isinstance(m, LSAAttention)
        ]
        self._n_layers = len(self._attn_layers)

        if self._n_layers == 0:
            raise RuntimeError(
                "LSALogger: model contains no LSAAttention layers. "
                "Ensure LSAAttention is imported from lsa_attention.py in "
                "both the model file and this logger — using two separate "
                "class definitions breaks isinstance() checks silently."
            )

        # Initialise history lists inside fold_results (no-op if already set).
        for key in self._HISTORY_KEYS:
            fold_results.setdefault(key, [])

        # Online entropy accumulators: one (sum, count) pair per layer.
        # Reset at start_validation_epoch, finalised at end_validation_epoch.
        self._entropy_sum:   List[float] = [0.0] * self._n_layers
        self._entropy_count: List[int]   = [0]   * self._n_layers

        self._hooks: list = []

    # ------------------------------------------------------------------
    # Public interface — call sites in training_logic.py
    # ------------------------------------------------------------------

    def start_validation_epoch(self) -> None:
        """
        Call BEFORE iterating over the validation DataLoader.
        Resets online entropy accumulators and registers forward hooks.
        """
        self._entropy_sum   = [0.0] * self._n_layers
        self._entropy_count = [0]   * self._n_layers
        self._register_entropy_hooks()

    def end_validation_epoch(self, epoch: int, verbose: bool = True) -> Dict:
        """
        Call AFTER the validation DataLoader is exhausted.
        Computes all LSA metrics for the current epoch and appends them
        to ``fold_results``.

        Parameters
        ----------
        epoch   : int   — current epoch index (0-based).
        verbose : bool  — print per-layer temperature table.

        Returns
        -------
        dict with keys ``'tau_raw'``, ``'tau_exp'``, ``'entropy'``
        (each a list over layers).
        """
        self._remove_entropy_hooks()

        temps   = self._read_temperatures()
        entropy = self._finalise_entropy()

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
        Extracts and prints per-layer temperature and entropy at the
        best checkpoint epoch.  Persists values to ``fold_results`` for
        JSON serialisation.

        Parameters
        ----------
        best_epoch : int  — 0-based index of the best epoch.

        Returns
        -------
        dict with ``'best_epoch_temperatures'`` and ``'best_epoch_entropy'``.
        """
        temps_history   = self.fold_results.get("lsa_temperatures", [])
        entropy_history = self.fold_results.get("lsa_attn_entropy",  [])

        if not temps_history or best_epoch >= len(temps_history):
            print(
                f"  [LSA] Warning: cannot retrieve temperatures for "
                f"best_epoch={best_epoch + 1} "
                f"(only {len(temps_history)} epochs recorded)."
            )
            return {}

        best_temps   = temps_history[best_epoch]
        best_entropy = (
            entropy_history[best_epoch] if entropy_history else []
        )

        print(
            f"\n  [LSA] Fold {self.fold_idx + 1} — "
            f"Temperature & entropy at best epoch ({best_epoch + 1}):"
        )
        for i, (t, e) in enumerate(zip(best_temps, best_entropy)):
            print(f"    Layer {i}: exp(τ)={t:.4f}, H̄={e:.4f}")

        # Persist in fold_results for downstream JSON serialisation.
        self.fold_results["lsa_best_epoch_temperatures"] = best_temps
        self.fold_results["lsa_best_epoch_entropy"]       = best_entropy

        return {
            "best_epoch_temperatures": best_temps,
            "best_epoch_entropy":      best_entropy,
        }

    # ------------------------------------------------------------------
    # Temperature reading
    # ------------------------------------------------------------------

    def _read_temperatures(self) -> Dict[str, List[float]]:
        """Read current ``τ`` values from all LSA layers."""
        tau_raw = []
        tau_exp = []
        warnings_raised = []

        for i, layer in enumerate(self._attn_layers):
            tau = layer.temperature.item()
            tau_raw.append(tau)
            # Mirror the forward clamp when computing the display value.
            tau_exp.append(math.exp(min(tau, 4.0)))

            if tau > self.tau_warn_threshold:
                warnings_raised.append((i, tau))

        for layer_i, tau_val in warnings_raised:
            warnings.warn(
                f"[LSA WARNING] Iter {self.iter_idx+1} Fold {self.fold_idx+1}: "
                f"Layer {layer_i} temperature τ={tau_val:.4f} > "
                f"{self.tau_warn_threshold}. "
                f"exp(τ)={math.exp(min(tau_val, 4.0)):.2f}. "
                "Attention is sharpening aggressively — approaching AMP "
                "saturation ceiling (clamp=4.0). Consider reducing LR or "
                "increasing weight decay.",
                RuntimeWarning,
                stacklevel=3,
            )

        return {"tau_raw": tau_raw, "tau_exp": tau_exp}

    # ------------------------------------------------------------------
    # Entropy hooks
    # ------------------------------------------------------------------

    def _register_entropy_hooks(self) -> None:
        """
        Register a forward hook on ``layer.attend`` (the ``nn.Softmax``)
        for each LSAAttention layer.

        Uses default-argument capture (``layer_i=i``) to bind the loop
        variable by value, avoiding the Python closure-over-loop-variable
        pitfall.
        """
        self._hooks.clear()
        for i, layer in enumerate(self._attn_layers):
            hook = layer.attend.register_forward_hook(
                lambda module, inp, out, layer_i=i:
                    self._accumulate_entropy(layer_i, out)
            )
            self._hooks.append(hook)

    def _remove_entropy_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _accumulate_entropy(
        self, layer_idx: int, attn_out: torch.Tensor
    ) -> None:
        """
        Hook callback: compute batch entropy on the GPU and accumulate
        the scalar result.  Only a 4-byte float crosses D2H.

        Parameters
        ----------
        layer_idx : int             — which LSA layer fired.
        attn_out  : (B, H, N, N)   — post-softmax weights, on GPU.
        """
        batch_entropy = _gpu_batch_attention_entropy(attn_out)
        if not math.isnan(batch_entropy):
            self._entropy_sum[layer_idx]   += batch_entropy
            self._entropy_count[layer_idx] += 1

    def _finalise_entropy(self) -> List[float]:
        """
        Return mean entropy per layer from the online accumulators and
        reset them for the next epoch.
        """
        result = []
        for i in range(self._n_layers):
            count = self._entropy_count[i]
            result.append(
                self._entropy_sum[i] / count if count > 0 else float("nan")
            )
        self._entropy_sum   = [0.0] * self._n_layers
        self._entropy_count = [0]   * self._n_layers
        return result

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_lsa_model(model: nn.Module) -> bool:
        """Return True if ``model`` contains at least one LSAAttention layer."""
        return any(isinstance(m, LSAAttention) for m in model.modules())
