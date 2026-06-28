"""
extractor.py

Architecture scope
------------------
This module is scoped exclusively to SimpleViT_LSA_ConvTok. The forward
path follows lsa_convtok_vit.py exactly:

    tokens_pre, h', w' = model.tokenizer(X)
    pe                 = _cached_posemb(h', w', ...)
    tokens_post        = tokens_pre + pe
    tokens             = model.transformer(tokens_post)
    pooled             = tokens.mean(dim=1)
    logits             = model.linear_head(pooled)

Final XAI scope
---------------
The production XAI path uses:

1. Attention rollout as the primary token-level explanation map.
2. Insertion/deletion AUC using rollout-ranked patches as the primary
   faithfulness test. Random and inverse-rollout rankings are optional
   negative controls.
3. Rollout-only review panels for qualitative plausibility assessment.

Faithfulness evaluation
-----------------------
compute_faithfulness_curves_from_maps()
    Accepts an explicit (N, P) attribution/relevance map. In the final thesis
    pipeline this map is attention rollout. Patches are ranked by decreasing
    rollout relevance and perturbed in image space. The primary perturbation
    score is the target-class logit; logit margin and probability are retained
    as secondary sensitivity outputs.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


# =============================================================================
# Forward decomposition dataclass
# =============================================================================

@dataclass(frozen=True)
class ForwardDecomposition:
    """
    Forward decomposition for SimpleViT_LSA_ConvTok.

    Fields
    ------
    patch_tokens_pre_pos  : (B, P, D) — ConvTokenizer output (post-LN), before PE.
    patch_tokens_post_pos : (B, P, D) — tokens after sincos PE addition.
    transformer_tokens    : (B, P, D) — output of LSATransformer.
    pooled                : (B, D)    — mean-pooled token representation.
    logits                : (B, C)    — classification logits.
    h_prime               : int       — token grid height (runtime value from tokenizer).
    w_prime               : int       — token grid width  (runtime value from tokenizer).
    """
    patch_tokens_pre_pos:  torch.Tensor
    patch_tokens_post_pos: torch.Tensor
    transformer_tokens:    torch.Tensor
    pooled:                torch.Tensor
    logits:                torch.Tensor
    h_prime:               int
    w_prime:               int


# =============================================================================
# Faithfulness result dataclasses
# =============================================================================

@dataclass
class FaithfulnessCurves:
    """
    Output of rollout-ranked insertion/deletion curves for one perturbation operator.

    Fields
    ------
    operator              : "zero" | "mean_patch"
    mode                  : "insertion" | "deletion"
    auc_scores            : (N,) primary AUC (logit-based, trapezoidal, normalised).
    score_curves          : (N, n_steps+1) target logit at each perturbation step.
                            PRIMARY faithfulness signal.
    logit_margin_curves   : (N, n_steps+1) logit margin at each step.
    secondary_score_curves: (N, n_steps+1) target-class probability (secondary
                            sensitivity analysis only).
    patch_order           : (N, P) patch indices sorted by decreasing rollout relevance.
    n_patches             : P — total number of patches.
    n_steps               : Number of perturbation steps recorded.

    """
    operator:               str
    mode:                   str
    auc_scores:             np.ndarray    # (N,) — raw logit-curve AUC
    score_curves:           np.ndarray    # (N, n_steps+1) — target logit
    logit_margin_curves:    np.ndarray    # (N, n_steps+1) — logit margin
    secondary_score_curves: np.ndarray    # (N, n_steps+1) — probability
    patch_order:            np.ndarray    # (N, P)
    n_patches:              int
    n_steps:                int
    ranking_variant:        str = "rollout"
    normalized_auc_scores:  Optional[np.ndarray] = None  # endpoint-normalised logit AUC/drop AUC
    endpoint_delta_scores:  Optional[np.ndarray] = None  # insertion gain or deletion drop

    def __post_init__(self) -> None:
        if self.normalized_auc_scores is None or self.endpoint_delta_scores is None:
            norm_auc, endpoint_delta = self.compute_normalized_metrics(
                self.score_curves, self.mode
            )
            if self.normalized_auc_scores is None:
                self.normalized_auc_scores = norm_auc
            if self.endpoint_delta_scores is None:
                self.endpoint_delta_scores = endpoint_delta

    @staticmethod
    def compute_normalized_metrics(
        score_curves: np.ndarray,
        mode: str,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute endpoint-normalised faithfulness metrics from logit curves.

        Insertion:  normalised curve = (s_t - s_0) / (s_T - s_0),
                    endpoint delta   = s_T - s_0.
        Deletion:   normalised curve = (s_0 - s_t) / (s_0 - s_T),
                    endpoint delta   = s_0 - s_T.

        If the endpoint delta is numerically zero or negative, the normalised
        AUC is set to NaN instead of forcing an artificial score. This keeps
        raw-logit AUC and endpoint deltas available for diagnosis while making
        pathological normalisations visible in the JSON/CSV output.
        """
        curves = np.asarray(score_curves, dtype=np.float32)
        if curves.ndim != 2:
            raise ValueError(f"score_curves must be 2D, got {curves.shape}")
        start = curves[:, 0]
        end = curves[:, -1]
        if mode == "insertion":
            delta = end - start
            norm_curve = (curves - start[:, None]) / np.maximum(delta[:, None], eps)
        elif mode == "deletion":
            delta = start - end
            norm_curve = (start[:, None] - curves) / np.maximum(delta[:, None], eps)
        else:
            raise ValueError(f"Unknown mode for normalisation: {mode}")

        invalid = ~(delta > eps)
        if np.any(invalid):
            norm_curve[invalid, :] = np.nan
        x_axis = np.linspace(0.0, 1.0, curves.shape[1], dtype=np.float32)
        norm_auc = np.trapz(norm_curve, x=x_axis, axis=1).astype(np.float32)
        return norm_auc, delta.astype(np.float32)

    @staticmethod
    def _auc_distribution_summary(vals: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Summarise per-sample AUC values using the same inferential convention
        as the spatial metrics module: mean, sample standard deviation, and a
        t-based confidence interval for the mean.

        Notes
        -----
        - AUC values are logit-curve AUCs, not probabilities.
        - The interval is descriptive/inferential over the analysed cohort.
        - For n < 2, the confidence interval collapses to the observed mean.
        """
        vals = np.asarray(vals, dtype=np.float32)
        n = int(vals.size)
        if n == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_95_lower": float("nan"),
                "ci_95_upper": float("nan"),
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
                "n": 0,
            }

        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1) if n > 1 else 0.0)
        if n > 1:
            alpha = 1.0 - float(confidence_level)
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
            se = std / float(np.sqrt(n))
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = ci_upper = mean

        return {
            "mean": mean,
            "std": std,
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "n": n,
        }

    def group_auc_summary(
        self,
        tp: np.ndarray,
        tn: np.ndarray,
        fp: np.ndarray,
        fn: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Summarise logit-based AUC by prediction group (TP/TN/FP/FN).

        The returned schema intentionally includes 95% confidence intervals so
        that faithfulness AUC reporting follows the same statistical style as
        rollout/attribution distribution summaries.
        """
        summary: Dict[str, Dict[str, float]] = {}
        for name, mask in [("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn)]:
            summary[name] = self._auc_distribution_summary(self.auc_scores[mask])
        return summary


@dataclass
class FaithfulnessResult:
    """
    Bundled rollout-ranked faithfulness output for all operators/modes.

    ranking_source records the map used to rank patches. In the production
    XAI pipeline this must be ``attention_rollout``. target_mode records
    whether the scored class is the predicted class or the true class.
    """
    ranking_source: str = "attention_rollout"
    target_mode: str = "predicted_class"
    curves: Dict[str, FaithfulnessCurves] = field(default_factory=dict)

    def add(self, curve: FaithfulnessCurves) -> None:
        variant = getattr(curve, "ranking_variant", "rollout")
        # Keep legacy keys for the primary rollout curves. Controls receive
        # explicit prefixes so downstream tables can compare them side by side.
        if variant in {"rollout", "attention_rollout"}:
            key = f"{curve.operator}_{curve.mode}"
        else:
            key = f"{variant}_{curve.operator}_{curve.mode}"
        self.curves[key] = curve

    def auc_array(self, operator: str, mode: str, ranking_variant: str = "rollout") -> np.ndarray:
        key = f"{operator}_{mode}" if ranking_variant in {"rollout", "attention_rollout"} else f"{ranking_variant}_{operator}_{mode}"
        return self.curves[key].auc_scores

    def to_summary_dict(
        self,
        tp: np.ndarray,
        tn: np.ndarray,
        fp: np.ndarray,
        fn: np.ndarray,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, curve in self.curves.items():
            out[key] = curve.group_auc_summary(tp, tn, fp, fn)
        return out


# =============================================================================
# LSA-ConvTok forward decomposition
# =============================================================================

class LSAConvTokForwardStrategy:
    """
    Forward decomposition for SimpleViT_LSA_ConvTok.

    Forward path (matches lsa_convtok_vit.py exactly):
        tokens_pre, h', w' = model.tokenizer(X)
        pe                  = _cached_posemb(h', w', model.dim, model.pe_temperature, dtype)
        tokens_post         = tokens_pre + pe
        tokens              = model.transformer(tokens_post)
        pooled              = tokens.mean(dim=1)
        logits              = model.linear_head(pooled)
    """

    def required_top_attrs(self) -> List[str]:
        return ["tokenizer", "transformer", "linear_head", "dim", "pe_temperature"]

    def required_attention_attrs(self) -> List[str]:
        return ["to_qkv", "attend", "heads", "temperature", "norm"]

    def decompose(self, model: nn.Module, X: torch.Tensor) -> ForwardDecomposition:
        from lsa_convtok_vit import _cached_posemb

        tokens_pre, h_prime, w_prime = model.tokenizer(X)
        pe         = _cached_posemb(
            h_prime, w_prime, model.dim, model.pe_temperature, tokens_pre.dtype
        )
        patch_post = tokens_pre + pe.to(device=X.device, dtype=tokens_pre.dtype)
        tokens     = model.transformer(patch_post)
        pooled     = tokens.mean(dim=1)
        logits     = model.linear_head(pooled)

        return ForwardDecomposition(
            patch_tokens_pre_pos=tokens_pre,
            patch_tokens_post_pos=patch_post,
            transformer_tokens=tokens,
            pooled=pooled,
            logits=logits,
            h_prime=h_prime,
            w_prime=w_prime,
        )


# =============================================================================
# Attention capture hook — LSA-only
# =============================================================================

class SpatialAttentionCapture:
    """
    Forward hook that reconstructs LSA attention transiently for inline rollout.

    This capture intentionally does not write full (B, heads, P, P) tensors to
    disk.  For each batch/layer it keeps only the current head-averaged matrix
    (B, P, P) long enough for the extractor to update the batch rollout matrix.
    """

    _LSA_ATTRS = ["heads", "temperature", "attend", "to_qkv", "norm"]

    def __init__(
        self,
        attention_module: nn.Module,
        layer_idx: int,
        output_dir: str,
        batch_counter: List[int],
        save_spatial: bool = False,
    ):
        self.attention_module = attention_module
        self.layer_idx        = layer_idx
        self.output_dir       = output_dir
        self.batch_counter    = batch_counter
        self.save_spatial     = False  # full attention storage is intentionally disabled
        self.latest_head_avg: Optional[torch.Tensor] = None  # CPU tensor (B,P,P)
        self.latest_token_importance: Optional[np.ndarray] = None

        missing = [a for a in self._LSA_ATTRS if not hasattr(attention_module, a)]
        if missing:
            raise AttributeError(
                f"Attention module at layer {layer_idx} is missing LSA attributes: "
                f"{missing}.  This extractor requires SimpleViT_LSA_ConvTok."
            )
        self.heads = attention_module.heads

    def clear(self) -> None:
        self.latest_head_avg = None
        self.latest_token_importance = None

    def _reconstruct_attention(
        self, module: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            x_norm = module.norm(x)
            qkv    = module.to_qkv(x_norm).chunk(3, dim=-1)
            q, k, _v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
            )
            scale = module.temperature.clamp(max=4.0).exp()
            dots  = torch.matmul(q, k.transpose(-1, -2)) * scale

            N = dots.shape[-1]
            diag_mask = torch.eye(N, device=dots.device, dtype=torch.bool)
            dots = dots.masked_fill(diag_mask, -torch.finfo(dots.dtype).max)
            attn = module.attend(dots)   # [B, heads, P, P]
        return attn

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        x = input[0].detach()
        with torch.no_grad():
            attn = self._reconstruct_attention(module, x)
            head_avg = attn.mean(dim=1).detach()        # (B,P,P)
            token_importance = head_avg.sum(dim=1)      # (B,P), diagnostic only

            # Store on CPU to avoid accumulating GPU memory across layers.
            self.latest_head_avg = head_avg.cpu().to(torch.float32)
            self.latest_token_importance = token_importance.cpu().numpy().astype(np.float32)

            del attn, head_avg, token_importance

    def load_all_batches(self) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Legacy storage path is removed; full attention tensors are never saved."""
        warnings.warn(
            "Full attention storage has been removed. Use embeddings_dict['rollout'] "
            "from inline rollout extraction instead."
        )
        return None, {}


# =============================================================================
# Main extractor class — LSA-ConvTok only
# =============================================================================

class SimpleViTEmbeddingExtractor:
    """
    Embedding and XAI extractor for SimpleViT_LSA_ConvTok.

    This class is scoped exclusively to the LSA-ConvTok architecture.  Passing
    any other model type raises AttributeError at construction time.

    Outputs (extract_embeddings)
    ----------------------------
    - pooled_embeddings        : (N, D)    — mean-pooled token vector
    - predictions              : (N,)      — argmax class predictions
    - probabilities            : (N, C)    — softmax probabilities
    - labels                   : (N,)      — ground-truth labels
    - indices                  : (N,)      — dataset sample indices
    - patch_tokens_pre_pos     : (N, P, D) — if requested
    - transformer_tokens       : (N, P, D) — if requested
    - rollout                  : (N, P)    — inline attention-rollout relevance, if requested
    - token_importance         : list[np.ndarray] — optional per-layer [N, P] summaries
    - token_grid               : (H', W')  — token spatial grid dimensions

    Architecture label
    ------------------
    arch_label = "SimpleViT_LSA_ConvTok"
    """

    arch_label: str = "SimpleViT_LSA_ConvTok"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        verbose: bool = True,
        temp_dir: Optional[str] = None,
        forward_check_atol: float = 1e-4,
        forward_check_rtol: float = 1e-4,
        completeness_warn_threshold: float = 0.1,
    ):
        self.model   = model
        self.device  = device
        self.verbose = verbose
        self.forward_check_atol = float(forward_check_atol)
        self.forward_check_rtol = float(forward_check_rtol)
        self.completeness_warn_threshold = float(completeness_warn_threshold)

        self.model.eval()

        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="vit_attention_spatial_")
        else:
            self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        if self.verbose:
            print(f"Architecture: {self.arch_label}")
            print(f"Temporary XAI checkpoint dir: {self.temp_dir}")

        self._strategy = LSAConvTokForwardStrategy()
        self._attention_captures: List[SpatialAttentionCapture] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle]     = []
        self._batch_counter = [0]

        self._verify_architecture()

    # -------------------------------------------------------------------------
    # Architecture validation
    # -------------------------------------------------------------------------

    def _verify_architecture(self) -> None:
        for attr in self._strategy.required_top_attrs():
            if not hasattr(self.model, attr):
                raise AttributeError(
                    f"Model missing required attribute '{attr}' for "
                    f"{self.arch_label}.  Got model type: {type(self.model).__name__}. "
                    "This extractor only supports SimpleViT_LSA_ConvTok."
                )

        if not hasattr(self.model.transformer, "layers"):
            raise AttributeError(
                "model.transformer.layers not found; cannot register attention hooks."
            )

        req_attn = self._strategy.required_attention_attrs()
        for li, layer in enumerate(self.model.transformer.layers):
            if not isinstance(layer, (list, tuple, nn.ModuleList)) or len(layer) < 1:
                raise AttributeError(
                    f"Unexpected transformer.layers[{li}] structure: {type(layer)}"
                )
            attn    = layer[0]
            missing = [a for a in req_attn if not hasattr(attn, a)]
            if missing:
                raise AttributeError(
                    f"Transformer layer {li} attention module missing LSA attrs: "
                    f"{missing}. Expected SimpleViT_LSA_ConvTok."
                )

    # -------------------------------------------------------------------------
    # Forward decomposition and equivalence check
    # -------------------------------------------------------------------------

    def _decompose_forward(self, X: torch.Tensor) -> ForwardDecomposition:
        return self._strategy.decompose(self.model, X)

    def _assert_forward_equivalence(
        self,
        logits_direct: torch.Tensor,
        logits_decomposed: torch.Tensor,
    ) -> None:
        if logits_direct.shape != logits_decomposed.shape:
            raise RuntimeError(
                f"Logit shape mismatch: direct={tuple(logits_direct.shape)} "
                f"vs decomposed={tuple(logits_decomposed.shape)}"
            )
        ok = torch.allclose(
            logits_direct, logits_decomposed,
            atol=self.forward_check_atol,
            rtol=self.forward_check_rtol,
        )
        if not ok:
            max_abs = (logits_direct - logits_decomposed).abs().max().item()
            raise RuntimeError(
                "Forward-pass alignment check FAILED. "
                f"max|Δ|={max_abs:.6e}. "
                "This invalidates attribution interpretability."
            )

    # -------------------------------------------------------------------------
    # Attention hooks
    # -------------------------------------------------------------------------

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attention_captures = []

    def _register_attention_hooks(
        self,
        layer_indices: Optional[List[int]],
        save_spatial: bool = False,
    ) -> None:
        self._remove_hooks()

        if layer_indices is None:
            layer_indices = list(range(len(self.model.transformer.layers)))

        for li in layer_indices:
            if li < 0 or li >= len(self.model.transformer.layers):
                warnings.warn(f"Layer {li} out of range; skipping.")
                continue
            attn_module = self.model.transformer.layers[li][0]
            capture = SpatialAttentionCapture(
                attention_module=attn_module,
                layer_idx=li,
                output_dir=self.temp_dir,
                batch_counter=self._batch_counter,
                save_spatial=False,
            )
            self._attention_captures.append(capture)
            self._hooks.append(attn_module.register_forward_hook(capture))

        if self.verbose and layer_indices:
            print(f"Registered inline LSA rollout hooks for layers: {layer_indices}")

    # -------------------------------------------------------------------------
    # Main extraction
    # -------------------------------------------------------------------------

    def extract_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_indices: np.ndarray,
        extract_patch_tokens: bool = False,
        extract_transformer_tokens: bool = False,
        extract_attention: bool = False,
        attention_layers: Optional[List[int]] = None,
        max_batches: Optional[int] = None,
        save_spatial_attention: bool = False,
        enforce_forward_equivalence: bool = True,
        rollout_discard_ratio: float = 0.9,
        max_attention_disk_gb: float = 0.0,
    ) -> Dict[str, Any]:

        expected_samples = len(dataloader.dataset)
        if len(dataset_indices) != expected_samples:
            raise ValueError(
                f"dataset_indices length ({len(dataset_indices)}) != "
                f"DataLoader dataset size ({expected_samples})"
            )

        if max_attention_disk_gb not in (None, 0, 0.0):
            warnings.warn(
                "max_attention_disk_gb is deprecated and ignored: this LSA-ConvTok "
                "pipeline never writes full attention tensors to disk; only final "
                "inline rollout relevance maps are retained."
            )

        if extract_attention:
            self._register_attention_hooks(attention_layers, save_spatial_attention)
        else:
            self._remove_hooks()

        pooled_list : List[np.ndarray] = []
        patch_list  : List[np.ndarray] = []
        tokens_list : List[np.ndarray] = []
        preds_list  : List[np.ndarray] = []
        probs_list  : List[np.ndarray] = []
        labels_list : List[np.ndarray] = []
        rollout_list: List[np.ndarray] = []
        token_importance_batches: List[List[np.ndarray]] = []

        h_prime_runtime: int = -1
        w_prime_runtime: int = -1

        total    = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
        iterator = tqdm(dataloader, total=total, desc="Extracting") if self.verbose else dataloader

        n_seen = 0
        with torch.inference_mode():
            for batch_idx, (X, y) in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                self._batch_counter[0] = int(batch_idx)
                X = X.to(self.device)

                # Only one normal forward decomposition is used per batch.
                # The direct model call is retained only for the first-batch
                # equivalence guard, where requested.
                decomp = self._decompose_forward(X)

                if enforce_forward_equivalence and batch_idx == 0:
                    logits_direct = self.model(X)
                    self._assert_forward_equivalence(logits_direct, decomp.logits)
                    del logits_direct

                probs = torch.softmax(decomp.logits, dim=1)
                preds = torch.argmax(decomp.logits, dim=1)

                if h_prime_runtime == -1:
                    h_prime_runtime = decomp.h_prime
                    w_prime_runtime = decomp.w_prime

                if extract_attention:
                    rollout_batch, token_importance_batch = self._consume_inline_rollout(
                        discard_ratio=float(rollout_discard_ratio)
                    )
                    rollout_list.append(rollout_batch)
                    token_importance_batches.append(token_importance_batch)

                pooled_list.append(decomp.pooled.detach().cpu().numpy().astype(np.float32))

                if extract_patch_tokens:
                    patch_list.append(
                        decomp.patch_tokens_pre_pos.detach().cpu().numpy().astype(np.float32)
                    )
                if extract_transformer_tokens:
                    tokens_list.append(
                        decomp.transformer_tokens.detach().cpu().numpy().astype(np.float32)
                    )

                preds_list.append(preds.detach().cpu().numpy().astype(np.int64))
                probs_list.append(probs.detach().cpu().numpy().astype(np.float32))
                labels_list.append(y.detach().cpu().numpy().astype(np.int64))

                n_seen += int(X.size(0))
                del X, probs, preds, decomp
                torch.cuda.empty_cache()
                if batch_idx % 10 == 0:
                    gc.collect()

        if n_seen != expected_samples:
            warnings.warn(
                f"Processed {n_seen} samples, expected {expected_samples} "
                "(check max_batches)."
            )

        results: Dict[str, Any] = {
            "pooled_embeddings": np.concatenate(pooled_list, axis=0),
            "predictions":       np.concatenate(preds_list,  axis=0),
            "probabilities":     np.concatenate(probs_list,  axis=0),
            "labels":            np.concatenate(labels_list, axis=0),
            "indices":           dataset_indices,
            "token_grid":        (h_prime_runtime, w_prime_runtime),
            "arch":              self.arch_label,
        }

        if extract_patch_tokens and patch_list:
            results["patch_tokens_pre_pos"] = np.concatenate(patch_list, axis=0)
        if extract_transformer_tokens and tokens_list:
            results["transformer_tokens"]   = np.concatenate(tokens_list, axis=0)

        if extract_attention and rollout_list:
            results["rollout"] = np.concatenate(rollout_list, axis=0).astype(np.float32)
            if token_importance_batches:
                n_layers = len(token_importance_batches[0])
                results["token_importance"] = [
                    np.concatenate([b[li] for b in token_importance_batches], axis=0).astype(np.float32)
                    for li in range(n_layers)
                ]
            results["rollout_mode"] = "inline_streaming"
            results["rollout_discard_ratio"] = float(rollout_discard_ratio)
            results["attention_disk_write_gb"] = 0.0
            self._remove_hooks()

        if self.verbose:
            print(
                f"\nExtraction complete: {n_seen} samples | "
                f"arch={self.arch_label} | "
                f"token_grid=({h_prime_runtime},{w_prime_runtime})"
            )

        return results

    # -------------------------------------------------------------------------
    # Attention rollout
    # -------------------------------------------------------------------------

    def _consume_inline_rollout(self, discard_ratio: float = 0.9) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Consume the current batch's transient head-averaged attention matrices
        and return final rollout relevance (B, P) without writing attention to disk.
        """
        if not self._attention_captures:
            raise RuntimeError("Inline rollout requested but no attention hooks are registered.")

        per_layer = []
        token_importance: List[np.ndarray] = []
        for cap in self._attention_captures:
            if cap.latest_head_avg is None:
                raise RuntimeError(
                    f"Missing captured attention for layer {cap.layer_idx}. "
                    "Ensure hooks are registered before the decomposed forward pass."
                )
            per_layer.append(cap.latest_head_avg.numpy())  # (B,P,P)
            if cap.latest_token_importance is not None:
                token_importance.append(cap.latest_token_importance)
            cap.clear()

        rollout = self._compute_rollout_from_head_avg(per_layer, discard_ratio=discard_ratio)
        return rollout, token_importance

    @staticmethod
    def _compute_rollout_from_head_avg(
        per_layer_head_avg: List[np.ndarray],
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """Compute Abnar-Zuidema rollout from transient (B,P,P) head-averaged layers."""
        if not per_layer_head_avg:
            raise ValueError("No layer attention matrices provided.")
        N, P, P2 = per_layer_head_avg[0].shape
        if P != P2:
            raise ValueError(f"Attention matrix must be square; got {P}x{P2}.")
        rollout = np.tile(np.eye(P, dtype=np.float32)[None, :, :], (N, 1, 1))
        I = np.eye(P, dtype=np.float32)[None, :, :]

        for A in per_layer_head_avg:
            A = np.asarray(A, dtype=np.float32)
            if A.shape != (N, P, P):
                raise ValueError(f"Layer attention shape mismatch: expected {(N,P,P)}, got {A.shape}")

            if discard_ratio > 0.0:
                flat = A.reshape(N, -1)
                thresh = np.quantile(flat, discard_ratio, axis=1, keepdims=True).reshape(N, 1, 1)
                A = np.where(A >= thresh, A, 0.0).astype(np.float32)

            row_sums = A.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums > 1e-8, row_sums, 1.0)
            A = A / row_sums
            A_aug = 0.5 * A + 0.5 * I
            rollout = np.einsum("bij,bjk->bik", A_aug, rollout).astype(np.float32)

        relevance = rollout.mean(axis=1)
        sums = relevance.sum(axis=1, keepdims=True)
        relevance = relevance / np.where(sums > 1e-8, sums, 1.0)
        return relevance.astype(np.float32)

    def compute_attention_rollout_mean_pooling(
        self,
        per_layer_attention: List[np.ndarray],
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Compute attention rollout with mean-pooling aggregation.

        Applies the Abnar & Zuidema (2020) residual augmentation:
            A_aug = 0.5 * A_head_avg + 0.5 * I

        For LSAAttention, the diagonal-masked attention is already row-stochastic
        without self-loops; adding 0.5*I restores identity flow and keeps rollout
        numerically stable.

        Parameters
        ----------
        per_layer_attention : list of (N, heads, P, P) arrays.
        discard_ratio       : Fraction of lowest-attention connections to zero out
                              before residual augmentation (improves signal/noise).

        Returns
        -------
        rollout : (N, P) relevance scores for each patch.
        """
        head_avg_layers = [np.asarray(a).mean(axis=1).astype(np.float32) for a in per_layer_attention]
        return self._compute_rollout_from_head_avg(head_avg_layers, discard_ratio=discard_ratio)

    def compute_last_layer_attention(
        self,
        attention_weights: List[np.ndarray],
    ) -> np.ndarray:
        """
        Return head-averaged last-layer attention for comparison with rollout.

        Parameters
        ----------
        attention_weights : list of (N, heads, P, P) arrays (one per layer).

        Returns
        -------
        last_layer_avg : (N, P, P) — head-averaged last-layer attention.
        """
        if not attention_weights:
            raise ValueError("No attention weights provided.")
        last = attention_weights[-1]
        return last.mean(axis=1).astype(np.float32)

    # -------------------------------------------------------------------------
    # Faithfulness evaluation — rollout-ranked insertion/deletion AUC
    # -------------------------------------------------------------------------

    def compute_faithfulness_curves_from_maps(
        self,
        dataloader: torch.utils.data.DataLoader,
        attribution_maps: np.ndarray,
        target_indices: np.ndarray,
        perturbation_steps: int = 20,
        operators: Sequence[str] = ("zero", "mean_patch"),
        modes: Sequence[str] = ("insertion", "deletion"),
        ranking_source: str = "attention_rollout",
        target_mode: str = "predicted_class",
        ranking_controls: Sequence[str] = ("random", "inverse"),
        random_seed: int = 42,
        max_batches: Optional[int] = None,
        resume: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_name: str = "faithfulness_rollout",
    ) -> FaithfulnessResult:
        """
        Compute insertion/deletion faithfulness curves from explicit patch maps.

        Final thesis semantics
        ----------------------
        ``attribution_maps`` is the attention-rollout relevance array (N, P).
        Patches are ranked by decreasing rollout relevance and perturbed in
        image space. This directly tests whether the patches with highest
        attention-flow relevance are behaviorally important for the model.

        Primary score  : target-class logit F(x)[c].
        Secondary score: logit margin F(x)[c] - max_{j != c} F(x)[j].
        Tertiary score : target-class probability (sensitivity analysis only).

        AUC is computed from the primary logit curve using trapezoidal
        integration over perturbation fractions. The function also reports
        endpoint-normalised AUC and endpoint delta metrics, so raw logit-scale
        effects can be interpreted together with confidence-independent curve
        shape.
        """
        if not operators:
            raise ValueError("At least one operator must be specified.")
        if not modes:
            raise ValueError("At least one mode must be specified.")
        valid_operators = {"zero", "mean_patch"}
        valid_modes = {"insertion", "deletion"}
        for op in operators:
            if op not in valid_operators:
                raise ValueError(f"Unknown operator '{op}'. Valid: {valid_operators}")
        for m in modes:
            if m not in valid_modes:
                raise ValueError(f"Unknown mode '{m}'. Valid: {valid_modes}")
        valid_controls = {"random", "inverse"}
        ranking_controls = tuple(str(c) for c in ranking_controls)
        for c in ranking_controls:
            if c not in valid_controls:
                raise ValueError(f"Unknown ranking control '{c}'. Valid: {valid_controls}")

        attribution_maps = np.asarray(attribution_maps, dtype=np.float32)
        if attribution_maps.ndim != 2:
            raise ValueError(f"attribution_maps must be (N, P), got {attribution_maps.shape}")
        N_total, P = attribution_maps.shape

        tgt_indices_arr = np.asarray(target_indices, dtype=np.int64)
        if tgt_indices_arr.shape != (N_total,):
            raise ValueError(
                f"target_indices must have shape ({N_total},), got {tgt_indices_arr.shape}"
            )

        h_prime, w_prime = self._get_token_grid()
        if h_prime == -1:
            h_w = int(P ** 0.5)
            if h_w * h_w != P:
                raise ValueError(f"Cannot infer square token grid from P={P}.")
            h_prime = w_prime = h_w

        # Patch orders. Primary rollout ranking uses highest relevance first.
        rollout_patch_order = np.argsort(-attribution_maps, axis=1).astype(np.int64)
        ranking_orders: Dict[str, np.ndarray] = {"rollout": rollout_patch_order}
        if "inverse" in ranking_controls:
            ranking_orders["inverse"] = rollout_patch_order[:, ::-1].copy()
        if "random" in ranking_controls:
            rng = np.random.RandomState(int(random_seed))
            ranking_orders["random"] = np.vstack([rng.permutation(P) for _ in range(N_total)]).astype(np.int64)

        step_fracs = np.linspace(0.0, 1.0, perturbation_steps + 1)

        if self.verbose:
            print(f"  Faithfulness ranking source: {ranking_source}")
            print(f"  Faithfulness target mode: {target_mode}")
            print(f"  Ranking variants: {list(ranking_orders.keys())}")
            print(f"  Attribution maps: {attribution_maps.shape}; token_grid=({h_prime},{w_prime})")

        result = FaithfulnessResult(ranking_source=str(ranking_source), target_mode=str(target_mode))
        faith_ckpt_root: Optional[Path] = None
        if checkpoint_dir is not None:
            faith_ckpt_root = Path(checkpoint_dir) / checkpoint_name
            faith_ckpt_root.mkdir(parents=True, exist_ok=True)

        for ranking_variant, batch_patch_order in ranking_orders.items():
            ranking_label = str(ranking_source) if ranking_variant == "rollout" else f"{ranking_source}_{ranking_variant}_control"
            for operator in operators:
                for mode in modes:
                    if self.verbose:
                        print(f"\n  Faithfulness: {ranking_label} | operator={operator} | mode={mode}")

                    score_curves = np.zeros((N_total, perturbation_steps + 1), dtype=np.float32)
                    margin_curves = np.zeros_like(score_curves)
                    prob_curves = np.zeros_like(score_curves)

                    pair_ckpt = None
                    if faith_ckpt_root is not None:
                        pair_ckpt = faith_ckpt_root / f"{ranking_variant}_{operator}_{mode}.npz"
                    if resume and pair_ckpt is not None and pair_ckpt.exists():
                        saved = np.load(str(pair_ckpt), allow_pickle=True)
                        result.add(FaithfulnessCurves(
                            operator=str(saved["operator"]),
                            mode=str(saved["mode"]),
                            auc_scores=saved["auc_scores"].astype(np.float32),
                            score_curves=saved["score_curves"].astype(np.float32),
                            logit_margin_curves=saved["logit_margin_curves"].astype(np.float32),
                            secondary_score_curves=saved["secondary_score_curves"].astype(np.float32),
                            patch_order=saved["patch_order"].astype(np.int64),
                            n_patches=int(saved["n_patches"]),
                            n_steps=int(saved["n_steps"]),
                            ranking_variant=str(saved["ranking_variant"]) if "ranking_variant" in saved.files else ranking_variant,
                            normalized_auc_scores=(
                                saved["normalized_auc_scores"].astype(np.float32)
                                if "normalized_auc_scores" in saved.files else None
                            ),
                            endpoint_delta_scores=(
                                saved["endpoint_delta_scores"].astype(np.float32)
                                if "endpoint_delta_scores" in saved.files else None
                            ),
                        ))
                        continue

                    sample_offset = 0
                    with torch.inference_mode():
                        for bi, (X, _) in enumerate(tqdm(
                            dataloader,
                            total=len(dataloader) if max_batches is None else min(max_batches, len(dataloader)),
                            desc=f"Faith {ranking_variant}/{operator}/{mode}",
                            disable=not self.verbose,
                        )):
                            if max_batches is not None and bi >= max_batches:
                                break

                            X = X.to(self.device)
                            B = X.size(0)
                            local_order = batch_patch_order[sample_offset: sample_offset + B]
                            local_tgt = tgt_indices_arr[sample_offset: sample_offset + B]
                            tgt_t = torch.from_numpy(local_tgt).long().to(self.device)

                            if operator == "zero":
                                X_baseline = self._build_zero_baseline(X)
                            else:
                                X_baseline = self._build_mean_patch_baseline(X, h_prime, w_prime)

                            for si, frac in enumerate(step_fracs):
                                n_patches_reveal = int(round(frac * P))
                                X_perturbed = self._apply_patch_mask(
                                    X_original=X,
                                    X_baseline=X_baseline,
                                    batch_patch_order=local_order,
                                    n_unmasked=n_patches_reveal,
                                    h_prime=h_prime,
                                    w_prime=w_prime,
                                    reveal_important=(mode == "insertion"),
                                )

                                logits = self.model(X_perturbed)
                                probs = torch.softmax(logits, dim=1)
                                C = logits.shape[1]
                                tgt_logit = logits.gather(1, tgt_t[:, None]).squeeze(1)
                                tgt_prob = probs.gather(1, tgt_t[:, None]).squeeze(1)
                                tgt_margin = _logit_margin(logits, tgt_t, C)

                                score_curves[sample_offset: sample_offset + B, si] = tgt_logit.cpu().numpy().astype(np.float32)
                                margin_curves[sample_offset: sample_offset + B, si] = tgt_margin.cpu().numpy().astype(np.float32)
                                prob_curves[sample_offset: sample_offset + B, si] = tgt_prob.cpu().numpy().astype(np.float32)

                                del X_perturbed, logits, probs, tgt_logit, tgt_prob, tgt_margin

                            del X_baseline
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            sample_offset += B
                            if bi % 5 == 0:
                                gc.collect()

                    x_axis = np.linspace(0.0, 1.0, perturbation_steps + 1, dtype=np.float32)
                    auc_scores = np.trapz(score_curves, x=x_axis, axis=1).astype(np.float32)
                    normalized_auc_scores, endpoint_delta_scores = FaithfulnessCurves.compute_normalized_metrics(
                        score_curves, mode
                    )

                    if pair_ckpt is not None:
                        np.savez_compressed(
                            str(pair_ckpt),
                            operator=np.array(operator),
                            mode=np.array(mode),
                            ranking_variant=np.array(ranking_variant),
                            auc_scores=auc_scores.astype(np.float32),
                            normalized_auc_scores=normalized_auc_scores.astype(np.float32),
                            endpoint_delta_scores=endpoint_delta_scores.astype(np.float32),
                            score_curves=score_curves,
                            logit_margin_curves=margin_curves,
                            secondary_score_curves=prob_curves,
                            patch_order=batch_patch_order.astype(np.int64),
                            n_patches=np.array(P, dtype=np.int64),
                            n_steps=np.array(perturbation_steps, dtype=np.int64),
                            ranking_source=np.array(str(ranking_source)),
                            target_mode=np.array(str(target_mode)),
                        )

                    result.add(FaithfulnessCurves(
                        operator=operator,
                        mode=mode,
                        auc_scores=auc_scores.astype(np.float32),
                        score_curves=score_curves,
                        logit_margin_curves=margin_curves,
                        secondary_score_curves=prob_curves,
                        patch_order=batch_patch_order,
                        n_patches=P,
                        n_steps=perturbation_steps,
                        ranking_variant=ranking_variant,
                        normalized_auc_scores=normalized_auc_scores,
                        endpoint_delta_scores=endpoint_delta_scores,
                    ))

        return result

    # -------------------------------------------------------------------------
    # Perturbation helpers
    # -------------------------------------------------------------------------

    def _build_zero_baseline(self, X: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(X)

    def _build_mean_patch_baseline(
        self, X: torch.Tensor, h_prime: int, w_prime: int
    ) -> torch.Tensor:
        """
        Vectorized mean-patch baseline.

        For each sample, computes one average patch template across all token
        positions and tiles that same template over the full image.  This
        preserves the original perturbation operator; it is not a local-patch
        mean replacement.
        """
        B, C, H, W = X.shape
        if H % h_prime != 0 or W % w_prime != 0:
            raise ValueError(
                f"Image size {(H, W)} must be divisible by token grid {(h_prime, w_prime)} "
                "for exact patch tiling."
            )
        ph = H // h_prime
        pw = W // w_prime

        patches = X.reshape(B, C, h_prime, ph, w_prime, pw).permute(0, 2, 4, 1, 3, 5)
        mean_patch = patches.reshape(B, h_prime * w_prime, C, ph, pw).mean(dim=1)
        tiled = mean_patch[:, None, None, :, :, :].expand(B, h_prime, w_prime, C, ph, pw)
        return tiled.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W).contiguous()

    def _apply_patch_mask(
        self,
        X_original: torch.Tensor,
        X_baseline: torch.Tensor,
        batch_patch_order: np.ndarray,
        n_unmasked: int,
        h_prime: int,
        w_prime: int,
        reveal_important: bool,
    ) -> torch.Tensor:
        """
        Vectorized image mixing by ranked patch mask.

        reveal_important=True  (insertion): reveal top-k important patches.
        reveal_important=False (deletion) : keep all except top-k important patches.
        """
        B, _C, H, W = X_original.shape
        P = h_prime * w_prime
        n_unmasked = int(np.clip(n_unmasked, 0, P))

        order = torch.as_tensor(batch_patch_order, device=X_original.device, dtype=torch.long)
        if order.shape != (B, P):
            raise ValueError(f"batch_patch_order must have shape {(B, P)}, got {tuple(order.shape)}")

        patch_mask = torch.zeros((B, P), device=X_original.device, dtype=X_original.dtype)
        if reveal_important:
            if n_unmasked > 0:
                patch_mask.scatter_(1, order[:, :n_unmasked], 1.0)
        else:
            patch_mask.fill_(1.0)
            if n_unmasked > 0:
                patch_mask.scatter_(1, order[:, :n_unmasked], 0.0)

        patch_mask = patch_mask.view(B, 1, h_prime, w_prime)
        image_mask = F.interpolate(patch_mask, size=(H, W), mode="nearest")
        return X_baseline * (1.0 - image_mask) + X_original * image_mask

    # -------------------------------------------------------------------------
    # Token-grid helpers
    # -------------------------------------------------------------------------

    def _get_token_grid(self) -> Tuple[int, int]:
        if hasattr(self.model, "_analytical_hw"):
            return self.model._analytical_hw
        if hasattr(self.model, "_expected_h") and self.model._expected_h is not None:
            return (self.model._expected_h, self.model._expected_w)
        return (-1, -1)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_embeddings(
        self,
        embeddings_dict: Dict[str, Any],
        output_path: str,
        compress: bool = True,
        include_statistics: bool = True,
    ) -> None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if include_statistics:
            stats = self.compute_embedding_statistics(embeddings_dict)
            stats_path = out_path.with_suffix(".statistics.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            if self.verbose:
                print(f"Saved embedding stats: {stats_path}")

        save_dict: Dict[str, Any] = {}
        for k, v in embeddings_dict.items():
            if k in {"attention_weights"}:
                continue
            if k == "token_grid" and isinstance(v, tuple):
                save_dict[k] = np.array(v, dtype=np.int32)
            else:
                save_dict[k] = v

        save_fn = np.savez_compressed if compress else np.savez
        save_fn(str(out_path), **save_dict)
        if self.verbose:
            print(f"Saved embeddings: {out_path}")

    @staticmethod
    def load_embeddings(embeddings_path: str, verbose: bool = True) -> Dict[str, Any]:
        p = Path(embeddings_path)
        if not p.exists():
            raise FileNotFoundError(f"Embeddings file not found: {p}")

        data = np.load(str(p), allow_pickle=True)
        d: Dict[str, Any] = {k: data[k] for k in data.files}

        if "token_grid" in d and hasattr(d["token_grid"], "__len__"):
            d["token_grid"] = tuple(int(x) for x in d["token_grid"])

        if verbose:
            keys = ", ".join(sorted(d.keys()))
            print(f"Loaded embeddings: {p.name} | keys=[{keys}]")

        return d

    def compute_embedding_statistics(self, embeddings_dict: Dict[str, Any]) -> Dict[str, Any]:
        pooled = np.asarray(embeddings_dict["pooled_embeddings"], dtype=np.float32)
        labels = np.asarray(embeddings_dict["labels"], dtype=np.int64)

        norms          = np.linalg.norm(pooled, axis=1)
        unique_classes = np.unique(labels)

        class_centroids  = []
        intraclass_means = []
        for c in unique_classes:
            m  = labels == c
            ec = pooled[m]
            if ec.shape[0] == 0:
                continue
            centroid = ec.mean(axis=0)
            class_centroids.append(centroid)
            dists = np.linalg.norm(ec - centroid[None, :], axis=1)
            intraclass_means.append(float(np.mean(dists)))

        inter_dists = []
        if len(class_centroids) >= 2:
            C = np.stack(class_centroids, axis=0)
            for i in range(C.shape[0]):
                for j in range(i + 1, C.shape[0]):
                    inter_dists.append(float(np.linalg.norm(C[i] - C[j])))

        return {
            "architecture":               self.arch_label,
            "embedding_dim":              int(pooled.shape[1]),
            "embedding_norm_mean":        float(norms.mean()),
            "embedding_norm_std":         float(norms.std()),
            "n_samples":                  int(pooled.shape[0]),
            "n_classes":                  int(len(unique_classes)),
            "intraclass_distance_mean":   float(np.mean(intraclass_means)) if intraclass_means else None,
            "interclass_distance_mean":   float(np.mean(inter_dists))      if inter_dists       else None,
        }

    def cleanup(self) -> None:
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            warnings.warn(f"Cleanup failed: {e}")


# =============================================================================
# Module-level helper
# =============================================================================

def _logit_margin(
    logits: torch.Tensor, target_indices: torch.Tensor, C: int
) -> torch.Tensor:
    """
    Compute logit margin: logit[target] - max logit of any other class.

    A positive margin means the target is the highest-scoring class.

    Parameters
    ----------
    logits         : (B, C)
    target_indices : (B,) long tensor
    C              : number of classes

    Returns
    -------
    margin : (B,) float tensor
    """
    B = logits.shape[0]
    tgt_logit = logits.gather(1, target_indices[:, None]).squeeze(1)  # (B,)

    # Build a mask that zeros out the target column, then take max of rest
    mask = torch.ones(B, C, device=logits.device, dtype=torch.bool)
    mask.scatter_(1, target_indices[:, None], False)
    other_max = logits.masked_fill(~mask, -1e9).max(dim=1).values  # (B,)

    return tgt_logit - other_max
