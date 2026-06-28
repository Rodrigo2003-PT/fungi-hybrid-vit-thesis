"""
analysis_attention.py

Architecture scope
------------------
This module loads only SimpleViT_LSA_ConvTok. The architecture label is
hard-coded to "SimpleViT_LSA_ConvTok" throughout for clean provenance.

Execution stages (fixed order)
------------------------------
1. Extraction / provenance    — embeddings, slide IDs, token grid
2. Embedding analysis (UMAP)  — optional
3. Rollout                    — attention rollout + distributional metrics
4. Rollout faithfulness       — insertion/deletion AUC using rollout-ranked patches
4B. Rollout representatives   — faithfulness-aware stratified MMD-critic maps
5. Review panels              — rollout-only model-input display-normalised panels
6. Consolidated summary       — single split-level JSON

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import get_config
from embeddings import EmbeddingAnalyzer
from extractor import (
    FaithfulnessResult,
    SimpleViTEmbeddingExtractor,
)
from helper import (
    create_slide_id_provenance_map,
    extract_slide_id_from_filename,
    validate_slide_grouping,
)
from stratified_sampling import (
    AttentionDistributionalMetrics,
    SpatialMetricsAggregator,
    SpatialMetricsSummary,
    StratifiedMMDCriticSelector,
    StrataDefinition,
    compare_groups_patch_level,
    compute_effect_size_cohens_d,
)
from training_utils import NPYImageFolder, TransformSubset, create_transforms, get_device_info
from visualization import (
    FaithfulnessVisualizer,
    SlideConditionedVisualizer,
    SpatialAttentionVisualizer,
    RolloutReviewPanel,
    plot_metric_distributions_by_group,
)

# Architecture label: hard-coded for provenance clarity.
_ARCH_LABEL = "SimpleViT_LSA_ConvTok"


# =============================================================================
# Project paths
# =============================================================================

@dataclass(frozen=True)
class ProjectPaths:
    project_root:      Path
    preprocessed_root: Path
    preprocessed_dir:  Path
    output_root:       Path
    analysis_root:     Path

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ProjectPaths":
        project_root      = Path(args.project_root).expanduser().resolve()
        preprocessed_root = Path(args.preprocessed_root).expanduser().resolve()
        preprocessed_dir  = preprocessed_root / args.dataset_subdir
        output_root       = Path(args.output_root).expanduser().resolve()
        analysis_root     = output_root / args.experiment / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        return ProjectPaths(
            project_root=project_root,
            preprocessed_root=preprocessed_root,
            preprocessed_dir=preprocessed_dir,
            output_root=output_root,
            analysis_root=analysis_root,
        )


# =============================================================================
# Split loading
# =============================================================================

def _normalize_split_entries(split_data: Dict[str, Any], split_name: str) -> List[str]:
    k1 = f"{split_name}_base_paths"
    if k1 in split_data:
        return list(split_data[k1])
    raise KeyError(f"Split file does not contain keys for split '{split_name}'")


def load_split_with_conversion(
    split_path: str, dataset: NPYImageFolder, split_name: str
) -> List[int]:
    with open(split_path, "r") as f:
        split_data = json.load(f)

    entries = _normalize_split_entries(split_data, split_name)

    basename_to_idx: Dict[str, int] = {}
    stem_to_idx:     Dict[str, int] = {}
    rel_to_idx:      Dict[str, int] = {}
    relstem_to_idx:  Dict[str, int] = {}

    root = Path(dataset.root).resolve()

    for idx, (filepath, _) in enumerate(dataset.samples):
        p = Path(filepath)
        basename_to_idx[p.name] = idx
        stem_to_idx[p.stem]     = idx
        try:
            rel = p.resolve().relative_to(root).as_posix()
        except Exception:
            rel = p.as_posix()
        rel_to_idx[rel]                                        = idx
        relstem_to_idx[Path(rel).with_suffix("").as_posix()]   = idx

    indices: List[int] = []
    missing: List[str] = []

    for e in entries:
        if isinstance(e, int):
            indices.append(int(e))
            continue

        s = str(e)
        p = Path(s)
        candidates = [
            p.name,
            p.stem,
            s,
            p.with_suffix(".npy").name,
            p.with_suffix(".npy").as_posix(),
            Path(s).with_suffix("").as_posix(),
        ]

        found = None
        for c in candidates:
            for lookup in (basename_to_idx, stem_to_idx, rel_to_idx, relstem_to_idx):
                if c in lookup:
                    found = lookup[c]
                    break
            if found is not None:
                break

        if found is None:
            missing.append(s)
        else:
            indices.append(int(found))

    if missing:
        warnings.warn(f"{len(missing)}/{len(entries)} entries missing. First: {missing[0]}")

    return indices


# =============================================================================
# Model loader — LSA-ConvTok only
# =============================================================================

def _infer_num_classes_from_state_dict(state: Dict[str, torch.Tensor]) -> int:
    if "linear_head.weight" in state:
        return int(state["linear_head.weight"].shape[0])
    for k, v in state.items():
        if k.endswith(".weight") and v.ndim == 2 and "head" in k:
            return int(v.shape[0])
    raise ValueError("Could not infer num_classes from state_dict.")


def load_lsa_convtok_model(
    checkpoint: Dict[str, Any],
    cli_config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """
    Instantiate and load SimpleViT_LSA_ConvTok from a checkpoint.

    This is the only model loading path in this module.  Any checkpoint that
    does not match the LSA-ConvTok architecture raises ValueError.

    Parameters
    ----------
    checkpoint  : Full checkpoint dict (must contain 'model_state_dict').
    cli_config  : Config from get_config() — used as fallback for missing keys.
    device      : Target device.

    Returns
    -------
    model : SimpleViT_LSA_ConvTok, loaded and in eval mode.
    """
    from lsa_convtok_vit import SimpleViT_LSA_ConvTok

    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError("Checkpoint missing 'model_state_dict'.")

    # Sanity-check: reject obvious non-LSA-ConvTok checkpoints
    has_tokenizer = any("tokenizer" in k for k in state.keys())
    has_temperature = any("temperature" in k for k in state.keys())
    if not has_tokenizer and not has_temperature:
        raise ValueError(
            "Checkpoint does not appear to contain a SimpleViT_LSA_ConvTok "
            "state dict (no 'tokenizer' or 'temperature' keys found).  "
            "This pipeline only supports SimpleViT_LSA_ConvTok."
        )

    ckpt_config = checkpoint.get("config", dict(cli_config))

    class_to_idx = checkpoint.get("class_to_idx", None)
    num_classes  = (
        len(class_to_idx)
        if class_to_idx is not None
        else _infer_num_classes_from_state_dict(state)
    )

    model_cfg = ckpt_config.get("model_config", ckpt_config.get("model_cfg", {}))
    if not model_cfg:
        model_cfg = cli_config.get("model_config", {})
        warnings.warn("No model_config in checkpoint; using CLI model_config.")

    convtok_blocks = ckpt_config.get(
        "convtok_blocks", cli_config.get("convtok_blocks", None)
    )
    if convtok_blocks is None:
        convtok_blocks = SimpleViT_LSA_ConvTok.DEFAULT_CONVTOK_BLOCKS
        warnings.warn("convtok_blocks not in checkpoint; using model defaults.")

    model = SimpleViT_LSA_ConvTok(
        image_size=int(ckpt_config.get("image_size", cli_config["image_size"])),
        num_classes=int(num_classes),
        convtok_blocks=convtok_blocks,
        convtok_hidden_channels=int(ckpt_config.get("convtok_hidden_channels", 64)),
        convtok_match_baseline_tokens=bool(ckpt_config.get("convtok_match_baseline_tokens", True)),
        convtok_expected_hw=ckpt_config.get("convtok_expected_hw", cli_config.get("convtok_expected_hw", None)),
        convtok_conv_bias=bool(ckpt_config.get("convtok_conv_bias", False)),
        init_policy=ckpt_config.get("init_policy", "default"),
        pe_temperature=int(ckpt_config.get("pe_temperature", 10000)),
        **{k: v for k, v in dict(model_cfg).items() if k not in ("channels", "pe_temperature")},
        channels=int(ckpt_config.get("channels", cli_config.get("channels", 1))),
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_model_and_checkpoint(
    model_path: str,
    cli_config: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Load checkpoint and instantiate SimpleViT_LSA_ConvTok.

    Returns
    -------
    model      : Loaded, eval-mode SimpleViT_LSA_ConvTok.
    checkpoint : Full checkpoint dictionary.
    ckpt_config: Config from checkpoint (or cli_config if absent).

    Note: arch_name is no longer returned; it is the module-level constant
    _ARCH_LABEL = "SimpleViT_LSA_ConvTok".
    """
    checkpoint  = torch.load(model_path, map_location=device)
    ckpt_config = checkpoint.get("config", None)
    if ckpt_config is None:
        warnings.warn("Checkpoint missing 'config'. Using CLI config.")
        ckpt_config = dict(cli_config)

    for k in ["image_size", "channels"]:
        if (
            k in cli_config
            and k in ckpt_config
            and cli_config[k] != ckpt_config[k]
        ):
            warnings.warn(
                f"Config mismatch '{k}': CLI={cli_config[k]} vs CKPT={ckpt_config[k]}"
        )

    model = load_lsa_convtok_model(checkpoint, cli_config, device)
    print(f"Loaded model: {_ARCH_LABEL}")
    return model, checkpoint, ckpt_config


# =============================================================================
# Data loader
# =============================================================================

def create_dataloader_with_provenance(
    dataset: NPYImageFolder,
    indices: Sequence[int],
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
    batch_size: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray]:
    mean = checkpoint.get("mean", None)
    std  = checkpoint.get("std",  None)
    if mean is None or std is None:
        warnings.warn("Checkpoint missing mean/std normalisation statistics.")

    transform   = create_transforms(mean, std, config, train=False)
    subset      = TransformSubset(dataset, list(indices), transform)
    bs          = int(batch_size if batch_size is not None else config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 2))

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dataset_indices = np.asarray(indices, dtype=np.int64)
    print(f"DataLoader: {len(indices)} samples | batch_size={bs}")
    return loader, dataset_indices


# =============================================================================
# Slide provenance
# =============================================================================

def reconstruct_slide_id_provenance(
    dataset: NPYImageFolder,
    dataset_indices: np.ndarray,
    validate: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    slide_ids: List[str] = []
    filepaths: List[str] = []
    labels:    List[int] = []

    for idx in dataset_indices:
        filepath, label = dataset.samples[int(idx)]
        filename        = Path(filepath).name
        slide_id        = extract_slide_id_from_filename(filename, validate=validate)
        slide_ids.append(slide_id)
        filepaths.append(filepath)
        labels.append(int(label))

    slide_ids_arr = np.asarray(slide_ids, dtype=object)
    labels_arr    = np.asarray(labels,    dtype=int)

    if validate:
        ok, err = validate_slide_grouping(slide_ids_arr, labels_arr)
        if not ok:
            raise ValueError(f"Slide validation failed: {err}")

    provenance_map = create_slide_id_provenance_map(
        file_paths=filepaths,
        slide_ids=slide_ids_arr,
        labels=labels_arr,
    )
    print(f"Slide provenance: {len(np.unique(slide_ids_arr))} unique slides")
    return slide_ids_arr, provenance_map


# =============================================================================
# Stage 3: Rollout population analysis
# =============================================================================

def run_rollout_population_analysis(
    embeddings_dict: Dict[str, Any],
    slide_ids: np.ndarray,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    paths: ProjectPaths,
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: torch.device,
    num_patches_side: int,
) -> Dict[str, Any]:
    """
    Rollout-based population analysis with stratified representatives.

    Returns serialisable results dict (written to rollout_stats_{split}.json).

    Notes
    -----
    LSA-ConvTok uses diagonal-masked attention; the Abnar & Zuidema (2020)
    residual augmentation (0.5A + 0.5I) is applied inside
    compute_attention_rollout_mean_pooling.  Last-layer attention is saved
    separately when --save-last-layer-attn is passed.
    """
    if "rollout" not in embeddings_dict:
        print("  No inline rollout maps found. Re-run extraction with --compute-attention-rollout or --extract-attention.")
        return {}

    print("\n" + "=" * 70)
    print(f"ROLLOUT POPULATION ANALYSIS  [{_ARCH_LABEL}]")
    print("=" * 70)

    y_true = np.asarray(embeddings_dict["labels"],      dtype=int)
    y_pred = np.asarray(embeddings_dict["predictions"], dtype=int)

    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    n_tp, n_tn, n_fp, n_fn = np.sum(tp), np.sum(tn), np.sum(fp), np.sum(fn)

    print(f"\nCohort:  TP={n_tp}  TN={n_tn}  FP={n_fp}  FN={n_fn}  Total={len(y_true)}")

    # ---- STEP 1: Load inline rollout ----
    print(f"\n{'='*70}")
    print("STEP 1: Loading Inline Attention Rollout")
    print(f"{'='*70}")

    rollout = np.asarray(embeddings_dict["rollout"], dtype=np.float32)
    print(f"  Rollout shape: {rollout.shape}  (N, P)")
    if getattr(args, "save_last_layer_attn", False):
        warnings.warn(
            "--save-last-layer-attn is ignored in the optimized pipeline: full "
            "attention tensors are no longer stored; only final rollout relevance maps are saved."
        )

    # ---- STEP 2: Distributional metrics ----
    print(f"\n{'='*70}")
    print("STEP 2: Distributional Metrics")
    print(f"{'='*70}")

    metrics     = AttentionDistributionalMetrics.compute_rollout_metrics_batch(rollout)
    gini_all    = metrics["gini_coefficients"]
    entropy_all = metrics["spatial_entropies"]

    aggregator = SpatialMetricsAggregator()

    gini_tp    = aggregator.compute_summary(gini_all[tp])    if n_tp > 0 else None
    gini_tn    = aggregator.compute_summary(gini_all[tn])    if n_tn > 0 else None
    gini_fp    = aggregator.compute_summary(gini_all[fp])    if n_fp > 0 else None
    gini_fn    = aggregator.compute_summary(gini_all[fn])    if n_fn > 0 else None

    entropy_tp = aggregator.compute_summary(entropy_all[tp]) if n_tp > 0 else None
    entropy_tn = aggregator.compute_summary(entropy_all[tn]) if n_tn > 0 else None
    entropy_fp = aggregator.compute_summary(entropy_all[fp]) if n_fp > 0 else None
    entropy_fn = aggregator.compute_summary(entropy_all[fn]) if n_fn > 0 else None

    print(f"\n  Gini Coefficient:")
    for name, g in [("TP", gini_tp), ("FP", gini_fp), ("TN", gini_tn), ("FN", gini_fn)]:
        if g:
            print(f"    {name}: {g.mean:.4f} ± {g.std:.4f}  (95% CI: [{g.ci_lower:.4f}, {g.ci_upper:.4f}])")

    print(f"\n  Spatial Entropy:")
    for name, e in [("TP", entropy_tp), ("FP", entropy_fp), ("TN", entropy_tn), ("FN", entropy_fn)]:
        if e:
            print(f"    {name}: {e.mean:.4f} ± {e.std:.4f}  (95% CI: [{e.ci_lower:.4f}, {e.ci_upper:.4f}])")

    if gini_tp and gini_fp:
        gini_effect = compute_effect_size_cohens_d(gini_all[tp], gini_all[fp])
        print(f"\n  Cohen's d (TP vs FP Gini): {gini_effect:.3f}")
        stat_test = aggregator.compare_strata(gini_all[tp], gini_all[fp])
        print(f"  Welch's t-test (TP vs FP Gini): p = {stat_test['p_value']:.4f}")

    # ---- STEP 3: Save population statistics ----
    results = {
        "experiment":  args.experiment,
        "arch":        _ARCH_LABEL,
        "split":       args.split,
        "n_samples":   int(len(y_true)),
        "n_tp": int(n_tp), "n_tn": int(n_tn),
        "n_fp": int(n_fp), "n_fn": int(n_fn),
        "rollout_config": {
            "mode": "inline_streaming",
            "discard_ratio": float(embeddings_dict.get("rollout_discard_ratio", args.rollout_discard_ratio)),
            "attention_disk_write_gb": float(embeddings_dict.get("attention_disk_write_gb", 0.0)),
            "deprecated_full_attention_storage": False,
            "note": (
                "LSA-ConvTok: diagonal-masked attention; residual augmentation "
                "(0.5A+0.5I) applied per Abnar & Zuidema (2020). Full attention "
                "tensors are not stored; only final rollout relevance maps are retained."
        ),
        },
        "gini": {
            "tp": gini_tp.to_dict()    if gini_tp    else None,
            "tn": gini_tn.to_dict()    if gini_tn    else None,
            "fp": gini_fp.to_dict()    if gini_fp    else None,
            "fn": gini_fn.to_dict()    if gini_fn    else None,
        },
        "entropy": {
            "tp": entropy_tp.to_dict() if entropy_tp else None,
            "tn": entropy_tn.to_dict() if entropy_tn else None,
            "fp": entropy_fp.to_dict() if entropy_fp else None,
            "fn": entropy_fn.to_dict() if entropy_fn else None,
        },
    }

    stats_path = paths.analysis_root / f"rollout_stats_{args.split}.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {stats_path}")

    # Faithfulness-aware representative rollout maps are generated after
    # rollout-ranked insertion/deletion AUC has been computed. This keeps
    # population statistics independent while making qualitative exemplars
    # representative of both rollout morphology and behavioural faithfulness.

    gini_dict    = {n: gini_all[m]    for m, n in [(tp,"TP"),(tn,"TN"),(fp,"FP"),(fn,"FN")] if np.sum(m) > 0}
    entropy_dict = {n: entropy_all[m] for m, n in [(tp,"TP"),(tn,"TN"),(fp,"FP"),(fn,"FN")] if np.sum(m) > 0}

    if len(gini_dict) >= 2:
        plot_metric_distributions_by_group(
            metric_dict=gini_dict,
            metric_name="Rollout Gini Coefficient",
            save_path=str(paths.analysis_root / f"rollout_gini_distribution_{args.split}.png"),
            ylabel="Gini Coefficient",
            title=f"Rollout Attention Focality [{_ARCH_LABEL}] — {args.split.upper()}",
        )

    if len(entropy_dict) >= 2:
        plot_metric_distributions_by_group(
            metric_dict=entropy_dict,
            metric_name="Rollout Spatial Entropy",
            save_path=str(paths.analysis_root / f"rollout_entropy_distribution_{args.split}.png"),
            ylabel="Shannon Entropy (nats)",
            title=f"Rollout Attention Diffuseness [{_ARCH_LABEL}] — {args.split.upper()}",
        )

    # Store rollout on results for downstream use (review panels need it)
    results["_rollout"] = rollout   # not JSON-serialisable; stripped before saving
    return results


# =============================================================================
# Stage 4: Faithfulness population analysis
# =============================================================================

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/NaN values to JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _outcome_group_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return TP/TN/FP/FN labels for binary outcome-stratified reporting."""
    groups = np.full(len(y_true), "OTHER", dtype=object)
    groups[(y_true == 1) & (y_pred == 1)] = "TP"
    groups[(y_true == 0) & (y_pred == 0)] = "TN"
    groups[(y_true == 0) & (y_pred == 1)] = "FP"
    groups[(y_true == 1) & (y_pred == 0)] = "FN"
    return groups


def _auc_group_masks(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Boolean masks for binary TP/TN/FP/FN faithfulness summaries."""
    return {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
    }


def _summary_values(vals: np.ndarray) -> Dict[str, Any]:
    """Safe wrapper around SpatialMetricsAggregator for empty/NaN-heavy arrays."""
    vals = np.asarray(vals, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    return SpatialMetricsAggregator.compute_summary(vals).to_dict()


def _compute_auc_group_summaries(
    faithfulness_result: FaithfulnessResult,
    group_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute group-level summaries for raw AUC, endpoint-normalised AUC, and
    endpoint delta metrics.

    The raw AUC remains the primary logit-scale faithfulness metric. The
    endpoint-normalised AUC reports curve shape relative to each sample's own
    baseline/full-image endpoints. Endpoint delta reports the absolute logit
    gain (insertion) or logit drop (deletion).
    """
    summaries: Dict[str, Dict[str, Any]] = {}
    for curve_key, curve in faithfulness_result.curves.items():
        summaries[curve_key] = {}
        norm_auc = getattr(curve, "normalized_auc_scores", None)
        endpoint_delta = getattr(curve, "endpoint_delta_scores", None)
        for group_name, mask in group_masks.items():
            summaries[curve_key][group_name] = {
                "raw_logit_auc": _summary_values(np.asarray(curve.auc_scores[mask], dtype=np.float32)),
                "normalized_logit_auc": _summary_values(
                    np.asarray(norm_auc[mask], dtype=np.float32)
                    if norm_auc is not None else np.array([], dtype=np.float32)
                ),
                "endpoint_delta": _summary_values(
                    np.asarray(endpoint_delta[mask], dtype=np.float32)
                    if endpoint_delta is not None else np.array([], dtype=np.float32)
                ),
            }
    return summaries

def _compute_auc_welch_tests(
    faithfulness_result: FaithfulnessResult,
    group_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    """
    Run predefined Welch tests for raw AUC, normalised AUC, and endpoint delta.
    """
    planned_pairs = [("FP", "TN"), ("FN", "TP")]
    metric_getters = {
        "raw_logit_auc": lambda c: c.auc_scores,
        "normalized_logit_auc": lambda c: getattr(c, "normalized_auc_scores", None),
        "endpoint_delta": lambda c: getattr(c, "endpoint_delta_scores", None),
    }
    tests: Dict[str, Dict[str, Any]] = {}

    for curve_key, curve in faithfulness_result.curves.items():
        tests[curve_key] = {}
        for metric_name, getter in metric_getters.items():
            metric_values = getter(curve)
            if metric_values is None:
                continue
            metric_values = np.asarray(metric_values, dtype=np.float32)
            tests[curve_key][metric_name] = {}
            for group_a, group_b in planned_pairs:
                vals_a = metric_values[group_masks[group_a]]
                vals_b = metric_values[group_masks[group_b]]
                vals_a = vals_a[np.isfinite(vals_a)]
                vals_b = vals_b[np.isfinite(vals_b)]
                test = SpatialMetricsAggregator.compare_strata(vals_a, vals_b)
                test.update({
                    "metric": metric_name,
                    "contrast": f"{group_a}_vs_{group_b}",
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(vals_a.size),
                    "n_b": int(vals_b.size),
                    "mean_a": float(np.mean(vals_a)) if vals_a.size else float("nan"),
                    "mean_b": float(np.mean(vals_b)) if vals_b.size else float("nan"),
                    "mean_difference_a_minus_b": (
                        float(np.mean(vals_a) - np.mean(vals_b))
                        if vals_a.size and vals_b.size else float("nan")
                    ),
                })
                tests[curve_key][metric_name][f"{group_a}_vs_{group_b}"] = _json_safe(test)
    return tests

def _save_per_sample_auc_table(
    faithfulness_result: FaithfulnessResult,
    embeddings_dict: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: ProjectPaths,
    split: str,
) -> str:
    """Save one row per analysed sample with all faithfulness AUC values."""
    sample_indices = np.asarray(
        embeddings_dict.get("indices", np.arange(len(y_true))), dtype=np.int64
    )
    groups = _outcome_group_labels(y_true, y_pred)

    curve_keys = list(faithfulness_result.curves.keys())
    prefix = "faithfulness_rollout"
    csv_path = paths.analysis_root / f"{prefix}_per_sample_auc_{split}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_index", "sample_idx", "label", "prediction", "group", "ranking_source"] +
                       [f"auc_{k}" for k in curve_keys] +
                       [f"normalized_auc_{k}" for k in curve_keys] +
                       [f"endpoint_delta_{k}" for k in curve_keys],
        )
        writer.writeheader()
        for i in range(len(y_true)):
            row = {
                "row_index": int(i),
                "sample_idx": int(sample_indices[i]),
                "label": int(y_true[i]),
                "prediction": int(y_pred[i]),
                "group": str(groups[i]),
                "ranking_source": getattr(faithfulness_result, "ranking_source", "unknown"),
            }
            for key in curve_keys:
                curve = faithfulness_result.curves[key]
                row[f"auc_{key}"] = float(curve.auc_scores[i])
                norm = getattr(curve, "normalized_auc_scores", None)
                delta = getattr(curve, "endpoint_delta_scores", None)
                row[f"normalized_auc_{key}"] = (
                    float(norm[i]) if norm is not None and np.isfinite(norm[i]) else ""
                )
                row[f"endpoint_delta_{key}"] = (
                    float(delta[i]) if delta is not None and np.isfinite(delta[i]) else ""
                )
            writer.writerow(row)

    return str(csv_path)


def run_faithfulness_population_analysis(
    faithfulness_result: FaithfulnessResult,
    embeddings_dict: Dict[str, Any],
    paths: ProjectPaths,
    args: argparse.Namespace,
    ranking_source: str = "attention_rollout",
    target_mode: str = "predicted_class",
) -> Dict[str, Any]:
    """
    Faithfulness population analysis: insertion/deletion AUC by TP/TN/FP/FN.

    Primary AUC is logit-based.  In the final pipeline, patch order comes from
    attention rollout, so this is a behavioural validation of
    rollout relevance maps. Inter-operator correlation is printed.
    Returns serialisable dict (written to faithfulness_rollout_stats_{split}.json).
    """
    y_true = np.asarray(embeddings_dict["labels"],      dtype=int)
    y_pred = np.asarray(embeddings_dict["predictions"], dtype=int)

    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)

    group_masks = _auc_group_masks(y_true, y_pred)

    # Group-level summaries now include mean, std, median, quartiles, n, and
    # t-based 95% confidence intervals for the primary logit-AUC.
    auc_summary = _compute_auc_group_summaries(faithfulness_result, group_masks)

    # Predefined Welch tests for scientifically relevant outcome contrasts.
    auc_welch_tests = _compute_auc_welch_tests(faithfulness_result, group_masks)

    # Per-sample AUC table for reproducibility and downstream statistical audit.
    per_sample_auc_path = _save_per_sample_auc_table(
        faithfulness_result=faithfulness_result,
        embeddings_dict=embeddings_dict,
        y_true=y_true,
        y_pred=y_pred,
        paths=paths,
        split=args.split,
    )

    # Inter-operator AUC correlation
    interop_corr: Dict[str, float] = {}
    for mode in ("insertion", "deletion"):
        keys = [
            k for k in faithfulness_result.curves
            if k.endswith(f"_{mode}") and not (k.startswith("random_") or k.startswith("inverse_"))
        ]
        if len(keys) == 2:
            a0   = faithfulness_result.curves[keys[0]].auc_scores
            a1   = faithfulness_result.curves[keys[1]].auc_scores
            corr = float(np.corrcoef(a0, a1)[0, 1])
            interop_corr[f"{mode}_interoperator_r"] = corr
            agreement = "high" if abs(corr) > 0.8 else ("moderate" if abs(corr) > 0.5 else "low")
            print(f"  Inter-operator AUC correlation ({mode}): r = {corr:.3f}  ({agreement} agreement)")

    # Visualisation
    faith_viz = FaithfulnessVisualizer(output_dir=str(paths.analysis_root), dpi=300)
    for key, curve in faithfulness_result.curves.items():
        faith_viz.plot_faithfulness_curves(
            curves=curve,
            y_true=y_true,
            y_pred=y_pred,
            arch_name=_ARCH_LABEL,
            split=args.split,
            show=False,
        )

    faith_viz.plot_auc_comparison(
        faithfulness_result=faithfulness_result,
        y_true=y_true,
        y_pred=y_pred,
        arch_name=_ARCH_LABEL,
        split=args.split,
        show=False,
    )

    results = {
        "experiment": args.experiment,
        "arch":       _ARCH_LABEL,
        "split":      args.split,
        "n_samples":  int(len(y_true)),
        "n_tp": int(np.sum(tp)), "n_tn": int(np.sum(tn)),
        "n_fp": int(np.sum(fp)), "n_fn": int(np.sum(fn)),
        "faithfulness_config": {
            "ranking_source":     str(ranking_source),
            "target_mode":        str(target_mode),
            "perturbation_steps": args.faithfulness_steps,
            "operators":          list(args.faithfulness_operators),
            "modes":              list(args.faithfulness_modes),
            "primary_score":      "target_logit",
            "secondary_score":    "logit_margin",
            "tertiary_score":     "probability (sensitivity analysis)",
            "ranking_controls":   list(args.faithfulness_controls),
            "random_control_seed": int(args.faithfulness_random_seed),
            "additional_metrics": [
                "endpoint_normalized_logit_auc",
                "endpoint_delta_logit_gain_or_drop",
            ],
        },
        "auc_by_group":              auc_summary,
        "auc_welch_tests":           auc_welch_tests,
        "per_sample_auc_csv":        per_sample_auc_path,
        "interoperator_correlation": interop_corr,
    }

    out_prefix = "faithfulness_rollout"
    stats_path = paths.analysis_root / f"{out_prefix}_stats_{args.split}.json"
    with open(stats_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"\n  Saved: {stats_path}")

    return results




# =============================================================================
# Review-panel prototype/criticism selection
# =============================================================================

_REVIEW_GROUP_ORDER = ("TP", "TN", "FP", "FN")


def _rollout_gini_vector(rollout: np.ndarray) -> np.ndarray:
    """Per-sample rollout Gini coefficient for non-negative patch maps."""
    return np.asarray([
        AttentionDistributionalMetrics.compute_gini_coefficient(v)
        for v in np.asarray(rollout, dtype=np.float32)
    ], dtype=np.float32)


def _rollout_entropy_vector(rollout: np.ndarray) -> np.ndarray:
    """Per-sample Shannon entropy after normalising rollout to a distribution."""
    return np.asarray([
        AttentionDistributionalMetrics.compute_spatial_entropy(v)
        for v in np.asarray(rollout, dtype=np.float32)
    ], dtype=np.float32)


def _review_group_masks(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Binary TP/TN/FP/FN masks using class 1 as positive class."""
    return {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
    }


def _pca_coordinates(X: np.ndarray, n_components: int = 8) -> np.ndarray:
    """Deterministic PCA coordinates via SVD for review-selection features."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0 or n_components <= 0:
        return np.empty((X.shape[0] if X.ndim == 2 else 0, 0), dtype=np.float32)
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X = np.where(np.isfinite(X), X, col_mean[None, :])
    X = X - X.mean(axis=0, keepdims=True)
    max_components = max(0, min(int(n_components), X.shape[0] - 1, X.shape[1]))
    if max_components == 0:
        return np.empty((X.shape[0], 0), dtype=np.float32)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ vt[:max_components].T
    except np.linalg.LinAlgError:
        warnings.warn("PCA feature extraction failed; omitting embedding coordinates from MMD-critic selection.")
        return np.empty((X.shape[0], 0), dtype=np.float32)
    return coords.astype(np.float32)


def _primary_rollout_curve_items(faithfulness_result: Optional[FaithfulnessResult]):
    if faithfulness_result is None:
        return []
    items = []
    for key, curve in faithfulness_result.curves.items():
        variant = getattr(curve, "ranking_variant", "rollout")
        if variant not in {"rollout", "attention_rollout"}:
            continue
        arr = getattr(curve, "normalized_auc_scores", None)
        if arr is None:
            arr = getattr(curve, "auc_scores", None)
        if arr is None:
            continue
        items.append((key, getattr(curve, "mode", "unknown"), np.asarray(arr, dtype=np.float32)))
    return items


def _faithfulness_auc_features(
    faithfulness_result: Optional[FaithfulnessResult],
    n_samples: int,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Build the faithfulness-aware MMD-critic feature block.

    The production selector uses the same explicit, operator-resolved rollout
    faithfulness features for representative rollout maps and review panels:

        insertion_AUC_zero, deletion_AUC_zero,
        insertion_AUC_mean_patch, deletion_AUC_mean_patch.

    Primary rollout curves only are used. Random and inverse rankings are
    deliberately excluded because they are negative controls, not explanation
    maps. Endpoint-normalised logit AUC is preferred when available because it
    compares curve shape on a common [0,1] scale; raw logit AUC is retained as
    a fallback for backwards-compatible checkpoints. Missing curves are left as
    NaN and are imputed inside StratifiedMMDCriticSelector.
    """
    desired = [
        ("zero_insertion", "insertion_AUC_zero"),
        ("zero_deletion", "deletion_AUC_zero"),
        ("mean_patch_insertion", "insertion_AUC_mean_patch"),
        ("mean_patch_deletion", "deletion_AUC_mean_patch"),
    ]
    names = [name for _, name in desired]
    feature_matrix = np.full((n_samples, len(desired)), np.nan, dtype=np.float32)

    if faithfulness_result is None or not getattr(faithfulness_result, "curves", None):
        return (
            feature_matrix,
            names,
            {
                "available": False,
                "used_curve_keys": [],
                "missing_curve_keys": [key for key, _ in desired],
                "score_source": "endpoint_normalized_logit_auc_preferred_raw_logit_auc_fallback",
            },
        )

    used: List[str] = []
    missing: List[str] = []
    skipped: List[str] = []
    for j, (curve_key, _feature_name) in enumerate(desired):
        curve = faithfulness_result.curves.get(curve_key)
        if curve is None:
            missing.append(curve_key)
            continue

        variant = getattr(curve, "ranking_variant", "rollout")
        if variant not in {"rollout", "attention_rollout"}:
            skipped.append(curve_key)
            continue

        arr = getattr(curve, "normalized_auc_scores", None)
        if arr is None:
            arr = getattr(curve, "auc_scores", None)
        if arr is None:
            missing.append(curve_key)
            continue

        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] != n_samples:
            warnings.warn(
                f"Skipping faithfulness curve '{curve_key}' in MMD-critic selection: "
                f"length {arr.shape[0]} != n_samples {n_samples}."
            )
            skipped.append(curve_key)
            continue

        feature_matrix[:, j] = arr
        used.append(curve_key)

    return (
        feature_matrix.astype(np.float32),
        names,
        {
            "available": bool(used),
            "used_curve_keys": used,
            "missing_curve_keys": missing,
            "skipped_curve_keys": skipped,
            "score_source": "endpoint_normalized_logit_auc_preferred_raw_logit_auc_fallback",
        },
    )


def _build_mmd_critic_review_features(
    embeddings_dict: Dict[str, Any],
    rollout: np.ndarray,
    faithfulness_result: Optional[FaithfulnessResult],
    embedding_dims: int = 8,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Construct the MMD-critic feature representation.

    Feature groups:
    - rollout morphology: Gini and Shannon entropy;
    - model behaviour: predicted-class confidence;
    - faithfulness: operator-resolved rollout-ranked insertion/deletion AUC;
    - representation geometry: PCA coordinates of mean-pooled embeddings.
    """
    y_pred = np.asarray(embeddings_dict["predictions"], dtype=int)
    probs = np.asarray(embeddings_dict["probabilities"], dtype=np.float32)
    n = int(len(y_pred))
    confidence = probs[np.arange(n), y_pred].astype(np.float32)

    base = np.vstack([
        _rollout_gini_vector(rollout),
        _rollout_entropy_vector(rollout),
        confidence,
    ]).T.astype(np.float32)
    names = ["rollout_gini", "rollout_entropy", "prediction_confidence"]

    faith, faith_names, faith_meta = _faithfulness_auc_features(faithfulness_result, n)
    feature_blocks = [base, faith]
    names.extend(faith_names)

    embedding_meta: Dict[str, Any] = {"available": False, "source": "pooled_embeddings_pca", "n_components": 0}
    if "pooled_embeddings" in embeddings_dict and int(embedding_dims) > 0:
        coords = _pca_coordinates(np.asarray(embeddings_dict["pooled_embeddings"], dtype=np.float32), n_components=int(embedding_dims))
        if coords.shape[1] > 0:
            feature_blocks.append(coords)
            names.extend([f"pooled_embedding_pc{i+1}" for i in range(coords.shape[1])])
            embedding_meta = {"available": True, "source": "pooled_embeddings_pca", "n_components": int(coords.shape[1])}

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    meta = {
        "feature_names": names,
        "faithfulness": faith_meta,
        "embedding_coordinates": embedding_meta,
    }
    return features, names, meta




def run_rollout_representative_map_selection(
    embeddings_dict: Dict[str, Any],
    rollout: np.ndarray,
    slide_ids: np.ndarray,
    paths: ProjectPaths,
    args: argparse.Namespace,
    num_patches_side: int,
    faithfulness_result: Optional[FaithfulnessResult],
    n_per_group: int = 3,
    embedding_dims: int = 4,
) -> Dict[str, Any]:
    """
    Generate faithfulness-aware representative rollout maps.

    This is intentionally run after Stage 4 when faithfulness is available.
    Selection uses the same scientific feature family as the review panels:
    rollout morphology, predicted-class confidence, operator-resolved
    rollout-ranked insertion/deletion AUC, and pooled-embedding PCA
    coordinates.  MMD-critic is applied independently within TP/TN/FP/FN
    strata so each outcome group receives representative prototypes and
    underrepresented criticisms.
    """
    print(f"\n{'='*70}")
    print("FAITHFULNESS-AWARE STRATIFIED MMD-CRITIC ROLLOUT REPRESENTATIVES")
    print(f"{'='*70}")

    rollout = np.asarray(rollout, dtype=np.float32)
    y_true = np.asarray(embeddings_dict["labels"], dtype=int)
    y_pred = np.asarray(embeddings_dict["predictions"], dtype=int)
    if rollout.shape[0] != len(y_true):
        raise ValueError(
            f"rollout has {rollout.shape[0]} samples but labels have {len(y_true)}."
        )

    masks = {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
    }

    strata: List[StrataDefinition] = []
    quotas_rep: Dict[str, int] = {}
    for group_name in _REVIEW_GROUP_ORDER:
        n_mask = int(np.sum(masks[group_name]))
        if n_mask <= 0:
            continue
        q = min(int(n_per_group), n_mask)
        quotas_rep[group_name] = q
        strata.append(StrataDefinition(
            stratum_id=group_name,
            mask=masks[group_name],
            n_samples=n_mask,
            target_samples=q,
        ))

    if not strata:
        warnings.warn("No non-empty TP/TN/FP/FN strata available for rollout representative selection.")
        return {"available": False, "reason": "no_nonempty_strata"}

    features, feature_names, feature_meta = _build_mmd_critic_review_features(
        embeddings_dict=embeddings_dict,
        rollout=rollout,
        faithfulness_result=faithfulness_result,
        embedding_dims=int(embedding_dims),
    )

    if not feature_meta.get("faithfulness", {}).get("available", False):
        warnings.warn(
            "Faithfulness AUC features are unavailable for rollout representative "
            "selection. The selector will impute missing AUC columns; run with "
            "--compute-faithfulness for the fully rigorous feature set."
        )

    selector = StratifiedMMDCriticSelector(random_seed=int(getattr(args, "faithfulness_random_seed", 42)))
    selection = selector.select_within_strata(
        strata=strata,
        features=features,
        quotas=quotas_rep,
        slide_ids=slide_ids,
        prototype_fraction=0.67,
        slide_aware=True,
    )

    representatives: Dict[str, List[int]] = {g: [] for g in _REVIEW_GROUP_ORDER}
    for idx, group in zip(selection.sample_indices, selection.stratum_ids):
        representatives[str(group)].append(int(idx))

    summary = {
        "available": True,
        "strategy": "stratified_mmd_critic_faithfulness_aware",
        "scientific_rationale": (
            "Representative rollout maps are selected in TP/TN/FP/FN strata using "
            "MMD-critic over rollout morphology, model confidence, rollout-ranked "
            "faithfulness AUC, and pooled-embedding PCA coordinates. Prototypes "
            "summarise each stratum distribution; criticisms expose samples poorly "
            "represented by the prototypes."
        ),
        "feature_names": feature_names,
        "feature_metadata": feature_meta,
        "quotas": quotas_rep,
        "selected_by_group": representatives,
        "selected": selection.metadata,
        "prototype_fraction": 0.67,
        "slide_aware_diversity": True,
        "n_per_group": int(n_per_group),
    }

    summary_path = paths.analysis_root / f"rollout_representative_selection_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    print(f"  Saved faithfulness-aware rollout representative selection: {summary_path}")

    viz = SpatialAttentionVisualizer(output_dir=str(paths.analysis_root), dpi=300)
    for group_name, grp_indices in representatives.items():
        for rank, orig_idx in enumerate(grp_indices):
            viz.plot_rollout_relevance_heatmap(
                rollout_relevance=rollout[orig_idx],
                num_patches_side=num_patches_side,
                title=(
                    f"Attention-flow Rollout | {_ARCH_LABEL} | "
                    f"{group_name} faithfulness-aware MMD-Critic sample {rank+1} "
                    f"(sample {orig_idx})"
                ),
                save_name=f"mmdcritic_{group_name}_{orig_idx:04d}_rollout_{args.split}.png",
                show=False,
            )

    panel_rollout: Dict[str, np.ndarray] = {}
    panel_indices: Dict[str, int] = {}
    for group_name in _REVIEW_GROUP_ORDER:
        if representatives.get(group_name):
            exemplar = int(representatives[group_name][0])
            panel_rollout[group_name] = rollout[exemplar]
            panel_indices[group_name] = exemplar

    comparison_path: Optional[str] = None
    if len(panel_rollout) >= 2:
        comparison_path = viz.plot_rollout_comparison_panel(
            rollout_dict=panel_rollout,
            num_patches_side=num_patches_side,
            sample_indices=panel_indices,
            title=f"Faithfulness-aware MMD-Critic Rollout Comparison [{_ARCH_LABEL}] — {args.split.upper()}",
            save_name=f"rollout_mmdcritic_comparison_{args.split}.png",
            show=False,
        )

    summary["selection_json"] = str(summary_path)
    summary["comparison_panel"] = comparison_path
    return _json_safe(summary)


def select_review_panel_indices_mmd_critic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    rollout: np.ndarray,
    slide_ids: Optional[np.ndarray],
    quotas: Dict[str, int],
    features: np.ndarray,
    feature_names: List[str],
    slide_aware: bool = True,
    prototype_fraction: float = 0.6,
    random_seed: int = 42,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stratified MMD-critic selection for rollout review panels.

    Within each TP/TN/FP/FN stratum, prototypes are greedily selected to
    minimise empirical MMD between the selected set and the stratum feature
    distribution. Criticisms are then selected by high MMD witness values,
    identifying samples underrepresented by the prototypes. Slide diversity is
    a soft constraint: candidates from unused slides are preferred where this
    does not prevent filling the requested quota.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    rollout = np.asarray(rollout, dtype=np.float32)
    slide_ids_arr = None if slide_ids is None else np.asarray(slide_ids, dtype=object)

    masks = _review_group_masks(y_true, y_pred)
    strata: List[StrataDefinition] = []
    for group in _REVIEW_GROUP_ORDER:
        n_avail = int(np.sum(masks[group]))
        quota = int(quotas.get(group, 0))
        if n_avail > 0 and quota > 0:
            strata.append(StrataDefinition(
                stratum_id=group,
                mask=masks[group],
                n_samples=n_avail,
                target_samples=min(quota, n_avail),
            ))

    selector = StratifiedMMDCriticSelector(random_seed=random_seed)
    selection = selector.select_within_strata(
        strata=strata,
        features=features,
        quotas=quotas,
        slide_ids=slide_ids_arr,
        prototype_fraction=prototype_fraction,
        slide_aware=slide_aware,
    )

    confidence = probabilities[np.arange(len(y_pred)), y_pred].astype(np.float32)
    gini = _rollout_gini_vector(rollout)
    entropy = _rollout_entropy_vector(rollout)
    rollout_l1 = rollout.sum(axis=1).astype(np.float32)
    rollout_l2 = np.linalg.norm(rollout, axis=1).astype(np.float32)

    feature_lookup = {name: features[:, j] for j, name in enumerate(feature_names)}
    final_meta: List[Dict[str, Any]] = []
    for raw_m in selection.metadata:
        i = int(raw_m["row_index"])
        m = dict(raw_m)
        m.update({
            "rollout_gini": float(gini[i]),
            "rollout_entropy": float(entropy[i]),
            "rollout_l1": float(rollout_l1[i]),
            "rollout_l2": float(rollout_l2[i]),
            "prediction_confidence": float(confidence[i]),
            "slide_id": str(slide_ids_arr[i]) if slide_ids_arr is not None else "",
        })
        for fname, fcol in feature_lookup.items():
            val = fcol[i]
            m[f"feature_{fname}"] = float(val) if np.isfinite(val) else float("nan")
        final_meta.append(m)

    summary: Dict[str, Any] = {
        "strategy": "stratified_mmd_critic",
        "references": [
            "Kim, Khanna & Koyejo (2016), Examples are not Enough, Learn to Criticize!",
            "Ribeiro, Singh & Guestrin (2016), Why Should I Trust You?",
        ],
        "slide_aware": bool(slide_aware),
        "prototype_fraction": float(prototype_fraction),
        "feature_names": list(feature_names),
        "quotas": {k: int(v) for k, v in quotas.items()},
        "groups": {},
    }
    for group in _REVIEW_GROUP_ORDER:
        group_meta = [m for m in final_meta if m.get("group") == group]
        group_mask = masks[group]
        summary["groups"][group] = {
            "n_available": int(np.sum(group_mask)),
            "quota_requested": int(quotas.get(group, 0)),
            "n_selected": int(len(group_meta)),
            "n_prototypes": int(sum(m.get("selection_role") == "prototype" for m in group_meta)),
            "n_criticisms": int(sum(m.get("selection_role") == "criticism" for m in group_meta)),
            "n_complete_small_stratum": int(sum(m.get("selection_role") == "complete_small_stratum" for m in group_meta)),
            "n_unique_slides_selected": int(len({m.get("slide_id", "") for m in group_meta if m.get("slide_id", "") != ""})),
            "selection_roles": [m.get("selection_role", "") for m in group_meta],
            "selection_reasons": [m.get("selection_reason", "") for m in group_meta],
        }

    return selection.sample_indices, final_meta, summary
# =============================================================================
# Stage 6: Plausibility review panel generation
# =============================================================================

def run_review_panel_generation(
    rollout: np.ndarray,
    embeddings_dict: Dict[str, Any],
    dataloader: torch.utils.data.DataLoader,
    paths: ProjectPaths,
    args: argparse.Namespace,
    num_patches_side: int,
    class_names: List[str],
    n_per_group: int = 5,
    faithfulness_result: Optional[FaithfulnessResult] = None,
) -> str:
    """
    Generate rollout-only structured review panels for plausibility assessment.

    Final selection strategy
    ------------------------
    Panels are selected by stratified MMD-critic. Within each TP/TN/FP/FN
    stratum, prototypes minimise empirical MMD in a feature space containing
    rollout morphology, prediction confidence, rollout-ranked faithfulness AUC,
    and pooled-embedding PCA coordinates. Criticisms are high-witness samples
    that are poorly represented by the prototypes; review examples are no
    longer selected by hand-written centrality or total-mass rules.
    """
    print(f"\n{'='*70}")
    print(f"ROLLOUT REVIEW PANEL GENERATION  [{_ARCH_LABEL}]")
    print(f"{'='*70}")

    y_true = np.asarray(embeddings_dict["labels"],      dtype=int)
    y_pred = np.asarray(embeddings_dict["predictions"], dtype=int)
    probs  = np.asarray(embeddings_dict["probabilities"], dtype=np.float32)
    sample_indices = np.asarray(embeddings_dict["indices"], dtype=np.int64)
    slide_ids = np.asarray(embeddings_dict.get("slide_ids", np.array([""] * len(y_true))), dtype=object)
    rollout = np.asarray(rollout, dtype=np.float32)

    if rollout.ndim != 2 or rollout.shape[0] != len(y_true):
        raise ValueError(
            f"rollout must have shape (N, P) with N={len(y_true)}, got {rollout.shape}"
        )

    masks = _review_group_masks(y_true, y_pred)
    group_counts = {g: int(np.sum(masks[g])) for g in _REVIEW_GROUP_ORDER}

    quotas = {
        "TP": int(getattr(args, "review_n_tp", 9)),
        "TN": int(getattr(args, "review_n_tn", 9)),
        "FP": int(getattr(args, "review_n_fp", 6)),
        "FN": int(getattr(args, "review_n_fn", -1)),
    }
    # ``-1`` means include all available false negatives. For other strata,
    # fall back to the historical common quota only if the user explicitly did
    # not provide a group-specific non-negative quota.
    for group in _REVIEW_GROUP_ORDER:
        if quotas[group] < 0:
            quotas[group] = group_counts[group] if group == "FN" else int(n_per_group)

    prototype_fraction = float(getattr(args, "review_mmd_prototype_fraction", 0.6))
    if not (0.0 < prototype_fraction <= 1.0):
        raise ValueError("--review-mmd-prototype-fraction must be in (0, 1].")

    feature_matrix, feature_names, feature_meta = _build_mmd_critic_review_features(
        embeddings_dict=embeddings_dict,
        rollout=rollout,
        faithfulness_result=faithfulness_result,
        embedding_dims=int(getattr(args, "review_embedding_dims", 8)),
    )

    print("  Review selection strategy: stratified_mmd_critic")
    print("  Outcome counts:", ", ".join(f"{g}={group_counts[g]}" for g in _REVIEW_GROUP_ORDER))
    print("  Requested quotas:", ", ".join(f"{g}={quotas[g]}" for g in _REVIEW_GROUP_ORDER))
    print("  MMD-critic features:", ", ".join(feature_names))
    if not feature_meta["faithfulness"].get("available", False):
        warnings.warn(
            "Faithfulness AUC features are unavailable for review selection. "
            "The selector will still run, but the final thesis configuration "
            "should compute faithfulness before review panels."
        )

    selected_indices, selection_metadata, selection_summary = select_review_panel_indices_mmd_critic(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probs,
        rollout=rollout,
        slide_ids=slide_ids if len(slide_ids) == len(y_true) else None,
        quotas=quotas,
        features=feature_matrix,
        feature_names=feature_names,
        slide_aware=bool(getattr(args, "review_slide_aware", True)),
        prototype_fraction=prototype_fraction,
        random_seed=int(getattr(args, "review_random_seed", 42)),
    )
    selection_summary["feature_construction"] = feature_meta

    if selected_indices.size == 0:
        warnings.warn("No samples selected for review panel generation.")
        return ""

    print(f"  Selected panels: {len(selected_indices)}")
    for g in _REVIEW_GROUP_ORDER:
        selected_g = [m for m in selection_metadata if m.get("group") == g]
        if selected_g:
            print(f"    {g}: {len(selected_g)} selected | reasons: " + ", ".join(m["selection_reason"] for m in selected_g))

    raw_images_all: List[np.ndarray] = []
    max_collect = getattr(args, "review_max_batches", None)
    for bi, (X, _) in enumerate(dataloader):
        if max_collect is not None and bi >= max_collect:
            break
        raw_images_all.append(X.cpu().numpy().astype(np.float32))

    if not raw_images_all:
        warnings.warn("DataLoader yielded no batches for review panel collection.")
        return ""

    raw_images_full = np.concatenate(raw_images_all, axis=0)[:len(y_true)]

    sel      = selected_indices
    raw_sel  = raw_images_full[sel]
    ro_sel   = rollout[sel]
    lab_sel  = y_true[sel]
    pred_sel = y_pred[sel]
    prob_sel = probs[sel]
    sidx_sel = sample_indices[sel]
    slide_sel = slide_ids[sel] if len(slide_ids) == len(y_true) else np.array([""] * len(sel), dtype=object)

    # Persist a machine-readable selection summary before rendering panels.
    selection_summary_path = paths.analysis_root / f"review_panel_selection_{args.split}.json"
    with open(selection_summary_path, "w") as f:
        json.dump(_json_safe(selection_summary), f, indent=2)
    print(f"  Saved review selection summary: {selection_summary_path}")

    base_dir = paths.analysis_root / "review_rollout_panels" / args.split
    within_dir = base_dir / "within_group_scale"
    global_dir = base_dir / "global_scale"
    per_sample_dir = base_dir / "per_sample_scale"
    for d in (within_dir, global_dir, per_sample_dir):
        d.mkdir(parents=True, exist_ok=True)

    reviewer_within = RolloutReviewPanel(
        output_dir=str(within_dir),
        dpi=200,
        class_names=class_names,
    )
    print(f"  Generating {len(sel)} rollout-only panels — within-group shared scale ...")
    reviewer_within.generate_panels(
        raw_images=raw_sel,
        rollout_maps=ro_sel,
        labels=lab_sel,
        predictions=pred_sel,
        probabilities=prob_sel,
        sample_indices=sidx_sel,
        num_patches_side=num_patches_side,
        slide_ids=slide_sel,
        selection_metadata=selection_metadata,
        split=args.split,
        scale_mode="within_group",
    )

    reviewer_global = RolloutReviewPanel(
        output_dir=str(global_dir),
        dpi=200,
        class_names=class_names,
    )
    print(f"  Generating {len(sel)} rollout-only panels — true global scale ...")
    reviewer_global.generate_panels(
        raw_images=raw_sel,
        rollout_maps=ro_sel,
        labels=lab_sel,
        predictions=pred_sel,
        probabilities=prob_sel,
        sample_indices=sidx_sel,
        num_patches_side=num_patches_side,
        slide_ids=slide_sel,
        selection_metadata=selection_metadata,
        split=args.split,
        scale_mode="global",
    )

    reviewer_per_sample = RolloutReviewPanel(
        output_dir=str(per_sample_dir),
        dpi=200,
        class_names=class_names,
    )
    print(f"  Generating {len(sel)} rollout-only panels — per-sample autoscale ...")
    reviewer_per_sample.generate_panels(
        raw_images=raw_sel,
        rollout_maps=ro_sel,
        labels=lab_sel,
        predictions=pred_sel,
        probabilities=prob_sel,
        sample_indices=sidx_sel,
        num_patches_side=num_patches_side,
        slide_ids=slide_sel,
        selection_metadata=selection_metadata,
        split=args.split,
        scale_mode="per_sample",
    )

    csv_path = reviewer_within.save_provenance_csv(
        csv_name=f"rollout_review_provenance_{args.split}_within_group_scale.csv"
    )
    print(f"  Review CSV (within-group scale): {csv_path}")

    csv_path_global = reviewer_global.save_provenance_csv(
        csv_name=f"rollout_review_provenance_{args.split}_global_scale.csv"
    )
    print(f"  Review CSV (global scale):       {csv_path_global}")

    csv_path_per_sample = reviewer_per_sample.save_provenance_csv(
        csv_name=f"rollout_review_provenance_{args.split}_per_sample_scale.csv"
    )
    print(f"  Review CSV (per-sample scale):   {csv_path_per_sample}")
    return csv_path


# =============================================================================
# Stage 7: Consolidated XAI summary export
# =============================================================================

def save_consolidated_xai_summary(
    paths: ProjectPaths,
    args: argparse.Namespace,
    rollout_results:      Optional[Dict[str, Any]],
    faithfulness_results: Optional[Dict[str, Any]],
    umap_metrics:         Optional[Dict[str, Any]],
    review_csv_path:      Optional[str],
) -> str:
    """
    Write one consolidated JSON summarising all XAI stages run in this session.

    Includes UMAP outputs, rollout metrics, rollout-ranked faithfulness summaries,
    and rollout-only review-panel provenance in one split-level report.

    Parameters
    ----------
    paths, args          : Pipeline context.
    rollout_results      : Output of run_rollout_population_analysis() (or None).
    faithfulness_results : Output of run_faithfulness_population_analysis() (or None).
    umap_metrics         : UMAP quality metrics dict (or None).
    review_csv_path      : Path to review provenance CSV (or None).

    Returns
    -------
    str — path to the saved summary JSON.
    """
    # Strip non-serialisable ndarray fields before saving
    def _strip_arrays(d: Optional[Dict]) -> Optional[Dict]:
        if d is None:
            return None
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    summary = {
        "experiment": args.experiment,
        "arch":       _ARCH_LABEL,
        "split":      args.split,
        "timestamp":  datetime.now().isoformat(),
        "stages_run": {
            "umap":          umap_metrics       is not None,
            "rollout":       rollout_results     is not None,
            "faithfulness":  faithfulness_results is not None,
            "review_panels": review_csv_path      is not None,
        },
        "methodology": {
            "primary_explanation": "attention_rollout",
            "faithfulness_ranking_source": "attention_rollout",
            "faithfulness_score": "target_class_logit",
            "references": [
                "Abnar & Zuidema (2020) attention rollout/flow",
                "Samek et al. (2017) perturbation-based explanation evaluation",
                "Petsiuk et al. (2018) insertion/deletion AUC / RISE",
            ],
        },
        "umap":          umap_metrics,
        "rollout":       _strip_arrays(rollout_results),
        "faithfulness":  faithfulness_results,
        "review_panels": {
            "provenance_csv":        review_csv_path,
            "image_modality_note": (
                "Rollout-only panels show model-input display-normalised images. "
                "They represent the normalised input space used during inference "
                "(checkpoint mean/std normalisation), then min-max normalised per "
                "sample to [0,1] for display. They are NOT acquisition-faithful "
                "raw intensity images."
        ),
        },
    }

    out_path = paths.analysis_root / f"xai_summary_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Consolidated XAI summary saved: {out_path}")
    return str(out_path)


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LSA-ConvTok XAI pipeline — embedding, rollout, rollout-faithfulness, review panels"
    )

    p.add_argument("--project-root",       type=str, default=".")
    p.add_argument("--preprocessed-root",  type=str, default="/content/data/preprocessed")
    p.add_argument("--dataset-subdir",     type=str, default="fungi")
    p.add_argument("--output-root",        type=str,
                   default="/content/drive/MyDrive/colab/analysis/output")
    p.add_argument("--split-file",         type=str,
                   default="/content/drive/MyDrive/colab/analysis/preprocessed/split_indices.json")

    p.add_argument("--experiment",   type=str, default="lsa_convtok")
    p.add_argument("--split",        type=str, default="test", choices=["train", "test"])
    p.add_argument("--batch-size",   type=int, default=8)
    p.add_argument("--force-recompute", action="store_true")

    p.add_argument("--extract-attention",    action="store_true")
    p.add_argument("--attention-layers",     type=int, nargs="+", default=None)

    p.add_argument("--run-embedding-analysis", action="store_true")

    p.add_argument("--compute-attention-rollout", action="store_true")
    p.add_argument("--rollout-inline", default=True, action=argparse.BooleanOptionalAction,
                   help="Use inline/streaming rollout extraction. Default: true.")
    p.add_argument("--max-attention-disk-gb", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--rollout-discard-ratio",     type=float, default=0.9)
    p.add_argument("--save-last-layer-attn",      action="store_true",
                   help="Deprecated: ignored because full attention tensors are not stored.")

    p.add_argument("--resume-xai", action="store_true",
                   help="Resume faithfulness from per-operator XAI checkpoints when available.")
    p.add_argument("--xai-checkpoint-dir", type=str, default=None,
                   help="Directory for resumable faithfulness checkpoint files. Default: analysis/xai_checkpoints.")

    p.add_argument("--compute-faithfulness",       action="store_true")
    p.add_argument("--faithfulness-steps",         type=int, default=20)
    p.add_argument("--faithfulness-operators",     type=str, nargs="+",
                   default=["zero", "mean_patch"])
    p.add_argument("--faithfulness-modes",         type=str, nargs="+",
                   default=["insertion", "deletion"])
    p.add_argument("--faithfulness-target", type=str, default="predicted",
                   choices=["predicted", "true"],
                   help="Class logit scored in faithfulness curves. Default explains the model decision.")
    p.add_argument("--faithfulness-ranking-source", type=str, default="rollout",
                   choices=["rollout", "attention_rollout"],
                   help=("Patch-ranking map for faithfulness. Only rollout is supported "
                         "in the production pipeline; 'attention_rollout' is accepted "
                         "as a provenance alias."))
    p.add_argument("--faithfulness-controls", type=str, nargs="*",
                   default=["random", "inverse"], choices=["random", "inverse"],
                   help=("Negative-control rankings to compute alongside rollout. "
                         "Use an empty value to disable controls."))
    p.add_argument("--faithfulness-random-seed", type=int, default=42,
                   help="Seed for reproducible random-ranking faithfulness control.")

    p.add_argument("--generate-review-panels",  action="store_true")
    p.add_argument("--review-n-per-group",       type=int, default=5,
                   help="Fallback quota for strata without an explicit non-negative group quota.")
    p.add_argument("--review-n-tp",              type=int, default=9,
                   help="Number of TP review panels selected by stratified MMD-critic.")
    p.add_argument("--review-n-tn",              type=int, default=9,
                   help="Number of TN review panels selected by stratified MMD-critic.")
    p.add_argument("--review-n-fp",              type=int, default=6,
                   help="Number of FP review panels selected by stratified MMD-critic.")
    p.add_argument("--review-n-fn",              type=int, default=-1,
                   help="Number of FN panels; -1 includes all available false negatives.")
    p.add_argument("--review-mmd-prototype-fraction", type=float, default=0.6,
                   help="Fraction of each stratum quota allocated to MMD prototypes; the rest are criticisms.")
    p.add_argument("--review-embedding-dims", type=int, default=8,
                   help="Number of pooled-embedding PCA coordinates included in the MMD-critic feature space.")
    p.add_argument("--review-random-seed", type=int, default=42,
                   help="Seed recorded for deterministic review selection provenance.")
    p.add_argument("--review-slide-aware",       action=argparse.BooleanOptionalAction, default=True,
                   help="Prefer one patch per slide within each review stratum when alternatives exist.")
    p.add_argument("--review-max-batches",       type=int, default=None)

    return p


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args  = build_arg_parser().parse_args()
    if getattr(args, "max_attention_disk_gb", None) not in (None, 0, 0.0):
        warnings.warn(
            "--max-attention-disk-gb is deprecated and ignored: inline rollout "
            "does not write full attention tensors to disk."
        )
    paths = ProjectPaths.from_args(args)
    if args.xai_checkpoint_dir is None:
        args.xai_checkpoint_dir = str(paths.analysis_root / "xai_checkpoints")

    cli_config = get_config(args.experiment)
    device     = get_device_info()

    dataset     = NPYImageFolder(str(paths.preprocessed_dir))
    class_names = getattr(dataset, "classes", ["class0", "class1"])
    print(f"Dataset: {len(dataset)} samples | classes={class_names}")

    indices = load_split_with_conversion(args.split_file, dataset, args.split)

    model_path = paths.output_root / args.experiment / f"final_model_{args.experiment}.pth"
    model, checkpoint, ckpt_config = load_model_and_checkpoint(
        str(model_path), cli_config, device
    )
    config = ckpt_config

    loader, dataset_indices = create_dataloader_with_provenance(
        dataset=dataset,
        indices=indices,
        checkpoint=checkpoint,
        config=config,
        batch_size=args.batch_size,
    )

    # =========================================================================
    # STAGE 1: Extraction / provenance
    # =========================================================================
    embeddings_path = paths.analysis_root / f"embeddings_{args.split}_{args.experiment}.npz"

    need_inline_rollout = bool(args.compute_attention_rollout or args.extract_attention)
    embeddings_dict: Dict[str, Any]
    recompute_embeddings = bool(args.force_recompute)
    if embeddings_path.exists() and not recompute_embeddings:
        embeddings_dict = SimpleViTEmbeddingExtractor.load_embeddings(
            str(embeddings_path), verbose=True
        )
        if need_inline_rollout and "rollout" not in embeddings_dict:
            warnings.warn(
                "Existing embeddings file does not contain inline rollout maps; "
                "recomputing extraction with inline rollout enabled."
        )
            recompute_embeddings = True
    else:
        embeddings_dict = {}
        recompute_embeddings = True

    if recompute_embeddings:
        extractor = SimpleViTEmbeddingExtractor(
            model=model, device=device, verbose=True
        )
        embeddings_dict = extractor.extract_embeddings(
            dataloader=loader,
            dataset_indices=dataset_indices,
            extract_patch_tokens=False,
            extract_transformer_tokens=False,
            extract_attention=need_inline_rollout,
            attention_layers=args.attention_layers,
            save_spatial_attention=False,
            enforce_forward_equivalence=True,
            rollout_discard_ratio=float(args.rollout_discard_ratio),
            max_attention_disk_gb=args.max_attention_disk_gb,
        )
        extractor.save_embeddings(embeddings_dict, str(embeddings_path), compress=False)

    y_true = np.asarray(embeddings_dict.get("labels"), dtype=int)
    y_pred = np.asarray(embeddings_dict.get("predictions"), dtype=int)
    embeddings_dict["labels"]      = y_true
    embeddings_dict["predictions"] = y_pred

    if "mean" not in embeddings_dict:
        embeddings_dict["mean"] = checkpoint.get("mean")
    if "std" not in embeddings_dict:
        embeddings_dict["std"] = checkpoint.get("std")

    # Resolve token grid ONCE — all downstream stages use these values.
    token_grid = embeddings_dict.get("token_grid", None)
    if token_grid is not None:
        h_prime, w_prime = int(token_grid[0]), int(token_grid[1])
    else:
        # Fallback via convtok_expected_hw (LSA-ConvTok always has this)
        hw = config.get("convtok_expected_hw", None)
        if hw is not None:
            h_prime, w_prime = int(hw[0]), int(hw[1])
        else:
            raise RuntimeError(
                "token_grid not found in embeddings_dict and "
                "convtok_expected_hw not in config.  "
                "Re-run extraction with --force-recompute."
        )
        warnings.warn(
            "token_grid not found in embeddings_dict; "
            f"using convtok_expected_hw = {h_prime}×{w_prime}."
        )

    num_patches_side = h_prime
    if h_prime != w_prime:
        warnings.warn(
            f"Non-square token grid ({h_prime}×{w_prime}); "
            "visualizations assume square grids."
        )

    slide_ids, provenance_map = reconstruct_slide_id_provenance(
        dataset, embeddings_dict["indices"], validate=True
    )
    embeddings_dict["slide_ids"] = slide_ids

    # Collect stage outputs for consolidated summary
    umap_metrics:         Optional[Dict[str, Any]] = None
    rollout_results:      Optional[Dict[str, Any]] = None
    faithfulness_result:  Optional[FaithfulnessResult] = None
    faithfulness_results: Optional[Dict[str, Any]] = None
    review_csv_path:      Optional[str] = None

    # =========================================================================
    # STAGE 2: Embedding analysis (UMAP)
    # =========================================================================
    if args.run_embedding_analysis:
        analyzer  = EmbeddingAnalyzer(
            output_dir=str(paths.analysis_root), dpi=300, random_seed=42
        )
        slide_viz = SlideConditionedVisualizer(
            output_dir=str(paths.analysis_root), dpi=300
        )
        pooled = np.asarray(embeddings_dict["pooled_embeddings"], dtype=np.float32)

        umap_proj = analyzer.compute_umap_projection(pooled, n_neighbors=15, min_dist=0.1)
        umap_metrics = analyzer.analyze_embedding_quality(
            embeddings=pooled,
            projection=umap_proj,
            labels=y_true,
            predictions=y_pred,
            slide_ids=slide_ids,
            method_name="UMAP",
        )
        umap_metrics["arch"] = _ARCH_LABEL

        umap_metrics_path = paths.analysis_root / f"umap_quality_metrics_{args.split}.json"
        with open(umap_metrics_path, "w") as f:
            json.dump(umap_metrics, f, indent=2)
        print(f"Saved UMAP metrics: {umap_metrics_path}")

        analyzer.plot_embedding_projection(
            projection=umap_proj,
            labels=y_true,
            predictions=y_pred,
            slide_ids=slide_ids,
            class_names=class_names,
            method_name="UMAP",
            save_name=f"umap_comprehensive_{args.split}.png",
            plot_errors_only=False,
            show=False,
        )
        slide_viz.plot_slide_conditioned_errors(
            projection=umap_proj,
            slide_ids=slide_ids,
            labels=y_true,
            predictions=y_pred,
            class_names=class_names,
            method_name="UMAP",
            save_name=f"umap_slide_conditioned_{args.split}.png",
            show=False,
        )

    # =========================================================================
    # STAGE 3: Rollout
    # =========================================================================
    if args.compute_attention_rollout:
        rollout_results = run_rollout_population_analysis(
            embeddings_dict=embeddings_dict,
            slide_ids=slide_ids,
            model=model,
            dataloader=loader,
            paths=paths,
            args=args,
            config=config,
            device=device,
            num_patches_side=num_patches_side,
        )

    # =========================================================================
    # STAGE 4: Rollout faithfulness
    # =========================================================================
    if args.compute_faithfulness:
        print("\n" + "=" * 70)
        print(f"ROLLOUT-BASED FAITHFULNESS ANALYSIS  [{_ARCH_LABEL}]")
        print("=" * 70)

        extractor_f = SimpleViTEmbeddingExtractor(
            model=model, device=device, verbose=True
        )

        rollout_for_faithfulness: Optional[np.ndarray] = None
        if rollout_results is not None:
            rollout_for_faithfulness = rollout_results.get("_rollout")
        if rollout_for_faithfulness is None and "rollout" in embeddings_dict:
            rollout_for_faithfulness = np.asarray(embeddings_dict["rollout"], dtype=np.float32)
        if rollout_for_faithfulness is None:
            raise RuntimeError(
                "--compute-faithfulness requires attention rollout maps. "
                "Run --compute-attention-rollout or load embeddings containing a 'rollout' array."
            )

        if args.faithfulness_target == "predicted":
            target_indices = np.asarray(embeddings_dict["predictions"], dtype=np.int64)
            target_mode = "predicted_class"
        else:
            target_indices = np.asarray(embeddings_dict["labels"], dtype=np.int64)
            target_mode = "true_class"

        # The production pipeline supports rollout-ranked faithfulness only.
        # The CLI accepts both names to keep the notebook-readable term
        # ("rollout") while storing explicit provenance ("attention_rollout").
        ranking_source = "attention_rollout"

        faithfulness_result = extractor_f.compute_faithfulness_curves_from_maps(
            dataloader=loader,
            attribution_maps=rollout_for_faithfulness,
            target_indices=target_indices,
            perturbation_steps=args.faithfulness_steps,
            operators=args.faithfulness_operators,
            modes=args.faithfulness_modes,
            ranking_source=ranking_source,
            target_mode=target_mode,
            ranking_controls=args.faithfulness_controls,
            random_seed=args.faithfulness_random_seed,
            resume=bool(args.resume_xai),
            checkpoint_dir=args.xai_checkpoint_dir,
            checkpoint_name=f"faithfulness_rollout_{args.split}_{args.experiment}",
        )
        faithfulness_results = run_faithfulness_population_analysis(
            faithfulness_result=faithfulness_result,
            embeddings_dict=embeddings_dict,
            paths=paths,
            args=args,
            ranking_source=ranking_source,
            target_mode=target_mode,
        )

    # =========================================================================
    # STAGE 4B: Faithfulness-aware representative rollout maps
    # =========================================================================
    if rollout_results is not None:
        rollout_for_representatives: Optional[np.ndarray] = rollout_results.get("_rollout")
        if rollout_for_representatives is None and "rollout" in embeddings_dict:
            rollout_for_representatives = np.asarray(embeddings_dict["rollout"], dtype=np.float32)
        if rollout_for_representatives is not None:
            representative_results = run_rollout_representative_map_selection(
                embeddings_dict=embeddings_dict,
                rollout=rollout_for_representatives,
                slide_ids=slide_ids,
                paths=paths,
                args=args,
                num_patches_side=num_patches_side,
                faithfulness_result=faithfulness_result,
                n_per_group=3,
                embedding_dims=4,
            )
            rollout_results["representative_selection"] = representative_results

    # =========================================================================
    # STAGE 5: Rollout review panels
    # =========================================================================
    if args.generate_review_panels:
        rollout_for_panels: Optional[np.ndarray] = None
        if rollout_results is not None:
            rollout_for_panels = rollout_results.get("_rollout")
        if rollout_for_panels is None and "rollout" in embeddings_dict:
            rollout_for_panels = np.asarray(embeddings_dict["rollout"], dtype=np.float32)
        if rollout_for_panels is None:
            raise RuntimeError(
                "--generate-review-panels requires attention rollout maps. "
                "Run --compute-attention-rollout or load embeddings containing 'rollout'."
            )

        review_csv_path = run_review_panel_generation(
            rollout=rollout_for_panels,
            embeddings_dict=embeddings_dict,
            dataloader=loader,
            paths=paths,
            args=args,
            num_patches_side=num_patches_side,
            class_names=class_names,
            n_per_group=args.review_n_per_group,
            faithfulness_result=faithfulness_result if args.compute_faithfulness else None,
        )

    # =========================================================================
    # STAGE 6: Consolidated summary
    # =========================================================================
    save_consolidated_xai_summary(
        paths=paths,
        args=args,
        rollout_results=rollout_results,
        faithfulness_results=faithfulness_results,
        umap_metrics=umap_metrics,
        review_csv_path=review_csv_path,
    )


if __name__ == "__main__":
    main()
