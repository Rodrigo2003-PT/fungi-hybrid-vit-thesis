"""
visualization.py — rollout-only XAI visualization module.

This module supports the final LSA-ConvTok XAI pipeline:
- rollout relevance heatmaps,
- rollout-ranked faithfulness curves,
- rollout-only review panels.

The module contains no gradient-based attribution visualisation code. Review
panels contain only the model-input display image, rollout overlay, rollout
heatmap, and metadata/rubric fields.

References
----------
- Abnar & Zuidema (2020): Quantifying Attention Flow in Transformers.
- Samek et al. (2017): Evaluating the Visualization of What a Deep Neural Network has Learned.
- Petsiuk et al. (2018): RISE.
- Cochran (1977): Sampling Techniques.
- Tukey (1977): Exploratory Data Analysis.

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from stratified_sampling import (
    StratifiedSample,
    SpatialMetricsSummary,
)

ArrayLike1D = Union[np.ndarray, Sequence[float]]


# =============================================================================
# Input validation helpers
# =============================================================================

def _as_1d_float(x: ArrayLike1D, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like. Got shape {arr.shape}.")
    return arr

def _as_2d_float(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {arr.shape}.")
    return arr

def _assert_square_patches(vec: np.ndarray, num_patches_side: int, name: str) -> None:
    expected = num_patches_side * num_patches_side
    if vec.size != expected:
        raise ValueError(
            f"{name} has length {vec.size}, but num_patches_side^2 = {expected}. "
            f"Check patch/grid parameters."
        )


# =============================================================================
# SpatialAttentionVisualizer — unchanged from baseline
# =============================================================================

class SpatialAttentionVisualizer:
    """
    Spatial map visualizer with stratified sampling support.
    """

    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._configure_matplotlib()

    def _configure_matplotlib(self) -> None:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "figure.facecolor": "white",
                "savefig.dpi": self.dpi,
            }
        )

    # ---------- Generic plotting primitive ----------

    def plot_patch_scalar_map(
        self,
        values: ArrayLike1D,
        num_patches_side: int,
        title: str,
        save_name: str,
        cbar_label: str,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = False,
        report_scalars: bool = True,
    ) -> str:
        """
        Plot a patch-level scalar map as a num_patches_side × num_patches_side heatmap.
        """
        values = _as_1d_float(values, "values")
        _assert_square_patches(values, num_patches_side, "values")

        grid = values.reshape(num_patches_side, num_patches_side)

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)

        if report_scalars:
            l1_norm = float(np.sum(np.abs(values)))
            l2_norm = float(np.linalg.norm(values))
            max_val = float(np.max(np.abs(values)))
            title_with_stats = (
                f"{title}\n"
                f"L1={l1_norm:.3f} | L2={l2_norm:.3f} | max={max_val:.3f}"
            )
        else:
            title_with_stats = title

        ax.set_title(title_with_stats, fontweight="bold", pad=12)
        ax.set_xlabel("Patch column", fontweight="bold")
        ax.set_ylabel("Patch row", fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return str(save_path)

    # ---------- Rollout and attribution ----------

    def plot_rollout_relevance_heatmap(
        self,
        rollout_relevance: Union[np.ndarray, ArrayLike1D],
        num_patches_side: int,
        sample_idx: int = 0,
        title: str = "Attention-flow rollout relevance (mean pooling)",
        save_name: str = "rollout_relevance.png",
        show: bool = False,
    ) -> str:
        arr = np.asarray(rollout_relevance, dtype=np.float32)
        if arr.ndim == 2:
            if sample_idx < 0 or sample_idx >= arr.shape[0]:
                raise IndexError(f"sample_idx {sample_idx} out of range")
            vec = arr[sample_idx]
            title_use = f"{title} | sample {sample_idx}"
        elif arr.ndim == 1:
            vec = arr
            title_use = title
        else:
            raise ValueError(f"rollout_relevance must be 1D or 2D. Got {arr.shape}")

        return self.plot_patch_scalar_map(
            values=vec,
            num_patches_side=num_patches_side,
            title=title_use,
            save_name=save_name,
            cbar_label="Attention-flow relevance (a.u.)",
            cmap="viridis",
            show=show,
        )

    def plot_rollout_comparison_panel(
        self,
        rollout_dict: Dict[str, np.ndarray],
        num_patches_side: int,
        sample_indices: Dict[str, int],
        title: str = "Rollout Comparison Panel",
        save_name: str = "rollout_comparison_panel.png",
        show: bool = False,
    ) -> str:
        """
        Plot rollout relevance maps with SHARED colorbar scale for valid comparison.
        """
        n_groups = len(rollout_dict)
        if n_groups == 0:
            raise ValueError("rollout_dict is empty")

        all_values = [rollout_dict[g] for g in rollout_dict]
        global_vmax = float(np.max([np.max(v) for v in all_values]))

        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5.5))
        if n_groups == 1:
            axes = [axes]

        for ax, (group_name, vec) in zip(axes, rollout_dict.items()):
            idx = sample_indices[group_name]
            vec_np = np.asarray(vec, dtype=np.float32)

            grid = vec_np.reshape(num_patches_side, num_patches_side)
            im = ax.imshow(grid, cmap="viridis", interpolation="nearest",
                           vmin=0, vmax=global_vmax)

            ax.set_title(f"{group_name} (sample {idx})", fontweight="bold", fontsize=10)
            ax.set_xlabel("Patch column")
            ax.set_ylabel("Patch row")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Rollout relevance", fontsize=9)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        print(f"  Saved: {save_path.name}")
        return str(save_path)




def _format_optional_float(value) -> str:
    try:
        x = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.6f}"

# =============================================================================
# RolloutReviewPanel
# =============================================================================

class RolloutReviewPanel:

    _RUBRIC_CATEGORIES = [
        "structure-centered",
        "boundary-centered",
        "background-dominant",
        "artifact-dominant",
        "diffuse",
        "mixed-focus",
    ]

    def __init__(
        self,
        output_dir: str,
        dpi: int = 200,
        class_names: Optional[List[str]] = None,
        rollout_cmap: str = "viridis",
        overlay_alpha: float = 0.55,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)
        self.class_names = class_names or ["class0", "class1"]
        self.rollout_cmap = rollout_cmap
        self.overlay_alpha = float(overlay_alpha)
        self._provenance_rows: List[Dict] = []
        plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})

    @staticmethod
    def _compute_group_scales(values: np.ndarray, group_labels: np.ndarray) -> Dict[str, float]:
        scales: Dict[str, float] = {}
        for group in np.unique(group_labels):
            mask = group_labels == group
            if np.any(mask):
                vmax = float(np.max(values[mask]))
                scales[str(group)] = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        return scales

    def save_provenance_csv(self, csv_name: Optional[str] = None) -> str:
        if not self._provenance_rows:
            warnings.warn("No rollout review provenance rows to save. Call generate_panels() first.")
            return ""
        csv_name = csv_name or "rollout_review_provenance.csv"
        csv_path = self.output_dir / csv_name
        fieldnames = list(self._provenance_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._provenance_rows)
        print(f"  Saved rollout review provenance: {csv_path.name}")
        return str(csv_path)

    """
    Rollout-only review panel generator for the final XAI pipeline.

    Produces one PNG panel per sample with:
      col 0 — model-input display-normalised image
      col 1 — attention-rollout overlay
      col 2 — standalone attention-rollout heatmap
      col 3 — metadata and fixed plausibility-review rubric

    Rollout is the primary architecture-aligned explanation map and is validated through rollout-ranked insertion/deletion AUC.
    """

    def generate_panels(
        self,
        raw_images: np.ndarray,
        rollout_maps: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        sample_indices: np.ndarray,
        num_patches_side: int,
        group_labels: Optional[np.ndarray] = None,
        slide_ids: Optional[np.ndarray] = None,
        selection_metadata: Optional[List[Dict]] = None,
        split: str = "test",
        shared_scale_within_group: Optional[bool] = None,
        scale_mode: str = "within_group",
    ) -> List[str]:
        """
        Render rollout review panels.

        Parameters
        ----------
        scale_mode:
            ``"within_group"`` — shared vmax separately for TP/TN/FP/FN;
            ``"global"``       — one vmax shared by all selected panels;
            ``"per_sample"``   — each panel autoscaled to its own maximum.

        ``shared_scale_within_group`` is retained for backward compatibility:
        True maps to ``within_group`` and False maps to ``per_sample``.
        """
        N = len(labels)
        if raw_images.shape[0] != N:
            raise ValueError(f"raw_images.shape[0]={raw_images.shape[0]} != N={N}")
        if rollout_maps.shape[0] != N:
            raise ValueError(f"rollout_maps.shape[0]={rollout_maps.shape[0]} != N={N}")
        if group_labels is None:
            group_labels = _assign_group_labels(labels, predictions)
        if slide_ids is None:
            slide_ids = np.array([""] * N, dtype=object)
        else:
            slide_ids = np.asarray(slide_ids, dtype=object)
            if slide_ids.shape[0] != N:
                raise ValueError(f"slide_ids length {slide_ids.shape[0]} != N={N}")
        if selection_metadata is None:
            selection_metadata = [{} for _ in range(N)]
        if len(selection_metadata) != N:
            raise ValueError(f"selection_metadata length {len(selection_metadata)} != N={N}")

        if shared_scale_within_group is not None:
            scale_mode = "within_group" if shared_scale_within_group else "per_sample"
        if scale_mode not in {"within_group", "global", "per_sample"}:
            raise ValueError("scale_mode must be 'within_group', 'global', or 'per_sample'")

        rollout_scales = self._compute_group_scales(rollout_maps, group_labels) if scale_mode == "within_group" else None
        global_vmax = float(np.max(rollout_maps)) if scale_mode == "global" else None
        if global_vmax is not None and (global_vmax <= 0 or not np.isfinite(global_vmax)):
            global_vmax = 1.0

        paths: List[str] = []

        for i in range(N):
            raw_img    = raw_images[i]
            ro_vec     = rollout_maps[i]
            label      = int(labels[i])
            pred       = int(predictions[i])
            prob_vec   = probabilities[i]
            confidence = float(prob_vec[pred])
            sample_idx = int(sample_indices[i])
            group      = str(group_labels[i])
            slide_id   = str(slide_ids[i])
            meta       = dict(selection_metadata[i])

            if scale_mode == "within_group":
                ro_vmax = rollout_scales.get(group, float(ro_vec.max())) if rollout_scales is not None else float(ro_vec.max())
            elif scale_mode == "global":
                ro_vmax = float(global_vmax)
            else:
                ro_vmax = float(ro_vec.max())
            if ro_vmax <= 0 or not np.isfinite(ro_vmax):
                ro_vmax = 1.0

            save_name = f"review_rollout_{split}_{group}_{sample_idx:05d}.png"
            saved_path = self._render_rollout_panel(
                raw_img=raw_img,
                ro_vec=ro_vec,
                label=label,
                pred=pred,
                confidence=confidence,
                group=group,
                sample_idx=sample_idx,
                num_patches_side=num_patches_side,
                ro_vmax=ro_vmax,
                save_name=save_name,
                slide_id=slide_id,
                selection_reason=str(meta.get("selection_reason", "")),
                selection_role=str(meta.get("selection_role", "")),
                mmd2_after_selection=meta.get("mmd2_after_selection", None),
                witness_score=meta.get("witness_score", None),
                scale_mode=scale_mode,
            )
            paths.append(saved_path)

            row = {
                "split": split,
                "sample_idx": sample_idx,
                "slide_id": slide_id,
                "group": group,
                "true_label": label,
                "true_class": self.class_names[label] if label < len(self.class_names) else str(label),
                "prediction": pred,
                "pred_class": self.class_names[pred] if pred < len(self.class_names) else str(pred),
                "confidence": round(confidence, 6),
                "selection_role": meta.get("selection_role", ""),
                "selection_reason": meta.get("selection_reason", ""),
                "selection_metric": meta.get("selection_metric", ""),
                "selection_target": meta.get("selection_target", ""),
                "mmd2_after_selection": meta.get("mmd2_after_selection", ""),
                "witness_score": meta.get("witness_score", ""),
                "kernel_gamma": meta.get("kernel_gamma", ""),
                "within_group_rank": meta.get("within_group_rank", ""),
                "slide_diversity_applied": meta.get("slide_diversity_applied", ""),
                "scale_mode": scale_mode,
                "rollout_gini": round(float(meta.get("rollout_gini", _spatial_gini(ro_vec))), 6),
                "rollout_entropy": round(float(meta.get("rollout_entropy", _spatial_entropy(ro_vec))), 6),
                "rollout_l1": round(float(meta.get("rollout_l1", np.sum(np.abs(ro_vec)))), 6),
                "rollout_l2": round(float(meta.get("rollout_l2", np.linalg.norm(ro_vec))), 6),
                "ranking_source": "attention_rollout",
                "panel_path": saved_path,
                "rubric_label": "",
            }
            self._provenance_rows.append(row)

        return paths

    def _render_rollout_panel(
        self,
        raw_img: np.ndarray,
        ro_vec: np.ndarray,
        label: int,
        pred: int,
        confidence: float,
        group: str,
        sample_idx: int,
        num_patches_side: int,
        ro_vmax: float,
        save_name: str,
        slide_id: str = "",
        selection_reason: str = "",
        selection_role: str = "",
        mmd2_after_selection = None,
        witness_score = None,
        scale_mode: str = "within_group",
    ) -> str:
        raw_display = _raw_image_to_display(raw_img)
        H, W = raw_display.shape[:2]
        ro_overlay = _upsample_map(ro_vec, num_patches_side, H, W)
        ro_grid = ro_vec.reshape(num_patches_side, num_patches_side)

        label_name = self.class_names[label] if label < len(self.class_names) else str(label)
        pred_name  = self.class_names[pred]  if pred  < len(self.class_names) else str(pred)
        correct    = "✓" if label == pred else "✗"

        meta_lines = [
            f"Sample:     {sample_idx}",
            f"Slide:      {slide_id}" if slide_id else "Slide:      n/a",
            f"Group:      {group}",
            "",
            f"True label: {label_name} ({label})",
            f"Prediction: {pred_name} ({pred}) {correct}",
            f"Confidence: {confidence:.4f}",
            "",
            "Explanation: attention rollout",
            f"Scale mode:      {scale_mode}",
            f"Selection:       {selection_reason or 'not recorded'}",
            f"Role:            {selection_role or 'not recorded'}",
            f"MMD² after sel.: {_format_optional_float(mmd2_after_selection)}",
            f"Witness score:   {_format_optional_float(witness_score)}",
            f"Rollout Gini:    {_spatial_gini(ro_vec):.4f}",
            f"Rollout entropy: {_spatial_entropy(ro_vec):.4f}",
            f"Rollout L1:      {float(np.sum(np.abs(ro_vec))):.4f}",
            f"Rollout L2:      {float(np.linalg.norm(ro_vec)):.4f}",
            "",
            "Rubric categories:",
        ]
        for cat in self._RUBRIC_CATEGORIES:
            meta_lines.append(f"  [ ] {cat}")

        n_cols = 4
        fig_w = n_cols * 3.4
        fig_h = 3.8
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        gs = gridspec.GridSpec(1, n_cols, figure=fig,
                               left=0.01, right=0.99, top=0.88, bottom=0.04,
                               wspace=0.16)

        _group_colors = {"TP": "#d4efdf", "TN": "#d6eaf8", "FP": "#fadbd8", "FN": "#fdebd0"}
        title_bg = _group_colors.get(group, "#f2f3f4")
        fig.suptitle(
            f"[{group}] Sample {sample_idx} | True: {label_name}  Pred: {pred_name}  Conf: {confidence:.3f}",
            fontsize=11, fontweight="bold", y=0.97,
            bbox=dict(facecolor=title_bg, edgecolor="none", boxstyle="round,pad=0.3"),
        )

        ax0 = fig.add_subplot(gs[0, 0])
        _plot_raw(ax0, raw_display, title="Model-input\n(display-norm.)")

        ax1 = fig.add_subplot(gs[0, 1])
        _plot_overlay(ax1, raw_display, ro_overlay,
                      cmap=self.rollout_cmap, alpha=self.overlay_alpha,
                      vmax=ro_vmax, title="Rollout overlay")

        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(ro_grid, cmap=self.rollout_cmap, vmin=0, vmax=ro_vmax, interpolation="nearest")
        ax2.set_title("Rollout heatmap", fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis("off")
        ax3.text(
            0.04, 0.97,
            "\n".join(meta_lines),
            transform=ax3.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.7, edgecolor="#cccccc"),
        )

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)


# =============================================================================
# FaithfulnessVisualizer — new class
# =============================================================================

class FaithfulnessVisualizer:
    """
    Visualizer for insertion/deletion faithfulness curves.

    Generates per-group mean curve plots (with ±1 SD band) and AUC boxplots
    for each operator × mode combination.
    """

    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.facecolor": "white",
            "savefig.dpi": dpi,
        })

    @staticmethod
    def _ranking_variant_label(curves: "FaithfulnessCurves") -> str:
        variant = getattr(curves, "ranking_variant", "rollout")
        if variant in {"rollout", "attention_rollout"}:
            return "rollout-ranked"
        if variant == "random":
            return "random-control ranked"
        if variant == "inverse":
            return "inverse-rollout-control ranked"
        return f"{variant}-ranked"

    @staticmethod
    def _curve_file_prefix(curves: "FaithfulnessCurves") -> str:
        variant = getattr(curves, "ranking_variant", "rollout")
        if variant in {"rollout", "attention_rollout"}:
            return "rollout"
        return str(variant)

    def plot_faithfulness_curves(
        self,
        curves: "FaithfulnessCurves",   # type annotation via string (forward ref)
        y_true: np.ndarray,
        y_pred: np.ndarray,
        arch_name: str = "",
        split: str = "test",
        show: bool = False,
    ) -> str:
        """
        Plot mean ± SD insertion/deletion curves by TP/TN/FP/FN group.

        Parameters
        ----------
        curves    : FaithfulnessCurves object from compute_faithfulness_curves().
        y_true    : (N,) ground-truth labels.
        y_pred    : (N,) predicted labels.
        arch_name : Architecture label for the plot title.
        split     : Split name for file naming.
        show      : Display figure interactively.

        Returns
        -------
        str — path to saved figure.
        """
        tp = (y_true == 1) & (y_pred == 1)
        tn = (y_true == 0) & (y_pred == 0)
        fp = (y_true == 0) & (y_pred == 1)
        fn = (y_true == 1) & (y_pred == 0)

        group_masks  = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        group_colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}

        # x-axis: normalised perturbation fraction 0 → 1
        n_steps = curves.n_steps
        x_axis  = np.linspace(0.0, 1.0, n_steps + 1)

        fig, ax = plt.subplots(figsize=(9, 5.5))

        for group, mask in group_masks.items():
            n = int(np.sum(mask))
            if n == 0:
                continue
            grp_curves = curves.score_curves[mask]   # (n, n_steps+1)
            mean_c = grp_curves.mean(axis=0)
            std_c  = grp_curves.std(axis=0)

            auc_mean = float(np.mean(curves.auc_scores[mask]))
            color    = group_colors[group]

            ax.plot(x_axis, mean_c, color=color, linewidth=2.0,
                    label=f"{group} (n={n}, AUC={auc_mean:.3f})")
            ax.fill_between(x_axis, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.15)

        op_label   = curves.operator.replace("_", " ")
        mode_label = curves.mode.capitalize()
        ax.set_xlabel("Fraction of patches perturbed", fontweight="bold")
        ax.set_ylabel("Target-class logit", fontweight="bold")
        rank_label = self._ranking_variant_label(curves)
        ax.set_title(
            f"{mode_label} curve — {rank_label} patches | operator: {op_label} | {arch_name} | {split.upper()}",
            fontweight="bold", pad=10
        )
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(0, 1)

        plt.tight_layout()
        prefix = self._curve_file_prefix(curves)
        save_name = (
            f"faithfulness_{prefix}_{curves.operator}_{curves.mode}_{split}.png"
        )
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        print(f"  Saved: {save_path.name}")
        return str(save_path)

    def plot_probability_sensitivity_curves(
        self,
        curves: "FaithfulnessCurves",
        y_true: np.ndarray,
        y_pred: np.ndarray,
        arch_name: str = "",
        split: str = "test",
        show: bool = False,
    ) -> str:
        """
        Plot the probability-valued sensitivity curves stored separately from
        the primary logit faithfulness curves.  These curves are clamped to
        [0, 1] because they are softmax probabilities.
        """
        tp = (y_true == 1) & (y_pred == 1)
        tn = (y_true == 0) & (y_pred == 0)
        fp = (y_true == 0) & (y_pred == 1)
        fn = (y_true == 1) & (y_pred == 0)
        group_masks  = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        group_colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}

        x_axis = np.linspace(0.0, 1.0, curves.n_steps + 1)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for group, mask in group_masks.items():
            n = int(np.sum(mask))
            if n == 0:
                continue
            grp_curves = curves.secondary_score_curves[mask]
            mean_c = grp_curves.mean(axis=0)
            std_c = grp_curves.std(axis=0)
            ax.plot(x_axis, mean_c, color=group_colors[group], linewidth=2.0, label=f"{group} (n={n})")
            ax.fill_between(x_axis, mean_c - std_c, mean_c + std_c, color=group_colors[group], alpha=0.15)

        ax.set_xlabel("Fraction of patches perturbed", fontweight="bold")
        ax.set_ylabel("Target-class probability", fontweight="bold")
        ax.set_title(
            f"Probability sensitivity — {self._ranking_variant_label(curves)} {curves.mode} / {curves.operator} | {arch_name} | {split.upper()}",
            fontweight="bold", pad=10,
        )
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        prefix = self._curve_file_prefix(curves)
        save_path = self.output_dir / f"faithfulness_{prefix}_probability_{curves.operator}_{curves.mode}_{split}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
        print(f"  Saved: {save_path.name}")
        return str(save_path)

    def plot_auc_comparison(
        self,
        faithfulness_result: "FaithfulnessResult",
        y_true: np.ndarray,
        y_pred: np.ndarray,
        arch_name: str = "",
        split: str = "test",
        show: bool = False,
    ) -> str:
        """
        Boxplot of AUC scores by group and operator × mode combination.

        Parameters
        ----------
        faithfulness_result : FaithfulnessResult containing all curve bundles.
        y_true, y_pred      : Label arrays for group assignment.
        arch_name           : Architecture label.
        split               : Split name.
        show                : Display interactively.

        Returns
        -------
        str — path to saved figure.
        """
        from scipy import stats as scipy_stats

        tp = (y_true == 1) & (y_pred == 1)
        tn = (y_true == 0) & (y_pred == 0)
        fp = (y_true == 0) & (y_pred == 1)
        fn = (y_true == 1) & (y_pred == 0)
        group_masks  = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        group_colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}

        n_combos = len(faithfulness_result.curves)
        if n_combos == 0:
            warnings.warn("FaithfulnessResult has no curves to plot.")
            return ""

        fig, axes = plt.subplots(1, n_combos, figsize=(5 * n_combos, 6), squeeze=False)
        axes = axes[0]

        for ax, (key, curve) in zip(axes, faithfulness_result.curves.items()):
            data    = []
            xlabels = []
            colors  = []
            for group, mask in group_masks.items():
                n = int(np.sum(mask))
                if n == 0:
                    continue
                vals = curve.auc_scores[mask]
                data.append(vals)
                xlabels.append(f"{group}\n(n={n})")
                colors.append(group_colors[group])

            if not data:
                ax.set_visible(False)
                continue

            bp = ax.boxplot(
                data, labels=xlabels, patch_artist=True,
                showmeans=True, meanline=True, widths=0.55,
                medianprops=dict(color="black", linewidth=2),
                meanprops=dict(color="darkred", linewidth=2, linestyle="--"),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            # Scatter overlay
            for j, (vals, c) in enumerate(zip(data, colors), start=1):
                n_pts = len(vals)
                xj = np.random.normal(j, 0.04, size=min(n_pts, 200))
                ax.scatter(xj, vals[:200], alpha=0.3, s=18, color=c, edgecolors="none")

            # TP vs FP Welch's t-test annotation
            tp_vals = [d for d, g in zip(data, [g for g in group_masks if np.sum(group_masks[g]) > 0]) if g == "TP"]
            fp_vals = [d for d, g in zip(data, [g for g in group_masks if np.sum(group_masks[g]) > 0]) if g == "FP"]
            if tp_vals and fp_vals and len(tp_vals[0]) >= 2 and len(fp_vals[0]) >= 2:
                _, p = scipy_stats.ttest_ind(tp_vals[0], fp_vals[0], equal_var=False)
                sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.text(0.98, 0.98, f"TP vs FP: {sig}\np={p:.4f}",
                        transform=ax.transAxes, fontsize=8,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

            op_label   = curve.operator.replace("_", " ")
            mode_label = curve.mode.capitalize()
            rank_label = getattr(curve, "ranking_variant", "rollout")
            if rank_label in {"rollout", "attention_rollout"}:
                rank_label = "rollout"
            ax.set_title(f"{rank_label}: {mode_label} / {op_label}", fontweight="bold")
            ax.set_ylabel("Raw target-logit AUC", fontweight="bold")
            ax.grid(axis="y", alpha=0.25, linestyle="--")

        fig.suptitle(
            f"Faithfulness AUC by Group | {arch_name} | {split.upper()}",
            fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()

        save_name = f"faithfulness_auc_comparison_{split}.png"
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        print(f"  Saved: {save_path.name}")
        return str(save_path)


# =============================================================================
# SlideConditionedVisualizer — unchanged from baseline
# =============================================================================

class SlideConditionedVisualizer:
    """
    Slide-conditioned error visualization.
    """

    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._configure_matplotlib()

    def _configure_matplotlib(self) -> None:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "figure.facecolor": "white",
                "savefig.dpi": self.dpi,
            }
        )

    def plot_slide_conditioned_errors(
        self,
        projection: np.ndarray,
        slide_ids: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        class_names: List[str],
        method_name: str = "t-SNE",
        save_name: str = "slide_conditioned_errors.png",
        show: bool = False,
        max_slides_per_error: int = 5,
    ) -> str:
        proj = np.asarray(projection, dtype=np.float32)
        if proj.ndim != 2 or proj.shape[1] != 2:
            raise ValueError(f"projection must be (N, 2). Got {proj.shape}")

        slide_ids   = np.asarray(slide_ids)
        labels      = np.asarray(labels).astype(int)
        predictions = np.asarray(predictions).astype(int)

        tp = (labels == 1) & (predictions == 1)
        tn = (labels == 0) & (predictions == 0)
        fp = (labels == 0) & (predictions == 1)
        fn = (labels == 1) & (predictions == 0)

        fp_slides = np.unique(slide_ids[fp])[:max_slides_per_error]
        fn_slides = np.unique(slide_ids[fn])[:max_slides_per_error]

        n_fp = len(fp_slides)
        n_fn = len(fn_slides)

        if n_fp == 0 and n_fn == 0:
            print("No FP/FN errors to visualize")
            return ""

        n_rows = max(n_fp, n_fn)
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, sid in enumerate(fp_slides):
            ax = axes[i, 0]
            slide_mask   = slide_ids == sid
            fp_in_slide  = slide_mask & fp
            tn_in_slide  = slide_mask & tn

            ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=10, alpha=0.08)
            if np.any(tn_in_slide):
                ax.scatter(proj[tn_in_slide, 0], proj[tn_in_slide, 1],
                           c="blue", s=90, alpha=0.75, marker="o",
                           edgecolors="black", linewidth=0.8,
                           label=f"TN (n={int(np.sum(tn_in_slide))})")
            if np.any(fp_in_slide):
                ax.scatter(proj[fp_in_slide, 0], proj[fp_in_slide, 1],
                           c="red", s=140, alpha=0.90, marker="X",
                           edgecolors="black", linewidth=1.2,
                           label=f"FP (n={int(np.sum(fp_in_slide))})")

            ax.set_title(f"Slide {sid} (GT={class_names[0]})", fontweight="bold")
            ax.set_xlabel(f"{method_name} dim 1")
            ax.set_ylabel(f"{method_name} dim 2")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        for i in range(len(fp_slides), n_rows):
            axes[i, 0].axis("off")

        for i, sid in enumerate(fn_slides):
            ax = axes[i, 1]
            slide_mask   = slide_ids == sid
            fn_in_slide  = slide_mask & fn
            tp_in_slide  = slide_mask & tp

            ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=10, alpha=0.08)
            if np.any(tp_in_slide):
                ax.scatter(proj[tp_in_slide, 0], proj[tp_in_slide, 1],
                           c="green", s=90, alpha=0.75, marker="o",
                           edgecolors="black", linewidth=0.8,
                           label=f"TP (n={int(np.sum(tp_in_slide))})")
            if np.any(fn_in_slide):
                ax.scatter(proj[fn_in_slide, 0], proj[fn_in_slide, 1],
                           c="orange", s=140, alpha=0.90, marker="X",
                           edgecolors="black", linewidth=1.2,
                           label=f"FN (n={int(np.sum(fn_in_slide))})")

            ax.set_title(f"Slide {sid} (GT={class_names[1]})", fontweight="bold")
            ax.set_xlabel(f"{method_name} dim 1")
            ax.set_ylabel(f"{method_name} dim 2")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        for i in range(len(fn_slides), n_rows):
            axes[i, 1].axis("off")

        fig.suptitle(
            f"Slide-conditioned errors in {method_name} space",
            fontsize=15, fontweight="bold", y=0.995
        )

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return str(save_path)

    def plot_slide_error_summary(
        self,
        slide_ids: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_name: str = "slide_error_summary.png",
        show: bool = False,
        statistical_classification: Optional[Dict] = None,
    ) -> str:
        slide_ids   = np.asarray(slide_ids)
        labels      = np.asarray(labels).astype(int)
        predictions = np.asarray(predictions).astype(int)

        unique_slides  = np.unique(slide_ids)
        slide_error_rates = []
        slide_sizes       = []
        slide_gt          = []

        for sid in unique_slides:
            mask = slide_ids == sid
            slide_sizes.append(int(np.sum(mask)))
            slide_gt.append(int(labels[mask][0]))
            slide_error_rates.append(float(np.mean(labels[mask] != predictions[mask])))

        slide_error_rates = np.asarray(slide_error_rates, dtype=np.float32)
        slide_sizes       = np.asarray(slide_sizes, dtype=np.int32)
        slide_gt          = np.asarray(slide_gt, dtype=np.int32)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        ax1 = axes[0]
        ax1.hist(slide_error_rates, bins=20, color="slategray", edgecolor="black", alpha=0.75)
        ax1.axvline(slide_error_rates.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {slide_error_rates.mean():.3f}")
        ax1.set_title("Slide-level error rate distribution", fontweight="bold")
        ax1.set_xlabel("Error rate")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        ax2 = axes[1]
        for cls, color in [(0, "royalblue"), (1, "seagreen")]:
            m = slide_gt == cls
            if np.any(m):
                ax2.hist(slide_error_rates[m], bins=15, alpha=0.55, color=color,
                         edgecolor="black", label=f"Class {cls} (n={int(np.sum(m))})")
        ax2.set_title("Error rate by GT class", fontweight="bold")
        ax2.set_xlabel("Error rate")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.25)
        ax2.legend()

        ax3 = axes[2]
        colors = np.where(slide_gt == 0, "royalblue", "seagreen")
        ax3.scatter(slide_sizes, slide_error_rates, c=colors, s=90, alpha=0.7,
                    edgecolors="black", linewidth=0.4)
        ax3.set_title("Slide size vs error rate", fontweight="bold")
        ax3.set_xlabel("Slide size (patches)")
        ax3.set_ylabel("Error rate")
        ax3.grid(True, alpha=0.25)

        if statistical_classification is not None:
            cls  = statistical_classification.get("classification", "N/A")
            conf = statistical_classification.get("confidence", None)
            parts = [f"Statistical: {cls}"]
            if conf is not None:
                parts.append(f"conf={conf:.3f}")
            fig.suptitle(" | ".join(parts), fontsize=12, fontweight="bold", y=1.02)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return str(save_path)


# =============================================================================
# Module-level helpers
# =============================================================================

def plot_metric_distributions_by_group(
    metric_dict: Dict[str, np.ndarray],
    metric_name: str,
    save_path: str,
    ylabel: str = "Metric Value",
    title: str = "Metric Distribution by Group",
    dpi: int = 300,
    show: bool = False,
) -> str:
    """
    Scientific Goal: Show statistically significant differences in attention
    metrics (Gini, Entropy) between True Positives and False Positives.

    References:
    - Tukey (1977): Exploratory Data Analysis
    - McGill et al. (1978): Variations of Box Plots
    """
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 7))

    color_map = {
        "TP": "#2ecc71",
        "TN": "#3498db",
        "FP": "#e74c3c",
        "FN": "#f39c12",
    }

    data    = []
    labels  = []
    colors  = []

    order = ["TN", "FP", "TP", "FN"]

    for group_name in order:
        if group_name in metric_dict and len(metric_dict[group_name]) > 0:
            data.append(metric_dict[group_name])
            n        = len(metric_dict[group_name])
            mean_val = np.mean(metric_dict[group_name])
            labels.append(f"{group_name}\n(n={n})\nμ={mean_val:.3f}")
            colors.append(color_map[group_name])

    if len(data) == 0:
        print(f"Warning: No data to plot for {metric_name}")
        return ""

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(color="darkred", linewidth=2, linestyle="--"),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (group_data, color) in enumerate(zip(data, colors), start=1):
        plot_data = (
            np.random.choice(group_data, size=200, replace=False)
            if len(group_data) > 200
            else group_data
        )
        x_jitter = np.random.normal(i, 0.04, size=len(plot_data))
        ax.scatter(x_jitter, plot_data, alpha=0.3, s=20, color=color, edgecolors="none")

    if "TP" in metric_dict and "FP" in metric_dict:
        tp_data = metric_dict["TP"]
        fp_data = metric_dict["FP"]
        if len(tp_data) >= 2 and len(fp_data) >= 2:
            t_stat, p_val = stats.ttest_ind(tp_data, fp_data, equal_var=False)
            pooled_std    = np.sqrt(
                (np.var(tp_data, ddof=1) + np.var(fp_data, ddof=1)) / 2
            )
            cohens_d = (
                (np.mean(tp_data) - np.mean(fp_data)) / pooled_std
                if pooled_std > 0 else 0
            )
            sig_text   = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                          else "*" if p_val < 0.05 else "ns")
            annot_text = f"TP vs FP: p={p_val:.4f} {sig_text}\nCohen's d={cohens_d:.3f}"
            ax.text(
                0.98, 0.98,
                annot_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black",   linewidth=2, label="Median"),
        Line2D([0], [0], color="darkred", linewidth=2, linestyle="--", label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {Path(save_path).name}")
    return save_path


# =============================================================================
# Internal rendering utilities for rollout review panels
# =============================================================================

def _assign_group_labels(
    labels: np.ndarray, predictions: np.ndarray
) -> np.ndarray:
    """Assign "TP"/"TN"/"FP"/"FN" string per sample."""
    groups = np.empty(len(labels), dtype=object)
    groups[(labels == 1) & (predictions == 1)] = "TP"
    groups[(labels == 0) & (predictions == 0)] = "TN"
    groups[(labels == 0) & (predictions == 1)] = "FP"
    groups[(labels == 1) & (predictions == 0)] = "FN"
    return groups


def _raw_image_to_display(raw_img: np.ndarray) -> np.ndarray:
    """
    Convert (C, H, W) float32 model-input tensor to display-ready array.

    Returns
    -------
    (H, W) float32 in [0, 1] for single-channel, or (H, W, 3) for RGB.
    """
    arr = np.asarray(raw_img, dtype=np.float32)
    if arr.ndim == 2:
        # Already (H, W)
        pass
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]             # (1, H, W) → (H, W)
        elif arr.shape[0] >= 3:
            arr = arr[:3].transpose(1, 2, 0)  # (C, H, W) → (H, W, 3)
        else:
            arr = arr.mean(axis=0)   # unknown C > 1 → mean
    else:
        raise ValueError(f"raw_img must be 2D or 3D, got shape {arr.shape}")

    # Min-max normalise to [0, 1]
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr


def _upsample_map(
    vec: np.ndarray, num_patches_side: int, H: int, W: int
) -> np.ndarray:
    """
    Bilinearly upsample a (P,) patch map to (H, W) for overlay rendering.

    Uses scipy zoom for bilinear interpolation.
    """
    from scipy.ndimage import zoom
    grid    = vec.reshape(num_patches_side, num_patches_side).astype(np.float32)
    scale_h = H / num_patches_side
    scale_w = W / num_patches_side
    upsampled = zoom(grid, (scale_h, scale_w), order=1)   # bilinear
    # Normalise to [0, 1] for overlay
    mn, mx = float(upsampled.min()), float(upsampled.max())
    if mx - mn > 1e-8:
        upsampled = (upsampled - mn) / (mx - mn)
    return upsampled.astype(np.float32)


def _plot_raw(ax: plt.Axes, raw_display: np.ndarray, title: str) -> None:
    """Render a pre-processed raw image on ax."""
    if raw_display.ndim == 2:
        ax.imshow(raw_display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(np.clip(raw_display, 0, 1), interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _plot_overlay(
    ax: plt.Axes,
    raw_display: np.ndarray,
    saliency: np.ndarray,
    cmap: str,
    alpha: float,
    vmax: float,
    title: str,
) -> None:
    """Render raw image with semi-transparent saliency overlay."""
    if raw_display.ndim == 2:
        ax.imshow(raw_display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(np.clip(raw_display, 0, 1), interpolation="nearest")
    ax.imshow(saliency, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6),
              alpha=alpha, interpolation="bilinear")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _spatial_entropy(vec: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a non-negative attribution map treated as a distribution."""
    v = np.asarray(vec, dtype=np.float64)
    v = np.abs(v)
    s = v.sum()
    if s < eps:
        return 0.0
    p = v / s
    return float(-np.sum(p * np.log(p + eps)))


def _spatial_gini(vec: np.ndarray, eps: float = 1e-12) -> float:
    """Gini coefficient for a non-negative patch-level rollout map."""
    v = np.asarray(vec, dtype=np.float64).flatten()
    v = np.abs(v)
    v = v[np.isfinite(v)]
    if v.size == 0 or float(v.sum()) < eps:
        return 0.0
    v = np.sort(v)
    n = v.size
    cumulative = np.cumsum(v)
    g = (n + 1.0 - 2.0 * np.sum(cumulative) / cumulative[-1]) / n
    return float(np.clip(g, 0.0, 1.0))
