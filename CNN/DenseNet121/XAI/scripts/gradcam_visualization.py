"""
Grad-CAM Visualization Utilities

Produces figures for the Grad-CAM XAI analysis of the
DenseNet-121 fungal microscopy classifier. All figures follow the same visual
language used in the ViT Attention Rollout analysis to allow direct
side-by-side comparison in the thesis.

Figure types:
    1. Qualitative panel — raw image | CAM heatmap | overlay.
    2. Descriptor distribution box-violin plot (SCI + Entropy) per class.
    3. Aggregate CAM grid — rank-1 exemplar per (class × correct/incorrect).
    4. Statistical summary table saved as CSV.
    5. Perturbation faithfulness curves — deletion & insertion vs. random.
    6. Faithfulness AUC distribution — per-class box-violin plot.
    7. Faithfulness summary CSV.

Author: Rodrigo Sá
Date: 2025
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from gradcam_engine import select_rank1_exemplar


# ---------------------------------------------------------------------------
# Color palette — consistent with thesis figures
# ---------------------------------------------------------------------------

HEATMAP_CMAP = "jet"           # Grad-CAM convention
OVERLAY_ALPHA = 0.50           # transparency for heatmap over raw image
DPI = 300
FONT_TITLE  = 14
FONT_LABEL  = 11
FONT_TICK   = 9


# ---------------------------------------------------------------------------
# Utility: denormalize + channel squeeze
# ---------------------------------------------------------------------------

def _to_display_image(raw: np.ndarray) -> np.ndarray:
    """
    Convert a raw patch (float32, arbitrary range) to a display-ready
    grayscale uint8 array in [0, 255].
    """
    img = raw.astype(np.float32)
    img = img.squeeze()                             # (H, W)
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-8:
        img = (img - lo) / (hi - lo)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def _apply_heatmap(cam: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Overlay a Grad-CAM map (float32, [0,1]) onto a grayscale image (uint8).

    Returns an RGB uint8 array for display.
    """
    cmap = plt.colormaps[HEATMAP_CMAP]
    heatmap_rgb = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)   # (H, W, 3)

    # Convert grayscale to RGB for blending
    img_rgb = np.stack([image, image, image], axis=-1)            # (H, W, 3)

    overlay = (
        (1 - OVERLAY_ALPHA) * img_rgb.astype(np.float32)
        + OVERLAY_ALPHA * heatmap_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return overlay


# ---------------------------------------------------------------------------
# Figure 1: Single-sample triplet (raw | heatmap | overlay)
# ---------------------------------------------------------------------------

def plot_gradcam_triplet(
    raw_image: np.ndarray,
    cam: np.ndarray,
    true_label: str,
    predicted_label: str,
    descriptors: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = False,
    title_prefix: str = "",
) -> str:
    """
    Three-panel figure for a single patch: raw image | pure heatmap | overlay.

    Args:
        raw_image:       Raw (un-normalized) patch, shape (H, W) or (C, H, W).
        cam:             Grad-CAM map, shape (H, W), values in [0, 1].
        true_label:      Ground-truth class name (string).
        predicted_label: Predicted class name (string).
        descriptors:     Dict from compute_map_descriptors().
        save_path:       File path to save figure (.png).
        show:            Whether to call plt.show().
        title_prefix:    Optional string prepended to the figure suptitle.
    """
    img_u8 = _to_display_image(raw_image)
    overlay = _apply_heatmap(cam, img_u8)
    cmap_obj = plt.colormaps[HEATMAP_CMAP]

    correct = true_label == predicted_label
    status = "Correct" if correct else "✗ Incorrect"
    color  = "green" if correct else "red"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel 1: raw image
    axes[0].imshow(img_u8, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input Patch", fontsize=FONT_TITLE)
    axes[0].axis("off")

    # Panel 2: pure CAM heatmap
    im = axes[1].imshow(cam, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=FONT_TITLE)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Normalized activation")

    # Panel 3: overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=FONT_TITLE)
    axes[2].axis("off")

    sci_val = descriptors.get("sci", float("nan"))
    ent_val = descriptors.get("entropy", float("nan"))
    af_val  = descriptors.get("active_fraction", float("nan"))

    suptitle = (
        f"{title_prefix}  GT: {true_label} | Pred: {predicted_label} "
        f"[{status}]\n"
        f"SCI={sci_val:.3f}  H={ent_val:.3f}  "
        f"Active fraction (τ=0.5)={af_val:.3f}"
    )
    fig.suptitle(suptitle, fontsize=FONT_TITLE, color=color, y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path or ""


# ---------------------------------------------------------------------------
# Figure 2: Rank-1 exemplar grid (class × correct | incorrect)
# ---------------------------------------------------------------------------

def plot_exemplar_grid(
    partitions: Dict[str, Dict[str, Any]],
    dataset,                                     # NPYPatchDataset
    class_names: List[str],
    output_dir: str,
    descriptor_key: str = "sci",
    save_name: str = "gradcam_exemplar_grid.png",
    show: bool = False,
) -> str:
    """
    Create a grid figure with one rank-1 exemplar per (class × outcome) cell.

    Layout:
        row 0          = column headers (dedicated header row, slim height).
        rows 1..n_cls  = one class per row.
        cols           = [Correct | Incorrect] × [Raw | Heatmap | Overlay]
        → 6 columns total, n_classes + 1 rows total.

    Row 0 is allocated exclusively to the column headers and contains no image
    content. This prevents the header text axes from sharing GridSpec cells
    with the first-class image axes, which would cause images to be rendered
    on top of the invisible header text.

    The rank-1 exemplar is selected as the sample whose SCI (or the chosen
    descriptor) is closest to the group median, following the same criterion
    used for Attention Rollout in the ViT analysis (Euclidean distance to
    median descriptor value).

    Args:
        partitions:      Output of partition_by_class_and_correctness().
        dataset:         NPYPatchDataset (used to retrieve raw images).
        class_names:     Ordered list of class name strings.
        output_dir:      Directory for saving the output PNG.
        descriptor_key:  Descriptor used for exemplar selection.
        save_name:       Filename for the output PNG.
        show:            Whether to display interactively.
    """
    n_classes = len(class_names)
    # Layout: row 0 = column headers (dedicated); rows 1..n_classes = one class per row.
    # Previously n_rows = n_classes caused row 0 to be shared between the
    # header text axes and the first-class image axes, producing invisible
    # headers overlaid by images. The extra row fully separates the two.
    n_cols = 6
    n_rows = n_classes + 1   # row 0: headers; rows 1..n_classes: image panels

    # Height: slim header row (0.6 units) + n_classes full image rows (3.5 each).
    fig = plt.figure(figsize=(n_cols * 3.5, n_classes * 3.5 + 0.6 + 0.5))
    gs  = GridSpec(
        n_rows, n_cols,
        figure=fig,
        height_ratios=[0.18] + [1.0] * n_classes,   # header row is slim
        hspace=0.35,
        wspace=0.05,
    )

    col_headers = [
        "Correct — Input", "Correct — CAM", "Correct — Overlay",
        "Incorrect — Input", "Incorrect — CAM", "Incorrect — Overlay",
    ]
    col_colors = ["#2ca02c"] * 3 + ["#d62728"] * 3

    # Column headers occupy row 0 exclusively — no image content shares these axes.
    for col_i, (hdr, clr) in enumerate(zip(col_headers, col_colors)):
        ax = fig.add_subplot(gs[0, col_i])
        ax.text(
            0.5, 0.5, hdr,
            ha="center", va="center",
            fontsize=FONT_LABEL, fontweight="bold", color=clr,
            transform=ax.transAxes,
        )
        ax.axis("off")

    # Image panels start at row 1, offset by +1 from the class index.
    for row_i, cname in enumerate(class_names):
        img_row = row_i + 1    # row 0 is reserved for headers

        group = partitions[cname]

        for outcome_i, outcome_key in enumerate(["correct", "incorrect"]):
            grp = group[outcome_key]
            col_base = outcome_i * 3           # 0 or 3

            if len(grp["cams"]) == 0:
                # No samples in this (class, outcome) cell → placeholder
                for c in range(3):
                    ax = fig.add_subplot(gs[img_row, col_base + c])
                    ax.text(
                        0.5, 0.5, "No samples",
                        ha="center", va="center",
                        fontsize=FONT_TICK, color="gray",
                        transform=ax.transAxes,
                    )
                    ax.axis("off")
                continue

            # Rank-1 exemplar selection
            cam_stack = np.stack(grp["cams"], axis=0)      # (N, H, W)
            rank1_idx = select_rank1_exemplar(cam_stack, descriptor_key)
            global_i  = grp["indices"][rank1_idx]          # index in dataset
            cam       = grp["cams"][rank1_idx]
            descs     = grp["descriptors"][rank1_idx]

            raw     = dataset.get_raw_image(global_i)
            img_u8  = _to_display_image(raw)
            overlay = _apply_heatmap(cam, img_u8)

            panels = [
                (img_u8,  "gray",       {}),
                (cam,     HEATMAP_CMAP, {"vmin": 0, "vmax": 1}),
                (overlay, None,         {}),
            ]
            for c, (data, cmap_str, kwargs) in enumerate(panels):
                ax = fig.add_subplot(gs[img_row, col_base + c])
                ax.imshow(data, cmap=cmap_str, **kwargs)
                ax.axis("off")

                if c == 0:
                    ax.set_ylabel(
                        cname, fontsize=FONT_LABEL, fontweight="bold",
                        rotation=90, labelpad=6,
                    )
                    ax.yaxis.set_label_position("left")

                if c == 1:
                    sci_v = descs.get("sci", float("nan"))
                    ent_v = descs.get("entropy", float("nan"))
                    ax.set_xlabel(
                        f"SCI={sci_v:.2f}  H={ent_v:.2f}",
                        fontsize=FONT_TICK,
                    )

    fig.suptitle(
        "Grad-CAM Rank-1 Exemplars per Class × Outcome\n"
        f"(exemplar selected by |SCI − median(SCI)|, "
        f"SCI = Spatial Concentration Index)",
        fontsize=FONT_TITLE,
        y=1.01,
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Exemplar grid saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Figure 3: Descriptor distribution plots (SCI + Entropy per class)
# ---------------------------------------------------------------------------

def plot_descriptor_distributions(
    results: Dict[str, Any],
    class_names: List[str],
    output_dir: str,
    save_name: str = "gradcam_descriptor_distributions.png",
    show: bool = False,
) -> str:
    """
    Box-violin plot of the SCI and Spatial Entropy distributions for each class
    (correct vs. incorrect predictions shown as separate hue levels).

    The dual-panel layout (SCI left, Entropy right) is directly analogous to
    the Gini/Entropy figure in the ViT Attention Rollout analysis, enabling
    a visual comparison of localization quality across architectures.

    Args:
        results:     Output of run_gradcam_inference().
        class_names: Ordered list of class name strings.
        output_dir:  Directory for saving the output PNG.
        save_name:   Filename for the output PNG.
        show:        Whether to display interactively.
    """
    rows = []
    y_true = results["true_labels"]
    y_pred = results["predicted_classes"]
    descs  = results["descriptors"]
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    for i, desc in enumerate(descs):
        cname   = idx_to_class.get(int(y_true[i]), f"class_{y_true[i]}")
        correct = "Correct" if y_true[i] == y_pred[i] else "Incorrect"
        rows.append({
            "class":   cname,
            "outcome": correct,
            "SCI":     desc["sci"],
            "Entropy": desc["entropy"],
            "Active fraction": desc["active_fraction"],
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"Correct": "#2ca02c", "Incorrect": "#d62728"}

    for ax, metric in zip(axes, ["SCI", "Entropy"]):
        sns.violinplot(
            data=df,
            x="class",
            y=metric,
            hue="outcome",
            split=True,
            inner="box",
            palette=palette,
            ax=ax,
            linewidth=0.8,
        )
        sns.stripplot(
            data=df,
            x="class",
            y=metric,
            hue="outcome",
            dodge=True,
            palette=palette,
            alpha=0.25,
            size=2.5,
            ax=ax,
            legend=False,
        )
        ax.set_title(
            f"Grad-CAM {metric} Distribution",
            fontsize=FONT_TITLE,
        )
        ax.set_xlabel("Class", fontsize=FONT_LABEL)
        ax.set_ylabel(metric, fontsize=FONT_LABEL)
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Outcome", fontsize=FONT_TICK, title_fontsize=FONT_TICK)

    fig.suptitle(
        "Grad-CAM Localization Quality Descriptors\n"
        "SCI = Spatial Concentration Index (Gini analog, ↑ = more focused)\n"
        "H = Spatial Entropy (↓ = more focused)",
        fontsize=FONT_TITLE,
        y=1.05,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Descriptor distributions saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Figure 4: Per-class average CAM (mean activation map)
# ---------------------------------------------------------------------------

def plot_mean_cam_per_class(
    partitions: Dict[str, Dict[str, Any]],
    class_names: List[str],
    output_dir: str,
    save_name: str = "gradcam_mean_cam_per_class.png",
    show: bool = False,
) -> str:
    """
    Compute and visualize the mean Grad-CAM map for each class (correctly
    classified samples only), providing an aggregate view of which spatial
    regions the model consistently attends to for each fungal category.

    This is inspired by average saliency map analysis in the CNN XAI
    literature and provides a complementary view to the single-exemplar panels.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5))
    if n_classes == 1:
        axes = [axes]

    for ax, cname in zip(axes, class_names):
        correct_cams = partitions[cname]["correct"]["cams"]
        if len(correct_cams) == 0:
            ax.text(
                0.5, 0.5, "No correct\npredictions",
                ha="center", va="center",
                fontsize=FONT_LABEL, color="gray",
                transform=ax.transAxes,
            )
            ax.axis("off")
            ax.set_title(cname, fontsize=FONT_TITLE)
            continue

        mean_cam = np.mean(np.stack(correct_cams, axis=0), axis=0)   # (H, W)

        # Normalize mean map to [0, 1]
        lo, hi = mean_cam.min(), mean_cam.max()
        if hi - lo > 1e-8:
            mean_cam = (mean_cam - lo) / (hi - lo)

        im = ax.imshow(mean_cam, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Mean activation")
        ax.set_title(
            f"{cname}\n(n={len(correct_cams)} correct predictions)",
            fontsize=FONT_TITLE,
        )
        ax.axis("off")

    fig.suptitle(
        "Mean Grad-CAM Activation (Correctly Classified Samples)",
        fontsize=FONT_TITLE,
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Mean CAM plot saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# CSV summary table
# ---------------------------------------------------------------------------

def save_descriptor_summary_csv(
    results: Dict[str, Any],
    class_names: List[str],
    output_path: str,
) -> str:
    """
    Save a per-sample descriptor table to CSV for statistical reporting.

    Columns: path, true_class, predicted_class, correct,
             sci, entropy, active_fraction, peak_response.
    """
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    rows = []
    y_true = results["true_labels"]
    y_pred = results["predicted_classes"]
    descs  = results["descriptors"]
    paths  = results["paths"]

    for i in range(len(y_true)):
        rows.append({
            "path":            paths[i],
            "true_class":      idx_to_class.get(int(y_true[i]), str(y_true[i])),
            "predicted_class": idx_to_class.get(int(y_pred[i]), str(y_pred[i])),
            "correct":         bool(y_true[i] == y_pred[i]),
            "sci":             descs[i]["sci"],
            "entropy":         descs[i]["entropy"],
            "active_fraction": descs[i]["active_fraction"],
            "peak_response":   descs[i]["peak_response"],
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Descriptor summary CSV saved: {output_path}")
    return output_path


def print_descriptor_statistics(
    results: Dict[str, Any],
    class_names: List[str],
) -> None:
    """
    Print a formatted summary table (mean ± std) of SCI and Entropy
    per (class, outcome) group to stdout.
    """
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    y_true = results["true_labels"]
    y_pred = results["predicted_classes"]
    descs  = results["descriptors"]

    groups: Dict[Tuple, List] = {}
    for i in range(len(y_true)):
        cname   = idx_to_class.get(int(y_true[i]), str(y_true[i]))
        correct = "Correct" if y_true[i] == y_pred[i] else "Incorrect"
        key = (cname, correct)
        if key not in groups:
            groups[key] = []
        groups[key].append(descs[i])

    print("\n" + "=" * 70)
    print("GRAD-CAM DESCRIPTOR SUMMARY")
    print("=" * 70)
    print(f"{'Class':<20} {'Outcome':<12} {'N':>5}  {'SCI mean±std':>18}  {'H mean±std':>18}")
    print("-" * 70)
    for (cname, outcome), dlist in sorted(groups.items()):
        scis  = [d["sci"]     for d in dlist]
        ents  = [d["entropy"] for d in dlist]
        print(
            f"{cname:<20} {outcome:<12} {len(dlist):>5}  "
            f"{np.mean(scis):6.3f} ± {np.std(scis):5.3f}       "
            f"{np.mean(ents):6.3f} ± {np.std(ents):5.3f}"
        )
    print("=" * 70)
    print("SCI  = Spatial Concentration Index (Gini analog; ↑ = more focussed)")
    print("H    = Spatial Entropy (↓ = more focussed)")
    print()


# ---------------------------------------------------------------------------
# Figure 5: Perturbation faithfulness curves (mean deletion + insertion)
# ---------------------------------------------------------------------------

def plot_faithfulness_curves(
    faith_results: Dict[str, Any],
    class_names: List[str],
    output_dir: str,
    save_name: str = "gradcam_faithfulness_curves.png",
    show: bool = False,
) -> str:
    """
    Plot mean deletion and insertion confidence curves for Grad-CAM vs.
    random baseline, stratified by class AND prediction correctness.

    Layout: 4 rows x (1 + n_classes) columns.
        Row 0: Deletion   — correct predictions
        Row 1: Insertion  — correct predictions
        Row 2: Deletion   — incorrect predictions
        Row 3: Insertion  — incorrect predictions
        Left column = all classes; remaining = per-class.

    Stratification by correctness is methodologically necessary:
    faithfulness on correct predictions measures whether the model attends to
    truly discriminative regions; faithfulness on incorrect predictions reveals
    whether the model was confidently focused on spurious regions — a distinct
    and diagnostically important failure mode (Hooker et al., 2019).

    Each panel shows:
        - Solid coloured line: Grad-CAM mean curve +/- 1 std shaded band.
        - Dashed grey line:    Random baseline mean curve +/- 1 std shaded band.
        - AUC scalars reported in the legend.
    """
    del_gc  = np.stack(faith_results["del_curves_gradcam"])   # (N, S+1)
    ins_gc  = np.stack(faith_results["ins_curves_gradcam"])
    del_rnd = np.stack(faith_results["del_curves_random"])
    ins_rnd = np.stack(faith_results["ins_curves_random"])
    y_true  = faith_results["true_labels"]
    y_pred  = faith_results["predicted_classes"]
    correct = (y_true == y_pred)
    n_steps = faith_results["n_steps"]
    x_axis  = np.linspace(0.0, 1.0, n_steps + 1)
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))

    n_cols = 1 + len(class_names)
    fig, axes = plt.subplots(4, n_cols, figsize=(5 * n_cols, 14),
                             sharex=True, sharey="row")
    if n_cols == 1:
        axes = axes.reshape(4, 1)

    row_specs = [
        (del_gc,  del_rnd,  "Deletion",  "Correct"),
        (ins_gc,  ins_rnd,  "Insertion", "Correct"),
        (del_gc,  del_rnd,  "Deletion",  "Incorrect"),
        (ins_gc,  ins_rnd,  "Insertion", "Incorrect"),
    ]
    row_colors = {"Correct": "#2ca02c", "Incorrect": "#d62728"}

    def _plot_panel(ax, gc_c, rnd_c, outcome, title, ylabel):
        if len(gc_c) == 0:
            ax.text(0.5, 0.5, f"No {outcome.lower()}\npredictions",
                    ha="center", va="center", fontsize=FONT_TICK,
                    color="gray", transform=ax.transAxes)
            ax.axis("off")
            return
        gc_mean  = gc_c.mean(axis=0)
        gc_std   = gc_c.std(axis=0)
        rnd_mean = rnd_c.mean(axis=0)
        rnd_std  = rnd_c.std(axis=0)
        auc_gc   = float(_trapz(gc_mean, x_axis))
        auc_rnd  = float(_trapz(rnd_mean, x_axis))
        clr      = row_colors[outcome]
        ax.plot(x_axis, gc_mean, color=clr, linewidth=2,
                label=f"Grad-CAM (AUC={auc_gc:.3f})")
        ax.fill_between(x_axis, np.clip(gc_mean - gc_std, 0, 1),
                        np.clip(gc_mean + gc_std, 0, 1),
                        alpha=0.18, color=clr)
        ax.plot(x_axis, rnd_mean, color="#7f7f7f", linewidth=1.5,
                linestyle="--", label=f"Random (AUC={auc_rnd:.3f})")
        ax.fill_between(x_axis, np.clip(rnd_mean - rnd_std, 0, 1),
                        np.clip(rnd_mean + rnd_std, 0, 1),
                        alpha=0.12, color="#7f7f7f")
        ax.set_title(title, fontsize=FONT_TITLE - 1)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.set_xlabel("Fraction of pixels perturbed", fontsize=FONT_LABEL)
        ax.legend(fontsize=FONT_TICK)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for row_i, (gc_all, rnd_all, protocol, outcome) in enumerate(row_specs):
        out_mask = correct if outcome == "Correct" else ~correct

        # Overall column
        _plot_panel(
            axes[row_i, 0], gc_all[out_mask], rnd_all[out_mask], outcome,
            f"{protocol} — All classes ({outcome}, n={out_mask.sum()})",
            "Predicted-class confidence",
        )

        # Per-class columns
        for col_i, cname in enumerate(class_names):
            cls_mask = (y_true == col_i) & out_mask
            _plot_panel(
                axes[row_i, col_i + 1],
                gc_all[cls_mask], rnd_all[cls_mask], outcome,
                f"{protocol} — {cname}\n({outcome}, n={cls_mask.sum()})",
                "",
            )

    fig.suptitle(
        "Perturbation Faithfulness — Stratified by Class and Prediction Outcome\n"
        "Deletion (MoRF): lower area = more faithful  |  "
        "Insertion (MoRF-Insertion): higher area = more faithful",
        fontsize=FONT_TITLE,
        y=1.01,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Faithfulness curves saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Figure 6: Faithfulness AUC distribution (box-violin per class)
# ---------------------------------------------------------------------------

def plot_faithfulness_auc_distributions(
    faith_results: Dict[str, Any],
    class_names: List[str],
    output_dir: str,
    save_name: str = "gradcam_faithfulness_auc_distributions.png",
    show: bool = False,
) -> str:
    """
    Box-violin plots of AUDC and AUIC per (class x correctness), with the
    random baseline mean overlaid as a horizontal dashed reference.

    Layout: 2 columns x 2 rows
        Row 0: AUDC (deletion, lower = better)
        Row 1: AUIC (insertion, higher = better)
        Each row has two panels side by side:
            Left:  Correct predictions
            Right: Incorrect predictions

    Stratification by correctness allows the reader to distinguish:
        - Faithful correct predictions  : model attends to truly discriminative
          regions and the decision follows from them.
        - Faithful incorrect predictions: model attended to regions that were
          internally consistent with the wrong class — a diagnostic that
          reveals the nature of the error, not merely that it occurred.
    """
    audc_gc  = faith_results["audc_gradcam"]
    auic_gc  = faith_results["auic_gradcam"]
    audc_rnd = faith_results["audc_random"]
    auic_rnd = faith_results["auic_random"]
    y_true   = faith_results["true_labels"]
    y_pred   = faith_results["predicted_classes"]
    correct  = (y_true == y_pred)
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    rows = []
    for i in range(len(audc_gc)):
        cname   = idx_to_class.get(int(y_true[i]), str(y_true[i]))
        outcome = "Correct" if correct[i] else "Incorrect"
        rows.append({
            "class":           cname,
            "outcome":         outcome,
            "AUDC (Grad-CAM)": float(audc_gc[i]),
            "AUIC (Grad-CAM)": float(auic_gc[i]),
            "AUDC (Random)":   float(audc_rnd[i]),
            "AUIC (Random)":   float(auic_rnd[i]),
        })
    df = pd.DataFrame(rows)

    outcome_palette = {"Correct": "#2ca02c", "Incorrect": "#d62728"}
    metrics = [
        ("AUDC (Grad-CAM)", "AUDC (Random)", "AUDC  (lower = more faithful)"),
        ("AUIC (Grad-CAM)", "AUIC (Random)", "AUIC  (higher = more faithful)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey="row")

    for row_i, (gc_col, rnd_col, row_label) in enumerate(metrics):
        for col_i, outcome in enumerate(["Correct", "Incorrect"]):
            ax  = axes[row_i, col_i]
            clr = outcome_palette[outcome]
            df_out = df[df["outcome"] == outcome]

            if df_out.empty:
                ax.text(0.5, 0.5, f"No {outcome.lower()} predictions",
                        ha="center", va="center", fontsize=FONT_LABEL,
                        color="gray", transform=ax.transAxes)
                ax.axis("off")
                continue

            sns.violinplot(
                data=df_out, x="class", y=gc_col,
                color=clr, inner="box",
                linewidth=0.8, ax=ax, alpha=0.70,
            )
            sns.stripplot(
                data=df_out, x="class", y=gc_col,
                color=clr, alpha=0.35, size=3, ax=ax,
            )

            # Random baseline mean as dashed horizontal line per class
            for cls_i, cname in enumerate(class_names):
                mask_cls = (df_out["class"] == cname)
                if mask_cls.sum() == 0:
                    continue
                rnd_mean = float(df_out.loc[mask_cls, rnd_col].mean())
                ax.hlines(rnd_mean, cls_i - 0.30, cls_i + 0.30,
                          colors="#7f7f7f", linewidths=1.5, linestyles="--")

            ax.set_title(
                f"{outcome} predictions — {row_label.split('  ')[0]}",
                fontsize=FONT_TITLE, color=clr,
            )
            ax.set_xlabel("Class", fontsize=FONT_LABEL)
            ax.set_ylabel(row_label, fontsize=FONT_LABEL)
            ax.tick_params(labelsize=FONT_TICK)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Faithfulness AUC Distributions per Class and Prediction Outcome\n"
        "(coloured violin = Grad-CAM; grey dashes = random baseline mean)\n"
        "delta_del = AUDC_random - AUDC_gradcam  |  "
        "delta_ins = AUIC_gradcam - AUIC_random  (positive = better than random)",
        fontsize=FONT_TITLE,
        y=1.03,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Faithfulness AUC distributions saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Faithfulness summary CSV
# ---------------------------------------------------------------------------

def save_faithfulness_summary_csv(
    faith_results: Dict[str, Any],
    class_names: List[str],
    output_path: str,
) -> str:
    """
    Save per-sample faithfulness AUC values to CSV.

    One row per test patch. Columns: true_class, predicted_class, correct,
    audc_gradcam, auic_gradcam, audc_random, auic_random,
    delta_deletion, delta_insertion.

    This is the sample-level table. For the group-level inferential summary
    (means, SDs, 95% CIs, and Wilcoxon p-values) use
    save_faithfulness_group_stats_csv().
    """
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    audc_gc  = faith_results["audc_gradcam"]
    auic_gc  = faith_results["auic_gradcam"]
    audc_rnd = faith_results["audc_random"]
    auic_rnd = faith_results["auic_random"]
    y_true   = faith_results["true_labels"]
    y_pred   = faith_results["predicted_classes"]

    rows = []
    for i in range(len(audc_gc)):
        rows.append({
            "true_class":       idx_to_class.get(int(y_true[i]), str(y_true[i])),
            "predicted_class":  idx_to_class.get(int(y_pred[i]), str(y_pred[i])),
            "correct":          bool(y_true[i] == y_pred[i]),
            "audc_gradcam":     float(audc_gc[i]),
            "auic_gradcam":     float(auic_gc[i]),
            "audc_random":      float(audc_rnd[i]),
            "auic_random":      float(auic_rnd[i]),
            "delta_deletion":   float(audc_rnd[i] - audc_gc[i]),
            "delta_insertion":  float(auic_gc[i]  - auic_rnd[i]),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Per-sample faithfulness CSV saved: {output_path}")
    return output_path


def save_faithfulness_group_stats_csv(
    faith_summary: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Save group-level faithfulness statistics — including inferential results —
    to a machine-readable CSV.

    One row per group. Groups are:
        - One "overall" row (all test samples).
        - One row per true class (class-level aggregation).
        - One row per (class × prediction outcome) combination.

    This table is the persistent complement to print_faithfulness_summary().
    It enables downstream statistical analysis, table generation for the
    thesis, and exact reproduction of reported numbers from saved artefacts
    without re-running the full pipeline.

    Columns
    -------
    group_type      : "overall" | "per_class" | "per_class_outcome"
    class           : class name, or "all" for the overall group.
    outcome         : "all" | "correct" | "incorrect"
    n_samples       : number of patches in the group.
    n_slides        : number of distinct slides in the group
                      (0 when slide_groups was not provided).

    Descriptive statistics (patch-level, both protocols):
    audc_gradcam_mean, audc_gradcam_std  — deletion AUC, Grad-CAM
    auic_gradcam_mean, auic_gradcam_std  — insertion AUC, Grad-CAM
    audc_random_mean,  audc_random_std   — deletion AUC, random baseline
    auic_random_mean,  auic_random_std   — insertion AUC, random baseline
    delta_deletion_mean, delta_deletion_std   — patch-level mean Δ_del
    delta_insertion_mean, delta_insertion_std — patch-level mean Δ_ins

    Slide-level Δ (primary estimand used for inferential statistics):
    delta_deletion_slide_mean   — mean of per-slide mean(Δ_del).
    delta_deletion_slide_std    — std of per-slide mean(Δ_del).
    delta_insertion_slide_mean  — mean of per-slide mean(Δ_ins).
    delta_insertion_slide_std   — std of per-slide mean(Δ_ins).
    (NaN when slide_groups was not provided.)

    Inferential statistics (deletion gap):
    deletion_wilcoxon_stat              — Wilcoxon W on slide-level means
                                          (NaN if n_slides < 10).
    deletion_wilcoxon_pvalue            — one-sided raw p, H1: Δ_del > 0.
    deletion_wilcoxon_pvalue_bonferroni — Bonferroni-corrected p (FWER).
    deletion_wilcoxon_pvalue_bh         — Benjamini-Hochberg corrected p (FDR).
    deletion_ci95_low                   — lower bound of 95% bootstrap CI
                                          for mean slide-level Δ_del.
    deletion_ci95_high                  — upper bound.

    Inferential statistics (insertion gap):
    insertion_wilcoxon_stat              — Wilcoxon W on slide-level means.
    insertion_wilcoxon_pvalue            — one-sided raw p, H1: Δ_ins > 0.
    insertion_wilcoxon_pvalue_bonferroni — Bonferroni-corrected p (FWER).
    insertion_wilcoxon_pvalue_bh         — BH corrected p (FDR).
    insertion_ci95_low                   — lower bound of 95% bootstrap CI
                                           for mean slide-level Δ_ins.
    insertion_ci95_high                  — upper bound.

    Statistical methods
    -------------------
    When slide_groups is provided (recommended):
        Per-patch differences d_i are aggregated to slide-level means first,
        giving one observation Δ_s per slide.  The Wilcoxon signed-rank test
        and the bootstrap CI are both applied to {Δ_s}, so the estimand is
        identical for both statistics: the mean slide-level Δ.  The bootstrap
        resamples slides (IID, because slide-level means are independent
        observations), consistent with the slide-level bootstrap used in the
        training evaluation pipeline.

    When slide_groups is not provided:
        Both statistics are computed on raw per-patch differences.  CIs will
        be anti-conservative because within-slide patches are positively
        correlated.  Use only for exploratory analysis.

    Args:
        faith_summary: Output of compute_faithfulness_summary().
        output_path:   Full path to the output .csv file.

    Returns:
        output_path (for chaining).
    """
    # Ordered columns for the output table
    _DESCRIPTIVE_COLS = [
        "audc_gradcam_mean", "audc_gradcam_std",
        "auic_gradcam_mean", "auic_gradcam_std",
        "audc_random_mean",  "audc_random_std",
        "auic_random_mean",  "auic_random_std",
        "delta_deletion_mean",  "delta_deletion_std",
        "delta_insertion_mean", "delta_insertion_std",
        "delta_deletion_slide_mean",  "delta_deletion_slide_std",
        "delta_insertion_slide_mean", "delta_insertion_slide_std",
        "n_slides",
    ]
    _INFERENTIAL_COLS = [
        "deletion_wilcoxon_stat",  "deletion_wilcoxon_pvalue",
        "deletion_wilcoxon_pvalue_bonferroni",
        "deletion_wilcoxon_pvalue_bh",
        "deletion_ci95_low",       "deletion_ci95_high",
        "insertion_wilcoxon_stat", "insertion_wilcoxon_pvalue",
        "insertion_wilcoxon_pvalue_bonferroni",
        "insertion_wilcoxon_pvalue_bh",
        "insertion_ci95_low",      "insertion_ci95_high",
    ]
    _ALL_STAT_COLS = _DESCRIPTIVE_COLS + _INFERENTIAL_COLS

    def _row_from_stats(
        group_type: str,
        class_name: str,
        outcome: str,
        stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "group_type": group_type,
            "class":      class_name,
            "outcome":    outcome,
            "n_samples":  stats.get("n_samples", 0),
        }
        for col in _ALL_STAT_COLS:
            row[col] = stats.get(col, float("nan"))
        return row

    rows: List[Dict[str, Any]] = []

    # Overall row
    if "overall" in faith_summary and faith_summary["overall"]:
        rows.append(_row_from_stats(
            "overall", "all", "all", faith_summary["overall"]
        ))

    # Per-class rows
    for cname, stats in faith_summary.get("per_class", {}).items():
        rows.append(_row_from_stats("per_class", cname, "all", stats))

    # Per (class × outcome) rows — sorted for deterministic column order
    for key, stats in sorted(faith_summary.get("per_class_outcome", {}).items()):
        rows.append(_row_from_stats(
            "per_class_outcome",
            stats.get("class", key),
            stats.get("outcome", "unknown"),
            stats,
        ))

    df = pd.DataFrame(rows)

    # Enforce column order: metadata first, then stats
    ordered_cols = (
        ["group_type", "class", "outcome", "n_samples"] + _ALL_STAT_COLS
    )
    # Keep only columns that actually exist (guard against partial summaries)
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"  Group-level faithfulness stats CSV saved: {output_path}")
    return output_path
