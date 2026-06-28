"""
Grad-CAM XAI Analysis
==================================================================

Entry point for the complete Grad-CAM explainability pipeline. This script is
fully independent from the training pipeline (densenet_fungi.py). It requires:
    1. The saved final model checkpoint (.pth).
    2. The preprocessed NPY dataset directory.
    3. The split_indices.json file (shared across all experiments).

Pipeline phases
---------------
Phase 1  Grad-CAM inference
    - Compute Grad-CAM maps for all test patches (model.features.denseblock4).
    - Compute per-map shape descriptors: SCI (Gini analog) and Spatial Entropy.
    - Produce qualitative figures (exemplar grid, distribution plots, mean CAMs).

Phase 2  Perturbation-based faithfulness evaluation
    - Deletion curve (MoRF): mask pixels in descending saliency order; track
      how the model's predicted-class confidence drops. A faithful explanation
      causes a steep early drop -> low AUDC.
    - Insertion curve (MoRF-Insertion): reveal pixels in descending saliency
      order (most salient first) from a blank baseline; track confidence
      recovery. Faithful -> steep early rise -> high AUIC.
    - Random baseline: average over N_RANDOM_TRIALS random permutations to
      provide a chance-level reference for both curves.
    - Faithfulness gaps: delta_del = AUDC_random - AUDC_gradcam (up better);
                         delta_ins = AUIC_gradcam - AUIC_random  (up better).
    - Figures: mean curve panels (overall + per class) and AUC distribution
      violin plots.

Author: Rodrigo Sa
Date: 2025
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Absolute path to the .pth saved by densenet_fungi.py::run_final_training
    "model_path": "/content/drive/MyDrive/colab/denseNet121/checkpoints/"
                  "final_model_standard_cnn.pth",

    # Root directory of the preprocessed NPY dataset
    "preprocessed_dir": "/content/data/preprocessed",

    # Path to the shared split_indices.json (identical across all experiments)
    "split_indices_path": "/content/drive/MyDrive/colab/denseNet121/"
                          "preprocessed/split_indices.json",

    # Directory where all outputs will be written
    "output_dir": "/content/drive/MyDrive/colab/denseNet121/output/"
                  "gradcam_results",

    # Grad-CAM settings
    "exemplar_descriptor": "sci",   # descriptor for rank-1 exemplar selection
    "batch_size": 8,
    "num_workers": 2,
    "target_class": None,           # MUST be None for thesis runs.
                                    # If set to a fixed integer, Grad-CAM
                                    # explains that class but perturbation
                                    # tracks the *predicted* class confidence,
                                    # which invalidates AUDC/AUIC when the
                                    # fixed target != the model's prediction.
                                    # Use None (argmax per sample) always.
    "save_individual_triplets": False,

    # Faithfulness evaluation settings
    "run_faithfulness": True,
    "faithfulness_n_steps": 100,        # curve resolution (100 is standard)
    "faithfulness_n_random_trials": 10, # random permutations for baseline
    "faithfulness_baseline_value": 0.0, # fill value: 0.0 = training-mean (norm.)
    "faithfulness_max_samples": None,   # None = full test set; int = quick check
    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("GRAD-CAM XAI PIPELINE -- DenseNet-121 Fungal Microscopy")
    print("=" * 70 + "\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from gradcam_engine import GradCAM
    from gradcam_data_utils import (
        load_final_model,
        NPYPatchDataset,
        run_gradcam_inference,
        partition_by_class_and_correctness,
    )
    from gradcam_faithfulness import (
        run_perturbation_faithfulness,
        compute_faithfulness_summary,
        print_faithfulness_summary,
    )
    from gradcam_visualization import (
        plot_exemplar_grid,
        plot_descriptor_distributions,
        plot_mean_cam_per_class,
        plot_gradcam_triplet,
        save_descriptor_summary_csv,
        print_descriptor_statistics,
        plot_faithfulness_curves,
        plot_faithfulness_auc_distributions,
        save_faithfulness_summary_csv,
        save_faithfulness_group_stats_csv,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\n[1/8] Loading trained DenseNet-121 checkpoint...")
    model, meta = load_final_model(CONFIG["model_path"], device)
    mean         = meta["mean"]
    std          = meta["std"]
    class_to_idx = meta["class_to_idx"]
    class_names  = sorted(class_to_idx, key=class_to_idx.get)
    num_classes  = len(class_names)
    print(f"  Classes: {class_names}")
    print(f"  Normalization mean: {[f'{v:.4f}' for v in mean]}")
    print(f"  Normalization std:  {[f'{v:.4f}' for v in std]}")

    # Reconstruct test indices
    print("\n[2/8] Loading test split indices...")
    if not os.path.exists(CONFIG["split_indices_path"]):
        raise FileNotFoundError(
            f"split_indices.json not found: {CONFIG['split_indices_path']}\n"
            "This file is shared with all ViT experiments and must not be regenerated."
        )
    with open(CONFIG["split_indices_path"], "r") as f:
        split_data = json.load(f)

    test_base_paths: List[Path] = [
        Path(p) for p in split_data["test_base_paths"]
    ]

    _all_samples: list = []
    for class_name in sorted(os.listdir(CONFIG["preprocessed_dir"])):
        class_dir = os.path.join(CONFIG["preprocessed_dir"], class_name)
        if not os.path.isdir(class_dir):
            continue
        label = class_to_idx.get(class_name)
        if label is None:
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(".npy"):
                _all_samples.append((os.path.join(class_dir, fname), label))

    npy_base_to_idx = {}
    for idx, (spath, _) in enumerate(_all_samples):
        rel  = Path(spath).relative_to(CONFIG["preprocessed_dir"])
        base = rel.with_suffix("")
        npy_base_to_idx[base.as_posix()] = idx

    test_idx = []
    for base in test_base_paths:
        key = base.as_posix()
        if key not in npy_base_to_idx:
            warnings.warn(f"Test path not found in dataset: {key}")
            continue
        test_idx.append(npy_base_to_idx[key])

    print(f"  Test set size: {len(test_idx)} patches")

    # Build dataset
    print("\n[3/8] Constructing test dataset (NPYPatchDataset)...")
    dataset = NPYPatchDataset(
        preprocessed_dir=CONFIG["preprocessed_dir"],
        indices=test_idx,
        mean=mean,
        std=std,
        class_to_idx=class_to_idx,
    )
    print(f"  Dataset length: {len(dataset)}")

    # Target layer + Grad-CAM inference
    # DenseNet-121 final convolutional block: model.features.denseblock4.
    # Deepest layer before GAP -- best semantic/spatial trade-off.
    target_layer = model.features.denseblock4
    print(
        "\n  Target layer: model.features.denseblock4\n"
        "  (last DenseBlock before GAP -- Selvaraju et al. 2020, sec. 3)"
    )

    if CONFIG["target_class"] is not None:
        raise ValueError(
            "CONFIG['target_class'] must be None for thesis runs. "
            "A fixed target class decouples the Grad-CAM explanation from the "
            "perturbation confidence tracking and invalidates AUDC/AUIC. "
            "See the CONFIG comment for a full explanation."
        )

    print(
        f"\nRunning Grad-CAM inference...\n"
        f"  Target class: argmax per sample (None enforced)"
    )

    results = run_gradcam_inference(
        model=model,
        dataset=dataset,
        target_layer=target_layer,
        device=device,
        target_class=CONFIG["target_class"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    n_total   = len(results["true_labels"])
    n_correct = int((results["true_labels"] == results["predicted_classes"]).sum())
    print(
        f"  Inference complete. Accuracy: {n_correct}/{n_total} "
        f"({100 * n_correct / n_total:.1f}%)"
    )
    print_descriptor_statistics(results, class_names)

    # Derive slide identifiers from patch file paths.
    # The extraction logic mirrors densenet_fungi.py::extract_slide_id_from_base
    # exactly so that grouping is identical across training and XAI pipelines.
    def _extract_slide_id(path: str) -> str:
        stem  = Path(path).stem
        parts = stem.split("_")
        if len(parts) < 3:
            return stem
        return f"{parts[0]}_{parts[1]}_{parts[2]}"

    test_slide_groups = np.array(
        [_extract_slide_id(p) for p in results["paths"]], dtype=object
    )
    n_slides = len(np.unique(test_slide_groups))
    print(f"  Slide groups: {n_total} patches across {n_slides} slides")

    partitions = partition_by_class_and_correctness(results, class_names)

    # Qualitative figures
    print("\n[5/8] Generating qualitative Grad-CAM figures...")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    plot_exemplar_grid(
        partitions=partitions,
        dataset=dataset,
        class_names=class_names,
        output_dir=CONFIG["output_dir"],
        descriptor_key=CONFIG["exemplar_descriptor"],
        save_name="gradcam_exemplar_grid.png",
        show=False,
    )

    plot_descriptor_distributions(
        results=results,
        class_names=class_names,
        output_dir=CONFIG["output_dir"],
        save_name="gradcam_descriptor_distributions.png",
        show=False,
    )

    plot_mean_cam_per_class(
        partitions=partitions,
        class_names=class_names,
        output_dir=CONFIG["output_dir"],
        save_name="gradcam_mean_cam_per_class.png",
        show=False,
    )

    if CONFIG["save_individual_triplets"]:
        triplet_dir = os.path.join(CONFIG["output_dir"], "gradcam_triplets")
        os.makedirs(triplet_dir, exist_ok=True)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"\n  Saving {n_total} triplet figures to {triplet_dir} ...")
        for i in range(n_total):
            raw_img    = dataset.get_raw_image(i)
            cam        = results["cams"][i]
            true_lbl   = idx_to_class[int(results["true_labels"][i])]
            pred_lbl   = idx_to_class[int(results["predicted_classes"][i])]
            descs      = results["descriptors"][i]
            patch_stem = Path(results["paths"][i]).stem
            plot_gradcam_triplet(
                raw_image=raw_img,
                cam=cam,
                true_label=true_lbl,
                predicted_label=pred_lbl,
                descriptors=descs,
                save_path=os.path.join(triplet_dir, f"{patch_stem}_gradcam.png"),
                show=False,
                title_prefix=f"[{i+1}/{n_total}]",
            )

    # Descriptor CSV
    print("\n[6/8] Saving descriptor summary CSV...")
    save_descriptor_summary_csv(
        results, class_names,
        os.path.join(CONFIG["output_dir"], "gradcam_descriptors.csv"),
    )

    # Perturbation faithfulness
    if not CONFIG["run_faithfulness"]:
        print("\n  Skipping faithfulness evaluation (run_faithfulness=False).")
        _print_completion(CONFIG["output_dir"])
        return

    print("\nRunning perturbation-based faithfulness evaluation...")
    print(
        f"  Protocol : deletion (MoRF) + insertion (MoRF-Insertion) vs. random baseline\n"
        f"  Steps    : {CONFIG['faithfulness_n_steps']}\n"
        f"  Trials   : {CONFIG['faithfulness_n_random_trials']} random permutations\n"
        f"  Baseline : {CONFIG['faithfulness_baseline_value']} "
        f"(= training-set mean in normalized space)\n"
        f"  Samples  : "
        f"{CONFIG['faithfulness_max_samples'] or 'all (' + str(n_total) + ')'}\n"
    )

    faith_results = run_perturbation_faithfulness(
        model=model,
        dataset=dataset,
        results=results,
        device=device,
        n_steps=CONFIG["faithfulness_n_steps"],
        n_random_trials=CONFIG["faithfulness_n_random_trials"],
        baseline_value=CONFIG["faithfulness_baseline_value"],
        rng_seed=CONFIG["random_seed"],
        max_samples=CONFIG["faithfulness_max_samples"],
        slide_groups=test_slide_groups,
    )

    faith_summary = compute_faithfulness_summary(faith_results, class_names)
    print_faithfulness_summary(faith_summary)

    print("\nGenerating faithfulness figures...")
    plot_faithfulness_curves(
        faith_results=faith_results,
        class_names=class_names,
        output_dir=CONFIG["output_dir"],
        save_name="gradcam_faithfulness_curves.png",
        show=False,
    )

    plot_faithfulness_auc_distributions(
        faith_results=faith_results,
        class_names=class_names,
        output_dir=CONFIG["output_dir"],
        save_name="gradcam_faithfulness_auc_distributions.png",
        show=False,
    )

    save_faithfulness_summary_csv(
        faith_results, class_names,
        os.path.join(CONFIG["output_dir"], "gradcam_faithfulness.csv"),
    )

    save_faithfulness_group_stats_csv(
        faith_summary,
        os.path.join(CONFIG["output_dir"], "gradcam_faithfulness_group_stats.csv"),
    )

    _print_completion(CONFIG["output_dir"])


def _print_completion(output_dir: str) -> None:
    print("\n" + "=" * 70)
    print("GRAD-CAM XAI PIPELINE COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
