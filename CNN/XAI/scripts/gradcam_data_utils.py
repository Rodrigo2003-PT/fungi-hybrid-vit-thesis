"""
Grad-CAM Data Utilities

Handles loading of the trained DenseNet-121 checkpoint, construction of the
NPY-based test dataset, per-sample inference with Grad-CAM, and result
aggregation for downstream visualization.  All preprocessing exactly mirrors
the evaluation phase of densenet_fungi.py so that no distribution shift is
introduced between training and XAI analysis.

Author: Rodrigo Sá
Date: 2025
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gradcam_engine import GradCAM, compute_map_descriptors


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_final_model(
    model_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load the trained DenseNet-121 from a .pth checkpoint saved by
    densenet_fungi.py::run_final_training().

    The checkpoint is expected to contain:
        'model_state_dict', 'config', 'mean', 'std',
        'class_to_idx', 'metadata'.

    Args:
        model_path: Absolute path to the final model .pth file.
        device:     Device on which to load and run the model.

    Returns:
        model:    DenseNet121Classifier in eval() mode.
        meta:     Dictionary with keys 'mean', 'std', 'class_to_idx',
                  'config', 'metadata'.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Import here to keep this file independent of the training path
    from densenet_model import DenseNet121Classifier

    ckpt = torch.load(model_path, map_location=device)

    config = ckpt["config"]
    model_kwargs = {
        k: v
        for k, v in config["model_config"].items()
        if k != "num_classes"
    }
    model = DenseNet121Classifier(
        num_classes=config["model_config"]["num_classes"],
        **model_kwargs,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "mean":         ckpt["mean"],
        "std":          ckpt["std"],
        "class_to_idx": ckpt["class_to_idx"],
        "config":       config,
        "metadata":     ckpt.get("metadata", {}),
    }

    print(
        f"  Loaded DenseNet121Classifier from {os.path.basename(model_path)}"
    )
    counts = model.count_parameters()
    print(
        f"  Parameters: {counts['total']:,} total, "
        f"{counts['trainable']:,} trainable"
    )
    return model, meta


# ---------------------------------------------------------------------------
# Minimal NPY dataset (mirrors NPYImageFolder from training_utils)
# ---------------------------------------------------------------------------

class NPYPatchDataset(Dataset):
    """
    Lightweight Dataset that reads .npy patch files from a preprocessed
    directory.  Applies the same deterministic (val/test) transform used
    during model evaluation — Normalize with the training-set statistics
    stored in the checkpoint.

    Only the test split is used for Grad-CAM analysis, consistent with the
    evaluation phase in densenet_fungi.py.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        indices: List[int],
        mean: List[float],
        std: List[float],
        class_to_idx: Dict[str, int],
    ):
        """
        Args:
            preprocessed_dir: Root directory of the NPY dataset
                               (same as `preprocessed_dir` in training).
            indices:          Global dataset indices for the test split.
            mean:             Per-channel mean from training set.
            std:              Per-channel std from training set.
            class_to_idx:     Class-name → integer-label mapping.
        """
        # Reconstruct samples list by scanning the preprocessed_dir,
        # mirroring NPYImageFolder in training_utils.py
        self._samples: List[Tuple[str, int]] = []
        idx_to_str = {v: k for k, v in class_to_idx.items()}

        # Walk directory in sorted order to match NPYImageFolder indexing
        for class_name in sorted(os.listdir(preprocessed_dir)):
            class_dir = os.path.join(preprocessed_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = class_to_idx.get(class_name)
            if label is None:
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".npy"):
                    self._samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

        # Subset to the requested indices
        self._subset = [(self._samples[i][0], self._samples[i][1])
                        for i in indices]

        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Store original paths and labels in order for later retrieval
        self.paths  = [p for p, _ in self._subset]
        self.labels = [l for _, l in self._subset]

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        path, label = self._subset[i]
        arr = np.load(path)                         # (H, W) or (H, W, C)

        # Ensure shape (C, H, W) as float32 in [0, 1]
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]             # (1, H, W)
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)            # (C, H, W)
        else:
            raise RuntimeError(
                f"Unexpected array shape {arr.shape} for {path}"
            )

        arr = arr.astype(np.float32)

        # Defensive clip — mirrors NPYImageFolder in training_utils.py.
        # Preprocessed .npy files should lie in [0, 1], but bilinear
        # interpolation during resizing can introduce values slightly outside
        # this range (e.g. 1.002 or -0.001). The training evaluation pipeline
        # clips these before normalization; not doing so here would introduce
        # a distribution shift between the model's training context and the
        # XAI inference context, violating the reproducibility requirement.
        if arr.min() < -0.01 or arr.max() > 1.01:
            arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr)

        # Normalize: (tensor - mean) / std, matching create_transforms(train=False)
        mean_t = torch.as_tensor(self.mean[:, None, None])
        std_t  = torch.as_tensor(self.std[:, None, None])
        tensor = (tensor - mean_t) / (std_t + 1e-8)

        return tensor, label

    def get_raw_image(self, i: int) -> np.ndarray:
        """
        Return the raw (un-normalized) patch as a float32 array in [0, 1],
        shape (H, W) for single-channel grayscale.  Used only for overlaying
        the CAM heatmap during visualization.
        """
        path, _ = self._subset[i]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]                     # take first channel
        # Normalize to [0, 1] if stored as integer values
        if arr.max() > 1.0:
            arr = arr / arr.max()
        return arr


# ---------------------------------------------------------------------------
# Batch-level Grad-CAM inference
# ---------------------------------------------------------------------------

def run_gradcam_inference(
    model: nn.Module,
    dataset: NPYPatchDataset,
    target_layer: nn.Module,
    device: torch.device,
    target_class: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Run Grad-CAM on the entire dataset and collect results.

    Each sample is processed individually through the Grad-CAM backward pass
    (batch_size > 1 is used for the forward pass only; the gradient signal is
    computed per-sample as required by Selvaraju et al. §3).

    DataLoader settings
    -------------------
    When num_workers > 0, persistent_workers=True is set so that worker
    processes are not respawned between batches, eliminating fork overhead
    on Colab/Linux.  pin_memory is enabled automatically when device is CUDA,
    overlapping host→device DMA with GPU compute.

    Mixed-precision inference (AMP)
    --------------------------------
    The fp16/fp32 AMP boundary is managed entirely inside GradCAM.__call__,
    not at this level.  Specifically:

      - The forward pass (model(inputs)) executes under torch.autocast(float16),
        letting the T4's Tensor Cores accelerate DenseNet's dense convolutions.

      - Immediately after the forward pass, logits and hook-captured activations
        are explicitly cast to fp32 via .float() before any backward work begins.

      - score.backward() and all CAM construction (alpha computation, weighted
        sum, ReLU, bilinear upsample) run in pure fp32.

    This design guarantees that the Grad-CAM maps are numerically identical to
    those produced by the original fp32-only implementation.  The fp32 cast of
    logits before backward() is the critical invariant: it prevents fp16
    overflow in the scalar score (DenseNet-121 logits can exceed the fp16
    dynamic range of ±65504) and ensures that gradient magnitudes and spatial
    structure are independent of batch size and autocast context.

    No autocast context is opened at this (run_gradcam_inference) level.
    Wrapping the GradCAM.__call__ in an outer autocast here would silently
    override the internal fp32 boundary and push backward computation into
    fp16, invalidating the maps.

    Args:
        model:        Trained DenseNet121Classifier in eval() mode.
        dataset:      NPYPatchDataset covering the test split.
        target_layer: The convolutional layer from which to extract Grad-CAM.
        device:       Inference device.
        target_class: Fixed target class index, or None for argmax.
        batch_size:   How many samples to forward-pass simultaneously.
        num_workers:  DataLoader worker count.

    Returns:
        results dict with keys:
            'cams'             : List[np.ndarray (H, W)] — one normalized CAM per sample.
            'predicted_classes': np.ndarray (N,) — argmax predictions.
            'true_labels'      : np.ndarray (N,) — ground-truth labels.
            'descriptors'      : List[Dict]       — SCI, entropy, etc. per sample.
            'paths'            : List[str]        — file paths.
    """
    gradcam = GradCAM(model, target_layer)

    _pin = device.type == "cuda"
    _persistent = (num_workers > 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=_persistent,
    )

    all_cams:      List[np.ndarray] = []
    all_preds:     List[int]        = []
    all_labels:    List[int]        = []
    all_descs:     List[Dict]       = []

    sample_offset = 0
    model.eval()

    for batch_inputs, batch_labels in loader:
        batch_inputs = batch_inputs.to(device, non_blocking=_pin)
        B = batch_inputs.size(0)

        # The AMP fp16/fp32 boundary is managed inside GradCAM.__call__:
        #   - forward pass runs under fp16 autocast
        #   - logits and hook-captured activations are explicitly cast to fp32
        #     before any backward computation
        # Wrapping this call in an outer autocast context here would override
        # the internal boundary and push the backward pass into fp16, which is
        # incorrect.  No autocast context is used at this level.
        batch_cams, batch_preds = gradcam(
            batch_inputs, target_class=target_class
        )

        for b in range(B):
            cam = batch_cams[b]                     # (H, W) in [0, 1]
            desc = compute_map_descriptors(cam)
            all_cams.append(cam)
            all_descs.append(desc)

        all_preds.extend(batch_preds.tolist())
        all_labels.extend(batch_labels.numpy().tolist())
        sample_offset += B

    gradcam.remove_hooks()

    return {
        "cams":              all_cams,
        "predicted_classes": np.array(all_preds, dtype=np.int64),
        "true_labels":       np.array(all_labels, dtype=np.int64),
        "descriptors":       all_descs,
        "paths":             dataset.paths,
    }


# ---------------------------------------------------------------------------
# Per-class group extraction
# ---------------------------------------------------------------------------

def partition_by_class_and_correctness(
    results: Dict[str, Any],
    class_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Partition Grad-CAM results by (true_class, correct/incorrect).

    Returns a nested dictionary:
        {
          class_name: {
              'correct':   {'cams': [...], 'indices': [...], 'descriptors': [...]},
              'incorrect': {'cams': [...], 'indices': [...], 'descriptors': [...]},
          },
          ...
        }

    This structure mirrors the stratified Attention Rollout analysis in the
    ViT baseline, enabling fair comparison across XAI methods.
    """
    partitions: Dict[str, Dict[str, Any]] = {}
    for cname in class_names:
        partitions[cname] = {
            "correct":   {"cams": [], "indices": [], "descriptors": []},
            "incorrect": {"cams": [], "indices": [], "descriptors": []},
        }

    y_true = results["true_labels"]
    y_pred = results["predicted_classes"]
    cams   = results["cams"]
    descs  = results["descriptors"]
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    for i in range(len(y_true)):
        cname = idx_to_class.get(int(y_true[i]))
        if cname is None:
            warnings.warn(
                f"Label {y_true[i]} not found in class_names; skipping."
            )
            continue
        key = "correct" if y_true[i] == y_pred[i] else "incorrect"
        partitions[cname][key]["cams"].append(cams[i])
        partitions[cname][key]["indices"].append(i)
        partitions[cname][key]["descriptors"].append(descs[i])

    return partitions
