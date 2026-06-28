"""
Grad-CAM Engine for DenseNet-121 Fungal Microscopy Classifier

Implements Gradient-weighted Class Activation Mapping (Grad-CAM)

For DenseNet-121, the target layer is the final convolutional block
(model.features.denseblock4), producing a (B, 1024, h', w') activation tensor
at 12×12 spatial resolution for 384×384 inputs. The ReLU in Eq. (2) of the
paper retains only positively contributing regions, as recommended by ablation
results in Table 3 of the Grad-CAM paper.

Author: Rodrigo Sá
Date: 2025
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hook-based Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for DenseNet-121 via forward/backward hooks.

    The forward hook captures the activation tensor A^k (shape B×K×h×w) from
    the target convolutional layer. The backward hook captures the gradient
    ∂y^c / ∂A^k flowing from the classification score for class c back to
    that same layer. The neuron importance weights are then computed by
    global-average-pooling the gradient maps (Eq. 1), and the final
    localization map is obtained via a weighted combination followed by a
    ReLU (Eq. 2), as specified in Selvaraju et al. (2020).

    The class explicitly supports:
        - Arbitrary target class (or argmax prediction if unspecified).
        - Batch inference (one map returned per sample).
        - Bilinear upsampling to the input spatial resolution.
        - Proper hook cleanup to avoid memory leaks across calls.

    Usage::
        gradcam = GradCAM(model, target_layer=model.features.denseblock4)
        cam, pred_class = gradcam(inputs, target_class=None)  # argmax
        cam, pred_class = gradcam(inputs, target_class=0)     # class 0
        gradcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model:        Trained DenseNet121Classifier in eval mode.
            target_layer: The convolutional layer from which to extract
                          Grad-CAM maps. For DenseNet-121, this is
                          model.features.denseblock4 (last dense block).
        """
        self.model = model
        self.target_layer = target_layer

        # Storage populated by hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register persistent hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ---------------------------------------------------------------------- #
    # Hook callbacks                                                          #
    # ---------------------------------------------------------------------- #

    def _save_activation(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook: store A^k detached to avoid holding onto the graph."""
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple,
        grad_output: Tuple,
    ) -> None:
        """Backward hook: store ∂y^c / ∂A^k from the first grad_output."""
        self._gradients = grad_output[0].detach()

    # ---------------------------------------------------------------------- #
    # Core Grad-CAM computation                                               #
    # ---------------------------------------------------------------------- #

    def __call__(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM localization maps for a batch of inputs.

        Steps (per Selvaraju et al., 2020, §3):
          1. Forward pass → obtain logits y^c.
          2. Zero all class gradients except c (set to 1); backpropagate.
          3. Compute α^c_k = GAP_ij (∂y^c / ∂A^k_ij)  [Eq. 1].
          4. L^c_Grad-CAM = ReLU(Σ_k α^c_k · A^k)      [Eq. 2].
          5. Bilinear upsample L^c to input spatial resolution.
          6. Normalize each map independently to [0, 1].

        Args:
            inputs:       Tensor of shape (B, C, H, W), already on the
                          correct device, normalized as during training.
            target_class: Integer class index (0-based) for which to
                          compute the CAM.  If None, uses argmax(y^c)
                          independently per sample.

        Returns:
            cams:          Float32 ndarray of shape (B, H, W), values ∈ [0,1].
            predicted_classes: Int ndarray of shape (B,) with predicted class
                               indices (argmax, regardless of target_class).

        AMP precision boundary
        ----------------------
        The forward pass is executed inside a torch.autocast(float16) context
        when a CUDA device is present, letting the T4's Tensor Cores accelerate
        DenseNet's dense convolutions.

        The boundary is drawn *between* the forward and backward passes.
        Concretely:

          (a) Forward (inside autocast):
                logits_fp16 = model(inputs)          # fp16 activations + weights
                activations_fp16 stored by fwd hook  # fp16

          (b) Explicit fp32 cast before any backward work:
                logits_fp32 = logits_fp16.float()
                activations saved by hook are also cast to fp32

          (c) Backward (outside autocast, pure fp32):
                score = logits_fp32[b, cls]          # scalar fp32 node
                score.backward()                     # gradients in fp32
                alpha = gradients_fp32.mean(...)     # fp32
                cam   = relu(alpha @ activations_fp32) # fp32
        """
        if inputs.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (B, C, H, W), got shape {inputs.shape}."
            )

        B, C, H, W = inputs.shape
        device = inputs.device
        _use_amp = (device.type == "cuda")

        # ---- 1. Forward pass (inside fp16 autocast on CUDA) ----
        # The computational graph is retained for the backward pass.
        # Activations captured by the forward hook are fp16 at this point.
        self.model.eval()
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16,
                            enabled=_use_amp):
            logits_raw = self.model(inputs)          # (B, num_classes), fp16 on CUDA

        # ---- fp32 rescue: cast logits and stored activations to fp32 --------
        # This is the hard boundary between the fp16 forward and the fp32
        # backward.  All subsequent tensor operations are in fp32, which
        # guarantees that gradient computation and CAM construction are
        # numerically identical to the original fp32-only implementation.
        #
        # logits_fp32 retains its gradient connection to the computation graph
        # (it was built by a .float() call on an fp16 leaf, not a detach).
        logits_fp32 = logits_raw.float()             # (B, num_classes), fp32
        predicted_classes = logits_fp32.argmax(dim=1).cpu().numpy()  # (B,)

        # Cast the hook-captured activations to fp32 in-place so that the
        # weighted combination (step 4) and all subsequent math are in fp32.
        if self._activations is not None:
            self._activations = self._activations.float()

        # ---- 2-6. Per-sample backward + CAM construction (pure fp32) --------
        cams = []
        for b in range(B):
            cls = target_class if target_class is not None else int(predicted_classes[b])

            self.model.zero_grad()

            # Scalar score for class cls (pre-softmax), as required by
            # Selvaraju et al.: "the score for class c, y^c (before softmax)".
            # logits_fp32[b, cls] is a proper fp32 scalar with a grad_fn that
            # traces back through the fp16 forward graph — PyTorch's autocast
            # scales gradients automatically via GradScaler-compatible paths,
            # but here we do NOT use a GradScaler because we are not training;
            # we call backward() directly on the fp32 scalar.  This is safe
            # because we only need the *direction* of the gradient, not an
            # unbiased estimate for parameter updates.
            score = logits_fp32[b, cls]
            score.backward(retain_graph=(b < B - 1))

            # ---- 3. Neuron importance weights α^c_k = GAP(∂y^c / ∂A^k) ----
            # _gradients populated by the backward hook.
            # Cast to fp32: the backward hook fires during the fp16→fp32
            # backward pass, so gradients may still arrive as fp16 tensors
            # depending on PyTorch version and autocast interaction.
            grads = self._gradients[b].float()   # (K, h, w), guaranteed fp32
            acts  = self._activations[b]         # (K, h, w), already fp32

            # Eq. (1): α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
            alpha = grads.mean(dim=(1, 2))       # (K,)

            # ---- 4. Weighted combination + ReLU  [Eq. 2] ----
            weighted = (alpha[:, None, None] * acts).sum(dim=0)  # (h, w), fp32
            cam = F.relu(weighted)                                # (h, w), fp32

            # ---- 5. Bilinear upsample to input resolution ----
            cam_up = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),    # (1, 1, h, w)
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()                           # (H, W), fp32

            # ---- 6. Min-max normalize to [0, 1] ----
            cam_np = cam_up.cpu().numpy()         # already float32
            c_min, c_max = cam_np.min(), cam_np.max()
            if c_max - c_min > 1e-8:
                cam_np = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_np = np.zeros_like(cam_np)

            cams.append(cam_np)

        return np.stack(cams, axis=0), predicted_classes   # (B, H, W), (B,)

    # ---------------------------------------------------------------------- #
    # Cleanup                                                                 #
    # ---------------------------------------------------------------------- #

    def remove_hooks(self) -> None:
        """Deregister hooks to free memory. Must be called when done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Map quality descriptors
# ---------------------------------------------------------------------------

def compute_map_descriptors(cam: np.ndarray) -> Dict[str, float]:
    """
    Compute quantitative descriptors for a single Grad-CAM map.

    Two complementary descriptors are computed to characterise the spatial
    structure of the localization map, directly analogous to the metrics used
    for Attention Rollout in the ViT baseline:

    Spatial Concentration Index (SCI) — analogous to Gini Coefficient (G)
    -----------------------------------------------------------------------
    Measures how *concentrated* (sparse) the activation is:

        SCI = (2 * Σ_{i=1}^{n} i * p_i) / (n * Σ p_i) - (n+1)/n

    where {p_i} are the sorted-ascending, normalized pixel values and n is the
    number of pixels. SCI ∈ [0, 1]; higher → more concentrated / sparse.

    Unlike the Gini coefficient for attention weights, Grad-CAM maps are not
    intrinsically normalized probability distributions; instead we first
    compute them as continuous activation values in [0,1] and define the
    concentration over that support. This preserves comparability with the
    ViT Gini metric while being appropriate for the continuous Grad-CAM output.

    Spatial Entropy (H) — directly analogous to ViT baseline
    ----------------------------------------------------------
    Measures how *diffuse* (spread) the activation is, treating the
    normalized map as a discrete probability distribution:

        p_i = v_i / Σ v_i,    H = -Σ p_i log(p_i + ε)

    H is large when activation is uniform (diffuse) and small when
    concentrated (sparse). ε = 1e-10 prevents log(0).

    Relationship: SCI and H are negatively correlated by construction;
    reporting both follows the dual-metric framework of the ViT analysis.

    Additional lightweight descriptors:
        active_fraction: fraction of pixels with CAM > 0.5 (activation density).
        peak_response:   maximum activation value after normalization (should be
                         1.0 post-normalization but useful as a sanity check).

    Args:
        cam: 2-D float32 array in [0, 1], shape (H, W).

    Returns:
        Dictionary with keys: 'sci', 'entropy', 'active_fraction',
        'peak_response'.
    """
    if cam.ndim != 2:
        raise ValueError(f"Expected 2-D cam, got shape {cam.shape}.")

    flat = cam.flatten().astype(np.float64)
    n = len(flat)

    # --- Spatial Concentration Index (SCI, Gini analog) ---
    sorted_vals = np.sort(flat)                         # ascending
    indices = np.arange(1, n + 1, dtype=np.float64)
    s = sorted_vals.sum()
    if s < 1e-12:
        sci = 0.0
    else:
        sci = float(
            (2.0 * (indices * sorted_vals).sum()) / (n * s) - (n + 1) / n
        )
    sci = float(np.clip(sci, 0.0, 1.0))

    # --- Spatial Entropy ---
    total = flat.sum()
    if total < 1e-12:
        entropy = 0.0
    else:
        probs = flat / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    # --- Active fraction (threshold τ = 0.5) ---
    active_fraction = float((flat > 0.5).mean())

    # --- Peak response ---
    peak_response = float(flat.max())

    return {
        "sci": sci,
        "entropy": entropy,
        "active_fraction": active_fraction,
        "peak_response": peak_response,
    }


def select_rank1_exemplar(
    cams: np.ndarray,
    descriptor_key: str = "sci",
) -> int:
    """
    Select the rank-1 exemplar within a group of Grad-CAM maps.

    Strategy (mirrors the ViT Attention Rollout exemplar selection):
        Compute the chosen descriptor for each map in the group, then return
        the index of the sample whose descriptor value is *closest* to the
        group median (i.e., minimises |descriptor_i - median(descriptors)|).

    This ensures the selected exemplar is representative of the group's
    central tendency rather than an outlier, providing a principled and
    reproducible visualization.

    Args:
        cams:           Float32 array, shape (N, H, W).
        descriptor_key: Which descriptor to use; default 'sci' (Gini analog).

    Returns:
        Index i* = argmin_i |descriptor_i - median(descriptors)|.
    """
    if cams.ndim != 3 or cams.shape[0] == 0:
        raise ValueError(
            f"Expected non-empty 3-D array (N, H, W), got {cams.shape}."
        )

    descriptors = np.array([
        compute_map_descriptors(cams[i])[descriptor_key]
        for i in range(len(cams))
    ])
    median_val = np.median(descriptors)
    distances = np.abs(descriptors - median_val)
    return int(np.argmin(distances))
