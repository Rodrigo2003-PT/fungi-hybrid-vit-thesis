"""
DenseNet-121 for Fungal Microscopy

Author: Rodrigo Sá
Date: 2025

"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import densenet121


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 adapted for single-channel fungal microscopy classification.

    Architectural modifications from the canonical ImageNet DenseNet-121:
    1.  Input layer: features.conv0 replaced with nn.Conv2d(1→64, k=7, s=2, p=3,
        bias=False) to handle grayscale input.
    2.  Classifier head: replaced with nn.Linear(1024, num_classes) for
        binary (or multi-class) fungal prediction.
    3.  All parameters initialized from scratch under the policy in
        _initialize_weights (He et al. for Conv2d, canonical identity for
        BatchNorm, truncated normal std=0.02 for Linear head). No ImageNet
        weights are loaded at any point.
    """

    def __init__(self, num_classes: int = 2, channels: int = 1):
        """
        Args:
            num_classes: Number of output classes
            channels: Must be 1 for grayscale
        """
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if channels != 1:
            raise ValueError(
                f"This baseline is designed for single-channel (grayscale) input. "
                f"Got channels={channels}."
            )

        self.num_classes = num_classes
        self.channels = channels

        # 1. Load torchvision DenseNet-121 architecture without weights.
        backbone = densenet121(weights=None)

        # 2. Replace features.conv0 for single-channel input
        original_conv0 = backbone.features.conv0
        backbone.features.conv0 = nn.Conv2d(
            in_channels=channels,
            out_channels=original_conv0.out_channels,   # 64
            kernel_size=original_conv0.kernel_size,     # (7, 7)
            stride=original_conv0.stride,               # (2, 2)
            padding=original_conv0.padding,             # (3, 3)
            bias=False,                                
        )

        # 3. Replace classifier head
        in_features = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Linear(in_features, num_classes)

        self.features = backbone.features
        self.classifier = backbone.classifier

        # 4. Apply the complete from-scratch initialization policy
        self._initialize_weights()

    # ---------------------------------------------------------------------- 
    # Forward pass                                                            
    # ---------------------------------------------------------------------- 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, H, W), float32, normalized.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.features(x)                         # (B, 1024, h', w')
        # ReLU before GAP follows torchvision's DenseNet forward exactly.
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))  # (B, 1024, 1, 1)
        out = torch.flatten(out, 1)                         # (B, 1024)
        out = self.classifier(out)                          # (B, num_classes)
        return out

    # ---------------------------------------------------------------------- 
    # Weight initialization                                                   
    # ---------------------------------------------------------------------- 

    def _initialize_weights(self) -> None:
        """

        nn.Conv2d   → Kaiming Normal, fan_out, nonlinearity='relu'  [He+2015]
                      Rationale: All DenseNet Conv2d layers are followed by
                      ReLU (via BN→ReLU in dense blocks). fan_out is preferred
                      for deep networks because it preserves gradient magnitude
                      in the backward pass (He et al., 2015 §2.2).
                      Bias: not applicable — all DenseNet Conv2d have bias=False.

        nn.BatchNorm2d → γ=1.0, β=0.0, running_mean=0, running_var=1,
                         num_batches_tracked=0  [Ioffe & Szegedy 2015]
                         Rationale: Identity transform at initialization.
                         Specified explicitly rather than relying on PyTorch
                         defaults to guarantee reproducibility across versions.

        nn.Linear (head) → Truncated Normal std=0.02, bias=0.0
                           Rationale: Matches the classifier head initialization
                           used in the ViT baseline and the Hassani et al.
                           ConvViT implementation (Hassani et al. 2021 §3.2),
                           making the final classification mapping comparable
                           across all architectures in this study. Truncated at
                           ±2σ to prevent extreme initializations.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming Normal with fan_out for ReLU networks.
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                # Bias is always False for DenseNet Conv2d;
                if module.bias is not None:
                    # Defensive: should never trigger in DenseNet-121.
                    warnings.warn(
                        f"Unexpected bias in Conv2d layer '{name}'. "
                        f"Initializing to zero."
                    )
                    nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.BatchNorm2d):
                # Identity transform at init.
                nn.init.constant_(module.weight, 1.0)   # γ = 1
                nn.init.constant_(module.bias, 0.0)     # β = 0
                # Running statistics: explicit reset (not relying on defaults).
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
                module.num_batches_tracked.zero_()

            elif isinstance(module, nn.Linear):
                # Truncated Normal std=0.02 matching ViT and ConvViT heads.
                # Truncation at ±2σ (PyTorch trunc_normal_ default).
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    # ----------------------------------------------------------------------
    # Parameter statistics
    # ---------------------------------------------------------------------- 

    def count_parameters(self) -> dict:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# --------------------------------------------------------------------------- #
# Sanity check: verify architecture produces correct output shape             #
# (run as a standalone script during development; not executed in training)   #
# --------------------------------------------------------------------------- #

def _architecture_sanity_check():
    import sys

    model = DenseNet121Classifier(num_classes=2, channels=1)
    model.eval()

    # Shape check
    dummy = torch.zeros(2, 1, 384, 384)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 2), f"Expected (2, 2) output, got {out.shape}"
    print(f"  Output shape: {out.shape}  ✓")

    # Parameter count
    counts = model.count_parameters()
    print(f"  Total parameters:     {counts['total']:,}")
    print(f"  Trainable parameters: {counts['trainable']:,}")
    # DenseNet-121 with 2-class head and 1-channel input ≈ 6.96M params
    # Tolerance: ±50K to allow for minor torchvision version differences
    expected = 6_956_298  # canonical value;
    if abs(counts['total'] - expected) > 100_000:
        print(
            f"  WARNING: Parameter count {counts['total']:,} deviates from "
            f"expected {expected:,}. Check torchvision version."
        )
    else:
        print(f"  Parameter count within expected range of {expected:,}  ✓")

    # Verify no NaN/Inf in initialized weights
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  FAIL: NaN/Inf in parameter '{name}'")
            sys.exit(1)
    print("  No NaN/Inf in initialized weights  ✓")

    # Verify input channel adaptation
    assert model.features.conv0.in_channels == 1, \
        "features.conv0 must accept 1 input channel"
    print("  Input channel adaptation correct (1-channel)  ✓")

    # Verify classifier head dimensions
    assert model.classifier.out_features == 2, \
        "Classifier head must output 2 logits"
    print("  Classifier head output dimension correct  ✓")

    print("Sanity check passed.\n")


if __name__ == "__main__":
    _architecture_sanity_check()
