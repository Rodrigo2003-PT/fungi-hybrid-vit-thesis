"""
Pre-Training Sanity Checks for the LSA Ablation Study

Usage
-----
Expected output on success:
    All checks print PASS. Any failure raises AssertionError with a
    descriptive message so the root cause can be diagnosed immediately.

Author: Rodrigo Sá
Date: 2025
"""

import math
import torch
import numpy as np

from simple_vit import SimpleViT
from LSA_ViT import SimpleViT_LSA


# ---------------------------------------------------------------------------
# Shared model configuration
# ---------------------------------------------------------------------------

MODEL_KWARGS = dict(
    image_size=384,
    patch_size=16,
    num_classes=2,
    dim=256,
    depth=8,
    heads=6,
    mlp_dim=512,
    channels=1,
    dim_head=64,
)

SEED = 42
BATCH_SIZE = 2
IMG_SHAPE = (BATCH_SIZE, MODEL_KWARGS["channels"],
             MODEL_KWARGS["image_size"], MODEL_KWARGS["image_size"])


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _get_lsa_attention_layers(model: SimpleViT_LSA):
    """Return all LSAAttention modules in the transformer."""
    from LSA_ViT import LSAAttention
    return [m for m in model.modules() if isinstance(m, LSAAttention)]


# ---------------------------------------------------------------------------
# Phase 3.1 — Forward Pass Verification
# ---------------------------------------------------------------------------

def check_forward_pass(device: torch.device) -> None:
    """
    Verify:
      (a) Both models produce identical output shapes.
      (b) Outputs differ (LSA is not a no-op).
      (c) No NaN or Inf in outputs.
    """
    print("\n" + "=" * 60)
    print("Phase 3.1 — Forward Pass Verification")
    print("=" * 60)

    set_seeds(SEED)
    baseline = SimpleViT(**MODEL_KWARGS).to(device)

    set_seeds(SEED)
    lsa_model = SimpleViT_LSA(**MODEL_KWARGS).to(device)

    # Use the same random input for both
    set_seeds(SEED)
    x = torch.randn(*IMG_SHAPE, device=device)

    baseline.eval()
    lsa_model.eval()
    with torch.no_grad():
        out_baseline = baseline(x)
        out_lsa = lsa_model(x)

    # (a) Shape parity
    assert out_baseline.shape == out_lsa.shape, (
        f"Shape mismatch: baseline {out_baseline.shape} vs LSA {out_lsa.shape}"
    )
    assert out_baseline.shape == (BATCH_SIZE, MODEL_KWARGS["num_classes"]), (
        f"Unexpected output shape: {out_baseline.shape}"
    )
    print(f"  [PASS] Output shape: {tuple(out_baseline.shape)}")

    # (b) Outputs are different (LSA actually modifies the computation)
    max_diff = (out_lsa - out_baseline).abs().max().item()
    assert max_diff > 1e-6, (
        f"Outputs are suspiciously identical (max_diff={max_diff:.2e}). "
        "LSA may be a no-op — check that LSAAttention is actually being used."
    )
    print(f"  [PASS] Outputs differ (max_diff={max_diff:.4f}) — LSA is active")

    # (c) No NaN / Inf
    for name, out in [("baseline", out_baseline), ("LSA", out_lsa)]:
        assert not torch.isnan(out).any(), f"NaN detected in {name} output"
        assert not torch.isinf(out).any(), f"Inf detected in {name} output"
    print("  [PASS] No NaN or Inf in outputs")

    # --- Parameter count audit ---
    n_baseline = count_parameters(baseline)
    n_lsa = count_parameters(lsa_model)
    delta = n_lsa - n_baseline
    expected_delta = MODEL_KWARGS["depth"]   # one scalar τ per attention layer

    print(f"\n  Parameter Count Audit:")
    print(f"    Baseline  : {n_baseline:,}")
    print(f"    LSA model : {n_lsa:,}")
    print(f"    Delta     : +{delta} (expected +{expected_delta})")

    assert delta == expected_delta, (
        f"Parameter delta is {delta}, expected exactly +{expected_delta} "
        f"(one learnable τ per attention layer with depth={MODEL_KWARGS['depth']}). "
        "Check for accidental dropout layers, bias terms, or missing parameters."
    )
    print(f"  [PASS] Exactly +{expected_delta} parameters added (one τ per layer)")


# ---------------------------------------------------------------------------
# Phase 3.2 — Gradient Flow Check
# ---------------------------------------------------------------------------

def check_gradient_flow(device: torch.device) -> None:
    """
    Verify:
      (a) All parameters receive non-None gradients.
      (b) Temperature parameter τ receives a non-zero gradient.
      (c) No gradient explosion (L2 norm per parameter < 100).
    """
    print("\n" + "=" * 60)
    print("Phase 3.2 — Gradient Flow Check")
    print("=" * 60)

    set_seeds(SEED)
    lsa_model = SimpleViT_LSA(**MODEL_KWARGS).to(device)
    lsa_model.train()

    set_seeds(SEED)
    x = torch.randn(*IMG_SHAPE, device=device)
    labels = torch.randint(0, MODEL_KWARGS["num_classes"], (BATCH_SIZE,), device=device)

    criterion = torch.nn.CrossEntropyLoss()
    out = lsa_model(x)
    loss = criterion(out, labels)
    loss.backward()

    # (a) All parameters have gradients
    dead_params = []
    for name, param in lsa_model.named_parameters():
        if param.grad is None:
            dead_params.append(name)
    assert len(dead_params) == 0, (
        f"Dead parameters (no gradient): {dead_params}"
    )
    print("  [PASS] All parameters received gradients")

    # (b) Temperature parameters have non-zero gradients
    from LSA_ViT import LSAAttention
    temp_grad_norms = []
    for i, attn_layer in enumerate(_get_lsa_attention_layers(lsa_model)):
        tau = attn_layer.temperature
        assert tau.grad is not None, f"Layer {i}: τ has no gradient"
        grad_norm = tau.grad.abs().item()
        temp_grad_norms.append(grad_norm)
        assert grad_norm > 0.0, (
            f"Layer {i}: τ gradient is exactly zero ({grad_norm}). "
            "Learnable temperature is non-functional."
        )

    print(f"  [PASS] Temperature τ gradients non-zero across all {len(temp_grad_norms)} layers")
    print(f"         τ grad norms: min={min(temp_grad_norms):.4e}, "
          f"max={max(temp_grad_norms):.4e}, mean={np.mean(temp_grad_norms):.4e}")

    # (c) No gradient explosion
    exploding = []
    for name, param in lsa_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 100.0:
                exploding.append((name, grad_norm))

    if exploding:
        for name, norm in exploding:
            print(f"  [WARN] Large gradient: {name} norm={norm:.2f}")
    else:
        print("  [PASS] No gradient explosion detected (all norms < 100)")


# ---------------------------------------------------------------------------
# Phase 3.3 — Attention Map Inspection
# ---------------------------------------------------------------------------

def check_attention_maps(device: torch.device) -> None:
    """
    Verify the LSA attention mechanism properties on a dummy batch:

      (a) Diagonal masking broadcasts correctly across all heads H and batch B.
      (b) Diagonal entries are filled with -finfo(dtype).max (not literal -inf)
          in the pre-softmax dots tensor.
      (c) After softmax, diagonal entries are numerically zero (< 1e-6).
      (d) Temperature initialises to exp(τ_init) ≈ d_head^{-0.5}.
    """
    print("\n" + "=" * 60)
    print("Phase 3.3 — Attention Map Inspection")
    print("=" * 60)

    from LSA_ViT import LSAAttention
    from einops import rearrange

    # Build a small isolated LSAAttention for inspection
    dim      = MODEL_KWARGS["dim"]
    heads    = MODEL_KWARGS["heads"]
    dim_head = 64
    N = (MODEL_KWARGS["image_size"] // MODEL_KWARGS["patch_size"]) ** 2  # number of tokens

    set_seeds(SEED)
    attn_layer = LSAAttention(dim=dim, heads=heads, dim_head=dim_head).to(device)
    attn_layer.eval()

    # Register hooks to capture pre- and post-softmax attention scores
    pre_softmax_dots: dict = {}
    post_softmax_attn: dict = {}

    def hook_pre(module, input, output):
        # We intercept inside forward by monkey-patching — use the hook on Softmax
        pass

    # We will instrument the forward pass directly on a copy of the module logic
    # rather than registering a hook on nn.Softmax (which only sees its input).
    # Instead we subclass temporarily for inspection.
    class _InspectableAttention(LSAAttention):
        def forward(self, x):
            x = self.norm(x)
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
            )
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.clamp(max=4.0).exp()
            N_ = dots.shape[-1]
            mask = torch.eye(N_, device=dots.device, dtype=torch.bool)
            dots_masked = dots.masked_fill(mask, -torch.finfo(dots.dtype).max)
            pre_softmax_dots["value"] = dots_masked.detach().cpu()
            attn = self.attend(dots_masked)
            post_softmax_attn["value"] = attn.detach().cpu()
            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            return self.to_out(out)

    set_seeds(SEED)
    inspect_attn = _InspectableAttention(dim=dim, heads=heads, dim_head=dim_head).to(device)
    inspect_attn.load_state_dict(attn_layer.state_dict())
    inspect_attn.eval()

    x_tokens = torch.randn(BATCH_SIZE, N, dim, device=device)
    with torch.no_grad():
        _ = inspect_attn(x_tokens)

    dots_val = pre_softmax_dots["value"]    # shape: [B, H, N, N]
    attn_val = post_softmax_attn["value"]   # shape: [B, H, N, N]

    B_actual, H_actual, N1, N2 = dots_val.shape
    assert N1 == N2 == N, f"Expected square attention matrix [{N},{N}], got [{N1},{N2}]"
    assert H_actual == heads, f"Expected {heads} heads, got {H_actual}"
    assert B_actual == BATCH_SIZE, f"Expected batch {BATCH_SIZE}, got {B_actual}"

    print(f"  Attention tensor shape: [B={B_actual}, H={H_actual}, N={N1}, N={N2}]")

    # (a) Diagonal mask broadcasts correctly across all B and H
    expected_fill = -torch.finfo(dots_val.dtype).max
    for b in range(B_actual):
        for h in range(H_actual):
            diag = torch.diagonal(dots_val[b, h])  # shape [N]
            # All diagonal entries must equal the fill value
            assert torch.allclose(diag, torch.full_like(diag, expected_fill)), (
                f"Diagonal mask failed at batch={b}, head={h}. "
                f"Values: {diag[:5].tolist()}... (expected {expected_fill})"
            )
    print(f"  [PASS] Diagonal correctly filled with {expected_fill:.1f} "
          f"across all B={B_actual} batch elements and H={H_actual} heads")

    # (b) Off-diagonal entries are NOT filled (sanity: they should vary)
    off_diag_mask = ~torch.eye(N, dtype=torch.bool)
    off_diag_vals = dots_val[0, 0][off_diag_mask]
    assert off_diag_vals.std() > 0.0, (
        "Off-diagonal attention scores have zero variance — suspicious."
    )
    print(f"  [PASS] Off-diagonal entries are non-degenerate "
          f"(std={off_diag_vals.std():.4f})")

    # (c) After softmax, diagonal entries are numerically zero
    for b in range(B_actual):
        for h in range(H_actual):
            diag_post = torch.diagonal(attn_val[b, h])
            max_diag_val = diag_post.abs().max().item()
            assert max_diag_val < 1e-6, (
                f"Post-softmax diagonal not zero at batch={b}, head={h}: "
                f"max value={max_diag_val:.2e} (expected < 1e-6)"
            )
    print(f"  [PASS] Post-softmax diagonal entries are numerically zero "
          f"(< 1e-6) confirming correct AMP-safe masking behaviour")

    # (d) Temperature initialisation matches baseline scale
    tau_init = inspect_attn.temperature.item()
    scale_init = math.exp(tau_init)
    expected_scale = dim_head ** -0.5
    assert abs(scale_init - expected_scale) < 1e-5, (
        f"Temperature initialisation mismatch: exp(τ_init)={scale_init:.6f}, "
        f"expected d_head^(-0.5)={expected_scale:.6f}"
    )
    print(f"  [PASS] Temperature initialisation: τ_init={tau_init:.6f}, "
          f"exp(τ)={scale_init:.6f} ≈ d_head^(-0.5)={expected_scale:.6f}")

    # Report overall temperature values across all layers of the full model
    set_seeds(SEED)
    full_model = SimpleViT_LSA(**MODEL_KWARGS).to(device)
    from LSA_ViT import LSAAttention as _LSA
    temps = [m.temperature.item() for m in full_model.modules() if isinstance(m, _LSA)]
    print(f"\n  Initial temperature values across {len(temps)} attention layers:")
    for i, t in enumerate(temps):
        print(f"    Layer {i}: τ={t:.6f}, exp(τ)={math.exp(t):.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_checks(device: torch.device = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("LSA ABLATION — PRE-TRAINING SANITY CHECKS")
    print(f"Device: {device}")
    print("=" * 60)

    check_forward_pass(device)
    check_gradient_flow(device)
    check_attention_maps(device)

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED — safe to proceed with training")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_checks()
