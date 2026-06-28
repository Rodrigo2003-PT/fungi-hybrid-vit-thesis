"""
Architecture
------------
Tokenizer  : ``ConvTokenizer`` (Hassani et al., 2021) — a cascade of
             (Conv2d → ReLU → MaxPool2d) blocks, followed by a
             post-reshape LayerNorm. Default two-block configuration
             downsamples a 384×384 image to a 24×24 token grid (N=576),
             matching the baseline patch grid exactly.

Attention  : ``LSAAttention`` (Lee et al., 2021) — dot-product attention
             with (a) diagonal masking that removes each token's
             self-similarity term, and (b) a per-layer learnable temperature
             τ in log-space, initialised so exp(τ₀) = dim_head ** -0.5.

Positional : 2-D sinusoidal PE computed from the runtime grid (h', w')
embedding    returned by the tokenizer and cached via lru_cache.

Initialisation policies
-----------------------
Policy A  "default"          — PyTorch default for all layers; LSA
                               temperatures at log(dim_head**-0.5).
Policy B  "explicit_kaiming" — Kaiming Normal for Conv2d (fan_out, relu);
                               trunc_normal(std=0.02) for Linear layers;
                               constant 1/0 for LayerNorm.
                               Temperature parameters are *excluded* from
                               Policy B — they must stay at their
                               log(dim_head**-0.5) starting point (Lee et
                               al., 2021 scale invariant).

Author: Rodrigo Sá
Date: 2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from lsa_attention import LSAAttention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------

def posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    temperature: int = 10_000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2-D sinusoidal positional embedding — identical to baseline SimpleViT."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "Feature dimension must be a multiple of 4 for sin-cos PE."
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


@lru_cache(maxsize=32)
def _cached_posemb(
    h: int,
    w: int,
    dim: int,
    temperature: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a cached sin-cos PE tensor (stored on CPU)."""
    return posemb_sincos_2d(h, w, dim, temperature=temperature, dtype=dtype)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Convolutional Tokenizer (Hassani et al., 2021 — CCT style)
# ---------------------------------------------------------------------------

class ConvTokenizerBlock(nn.Module):
    """Single (Conv2d → ReLU → MaxPool2d) block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_k: int,
        conv_s: int,
        conv_p: int,
        pool_k: int,
        pool_s: int,
        pool_p: int,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=conv_k, stride=conv_s, padding=conv_p,
            bias=conv_bias,
        )
        self.act  = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_k, stride=pool_s, padding=pool_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.conv(x)))


class ConvTokenizer(nn.Module):
    """
    Convolutional tokenizer: cascade of ConvTokenizerBlock instances.

    Default two-block configuration (384×384 → 24×24, N=576):
        Block 1: Conv2d(C → hidden, k=7, s=2, p=3) → ReLU → MaxPool2d(k=3, s=2, p=1)
        Block 2: Conv2d(hidden → dim, k=3, s=2, p=1) → ReLU → MaxPool2d(k=3, s=2, p=1)

    Post-reshape LayerNorm normalises token statistics before PE addition,
    removing the confound where ReLU-skewed activations would otherwise
    enter the Transformer with different statistics than the baseline's
    normalised patch embeddings.

    Parameters
    ----------
    in_channels      : int  — image channels.
    dim              : int  — output token embedding dimension.
    blocks           : list[dict]  — ordered block configurations.
    hidden_channels  : int  — intermediate channel width (default 64).
    conv_bias        : bool — Conv2d bias flag (Hassani et al.: False).
    init_policy      : str  — "default" | "explicit_kaiming".
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        blocks: List[dict],
        hidden_channels: int = 64,
        conv_bias: bool = False,
        init_policy: str = "default",
    ):
        super().__init__()

        if len(blocks) == 0:
            raise ValueError("ConvTokenizer requires at least one block.")
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {hidden_channels}.")

        layers = []
        c_in   = in_channels
        n      = len(blocks)

        for i, blk in enumerate(blocks):
            c_out = dim if (i == n - 1) else hidden_channels
            layers.append(
                ConvTokenizerBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    conv_k=blk["conv_k"], conv_s=blk["conv_s"], conv_p=blk["conv_p"],
                    pool_k=blk["pool_k"], pool_s=blk["pool_s"], pool_p=blk["pool_p"],
                    conv_bias=conv_bias,
                )
            )
            c_in = c_out

        self.conv_blocks = nn.Sequential(*layers)
        self.post_ln     = nn.LayerNorm(dim)

        if init_policy == "explicit_kaiming":
            self._init_explicit_kaiming()

    def _init_explicit_kaiming(self) -> None:
        """Policy B: Kaiming Normal for Conv2d; constant 1/0 for LayerNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        tokens   : (B, N, dim)  — layer-normalised token sequence
        h_prime  : int          — output grid height
        w_prime  : int          — output grid width
        """
        feat            = self.conv_blocks(x)                      # (B, dim, H', W')
        h_prime, w_prime = feat.shape[2], feat.shape[3]
        tokens          = rearrange(feat, "b d h w -> b (h w) d")  # (B, N, dim)
        tokens          = self.post_ln(tokens)
        return tokens, h_prime, w_prime


# ---------------------------------------------------------------------------
# LSA Transformer
# ---------------------------------------------------------------------------

class LSATransformer(nn.Module):
    """
    Standard SimpleViT transformer encoder where every attention layer is
    ``LSAAttention``.  Residual connections and terminal LayerNorm are
    identical to the baseline and CTT transformers.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    LSAAttention(dim, heads=heads, dim_head=dim_head),
                    FeedForward(dim, mlp_dim),
                ])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)   + x
        return self.norm(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SimpleViT_LSA_ConvTok(nn.Module):
    """
    Parameters
    ----------
    image_size                  : int
    num_classes                 : int
    dim                         : int   — transformer / token embedding dimension
    depth                       : int   — number of encoder layers
    heads                       : int   — attention heads
    mlp_dim                     : int   — FFN hidden dimension
    channels                    : int   — image channels (default 1 for grayscale)
    dim_head                    : int   — dimension per head (default 64)
    convtok_hidden_channels     : int   — tokenizer intermediate channels (default 64)
    convtok_blocks              : list[dict]  — block specs for ConvTokenizer
    convtok_match_baseline_tokens : bool — enforce N == convtok_expected_hw product
    convtok_expected_hw         : list[int]  — expected [H', W'] from tokenizer
    convtok_conv_bias           : bool  — Conv2d bias (Hassani et al.: False)
    init_policy                 : str   — "default" | "explicit_kaiming"
    pe_temperature              : int   — sin-cos PE temperature (default 10000)
    """

    DEFAULT_CONVTOK_BLOCKS: List[dict] = [
        {"conv_k": 7, "conv_s": 2, "conv_p": 3, "pool_k": 3, "pool_s": 2, "pool_p": 1},
        {"conv_k": 3, "conv_s": 2, "conv_p": 1, "pool_k": 3, "pool_s": 2, "pool_p": 1},
    ]

    def __init__(
        self,
        *,
        image_size: int = 384,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 1,
        dim_head: int = 64,
        convtok_hidden_channels: int = 64,
        convtok_blocks: Optional[List[dict]] = None,
        convtok_match_baseline_tokens: bool = True,
        convtok_expected_hw: Optional[List[int]] = None,
        convtok_conv_bias: bool = False,
        init_policy: str = "default",
        pe_temperature: int = 10_000,
    ):
        super().__init__()

        if convtok_blocks is None:
            convtok_blocks = self.DEFAULT_CONVTOK_BLOCKS

        self.image_size                   = image_size
        self.channels                     = channels
        self.dim                          = dim
        self.convtok_match_baseline_tokens = convtok_match_baseline_tokens
        self.pe_temperature               = pe_temperature

        # ---------------------------------------------------------------
        # Expected grid from config
        # ---------------------------------------------------------------
        if convtok_expected_hw is not None:
            self._expected_h = int(convtok_expected_hw[0])
            self._expected_w = int(convtok_expected_hw[1])
        else:
            self._expected_h = None
            self._expected_w = None

        # ---------------------------------------------------------------
        # Analytical grid check at construction time
        # ---------------------------------------------------------------
        self._analytical_hw = self._compute_analytical_output_size(
            image_size, convtok_blocks
        )
        _analytical_n = self._analytical_hw[0] * self._analytical_hw[1]

        logger.info(
            "[LSA-ConvTok] Analytical output grid: %s (N=%d) for input %d×%d",
            self._analytical_hw, _analytical_n, image_size, image_size,
        )
        print(
            f"[LSA-ConvTok] Analytical output grid: {self._analytical_hw} "
            f"(N={_analytical_n}) for {image_size}×{image_size} input"
        )

        if convtok_match_baseline_tokens and self._expected_h is not None:
            assert (
                self._analytical_hw[0] == self._expected_h
                and self._analytical_hw[1] == self._expected_w
            ), (
                f"[LSA-ConvTok] Analytical matched-N FAILED at construction: "
                f"computed {self._analytical_hw}, expected "
                f"({self._expected_h}, {self._expected_w}). "
                "Check convtok_blocks configuration."
            )
            print(
                f"[LSA-ConvTok] Analytical matched-N passed: "
                f"{self._analytical_hw} == ({self._expected_h}, {self._expected_w}), "
                f"N={_analytical_n}"
            )

        # ---------------------------------------------------------------
        # Tokenizer
        # ---------------------------------------------------------------
        self.tokenizer = ConvTokenizer(
            in_channels=channels,
            dim=dim,
            blocks=convtok_blocks,
            hidden_channels=convtok_hidden_channels,
            conv_bias=convtok_conv_bias,
            init_policy=init_policy,
        )

        # ---------------------------------------------------------------
        # Transformer (LSA attention)
        # ---------------------------------------------------------------
        self.transformer = LSATransformer(dim, depth, heads, dim_head, mlp_dim)

        # ---------------------------------------------------------------
        # Classification head
        # ---------------------------------------------------------------
        self.pool        = "mean"
        self.to_latent   = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

        # ---------------------------------------------------------------
        # Initialisation — Policy B
        # Temperature parameters are EXCLUDED from Policy B.
        # ---------------------------------------------------------------
        if init_policy == "explicit_kaiming":
            # (1) ConvTokenizer Conv2d + LayerNorm — handled in ConvTokenizer.__init__
            # (2) Transformer Linear + LayerNorm
            self._init_transformer_explicit()
            # (3) Classification head
            self._init_linear_head_explicit()

        # ---------------------------------------------------------------
        # Runtime forward counter (matched-N check fires on first pass)
        # ---------------------------------------------------------------
        self._forward_count: int = 0

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_output_size(h_in: int, k: int, s: int, p: int) -> int:
        return (h_in + 2 * p - k) // s + 1

    @classmethod
    def _compute_analytical_output_size(
        cls, image_size: int, blocks: List[dict]
    ) -> Tuple[int, int]:
        h = image_size
        for blk in blocks:
            h = cls._conv_output_size(h, blk["conv_k"], blk["conv_s"], blk["conv_p"])
            h = cls._conv_output_size(h, blk["pool_k"], blk["pool_s"], blk["pool_p"])
        return (h, h)

    # ------------------------------------------------------------------
    # Policy B initialisation helpers
    # ------------------------------------------------------------------

    def _init_transformer_explicit(self) -> None:
        """
        Policy B: trunc_normal(std=0.02) for Linear; constant 1/0 for
        LayerNorm.  Temperature parameters are skipped — they are
        nn.Parameter scalars, not weight matrices, and must stay at their
        log(dim_head**-0.5) starting value (Lee et al., 2021).

        Coverage (depth=8, heads=6, dim_head=64, mlp_dim=512):
            Per layer × 8:
              LSAAttention.norm      LayerNorm(256)
              LSAAttention.to_qkv   Linear(256 → 1152, bias=False)
              LSAAttention.to_out   Linear(384 → 256,  bias=False)
              FeedForward.net[0]    LayerNorm(256)
              FeedForward.net[1]    Linear(256 → 512)
              FeedForward.net[3]    Linear(512 → 256)
            Terminal:
              LSATransformer.norm   LayerNorm(256)
        """
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            # nn.Parameter (temperature) — intentionally not touched.

    def _init_linear_head_explicit(self) -> None:
        """Policy B: trunc_normal(std=0.02) for the classification head."""
        nn.init.trunc_normal_(self.linear_head.weight, std=0.02)
        nn.init.constant_(self.linear_head.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        device = img.device
        dtype  = img.dtype

        # 1. Convolutional tokenisation → (B, N, dim)
        tokens, h_prime, w_prime = self.tokenizer(img)

        # 2. Runtime matched-N assertion (first forward pass only)
        self._forward_count += 1
        if self._forward_count == 1:
            self._runtime_token_check(h_prime, w_prime)

        # 3. Sinusoidal 2-D PE (cached)
        pe     = _cached_posemb(h_prime, w_prime, self.dim, self.pe_temperature, dtype)
        tokens = tokens + pe.to(device=device, dtype=dtype)

        # 4. LSA Transformer encoder
        tokens = self.transformer(tokens)

        # 5. Global mean pooling + classification head
        tokens = tokens.mean(dim=1)
        tokens = self.to_latent(tokens)
        return self.linear_head(tokens)

    def _runtime_token_check(self, h_prime: int, w_prime: int) -> None:
        """
        Enforce the matched-N guarantee on the first forward pass and log
        the runtime token grid.
        """
        n_tokens = h_prime * w_prime
        logger.info(
            "[LSA-ConvTok] Runtime token grid (forward #%d): H'=%d, W'=%d, N=%d",
            self._forward_count, h_prime, w_prime, n_tokens,
        )
        print(
            f"[LSA-ConvTok] Runtime token grid (forward #{self._forward_count}): "
            f"H'={h_prime}, W'={w_prime}, N={n_tokens}"
        )

        if self.convtok_match_baseline_tokens:
            expected_h = self._expected_h if self._expected_h is not None else 24
            expected_w = self._expected_w if self._expected_w is not None else 24
            assert (h_prime == expected_h) and (w_prime == expected_w), (
                f"[LSA-ConvTok] Matched-N FAILED (forward #{self._forward_count}): "
                f"got ({h_prime}, {w_prime}), expected ({expected_h}, {expected_w})."
            )
            print(
                f"[LSA-ConvTok] Matched-N assertion passed: "
                f"({h_prime}, {w_prime}) == ({expected_h}, {expected_w}), N={n_tokens}"
            )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_arch_hash(config: dict) -> str:
    """
    Compute a 16-char SHA-256 hash of architecture-determining config fields
    for ``SimpleViT_LSA_ConvTok``.

    ``"attention_type": "lsa_diagonal"`` is included as a fixed discriminator
    field so that the aggregate hash is always distinct from the CTT-only
    hash (``convViT.py``), even when all other fields are identical.
    """
    arch_fields = {
        # Attention mechanism identity — fixed discriminator
        "attention_type":              "lsa_diagonal",
        # Tokenizer geometry
        "convtok_blocks":              config.get("convtok_blocks"),
        "convtok_hidden_channels":     config.get("convtok_hidden_channels", 64),
        "convtok_post_ln":             config.get("convtok_post_ln", True),
        "convtok_expected_hw":         config.get("convtok_expected_hw"),
        # Transformer
        "init_policy":                 config.get("init_policy", "default"),
        "model_config":                config.get("model_config"),
        # Input geometry
        "image_size":                  config.get("image_size"),
        "channels":                    config.get("channels"),
        "num_classes":                 config.get("num_classes", None),
    }
    serialised = json.dumps(arch_fields, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:  # noqa: C901
    import math
    print("=" * 70)
    print("SMOKE TEST: lsa_convtok_vit.py")
    print("=" * 70)

    from lsa_logger import LSALogger

    BLOCKS = [
        {"conv_k": 7, "conv_s": 2, "conv_p": 3, "pool_k": 3, "pool_s": 2, "pool_p": 1},
        {"conv_k": 3, "conv_s": 2, "conv_p": 1, "pool_k": 3, "pool_s": 2, "pool_p": 1},
    ]
    NUM_CLASSES = 2
    DIM         = 256
    DEPTH       = 8
    HEADS       = 6
    DIM_HEAD    = 64
    BATCH       = 2

    # ------------------------------------------------------------------
    # [1] Instantiation and analytical grid
    # ------------------------------------------------------------------
    print("\n[1] Instantiation")
    model = SimpleViT_LSA_ConvTok(
        image_size=384,
        num_classes=NUM_CLASSES,
        dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=512,
        channels=1, dim_head=DIM_HEAD,
        convtok_hidden_channels=64,
        convtok_blocks=BLOCKS,
        convtok_match_baseline_tokens=True,
        convtok_expected_hw=[24, 24],
        convtok_conv_bias=False,
        init_policy="default",
    )
    model.eval()
    assert model._analytical_hw == (24, 24), f"Bad grid: {model._analytical_hw}"

    total, trainable = count_parameters(model)
    print(f"    Parameters: {total:,} total, {trainable:,} trainable")

    # ------------------------------------------------------------------
    # [2] Tokenizer output shape and post-LN statistics
    # ------------------------------------------------------------------
    print("\n[2] Tokenizer output")
    dummy = torch.randn(BATCH, 1, 384, 384)
    with torch.no_grad():
        toks, hp, wp = model.tokenizer(dummy)
    assert toks.shape == (BATCH, 576, DIM),  f"Bad token shape: {toks.shape}"
    assert (hp, wp) == (24, 24),             f"Bad grid: ({hp}, {wp})"
    assert abs(toks.mean().item()) < 0.5,    "post-LN mean too large"
    print(f"    token shape={toks.shape}, grid=({hp},{wp}), "
          f"mean={toks.mean().item():.4f}, std={toks.std().item():.4f}")

    # ------------------------------------------------------------------
    # [3] Temperature parameter coverage
    # ------------------------------------------------------------------
    print("\n[3] Temperature parameters")
    temp_params = [(n, p) for n, p in model.named_parameters() if "temperature" in n]
    assert len(temp_params) == DEPTH, (
        f"Expected {DEPTH} temperature params, got {len(temp_params)}"
    )
    expected_tau0 = math.log(DIM_HEAD ** -0.5)
    for name, param in temp_params:
        val = param.item()
        assert abs(val - expected_tau0) < 1e-5, (
            f"{name}: expected {expected_tau0:.4f}, got {val:.4f}"
        )
    print(f"    {len(temp_params)} temperature params, all τ₀ ≈ {expected_tau0:.4f}")

    # ------------------------------------------------------------------
    # [4] Full forward pass
    # ------------------------------------------------------------------
    print("\n[4] Full forward pass")
    assert model._forward_count == 0
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == (BATCH, NUM_CLASSES), f"Bad logit shape: {logits.shape}"
    assert not torch.isnan(logits).any(),         "NaN in logits"
    assert model._forward_count == 1
    print(f"    logits shape={logits.shape}, forward_count={model._forward_count}")

    # ------------------------------------------------------------------
    # [5] Policy B initialisation
    # ------------------------------------------------------------------
    print("\n[5] Policy B (explicit_kaiming)")
    model_b = SimpleViT_LSA_ConvTok(
        image_size=384,
        num_classes=NUM_CLASSES,
        dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=512,
        channels=1, dim_head=DIM_HEAD,
        convtok_hidden_channels=64,
        convtok_blocks=BLOCKS,
        convtok_match_baseline_tokens=True,
        convtok_expected_hw=[24, 24],
        init_policy="explicit_kaiming",
    )
    model_b.eval()

    tf_linears = [m for m in model_b.transformer.modules() if isinstance(m, nn.Linear)]
    for m in tf_linears:
        std = m.weight.std().item()
        assert std < 0.035, f"Policy B Linear std={std:.4f}; expected ≈0.02"

    # Temperature parameters must be UNCHANGED by Policy B
    for name, param in model_b.named_parameters():
        if "temperature" in name:
            val = param.item()
            assert abs(val - expected_tau0) < 1e-5, (
                f"Policy B changed temperature {name}: {val:.4f} ≠ {expected_tau0:.4f}"
            )
    print("    Policy B assertions passed (temperatures untouched)")

    # ------------------------------------------------------------------
    # [6] LSALogger compatibility
    # ------------------------------------------------------------------
    print("\n[6] LSALogger compatibility")
    fold_results = {}
    lsa_log = LSALogger(
        model=model, fold_results=fold_results,
        iter_idx=0, fold_idx=0,
    )
    assert lsa_log._n_layers == DEPTH, (
        f"Logger found {lsa_log._n_layers} LSA layers, expected {DEPTH}"
    )
    lsa_log.start_validation_epoch()
    with torch.no_grad():
        _ = model(dummy)
    lsa_log.end_validation_epoch(epoch=0, verbose=False)
    assert len(fold_results["lsa_attn_entropy"]) == 1
    assert len(fold_results["lsa_temperatures"]) == 1
    print(f"    Logger attached to {lsa_log._n_layers} layers, entropy logged OK")

    # ------------------------------------------------------------------
    # [7] Arch hash distinction across ablation cells
    # ------------------------------------------------------------------
    print("\n[7] Arch hash distinction")
    # Simulate configs for all four cells
    model_cfg = {"dim": 256, "depth": 8, "heads": 6, "mlp_dim": 512,
                 "dim_head": 64, "pe_temperature": 10000}
    base_cfg = {
        "convtok_blocks": BLOCKS, "convtok_hidden_channels": 64,
        "convtok_post_ln": True, "convtok_expected_hw": [24, 24],
        "init_policy": "default", "model_config": model_cfg,
        "image_size": 384, "channels": 1,
    }

    import convViT as _ctt_module
    hash_ctt     = _ctt_module.compute_arch_hash(base_cfg)
    hash_agg     = compute_arch_hash(base_cfg)
    assert hash_ctt != hash_agg, "CTT and aggregate hashes must differ"
    print(f"    CTT hash      : {hash_ctt}")
    print(f"    Aggregate hash: {hash_agg}")
    print("    Hash distinction assertion passed")

    print("\n" + "=" * 70)
    print("ALL SMOKE-TEST ASSERTIONS PASSED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
