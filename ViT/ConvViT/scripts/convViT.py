"""
SimpleViT + Convolutional Tokenization
---------------------------------------------------------------------------------
PatchTok  → Rearrange + LayerNorm + Linear + LayerNorm
ConvTok   → Conv2d×n + ReLU + MaxPool2d×n + reshape + LayerNorm

Block layout (default, named "matched_N_2block"):
    Block 1: Conv2d(1 → hidden, k=7, s=2, p=3) → ReLU → MaxPool2d(k=3, s=2, p=1)
    Block 2: Conv2d(hidden → dim, k=3, s=2, p=1) → ReLU → MaxPool2d(k=3, s=2, p=1)

Positional embedding
--------------------
Sin-cos 2-D PE is computed dynamically from (H', W') in forward() and cached by
(H', W', dim, temperature, dtype) to avoid throughput distortion in benchmarks.

Normalisation
-------------
After reshape to (B, N, dim) a LayerNorm(dim) is applied before adding PE and
passing to the Transformer, matching the normalised-token interface of the
baseline SimpleViT. This removes the confound where ConvTok tokens would
otherwise enter the Transformer with ReLU-skewed activation statistics.

Initialisation policy
---------------------
Policy A  "default"          – PyTorch default init for all layers (used for the
                               primary tokenizer ablation).
Policy B  "explicit_kaiming" – Applied to all three model components:
                               (1) ConvTokenizer Conv2d: Kaiming Normal, fan_out,
                                   nonlinearity='relu' (He et al., 2015); bias→0.
                                   ConvTokenizer LayerNorm: weight→1, bias→0.
                               (2) Transformer all nn.Linear: trunc_normal std=0.02,
                                   bias→0. All nn.LayerNorm: weight→1, bias→0.
                                   Matches Hassani et al. (2021)
                                   TransformerClassifier.init_weight exactly.
                               (3) Classification head nn.Linear: trunc_normal
                                   std=0.02, bias→0.
                               Used only when init_policy="explicit_kaiming" is
                               set in config, as a *separate* ablation factor.

Author: Rodrigo Sá (2025)
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    temperature: int = 10_000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2-D sin-cos positional embedding – identical to baseline SimpleViT."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "Feature dimension must be a multiple of 4 for sin-cos PE."
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# ---------------------------------------------------------------------------
# Transformer blocks
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


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ---------------------------------------------------------------------------
# Convolutional Tokenizer (CCT-style)
# ---------------------------------------------------------------------------

class ConvTokenizerBlock(nn.Module):
    """
    Single (Conv2d → ReLU → MaxPool2d) block.
    """

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
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_k, stride=pool_s, padding=pool_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.conv(x)))


class ConvTokenizer(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    dim : int
        Output embedding dimension.

        For a 2-block default configuration with ``in_channels=1``,
        ``hidden_channels=64``, ``dim=256``:
            Block 1 weight: 7×7×1×64  =   3,136 params
            Block 2 weight: 3×3×64×256 = 147,456 params
        Total tokenizer Conv weights: ~150,592 params.

    hidden_channels : int
        Channel width for all blocks except the final one. Default: 64.
    blocks : list[dict]
        Ordered list of block configurations. Each dict must contain keys:
        ``conv_k``, ``conv_s``, ``conv_p``, ``pool_k``, ``pool_s``, ``pool_p``.
    conv_bias : bool
        Whether to use bias in Conv2d layers.
    init_policy : str
        "default" — PyTorch default init for all layers in this tokenizer.
        "explicit_kaiming" — Kaiming Normal (fan_out, nonlinearity='relu') for
            Conv2d; weight→1, bias→0 for the post-reshape LayerNorm. Initialisation
            of the Transformer body and classification head under Policy B is handled
            by ``SimpleViT_ConvTok._init_transformer_explicit()`` and
            ``SimpleViT_ConvTok._init_linear_head_explicit()`` respectively.
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
            raise ValueError(
                f"hidden_channels must be >= 1, got {hidden_channels}."
            )
        layers = []
        c_in = in_channels
        n_blocks = len(blocks)

        for i, blk in enumerate(blocks):
            is_final = (i == n_blocks - 1)
            c_out = dim if is_final else hidden_channels
            layers.append(
                ConvTokenizerBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    conv_k=blk["conv_k"],
                    conv_s=blk["conv_s"],
                    conv_p=blk["conv_p"],
                    pool_k=blk["pool_k"],
                    pool_s=blk["pool_s"],
                    pool_p=blk["pool_p"],
                    conv_bias=conv_bias,
                )
            )
            c_in = c_out

        self.conv_blocks = nn.Sequential(*layers)

        self.post_ln = nn.LayerNorm(dim)

        if init_policy == "explicit_kaiming":
            self._init_explicit_kaiming()
        # else: rely on PyTorch default init

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_explicit_kaiming(self):
        """
        Policy B: Kaiming Normal for Conv2d, constant 0/1 for LayerNorm.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # fan_out + nonlinearity='relu' per He et al. (2015)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        tokens   : (B, N, dim)   normalised token sequence
        h_prime  : int           output grid height
        w_prime  : int           output grid width
        """
        feat = self.conv_blocks(x)                         # (B, dim, H', W')
        h_prime, w_prime = feat.shape[2], feat.shape[3]
        tokens = rearrange(feat, "b d h w -> b (h w) d")   # (B, N, dim)
        tokens = self.post_ln(tokens)
        return tokens, h_prime, w_prime


# ---------------------------------------------------------------------------
# Positional-embedding cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _cached_posemb(
    h: int,
    w: int,
    dim: int,
    temperature: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return a cached sin-cos PE tensor stored on CPU.
    """
    return posemb_sincos_2d(h, w, dim, temperature=temperature, dtype=dtype)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SimpleViT_ConvTok(nn.Module):
    """
    SimpleViT with convolutional tokenisation (CCT-style).

    Parameters
    ----------
    image_size : int
        Input image spatial size. Default 384.
    num_classes : int
        Number of output classes.
    dim : int
        Transformer / token embedding dimension.
    depth : int
        Number of Transformer encoder layers.
    heads : int
        Number of attention heads.
    mlp_dim : int
        FFN hidden dimension.
    channels : int
        Number of input image channels.
    dim_head : int
        Per-head dimension. Default 64.
    convtok_hidden_channels : int
        Channel width for all tokenizer blocks except the final projection.
    convtok_blocks : list[dict]
        Block configurations for ConvTokenizer.
    convtok_match_baseline_tokens : bool
        If True, enforce runtime assertion that (H', W') == convtok_expected_hw.
    convtok_expected_hw : list[int]
        Expected [H', W']. Stored as a list for JSON round-trip safety.
    convtok_conv_bias : bool
        Conv2d bias flag (Hassani et al.: False).
    init_policy : str
        ``"default"`` or ``"explicit_kaiming"``.
    pe_temperature : int
        Temperature for sin-cos PE. Default 10000.
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

        self.image_size = image_size
        self.channels = channels
        self.dim = dim
        self.convtok_match_baseline_tokens = convtok_match_baseline_tokens
        self.pe_temperature = pe_temperature

        if convtok_expected_hw is not None:
            self._expected_h = int(convtok_expected_hw[0])
            self._expected_w = int(convtok_expected_hw[1])
        else:
            self._expected_h = None
            self._expected_w = None

        self._analytical_hw = self._compute_analytical_output_size(
            image_size, convtok_blocks
        )
        _analytical_n = self._analytical_hw[0] * self._analytical_hw[1]
        logger.info(
            "[ConvTok] Analytical output grid: %s (N=%d) for input %d×%d",
            self._analytical_hw, _analytical_n, image_size, image_size,
        )
        print(
            f"[ConvTok] Analytical output grid: {self._analytical_hw} "
            f"(N={_analytical_n}) for {image_size}×{image_size} input"
        )

        if convtok_match_baseline_tokens and self._expected_h is not None:
            assert (
                self._analytical_hw[0] == self._expected_h
                and self._analytical_hw[1] == self._expected_w
            ), (
                f"[ConvTok] Analytical matched-N FAILED at construction: "
                f"computed {self._analytical_hw}, expected "
                f"({self._expected_h}, {self._expected_w}). "
                f"Check convtok_blocks configuration."
            )
            print(
                f"[ConvTok] Analytical matched-N assertion passed at construction: "
                f"{self._analytical_hw} = ({self._expected_h}, {self._expected_w}), "
                f"N={_analytical_n}"
            )

        # ------------------------------------------------------------------
        # Tokenizer
        # ------------------------------------------------------------------
        self.tokenizer = ConvTokenizer(
            in_channels=channels,
            dim=dim,
            blocks=convtok_blocks,
            hidden_channels=convtok_hidden_channels,
            conv_bias=convtok_conv_bias,
            init_policy=init_policy,
        )

        # ------------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------------
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

        if init_policy == "explicit_kaiming":
            # Policy B initialisation — three components, called in module order:
            # (1) ConvTokenizer Conv2d + LayerNorm: handled inside
            #     ConvTokenizer._init_explicit_kaiming(), called during
            #     ConvTokenizer.__init__() above.
            # (2) Transformer all Linear + LayerNorm:
            self._init_transformer_explicit()
            # (3) Classification head Linear:
            self._init_linear_head_explicit()

        # ------------------------------------------------------------------
        # Runtime token-check counter.
        # ------------------------------------------------------------------
        self._forward_count: int = 0

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_output_size(h_in: int, k: int, s: int, p: int) -> int:
        """Standard Conv2d / MaxPool2d output-size formula (floor division)."""
        return (h_in + 2 * p - k) // s + 1

    @classmethod
    def _compute_analytical_output_size(
        cls, image_size: int, blocks: List[dict]
    ) -> Tuple[int, int]:
        """
        Compute expected (H', W') analytically from block configuration.
        Assumes square input and symmetric conv/pool parameters.
        """
        h = image_size
        for blk in blocks:
            h = cls._conv_output_size(h, blk["conv_k"], blk["conv_s"], blk["conv_p"])
            h = cls._conv_output_size(h, blk["pool_k"], blk["pool_s"], blk["pool_p"])
        return (h, h)

    def _init_transformer_explicit(self) -> None:
        """
        Policy B: initialise all nn.Linear and nn.LayerNorm inside
        ``self.transformer`` to match Hassani et al. (2021)
        ``TransformerClassifier.init_weight``.

        Scheme
        ------
        ``nn.Linear``   weight ← trunc_normal(mean=0, std=0.02); bias ← 0.
        ``nn.LayerNorm`` weight ← 1;                              bias ← 0.

        Rationale
        ---------
        Truncated normal with std=0.02 is the canonical weight initialisation
        for Vision Transformer linear layers, established by Dosovitskiy et al.
        (2021) and reproduced verbatim in Hassani et al. (2021). Without this,
        Transformer Linear layers retain PyTorch's Kaiming Uniform default
        (std ≈ sqrt(1/fan_in)), which for the QKV projection (dim=256 →
        inner_dim=384) gives std ≈ 0.063 — more than 3× the intended value.

        Coverage (depth=8, heads=6, dim_head=64, mlp_dim=512)
        -------------------------------------------------------
        Per encoder layer × 8:
          Attention.norm       LayerNorm(256)
          Attention.to_qkv     Linear(256 → 1152, bias=False) — bias branch skipped
          Attention.to_out     Linear(384 → 256,  bias=False) — bias branch skipped
          FeedForward.net[0]   LayerNorm(256)
          FeedForward.net[1]   Linear(256 → 512)
          FeedForward.net[3]   Linear(512 → 256)
        Terminal:
          Transformer.norm     LayerNorm(256)
        """
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _init_linear_head_explicit(self) -> None:
        """
        Policy B: initialise the classification head to match Hassani et al.

        ``nn.Linear`` weight ← trunc_normal(mean=0, std=0.02); bias ← 0.
        """
        nn.init.trunc_normal_(self.linear_head.weight, std=0.02)
        nn.init.constant_(self.linear_head.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        device = img.device
        dtype = img.dtype

        # 1. Convolutional tokenisation  →  (B, N, dim)
        tokens, h_prime, w_prime = self.tokenizer(img)
        n_tokens = h_prime * w_prime

        self._forward_count += 1
        if self._forward_count == 1:
            self._runtime_token_check(h_prime, w_prime, n_tokens)

        # 3. Positional embedding
        pe = _cached_posemb(h_prime, w_prime, self.dim, self.pe_temperature, dtype)
        tokens = tokens + pe.to(device=device, dtype=dtype)

        # 4. Transformer encoder
        tokens = self.transformer(tokens)

        # 5. Global mean pooling + head
        tokens = tokens.mean(dim=1)
        tokens = self.to_latent(tokens)
        return self.linear_head(tokens)

    def _runtime_token_check(self, h_prime: int, w_prime: int, n_tokens: int):
        """
        Enforce matched-N guarantee and log token statistics.

        Raises ``AssertionError`` if ``convtok_match_baseline_tokens=True``
        and (H', W') ≠ ``convtok_expected_hw``.

        Called from ``forward()`` on a configurable interval so that
        mid-training resolution drift is caught beyond the first epoch.
        """
        logger.info(
            "[ConvTok] Runtime token grid (forward #%d): H'=%d, W'=%d, N=%d",
            self._forward_count, h_prime, w_prime, n_tokens,
        )
        print(
            f"[ConvTok] Runtime token grid (forward #{self._forward_count}): "
            f"H'={h_prime}, W'={w_prime}, N={n_tokens}"
        )

        if self.convtok_match_baseline_tokens:
            expected_h = self._expected_h if self._expected_h is not None else 24
            expected_w = self._expected_w if self._expected_w is not None else 24
            assert (h_prime == expected_h) and (w_prime == expected_w), (
                f"[ConvTok] Matched-N FAILED (forward #{self._forward_count}): "
                f"got ({h_prime}, {w_prime}), expected ({expected_h}, {expected_w}). "
                f"Check convtok_blocks configuration or input resolution."
            )
            print(
                f"[ConvTok] Matched-N assertion passed (forward #{self._forward_count}): "
                f"({h_prime}, {w_prime}) = ({expected_h}, {expected_w}), N={n_tokens}"
            )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return ``(total_params, trainable_params)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compute_arch_hash(config: dict) -> str:
    """
    Compute a short SHA-256 hash of architecture-determining config fields.
    """
    arch_fields = {
        "convtok_blocks": config.get("convtok_blocks"),
        "convtok_hidden_channels": config.get("convtok_hidden_channels", 64),
        "convtok_post_ln": config.get("convtok_post_ln", True),
        "convtok_expected_hw": config.get("convtok_expected_hw"),
        "init_policy": config.get("init_policy", "default"),
        "model_config": config.get("model_config"),
        "image_size": config.get("image_size"),
        "channels": config.get("channels"),
        "num_classes": config.get("num_classes", None),
    }
    serialised = json.dumps(arch_fields, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    print("=" * 70)
    print("SMOKE TEST: convViT.py")
    print("=" * 70)

    default_blocks = [
        {"conv_k": 7, "conv_s": 2, "conv_p": 3, "pool_k": 3, "pool_s": 2, "pool_p": 1},
        {"conv_k": 3, "conv_s": 2, "conv_p": 1, "pool_k": 3, "pool_s": 2, "pool_p": 1},
    ]
    NUM_CLASSES = 2
    DIM = 256
    HIDDEN_CH = 64
    BATCH = 2

    # ------------------------------------------------------------------
    # Check 1: Instantiation
    # ------------------------------------------------------------------
    print("\n Model instantiation (analytical grid check fires here)")
    model = SimpleViT_ConvTok(
        image_size=384,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=8,
        heads=6,
        mlp_dim=512,
        channels=1,
        dim_head=64,
        convtok_hidden_channels=HIDDEN_CH,
        convtok_blocks=default_blocks,
        convtok_match_baseline_tokens=True,
        convtok_expected_hw=[24, 24],
        convtok_conv_bias=False,
        init_policy="default",
    )
    model.eval()

    total, trainable = count_parameters(model)
    print(f"    Parameters: {total:,} total, {trainable:,} trainable")

    print("\n Direct tokenizer forward (ConvTokenizer only — no _runtime_token_check)")
    dummy = torch.randn(BATCH, 1, 384, 384)
    with torch.no_grad():
        toks, h_p, w_p = model.tokenizer(dummy)

    print(f"    Token shape : {toks.shape}  (expect ({BATCH}, 576, {DIM}))")
    print(f"    Grid        : ({h_p}, {w_p})  (expect (24, 24))")
    print(f"    Token mean  : {toks.mean().item():.4f}  (expect ≈ 0 due to LayerNorm)")
    print(f"    Token std   : {toks.std().item():.4f}  (expect ≈ 1 due to LayerNorm)")
    assert toks.shape == (BATCH, 576, DIM), (
        f"Unexpected token shape: {toks.shape}"
    )
    assert (h_p, w_p) == (24, 24), f"Unexpected grid: ({h_p}, {w_p})"
    # LayerNorm: mean should be near 0. Loose tolerance for random inputs.
    assert abs(toks.mean().item()) < 0.5, (
        f"Token mean too large ({toks.mean().item():.4f}); LayerNorm may not be applied."
    )
    print("    Tokenizer assertions passed")

    print("\n Full model forward (SimpleViT_ConvTok.forward — _runtime_token_check fires here)")
    assert model._forward_count == 0, "Counter should be 0 before first full forward."

    with torch.no_grad():
        for call_idx in range(1, 4):
            logits = model(dummy)
            assert model._forward_count == call_idx, (
                f"Expected _forward_count={call_idx}, got {model._forward_count}"
            )
            assert logits.shape == (BATCH, NUM_CLASSES), (
                f"Unexpected logit shape: {logits.shape}"
            )
    print(f"    Logits shape: {logits.shape}")
    print(f"    _forward_count after 3 calls: {model._forward_count}")
    print(
        "   _runtime_token_check fired on calls 1, 2, 3 "
        "(interval=1, visible in output above)"
    )

    print("\n Parameter count comparison: hidden_channels=64 (fixed) vs hidden_channels=256 (prior)")
    model_old = SimpleViT_ConvTok(
        image_size=384,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=8,
        heads=6,
        mlp_dim=512,
        channels=1,
        dim_head=64,
        convtok_hidden_channels=DIM,   # reproduces original bug
        convtok_blocks=default_blocks,
        convtok_match_baseline_tokens=True,
        convtok_expected_hw=[24, 24],
        convtok_conv_bias=False,
        init_policy="default",
    )
    total_fixed, _ = count_parameters(model)
    total_old, _ = count_parameters(model_old)
    tok_fixed, _ = count_parameters(model.tokenizer)
    tok_old, _ = count_parameters(model_old.tokenizer)

    print(f"    Tokenizer params (hidden=64,  fixed): {tok_fixed:>10,}")
    print(f"    Tokenizer params (hidden=256, prior): {tok_old:>10,}")
    print(f"    Total model params (fixed) : {total_fixed:>10,}")
    print(f"    Total model params (prior) : {total_old:>10,}")
    assert tok_fixed < tok_old, (
        "Fixed tokenizer should have fewer parameters than the prior implementation."
    )
    reduction = (tok_old - tok_fixed) / tok_old * 100
    print(f" Tokenizer parameter reduction: {reduction:.1f}%")

    # ------------------------------------------------------------------
    # Check 5: Policy B covers the Transformer body
    # ------------------------------------------------------------------
    print("\n Policy B ('explicit_kaiming') — transformer body initialisation")
    model_b = SimpleViT_ConvTok(
        image_size=384,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=8,
        heads=6,
        mlp_dim=512,
        channels=1,
        dim_head=64,
        convtok_hidden_channels=HIDDEN_CH,
        convtok_blocks=default_blocks,
        convtok_match_baseline_tokens=True,
        convtok_expected_hw=[24, 24],
        convtok_conv_bias=False,
        init_policy="explicit_kaiming",
    )
    model_b.eval()

    tf_linears_b = [m for m in model_b.transformer.modules() if isinstance(m, nn.Linear)]
    tf_linears_a = [m for m in model.transformer.modules()   if isinstance(m, nn.Linear)]
    assert tf_linears_b, "No nn.Linear found in Policy B transformer — check architecture."

    # (a) All transformer Linear weights must have std ≈ 0.02, not Kaiming default.
    for m in tf_linears_b:
        std = m.weight.std().item()
        assert std < 0.035, (
            f"Policy B transformer Linear std={std:.4f}; expected ≈0.02 "
            f"(trunc_normal). _init_transformer_explicit() may not have run."
        )

    # (b) All transformer LayerNorms must have weight=1, bias=0.
    for m in model_b.transformer.modules():
        if isinstance(m, nn.LayerNorm):
            assert torch.allclose(m.weight, torch.ones_like(m.weight)), (
                "Policy B LayerNorm weight ≠ 1 in transformer body."
            )
            assert torch.allclose(m.bias, torch.zeros_like(m.bias)), (
                "Policy B LayerNorm bias ≠ 0 in transformer body."
            )

    # (c) Classification head: trunc_normal weight, zero bias.
    head_std = model_b.linear_head.weight.std().item()
    assert head_std < 0.035, (
        f"Policy B head std={head_std:.4f}; expected ≈0.02."
    )
    assert torch.allclose(
        model_b.linear_head.bias, torch.zeros_like(model_b.linear_head.bias)
    ), "Policy B head bias ≠ 0."

    # (d) Confirm Policy A transformer stds are larger, i.e. the two policies
    #     are empirically distinguishable — a necessary condition for the
    #     initialisation ablation to be interpretable.
    mean_std_a = sum(m.weight.std().item() for m in tf_linears_a) / len(tf_linears_a)
    mean_std_b = sum(m.weight.std().item() for m in tf_linears_b) / len(tf_linears_b)
    assert mean_std_b < mean_std_a, (
        "Policy B transformer weights are not detectably different from Policy A. "
        "Ablation would be uninterpretable."
    )
    print(f"    Policy A  mean transformer Linear std : {mean_std_a:.4f}  (Kaiming Uniform default)")
    print(f"    Policy B  mean transformer Linear std : {mean_std_b:.4f}  (trunc_normal 0.02)")
    print("    Policy B transformer body assertions passed ✓")

    print("\n" + "=" * 70)
    print("ALL SMOKE-TEST ASSERTIONS PASSED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
