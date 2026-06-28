"""
LSA Attention

Mechanism
---------
Locality Self-Attention (Lee et al., 2021)

  1. **Diagonal masking** — the self-similarity term (i == j) is set to
     -inf before softmax, removing the dominant diagonal that allows a token
     to attend primarily to itself.  This forces information aggregation from
     neighbouring tokens even in shallow networks.  This is a *softer*
     inductive bias than a hard local window: all off-diagonal positions
     remain unconstrained, so the model retains global context capacity.

  2. **Learnable temperature** — the fixed scale factor ``dim_head ** -0.5``
     is replaced by a per-layer learnable scalar ``τ`` parameterised in
     log-space: ``temperature = log(dim_head ** -0.5)``, applied as
     ``dots * exp(τ)``.  Parameterising in log-space keeps τ unconstrained
     (no positivity projection needed) while ensuring ``exp(τ) > 0`` always.
     τ is initialised so ``exp(τ) = dim_head ** -0.5``, matching the
     baseline scale at epoch 0.  A hard clamp at τ ≤ 4.0 (``exp(4) ≈ 54``)
     prevents fp16 overflow under AMP, consistent with the AMP saturation
     warning threshold used in ``lsa_logger.py``.

Author: Rodrigo Sá
Date: 2026
"""

import torch
import torch.nn as nn
from einops import rearrange


class LSAAttention(nn.Module):
    """
    Locality Self-Attention (Lee et al., 2021).

    Parameters
    ----------
    dim : int
        Token embedding dimension (input and output).
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per head.  Default 64 matches the SimpleViT baseline and
        the CTT model, keeping ``exp(τ_0) = 64 ** -0.5 ≈ 0.125`` across
        all ablation cells.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        # ----------------------------------------------------------------
        # Learnable temperature in log-space
        # Initialised so exp(τ) == dim_head ** -0.5
        # ----------------------------------------------------------------
        self.temperature = nn.Parameter(
            torch.log(torch.tensor(dim_head ** -0.5))
        )

        self.norm    = nn.LayerNorm(dim)
        # attaches its forward hook to layer.attend.
        self.attend  = nn.Softmax(dim=-1)

        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out  = nn.Linear(inner_dim, dim,     bias=False)

        # Diagonal mask cache
        self._mask_cache: torch.Tensor = None
        self._mask_N: int = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, dim)

        Returns
        -------
        out : (B, N, dim)
        """
        x = self.norm(x)

        qkv    = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        # Scaled dot-product with learnable temperature (clamped for AMP).
        scale = self.temperature.clamp(max=4.0).exp()
        dots  = torch.matmul(q, k.transpose(-1, -2)) * scale

        # ----------------------------------------------------------------
        # Diagonal mask: set a_{i,i} = -inf so each token cannot attend
        # to itself, enforcing aggregation from other tokens.
        # ----------------------------------------------------------------
        N = dots.shape[-1]
        if (
            self._mask_N != N
            or self._mask_cache is None
            or self._mask_cache.device != dots.device
        ):
            self._mask_cache = torch.eye(N, device=dots.device, dtype=torch.bool)
            self._mask_N = N

        dots = dots.masked_fill(self._mask_cache, -torch.finfo(dots.dtype).max)

        # self.attend is the hook target for LSALogger entropy computation.
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
