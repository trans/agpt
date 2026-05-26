#!/usr/bin/env python3
"""Asym-DFT harmonic-bias prototype for AGPT attention.

Forward-only design implemented here (autograd handles the backward):
For each (Q at corpus position p_Q, K at corpus position p_K), let
substring(p_K) be the d-char substring ending at p_K, and let z_K be
its position-mod-W DFT chord. The bias term is:

    bias(K, p_Q) = (1 / C_K) Σ_j Re[ z_K[j] · exp(-i · p_Q · ω_j) ]
                 = (1 / C_K) Σ_j (z_re[j] cos(p_Q ω_j) + z_im[j] sin(p_Q ω_j))

with ω_j = 2π(j+1)/W (skip DC). The attention score becomes:

    score(Q, K) = (Q · K) / sqrt(d_head) + β_h · bias(K, p_Q)

β_h is a per-head learnable scalar, init 0 (no-op at start; the model
learns to use it). See notes/seq-len-extension/harmonic-filter-asymmetric.md
for derivation.

This module:
  - precomputes chords from a corpus (all unique d-substrings)
  - implements a forward_with_bias for AGPTForCausalLM
  - provides byte_perplexity_pytorch for direct PyTorch eval (so β is
    actually exercised at eval, which agpt_lm_eval can't do — its
    .model loader doesn't know about β).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_SRC_TOOLS = _REPO_ROOT / "src" / "tools"
for p in (str(_HERE), str(_SRC_TOOLS)):
    if p not in sys.path:
        sys.path.insert(0, p)
from agpt_ppl import build_vocab  # noqa: E402
from agpt_hf import AGPTForCausalLM  # noqa: E402


# ---------------------------------------------------------------------------
# Chord precompute
# ---------------------------------------------------------------------------

class ChordTable:
    """All d-substrings in the corpus, with their position-mod-W chords.

    Fields (all torch tensors, on chosen device):
      z_re        (n_unique, n_freq)  Σ count(p) cos(p · ω_j)
      z_im        (n_unique, n_freq)  Σ count(p) sin(p · ω_j)
      mass        (n_unique,)         Σ count(p)
      omega       (n_freq,)           2π(j+1)/W
      pos_to_idx  (N,)                pos_to_idx[p] = chord row for substring ending at p

    Positions p < d-1 use the variable-length prefix substring corpus[0:p+1].
    """

    def __init__(self, z_re: torch.Tensor, z_im: torch.Tensor,
                 mass: torch.Tensor, omega: torch.Tensor,
                 pos_to_idx: torch.Tensor):
        self.z_re = z_re
        self.z_im = z_im
        self.mass = mass
        self.omega = omega
        self.pos_to_idx = pos_to_idx
        self.n_freq = omega.numel()

    @property
    def n_unique(self) -> int:
        return self.z_re.shape[0]

    def to(self, device: torch.device) -> "ChordTable":
        return ChordTable(
            self.z_re.to(device), self.z_im.to(device),
            self.mass.to(device), self.omega.to(device),
            self.pos_to_idx.to(device),
        )


def precompute_chords(corpus_tokens: torch.Tensor, d: int, window_W: int,
                      n_freq: int) -> ChordTable:
    """Walk the corpus, hash d-substrings, build chord table. CPU-bound
    Python; for Shakespeare-scale (~1M chars) takes ~5–15s."""
    t0 = time.time()
    N = corpus_tokens.numel()
    tokens = corpus_tokens.cpu().tolist()

    # Substring tuple → unique_id
    sub_to_id: dict[tuple[int, ...], int] = {}
    pos_to_idx_list: list[int] = []
    # Per substring_id: list of positions (where it ends)
    occurrences: list[list[int]] = []

    for p in range(N):
        start = max(0, p - d + 1)
        key = tuple(tokens[start:p + 1])
        sid = sub_to_id.get(key)
        if sid is None:
            sid = len(sub_to_id)
            sub_to_id[key] = sid
            occurrences.append([])
        occurrences[sid].append(p)
        pos_to_idx_list.append(sid)

    n_unique = len(sub_to_id)
    # Frequencies: ω_j = 2π(j+1)/W (skip DC).
    omega = torch.tensor(
        [2.0 * math.pi * (j + 1) / window_W for j in range(n_freq)],
        dtype=torch.float32,
    )

    # Aggregate chords. For each substring, sum cos/sin over its positions,
    # using positions taken mod W. This matches the diagnostic.
    z_re = torch.zeros(n_unique, n_freq, dtype=torch.float32)
    z_im = torch.zeros(n_unique, n_freq, dtype=torch.float32)
    mass = torch.zeros(n_unique, dtype=torch.float32)
    for sid, plist in enumerate(occurrences):
        # Convert positions to bin (p mod W) and count duplicates.
        bins = torch.tensor([p % window_W for p in plist],
                            dtype=torch.float32)
        mass[sid] = float(len(plist))
        # angles: (n_pos, n_freq)
        angles = bins.unsqueeze(1) * omega.unsqueeze(0)
        z_re[sid] = torch.cos(angles).sum(dim=0)
        z_im[sid] = torch.sin(angles).sum(dim=0)

    pos_to_idx = torch.tensor(pos_to_idx_list, dtype=torch.long)
    print(f"  precompute_chords: {n_unique} unique substrings "
          f"of length up to {d} from {N} positions, "
          f"{time.time() - t0:.1f}s",
          file=sys.stderr)
    return ChordTable(z_re, z_im, mass, omega, pos_to_idx)


# ---------------------------------------------------------------------------
# Bias-aware attention
# ---------------------------------------------------------------------------

def _bias_term(z_re_b: torch.Tensor,   # (B, T_K, n_freq)
               z_im_b: torch.Tensor,   # (B, T_K, n_freq)
               mass_b: torch.Tensor,   # (B, T_K)
               omega: torch.Tensor,    # (n_freq,)
               pos_q_b: torch.Tensor   # (B, T_Q) int corpus positions
               ) -> torch.Tensor:      # (B, T_Q, T_K)
    # angles: (B, T_Q, n_freq)
    angles = pos_q_b.float().unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    cos_a = torch.cos(angles)  # (B, T_Q, n_freq)
    sin_a = torch.sin(angles)
    # broadcast cos_a (B, T_Q, 1, n_freq) × z_re (B, 1, T_K, n_freq)
    contrib = (cos_a.unsqueeze(2) * z_re_b.unsqueeze(1)
               + sin_a.unsqueeze(2) * z_im_b.unsqueeze(1))
    bias = contrib.sum(dim=-1)  # (B, T_Q, T_K)
    mass_clamp = mass_b.clamp_min(1.0).unsqueeze(1)  # (B, 1, T_K)
    return bias / mass_clamp


def attention_with_bias_forward(layer, x: torch.Tensor,
                                cos: torch.Tensor, sin: torch.Tensor,
                                bias_BTqTk: Optional[torch.Tensor],
                                beta: Optional[torch.Tensor]) -> torch.Tensor:
    """Forward pass for one AGPTLayer, with the harmonic bias added to
    attention scores. Mirrors AGPTLayer.forward exactly except for the
    `scores = scores + beta * bias_BTqTk[:, None]` line.

    beta: scalar tensor (broadcast across heads) OR (H,) per-head tensor.
    """
    from agpt_ppl import rope_apply  # local import to avoid circular issues
    B, T, D = x.shape
    H, HD = layer.n_heads, layer.head_dim

    residual = x
    h = layer.ln1(x)
    q = layer.wq(h).view(B, T, H, HD).transpose(1, 2)
    k = layer.wk(h).view(B, T, H, HD).transpose(1, 2)
    v = layer.wv(h).view(B, T, H, HD).transpose(1, 2)

    cos_b = cos.view(1, 1, T, HD)
    sin_b = sin.view(1, 1, T, HD)
    q = rope_apply(q, cos_b, sin_b)
    k = rope_apply(k, cos_b, sin_b)

    scale = 1.0 / math.sqrt(HD)
    scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

    if bias_BTqTk is not None and beta is not None:
        # bias_BTqTk: (B, T_Q, T_K) — same across heads, scaled by per-head β
        if beta.dim() == 0:
            scores = scores + beta * bias_BTqTk.unsqueeze(1)
        else:
            # (H,) → (1, H, 1, 1)
            scores = scores + beta.view(1, H, 1, 1) * bias_BTqTk.unsqueeze(1)

    causal = torch.triu(
        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    attn = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
    x = residual + layer.wo(attn)

    residual = x
    h = layer.ln2(x)
    h = F.relu(layer.l1(h))
    x = residual + layer.l2(h)
    return x


class HarmonicBiasModel(nn.Module):
    """Wraps an AGPTForCausalLM, replacing per-layer attention with the
    bias-aware variant. Holds the per-layer learnable β parameters.
    """

    def __init__(self, hf_model: AGPTForCausalLM, n_freq: int):
        super().__init__()
        self.hf = hf_model
        self.n_freq = n_freq
        n_layers = hf_model.config.n_layers
        n_heads = hf_model.config.n_heads
        # Per-(layer, head) β; init 0.
        self.beta = nn.Parameter(torch.zeros(n_layers, n_heads))

    @property
    def config(self):
        return self.hf.config

    def forward(self, input_ids: torch.Tensor,
                chord_z_re: Optional[torch.Tensor] = None,
                chord_z_im: Optional[torch.Tensor] = None,
                chord_mass: Optional[torch.Tensor] = None,
                omega: Optional[torch.Tensor] = None,
                pos_q: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns logits (B, T, V).
        chord_z_re/im: (B, T_K, n_freq)   — chord of K-substring at each position
        chord_mass:    (B, T_K)
        omega:         (n_freq,)
        pos_q:         (B, T_Q)           — absolute corpus positions
        When any of the chord_* is None, falls through to standard attention.
        """
        backbone = self.hf.backbone
        B, T = input_ids.shape

        if (chord_z_re is not None and chord_z_im is not None
                and chord_mass is not None and omega is not None
                and pos_q is not None):
            bias_BTqTk = _bias_term(chord_z_re, chord_z_im, chord_mass,
                                    omega, pos_q)
        else:
            bias_BTqTk = None

        x = backbone.tok_emb(input_ids)
        cos = backbone.rope_cos[:T]
        sin = backbone.rope_sin[:T]
        for layer_idx, layer in enumerate(backbone.layers):
            beta_layer = self.beta[layer_idx] if bias_BTqTk is not None else None
            x = attention_with_bias_forward(
                layer, x, cos, sin, bias_BTqTk, beta_layer,
            )
        x = backbone.final_ln(x)
        return backbone.out_head(x)


# ---------------------------------------------------------------------------
# Direct PyTorch byte_perplexity (so β is exercised at eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def byte_perplexity_pytorch(model: HarmonicBiasModel,
                            corpus_tokens: torch.Tensor,
                            chord_table: Optional[ChordTable],
                            d_window: int, device: torch.device,
                            batch_size: int = 64,
                            use_bias: bool = True) -> dict:
    """Compute byte_perplexity in the lm-eval-harness loglikelihood-rolling
    style: predict each token from its d-1 prior tokens; sum -log P and
    divide by token count. Returns dict with byte_perplexity + bits_per_byte.
    """
    model.eval()
    N = corpus_tokens.numel()
    if N < d_window + 1:
        raise ValueError(f"corpus too short for d={d_window}")
    ids = corpus_tokens.to(device)
    targets = torch.arange(d_window, N, dtype=torch.long, device=device)
    total_nll = 0.0
    n_scored = 0
    omega = chord_table.omega.to(device) if chord_table is not None else None
    for i in range(0, targets.numel(), batch_size):
        bp = targets[i:i + batch_size]
        offs = torch.arange(-d_window, 0, device=device)
        ctx_idx = bp.unsqueeze(1) + offs.unsqueeze(0)  # (B, T)
        ctx = ids[ctx_idx]
        tgt = ids[bp]

        if use_bias and chord_table is not None:
            chord_idx = chord_table.pos_to_idx[ctx_idx]  # (B, T)
            z_re = chord_table.z_re[chord_idx]
            z_im = chord_table.z_im[chord_idx]
            mass = chord_table.mass[chord_idx]
            pos_q = ctx_idx  # absolute corpus positions
            logits = model(input_ids=ctx, chord_z_re=z_re, chord_z_im=z_im,
                           chord_mass=mass, omega=omega, pos_q=pos_q)
        else:
            logits = model(input_ids=ctx)
        last_logits = logits[:, -1, :]
        nll = F.cross_entropy(last_logits, tgt, reduction='sum').item()
        total_nll += nll
        n_scored += bp.numel()
    mean_nll = total_nll / n_scored
    return {
        "byte_perplexity": math.exp(mean_nll),
        "bits_per_byte": mean_nll / math.log(2),
        "n_scored": n_scored,
    }
