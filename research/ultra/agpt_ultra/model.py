from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TinyCharRNN(nn.Module):
    """A small recurrent char LM whose prefix states are exactly cacheable."""

    def __init__(self, vocab_size: int, n_embd: int = 64, n_hidden: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.cell = nn.GRUCell(n_embd, n_hidden)
        self.head = nn.Linear(n_hidden, vocab_size)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.n_hidden, device=device)

    def step(self, token_ids: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        emb = self.embed(token_ids)
        next_state = self.cell(emb, state)
        logits = self.head(next_state)
        return logits, next_state

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, length = input_ids.shape
        state = self.initial_state(batch, input_ids.device)
        logits = []
        for t in range(length):
            step_logits, state = self.step(input_ids[:, t], state)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, length, dim)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyCharTransformer(nn.Module):
    """A small causal char transformer for sequence baselines and AGPT body tests."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_hidden = n_embd
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(n_embd, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        _, length = input_ids.shape
        if length > self.block_size:
            raise ValueError("input length exceeds configured block_size")
        positions = torch.arange(length, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
