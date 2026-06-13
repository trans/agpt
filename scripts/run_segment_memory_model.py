from __future__ import annotations

import argparse
import csv
import hashlib
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, read_text
from agpt_ultra.eval import split_text
from agpt_ultra.count_gate import CountModel, expand_extra_features, select_positions, train_gate
from agpt_ultra.run_ids import prefixed_path, resolve_run_id


MemoryEntry = tuple[torch.Tensor, int]


@dataclass
class EpochProfile:
    zero_sec: float = 0.0
    forward_sec: float = 0.0
    backward_sec: float = 0.0
    clip_sec: float = 0.0
    optimizer_sec: float = 0.0
    detach_sec: float = 0.0
    eval_sec: float = 0.0
    checkpoint_sec: float = 0.0
    updates: int = 0
    tokens: int = 0

    def add(self, other: "EpochProfile") -> None:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    @property
    def train_sec(self) -> float:
        return self.zero_sec + self.forward_sec + self.backward_sec + self.clip_sec + self.optimizer_sec + self.detach_sec


def profile_summary(profile: EpochProfile) -> str:
    train_sec = max(profile.train_sec, 1e-9)
    return (
        f"profile updates={profile.updates} tokens={profile.tokens} train_sec={train_sec:.2f} "
        f"tok_per_sec={profile.tokens / train_sec if profile.tokens else 0.0:.1f} "
        f"zero={profile.zero_sec:.2f}s/{100.0 * profile.zero_sec / train_sec:.1f}% "
        f"forward={profile.forward_sec:.2f}s/{100.0 * profile.forward_sec / train_sec:.1f}% "
        f"backward={profile.backward_sec:.2f}s/{100.0 * profile.backward_sec / train_sec:.1f}% "
        f"clip={profile.clip_sec:.2f}s/{100.0 * profile.clip_sec / train_sec:.1f}% "
        f"optim={profile.optimizer_sec:.2f}s/{100.0 * profile.optimizer_sec / train_sec:.1f}% "
        f"detach={profile.detach_sec:.2f}s/{100.0 * profile.detach_sec / train_sec:.1f}% "
        f"eval={profile.eval_sec:.2f}s checkpoint={profile.checkpoint_sec:.2f}s"
    )


def empty_stats() -> dict[str, float]:
    return {
        "gate_sum": 0.0,
        "gate_count": 0.0,
        "context_norm_sum": 0.0,
        "delta_norm_sum": 0.0,
        "token_count": 0.0,
        "no_memory_loss": 0.0,
        "context_only_loss": 0.0,
        "diagnostic_token_count": 0.0,
        "attn_entropy_sum": 0.0,
        "attn_top_weight_sum": 0.0,
        "attn_char_distance_sum": 0.0,
        "attn_token_count": 0.0,
        "attn_prev_weight_sum": 0.0,
        "attn_current_weight_sum": 0.0,
        "attn_top_is_prev_sum": 0.0,
        "first_token_loss": 0.0,
        "first_token_count": 0.0,
        "later_token_loss": 0.0,
        "later_token_count": 0.0,
        "retrieval_loss": 0.0,
        "retrieval_correct": 0.0,
        "retrieval_count": 0.0,
        "utility_loss": 0.0,
        "utility_count": 0.0,
        "utility_target_entropy_sum": 0.0,
    }


def merge_stats(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] += value


def mean_or_none(total: float, count: float) -> float | None:
    if count <= 0:
        return None
    return total / count


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        state = self.norm_attn(state + context)
        return self.norm_mlp(state + self.mlp(state))


class AttentionDecisionBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.state_to_context = nn.Linear(hidden_size, hidden_size)
        self.norm_state = nn.LayerNorm(hidden_size)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.norm_update = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        state_signal = self.state_to_context(self.norm_state(state))
        context_norm = self.norm_context(context)
        gate = torch.sigmoid(self.gate(torch.cat([state_signal, context_norm], dim=1)))
        decision = self.norm_update(context + gate * state_signal)
        return self.norm_mlp(decision + self.mlp(decision))


class LMATokenBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm_attn(tokens + attn_out)
        return self.norm_mlp(tokens + self.mlp(tokens))


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, memory_size: int | None = None, mlp_mult: int = 4) -> None:
        super().__init__()
        memory_size = memory_size or hidden_size
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(memory_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            heads,
            kdim=memory_size,
            vdim=memory_size,
            batch_first=True,
        )
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.ln_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_mult * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_mult * hidden_size, hidden_size),
        )
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if memory.shape[1] > 0:
            memory_norm = self.ln_kv(memory)
            attn_out, _ = self.attn(self.ln_q(hidden), memory_norm, memory_norm, need_weights=False)
        else:
            attn_out = torch.zeros_like(hidden)
        hidden = hidden + torch.tanh(self.gate_attn) * attn_out
        return hidden + torch.tanh(self.gate_mlp) * self.mlp(self.ln_mlp(hidden))


class PrefixCountTrie:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.counts: list[int] = [0]
        self.children: list[dict[int, int]] = [{}]

    def insert_suffixes(self, ids: list[int], max_depth: int) -> None:
        limit = len(ids)
        for start in range(limit):
            node = 0
            self.counts[node] += 1
            end = min(limit, start + max_depth)
            for pos in range(start, end):
                token = ids[pos]
                child = self.children[node].get(token)
                if child is None:
                    child = len(self.counts)
                    self.children[node][token] = child
                    self.counts.append(0)
                    self.children.append({})
                node = child
                self.counts[node] += 1


def segment_ids(
    ids: list[int],
    trie: PrefixCountTrie,
    max_depth: int,
    unique_threshold: int,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    pos = 0
    input_limit = len(ids) - 1
    while pos < input_limit:
        start = pos
        node = 0
        depth = 0
        while pos < input_limit and depth < max_depth:
            token = ids[pos]
            child = trie.children[node].get(token)
            pos += 1
            depth += 1
            if child is None:
                break
            node = child
            if trie.counts[node] <= unique_threshold:
                break
        if pos == start:
            pos += 1
        segments.append((start, pos))
    return segments


class SegmentMemoryLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        use_rope: bool,
        rope_positions: str,
        memory_state: str,
        memory_record: str,
        rnn_core: str,
        attn_interface_norm: str,
        mixing: str,
        attention_layers: int,
        attention_heads: int,
        input_rope: bool,
        feedback_gate_bias: float,
        feedback_delta_cap: float,
        prior_residual_scale: float,
        prior_residual_l2: float,
        record_aux_weight: float,
        terminal_record_aux_weight: float,
        retrieval_aux_weight: float,
        utility_aux_weight: float,
        utility_temperature: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.rope_positions = rope_positions
        self.memory_state = memory_state
        self.memory_record = memory_record
        self.rnn_core = rnn_core
        self.attn_interface_norm = attn_interface_norm
        self.mixing = mixing
        self.attention_heads = attention_heads
        self.input_rope = input_rope
        self.feedback_delta_cap = feedback_delta_cap
        self.prior_residual_scale = prior_residual_scale
        self.prior_residual_l2 = prior_residual_l2
        self.record_aux_weight = record_aux_weight
        self.terminal_record_aux_weight = terminal_record_aux_weight
        self.retrieval_aux_weight = retrieval_aux_weight
        self.utility_aux_weight = utility_aux_weight
        self.utility_temperature = utility_temperature
        self.rope_dim = hidden_size - hidden_size % 2
        self.embed_rope_dim = embedding_size - embedding_size % 2
        self.embed = nn.Embedding(vocab_size, embedding_size)
        if rnn_core == "gru":
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
            self.cell = nn.GRUCell(embedding_size, hidden_size)
        elif rnn_core == "tanh":
            self.rnn = nn.RNN(embedding_size, hidden_size, nonlinearity="tanh", batch_first=True)
            self.cell = nn.RNNCell(embedding_size, hidden_size, nonlinearity="tanh")
        else:
            raise ValueError(f"unknown rnn_core: {rnn_core}")
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_rnn = nn.Linear(hidden_size, hidden_size, bias=False)
        if attn_interface_norm == "layer":
            self.query_interface_norm = nn.LayerNorm(hidden_size)
            self.memory_interface_norm = nn.LayerNorm(hidden_size)
            self.head_state_interface_norm = nn.LayerNorm(hidden_size)
            self.head_context_interface_norm = nn.LayerNorm(hidden_size)
        elif attn_interface_norm == "context":
            self.query_interface_norm = nn.LayerNorm(hidden_size)
            self.memory_interface_norm = nn.LayerNorm(hidden_size)
            self.head_state_interface_norm = nn.Identity()
            self.head_context_interface_norm = nn.LayerNorm(hidden_size)
        elif attn_interface_norm == "attention":
            self.query_interface_norm = nn.LayerNorm(hidden_size)
            self.memory_interface_norm = nn.LayerNorm(hidden_size)
            self.head_state_interface_norm = nn.Identity()
            self.head_context_interface_norm = nn.Identity()
        elif attn_interface_norm == "none":
            self.query_interface_norm = nn.Identity()
            self.memory_interface_norm = nn.Identity()
            self.head_state_interface_norm = nn.Identity()
            self.head_context_interface_norm = nn.Identity()
        else:
            raise ValueError(f"unknown attn_interface_norm: {attn_interface_norm}")
        self.memory_writer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(hidden_size) for _ in range(attention_layers)])
        self.decision_blocks = nn.ModuleList([AttentionDecisionBlock(hidden_size) for _ in range(attention_layers)])
        self.lma_blocks = nn.ModuleList([LMATokenBlock(hidden_size, attention_heads) for _ in range(attention_layers)])
        self.gated_xattn_blocks = nn.ModuleList(
            [GatedCrossAttentionBlock(hidden_size, attention_heads) for _ in range(attention_layers)]
        )
        self.gate_update = nn.Sequential(
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.feedback_update = nn.Sequential(
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.feedback_gate = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.zeros_(self.feedback_gate.weight)
        nn.init.constant_(self.feedback_gate.bias, feedback_gate_bias)
        self.cap_fuse = nn.Linear(2 * hidden_size, hidden_size)
        self.cap_norm = nn.LayerNorm(hidden_size)
        self.gru_head = nn.Linear(hidden_size, vocab_size)
        self.late_head = nn.Linear(2 * hidden_size, vocab_size)
        self.attn_head = nn.Linear(hidden_size, vocab_size)
        self.dual_attn_head = nn.Linear(2 * hidden_size, vocab_size)
        self.record_head = nn.Linear(hidden_size, vocab_size)
        self.gated_head = nn.Linear(3 * hidden_size, vocab_size)
        self.state_head = nn.Linear(hidden_size, vocab_size)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / max(1, self.rope_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        input_inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.embed_rope_dim, 2, dtype=torch.float32) / max(1, self.embed_rope_dim))
        )
        self.register_buffer("input_rope_inv_freq", input_inv_freq, persistent=False)

    def zero_prediction_heads(self) -> None:
        for head in [
            self.gru_head,
            self.late_head,
            self.attn_head,
            self.dual_attn_head,
            self.gated_head,
            self.state_head,
        ]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def zero_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def write_memory(self, state: torch.Tensor) -> torch.Tensor:
        if self.memory_record == "raw":
            return state
        if self.memory_record == "written":
            return self.memory_writer(state)
        raise ValueError(f"unknown memory_record: {self.memory_record}")

    def late_logits(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        state_for_head = self.head_state_interface_norm(state)
        context_for_head = self.head_context_interface_norm(context)
        return self.late_head(torch.cat([state_for_head, context_for_head], dim=1))

    def previous_record_loss(
        self,
        query_state: torch.Tensor,
        query_position: int,
        memory: list[MemoryEntry],
        max_memory: int,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        stats = empty_stats()
        window = memory[-max_memory:]
        if not window:
            return None, stats
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in window], dim=0))
        query = self.query(self.query_interface_norm(query_state))
        keys = self.key(mem)
        if self.rope_positions == "char":
            query_positions = torch.tensor([query_position], device=query_state.device)
            key_positions = torch.tensor([entry[1] for entry in window], device=query_state.device)
        else:
            query_positions = torch.zeros(1, device=query_state.device)
            key_positions = torch.arange(-mem.shape[0], 0, device=query_state.device)
        query = self.apply_rope(query, query_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        target = torch.tensor([mem.shape[0] - 1], device=query_state.device)
        loss = F.cross_entropy(scores, target, reduction="sum")
        prediction = scores.argmax(dim=1)
        stats["retrieval_loss"] = float(loss.detach().item())
        stats["retrieval_correct"] = float((prediction == target).detach().sum().item())
        stats["retrieval_count"] = 1.0
        return loss, stats

    def utility_attention_loss(
        self,
        weights: torch.Tensor | None,
        values: torch.Tensor | None,
        allowed: torch.Tensor | None,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        stats = empty_stats()
        if weights is None or values is None or allowed is None:
            return None, stats
        if weights.numel() == 0 or values.numel() == 0:
            return None, stats
        candidate_logits = self.attn_head(values)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=1)
        candidate_nll = -candidate_log_probs[:, targets].T
        quality = -candidate_nll
        quality = quality.masked_fill(~allowed, -torch.inf)
        valid = allowed.any(dim=1)
        if not bool(valid.any()):
            return None, stats
        temperature = max(self.utility_temperature, 1e-6)
        target_distribution = torch.softmax(quality[valid] / temperature, dim=1).detach()
        eps = torch.finfo(weights.dtype).eps
        log_weights = weights[valid].clamp_min(eps).log()
        loss_by_token = -(target_distribution * log_weights).sum(dim=1)
        target_entropy = -(target_distribution * target_distribution.clamp_min(eps).log()).sum(dim=1)
        loss = loss_by_token.sum()
        stats["utility_loss"] = float(loss.detach().item())
        stats["utility_count"] = float(loss_by_token.numel())
        stats["utility_target_entropy_sum"] = float(target_entropy.detach().sum().item())
        return loss, stats

    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not self.use_rope or self.rope_dim == 0:
            return x
        rope_part = x[:, : self.rope_dim]
        pass_part = x[:, self.rope_dim :]
        angles = positions.to(device=x.device, dtype=x.dtype).unsqueeze(1) * self.rope_inv_freq.to(
            dtype=x.dtype
        ).unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        even = rope_part[:, 0::2]
        odd = rope_part[:, 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(1)
        if pass_part.numel() == 0:
            return rotated
        return torch.cat([rotated, pass_part], dim=1)

    def apply_input_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not self.input_rope or self.embed_rope_dim == 0:
            return x
        rope_part = x[:, : self.embed_rope_dim]
        pass_part = x[:, self.embed_rope_dim :]
        angles = positions.to(device=x.device, dtype=x.dtype).unsqueeze(1) * self.input_rope_inv_freq.to(
            dtype=x.dtype
        ).unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        even = rope_part[:, 0::2]
        odd = rope_part[:, 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(1)
        if pass_part.numel() == 0:
            return rotated
        return torch.cat([rotated, pass_part], dim=1)

    def attend(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = self.attend_with_diagnostics(hidden, memory, query_positions)
        return context

    def attend_with_diagnostics(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        stats = empty_stats()
        if not memory:
            return torch.zeros_like(hidden), stats
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in memory], dim=0))
        mem_count = mem.shape[0]
        query = self.query(self.query_interface_norm(hidden))
        keys = self.key(mem)
        values = self.value(mem)
        if self.rope_positions == "char":
            query_rope_positions = query_positions
            key_positions = torch.tensor([entry[1] for entry in memory], device=hidden.device)
        else:
            query_rope_positions = torch.zeros(hidden.shape[0], device=hidden.device)
            key_positions = torch.arange(-mem_count, 0, device=hidden.device)
        query = self.apply_rope(query, query_rope_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=1)
        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=1)
        top_weight = weights.max(dim=1).values
        top_indices = weights.argmax(dim=1)
        if self.rope_positions == "char":
            distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype=weights.dtype)
        else:
            distances = torch.arange(mem_count, 0, -1, device=hidden.device, dtype=weights.dtype).unsqueeze(0)
        mean_distance = (weights * distances).sum(dim=1)
        stats["attn_entropy_sum"] = float(entropy.detach().sum().item())
        stats["attn_top_weight_sum"] = float(top_weight.detach().sum().item())
        stats["attn_char_distance_sum"] = float(mean_distance.detach().sum().item())
        stats["attn_token_count"] = float(hidden.shape[0])
        stats["attn_prev_weight_sum"] = float(weights.detach().sum().item())
        stats["attn_top_is_prev_sum"] = float(torch.ones_like(top_indices, dtype=weights.dtype).sum().item())
        return weights @ values, stats

    def attend_dual_value(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        context_pred, stats = self.attend_with_diagnostics(hidden, memory, query_positions)
        if not memory:
            return context_pred, torch.zeros_like(hidden), stats
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in memory], dim=0))
        mem_count = mem.shape[0]
        query = self.query(self.query_interface_norm(hidden))
        keys = self.key(mem)
        values_rnn = self.value_rnn(mem)
        if self.rope_positions == "char":
            query_rope_positions = query_positions
            key_positions = torch.tensor([entry[1] for entry in memory], device=hidden.device)
        else:
            query_rope_positions = torch.zeros(hidden.shape[0], device=hidden.device)
            key_positions = torch.arange(-mem_count, 0, device=hidden.device)
        query = self.apply_rope(query, query_rope_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=1)
        return context_pred, weights @ values_rnn, stats

    def attend_causal_current_full(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
        include_self: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        stats = empty_stats()
        if hidden.numel() == 0:
            return torch.zeros_like(hidden), stats, None, None, None
        current_entries = [
            (self.write_memory(hidden[index : index + 1]), int(query_positions[index].item()))
            for index in range(hidden.shape[0])
        ]
        all_entries = memory + current_entries
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in all_entries], dim=0))
        mem_count = len(memory)
        total_count = mem.shape[0]
        query = self.query(self.query_interface_norm(hidden))
        keys = self.key(mem)
        values = self.value(mem)
        if self.rope_positions == "char":
            query_rope_positions = query_positions
            key_positions = torch.tensor([entry[1] for entry in all_entries], device=hidden.device)
        else:
            query_rope_positions = torch.zeros(hidden.shape[0], device=hidden.device)
            key_positions = torch.arange(-total_count, 0, device=hidden.device)
        query = self.apply_rope(query, query_rope_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        current_indices = torch.arange(hidden.shape[0], device=hidden.device)
        key_indices = torch.arange(total_count, device=hidden.device)
        current_limit = mem_count + current_indices[:, None] + (1 if include_self else 0)
        allowed = key_indices[None, :] < current_limit
        no_allowed = ~allowed.any(dim=1)
        scores = scores.masked_fill(~allowed, -torch.inf)
        if no_allowed.any():
            scores = scores.clone()
            scores[no_allowed] = 0.0
        weights = torch.softmax(scores, dim=1)
        if no_allowed.any():
            weights = weights.masked_fill(no_allowed[:, None], 0.0)
        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=1)
        top_weight = weights.max(dim=1).values
        top_indices = weights.argmax(dim=1)
        if self.rope_positions == "char":
            distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype=weights.dtype)
        else:
            distances = torch.arange(total_count, 0, -1, device=hidden.device, dtype=weights.dtype).unsqueeze(0)
        mean_distance = (weights * distances).sum(dim=1)
        if mem_count > 0:
            prev_weight = weights[:, :mem_count].sum(dim=1)
            top_is_prev = (top_indices < mem_count).to(dtype=weights.dtype)
        else:
            prev_weight = torch.zeros(hidden.shape[0], device=hidden.device, dtype=weights.dtype)
            top_is_prev = torch.zeros(hidden.shape[0], device=hidden.device, dtype=weights.dtype)
        current_weight = 1.0 - prev_weight
        stats["attn_entropy_sum"] = float(entropy.detach().sum().item())
        stats["attn_top_weight_sum"] = float(top_weight.detach().sum().item())
        stats["attn_char_distance_sum"] = float(mean_distance.detach().sum().item())
        stats["attn_token_count"] = float(hidden.shape[0])
        stats["attn_prev_weight_sum"] = float(prev_weight.detach().sum().item())
        stats["attn_current_weight_sum"] = float(current_weight.detach().sum().item())
        stats["attn_top_is_prev_sum"] = float(top_is_prev.detach().sum().item())
        return weights @ values, stats, weights, values, allowed

    def attend_causal_current(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
        include_self: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        context, stats, _, _, _ = self.attend_causal_current_full(
            hidden,
            memory,
            query_positions,
            include_self=include_self,
        )
        return context, stats

    def lma_token_decision(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        max_memory: int,
    ) -> torch.Tensor:
        if hidden.numel() == 0:
            return hidden
        mem_entries = memory[-max_memory:]
        if mem_entries:
            mem = self.memory_interface_norm(torch.cat([entry[0] for entry in mem_entries], dim=0))
        else:
            mem = hidden.new_zeros((0, self.hidden_size))
        current = self.query_interface_norm(hidden).unsqueeze(1)
        if mem.shape[0] > 0:
            memory_tokens = mem.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            tokens = torch.cat([current, memory_tokens], dim=1)
        else:
            tokens = current
        for block in self.lma_blocks:
            tokens = block(tokens)
        return tokens[:, 0, :]

    def segment_loss_token_feedback(
        self,
        ids: torch.Tensor,
        segments: list[tuple[int, int]],
        memory: list[MemoryEntry],
        max_memory: int,
        collect_diagnostics: bool,
        carried_hidden: torch.Tensor | None,
        carry_hidden: bool,
        token_feedback: str,
        prior_log_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, list[MemoryEntry], torch.Tensor | None, dict[str, float]]:
        losses: list[torch.Tensor] = []
        token_count = 0
        stats = empty_stats()
        next_carried_hidden = carried_hidden
        for start, end in segments:
            prev_state = (
                next_carried_hidden.squeeze(0)
                if carry_hidden and next_carried_hidden is not None
                else torch.zeros(1, self.hidden_size, device=ids.device)
            )
            current_entries: list[MemoryEntry] = []
            final_state = prev_state
            final_context = torch.zeros_like(prev_state)
            for pos in range(start, end):
                token = ids[pos].view(1)
                emb = self.embed(token)
                emb = self.apply_input_rope(emb, torch.tensor([pos], device=ids.device))
                raw_state = self.cell(emb, prev_state)
                query_positions = torch.tensor([pos], device=ids.device)
                history = memory[-max_memory:] + current_entries
                if token_feedback == "dual-gated":
                    context, context_rnn, attn_stats = self.attend_dual_value(raw_state, history, query_positions)
                else:
                    context, attn_stats = self.attend_with_diagnostics(raw_state, history, query_positions)
                    context_rnn = context
                has_context = bool(history)
                if self.mixing == "attn-only":
                    logits = self.attn_head(context)
                else:
                    logits = self.late_logits(raw_state, context)
                if prior_log_probs is not None:
                    residual_logits = self.prior_residual_scale * logits
                    logits = prior_log_probs[pos : pos + 1] + residual_logits
                else:
                    residual_logits = None
                target = ids[pos + 1].view(1)
                token_loss = F.cross_entropy(logits, target, reduction="none")
                token_segment_loss = token_loss.sum()
                if self.training and residual_logits is not None and self.prior_residual_l2 > 0.0:
                    token_segment_loss = token_segment_loss + self.prior_residual_l2 * residual_logits.square().mean()
                losses.append(token_segment_loss)
                token_count += 1
                merge_stats(stats, attn_stats)
                stats["context_norm_sum"] += float(context.detach().norm(dim=1).sum().item())
                stats["token_count"] += 1.0
                if collect_diagnostics:
                    if pos == start:
                        stats["first_token_loss"] += float(token_loss.detach().sum().item())
                        stats["first_token_count"] += 1.0
                    else:
                        stats["later_token_loss"] += float(token_loss.detach().sum().item())
                        stats["later_token_count"] += 1.0

                if token_feedback == "context":
                    next_state = context if has_context else raw_state
                else:
                    feedback_input = torch.cat([raw_state, context_rnn], dim=1)
                    feedback_update = self.feedback_update(feedback_input)
                    feedback_gate = torch.sigmoid(self.feedback_gate(feedback_input))
                    delta = feedback_gate * feedback_update
                    delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    delta = delta * (self.feedback_delta_cap / delta_norm).clamp_max(1.0)
                    next_state = raw_state + delta
                    stats["gate_sum"] += float(feedback_gate.detach().sum().item())
                    stats["gate_count"] += float(feedback_gate.numel())
                    stats["delta_norm_sum"] += float(delta.detach().norm(dim=1).sum().item())
                current_entries.append((self.write_memory(next_state), pos))
                final_state = next_state
                final_context = context
                prev_state = next_state
            if self.memory_state == "fused":
                cap_input = torch.cat([final_state, final_context], dim=1)
                cap_state = self.cap_norm(torch.tanh(self.cap_fuse(cap_input)))
            else:
                cap_state = final_state
            memory.append((self.write_memory(cap_state), end - 1))
            if len(memory) > max_memory:
                memory = memory[-max_memory:]
            next_carried_hidden = final_state.unsqueeze(0)
        if not losses:
            return torch.zeros((), device=ids.device), 0, memory, next_carried_hidden, stats
        return torch.stack(losses).sum(), token_count, memory, next_carried_hidden, stats

    def segment_loss(
        self,
        ids: torch.Tensor,
        segments: list[tuple[int, int]],
        memory: list[MemoryEntry],
        max_memory: int,
        collect_diagnostics: bool = False,
        carried_hidden: torch.Tensor | None = None,
        carry_hidden: bool = False,
        feedback_state: str = "gru",
        token_feedback: str = "none",
        prior_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, list[MemoryEntry], torch.Tensor | None, dict[str, float]]:
        if token_feedback != "none":
            return self.segment_loss_token_feedback(
                ids,
                segments,
                memory,
                max_memory,
                collect_diagnostics,
                carried_hidden,
                carry_hidden,
                token_feedback,
                prior_log_probs,
            )
        losses: list[torch.Tensor] = []
        token_count = 0
        stats = empty_stats()
        next_carried_hidden = carried_hidden
        for start, end in segments:
            tokens = ids[start:end]
            initial_hidden = next_carried_hidden if carry_hidden and next_carried_hidden is not None else self.zero_state(ids.device)
            embeddings = self.embed(tokens)
            embeddings = self.apply_input_rope(embeddings, torch.arange(start, end, device=ids.device))
            hidden_seq, hidden = self.rnn(embeddings.unsqueeze(0), initial_hidden)
            hidden_seq = hidden_seq.squeeze(0)
            query_positions = torch.arange(start, end, device=ids.device)
            utility_weights: torch.Tensor | None = None
            utility_values: torch.Tensor | None = None
            utility_allowed: torch.Tensor | None = None
            if self.mixing == "none":
                context = torch.zeros_like(hidden_seq)
                attn_stats = empty_stats()
                state = hidden_seq
                logits = self.gru_head(hidden_seq)
            elif self.mixing == "cross":
                state = hidden_seq
                for block in self.cross_blocks:
                    state = block(state, self.attend(state, memory[-max_memory:], query_positions))
                context, attn_stats = self.attend_with_diagnostics(state, memory[-max_memory:], query_positions)
                logits = self.state_head(state)
            elif self.mixing == "gated":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                gate_input = torch.cat([hidden_seq, context], dim=1)
                update = self.gate_update(gate_input)
                gate = torch.sigmoid(self.gate(gate_input))
                delta = gate * update
                state = hidden_seq + delta
                logits = self.gated_head(torch.cat([hidden_seq, state, context], dim=1))
                stats["gate_sum"] += float(gate.detach().sum().item())
                stats["gate_count"] += float(gate.numel())
                stats["delta_norm_sum"] += float(delta.detach().norm(dim=1).sum().item())
            elif self.mixing == "attn-only":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "current-attn":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                )
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
            elif self.mixing == "current-only-attn":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    [],
                    query_positions,
                )
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
            elif self.mixing == "prefix-attn":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                    include_self=False,
                )
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
            elif self.mixing == "prefix-attn-only":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                    include_self=False,
                )
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "current-attn-only-final":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                    include_self=True,
                )
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "local-attn-final":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    [],
                    query_positions,
                    include_self=True,
                )
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "dual-attn-final":
                local_context, local_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    [],
                    query_positions,
                    include_self=True,
                )
                long_context, long_stats = self.attend_with_diagnostics(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                )
                context = long_context
                attn_stats = long_stats
                state = hidden_seq
                logits = self.dual_attn_head(torch.cat([local_context, long_context], dim=1))
            elif self.mixing == "late-block":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = hidden_seq
                for block in self.cross_blocks:
                    state = block(state, context)
                logits = self.state_head(state)
            elif self.mixing == "attention-decision":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = context
                for block in self.decision_blocks:
                    state = block(hidden_seq, state)
                logits = self.state_head(state)
            elif self.mixing == "lma-token":
                context = torch.zeros_like(hidden_seq)
                attn_stats = empty_stats()
                state = self.lma_token_decision(hidden_seq, memory, max_memory)
                logits = self.state_head(state)
            elif self.mixing == "gated-xattn":
                mem_entries = memory[-max_memory:]
                if mem_entries:
                    mem = self.memory_interface_norm(torch.cat([entry[0] for entry in mem_entries], dim=0)).unsqueeze(0)
                else:
                    mem = hidden_seq.new_zeros((1, 0, self.hidden_size))
                state_batch = hidden_seq.unsqueeze(0)
                for block in self.gated_xattn_blocks:
                    state_batch = block(state_batch, mem)
                state = state_batch.squeeze(0)
                context = state - hidden_seq
                attn_stats = empty_stats()
                logits = self.gru_head(state)
                for block in self.gated_xattn_blocks:
                    stats["gate_sum"] += float(torch.tanh(block.gate_attn).detach().abs().item())
                    stats["gate_sum"] += float(torch.tanh(block.gate_mlp).detach().abs().item())
                    stats["gate_count"] += 2.0
            else:
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
                if collect_diagnostics:
                    zero_context = torch.zeros_like(context)
                    zero_hidden = torch.zeros_like(hidden_seq)
                    no_memory_logits = self.late_logits(hidden_seq, zero_context)
                    context_only_logits = self.late_logits(zero_hidden, context)
                    targets = ids[start + 1 : end + 1]
                    stats["no_memory_loss"] += float(
                        F.cross_entropy(no_memory_logits, targets, reduction="sum").detach().item()
                    )
                    stats["context_only_loss"] += float(
                        F.cross_entropy(context_only_logits, targets, reduction="sum").detach().item()
                    )
                    stats["diagnostic_token_count"] += float(tokens.numel())
            if prior_log_probs is not None:
                residual_logits = self.prior_residual_scale * logits
                logits = prior_log_probs[start:end] + residual_logits
            else:
                residual_logits = None
            merge_stats(stats, attn_stats)
            stats["context_norm_sum"] += float(context.detach().norm(dim=1).sum().item())
            stats["token_count"] += float(tokens.numel())
            targets = ids[start + 1 : end + 1]
            token_losses = F.cross_entropy(logits, targets, reduction="none")
            segment_loss = token_losses.sum()
            if self.training and residual_logits is not None and self.prior_residual_l2 > 0.0:
                segment_loss = segment_loss + self.prior_residual_l2 * residual_logits.square().mean(dim=1).sum()
            terminal = state[-1:].contiguous()
            if self.training and self.record_aux_weight > 0.0:
                record_seq = self.write_memory(hidden_seq)
                record_logits = self.record_head(record_seq)
                record_losses = F.cross_entropy(record_logits, targets, reduction="none")
                segment_loss = segment_loss + self.record_aux_weight * record_losses.sum()
            if self.training and self.retrieval_aux_weight > 0.0:
                retrieval_loss, retrieval_stats = self.previous_record_loss(
                    terminal,
                    end - 1,
                    memory,
                    max_memory,
                )
                merge_stats(stats, retrieval_stats)
                if retrieval_loss is not None:
                    segment_loss = segment_loss + self.retrieval_aux_weight * retrieval_loss
            if self.training and self.utility_aux_weight > 0.0:
                utility_loss, utility_stats = self.utility_attention_loss(
                    utility_weights,
                    utility_values,
                    utility_allowed,
                    targets,
                )
                merge_stats(stats, utility_stats)
                if utility_loss is not None:
                    segment_loss = segment_loss + self.utility_aux_weight * utility_loss
            losses.append(segment_loss)
            if collect_diagnostics and token_losses.numel() > 0:
                stats["first_token_loss"] += float(token_losses[:1].detach().sum().item())
                stats["first_token_count"] += 1.0
                if token_losses.numel() > 1:
                    stats["later_token_loss"] += float(token_losses[1:].detach().sum().item())
                    stats["later_token_count"] += float(token_losses.numel() - 1)
            token_count += int(tokens.numel())
            if self.memory_state == "fused":
                cap_input = torch.cat([hidden_seq[-1:].contiguous(), context[-1:].contiguous()], dim=1)
                cap_state = self.cap_norm(torch.tanh(self.cap_fuse(cap_input)))
            else:
                cap_state = terminal
            memory_record = self.write_memory(cap_state)
            if self.training and self.terminal_record_aux_weight > 0.0:
                terminal_record_logits = self.record_head(memory_record)
                terminal_record_target = ids[end : end + 1]
                terminal_record_loss = F.cross_entropy(terminal_record_logits, terminal_record_target, reduction="sum")
                losses.append(self.terminal_record_aux_weight * terminal_record_loss)
            memory.append((memory_record, end - 1))
            if len(memory) > max_memory:
                memory = memory[-max_memory:]
            if feedback_state == "context":
                next_state = context[-1:].contiguous()
            elif feedback_state == "gated":
                feedback_input = torch.cat([terminal, context[-1:].contiguous()], dim=1)
                feedback_update = self.feedback_update(feedback_input)
                feedback_gate = torch.sigmoid(self.feedback_gate(feedback_input))
                next_state = terminal + feedback_gate * feedback_update
            else:
                next_state = terminal
            next_carried_hidden = next_state.unsqueeze(0)
        if not losses:
            return torch.zeros((), device=ids.device), 0, memory, next_carried_hidden, stats
        return torch.stack(losses).sum(), token_count, memory, next_carried_hidden, stats


@dataclass(frozen=True)
class SegmentMemoryRow:
    epoch: int
    step: int
    train_nll: float | None
    val_nll: float | None
    val_ppl: float | None
    val_bpc: float | None
    segment_count: int
    mean_segment_len: float
    max_segment_len: int
    mean_gate: float | None
    mean_context_norm: float | None
    mean_delta_norm: float | None
    no_memory_ppl: float | None
    context_only_ppl: float | None
    mean_attn_entropy: float | None
    mean_attn_top_weight: float | None
    mean_attn_char_distance: float | None
    mean_attn_prev_weight: float | None
    mean_attn_current_weight: float | None
    mean_attn_top_is_prev: float | None
    first_token_ppl: float | None
    later_token_ppl: float | None
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a char LM with trie-derived variable segments and attention over segment states.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--train-input", type=Path, default=None)
    parser.add_argument("--eval-input", type=Path, default=None)
    parser.add_argument("--vocab-input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/segment_memory_model.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--unique-threshold", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--rnn-core", choices=["gru", "tanh"], default="gru")
    parser.add_argument("--attn-interface-norm", choices=["none", "attention", "context", "layer"], default="none")
    parser.add_argument("--max-memory", type=int, default=32)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--input-rope", action="store_true")
    parser.add_argument("--rope-positions", choices=["segment", "char"], default="segment")
    parser.add_argument("--memory-state", choices=["terminal", "fused"], default="terminal")
    parser.add_argument("--memory-record", choices=["raw", "written"], default="raw")
    parser.add_argument("--record-aux-weight", type=float, default=0.0)
    parser.add_argument("--terminal-record-aux-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-aux-weight", type=float, default=0.0)
    parser.add_argument("--utility-aux-weight", type=float, default=0.0)
    parser.add_argument("--utility-temperature", type=float, default=1.0)
    parser.add_argument("--carry-hidden", action="store_true")
    parser.add_argument("--feedback-state", choices=["gru", "context", "gated"], default="gru")
    parser.add_argument("--token-feedback", choices=["none", "context", "gated", "dual-gated"], default="none")
    parser.add_argument("--feedback-gate-bias", type=float, default=-6.0)
    parser.add_argument("--feedback-delta-cap", type=float, default=1.0)
    parser.add_argument("--prior-residual-scale", type=float, default=1.0)
    parser.add_argument("--prior-residual-l2", type=float, default=0.0)
    parser.add_argument(
        "--mixing",
        choices=[
            "none",
            "late",
            "cross",
            "gated",
            "attn-only",
            "current-attn",
            "current-only-attn",
            "prefix-attn",
            "prefix-attn-only",
            "current-attn-only-final",
            "local-attn-final",
            "dual-attn-final",
            "late-block",
            "attention-decision",
            "lma-token",
            "gated-xattn",
        ],
        default="late",
    )
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--segments-per-update", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-chars", type=int, default=None)
    parser.add_argument("--eval-max-chars", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-initial", action="store_true")
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--best-checkpoint-output", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--progress-every-steps", type=int, default=0)
    parser.add_argument("--count-prior", choices=["none", "frozen"], default="none")
    parser.add_argument("--count-prior-depth", type=int, default=8)
    parser.add_argument("--count-prior-valid-ratio", type=float, default=0.05)
    parser.add_argument("--count-prior-epochs", type=int, default=100)
    parser.add_argument("--count-prior-lr", type=float, default=0.05)
    parser.add_argument("--count-prior-max-fit-positions", type=int, default=50000)
    parser.add_argument("--count-prior-extra-features", default="entropy_delta,suffix_stats")
    parser.add_argument("--count-prior-cache", type=Path, default=None)
    return parser.parse_args()


def segment_stats(segments: list[tuple[int, int]]) -> tuple[float, int]:
    lengths = [end - start for start, end in segments]
    return sum(lengths) / max(1, len(lengths)), max(lengths, default=0)


def fit_count_gate_prior(
    train_ids: list[int],
    vocab_size: int,
    depth: int,
    valid_ratio: float,
    epochs: int,
    lr: float,
    max_fit_positions: int,
    extra_features: str,
    seed: int,
) -> tuple[CountModel, list[float], list[dict[str, float]]]:
    split = int(len(train_ids) * (1.0 - valid_ratio))
    split = max(depth + 1, min(split, len(train_ids) - depth - 1))
    count_train = bytes(train_ids[:split])
    valid = bytes(train_ids[split:])
    model = CountModel(count_train, vocab_size, depth, expand_extra_features(extra_features))
    model.build()
    fit_positions = select_positions(len(valid), max_fit_positions, 1, seed)
    theta, history = train_gate(model, valid, depth, fit_positions, epochs, lr)
    return model, theta, history


def precompute_count_prior_log_probs(
    ids: list[int],
    model: CountModel,
    theta: list[float],
    depth: int,
    device: torch.device,
) -> torch.Tensor:
    if len(ids) < 2:
        return torch.empty((0, model.vocab_size), dtype=torch.float32, device=device)
    tokens = bytes(ids)
    rows = torch.empty((len(ids) - 1, model.vocab_size), dtype=torch.float32)
    for pos in range(len(ids) - 1):
        target_pos = pos + 1
        ctx = tokens[max(0, target_pos - depth) : target_pos]
        rows[pos] = torch.tensor(model.gated_distribution(ctx, theta), dtype=torch.float32).log()
    return rows.to(device)


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(bytes(ids)).hexdigest()


def count_prior_cache_meta(args: argparse.Namespace, train_ids: list[int], eval_ids: list[int], vocab: CharVocab) -> dict[str, object]:
    return {
        "train_digest": ids_digest(train_ids),
        "eval_digest": ids_digest(eval_ids),
        "vocab_chars": vocab.chars,
        "depth": args.count_prior_depth,
        "valid_ratio": args.count_prior_valid_ratio,
        "epochs": args.count_prior_epochs,
        "lr": args.count_prior_lr,
        "max_fit_positions": args.count_prior_max_fit_positions,
        "extra_features": args.count_prior_extra_features,
        "seed": args.seed,
    }


def count_prior_cache_matches(cache: dict, meta: dict[str, object]) -> bool:
    return cache.get("meta") == meta


@torch.no_grad()
def evaluate(
    model: SegmentMemoryLM,
    ids: torch.Tensor,
    segments: list[tuple[int, int]],
    max_memory: int,
    carry_hidden: bool,
    feedback_state: str,
    token_feedback: str,
    prior_log_probs: torch.Tensor | None = None,
) -> tuple[float, float, float, dict[str, float]]:
    model.eval()
    memory: list[MemoryEntry] = []
    loss, tokens, _, _, stats = model.segment_loss(
        ids,
        segments,
        memory,
        max_memory,
        collect_diagnostics=True,
        carry_hidden=carry_hidden,
        feedback_state=feedback_state,
        token_feedback=token_feedback,
        prior_log_probs=prior_log_probs,
    )
    if tokens == 0:
        raise ValueError("cannot evaluate no tokens")
    nll = float(loss.item()) / tokens
    return nll, math.exp(nll), nll / math.log(2.0), stats


def row_stats(stats: dict[str, float]) -> tuple[float | None, float | None, float | None]:
    return (
        mean_or_none(stats["gate_sum"], stats["gate_count"]),
        mean_or_none(stats["context_norm_sum"], stats["token_count"]),
        mean_or_none(stats["delta_norm_sum"], stats["token_count"]),
    )


def ppl_from_loss(total_loss: float, count: float) -> float | None:
    if count <= 0:
        return None
    nll = total_loss / count
    if nll > 700.0:
        return float("inf")
    return math.exp(nll)


def diagnostic_row_stats(
    stats: dict[str, float],
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    return (
        ppl_from_loss(stats["no_memory_loss"], stats["diagnostic_token_count"]),
        ppl_from_loss(stats["context_only_loss"], stats["diagnostic_token_count"]),
        mean_or_none(stats["attn_entropy_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_top_weight_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_char_distance_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_prev_weight_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_current_weight_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_top_is_prev_sum"], stats["attn_token_count"]),
        ppl_from_loss(stats["first_token_loss"], stats["first_token_count"]),
        ppl_from_loss(stats["later_token_loss"], stats["later_token_count"]),
    )


def save_checkpoint(
    path: Path,
    model: SegmentMemoryLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    vocab: CharVocab,
    args: argparse.Namespace,
    best_val_ppl: float | None = None,
    best_epoch: int | None = None,
    best_step: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "vocab_chars": vocab.chars,
        "args": vars(args),
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "best_step": best_step,
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    args.run_id = resolve_run_id(args.run_id, resume=args.resume_checkpoint is not None)
    args.output = prefixed_path(args.output, args.run_id)
    if args.checkpoint_output is None:
        args.checkpoint_output = args.output.with_suffix(".pt")
    else:
        args.checkpoint_output = prefixed_path(args.checkpoint_output, args.run_id)
    if args.best_checkpoint_output is None:
        args.best_checkpoint_output = args.output.with_suffix(".best.pt")
    else:
        args.best_checkpoint_output = prefixed_path(args.best_checkpoint_output, args.run_id)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    text = read_text(args.input)
    split = None if args.train_input is not None and args.eval_input is not None else split_text(text, train_fraction=args.train_fraction)
    train_source_text = read_text(args.train_input) if args.train_input is not None else split.train_text
    if args.max_train_chars is not None:
        train_text = train_source_text[: args.max_train_chars]
    else:
        train_text = train_source_text
    if args.eval_input is not None:
        eval_source_text = read_text(args.eval_input)
    else:
        if split is None:
            split = split_text(text, train_fraction=args.train_fraction)
        eval_source_text = split.val_text
    eval_text = eval_source_text[: args.eval_max_chars] if args.eval_max_chars is not None else eval_source_text
    vocab_text = read_text(args.vocab_input) if args.vocab_input is not None else text
    vocab = CharVocab.from_text(vocab_text)
    train_ids_list = vocab.encode(train_text)
    eval_ids_list = vocab.encode(eval_text)

    trie = PrefixCountTrie(vocab.size)
    trie.insert_suffixes(train_ids_list, args.max_depth)
    train_segments = segment_ids(train_ids_list, trie, args.max_depth, args.unique_threshold)
    eval_segments = segment_ids(eval_ids_list, trie, args.max_depth, args.unique_threshold)
    mean_segment_len, max_segment_len = segment_stats(train_segments)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids = torch.tensor(train_ids_list, dtype=torch.long, device=device)
    eval_ids = torch.tensor(eval_ids_list, dtype=torch.long, device=device)
    train_prior_log_probs: torch.Tensor | None = None
    eval_prior_log_probs: torch.Tensor | None = None
    if args.count_prior == "frozen":
        prior_started = time.perf_counter()
        count_history: list[dict[str, float]] = []
        cache_meta = count_prior_cache_meta(args, train_ids_list, eval_ids_list, vocab)
        cache_loaded = False
        if args.count_prior_cache is not None and args.count_prior_cache.exists():
            cache = torch.load(args.count_prior_cache, map_location="cpu", weights_only=False)
            if count_prior_cache_matches(cache, cache_meta):
                train_prior_log_probs = cache["train_log_probs"].to(device)
                eval_prior_log_probs = cache["eval_log_probs"].to(device)
                count_history = cache.get("history", [])
                cache_loaded = True
            else:
                print(f"count_prior_cache_miss={args.count_prior_cache}", flush=True)
        if train_prior_log_probs is None or eval_prior_log_probs is None:
            count_model, count_theta, count_history = fit_count_gate_prior(
                train_ids_list,
                vocab.size,
                args.count_prior_depth,
                args.count_prior_valid_ratio,
                args.count_prior_epochs,
                args.count_prior_lr,
                args.count_prior_max_fit_positions,
                args.count_prior_extra_features,
                args.seed,
            )
            train_prior_log_probs = precompute_count_prior_log_probs(
                train_ids_list,
                count_model,
                count_theta,
                args.count_prior_depth,
                device,
            )
            eval_prior_log_probs = precompute_count_prior_log_probs(
                eval_ids_list,
                count_model,
                count_theta,
                args.count_prior_depth,
                device,
            )
            if args.count_prior_cache is not None:
                args.count_prior_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp_cache = args.count_prior_cache.with_suffix(f"{args.count_prior_cache.suffix}.tmp")
                torch.save(
                    {
                        "meta": cache_meta,
                        "history": count_history,
                        "train_log_probs": train_prior_log_probs.detach().cpu(),
                        "eval_log_probs": eval_prior_log_probs.detach().cpu(),
                    },
                    tmp_cache,
                )
                tmp_cache.replace(args.count_prior_cache)
        print(
            f"count_prior=frozen depth={args.count_prior_depth} "
            f"features={args.count_prior_extra_features} "
            f"fit_ppl={count_history[-1]['ppl'] if count_history else float('nan'):.3f} "
            f"cache_loaded={cache_loaded} cache={args.count_prior_cache} "
            f"build_precompute_sec={time.perf_counter() - prior_started:.2f}",
            flush=True,
        )
    model = SegmentMemoryLM(
        vocab.size,
        args.embedding_size,
        args.hidden_size,
        use_rope=not args.no_rope,
        rope_positions=args.rope_positions,
        memory_state=args.memory_state,
        memory_record=args.memory_record,
        rnn_core=args.rnn_core,
        attn_interface_norm=args.attn_interface_norm,
        mixing=args.mixing,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        input_rope=args.input_rope,
        feedback_gate_bias=args.feedback_gate_bias,
        feedback_delta_cap=args.feedback_delta_cap,
        prior_residual_scale=args.prior_residual_scale,
        prior_residual_l2=args.prior_residual_l2,
        record_aux_weight=args.record_aux_weight,
        terminal_record_aux_weight=args.terminal_record_aux_weight,
        retrieval_aux_weight=args.retrieval_aux_weight,
        utility_aux_weight=args.utility_aux_weight,
        utility_temperature=args.utility_temperature,
    ).to(device)
    if args.count_prior == "frozen":
        model.zero_prediction_heads()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    step = 0
    best_val_ppl: float | None = None
    best_epoch: int | None = None
    best_step: int | None = None
    checkpoint = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    if checkpoint is not None:
        if tuple(checkpoint["vocab_chars"]) != vocab.chars:
            raise ValueError("checkpoint vocab does not match input text vocab")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        completed_epoch = int(checkpoint["epoch"])
        step = int(checkpoint["step"])
        start_epoch = completed_epoch + 1
        best_val_ppl = checkpoint.get("best_val_ppl")
        best_epoch = checkpoint.get("best_epoch")
        best_step = checkpoint.get("best_step")
        print(
            f"resumed checkpoint={args.resume_checkpoint} completed_epoch={completed_epoch} "
            f"start_epoch={start_epoch} step={step}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(
        asdict(
            SegmentMemoryRow(
                epoch=0,
                step=0,
                train_nll=None,
                val_nll=None,
                val_ppl=None,
                val_bpc=None,
                segment_count=0,
                mean_segment_len=0.0,
                max_segment_len=0,
                mean_gate=None,
                mean_context_norm=None,
                mean_delta_norm=None,
                no_memory_ppl=None,
                context_only_ppl=None,
                mean_attn_entropy=None,
                mean_attn_top_weight=None,
                mean_attn_char_distance=None,
                mean_attn_prev_weight=None,
                mean_attn_current_weight=None,
                mean_attn_top_is_prev=None,
                first_token_ppl=None,
                later_token_ppl=None,
                runtime_sec=0.0,
                peak_rss_kb=0,
            )
        ).keys()
    )
    append_output = args.resume_checkpoint is not None and args.output.exists()
    with args.output.open("a" if append_output else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append_output:
            writer.writeheader()
        print(
            f"train_chars={len(train_ids_list)} train_segments={len(train_segments)} "
            f"eval_chars={len(eval_ids_list)} eval_segments={len(eval_segments)} "
            f"mean_segment_len={mean_segment_len:.2f} max_segment_len={max_segment_len} "
            f"memory_state={args.memory_state} rope_positions={args.rope_positions} "
            f"memory_record={args.memory_record} rnn_core={args.rnn_core} "
            f"attn_interface_norm={args.attn_interface_norm} "
            f"mixing={args.mixing} attention_layers={args.attention_layers} attention_heads={args.attention_heads} "
            f"carry_hidden={args.carry_hidden} "
            f"feedback_state={args.feedback_state} token_feedback={args.token_feedback} input_rope={args.input_rope} "
            f"feedback_gate_bias={args.feedback_gate_bias} feedback_delta_cap={args.feedback_delta_cap} "
            f"prior_residual_scale={args.prior_residual_scale} prior_residual_l2={args.prior_residual_l2} "
            f"record_aux_weight={args.record_aux_weight} terminal_record_aux_weight={args.terminal_record_aux_weight} "
            f"retrieval_aux_weight={args.retrieval_aux_weight} "
            f"utility_aux_weight={args.utility_aux_weight} utility_temperature={args.utility_temperature} "
            f"count_prior={args.count_prior} count_prior_depth={args.count_prior_depth} "
            f"checkpoint_output={args.checkpoint_output} best_checkpoint_output={args.best_checkpoint_output} "
            f"profile={args.profile} "
            f"device={device} output={args.output}",
            flush=True,
        )
        if args.eval_initial:
            t0 = time.perf_counter()
            val_nll, val_ppl, val_bpc, val_stats = evaluate(
                model,
                eval_ids,
                eval_segments,
                args.max_memory,
                carry_hidden=args.carry_hidden,
                feedback_state=args.feedback_state,
                token_feedback=args.token_feedback,
                prior_log_probs=eval_prior_log_probs,
            )
            mean_gate, mean_context_norm, mean_delta_norm = row_stats(val_stats)
            (
                no_memory_ppl,
                context_only_ppl,
                mean_attn_entropy,
                mean_attn_top_weight,
                mean_attn_char_distance,
                mean_attn_prev_weight,
                mean_attn_current_weight,
                mean_attn_top_is_prev,
                first_token_ppl,
                later_token_ppl,
            ) = diagnostic_row_stats(val_stats)
            row = SegmentMemoryRow(
                epoch=0,
                step=step,
                train_nll=None,
                val_nll=val_nll,
                val_ppl=val_ppl,
                val_bpc=val_bpc,
                segment_count=len(train_segments),
                mean_segment_len=mean_segment_len,
                max_segment_len=max_segment_len,
                mean_gate=mean_gate,
                mean_context_norm=mean_context_norm,
                mean_delta_norm=mean_delta_norm,
                no_memory_ppl=no_memory_ppl,
                context_only_ppl=context_only_ppl,
                mean_attn_entropy=mean_attn_entropy,
                mean_attn_top_weight=mean_attn_top_weight,
                mean_attn_char_distance=mean_attn_char_distance,
                mean_attn_prev_weight=mean_attn_prev_weight,
                mean_attn_current_weight=mean_attn_current_weight,
                mean_attn_top_is_prev=mean_attn_top_is_prev,
                first_token_ppl=first_token_ppl,
                later_token_ppl=later_token_ppl,
                runtime_sec=time.perf_counter() - started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(row))
            handle.flush()
            print(
                f"epoch=0 initial step={step} val_ppl={val_ppl:.3f} "
                f"eval_sec={time.perf_counter() - t0:.2f} runtime_sec={row.runtime_sec:.2f} "
                f"peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
            if best_val_ppl is None or val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_epoch = 0
                best_step = step
                save_checkpoint(
                    args.best_checkpoint_output,
                    model,
                    optimizer,
                    0,
                    step,
                    vocab,
                    args,
                    best_val_ppl=best_val_ppl,
                    best_epoch=best_epoch,
                    best_step=best_step,
                )
                print(
                    f"best_checkpoint={args.best_checkpoint_output} "
                    f"best_epoch={best_epoch} best_step={best_step} best_val_ppl={best_val_ppl:.3f}",
                    flush=True,
                )
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            memory: list[MemoryEntry] = []
            carried_hidden: torch.Tensor | None = None
            epoch_profile = EpochProfile()
            for offset in range(0, len(train_segments), args.segments_per_update):
                step += 1
                batch_segments = train_segments[offset : offset + args.segments_per_update]
                step_profile = EpochProfile()
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                step_profile.zero_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                loss, tokens, memory, carried_hidden, train_stats = model.segment_loss(
                    train_ids,
                    batch_segments,
                    memory,
                    args.max_memory,
                    carried_hidden=carried_hidden,
                    carry_hidden=args.carry_hidden,
                    feedback_state=args.feedback_state,
                    token_feedback=args.token_feedback,
                    prior_log_probs=train_prior_log_probs,
                )
                step_profile.forward_sec += time.perf_counter() - t0
                if tokens == 0:
                    continue
                step_profile.updates += 1
                step_profile.tokens += tokens
                train_nll = float(loss.item()) / tokens
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch={epoch} step={step}")
                t0 = time.perf_counter()
                (loss / tokens).backward()
                step_profile.backward_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                step_profile.clip_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                optimizer.step()
                step_profile.optimizer_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                memory = [(state.detach(), position) for state, position in memory[-args.max_memory:]]
                if carried_hidden is not None:
                    carried_hidden = carried_hidden.detach()
                step_profile.detach_sec += time.perf_counter() - t0
                epoch_profile.add(step_profile)
                if args.progress_every_steps > 0 and step % args.progress_every_steps == 0:
                    print(
                        f"progress epoch={epoch} step={step} offset={offset + len(batch_segments)}/{len(train_segments)} "
                        f"train_nll={train_nll:.4f} {profile_summary(epoch_profile)}",
                        flush=True,
                    )
                if args.eval_every > 0 and step % args.eval_every == 0:
                    t0 = time.perf_counter()
                    val_nll, val_ppl, val_bpc, val_stats = evaluate(
                        model,
                        eval_ids,
                        eval_segments,
                        args.max_memory,
                        carry_hidden=args.carry_hidden,
                        feedback_state=args.feedback_state,
                        token_feedback=args.token_feedback,
                        prior_log_probs=eval_prior_log_probs,
                    )
                    epoch_profile.eval_sec += time.perf_counter() - t0
                    mean_gate, mean_context_norm, mean_delta_norm = row_stats(val_stats)
                    (
                        no_memory_ppl,
                        context_only_ppl,
                        mean_attn_entropy,
                        mean_attn_top_weight,
                        mean_attn_char_distance,
                        mean_attn_prev_weight,
                        mean_attn_current_weight,
                        mean_attn_top_is_prev,
                        first_token_ppl,
                        later_token_ppl,
                    ) = diagnostic_row_stats(val_stats)
                    row = SegmentMemoryRow(
                        epoch=epoch,
                        step=step,
                        train_nll=train_nll,
                        val_nll=val_nll,
                        val_ppl=val_ppl,
                        val_bpc=val_bpc,
                        segment_count=len(train_segments),
                        mean_segment_len=mean_segment_len,
                        max_segment_len=max_segment_len,
                        mean_gate=mean_gate,
                        mean_context_norm=mean_context_norm,
                        mean_delta_norm=mean_delta_norm,
                        no_memory_ppl=no_memory_ppl,
                        context_only_ppl=context_only_ppl,
                        mean_attn_entropy=mean_attn_entropy,
                        mean_attn_top_weight=mean_attn_top_weight,
                        mean_attn_char_distance=mean_attn_char_distance,
                        mean_attn_prev_weight=mean_attn_prev_weight,
                        mean_attn_current_weight=mean_attn_current_weight,
                        mean_attn_top_is_prev=mean_attn_top_is_prev,
                        first_token_ppl=first_token_ppl,
                        later_token_ppl=later_token_ppl,
                        runtime_sec=time.perf_counter() - started,
                        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    )
                    writer.writerow(asdict(row))
                    handle.flush()
                    print(
                        f"epoch={epoch} step={step} train_nll={train_nll:.4f} "
                        f"val_ppl={val_ppl:.3f} runtime_sec={row.runtime_sec:.2f} "
                        f"peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                        flush=True,
                    )
            t0 = time.perf_counter()
            val_nll, val_ppl, val_bpc, val_stats = evaluate(
                model,
                eval_ids,
                eval_segments,
                args.max_memory,
                carry_hidden=args.carry_hidden,
                feedback_state=args.feedback_state,
                token_feedback=args.token_feedback,
                prior_log_probs=eval_prior_log_probs,
            )
            epoch_profile.eval_sec += time.perf_counter() - t0
            mean_gate, mean_context_norm, mean_delta_norm = row_stats(val_stats)
            (
                no_memory_ppl,
                context_only_ppl,
                mean_attn_entropy,
                mean_attn_top_weight,
                mean_attn_char_distance,
                mean_attn_prev_weight,
                mean_attn_current_weight,
                mean_attn_top_is_prev,
                first_token_ppl,
                later_token_ppl,
            ) = diagnostic_row_stats(val_stats)
            row = SegmentMemoryRow(
                epoch=epoch,
                step=step,
                train_nll=None,
                val_nll=val_nll,
                val_ppl=val_ppl,
                val_bpc=val_bpc,
                segment_count=len(train_segments),
                mean_segment_len=mean_segment_len,
                max_segment_len=max_segment_len,
                mean_gate=mean_gate,
                mean_context_norm=mean_context_norm,
                mean_delta_norm=mean_delta_norm,
                no_memory_ppl=no_memory_ppl,
                context_only_ppl=context_only_ppl,
                mean_attn_entropy=mean_attn_entropy,
                mean_attn_top_weight=mean_attn_top_weight,
                mean_attn_char_distance=mean_attn_char_distance,
                mean_attn_prev_weight=mean_attn_prev_weight,
                mean_attn_current_weight=mean_attn_current_weight,
                mean_attn_top_is_prev=mean_attn_top_is_prev,
                first_token_ppl=first_token_ppl,
                later_token_ppl=later_token_ppl,
                runtime_sec=time.perf_counter() - started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(row))
            handle.flush()
            print(
                f"epoch={epoch} done step={step} val_ppl={val_ppl:.3f} "
                f"mean_gate={mean_gate if mean_gate is not None else float('nan'):.4f} "
                f"mean_context_norm={mean_context_norm if mean_context_norm is not None else float('nan'):.4f} "
                f"mean_delta_norm={mean_delta_norm if mean_delta_norm is not None else float('nan'):.4f} "
                f"no_memory_ppl={no_memory_ppl if no_memory_ppl is not None else float('nan'):.3f} "
                f"context_only_ppl={context_only_ppl if context_only_ppl is not None else float('nan'):.3f} "
                f"attn_top={mean_attn_top_weight if mean_attn_top_weight is not None else float('nan'):.3f} "
                f"prev_w={mean_attn_prev_weight if mean_attn_prev_weight is not None else float('nan'):.3f} "
                f"top_prev={mean_attn_top_is_prev if mean_attn_top_is_prev is not None else float('nan'):.3f} "
                f"runtime_sec={row.runtime_sec:.2f} peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
            t0 = time.perf_counter()
            improved_best = best_val_ppl is None or val_ppl < best_val_ppl
            if improved_best:
                best_val_ppl = val_ppl
                best_epoch = epoch
                best_step = step
                save_checkpoint(
                    args.best_checkpoint_output,
                    model,
                    optimizer,
                    epoch,
                    step,
                    vocab,
                    args,
                    best_val_ppl=best_val_ppl,
                    best_epoch=best_epoch,
                    best_step=best_step,
                )
            save_checkpoint(
                args.checkpoint_output,
                model,
                optimizer,
                epoch,
                step,
                vocab,
                args,
                best_val_ppl=best_val_ppl,
                best_epoch=best_epoch,
                best_step=best_step,
            )
            epoch_profile.checkpoint_sec += time.perf_counter() - t0
            if args.profile:
                print(f"epoch={epoch} {profile_summary(epoch_profile)}", flush=True)
            print(f"checkpoint={args.checkpoint_output}", flush=True)
            if improved_best:
                print(
                    f"best_checkpoint={args.best_checkpoint_output} "
                    f"best_epoch={best_epoch} best_step={best_step} best_val_ppl={best_val_ppl:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
