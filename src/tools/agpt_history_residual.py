#!/usr/bin/env python3
"""Prototype history-attention residual on top of the learned AGPT count prior."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import random
import struct
import time
from array import array
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_vocab(vocab_text: str) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set(vocab_text))
    return chars, {ch: i for i, ch in enumerate(chars)}


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi.get(ch, 0) for ch in text]


def build_prefix_counts(tokens: list[int], depth: int) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for i in range(len(tokens)):
        prefix = []
        for j in range(i, min(len(tokens), i + depth)):
            prefix.append(tokens[j])
            key = tuple(prefix)
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_mass_segments(
    tokens: list[int], counts: dict[tuple[int, ...], int], depth: int
) -> list[tuple[int, int]]:
    segments = []
    cursor = 0
    while cursor < len(tokens):
        prefix = []
        chosen = 1
        for j in range(cursor, min(len(tokens), cursor + depth)):
            prefix.append(tokens[j])
            chosen = j - cursor + 1
            if counts.get(tuple(prefix), 0) <= 1:
                break
        end = min(len(tokens), cursor + chosen)
        segments.append((cursor, end))
        cursor = end
    return segments


def segment_summary(segments: list[tuple[int, int]]) -> dict[str, float | int]:
    lengths = [end - start for start, end in segments]
    if not lengths:
        return {"segments": 0, "mean_len": 0.0, "max_len": 0, "min_len": 0}
    return {
        "segments": len(lengths),
        "mean_len": sum(lengths) / len(lengths),
        "max_len": max(lengths),
        "min_len": min(lengths),
    }


def read_substring_catalog(path: str) -> dict[bytes, int]:
    data = Path(path).read_bytes()
    if len(data) < 8 or data[:4] != b"ASUB":
        raise ValueError(f"bad substring catalog: {path}")
    count = struct.unpack_from("<I", data, 4)[0]
    offset = 8
    out: dict[bytes, int] = {}
    for sid in range(count):
        if offset >= len(data):
            raise ValueError(f"truncated substring catalog: {path}")
        length = data[offset]
        offset += 1
        if offset + length > len(data):
            raise ValueError(f"truncated substring payload: {path}")
        out[bytes(data[offset : offset + length])] = sid
        offset += length
    return out


class TargetSidecar:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mm[:4] != b"AGTS":
            raise ValueError(f"bad AGTS sidecar: {path}")
        version, self.scale, self.substring_count, self.total_entries = struct.unpack_from(
            "<HIIQ", self.mm, 4
        )
        if version != 1:
            raise ValueError(f"unsupported AGTS version: {version}")
        self.offset_base = 4 + struct.calcsize("<HIIQ")
        offsets_bytes = (self.substring_count + 1) * 4
        self.entry_base = self.offset_base + offsets_bytes
        self.offsets = array("i")
        self.offsets.frombytes(self.mm[self.offset_base : self.entry_base])
        if struct.pack("=i", 1) != struct.pack("<i", 1):
            self.offsets.byteswap()

    def log_probs(
        self,
        contexts: list[bytes],
        catalog: dict[bytes, int],
        vocab_size: int,
        floor: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, int, int]:
        rows = torch.full(
            (len(contexts), vocab_size),
            math.log(floor),
            dtype=torch.float32,
            device=device,
        )
        exact = 0
        backoff = 0
        miss = 0
        inv_scale = 1.0 / float(self.scale)
        for r, ctx in enumerate(contexts):
            sid = None
            matched = 0
            for start in range(len(ctx)):
                probe = ctx[start:]
                sid = catalog.get(probe)
                if sid is not None:
                    matched = len(probe)
                    break
            if sid is None:
                miss += 1
                continue
            if matched == len(ctx):
                exact += 1
            else:
                backoff += 1
            off0 = self.offsets[sid]
            off1 = self.offsets[sid + 1]
            base = self.entry_base + off0 * 6
            for i in range(off1 - off0):
                tok, count = struct.unpack_from("<HI", self.mm, base + i * 6)
                if tok < vocab_size and count > 0:
                    rows[r, tok] = math.log(max(count * inv_scale, floor))
        return rows, exact, backoff, miss

    def close(self) -> None:
        self.mm.close()
        self.file.close()


class LiveCountGatePrior:
    def __init__(self, result_path: str, vocab_chars: list[str]):
        from agpt_count_gate import CountModel, FeatureScaler, expand_extra_features

        self.path = result_path
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        cfg = result["config"]
        depth = int(cfg["depth"])
        valid_ratio = float(cfg.get("valid_ratio", 0.05))
        train_all = encode(read_text(cfg["train"]), {ch: i for i, ch in enumerate(vocab_chars)})
        split = int(len(train_all) * (1.0 - valid_ratio))
        split = max(depth + 1, min(split, len(train_all) - depth - 1))
        count_train = bytes(train_all[:split])
        base_mode = cfg.get("base_features", "core")
        base_features = (
            ["reliability", "kl_gain", "entropy_norm", "depth_norm"]
            if base_mode == "core"
            else []
        )
        self.depth = depth
        self.model = CountModel(
            count_train,
            len(vocab_chars),
            depth,
            expand_extra_features(cfg.get("extra_features", "")),
            base_features=base_features,
        )
        self.model.build()
        self.theta = [float(result["theta"][name]) for name in result["feature_names"]]
        self.scaler = None
        if result.get("feature_standardization", {}).get("enabled"):
            fs = result["feature_standardization"]
            self.scaler = FeatureScaler(
                [float(fs["mean"][name]) for name in result["feature_names"]],
                [float(fs["std"][name]) for name in result["feature_names"]],
            )

    def matched_depth(self, ctx: bytes) -> int:
        max_depth = min(self.depth, len(ctx))
        for d in range(max_depth, 0, -1):
            if self.model.stats(ctx[-d:]) is not None:
                return d
        return 0

    def log_probs(
        self,
        contexts: list[bytes],
        catalog: dict[bytes, int],
        vocab_size: int,
        floor: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, int, int]:
        rows = torch.empty((len(contexts), vocab_size), dtype=torch.float32, device=device)
        exact = 0
        backoff = 0
        miss = 0
        for r, ctx in enumerate(contexts):
            matched = self.matched_depth(ctx)
            if matched == len(ctx):
                exact += 1
            elif matched > 0:
                backoff += 1
            else:
                miss += 1
            probs = self.model.gated_distribution(ctx, self.theta, self.scaler)
            rows[r] = torch.tensor(
                [math.log(max(p, floor)) for p in probs],
                dtype=torch.float32,
                device=device,
            )
        return rows, exact, backoff, miss

    def close(self) -> None:
        pass


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even d_model")
    half = dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, half, dtype=x.dtype, device=x.device) / max(half, 1))
    )
    angles = positions.to(dtype=x.dtype, device=x.device).unsqueeze(-1) * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class HistoryResidual(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        position_encoding: str = "none",
        prior_feature_dim: int = 0,
    ):
        super().__init__()
        if position_encoding == "rope" and d_model % 2 != 0:
            raise ValueError("RoPE requires an even d_model")
        self.position_encoding = position_encoding
        self.prior_feature_dim = prior_feature_dim
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, vocab_size)
        self.alpha = nn.Linear(d_model * 2, 1)
        self.alpha_prior = (
            nn.Linear(prior_feature_dim, 1, bias=False) if prior_feature_dim > 0 else None
        )
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.alpha.weight)
        nn.init.constant_(self.alpha.bias, -2.0)
        if self.alpha_prior is not None:
            nn.init.zeros_(self.alpha_prior.weight)

    def recurrent_states(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens.unsqueeze(0))
        out, _ = self.rnn(emb)
        return out.squeeze(0)

    def recurrent_states_with_hidden(
        self, tokens: torch.Tensor, h0: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(tokens.unsqueeze(0))
        out, h_n = self.rnn(emb, h0)
        return out.squeeze(0), h_n

    def attend(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.w_q(query).unsqueeze(1)
        k = self.w_k(memory)
        v = self.w_v(memory)
        if self.position_encoding == "rope":
            if query_positions is None or memory_positions is None:
                raise ValueError("RoPE attention requires query and memory positions")
            q = apply_rope(q, query_positions.unsqueeze(1))
            k = apply_rope(k, memory_positions)
        scores = (q * k).sum(dim=-1) / math.sqrt(query.shape[-1])
        attn = torch.softmax(scores, dim=-1)
        attended = (attn.unsqueeze(-1) * v).sum(dim=1)
        residual = self.out(attended)
        alpha = torch.sigmoid(self.alpha(torch.cat([query, attended], dim=-1))).squeeze(-1)
        return residual, alpha, attn

    def mix_prior_alpha(self, alpha: torch.Tensor, prior_features: torch.Tensor | None) -> torch.Tensor:
        if self.alpha_prior is None:
            return alpha
        if prior_features is None:
            raise ValueError("prior features are required by this model")
        base = torch.logit(alpha.clamp(1.0e-6, 1.0 - 1.0e-6))
        correction = self.alpha_prior(prior_features).squeeze(-1)
        return torch.sigmoid(base + correction)

    def forward(self, seqs: torch.Tensor, k_history: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Legacy independent-segment path kept for comparison/smoke runs.
        batch = seqs.shape[0]
        flat = seqs.reshape(batch * (k_history + 1), seqs.shape[-1])
        emb = self.embedding(flat)
        _, h_n = self.rnn(emb)
        states = h_n[-1].reshape(batch, k_history + 1, -1)
        current = states[:, 0, :]
        history = states[:, 1:, :]
        return self.attend(current, history)


def make_context_batch(tokens: list[int], positions: list[int], depth: int, k_history: int) -> torch.Tensor:
    rows = []
    for pos in positions:
        seqs = []
        for lag in range(k_history + 1):
            end = pos - lag * depth
            start = end - depth
            seqs.append(tokens[start:end])
        rows.append(seqs)
    return torch.tensor(rows, dtype=torch.long)


def prior_contexts(tokens: list[int], positions: list[int], depth: int) -> list[bytes]:
    return [bytes(tokens[pos - depth : pos]) for pos in positions]


def matched_depths(
    contexts: list[bytes],
    catalog: dict[bytes, int],
    depth_lookup=None,
) -> list[int]:
    if depth_lookup is not None:
        return [int(depth_lookup(ctx)) for ctx in contexts]
    depths = []
    for ctx in contexts:
        matched = 0
        for start in range(len(ctx)):
            if ctx[start:] in catalog:
                matched = len(ctx) - start
                break
        depths.append(matched)
    return depths


def prior_summary_features(
    log_prior: torch.Tensor,
    contexts: list[bytes],
    catalog: dict[bytes, int],
    depth: int,
    depth_lookup=None,
) -> torch.Tensor:
    probs = log_prior.exp()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
    entropy = -(probs * probs.clamp_min(1.0e-30).log()).sum(dim=-1)
    entropy_norm = entropy / math.log(log_prior.shape[-1])
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    top_mass = top2[:, 0]
    if top2.shape[-1] > 1:
        top_margin = top2[:, 0] - top2[:, 1]
    else:
        top_margin = top2[:, 0]
    hit_depth = torch.tensor(
        [d / max(depth, 1) for d in matched_depths(contexts, catalog, depth_lookup)],
        dtype=log_prior.dtype,
        device=log_prior.device,
    )
    return torch.stack([entropy_norm, top_mass, top_margin, hit_depth], dim=-1)


def prior_dropout_baseline(
    mode: str,
    sidecar: TargetSidecar,
    catalog: dict[bytes, int],
    vocab_size: int,
    floor: float,
    device: torch.device,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.full((vocab_size,), -math.log(vocab_size), dtype=torch.float32, device=device)
    root, _ex, _bo, miss = sidecar.log_probs([b""], catalog, vocab_size, floor, device)
    if miss:
        return torch.full((vocab_size,), -math.log(vocab_size), dtype=torch.float32, device=device)
    return root.squeeze(0)


def select_log_prior(
    log_prior: torch.Tensor,
    mode: str,
    root_prior: torch.Tensor,
    uniform_prior: torch.Tensor,
) -> torch.Tensor:
    if mode == "full":
        return log_prior
    if mode == "root":
        return root_prior.unsqueeze(0).expand_as(log_prior)
    if mode == "uniform":
        return uniform_prior.unsqueeze(0).expand_as(log_prior)
    raise ValueError(f"unknown prior mode: {mode}")


def apply_residual(
    residual: torch.Tensor,
    alpha: torch.Tensor,
    residual_scale: float,
    learn_alpha: bool,
    max_delta: float,
) -> torch.Tensor:
    applied = (alpha.unsqueeze(-1) if learn_alpha else residual_scale) * residual
    if max_delta > 0.0:
        applied = max_delta * torch.tanh(applied / max_delta)
    return applied


def kl_prior_to_model(prior_logits: torch.Tensor, model_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(model_logits, dim=-1),
        F.softmax(prior_logits, dim=-1),
        reduction="batchmean",
    )


def select_eval_positions(length: int, depth: int, k_history: int, n: int, seed: int) -> list[int]:
    start = depth * (k_history + 1)
    positions = list(range(start, length))
    if n > 0 and n < len(positions):
        rng = random.Random(seed)
        positions = sorted(rng.sample(positions, n))
    return positions


def stream_residuals(
    model: HistoryResidual,
    tokens: list[int],
    start: int,
    n_targets: int,
    depth: int,
    k_history: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    warmup = depth * k_history
    n_available = min(n_targets, len(tokens) - start - warmup)
    if n_available <= 0:
        raise ValueError("stream chunk has no available target positions")
    end = start + warmup + n_available
    chunk = torch.tensor(tokens[start:end], dtype=torch.long, device=device)
    states = model.recurrent_states(chunk)

    queries = []
    memories = []
    query_positions = []
    memory_positions = []
    positions = []
    for i in range(n_available):
        local_pos = warmup + i
        global_pos = start + local_pos
        queries.append(states[local_pos - 1])
        rows = [states[local_pos - 1]]
        row_positions = [global_pos - 1]
        for lag_i in range(1, k_history + 1):
            rows.append(states[local_pos - lag_i * depth])
            row_positions.append(global_pos - lag_i * depth)
        memories.append(torch.stack(rows, dim=0))
        query_positions.append(global_pos - 1)
        memory_positions.append(row_positions)
        positions.append(global_pos)
    query = torch.stack(queries, dim=0)
    memory = torch.stack(memories, dim=0)
    query_pos = torch.tensor(query_positions, dtype=torch.long, device=device)
    memory_pos = torch.tensor(memory_positions, dtype=torch.long, device=device)
    residual, alpha, attn = model.attend(query, memory, query_pos, memory_pos)
    return residual, alpha, attn, positions


def ordered_stream_residuals(
    model: HistoryResidual,
    tokens: list[int],
    cursor: int,
    n_targets: int,
    depth: int,
    k_history: int,
    previous_hidden: torch.Tensor | None,
    previous_states: list[tuple[int, torch.Tensor]],
    device: torch.device,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    list[int],
    int,
    torch.Tensor,
    list[tuple[int, torch.Tensor]],
    int,
]:
    n_available = min(n_targets, len(tokens) - cursor)
    if n_available <= 0:
        raise ValueError("ordered cursor has no available target positions")

    input_start = cursor - 1
    input_end = input_start + n_available
    chunk = torch.tensor(tokens[input_start:input_end], dtype=torch.long, device=device)
    states, hidden = model.recurrent_states_with_hidden(chunk, previous_hidden)
    state_positions = list(range(input_start, input_end))
    current_states = {pos: states[i] for i, pos in enumerate(state_positions)}
    old_states = {pos: state for pos, state in previous_states}

    def state_at(pos: int) -> torch.Tensor:
        state = current_states.get(pos)
        if state is not None:
            return state
        return old_states[pos]

    queries = []
    memories = []
    query_positions = []
    memory_positions = []
    positions = []
    warmup = depth * k_history
    for target_pos in range(cursor, cursor + n_available):
        if target_pos - warmup < 0:
            continue
        try:
            query = state_at(target_pos - 1)
            rows = [query]
            row_positions = [target_pos - 1]
            for lag_i in range(1, k_history + 1):
                rows.append(state_at(target_pos - lag_i * depth))
                row_positions.append(target_pos - lag_i * depth)
        except KeyError:
            continue
        queries.append(query)
        memories.append(torch.stack(rows, dim=0))
        query_positions.append(target_pos - 1)
        memory_positions.append(row_positions)
        positions.append(target_pos)

    next_cursor = cursor + n_available
    min_keep = max(0, next_cursor - warmup)
    detached_states = [
        (pos, state.detach()) for pos, state in previous_states if pos >= min_keep
    ]
    detached_states.extend(
        (pos, states[i].detach()) for i, pos in enumerate(state_positions) if pos >= min_keep
    )

    if not positions:
        return None, None, None, [], next_cursor, hidden.detach(), detached_states, n_available

    query_batch = torch.stack(queries, dim=0)
    memory_batch = torch.stack(memories, dim=0)
    query_pos = torch.tensor(query_positions, dtype=torch.long, device=device)
    memory_pos = torch.tensor(memory_positions, dtype=torch.long, device=device)
    residual, alpha, attn = model.attend(query_batch, memory_batch, query_pos, memory_pos)
    return residual, alpha, attn, positions, next_cursor, hidden.detach(), detached_states, n_available


def ordered_segment_residuals(
    model: HistoryResidual,
    tokens: list[int],
    cursor: int,
    n_targets: int,
    k_memory: int,
    previous_hidden: torch.Tensor | None,
    previous_states: list[tuple[int, torch.Tensor]],
    previous_segment_states: list[tuple[int, torch.Tensor]],
    segment_ends: list[int],
    segment_index: int,
    device: torch.device,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    list[int],
    int,
    torch.Tensor,
    list[tuple[int, torch.Tensor]],
    list[tuple[int, torch.Tensor]],
    int,
    int,
]:
    n_available = min(n_targets, len(tokens) - cursor)
    if n_available <= 0:
        raise ValueError("ordered cursor has no available target positions")

    input_start = cursor - 1
    input_end = input_start + n_available
    chunk = torch.tensor(tokens[input_start:input_end], dtype=torch.long, device=device)
    states, hidden = model.recurrent_states_with_hidden(chunk, previous_hidden)
    state_positions = list(range(input_start, input_end))
    current_states = {pos: states[i] for i, pos in enumerate(state_positions)}
    old_states = {pos: state for pos, state in previous_states}

    def state_at(pos: int) -> torch.Tensor:
        state = current_states.get(pos)
        if state is not None:
            return state
        return old_states[pos]

    segment_states = list(previous_segment_states)
    queries = []
    memories = []
    query_positions = []
    memory_positions = []
    positions = []

    for target_pos in range(cursor, cursor + n_available):
        while segment_index < len(segment_ends) and segment_ends[segment_index] <= target_pos:
            terminal_pos = segment_ends[segment_index] - 1
            try:
                terminal_state = state_at(terminal_pos)
            except KeyError:
                terminal_state = None
            if terminal_state is not None:
                segment_states.append((terminal_pos, terminal_state))
                if len(segment_states) > k_memory:
                    segment_states = segment_states[-k_memory:]
            segment_index += 1

        if len(segment_states) < k_memory:
            continue
        query = state_at(target_pos - 1)
        rows = [state for _pos, state in segment_states[-k_memory:]]
        row_positions = [pos for pos, _state in segment_states[-k_memory:]]
        queries.append(query)
        memories.append(torch.stack(rows, dim=0))
        query_positions.append(target_pos - 1)
        memory_positions.append(row_positions)
        positions.append(target_pos)

    next_cursor = cursor + n_available
    min_keep = max(0, next_cursor - max(k_memory * 16, 1024))
    detached_states = [
        (pos, state.detach()) for pos, state in previous_states if pos >= min_keep
    ]
    detached_states.extend(
        (pos, states[i].detach()) for i, pos in enumerate(state_positions) if pos >= min_keep
    )
    detached_segment_states = [
        (pos, state.detach()) for pos, state in segment_states[-k_memory:]
    ]

    if not positions:
        return (
            None,
            None,
            None,
            [],
            next_cursor,
            hidden.detach(),
            detached_states,
            detached_segment_states,
            segment_index,
            n_available,
        )

    query_batch = torch.stack(queries, dim=0)
    memory_batch = torch.stack(memories, dim=0)
    query_pos = torch.tensor(query_positions, dtype=torch.long, device=device)
    memory_pos = torch.tensor(memory_positions, dtype=torch.long, device=device)
    residual, alpha, attn = model.attend(query_batch, memory_batch, query_pos, memory_pos)
    return (
        residual,
        alpha,
        attn,
        positions,
        next_cursor,
        hidden.detach(),
        detached_states,
        detached_segment_states,
        segment_index,
        n_available,
    )


@torch.no_grad()
def evaluate_stream(
    model: HistoryResidual,
    tokens: list[int],
    starts: list[int],
    sidecar: TargetSidecar,
    catalog: dict[bytes, int],
    vocab_size: int,
    depth: int,
    k_history: int,
    chunk_targets: int,
    floor: float,
    residual_scale: float,
    learn_alpha: bool,
    residual_max_delta: float,
    prior_mode: str,
    root_prior: torch.Tensor,
    uniform_prior: torch.Tensor,
    device: torch.device,
    depth_lookup=None,
) -> dict[str, float]:
    model.eval()
    total_prior = 0.0
    total_full = 0.0
    total = 0
    exact = backoff = miss = 0
    residual_norm_sum = 0.0
    applied_residual_norm_sum = 0.0
    alpha_sum = 0.0
    attn_entropy_sum = 0.0
    lag_mass = torch.zeros(k_history + 1, dtype=torch.float64)

    for start in starts:
        residual, alpha, attn, positions = stream_residuals(
            model, tokens, start, chunk_targets, depth, k_history, device
        )
        target = torch.tensor([tokens[pos] for pos in positions], dtype=torch.long, device=device)
        contexts = prior_contexts(tokens, positions, depth)
        log_prior, ex, bo, mi = sidecar.log_probs(contexts, catalog, vocab_size, floor, device)
        prediction_prior = select_log_prior(log_prior, prior_mode, root_prior, uniform_prior)
        if learn_alpha and model.alpha_prior is not None:
            alpha = model.mix_prior_alpha(
                alpha,
                prior_summary_features(prediction_prior, contexts, catalog, depth, depth_lookup),
            )
        applied = apply_residual(residual, alpha, residual_scale, learn_alpha, residual_max_delta)
        logits = prediction_prior + applied
        total_prior += F.cross_entropy(prediction_prior, target, reduction="sum").item()
        total_full += F.cross_entropy(logits, target, reduction="sum").item()
        total += len(positions)
        exact += ex
        backoff += bo
        miss += mi
        residual_norm_sum += residual.norm(dim=-1).sum().item()
        applied_residual_norm_sum += applied.norm(dim=-1).sum().item()
        alpha_sum += alpha.sum().item() if learn_alpha else residual_scale * len(positions)
        attn_entropy_sum += (-(attn * torch.log(attn.clamp_min(1e-30))).sum(dim=-1)).sum().item()
        lag_mass += attn.detach().cpu().double().sum(dim=0)

    mean_prior = total_prior / max(total, 1)
    mean_full = total_full / max(total, 1)
    lag_mass = lag_mass / max(total, 1)
    labels = [1] + [depth * i for i in range(1, k_history + 1)]
    top_lags = sorted(
        [(labels[i], float(mass)) for i, mass in enumerate(lag_mass.tolist())],
        key=lambda row: row[1],
        reverse=True,
    )[:8]
    return {
        "positions": total,
        "prior_loss_nats": mean_prior,
        "prior_ppl": math.exp(mean_prior),
        "loss_nats": mean_full,
        "ppl": math.exp(mean_full),
        "prior_exact_hits": exact,
        "prior_backoff_hits": backoff,
        "prior_misses": miss,
        "avg_residual_norm": residual_norm_sum / max(total, 1),
        "avg_applied_residual_norm": applied_residual_norm_sum / max(total, 1),
        "avg_alpha": alpha_sum / max(total, 1),
        "avg_attention_entropy": attn_entropy_sum / max(total, 1),
        "top_lags": top_lags,
    }


@torch.no_grad()
def evaluate_ordered_segments(
    model: HistoryResidual,
    tokens: list[int],
    sidecar: TargetSidecar,
    catalog: dict[bytes, int],
    vocab_size: int,
    depth: int,
    k_memory: int,
    chunk_targets: int,
    floor: float,
    residual_scale: float,
    learn_alpha: bool,
    residual_max_delta: float,
    prior_mode: str,
    root_prior: torch.Tensor,
    uniform_prior: torch.Tensor,
    segment_ends: list[int],
    device: torch.device,
    depth_lookup=None,
) -> dict[str, float]:
    model.eval()
    total_prior = 0.0
    total_full = 0.0
    total = 0
    exact = backoff = miss = 0
    residual_norm_sum = 0.0
    applied_residual_norm_sum = 0.0
    alpha_sum = 0.0
    attn_entropy_sum = 0.0
    lag_mass = torch.zeros(k_memory, dtype=torch.float64)

    cursor = 1
    hidden: torch.Tensor | None = None
    states: list[tuple[int, torch.Tensor]] = []
    segment_states: list[tuple[int, torch.Tensor]] = []
    segment_index = 0

    while cursor < len(tokens):
        (
            residual,
            alpha,
            attn,
            positions,
            cursor,
            hidden,
            states,
            segment_states,
            segment_index,
            _processed,
        ) = ordered_segment_residuals(
            model,
            tokens,
            cursor,
            chunk_targets,
            k_memory,
            hidden,
            states,
            segment_states,
            segment_ends,
            segment_index,
            device,
        )
        if residual is None or alpha is None or attn is None or not positions:
            continue
        target = torch.tensor([tokens[pos] for pos in positions], dtype=torch.long, device=device)
        contexts = prior_contexts(tokens, positions, depth)
        log_prior, ex, bo, mi = sidecar.log_probs(contexts, catalog, vocab_size, floor, device)
        prediction_prior = select_log_prior(log_prior, prior_mode, root_prior, uniform_prior)
        if learn_alpha and model.alpha_prior is not None:
            alpha = model.mix_prior_alpha(
                alpha,
                prior_summary_features(prediction_prior, contexts, catalog, depth, depth_lookup),
            )
        applied = apply_residual(residual, alpha, residual_scale, learn_alpha, residual_max_delta)
        logits = prediction_prior + applied
        total_prior += F.cross_entropy(prediction_prior, target, reduction="sum").item()
        total_full += F.cross_entropy(logits, target, reduction="sum").item()
        total += len(positions)
        exact += ex
        backoff += bo
        miss += mi
        residual_norm_sum += residual.norm(dim=-1).sum().item()
        applied_residual_norm_sum += applied.norm(dim=-1).sum().item()
        alpha_sum += alpha.sum().item() if learn_alpha else residual_scale * len(positions)
        attn_entropy_sum += (-(attn * torch.log(attn.clamp_min(1e-30))).sum(dim=-1)).sum().item()
        lag_mass += attn.detach().cpu().double().sum(dim=0)

    mean_prior = total_prior / max(total, 1)
    mean_full = total_full / max(total, 1)
    lag_mass = lag_mass / max(total, 1)
    top_lags = sorted(
        [(i + 1, float(mass)) for i, mass in enumerate(lag_mass.tolist())],
        key=lambda row: row[1],
        reverse=True,
    )[:8]
    return {
        "positions": total,
        "prior_loss_nats": mean_prior,
        "prior_ppl": math.exp(mean_prior),
        "loss_nats": mean_full,
        "ppl": math.exp(mean_full),
        "prior_exact_hits": exact,
        "prior_backoff_hits": backoff,
        "prior_misses": miss,
        "avg_residual_norm": residual_norm_sum / max(total, 1),
        "avg_applied_residual_norm": applied_residual_norm_sum / max(total, 1),
        "avg_alpha": alpha_sum / max(total, 1),
        "avg_attention_entropy": attn_entropy_sum / max(total, 1),
        "top_lags": top_lags,
    }


@torch.no_grad()
def evaluate(
    model: HistoryResidual,
    tokens: list[int],
    positions: list[int],
    sidecar: TargetSidecar,
    catalog: dict[bytes, int],
    vocab_size: int,
    depth: int,
    k_history: int,
    batch_size: int,
    floor: float,
    residual_scale: float,
    learn_alpha: bool,
    residual_max_delta: float,
    prior_mode: str,
    root_prior: torch.Tensor,
    uniform_prior: torch.Tensor,
    device: torch.device,
    depth_lookup=None,
) -> dict[str, float]:
    model.eval()
    total_prior = 0.0
    total_full = 0.0
    total = 0
    exact = backoff = miss = 0
    residual_norm_sum = 0.0
    applied_residual_norm_sum = 0.0
    alpha_sum = 0.0
    attn_entropy_sum = 0.0
    lag_mass = torch.zeros(k_history, dtype=torch.float64)

    for offset in range(0, len(positions), batch_size):
        batch_pos = positions[offset : offset + batch_size]
        seqs = make_context_batch(tokens, batch_pos, depth, k_history).to(device)
        target = torch.tensor([tokens[pos] for pos in batch_pos], dtype=torch.long, device=device)
        contexts = prior_contexts(tokens, batch_pos, depth)
        log_prior, ex, bo, mi = sidecar.log_probs(contexts, catalog, vocab_size, floor, device)
        residual, alpha, attn = model(seqs, k_history)
        prediction_prior = select_log_prior(log_prior, prior_mode, root_prior, uniform_prior)
        if learn_alpha and model.alpha_prior is not None:
            alpha = model.mix_prior_alpha(
                alpha,
                prior_summary_features(prediction_prior, contexts, catalog, depth, depth_lookup),
            )
        applied = apply_residual(residual, alpha, residual_scale, learn_alpha, residual_max_delta)
        logits = prediction_prior + applied
        total_prior += F.cross_entropy(prediction_prior, target, reduction="sum").item()
        total_full += F.cross_entropy(logits, target, reduction="sum").item()
        total += len(batch_pos)
        exact += ex
        backoff += bo
        miss += mi
        residual_norm_sum += residual.norm(dim=-1).sum().item()
        applied_residual_norm_sum += applied.norm(dim=-1).sum().item()
        alpha_sum += alpha.sum().item() if learn_alpha else residual_scale * len(batch_pos)
        attn_entropy_sum += (-(attn * torch.log(attn.clamp_min(1e-30))).sum(dim=-1)).sum().item()
        lag_mass += attn.detach().cpu().double().sum(dim=0)

    mean_prior = total_prior / max(total, 1)
    mean_full = total_full / max(total, 1)
    lag_mass = lag_mass / max(total, 1)
    top_lags = sorted(
        [(i + 1, float(mass)) for i, mass in enumerate(lag_mass.tolist())],
        key=lambda row: row[1],
        reverse=True,
    )[:8]
    return {
        "positions": total,
        "prior_loss_nats": mean_prior,
        "prior_ppl": math.exp(mean_prior),
        "loss_nats": mean_full,
        "ppl": math.exp(mean_full),
        "prior_exact_hits": exact,
        "prior_backoff_hits": backoff,
        "prior_misses": miss,
        "avg_residual_norm": residual_norm_sum / max(total, 1),
        "avg_applied_residual_norm": applied_residual_norm_sum / max(total, 1),
        "avg_alpha": alpha_sum / max(total, 1),
        "avg_attention_entropy": attn_entropy_sum / max(total, 1),
        "top_lags": top_lags,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--vocab-file", required=True)
    ap.add_argument("--position-data", required=True)
    ap.add_argument("--prior-sidecar", default="")
    ap.add_argument(
        "--count-prior-result",
        default="",
        help=(
            "Use a live learned count-gate prior from an agpt_count_gate.py JSON "
            "result instead of an AGTS sidecar. This leaves --prior-sidecar "
            "unchanged for existing sidecar-based runs."
        ),
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--k-history", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--position-encoding", choices=["none", "rope"], default="none")
    ap.add_argument("--memory-mode", choices=["fixed", "segment"], default="fixed")
    ap.add_argument("--zero-output-head", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-positions", type=int, default=8192)
    ap.add_argument("--eval-all", action="store_true")
    ap.add_argument("--mode", choices=["stream", "ordered", "independent"], default="stream")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--residual-l2", type=float, default=5.0)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--residual-max-delta", type=float, default=0.0)
    ap.add_argument("--trust-kl", type=float, default=0.0)
    ap.add_argument("--learn-alpha", action="store_true")
    ap.add_argument(
        "--alpha-prior-features",
        action="store_true",
        help=(
            "Add prior entropy/top-mass/top-margin/hit-depth features to the "
            "learned alpha gate. Requires --learn-alpha to affect logits."
        ),
    )
    ap.add_argument("--prior-mode", choices=["full", "root", "uniform"], default="full")
    ap.add_argument("--prior-dropout", type=float, default=0.0)
    ap.add_argument("--prior-dropout-mode", choices=["root", "uniform"], default="root")
    ap.add_argument(
        "--surprise-top-frac",
        type=float,
        default=0.0,
        help=(
            "Train residual only on the top fraction of batch rows by prior "
            "surprise -log p_prior(target). 0 disables; eval remains all rows."
        ),
    )
    ap.add_argument("--prior-floor", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.prior_dropout < 0.0 or args.prior_dropout >= 1.0:
        raise ValueError("--prior-dropout must be in [0, 1)")
    if args.surprise_top_frac < 0.0 or args.surprise_top_frac > 1.0:
        raise ValueError("--surprise-top-frac must be in [0, 1]")
    if args.residual_max_delta < 0.0:
        raise ValueError("--residual-max-delta must be non-negative")
    if args.trust_kl < 0.0:
        raise ValueError("--trust-kl must be non-negative")
    if args.memory_mode == "segment" and args.mode != "ordered":
        raise ValueError("--memory-mode segment currently requires --mode ordered")
    if bool(args.prior_sidecar) == bool(args.count_prior_result):
        raise ValueError("provide exactly one of --prior-sidecar or --count-prior-result")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    chars, stoi = build_vocab(read_text(args.vocab_file))
    train_tokens = encode(read_text(args.train), stoi)
    heldout_tokens = encode(read_text(args.heldout), stoi)
    vocab_size = len(chars)

    t0 = time.time()
    train_segments = None
    heldout_segments = None
    if args.memory_mode == "segment":
        prefix_counts = build_prefix_counts(train_tokens, args.depth)
        train_segments = build_mass_segments(train_tokens, prefix_counts, args.depth)
        heldout_segments = build_mass_segments(heldout_tokens, prefix_counts, args.depth)
    catalog = read_substring_catalog(str(Path(args.position_data) / "substrings.bin"))
    prior = (
        LiveCountGatePrior(args.count_prior_result, chars)
        if args.count_prior_result
        else TargetSidecar(args.prior_sidecar)
    )
    model = HistoryResidual(
        vocab_size,
        args.d_model,
        args.position_encoding,
        prior_feature_dim=4 if args.alpha_prior_features else 0,
    ).to(device)
    if args.zero_output_head:
        nn.init.zeros_(model.out.weight)
        nn.init.zeros_(model.out.bias)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    root_prior = prior_dropout_baseline("root", prior, catalog, vocab_size, args.prior_floor, device)
    uniform_prior = prior_dropout_baseline(
        "uniform", prior, catalog, vocab_size, args.prior_floor, device
    )
    prior_depth_lookup = getattr(prior, "matched_depth", None)
    dropout_prior = root_prior if args.prior_dropout_mode == "root" else uniform_prior

    min_pos = args.depth * (args.k_history + 1)
    train_positions = list(range(min_pos, len(train_tokens)))
    eval_positions = select_eval_positions(
        len(heldout_tokens),
        args.depth,
        args.k_history,
        0 if args.eval_all else args.eval_positions,
        args.seed + 100,
    )
    max_train_start = len(train_tokens) - args.depth * args.k_history - args.batch_size
    if max_train_start <= 0:
        raise ValueError("train corpus too short for stream warmup + batch")
    eval_target_count = max(0, len(heldout_tokens) - args.depth * args.k_history)
    if args.eval_all:
        eval_starts = list(range(0, eval_target_count, max(args.batch_size, 1)))
    else:
        n_eval_chunks = max(1, math.ceil(args.eval_positions / max(args.batch_size, 1)))
        max_eval_start = len(heldout_tokens) - args.depth * args.k_history - args.batch_size
        eval_starts = list(range(0, max_eval_start + 1, max(1, (max_eval_start + 1) // n_eval_chunks)))[:n_eval_chunks]
    if not train_positions:
        raise ValueError("no train positions after depth/history warmup")
    if args.mode == "independent" and not eval_positions:
        raise ValueError("no eval positions after depth/history warmup")
    if args.mode in {"stream", "ordered"} and not eval_starts:
        raise ValueError("no stream eval chunks after depth/history warmup")

    history = []
    ordered_cursor = 1
    ordered_epoch = 0
    ordered_hidden: torch.Tensor | None = None
    ordered_states: list[tuple[int, torch.Tensor]] = []
    ordered_segment_states: list[tuple[int, torch.Tensor]] = []
    ordered_segment_index = 0
    train_segment_ends = [end for _start, end in train_segments] if train_segments is not None else []
    for step in range(1, args.steps + 1):
        model.train()
        if args.mode == "stream":
            start = random.randint(0, max_train_start)
            residual, alpha, attn, batch_pos = stream_residuals(
                model, train_tokens, start, args.batch_size, args.depth, args.k_history, device
            )
            ordered_processed_inputs = args.depth * args.k_history + len(batch_pos)
        elif args.mode == "ordered":
            ordered_processed_inputs = 0
            while True:
                if ordered_cursor >= len(train_tokens):
                    ordered_cursor = 1
                    ordered_epoch += 1
                    ordered_hidden = None
                    ordered_states = []
                    ordered_segment_states = []
                    ordered_segment_index = 0
                if args.memory_mode == "segment":
                    (
                        residual,
                        alpha,
                        attn,
                        batch_pos,
                        ordered_cursor,
                        ordered_hidden,
                        ordered_states,
                        ordered_segment_states,
                        ordered_segment_index,
                        processed,
                    ) = ordered_segment_residuals(
                        model,
                        train_tokens,
                        ordered_cursor,
                        args.batch_size,
                        args.k_history,
                        ordered_hidden,
                        ordered_states,
                        ordered_segment_states,
                        train_segment_ends,
                        ordered_segment_index,
                        device,
                    )
                else:
                    (
                        residual,
                        alpha,
                        attn,
                        batch_pos,
                        ordered_cursor,
                        ordered_hidden,
                        ordered_states,
                        processed,
                    ) = ordered_stream_residuals(
                        model,
                        train_tokens,
                        ordered_cursor,
                        args.batch_size,
                        args.depth,
                        args.k_history,
                        ordered_hidden,
                        ordered_states,
                        device,
                    )
                ordered_processed_inputs += processed
                if residual is not None and alpha is not None and attn is not None and batch_pos:
                    break
        else:
            batch_pos = random.sample(train_positions, min(args.batch_size, len(train_positions)))
            seqs = make_context_batch(train_tokens, batch_pos, args.depth, args.k_history).to(device)
            residual, alpha, attn = model(seqs, args.k_history)
            ordered_processed_inputs = len(batch_pos) * (args.k_history + 1) * args.depth
        target = torch.tensor([train_tokens[pos] for pos in batch_pos], dtype=torch.long, device=device)
        contexts = prior_contexts(train_tokens, batch_pos, args.depth)
        log_prior, _ex, _bo, _mi = prior.log_probs(
            contexts,
            catalog,
            vocab_size,
            args.prior_floor,
            device,
        )
        dropout_fraction = 0.0
        train_log_prior = select_log_prior(log_prior, args.prior_mode, root_prior, uniform_prior)
        if args.prior_dropout > 0.0:
            drop_mask = torch.rand((log_prior.shape[0], 1), device=device) < args.prior_dropout
            train_log_prior = torch.where(drop_mask, dropout_prior.unsqueeze(0), train_log_prior)
            dropout_fraction = float(drop_mask.float().mean().item())
        if args.learn_alpha and model.alpha_prior is not None:
            alpha = model.mix_prior_alpha(
                alpha,
                prior_summary_features(
                    train_log_prior,
                    contexts,
                    catalog,
                    args.depth,
                    prior_depth_lookup,
                ),
            )
        applied = apply_residual(
            residual, alpha, args.residual_scale, args.learn_alpha, args.residual_max_delta
        )
        logits = train_log_prior + applied
        prior_surprise = -train_log_prior.gather(1, target.unsqueeze(1)).squeeze(1)
        if args.surprise_top_frac > 0.0 and args.surprise_top_frac < 1.0:
            keep_n = max(1, math.ceil(float(len(target)) * args.surprise_top_frac))
            keep_idx = torch.topk(prior_surprise, k=keep_n, largest=True).indices
            keep_mask = torch.zeros_like(prior_surprise, dtype=torch.bool)
            keep_mask[keep_idx] = True
        else:
            keep_mask = torch.ones_like(prior_surprise, dtype=torch.bool)
        kept_logits = logits[keep_mask]
        kept_target = target[keep_mask]
        kept_prior = train_log_prior[keep_mask]
        kept_applied = applied[keep_mask]
        ce = F.cross_entropy(kept_logits, kept_target)
        l2_reg = args.residual_l2 * kept_applied.pow(2).mean()
        kl_reg = args.trust_kl * kl_prior_to_model(kept_prior, kept_logits)
        loss = ce + l2_reg + kl_reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            row = {
                "step": step,
                "train_ce": float(ce.item()),
                "train_ppl": float(math.exp(ce.item())),
                "reg": float((l2_reg + kl_reg).item()),
                "l2_reg": float(l2_reg.item()),
                "kl_reg": float(kl_reg.item()),
                "avg_residual_norm": float(residual.detach().norm(dim=-1).mean().item()),
                "avg_applied_residual_norm": float(applied.detach().norm(dim=-1).mean().item()),
                "avg_alpha": float(alpha.detach().mean().item() if args.learn_alpha else args.residual_scale),
                "prior_dropout_fraction": dropout_fraction,
                "surprise_top_frac": args.surprise_top_frac,
                "surprise_kept_fraction": float(keep_mask.float().mean().item()),
                "avg_prior_surprise": float(prior_surprise.detach().mean().item()),
                "avg_kept_prior_surprise": float(prior_surprise.detach()[keep_mask].mean().item()),
                "processed_inputs": ordered_processed_inputs,
                "ordered_cursor": ordered_cursor if args.mode == "ordered" else None,
                "ordered_epoch": ordered_epoch if args.mode == "ordered" else None,
                "ordered_segment_index": ordered_segment_index if args.memory_mode == "segment" else None,
                "avg_attention_entropy": float(
                    (-(attn.detach() * torch.log(attn.detach().clamp_min(1e-30))).sum(dim=-1))
                    .mean()
                    .item()
                ),
            }
            print(json.dumps(row), flush=True)
            history.append(row)

    if args.memory_mode == "segment":
        heldout_segment_ends = [end for _start, end in heldout_segments] if heldout_segments is not None else []
        eval_full = evaluate_ordered_segments(
            model,
            heldout_tokens,
            prior,
            catalog,
            vocab_size,
            args.depth,
            args.k_history,
            args.batch_size,
            args.prior_floor,
            args.residual_scale,
            args.learn_alpha,
            args.residual_max_delta,
            args.prior_mode,
            root_prior,
            uniform_prior,
            heldout_segment_ends,
            device,
            prior_depth_lookup,
        )
        eval_half = evaluate_ordered_segments(
            model,
            heldout_tokens,
            prior,
            catalog,
            vocab_size,
            args.depth,
            args.k_history,
            args.batch_size,
            args.prior_floor,
            0.5,
            False,
            args.residual_max_delta,
            args.prior_mode,
            root_prior,
            uniform_prior,
            heldout_segment_ends,
            device,
            prior_depth_lookup,
        )
    elif args.mode in {"stream", "ordered"}:
        eval_full = evaluate_stream(
            model,
            heldout_tokens,
            eval_starts,
            prior,
            catalog,
            vocab_size,
            args.depth,
            args.k_history,
            args.batch_size,
            args.prior_floor,
            args.residual_scale,
            args.learn_alpha,
            args.residual_max_delta,
            args.prior_mode,
            root_prior,
            uniform_prior,
            device,
            prior_depth_lookup,
        )
        eval_half = evaluate_stream(
            model,
            heldout_tokens,
            eval_starts,
            prior,
            catalog,
            vocab_size,
            args.depth,
            args.k_history,
            args.batch_size,
            args.prior_floor,
            0.5,
            False,
            args.residual_max_delta,
            args.prior_mode,
            root_prior,
            uniform_prior,
            device,
            prior_depth_lookup,
        )
    else:
        eval_full = evaluate(
            model,
            heldout_tokens,
            eval_positions,
            prior,
            catalog,
            vocab_size,
            args.depth,
            args.k_history,
            args.batch_size,
            args.prior_floor,
            args.residual_scale,
            args.learn_alpha,
            args.residual_max_delta,
            args.prior_mode,
            root_prior,
            uniform_prior,
            device,
            prior_depth_lookup,
        )
        eval_half = evaluate(
            model,
            heldout_tokens,
            eval_positions,
            prior,
            catalog,
            vocab_size,
            args.depth,
            args.k_history,
            args.batch_size,
            args.prior_floor,
            0.5,
            False,
            args.residual_max_delta,
            args.prior_mode,
            root_prior,
            uniform_prior,
            device,
            prior_depth_lookup,
        )
    result = {
        "config": vars(args),
        "vocab_size": vocab_size,
        "train_positions": len(train_positions),
        "eval_positions": eval_full["positions"],
        "eval_chunks": len(eval_starts) if args.mode in {"stream", "ordered"} else None,
        "train_segments": segment_summary(train_segments) if train_segments is not None else None,
        "heldout_segments": segment_summary(heldout_segments) if heldout_segments is not None else None,
        "history": history,
        "eval": {
            "scale_1_0": eval_full,
            "scale_0_5": eval_half,
        },
        "wall_seconds": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    prior.close()


if __name__ == "__main__":
    main()
