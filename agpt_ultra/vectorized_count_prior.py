from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agpt_ultra.count_gate import (
    EPS,
    SUFFIX_FEATURES,
    PackedCountModel,
    PackedNgramTable,
    packed_keys_for_windows,
)


@dataclass
class DepthPriorTable:
    table: PackedNgramTable
    features: np.ndarray
    gates: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _table_entropy_norm(table: PackedNgramTable, vocab_size: int) -> np.ndarray:
    n = len(table.keys)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    pair_rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(table.offsets).astype(np.int64))
    probs = table.counts.astype(np.float64) / table.totals[pair_rows].astype(np.float64)
    contrib = -(probs * np.log(np.maximum(probs, EPS)))
    entropy = np.add.reduceat(contrib, table.offsets[:-1].astype(np.int64))
    return (entropy / np.log(vocab_size)).astype(np.float32)


def _kl_gain_from_backoff(
    table: PackedNgramTable,
    back_indices: np.ndarray,
    back_wb: np.ndarray,
) -> np.ndarray:
    n = len(table.keys)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    pair_rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(table.offsets).astype(np.int64))
    probs = table.counts.astype(np.float64) / table.totals[pair_rows].astype(np.float64)
    p_back = back_wb[back_indices[pair_rows], table.tokens].astype(np.float64)
    contrib = probs * np.log(np.maximum(probs, EPS) / np.maximum(p_back, EPS))
    return np.add.reduceat(contrib, table.offsets[:-1].astype(np.int64)).astype(np.float32)


def _dense_wb_from_backoff(
    table: PackedNgramTable,
    back_indices: np.ndarray,
    back_wb: np.ndarray,
) -> np.ndarray:
    n = len(table.keys)
    vocab_size = back_wb.shape[1]
    if n == 0:
        return np.empty((0, vocab_size), dtype=np.float32)
    totals = table.totals.astype(np.float32)
    lambdas = totals / (totals + table.types.astype(np.float32))
    wb = (1.0 - lambdas[:, None]) * back_wb[back_indices]
    pair_rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(table.offsets).astype(np.int64))
    additions = lambdas[pair_rows] * table.counts.astype(np.float32) / totals[pair_rows]
    np.add.at(wb, (pair_rows, table.tokens), additions)
    return np.maximum(wb, EPS).astype(np.float32, copy=False)


def _back_indices_for_prefix(table: PackedNgramTable, previous: PackedNgramTable, depth: int) -> np.ndarray:
    if depth == 1:
        return np.zeros(len(table.keys), dtype=np.int64)
    mask = np.uint64((1 << (8 * (depth - 1))) - 1)
    back_keys = table.keys & mask
    return np.searchsorted(previous.keys, back_keys).astype(np.int64)


def _back_indices_for_suffix(table: PackedNgramTable, previous: PackedNgramTable, depth: int) -> np.ndarray:
    if depth == 1:
        return np.zeros(len(table.keys), dtype=np.int64)
    back_keys = table.keys >> np.uint64(8)
    return np.searchsorted(previous.keys, back_keys).astype(np.int64)


def _suffix_feature_maps(model: PackedCountModel) -> dict[str, list[np.ndarray]]:
    vocab_size = model.vocab_size
    root_entropy = -sum(p * np.log(max(p, EPS)) for p in model.suffix_unigram_probs) / np.log(vocab_size)
    previous_wb = np.asarray(model.suffix_unigram_probs, dtype=np.float32)[None, :]
    previous_entropy = np.asarray([root_entropy], dtype=np.float32)
    previous_table = model.suffix_tables[0]
    result = {name: [np.empty(0, dtype=np.float32)] for name in SUFFIX_FEATURES}

    for depth in range(1, model.depth + 1):
        table = model.suffix_tables[depth]
        n = len(table.keys)
        if n == 0:
            for name in SUFFIX_FEATURES:
                result[name].append(np.empty(0, dtype=np.float32))
            previous_table = table
            previous_wb = np.empty((0, vocab_size), dtype=np.float32)
            previous_entropy = np.empty(0, dtype=np.float32)
            continue
        back_idx = _back_indices_for_suffix(table, previous_table, depth)
        entropy = _table_entropy_norm(table, vocab_size)
        kl_gain = _kl_gain_from_backoff(table, back_idx, previous_wb)
        total = table.totals.astype(np.float32)
        types = table.types.astype(np.float32)
        result["suffix_mass_norm"].append(
            (np.log1p(total) / np.log1p(len(model.tokens))).astype(np.float32)
        )
        result["suffix_reliability"].append((total / (total + types)).astype(np.float32))
        result["suffix_entropy_norm"].append(entropy)
        result["suffix_kl_gain"].append(kl_gain)
        result["suffix_entropy_delta"].append((previous_entropy[back_idx] - entropy).astype(np.float32))
        previous_wb = _dense_wb_from_backoff(table, back_idx, previous_wb)
        previous_entropy = entropy
        previous_table = table
    return result


def build_depth_prior_tables(model: PackedCountModel, theta: list[float]) -> list[DepthPriorTable]:
    feature_names = model.feature_names()
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    suffix_maps = _suffix_feature_maps(model)
    vocab_size = model.vocab_size
    root_entropy = -sum(p * np.log(max(p, EPS)) for p in model.unigram_probs) / np.log(vocab_size)
    previous_wb = np.asarray(model.unigram_probs, dtype=np.float32)[None, :]
    previous_entropy = np.asarray([root_entropy], dtype=np.float32)
    previous_table = model.tables[0]
    theta_arr = np.asarray(theta, dtype=np.float32)
    output: list[DepthPriorTable] = []

    for depth in range(1, model.depth + 1):
        table = model.tables[depth]
        n = len(table.keys)
        features = np.zeros((n, len(feature_names)), dtype=np.float32)
        if n == 0:
            output.append(DepthPriorTable(table=table, features=features, gates=np.empty(0, dtype=np.float32)))
            previous_table = table
            previous_wb = np.empty((0, vocab_size), dtype=np.float32)
            previous_entropy = np.empty(0, dtype=np.float32)
            continue

        back_idx = _back_indices_for_prefix(table, previous_table, depth)
        entropy = _table_entropy_norm(table, vocab_size)
        kl_gain = _kl_gain_from_backoff(table, back_idx, previous_wb)
        totals = table.totals.astype(np.float32)
        types = table.types.astype(np.float32)
        features[:, feature_index["reliability"]] = totals / (totals + types)
        features[:, feature_index["kl_gain"]] = kl_gain
        features[:, feature_index["entropy_norm"]] = entropy
        features[:, feature_index["depth_norm"]] = depth / model.depth
        if "entropy_delta" in feature_index:
            features[:, feature_index["entropy_delta"]] = previous_entropy[back_idx] - entropy

        if any(name in feature_index for name in SUFFIX_FEATURES):
            suffix_table = model.suffix_tables[depth]
            suffix_idx = np.searchsorted(suffix_table.keys, table.keys)
            suffix_mask = suffix_idx < len(suffix_table.keys)
            if np.any(suffix_mask):
                suffix_mask[suffix_mask] &= suffix_table.keys[suffix_idx[suffix_mask]] == table.keys[suffix_mask]
            for name in SUFFIX_FEATURES:
                if name not in feature_index:
                    continue
                values = suffix_maps[name][depth]
                if len(values) > 0:
                    features[suffix_mask, feature_index[name]] = values[suffix_idx[suffix_mask]]

        features[:, feature_index["bias"]] = 1.0
        gates = _sigmoid(features @ theta_arr)
        output.append(DepthPriorTable(table=table, features=features, gates=gates))
        previous_wb = _dense_wb_from_backoff(table, back_idx, previous_wb)
        previous_entropy = entropy
        previous_table = table
    return output


def compile_prior_rows_numpy(
    ids: list[int],
    model: PackedCountModel,
    depth_tables: list[DepthPriorTable],
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = max(0, len(ids) - 1)
    log_rows = np.empty((row_count, model.vocab_size), dtype=np.float32)
    feature_rows = np.zeros((row_count, len(model.feature_names())), dtype=np.float32)
    if feature_rows.shape[1] > 0:
        feature_rows[:, -1] = 1.0
    if row_count == 0:
        return log_rows, feature_rows

    arr = np.asarray(ids, dtype=np.uint8)
    for start in range(0, row_count, chunk_size):
        end = min(row_count, start + chunk_size)
        log_chunk, feature_chunk = _compile_prior_chunk(arr, model, depth_tables, start, end)
        log_rows[start:end] = log_chunk
        feature_rows[start:end] = feature_chunk
    return log_rows, feature_rows


def compile_prior_rows_numpy_mmap(
    ids: list[int],
    model: PackedCountModel,
    depth_tables: list[DepthPriorTable],
    log_path,
    feature_path,
    log_dtype=np.float16,
    feature_dtype=np.float32,
    chunk_size: int = 100_000,
) -> tuple[np.memmap, np.memmap]:
    log_path = Path(log_path)
    feature_path = Path(feature_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = max(0, len(ids) - 1)
    log_shape = (row_count, model.vocab_size)
    feature_shape = (row_count, len(model.feature_names()))
    log_rows = np.memmap(log_path, dtype=np.dtype(log_dtype), mode="w+", shape=log_shape)
    feature_rows = np.memmap(feature_path, dtype=np.dtype(feature_dtype), mode="w+", shape=feature_shape)
    if row_count == 0:
        log_rows.flush()
        feature_rows.flush()
        return (
            np.memmap(log_path, dtype=np.dtype(log_dtype), mode="r", shape=log_shape),
            np.memmap(feature_path, dtype=np.dtype(feature_dtype), mode="r", shape=feature_shape),
        )
    arr = np.asarray(ids, dtype=np.uint8)
    for start in range(0, row_count, chunk_size):
        end = min(row_count, start + chunk_size)
        log_chunk, feature_chunk = _compile_prior_chunk(arr, model, depth_tables, start, end)
        log_rows[start:end] = log_chunk
        feature_rows[start:end] = feature_chunk
    log_rows.flush()
    feature_rows.flush()
    del log_rows, feature_rows
    return (
        np.memmap(log_path, dtype=np.dtype(log_dtype), mode="r", shape=log_shape),
        np.memmap(feature_path, dtype=np.dtype(feature_dtype), mode="r", shape=feature_shape),
    )


def _compile_prior_chunk(
    arr: np.ndarray,
    model: PackedCountModel,
    depth_tables: list[DepthPriorTable],
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    root = np.asarray(model.unigram_probs, dtype=np.float32)
    q = np.broadcast_to(root, (end - start, model.vocab_size)).copy()
    chunk_features = np.zeros((end - start, len(model.feature_names())), dtype=np.float32)
    if chunk_features.shape[1] > 0:
        chunk_features[:, -1] = 1.0

    for depth, prior_table in enumerate(depth_tables, start=1):
        valid_start = max(start, depth - 1)
        if valid_start >= end:
            continue
        keys = packed_keys_for_windows(arr, valid_start + 1 - depth, end - valid_start, depth)
        table = prior_table.table
        idx = np.searchsorted(table.keys, keys)
        mask = idx < len(table.keys)
        if np.any(mask):
            mask[mask] &= table.keys[idx[mask]] == keys[mask]
        if not np.any(mask):
            continue
        local_rows = np.nonzero(mask)[0] + (valid_start - start)
        table_rows = idx[mask].astype(np.int64, copy=False)
        w = prior_table.gates[table_rows]
        q[local_rows] *= (1.0 - w[:, None])
        repeat_counts = (
            table.offsets[table_rows + 1].astype(np.int64) - table.offsets[table_rows].astype(np.int64)
        )
        expanded_local = np.repeat(local_rows, repeat_counts)
        starts = table.offsets[table_rows].astype(np.int64)
        pair_count = int(repeat_counts.sum())
        pair_indices = (
            np.repeat(starts, repeat_counts)
            + np.arange(pair_count, dtype=np.int64)
            - np.repeat(np.cumsum(repeat_counts) - repeat_counts, repeat_counts)
        )
        additions = (
            np.repeat(w / table.totals[table_rows].astype(np.float32), repeat_counts)
            * table.counts[pair_indices].astype(np.float32)
        )
        np.add.at(q, (expanded_local, table.tokens[pair_indices]), additions)
        chunk_features[local_rows] = prior_table.features[table_rows]

    q /= q.sum(axis=1, keepdims=True)
    return np.log(np.maximum(q, EPS)), chunk_features
