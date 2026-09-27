#!/usr/bin/env python3
"""Count/backoff baselines with a learned pointwise trust gate.

This is intentionally independent from the neural AGPT trainers. It asks whether
simple count statistics can learn when to trust a deeper context versus backing
off, using a held-out validation tail that is not included in the count table.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EPS = 1.0e-12
PREFIX_SIMPLE_FEATURES = [
    "mass_norm",
    "branch_norm",
    "entropy_delta",
]
PREFIX_RATE_FEATURES = [
    "mass_delta",
    "branch_delta",
    "reliability_delta",
    "kl_gain_delta",
    "entropy_slope",
    "mass_slope",
    "branch_slope",
    "reliability_slope",
    "kl_gain_slope",
]
SUFFIX_FEATURES = [
    "suffix_branch_norm",
    "suffix_reliability",
    "suffix_entropy_norm",
    "suffix_kl_gain",
    "suffix_entropy_delta",
]
SUFFIX_RATE_FEATURES = [
    "suffix_mass_delta",
    "suffix_branch_delta",
    "suffix_reliability_delta",
    "suffix_kl_gain_delta",
    "suffix_entropy_slope",
    "suffix_mass_slope",
    "suffix_branch_slope",
    "suffix_reliability_slope",
    "suffix_kl_gain_slope",
]


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_vocab(vocab_text: str) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set(vocab_text))
    return chars, {ch: idx for idx, ch in enumerate(chars)}


def encode(text: str, stoi: dict[str, int]) -> bytes:
    missing = sorted(set(text) - set(stoi))
    if missing:
        raise ValueError(f"text contains {len(missing)} chars missing from vocab: {missing[:8]}")
    return bytes(stoi[ch] for ch in text)


def read_substring_catalog(path: str) -> list[bytes]:
    data = Path(path).read_bytes()
    if len(data) < 8 or data[:4] != b"ASUB":
        raise ValueError(f"bad substring catalog: {path}")
    count = struct.unpack_from("<I", data, 4)[0]
    offset = 8
    out: list[bytes] = []
    for _sid in range(count):
        if offset >= len(data):
            raise ValueError(f"truncated substring catalog: {path}")
        length = data[offset]
        offset += 1
        if offset + length > len(data):
            raise ValueError(f"truncated substring payload: {path}")
        out.append(bytes(data[offset : offset + length]))
        offset += length
    return out


def write_target_sidecar(
    path: str,
    substrings: list[bytes],
    model: "CountModel",
    theta: list[float],
    scaler: "FeatureScaler | None",
    scale: int,
    top_k: int,
    clear_cache_every: int = 10000,
) -> dict[str, int | float]:
    offsets = [0]
    entry_count = 0
    covered = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header_size = 4 + struct.calcsize("<HIIQ")
    offsets_size = (len(substrings) + 1) * 4
    with open(path, "wb+") as f:
        f.write(b"\0" * (header_size + offsets_size))
        for ctx in substrings:
            if clear_cache_every > 0 and covered > 0 and covered % clear_cache_every == 0:
                model.clear_runtime_caches()
            if len(ctx) == 0:
                offsets.append(entry_count)
                continue
            probs = model.gated_distribution(ctx, theta, scaler)
            if top_k > 0 and top_k < len(probs):
                keep = set(sorted(range(len(probs)), key=lambda tok: probs[tok], reverse=True)[:top_k])
                kept_sum = sum(probs[tok] for tok in keep)
                if kept_sum > 0.0:
                    probs = [probs[tok] / kept_sum if tok in keep else 0.0 for tok in range(len(probs))]
            total = 0
            local: list[tuple[int, int, float]] = []
            for tok, p in enumerate(probs):
                q = int(round(max(0.0, p) * scale))
                if q <= 0:
                    continue
                local.append((tok, q, p))
                total += q
            if total <= 0:
                tok = max(range(model.vocab_size), key=lambda t: probs[t])
                local.append((tok, scale, probs[tok]))
                total = scale
            if total != scale:
                tok, _q, _p = max(local, key=lambda row: row[2])
                local = [(t, c + (scale - total) if t == tok else c, pp) for t, c, pp in local]
            chunk = bytearray()
            for tok, q, _p in local:
                if q > 0:
                    chunk += struct.pack("<HI", tok, q)
                    entry_count += 1
            if chunk:
                f.write(chunk)
            covered += 1
            offsets.append(entry_count)

        offset_blob = bytearray(offsets_size)
        for i, off in enumerate(offsets):
            struct.pack_into("<i", offset_blob, i * 4, off)
        f.seek(0)
        f.write(b"AGTS")
        f.write(struct.pack("<HIIQ", 1, scale, len(substrings), entry_count))
        f.write(offset_blob)
    return {
        "substring_count": len(substrings),
        "covered_substrings": covered,
        "entries": entry_count,
        "scale": scale,
        "top_k": top_k,
        "avg_entries_per_substring": entry_count / max(len(substrings), 1),
    }


def select_positions(length: int, max_positions: int, min_pos: int, seed: int) -> list[int]:
    positions = list(range(min_pos, length))
    if max_positions <= 0 or max_positions >= len(positions):
        return positions
    rng = random.Random(seed)
    return sorted(rng.sample(positions, max_positions))


def expand_extra_features(raw: str) -> list[str]:
    features: list[str] = []
    for name in [part.strip() for part in raw.split(",") if part.strip()]:
        if name == "suffix_stats":
            features.extend(SUFFIX_FEATURES)
        elif name == "rate_stats":
            features.extend(PREFIX_RATE_FEATURES)
        elif name == "suffix_rate_stats":
            features.extend(SUFFIX_RATE_FEATURES)
        elif name == "all_stats":
            features.extend(PREFIX_SIMPLE_FEATURES)
            features.extend(SUFFIX_FEATURES)
            features.extend(PREFIX_RATE_FEATURES)
            features.extend(SUFFIX_RATE_FEATURES)
        else:
            features.append(name)
    deduped: list[str] = []
    for name in features:
        if name not in deduped:
            deduped.append(name)
    return deduped


@dataclass
class ContextStats:
    counts: Counter[int]
    total: int
    types: int


@dataclass
class FeatureScaler:
    mean: list[float]
    std: list[float]

    def transform(self, feats: tuple[float, ...]) -> tuple[float, ...]:
        # Keep the final bias feature as a literal 1.0.
        return tuple(
            (feats[i] - self.mean[i]) / self.std[i] if i < len(feats) - 1 else feats[i]
            for i in range(len(feats))
        )


class CountModel:
    def __init__(
        self,
        tokens: bytes,
        vocab_size: int,
        depth: int,
        extra_features: list[str],
        base_features: list[str] | None = None,
    ):
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.depth = depth
        self.extra_features = extra_features
        self.base_features = (
            base_features
            if base_features is not None
            else ["reliability", "kl_gain", "entropy_norm", "depth_norm"]
        )
        self.tables: list[defaultdict[bytes, Counter[int]]] = [
            defaultdict(Counter) for _ in range(depth + 1)
        ]
        self.suffix_tables: list[defaultdict[bytes, Counter[int]]] = [
            defaultdict(Counter) for _ in range(depth + 1)
        ]
        self._stats_cache: dict[bytes, ContextStats | None] = {}
        self._suffix_stats_cache: dict[bytes, ContextStats | None] = {}
        self._wb_prob_cache: dict[tuple[bytes, int], float] = {}
        self._suffix_wb_prob_cache: dict[tuple[bytes, int], float] = {}
        self._entropy_cache: dict[bytes, float] = {}
        self._suffix_entropy_cache: dict[bytes, float] = {}
        self._feature_cache: dict[bytes, tuple[float, ...] | None] = {}
        self.unigram_probs = [1.0 / vocab_size] * vocab_size
        self.suffix_unigram_probs = [1.0 / vocab_size] * vocab_size

    def clear_runtime_caches(self) -> None:
        self._stats_cache.clear()
        self._suffix_stats_cache.clear()
        self._wb_prob_cache.clear()
        self._suffix_wb_prob_cache.clear()
        self._entropy_cache.clear()
        self._suffix_entropy_cache.clear()
        self._feature_cache.clear()

    def feature_names(self) -> list[str]:
        return [
            *self.base_features,
            *self.extra_features,
            "bias",
        ]

    def build(self) -> None:
        for i, target in enumerate(self.tokens):
            max_depth = min(self.depth, i)
            for d in range(max_depth + 1):
                ctx = self.tokens[i - d : i]
                self.tables[d][ctx][target] += 1

        for start in range(1, len(self.tokens)):
            prev_token = self.tokens[start - 1]
            max_depth = min(self.depth, len(self.tokens) - start)
            for d in range(max_depth + 1):
                ctx = self.tokens[start : start + d]
                self.suffix_tables[d][ctx][prev_token] += 1

        root = self.tables[0][b""]
        total = sum(root.values())
        if total > 0:
            self.unigram_probs = [
                max(root.get(tok, 0) / total, EPS) for tok in range(self.vocab_size)
            ]
            z = sum(self.unigram_probs)
            self.unigram_probs = [p / z for p in self.unigram_probs]

        suffix_root = self.suffix_tables[0][b""]
        suffix_total = sum(suffix_root.values())
        if suffix_total > 0:
            self.suffix_unigram_probs = [
                max(suffix_root.get(tok, 0) / suffix_total, EPS)
                for tok in range(self.vocab_size)
            ]
            z = sum(self.suffix_unigram_probs)
            self.suffix_unigram_probs = [p / z for p in self.suffix_unigram_probs]

    def stats(self, ctx: bytes) -> ContextStats | None:
        cached = self._stats_cache.get(ctx)
        if cached is not None or ctx in self._stats_cache:
            return cached
        d = len(ctx)
        if d > self.depth:
            ctx = ctx[-self.depth :]
            d = self.depth
        counts = self.tables[d].get(ctx)
        if not counts:
            self._stats_cache[ctx] = None
            return None
        stats = ContextStats(counts=counts, total=sum(counts.values()), types=len(counts))
        self._stats_cache[ctx] = stats
        return stats

    def suffix_stats(self, ctx: bytes) -> ContextStats | None:
        cached = self._suffix_stats_cache.get(ctx)
        if cached is not None or ctx in self._suffix_stats_cache:
            return cached
        d = len(ctx)
        if d > self.depth:
            ctx = ctx[: self.depth]
            d = self.depth
        counts = self.suffix_tables[d].get(ctx)
        if not counts:
            self._suffix_stats_cache[ctx] = None
            return None
        stats = ContextStats(counts=counts, total=sum(counts.values()), types=len(counts))
        self._suffix_stats_cache[ctx] = stats
        return stats

    def ml_prob(self, ctx: bytes, target: int) -> float:
        stats = self.stats(ctx)
        if stats is None or stats.total <= 0:
            return 0.0
        return stats.counts.get(target, 0) / stats.total

    def longest_seen_prob(self, ctx: bytes, target: int) -> float:
        max_depth = min(self.depth, len(ctx))
        for d in range(max_depth, 0, -1):
            sub = ctx[-d:]
            stats = self.stats(sub)
            if stats is not None and stats.counts.get(target, 0) > 0:
                return max(stats.counts[target] / stats.total, EPS)
        return self.unigram_probs[target]

    def deepest_mle_floor_prob(self, ctx: bytes, target: int) -> float:
        max_depth = min(self.depth, len(ctx))
        for d in range(max_depth, 0, -1):
            sub = ctx[-d:]
            stats = self.stats(sub)
            if stats is not None and stats.total > 0:
                return max(stats.counts.get(target, 0) / stats.total, EPS)
        return self.unigram_probs[target]

    def wb_prob(self, ctx: bytes, target: int) -> float:
        if len(ctx) > self.depth:
            ctx = ctx[-self.depth :]
        key = (ctx, target)
        cached = self._wb_prob_cache.get(key)
        if cached is not None:
            return cached
        if not ctx:
            value = self.unigram_probs[target]
        else:
            stats = self.stats(ctx)
            back = self.wb_prob(ctx[1:], target)
            if stats is None or stats.total <= 0:
                value = back
            else:
                lam = stats.total / (stats.total + stats.types)
                p_ml = stats.counts.get(target, 0) / stats.total
                value = lam * p_ml + (1.0 - lam) * back
        value = max(value, EPS)
        self._wb_prob_cache[key] = value
        return value

    def suffix_wb_prob(self, ctx: bytes, target: int) -> float:
        if len(ctx) > self.depth:
            ctx = ctx[: self.depth]
        key = (ctx, target)
        cached = self._suffix_wb_prob_cache.get(key)
        if cached is not None:
            return cached
        if not ctx:
            value = self.suffix_unigram_probs[target]
        else:
            stats = self.suffix_stats(ctx)
            back = self.suffix_wb_prob(ctx[:-1], target)
            if stats is None or stats.total <= 0:
                value = back
            else:
                lam = stats.total / (stats.total + stats.types)
                p_ml = stats.counts.get(target, 0) / stats.total
                value = lam * p_ml + (1.0 - lam) * back
        value = max(value, EPS)
        self._suffix_wb_prob_cache[key] = value
        return value

    def entropy_norm(self, ctx: bytes) -> float | None:
        if len(ctx) > self.depth:
            ctx = ctx[-self.depth :]
        cached = self._entropy_cache.get(ctx)
        if cached is not None:
            return cached
        if not ctx:
            entropy = -sum(p * math.log(max(p, EPS)) for p in self.unigram_probs)
            value = entropy / math.log(self.vocab_size)
            self._entropy_cache[ctx] = value
            return value
        stats = self.stats(ctx)
        if stats is None or stats.total <= 0:
            return None
        entropy = 0.0
        for count in stats.counts.values():
            p = count / stats.total
            entropy -= p * math.log(max(p, EPS))
        value = entropy / math.log(self.vocab_size)
        self._entropy_cache[ctx] = value
        return value

    def suffix_entropy_norm(self, ctx: bytes) -> float | None:
        if len(ctx) > self.depth:
            ctx = ctx[: self.depth]
        cached = self._suffix_entropy_cache.get(ctx)
        if cached is not None:
            return cached
        if not ctx:
            entropy = -sum(p * math.log(max(p, EPS)) for p in self.suffix_unigram_probs)
            value = entropy / math.log(self.vocab_size)
            self._suffix_entropy_cache[ctx] = value
            return value
        stats = self.suffix_stats(ctx)
        if stats is None or stats.total <= 0:
            return None
        entropy = 0.0
        for count in stats.counts.values():
            p = count / stats.total
            entropy -= p * math.log(max(p, EPS))
        value = entropy / math.log(self.vocab_size)
        self._suffix_entropy_cache[ctx] = value
        return value

    def prefix_scalar_values(self, ctx: bytes) -> dict[str, float] | None:
        if len(ctx) > self.depth:
            ctx = ctx[-self.depth :]
        stats = self.stats(ctx)
        entropy_norm = self.entropy_norm(ctx)
        if stats is None or stats.total <= 0 or entropy_norm is None:
            return None
        reliability = stats.total / (stats.total + stats.types)
        mass_norm = math.log1p(stats.total) / math.log1p(len(self.tokens))
        branch_norm = math.log1p(stats.types) / math.log1p(self.vocab_size)
        kl_gain = 0.0
        if ctx:
            back_ctx = ctx[1:]
            for target, count in stats.counts.items():
                p = count / stats.total
                p_back = self.wb_prob(back_ctx, target)
                kl_gain += p * math.log(max(p, EPS) / max(p_back, EPS))
        return {
            "mass_norm": mass_norm,
            "branch_norm": branch_norm,
            "reliability": reliability,
            "kl_gain": kl_gain,
            "entropy_norm": entropy_norm,
            "depth_norm": len(ctx) / self.depth,
        }

    def suffix_scalar_values(self, ctx: bytes) -> dict[str, float] | None:
        if len(ctx) > self.depth:
            ctx = ctx[: self.depth]
        stats = self.suffix_stats(ctx)
        entropy_norm = self.suffix_entropy_norm(ctx)
        if stats is None or stats.total <= 0 or entropy_norm is None:
            return None
        reliability = stats.total / (stats.total + stats.types)
        mass_norm = math.log1p(stats.total) / math.log1p(len(self.tokens))
        branch_norm = math.log1p(stats.types) / math.log1p(self.vocab_size)
        kl_gain = 0.0
        if ctx:
            back_ctx = ctx[:-1]
            for target, count in stats.counts.items():
                p = count / stats.total
                p_back = self.suffix_wb_prob(back_ctx, target)
                kl_gain += p * math.log(max(p, EPS) / max(p_back, EPS))
        return {
            "suffix_branch_norm": branch_norm,
            "suffix_reliability": reliability,
            "suffix_kl_gain": kl_gain,
            "suffix_entropy_norm": entropy_norm,
            "_suffix_mass_norm": mass_norm,
        }

    def features(self, ctx: bytes) -> tuple[float, ...] | None:
        if len(ctx) > self.depth:
            ctx = ctx[-self.depth :]
        cached = self._feature_cache.get(ctx)
        if cached is not None or ctx in self._feature_cache:
            return cached
        if not ctx:
            self._feature_cache[ctx] = None
            return None
        current = self.prefix_scalar_values(ctx)
        if current is None:
            self._feature_cache[ctx] = None
            return None
        back_ctx = ctx[1:]
        back = self.prefix_scalar_values(back_ctx) or current
        root = self.prefix_scalar_values(b"") or current
        depth = max(len(ctx), 1)
        feature_values = dict(current)
        feature_values.update(
            {
                "entropy_delta": back["entropy_norm"] - current["entropy_norm"],
                "mass_delta": current["mass_norm"] - back["mass_norm"],
                "branch_delta": current["branch_norm"] - back["branch_norm"],
                "reliability_delta": current["reliability"] - back["reliability"],
                "kl_gain_delta": current["kl_gain"] - back["kl_gain"],
                "entropy_slope": (root["entropy_norm"] - current["entropy_norm"]) / depth,
                "mass_slope": (current["mass_norm"] - root["mass_norm"]) / depth,
                "branch_slope": (current["branch_norm"] - root["branch_norm"]) / depth,
                "reliability_slope": (current["reliability"] - root["reliability"]) / depth,
                "kl_gain_slope": (current["kl_gain"] - root["kl_gain"]) / depth,
            }
        )
        if any(
            name in self.extra_features
            for name in [*SUFFIX_FEATURES, *SUFFIX_RATE_FEATURES]
        ):
            suffix_current = self.suffix_scalar_values(ctx)
            suffix_back_ctx = ctx[:-1]
            suffix_back = self.suffix_scalar_values(suffix_back_ctx)
            suffix_root = self.suffix_scalar_values(b"")
            if suffix_current is None:
                suffix_current = {
                    "_suffix_mass_norm": 0.0,
                    "suffix_branch_norm": 0.0,
                    "suffix_reliability": 0.0,
                    "suffix_entropy_norm": 0.0,
                    "suffix_kl_gain": 0.0,
                }
            suffix_back = suffix_back or suffix_current
            suffix_root = suffix_root or suffix_current
            suffix_values = dict(suffix_current)
            suffix_values.update(
                {
                    "suffix_entropy_delta": (
                        suffix_back["suffix_entropy_norm"]
                        - suffix_current["suffix_entropy_norm"]
                    ),
                    "suffix_mass_delta": (
                        suffix_current["_suffix_mass_norm"]
                        - suffix_back["_suffix_mass_norm"]
                    ),
                    "suffix_branch_delta": (
                        suffix_current["suffix_branch_norm"]
                        - suffix_back["suffix_branch_norm"]
                    ),
                    "suffix_reliability_delta": (
                        suffix_current["suffix_reliability"]
                        - suffix_back["suffix_reliability"]
                    ),
                    "suffix_kl_gain_delta": (
                        suffix_current["suffix_kl_gain"]
                        - suffix_back["suffix_kl_gain"]
                    ),
                    "suffix_entropy_slope": (
                        suffix_root["suffix_entropy_norm"]
                        - suffix_current["suffix_entropy_norm"]
                    )
                    / depth,
                    "suffix_mass_slope": (
                        suffix_current["_suffix_mass_norm"]
                        - suffix_root["_suffix_mass_norm"]
                    )
                    / depth,
                    "suffix_branch_slope": (
                        suffix_current["suffix_branch_norm"]
                        - suffix_root["suffix_branch_norm"]
                    )
                    / depth,
                    "suffix_reliability_slope": (
                        suffix_current["suffix_reliability"]
                        - suffix_root["suffix_reliability"]
                    )
                    / depth,
                    "suffix_kl_gain_slope": (
                        suffix_current["suffix_kl_gain"]
                        - suffix_root["suffix_kl_gain"]
                    )
                    / depth,
                }
            )
            feature_values.update(suffix_values)
        feats = [feature_values[name] for name in self.base_features]
        for name in self.extra_features:
            feats.append(feature_values[name])
        feats.append(1.0)
        result = tuple(feats)
        self._feature_cache[ctx] = result
        return result

    def scaled_features(
        self, ctx: bytes, scaler: FeatureScaler | None = None
    ) -> tuple[float, ...] | None:
        feats = self.features(ctx)
        if feats is None:
            return None
        if scaler is None:
            return feats
        return scaler.transform(feats)

    def gated_prob_and_grad(
        self, ctx: bytes, target: int, theta: list[float], scaler: FeatureScaler | None = None
    ) -> tuple[float, list[float], list[float]]:
        max_depth = min(self.depth, len(ctx))
        q = self.unigram_probs[target]
        grad = [0.0] * len(theta)
        weights: list[float] = []

        for d in range(1, max_depth + 1):
            sub = ctx[-d:]
            feats = self.scaled_features(sub, scaler)
            if feats is None:
                continue
            p_ml = self.ml_prob(sub, target)
            z = sum(theta[j] * feats[j] for j in range(len(theta)))
            if z >= 0:
                ez = math.exp(-z)
                w = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(z)
                w = ez / (1.0 + ez)
            old_q = q
            old_grad = grad
            dw = w * (1.0 - w)
            q = w * p_ml + (1.0 - w) * old_q
            grad = [
                (1.0 - w) * old_grad[j] + dw * feats[j] * (p_ml - old_q)
                for j in range(len(theta))
            ]
            weights.append(w)

        return max(q, EPS), grad, weights

    def gated_prob(
        self, ctx: bytes, target: int, theta: list[float], scaler: FeatureScaler | None = None
    ) -> tuple[float, list[float]]:
        q, _grad, weights = self.gated_prob_and_grad(ctx, target, theta, scaler)
        return q, weights

    def gated_distribution(
        self, ctx: bytes, theta: list[float], scaler: FeatureScaler | None = None
    ) -> list[float]:
        probs = self.unigram_probs[:]
        for d in range(1, min(self.depth, len(ctx)) + 1):
            sub = ctx[-d:]
            feats = self.scaled_features(sub, scaler)
            if feats is None:
                continue
            stats = self.stats(sub)
            if stats is None or stats.total <= 0:
                continue
            z = sum(theta[j] * feats[j] for j in range(len(theta)))
            if z >= 0:
                ez = math.exp(-z)
                w = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(z)
                w = ez / (1.0 + ez)
            next_probs = [(1.0 - w) * p for p in probs]
            scale = w / stats.total
            for tok, count in stats.counts.items():
                next_probs[tok] += scale * count
            probs = next_probs
        z = sum(probs)
        if z <= 0.0:
            return [1.0 / self.vocab_size] * self.vocab_size
        return [max(p / z, EPS) for p in probs]


def mean_loss_for_positions(
    tokens: bytes,
    positions: Iterable[int],
    depth: int,
    prob_fn,
) -> tuple[float, float, int]:
    total_loss = 0.0
    n = 0
    for i in positions:
        ctx = tokens[max(0, i - depth) : i]
        target = tokens[i]
        p = max(prob_fn(ctx, target), EPS)
        total_loss -= math.log(p)
        n += 1
    loss = total_loss / max(n, 1)
    return loss, math.exp(loss), n


def fit_feature_scaler(
    model: CountModel,
    tokens: bytes,
    positions: list[int],
    depth: int,
) -> FeatureScaler:
    n_features = len(model.feature_names())
    sums = [0.0] * n_features
    sums_sq = [0.0] * n_features
    count = 0
    for i in positions:
        ctx = tokens[max(0, i - depth) : i]
        for d in range(1, min(depth, len(ctx)) + 1):
            feats = model.features(ctx[-d:])
            if feats is None:
                continue
            for j, value in enumerate(feats):
                if j == n_features - 1:
                    continue
                sums[j] += value
                sums_sq[j] += value * value
            count += 1
    if count <= 0:
        raise ValueError("cannot standardize features: no train contexts produced features")

    mean = [0.0] * n_features
    std = [1.0] * n_features
    for j in range(n_features - 1):
        mean[j] = sums[j] / count
        var = max(sums_sq[j] / count - mean[j] * mean[j], 0.0)
        std[j] = max(math.sqrt(var), 1.0e-6)
    mean[-1] = 0.0
    std[-1] = 1.0
    return FeatureScaler(mean=mean, std=std)


def train_gate(
    model: CountModel,
    valid_tokens: bytes,
    depth: int,
    positions: list[int],
    epochs: int,
    lr: float,
    scaler: FeatureScaler | None = None,
) -> tuple[list[float], list[dict[str, float]]]:
    theta = [0.0] * len(model.feature_names())
    m = [0.0] * len(theta)
    v = [0.0] * len(theta)
    beta1 = 0.9
    beta2 = 0.999
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_weights = 0.0
        total_weight_count = 0
        grad_sum = [0.0] * len(theta)
        random.Random(epoch).shuffle(positions)

        for i in positions:
            ctx = valid_tokens[max(0, i - depth) : i]
            target = valid_tokens[i]
            q, grad_q, weights = model.gated_prob_and_grad(ctx, target, theta, scaler)
            total_loss -= math.log(q)
            for j in range(len(theta)):
                grad_sum[j] += -grad_q[j] / q
            total_weights += sum(weights)
            total_weight_count += len(weights)

        scale = 1.0 / max(len(positions), 1)
        step += 1
        for j in range(len(theta)):
            g = grad_sum[j] * scale
            m[j] = beta1 * m[j] + (1.0 - beta1) * g
            v[j] = beta2 * v[j] + (1.0 - beta2) * g * g
            m_hat = m[j] / (1.0 - beta1**step)
            v_hat = v[j] / (1.0 - beta2**step)
            theta[j] -= lr * m_hat / (math.sqrt(v_hat) + 1.0e-8)

        loss = total_loss * scale
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "ppl": math.exp(loss),
                "avg_gate_weight": total_weights / max(total_weight_count, 1),
            }
        )

    return theta, history


def gate_weight_profile(
    model: CountModel,
    tokens: bytes,
    positions: list[int],
    depth: int,
    theta: list[float],
    scaler: FeatureScaler | None = None,
) -> list[dict[str, float]]:
    feature_names = model.feature_names()
    sums = [
        {"weight": 0.0, "count": 0.0, **{name: 0.0 for name in feature_names[:-1]}}
        for _ in range(depth + 1)
    ]
    for i in positions:
        ctx = tokens[max(0, i - depth) : i]
        for d in range(1, min(depth, len(ctx)) + 1):
            sub = ctx[-d:]
            feats = model.scaled_features(sub, scaler)
            if feats is None:
                continue
            z = sum(theta[j] * feats[j] for j in range(len(theta)))
            w = 1.0 / (1.0 + math.exp(-z)) if -60.0 < z < 60.0 else (1.0 if z >= 0 else 0.0)
            row = sums[d]
            row["weight"] += w
            row["count"] += 1.0
            for j, name in enumerate(feature_names[:-1]):
                row[name] += feats[j]
    profile = []
    for d in range(1, depth + 1):
        count = sums[d]["count"]
        if count <= 0:
            continue
        row = {
            "depth": d,
            "count": int(count),
            "avg_weight": sums[d]["weight"] / count,
        }
        for name in feature_names[:-1]:
            row[f"avg_{name}"] = sums[d][name] / count
        profile.append(row)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/.splits/4fa9aec1db6b3aea/train_corpus.txt")
    parser.add_argument("--heldout", default="data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt")
    parser.add_argument("--vocab-file", default="data/input.txt")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-fit-positions", type=int, default=50000)
    parser.add_argument("--max-eval-positions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--standardize-features",
        action="store_true",
        help="Z-score non-bias gate features using train-fit contexts only.",
    )
    parser.add_argument(
        "--extra-features",
        default="",
        help=(
            "Comma-separated extra gate features. Supported: entropy_delta, "
            "mass_norm, branch_norm, suffix_stats, rate_stats, "
            "suffix_rate_stats, all_stats, or individual feature names."
        ),
    )
    parser.add_argument(
        "--base-features",
        choices=["core", "none"],
        default="core",
        help="Use the four core gate features, or only the requested extra features plus bias.",
    )
    parser.add_argument(
        "--substring-catalog",
        default="",
        help="Optional position-data substrings.bin catalog for exporting smoothed targets.",
    )
    parser.add_argument(
        "--target-sidecar-out",
        default="",
        help="Optional AGTS smoothed target sidecar path keyed by substring id.",
    )
    parser.add_argument("--target-sidecar-scale", type=int, default=1000000)
    parser.add_argument(
        "--target-sidecar-top-k",
        type=int,
        default=0,
        help="Keep only top-k target probabilities per substring before quantization; 0 keeps all.",
    )
    parser.add_argument(
        "--target-sidecar-clear-cache-every",
        type=int,
        default=10000,
        help="Clear count-model runtime caches every N exported substrings; 0 disables.",
    )
    parser.add_argument("--output", default="rnd/count-backoff-gate/result.json")
    args = parser.parse_args()

    t0 = time.time()
    vocab_chars, stoi = build_vocab(read_text(args.vocab_file))
    train_all = encode(read_text(args.train), stoi)
    heldout = encode(read_text(args.heldout), stoi)
    split = int(len(train_all) * (1.0 - args.valid_ratio))
    split = max(args.depth + 1, min(split, len(train_all) - args.depth - 1))
    count_train = train_all[:split]
    valid = train_all[split:]

    extra_features = expand_extra_features(args.extra_features)
    supported = {
        "reliability",
        "kl_gain",
        "entropy_norm",
        "depth_norm",
        "entropy_delta",
        "mass_norm",
        "branch_norm",
        *PREFIX_RATE_FEATURES,
        *SUFFIX_FEATURES,
        *SUFFIX_RATE_FEATURES,
    }
    unsupported = sorted(set(extra_features) - supported)
    if unsupported:
        raise ValueError(f"unsupported extra features: {unsupported}")

    base_features = (
        ["reliability", "kl_gain", "entropy_norm", "depth_norm"]
        if args.base_features == "core"
        else []
    )

    model = CountModel(
        count_train,
        len(vocab_chars),
        args.depth,
        extra_features,
        base_features=base_features,
    )
    model.build()
    build_wall = time.time() - t0

    fit_positions = select_positions(len(valid), args.max_fit_positions, 1, args.seed)
    eval_valid_positions = select_positions(len(valid), args.max_eval_positions, 1, args.seed + 1)
    eval_held_positions = select_positions(len(heldout), args.max_eval_positions, 1, args.seed + 2)
    eval_held_fixed_positions = select_positions(
        len(heldout), args.max_eval_positions, args.depth, args.seed + 3
    )

    scaler = (
        fit_feature_scaler(model, valid, fit_positions, args.depth)
        if args.standardize_features
        else None
    )

    t1 = time.time()
    theta, history = train_gate(
        model,
        valid,
        args.depth,
        fit_positions,
        args.epochs,
        args.lr,
        scaler,
    )
    train_wall = time.time() - t1

    def eval_split(tokens: bytes, positions: list[int]) -> dict[str, dict[str, float]]:
        rows: dict[str, dict[str, float]] = {}
        variants = {
            "unigram": lambda ctx, y: model.unigram_probs[y],
            "deepest_mle_floor": lambda ctx, y: model.deepest_mle_floor_prob(ctx, y),
            "witten_bell": lambda ctx, y: model.wb_prob(ctx[-args.depth :], y),
            "learned_gate": lambda ctx, y: model.gated_prob(ctx, y, theta, scaler)[0],
            "target_backoff_oracle": lambda ctx, y: model.longest_seen_prob(ctx, y),
        }
        for name, fn in variants.items():
            loss, ppl, n = mean_loss_for_positions(tokens, positions, args.depth, fn)
            rows[name] = {"loss_nats": loss, "ppl_e": ppl, "positions": n}
        return rows

    valid_scores = eval_split(valid, eval_valid_positions)
    heldout_scores = eval_split(heldout, eval_held_positions)
    heldout_fixed_scores = eval_split(heldout, eval_held_fixed_positions)
    profile_positions = eval_held_fixed_positions[: min(10000, len(eval_held_fixed_positions))]
    profile = gate_weight_profile(model, heldout, profile_positions, args.depth, theta, scaler)
    target_sidecar = None
    if args.target_sidecar_out:
        if not args.substring_catalog:
            raise ValueError("--target-sidecar-out requires --substring-catalog")
        substrings = read_substring_catalog(args.substring_catalog)
        target_sidecar = write_target_sidecar(
            args.target_sidecar_out,
            substrings,
            model,
            theta,
            scaler,
            args.target_sidecar_scale,
            args.target_sidecar_top_k,
            args.target_sidecar_clear_cache_every,
        )
        target_sidecar["path"] = args.target_sidecar_out

    result = {
        "config": vars(args),
        "corpus": {
            "vocab_size": len(vocab_chars),
            "train_total_chars": len(train_all),
            "count_train_chars": len(count_train),
            "valid_chars": len(valid),
            "heldout_chars": len(heldout),
            "valid_ratio": args.valid_ratio,
        },
        "feature_names": model.feature_names(),
        "feature_standardization": {
            "enabled": scaler is not None,
            "mean": dict(zip(model.feature_names(), scaler.mean)) if scaler else None,
            "std": dict(zip(model.feature_names(), scaler.std)) if scaler else None,
        },
        "theta": dict(zip(model.feature_names(), theta)),
        "history": history,
        "scores": {
            "valid_rolling": valid_scores,
            "heldout_rolling": heldout_scores,
            "heldout_fixed_skip_depth": heldout_fixed_scores,
        },
        "gate_profile_heldout_fixed": profile,
        "target_sidecar": target_sidecar,
        "wall_seconds": {
            "build_counts": build_wall,
            "fit_gate": train_wall,
            "total": time.time() - t0,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
