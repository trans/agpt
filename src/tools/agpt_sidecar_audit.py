#!/usr/bin/env python3
"""Compare live count-gate probabilities with an exported AGTS sidecar."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from agpt_count_gate import (
    CountModel,
    FeatureScaler,
    build_vocab,
    encode,
    expand_extra_features,
    read_text,
)
from agpt_history_residual import TargetSidecar, prior_contexts, read_substring_catalog


def positions_for_mode(length: int, depth: int, k_history: int, mode: str) -> list[int]:
    if mode == "fixed":
        start = depth
    elif mode == "history":
        # Matches agpt_history_residual.py stream --eval-all: each eval chunk
        # warms up k_history depth-sized lags, then scores the next position.
        start = depth * k_history
    else:
        raise ValueError(f"unknown mode: {mode}")
    return list(range(start, length))


def sidecar_matched_context(ctx: bytes, catalog: dict[bytes, int]) -> bytes:
    for start in range(len(ctx)):
        probe = ctx[start:]
        if probe in catalog:
            return probe
    return b""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-result", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--position-data", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--heldout", default="")
    ap.add_argument("--vocab-file", default="")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--k-history", type=int, default=64)
    ap.add_argument("--mode", choices=["fixed", "history"], default="history")
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--prior-floor", type=float, default=1.0e-12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    result = json.loads(Path(args.count_result).read_text(encoding="utf-8"))
    cfg = result["config"]
    train_path = args.train or cfg["train"]
    heldout_path = args.heldout or cfg["heldout"]
    vocab_path = args.vocab_file or cfg["vocab_file"]
    depth = args.depth or int(cfg["depth"])
    valid_ratio = float(cfg.get("valid_ratio", 0.05))
    base_mode = cfg.get("base_features", "core")
    base_features = (
        ["reliability", "kl_gain", "entropy_norm", "depth_norm"]
        if base_mode == "core"
        else []
    )

    vocab_chars, stoi = build_vocab(read_text(vocab_path))
    train_all = encode(read_text(train_path), stoi)
    heldout = encode(read_text(heldout_path), stoi)
    split = int(len(train_all) * (1.0 - valid_ratio))
    split = max(depth + 1, min(split, len(train_all) - depth - 1))
    count_train = train_all[:split]

    model = CountModel(
        count_train,
        len(vocab_chars),
        depth,
        expand_extra_features(cfg.get("extra_features", "")),
        base_features=base_features,
    )
    model.build()
    theta = [float(result["theta"][name]) for name in result["feature_names"]]
    scaler = None
    if result.get("feature_standardization", {}).get("enabled"):
        fs = result["feature_standardization"]
        scaler = FeatureScaler(
            [float(fs["mean"][name]) for name in result["feature_names"]],
            [float(fs["std"][name]) for name in result["feature_names"]],
        )

    all_positions = positions_for_mode(len(heldout), depth, args.k_history, args.mode)
    if args.max_positions > 0:
        all_positions = all_positions[: args.max_positions]

    catalog = read_substring_catalog(str(Path(args.position_data) / "substrings.bin"))
    sidecar = TargetSidecar(args.sidecar)
    device = torch.device("cpu")

    direct_prob_loss = 0.0
    direct_dist_loss = 0.0
    direct_matched_loss = 0.0
    sidecar_loss = 0.0
    direct_vs_matched_target_nll_delta = 0.0
    abs_target_nll_delta = 0.0
    max_target_nll_delta = 0.0
    l1_sum = 0.0
    l1_max = 0.0
    matched_l1_sum = 0.0
    matched_l1_max = 0.0
    exact = backoff = miss = 0
    n = 0

    for offset in range(0, len(all_positions), args.batch_size):
        positions = all_positions[offset : offset + args.batch_size]
        contexts = prior_contexts(heldout, positions, depth)
        targets = [heldout[pos] for pos in positions]
        sidecar_logits, ex, bo, mi = sidecar.log_probs(
            contexts, catalog, len(vocab_chars), args.prior_floor, device
        )
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        sidecar_loss += F.cross_entropy(sidecar_logits, target_tensor, reduction="sum").item()
        sidecar_probs = torch.softmax(sidecar_logits, dim=-1).cpu()
        exact += ex
        backoff += bo
        miss += mi

        for row, (ctx, target) in enumerate(zip(contexts, targets)):
            p, _weights = model.gated_prob(ctx, target, theta, scaler)
            direct_prob_loss -= math.log(max(p, 1.0e-12))
            dist = model.gated_distribution(ctx, theta, scaler)
            z = sum(dist)
            dist = [p_i / z for p_i in dist]
            direct_dist_loss -= math.log(max(dist[target], 1.0e-12))
            matched_ctx = sidecar_matched_context(ctx, catalog)
            matched_dist = model.gated_distribution(matched_ctx, theta, scaler)
            matched_z = sum(matched_dist)
            matched_dist = [p_i / matched_z for p_i in matched_dist]
            direct_matched_loss -= math.log(max(matched_dist[target], 1.0e-12))
            direct_vs_matched_target_nll_delta += abs(
                -math.log(max(dist[target], 1.0e-12))
                + math.log(max(matched_dist[target], 1.0e-12))
            )
            side_target = float(sidecar_probs[row, target].item())
            delta = abs(-math.log(max(dist[target], 1.0e-12)) + math.log(max(side_target, 1.0e-12)))
            abs_target_nll_delta += delta
            max_target_nll_delta = max(max_target_nll_delta, delta)
            l1 = sum(abs(dist[i] - float(sidecar_probs[row, i].item())) for i in range(len(dist)))
            l1_sum += l1
            l1_max = max(l1_max, l1)
            matched_l1 = sum(
                abs(matched_dist[i] - float(sidecar_probs[row, i].item()))
                for i in range(len(matched_dist))
            )
            matched_l1_sum += matched_l1
            matched_l1_max = max(matched_l1_max, matched_l1)
            n += 1

    sidecar.close()
    summary = {
        "config": {
            "count_result": args.count_result,
            "sidecar": args.sidecar,
            "position_data": args.position_data,
            "mode": args.mode,
            "depth": depth,
            "k_history": args.k_history,
            "positions": n,
            "position_start": all_positions[0] if all_positions else None,
            "position_end_inclusive": all_positions[-1] if all_positions else None,
        },
        "loss": {
            "direct_gated_prob_nats": direct_prob_loss / max(n, 1),
            "direct_gated_prob_ppl": math.exp(direct_prob_loss / max(n, 1)),
            "direct_distribution_nats": direct_dist_loss / max(n, 1),
            "direct_distribution_ppl": math.exp(direct_dist_loss / max(n, 1)),
            "direct_matched_suffix_nats": direct_matched_loss / max(n, 1),
            "direct_matched_suffix_ppl": math.exp(direct_matched_loss / max(n, 1)),
            "sidecar_logits_nats": sidecar_loss / max(n, 1),
            "sidecar_logits_ppl": math.exp(sidecar_loss / max(n, 1)),
        },
        "sidecar_hits": {
            "exact": exact,
            "backoff": backoff,
            "miss": miss,
        },
        "distribution_delta": {
            "avg_abs_target_nll_delta": abs_target_nll_delta / max(n, 1),
            "avg_abs_direct_vs_matched_target_nll_delta": (
                direct_vs_matched_target_nll_delta / max(n, 1)
            ),
            "max_abs_target_nll_delta": max_target_nll_delta,
            "avg_l1_distribution_delta": l1_sum / max(n, 1),
            "max_l1_distribution_delta": l1_max,
            "avg_l1_matched_suffix_delta": matched_l1_sum / max(n, 1),
            "max_l1_matched_suffix_delta": matched_l1_max,
        },
    }

    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
