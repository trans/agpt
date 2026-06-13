from __future__ import annotations

import argparse
import csv
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, make_circular_samples, make_samples, read_text
from agpt_ultra.eval import split_text
from agpt_ultra.flat_ops import compute_flat_hidden_states, samples_to_flat_trie
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.hybrid import prefix_hidden_state
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.state_fisher import parse_step_scale, state_fisher_correct_hidden


@dataclass(frozen=True)
class EvalRow:
    prefix: str
    sample_count: int
    node_count: int
    transition_mass: int
    train_prior_mass: int
    val_count_prior_ppl: float
    val_prior_ppl: float
    val_residual_ppl: float
    val_loss: float
    runtime_sec: float
    peak_rss_kb: int


@dataclass
class BackoffStats:
    tokens: int = 0
    count_prior_loss: float = 0.0
    prior_loss: float = 0.0
    residual_loss: float = 0.0
    nodes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out evaluation for Fisher-residual checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/fisher_residual_eval.csv"))
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--prefix-length", type=int, default=None)
    parser.add_argument("--state-iterations", type=int, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--step-scale", default=None)
    parser.add_argument("--max-auto-eta", type=float, default=None)
    parser.add_argument("--auto-newton-steps", type=int, default=None)
    parser.add_argument("--curvature", choices=["model", "empirical"], default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--val-circular", action="store_true")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--eval-split", choices=["val", "train"], default="val")
    parser.add_argument("--count-prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-hidden-context", choices=["full", "backoff"], default="full")
    return parser.parse_args()


def checkpoint_arg(checkpoint: dict, args: argparse.Namespace, name: str, default):
    value = getattr(args, name)
    if value is not None:
        return value
    value = checkpoint.get("args", {}).get(name, default)
    return default if value is None else value


def prefix_ranges(sorted_samples: list[str], prefix_length: int) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    if not sorted_samples:
        return ranges
    start = 0
    current = sorted_samples[0][:prefix_length]
    for index, sample in enumerate(sorted_samples[1:], start=1):
        prefix = sample[:prefix_length]
        if prefix != current:
            ranges.append((current, start, index))
            current = prefix
            start = index
    ranges.append((current, start, len(sorted_samples)))
    return ranges


def trie_node_paths(trie: FlatTrie) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = [()]
    for node_id in range(1, trie.node_count):
        parent = int(trie.parents[node_id].item())
        paths.append(paths[parent] + (int(trie.tokens[node_id].item()),))
    return paths


def collect_needed_contexts(encoded_samples: list[list[int]]) -> set[tuple[int, ...]]:
    needed: set[tuple[int, ...]] = {()}
    for ids in encoded_samples:
        for end in range(1, len(ids)):
            context = tuple(ids[:end])
            while context:
                needed.add(context)
                context = context[1:]
    return needed


def build_train_context_counts(
    encoded_samples: list[list[int]],
    needed_contexts: set[tuple[int, ...]],
    vocab_size: int,
) -> dict[tuple[int, ...], torch.Tensor]:
    counts: dict[tuple[int, ...], torch.Tensor] = {(): torch.zeros(vocab_size, dtype=torch.float32)}
    for ids in encoded_samples:
        for end in range(1, len(ids)):
            target = ids[end]
            counts[()][target] += 1.0
            context = tuple(ids[:end])
            if context not in needed_contexts:
                continue
            row = counts.get(context)
            if row is None:
                row = torch.zeros(vocab_size, dtype=torch.float32)
                counts[context] = row
            row[target] += 1.0
    return counts


def lookup_backoff_counts(
    context: tuple[int, ...],
    train_counts: dict[tuple[int, ...], torch.Tensor],
) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    cursor = context
    dropped = 0
    while cursor:
        row = train_counts.get(cursor)
        if row is not None and float(row.sum().item()) > 0.0:
            return row, dropped, cursor
        cursor = cursor[1:]
        dropped += 1
    return train_counts[()], dropped, ()


def train_prior_counts_for_trie(
    prefix_ids: tuple[int, ...],
    trie: FlatTrie,
    train_counts: dict[tuple[int, ...], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, ...]], int]:
    rows: list[torch.Tensor] = []
    dropped_rows: list[int] = []
    matched_contexts: list[tuple[int, ...]] = []
    mass = 0
    for path in trie_node_paths(trie):
        row, dropped, matched_context = lookup_backoff_counts(prefix_ids + path, train_counts)
        rows.append(row)
        dropped_rows.append(dropped)
        matched_contexts.append(matched_context)
        mass += int(row.sum().item())
    return torch.stack(rows, dim=0), torch.tensor(dropped_rows, dtype=torch.long), matched_contexts, mass


@torch.no_grad()
def hidden_states_for_contexts(model: TinyCharRNN, contexts: list[tuple[int, ...]]) -> torch.Tensor:
    device = next(model.parameters()).device
    cache: dict[tuple[int, ...], torch.Tensor] = {}
    rows: list[torch.Tensor] = []
    for context in contexts:
        state = cache.get(context)
        if state is None:
            hidden = model.initial_state(1, device)
            for token_id in context:
                token = torch.tensor([token_id], dtype=torch.long, device=device)
                _, hidden = model.step(token, hidden)
            state = hidden.squeeze(0)
            cache[context] = state
        rows.append(state)
    return torch.stack(rows, dim=0)


@torch.no_grad()
def score_subtree(
    model: TinyCharRNN,
    residual_head: nn.Linear,
    vocab: CharVocab,
    prefix: str,
    trie: FlatTrie,
    train_counts: dict[tuple[int, ...], torch.Tensor],
    damping: float,
    step_scale: float | str,
    max_auto_eta: float,
    auto_newton_steps: int,
    state_iterations: int,
    curvature: str,
    chunk_size: int,
    residual_scale: float,
    count_prior_alpha: float,
    prior_hidden_context: str,
) -> tuple[float, float, float, int, int, dict[int, BackoffStats]]:
    initial_hidden = prefix_hidden_state(model, vocab, prefix)
    hidden = compute_flat_hidden_states(model, trie, initial_hidden=initial_hidden)
    prior_counts, dropped_by_node, matched_contexts, prior_mass = train_prior_counts_for_trie(
        tuple(vocab.encode(prefix)), trie, train_counts
    )
    prior_base_hidden = hidden
    if prior_hidden_context == "backoff":
        prior_base_hidden = hidden_states_for_contexts(model, matched_contexts)
    correction = state_fisher_correct_hidden(
        model,
        prior_base_hidden,
        prior_counts,
        damping=damping,
        step_scale=step_scale,
        max_auto_eta=max_auto_eta,
        auto_newton_steps=auto_newton_steps,
        state_iterations=state_iterations,
        curvature=curvature,
        chunk_size=chunk_size,
    )
    val_counts = trie.transition_counts.to(device=hidden.device, dtype=hidden.dtype)
    totals = val_counts.sum(dim=1)
    active = torch.nonzero(totals > 0, as_tuple=False).flatten()
    unigram_counts = train_counts[()].to(device=hidden.device, dtype=hidden.dtype)
    unigram_probs = unigram_counts / unigram_counts.sum().clamp_min(1.0)
    count_prior_loss = hidden.new_tensor(0.0)
    prior_loss = hidden.new_tensor(0.0)
    residual_loss = hidden.new_tensor(0.0)
    backoff_stats: dict[int, BackoffStats] = {}
    for offset in range(0, active.numel(), chunk_size):
        node_ids = active[offset : offset + chunk_size]
        count_rows = prior_counts[node_ids].to(device=hidden.device, dtype=hidden.dtype)
        count_probs = (count_rows + count_prior_alpha * unigram_probs.unsqueeze(0)) / (
            count_rows.sum(dim=1, keepdim=True) + count_prior_alpha
        ).clamp_min(1e-12)
        prior_logits = model.head(correction.corrected_hidden[node_ids])
        prior_log_probs = F.log_softmax(prior_logits, dim=1)
        residual_logits = prior_logits + residual_scale * residual_head(hidden[node_ids])
        residual_log_probs = F.log_softmax(residual_logits, dim=1)
        count_prior_log_probs = count_probs.clamp_min(1e-12).log()
        count_prior_loss = count_prior_loss - (val_counts[node_ids] * count_prior_log_probs).sum()
        prior_loss = prior_loss - (val_counts[node_ids] * prior_log_probs).sum()
        residual_loss = residual_loss - (val_counts[node_ids] * residual_log_probs).sum()
        node_count_prior_losses = -(val_counts[node_ids] * count_prior_log_probs).sum(dim=1)
        node_prior_losses = -(val_counts[node_ids] * prior_log_probs).sum(dim=1)
        node_residual_losses = -(val_counts[node_ids] * residual_log_probs).sum(dim=1)
        for local_index, node_id in enumerate(node_ids.tolist()):
            dropped = int(dropped_by_node[node_id].item())
            stats = backoff_stats.setdefault(dropped, BackoffStats())
            node_tokens = int(totals[node_id].item())
            stats.tokens += node_tokens
            stats.nodes += 1
            stats.count_prior_loss += float(node_count_prior_losses[local_index].item())
            stats.prior_loss += float(node_prior_losses[local_index].item())
            stats.residual_loss += float(node_residual_losses[local_index].item())
    return (
        float(count_prior_loss.item()),
        float(prior_loss.item()),
        float(residual_loss.item()),
        int(val_counts.sum().item()),
        prior_mass,
        backoff_stats,
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_fraction = float(checkpoint_arg(checkpoint, args, "train_fraction", 0.9))
    block_size = int(checkpoint_arg(checkpoint, args, "block_size", 16))
    stride = int(checkpoint_arg(checkpoint, args, "stride", 1))
    prefix_length = int(checkpoint_arg(checkpoint, args, "prefix_length", 1))
    hidden_size = int(checkpoint["args"]["hidden_size"])
    embedding_size = int(checkpoint["args"]["embedding_size"])
    damping = float(checkpoint_arg(checkpoint, args, "damping", 10.0))
    step_scale_value = checkpoint_arg(checkpoint, args, "step_scale", "auto-node")
    step_scale = parse_step_scale(str(step_scale_value))
    max_auto_eta = float(checkpoint_arg(checkpoint, args, "max_auto_eta", 32.0))
    auto_newton_steps = int(checkpoint_arg(checkpoint, args, "auto_newton_steps", 8))
    state_iterations = int(checkpoint_arg(checkpoint, args, "state_iterations", 5))
    curvature = str(checkpoint_arg(checkpoint, args, "curvature", "empirical"))
    chunk_size = int(checkpoint_arg(checkpoint, args, "chunk_size", 512))
    residual_scale = float(checkpoint_arg(checkpoint, args, "residual_scale", 1.0))

    text = read_text(args.input)
    split = split_text(text, train_fraction=train_fraction)
    vocab = CharVocab.from_text(text)
    if tuple(checkpoint["vocab_chars"]) != vocab.chars:
        raise ValueError("checkpoint vocab does not match input text vocab")
    model = TinyCharRNN(vocab.size, n_embd=embedding_size, n_hidden=hidden_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    residual_head = nn.Linear(hidden_size, vocab.size)
    residual_head.load_state_dict(checkpoint["residual_head_state_dict"])
    residual_head.eval()

    train_samples = make_circular_samples(split.train_text, block_size=block_size, stride=stride)
    eval_text = split.train_text if args.eval_split == "train" else split.val_text
    val_make = make_circular_samples if args.val_circular else make_samples
    val_samples = val_make(eval_text, block_size=block_size, stride=stride)
    if args.max_val_samples is not None:
        val_samples = val_samples[: args.max_val_samples]
    train_encoded = [vocab.encode(sample) for sample in train_samples]
    val_encoded = [vocab.encode(sample) for sample in val_samples]

    started = time.perf_counter()
    needed_contexts = collect_needed_contexts(val_encoded)
    train_counts = build_train_context_counts(train_encoded, needed_contexts, vocab.size)
    count_sec = time.perf_counter() - started
    val_samples.sort()
    ranges = prefix_ranges(val_samples, prefix_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"checkpoint={args.checkpoint} val_samples={len(val_samples)} prefixes={len(ranges)} "
        f"eval_split={args.eval_split} block_size={block_size} state_iterations={state_iterations} "
        f"needed_contexts={len(needed_contexts)} train_count_rows={len(train_counts)} "
        f"context_count_sec={count_sec:.2f} output={args.output}",
        flush=True,
    )

    total_count_prior_loss = 0.0
    total_prior_loss = 0.0
    total_residual_loss = 0.0
    total_tokens = 0
    total_prior_mass = 0
    total_backoff_stats: dict[int, BackoffStats] = {}
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(EvalRow("", 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        for index, (prefix, lo, hi) in enumerate(ranges, start=1):
            row_started = time.perf_counter()
            suffix_samples = [sample[prefix_length:] for sample in val_samples[lo:hi]]
            trie = samples_to_flat_trie(suffix_samples, vocab)
            count_prior_loss, prior_loss, residual_loss_value, tokens, prior_mass, backoff_stats = score_subtree(
                model,
                residual_head,
                vocab,
                prefix,
                trie,
                train_counts,
                damping=damping,
                step_scale=step_scale,
                max_auto_eta=max_auto_eta,
                auto_newton_steps=auto_newton_steps,
                state_iterations=state_iterations,
                curvature=curvature,
        chunk_size=chunk_size,
        residual_scale=residual_scale,
        count_prior_alpha=args.count_prior_alpha,
        prior_hidden_context=args.prior_hidden_context,
    )
            row = EvalRow(
                prefix=prefix,
                sample_count=hi - lo,
                node_count=trie.node_count,
                transition_mass=tokens,
                train_prior_mass=prior_mass,
                val_count_prior_ppl=math.exp(count_prior_loss / tokens),
                val_prior_ppl=math.exp(prior_loss / tokens),
                val_residual_ppl=math.exp(residual_loss_value / tokens),
                val_loss=residual_loss_value,
                runtime_sec=time.perf_counter() - row_started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(row))
            handle.flush()
            total_count_prior_loss += count_prior_loss
            total_prior_loss += prior_loss
            total_residual_loss += residual_loss_value
            total_tokens += tokens
            total_prior_mass += prior_mass
            for dropped, stats in backoff_stats.items():
                merged = total_backoff_stats.setdefault(dropped, BackoffStats())
                merged.tokens += stats.tokens
                merged.nodes += stats.nodes
                merged.count_prior_loss += stats.count_prior_loss
                merged.prior_loss += stats.prior_loss
                merged.residual_loss += stats.residual_loss
            if args.progress:
                print(
                    f"prefix {index}/{len(ranges)} {prefix!r} nodes={trie.node_count} "
                    f"val_ppl=count {row.val_count_prior_ppl:.3f}, fisher {row.val_prior_ppl:.3f}->{row.val_residual_ppl:.3f} "
                    f"sec={row.runtime_sec:.2f}",
                    flush=True,
                )
        summary = EvalRow(
            prefix="",
            sample_count=len(val_samples),
            node_count=0,
            transition_mass=total_tokens,
            train_prior_mass=total_prior_mass,
            val_count_prior_ppl=math.exp(total_count_prior_loss / total_tokens),
            val_prior_ppl=math.exp(total_prior_loss / total_tokens),
            val_residual_ppl=math.exp(total_residual_loss / total_tokens),
            val_loss=total_residual_loss,
            runtime_sec=0.0,
            peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        writer.writerow(asdict(summary))
    print(
        f"summary tokens={total_tokens} val_ppl=count {summary.val_count_prior_ppl:.4f}, "
        f"fisher {summary.val_prior_ppl:.4f}->{summary.val_residual_ppl:.4f} "
        f"peak_rss_mb={summary.peak_rss_kb / 1024:.1f}",
        flush=True,
    )
    backoff_path = args.output.with_name(f"{args.output.stem}_backoff.csv")
    with backoff_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dropped",
            "nodes",
            "tokens",
            "token_fraction",
            "count_prior_ppl",
            "prior_ppl",
            "residual_ppl",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dropped, stats in sorted(total_backoff_stats.items()):
            writer.writerow(
                {
                    "dropped": dropped,
                    "nodes": stats.nodes,
                    "tokens": stats.tokens,
                    "token_fraction": stats.tokens / total_tokens if total_tokens else 0.0,
                    "count_prior_ppl": math.exp(stats.count_prior_loss / stats.tokens)
                    if stats.tokens
                    else float("nan"),
                    "prior_ppl": math.exp(stats.prior_loss / stats.tokens) if stats.tokens else float("nan"),
                    "residual_ppl": math.exp(stats.residual_loss / stats.tokens) if stats.tokens else float("nan"),
                }
            )
    print(f"backoff={backoff_path}", flush=True)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
