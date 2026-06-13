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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, make_circular_samples, read_text
from agpt_ultra.eval import split_text
from agpt_ultra.flat_ops import compute_flat_hidden_states, samples_to_flat_trie
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.hybrid import prefix_hidden_state
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id
from agpt_ultra.state_fisher import parse_step_scale, state_fisher_correct_hidden


@dataclass(frozen=True)
class FidelityRow:
    epoch: int
    prefix: str
    sample_count: int
    node_count: int
    transition_mass: int
    teacher_ppl_before: float
    teacher_ppl_after: float
    train_loss: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fisher prior to imitate backed-off count rows.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/prior_fidelity.csv"))
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--prefix-length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--forced-drop-min", type=int, default=0)
    parser.add_argument("--forced-drop-max", type=int, default=8)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--step-scale", default=None)
    parser.add_argument("--max-auto-eta", type=float, default=None)
    parser.add_argument("--auto-newton-steps", type=int, default=None)
    parser.add_argument("--state-iterations", type=int, default=None)
    parser.add_argument("--curvature", choices=["model", "empirical"], default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--limit-prefixes", type=str, default=None)
    parser.add_argument("--max-subtrees", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def checkpoint_arg(checkpoint: dict, args: argparse.Namespace, name: str, default):
    value = getattr(args, name)
    if value is not None:
        return value
    return checkpoint.get("args", {}).get(name, default)


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


def collect_needed_contexts(
    encoded_samples: list[list[int]],
    prefix_length: int,
    forced_drop_min: int,
    forced_drop_max: int,
) -> set[tuple[int, ...]]:
    needed: set[tuple[int, ...]] = {()}
    for ids in encoded_samples:
        for end in range(prefix_length, len(ids)):
            full_context = tuple(ids[:end])
            for requested_drop in range(forced_drop_min, forced_drop_max + 1):
                drop = min(requested_drop, len(full_context))
                context = full_context[drop:]
                while context:
                    needed.add(context)
                    context = context[1:]
    return needed


def build_train_context_counts_from_corpus(
    encoded_text: list[int],
    block_size: int,
    stride: int,
    needed_contexts: set[tuple[int, ...]],
    vocab_size: int,
) -> dict[tuple[int, ...], torch.Tensor]:
    counts: dict[tuple[int, ...], torch.Tensor] = {(): torch.zeros(vocab_size, dtype=torch.float32)}
    if not needed_contexts:
        return counts
    bits_per_token = 7
    split_tokens = 9
    needed_by_len: dict[int, list[tuple[int, int, tuple[int, ...]]]] = {}
    max_len = 0
    for context in needed_contexts:
        if not context:
            continue
        low = 0
        high = 0
        for index, token_id in enumerate(context):
            value = token_id + 1
            if index < split_tokens:
                low |= value << (bits_per_token * index)
            else:
                high |= value << (bits_per_token * (index - split_tokens))
        context_len = len(context)
        max_len = max(max_len, context_len)
        needed_by_len.setdefault(context_len, []).append((high, low, context))

    padded = np.asarray(encoded_text + encoded_text[:block_size], dtype=np.uint64)
    starts = np.arange(0, len(encoded_text), stride, dtype=np.int64)
    max_context_len = min(max_len, block_size - 1)
    empty_counts = np.zeros(vocab_size, dtype=np.float64)
    for context_len in range(1, block_size):
        targets = padded[starts + context_len].astype(np.int64, copy=False)
        empty_counts += np.bincount(targets, minlength=vocab_size)
    counts[()] = torch.tensor(empty_counts, dtype=torch.float32)

    low = np.zeros(starts.shape[0], dtype=np.uint64)
    high = np.zeros(starts.shape[0], dtype=np.uint64)
    pair_dtype = np.dtype([("high", np.uint64), ("low", np.uint64)])
    for context_len in range(1, max_context_len + 1):
        values = padded[starts + context_len - 1] + np.uint64(1)
        if context_len <= split_tokens:
            low |= values << np.uint64(bits_per_token * (context_len - 1))
        else:
            high |= values << np.uint64(bits_per_token * (context_len - split_tokens - 1))
        needed = needed_by_len.get(context_len)
        if not needed:
            continue

        pairs = np.empty(starts.shape[0], dtype=pair_dtype)
        pairs["high"] = high
        pairs["low"] = low
        order = np.argsort(pairs, order=("high", "low"))
        sorted_pairs = pairs[order]
        sorted_targets = padded[starts[order] + context_len].astype(np.int64, copy=False)
        needed_pairs = np.empty(len(needed), dtype=pair_dtype)
        for index, (needed_high, needed_low, _) in enumerate(needed):
            needed_pairs[index] = (needed_high, needed_low)
        lefts = np.searchsorted(sorted_pairs, needed_pairs, side="left", sorter=None)
        rights = np.searchsorted(sorted_pairs, needed_pairs, side="right", sorter=None)
        for (left, right, (_, _, context)) in zip(lefts, rights, needed, strict=True):
            row_counts = np.bincount(sorted_targets[left:right], minlength=vocab_size)
            counts[context] = torch.tensor(row_counts, dtype=torch.float32)
    return counts


def lookup_counts(context: tuple[int, ...], counts: dict[tuple[int, ...], torch.Tensor]) -> torch.Tensor:
    cursor = context
    while cursor:
        row = counts.get(cursor)
        if row is not None and float(row.sum().item()) > 0.0:
            return row
        cursor = cursor[1:]
    return counts[()]


def teacher_counts_for_trie(
    prefix_ids: tuple[int, ...],
    trie: FlatTrie,
    train_counts: dict[tuple[int, ...], torch.Tensor],
    forced_drop_min: int,
    forced_drop_max: int,
    epoch: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    paths = trie_node_paths(trie)
    node_totals = trie.transition_counts.sum(dim=1)
    span = forced_drop_max - forced_drop_min + 1
    for node_id, path in enumerate(paths):
        full_context = prefix_ids + path
        if span > 1:
            requested_drop = forced_drop_min + ((node_id + epoch - 1) % span)
        else:
            requested_drop = forced_drop_min
        drop = min(requested_drop, len(full_context))
        row = lookup_counts(full_context[drop:], train_counts)
        total = float(node_totals[node_id].item())
        row_total = float(row.sum().item())
        if total <= 0.0 or row_total <= 0.0:
            rows.append(torch.zeros_like(row))
        else:
            rows.append(row * (total / row_total))
    return torch.stack(rows, dim=0)


def teacher_loss(
    model: TinyCharRNN,
    corrected_hidden: torch.Tensor,
    teacher_counts: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    device = corrected_hidden.device
    dtype = corrected_hidden.dtype
    teacher_counts = teacher_counts.to(device=device, dtype=dtype)
    totals = teacher_counts.sum(dim=1)
    active = torch.nonzero(totals > 0, as_tuple=False).flatten()
    loss = corrected_hidden.new_tensor(0.0)
    for offset in range(0, active.numel(), chunk_size):
        node_ids = active[offset : offset + chunk_size]
        log_probs = F.log_softmax(model.head(corrected_hidden[node_ids]), dim=1)
        loss = loss - (teacher_counts[node_ids] * log_probs).sum()
    return loss


def main() -> None:
    args = parse_args()
    if args.forced_drop_min < 0 or args.forced_drop_max < args.forced_drop_min:
        raise ValueError("forced drop range is invalid")
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    if args.checkpoint_output is None:
        args.checkpoint_output = args.output.with_suffix(".pt")
    else:
        args.checkpoint_output = prefixed_path(args.checkpoint_output, args.run_id)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_fraction = float(checkpoint_arg(checkpoint, args, "train_fraction", 0.9))
    block_size = int(checkpoint_arg(checkpoint, args, "block_size", 16))
    stride = int(checkpoint_arg(checkpoint, args, "stride", 1))
    prefix_length = int(checkpoint_arg(checkpoint, args, "prefix_length", 1))
    hidden_size = int(checkpoint["args"]["hidden_size"])
    embedding_size = int(checkpoint["args"]["embedding_size"])
    damping = float(checkpoint_arg(checkpoint, args, "damping", 10.0))
    step_scale = parse_step_scale(str(checkpoint_arg(checkpoint, args, "step_scale", "auto-node")))
    max_auto_eta = float(checkpoint_arg(checkpoint, args, "max_auto_eta", 32.0))
    auto_newton_steps = int(checkpoint_arg(checkpoint, args, "auto_newton_steps", 8))
    state_iterations = int(checkpoint_arg(checkpoint, args, "state_iterations", 5))
    curvature = str(checkpoint_arg(checkpoint, args, "curvature", "empirical"))
    chunk_size = int(checkpoint_arg(checkpoint, args, "chunk_size", 512))

    text = read_text(args.input)
    split = split_text(text, train_fraction=train_fraction)
    vocab = CharVocab.from_text(text)
    if tuple(checkpoint["vocab_chars"]) != vocab.chars:
        raise ValueError("checkpoint vocab does not match input text vocab")
    samples = make_circular_samples(split.train_text, block_size=block_size, stride=stride)
    samples.sort()
    ranges = prefix_ranges(samples, prefix_length)
    if args.limit_prefixes is not None:
        if prefix_length != 1:
            raise ValueError("--limit-prefixes currently supports prefix length 1 only")
        allowed = set(args.limit_prefixes)
        ranges = [item for item in ranges if item[0] in allowed]
    if args.max_subtrees is not None:
        ranges = ranges[: args.max_subtrees]
    selected_samples: list[str] = []
    for _, lo, hi in ranges:
        selected_samples.extend(samples[lo:hi])
    selected_encoded = [vocab.encode(sample) for sample in selected_samples]
    train_encoded_text = vocab.encode(split.train_text)
    needed_contexts = collect_needed_contexts(
        selected_encoded,
        prefix_length=prefix_length,
        forced_drop_min=args.forced_drop_min,
        forced_drop_max=args.forced_drop_max,
    )
    count_started = time.perf_counter()
    train_counts = build_train_context_counts_from_corpus(
        train_encoded_text,
        block_size=block_size,
        stride=stride,
        needed_contexts=needed_contexts,
        vocab_size=vocab.size,
    )
    count_sec = time.perf_counter() - count_started

    model = TinyCharRNN(vocab.size, n_embd=embedding_size, n_hidden=hidden_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    residual_head = nn.Linear(hidden_size, vocab.size)
    residual_head.load_state_dict(checkpoint["residual_head_state_dict"])
    residual_head.eval()
    for param in residual_head.parameters():
        param.requires_grad_(False)
    for param in model.embed.parameters():
        param.requires_grad_(False)
    for param in model.cell.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(FidelityRow(0, "", 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        print(
            f"samples={len(samples)} prefixes={len(ranges)} epochs={args.epochs} block_size={block_size} "
            f"drop={args.forced_drop_min}..{args.forced_drop_max} lr={args.lr} "
            f"state_iterations={state_iterations} needed_contexts={len(needed_contexts)} "
            f"train_count_rows={len(train_counts)} context_count_sec={count_sec:.2f} "
            f"run_id={args.run_id} output={args.output}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            total_before = 0.0
            total_after = 0.0
            total_loss = 0.0
            total_mass = 0
            for prefix, lo, hi in ranges:
                started = time.perf_counter()
                suffix_samples = [sample[prefix_length:] for sample in samples[lo:hi]]
                trie = samples_to_flat_trie(suffix_samples, vocab)
                teacher_counts = teacher_counts_for_trie(
                    tuple(vocab.encode(prefix)),
                    trie,
                    train_counts,
                    forced_drop_min=args.forced_drop_min,
                    forced_drop_max=args.forced_drop_max,
                    epoch=epoch,
                )
                mass = int(teacher_counts.sum().item())
                with torch.no_grad():
                    hidden = compute_flat_hidden_states(model, trie, initial_hidden=prefix_hidden_state(model, vocab, prefix))
                    correction = state_fisher_correct_hidden(
                        model,
                        hidden,
                        teacher_counts,
                        damping=damping,
                        step_scale=step_scale,
                        max_auto_eta=max_auto_eta,
                        auto_newton_steps=auto_newton_steps,
                        state_iterations=state_iterations,
                        curvature=curvature,
                        chunk_size=chunk_size,
                    )
                    corrected_hidden = correction.corrected_hidden.detach()
                loss = teacher_loss(model, corrected_hidden, teacher_counts, chunk_size=chunk_size)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.max_grad_norm)
                optimizer.step()
                loss_value = float(loss.item())
                row = FidelityRow(
                    epoch=epoch,
                    prefix=prefix,
                    sample_count=hi - lo,
                    node_count=trie.node_count,
                    transition_mass=mass,
                    teacher_ppl_before=math.exp(correction.before_loss / mass) if mass else float("nan"),
                    teacher_ppl_after=math.exp(correction.after_loss / mass) if mass else float("nan"),
                    train_loss=loss_value,
                    runtime_sec=time.perf_counter() - started,
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                writer.writerow(asdict(row))
                handle.flush()
                if args.progress:
                    print(
                        f"epoch={epoch:03d} prefix={prefix!r} samples={hi - lo} nodes={trie.node_count} "
                        f"teacher_ppl={row.teacher_ppl_before:.3f}->{row.teacher_ppl_after:.3f} "
                        f"sec={row.runtime_sec:.2f}",
                        flush=True,
                    )
                total_before += correction.before_loss
                total_after += correction.after_loss
                total_loss += loss_value
                total_mass += mass
            summary = FidelityRow(
                epoch=epoch,
                prefix="",
                sample_count=len(samples),
                node_count=0,
                transition_mass=total_mass,
                teacher_ppl_before=math.exp(total_before / total_mass),
                teacher_ppl_after=math.exp(total_after / total_mass),
                train_loss=total_loss,
                runtime_sec=time.perf_counter() - epoch_started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(summary))
            handle.flush()
            merged_args = dict(checkpoint["args"])
            merged_args.update({key: value for key, value in vars(args).items() if value is not None})
            checkpoint_out = {
                "epoch": epoch,
                "args": {**merged_args, "source_checkpoint": str(args.checkpoint)},
                "vocab_chars": vocab.chars,
                "model_state_dict": model.state_dict(),
                "residual_head_state_dict": residual_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint_out, args.checkpoint_output)
            print(
                f"epoch={epoch:03d} teacher_ppl={summary.teacher_ppl_before:.3f}->{summary.teacher_ppl_after:.3f} "
                f"trained_head_ppl={math.exp(total_loss / total_mass):.3f} runtime_sec={summary.runtime_sec:.2f} "
                f"peak_rss_mb={summary.peak_rss_kb / 1024:.1f} checkpoint={args.checkpoint_output}",
                flush=True,
            )
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
