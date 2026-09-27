from __future__ import annotations

import argparse
import csv
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, make_circular_samples, make_samples, read_text
from agpt_ultra.eval import split_text
from agpt_ultra.flat_ops import compute_flat_hidden_states, samples_to_flat_trie
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.hybrid import prefix_hidden_state
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id


@dataclass(frozen=True)
class DirectPriorResidualRow:
    epoch: int
    split: str
    prefix: str
    sample_count: int
    node_count: int
    tokens: int
    prior_ppl: float
    residual_ppl: float
    loss: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRU residual on top of explicit log-count tree priors.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/direct_prior_residual.csv"))
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--forced-drop-min", type=int, default=1)
    parser.add_argument("--forced-drop-max", type=int, default=8)
    parser.add_argument("--eval-drop-min", type=int, default=0)
    parser.add_argument("--eval-drop-max", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--limit-prefixes", type=str, default=None)
    parser.add_argument("--max-subtrees", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=2000)
    parser.add_argument("--head-init", choices=["zero", "checkpoint_model", "checkpoint_residual"], default="zero")
    parser.add_argument("--train-body", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def checkpoint_arg(checkpoint: dict | None, name: str, default):
    if checkpoint is None:
        return default
    value = checkpoint.get("args", {}).get(name, default)
    return default if value is None else value


def prefix_ranges(sorted_samples: list[str], prefix_length: int) -> list[tuple[str, int, int]]:
    if not sorted_samples:
        return []
    ranges: list[tuple[str, int, int]] = []
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
    drop_min: int,
    drop_max: int,
) -> set[tuple[int, ...]]:
    needed: set[tuple[int, ...]] = {()}
    for ids in encoded_samples:
        for end in range(prefix_length, len(ids)):
            full_context = tuple(ids[:end])
            for requested_drop in range(drop_min, drop_max + 1):
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
        max_len = max(max_len, len(context))
        needed_by_len.setdefault(len(context), []).append((high, low, context))

    padded = np.asarray(encoded_text + encoded_text[:block_size], dtype=np.uint64)
    starts = np.arange(0, len(encoded_text), stride, dtype=np.int64)
    empty_counts = np.zeros(vocab_size, dtype=np.float64)
    for context_len in range(1, block_size):
        targets = padded[starts + context_len].astype(np.int64, copy=False)
        empty_counts += np.bincount(targets, minlength=vocab_size)
    counts[()] = torch.tensor(empty_counts, dtype=torch.float32)

    low = np.zeros(starts.shape[0], dtype=np.uint64)
    high = np.zeros(starts.shape[0], dtype=np.uint64)
    pair_dtype = np.dtype([("high", np.uint64), ("low", np.uint64)])
    for context_len in range(1, min(max_len, block_size - 1) + 1):
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
        for left, right, (_, _, context) in zip(lefts, rights, needed, strict=True):
            counts[context] = torch.tensor(np.bincount(sorted_targets[left:right], minlength=vocab_size), dtype=torch.float32)
    return counts


def lookup_counts(context: tuple[int, ...], counts: dict[tuple[int, ...], torch.Tensor]) -> torch.Tensor:
    cursor = context
    while cursor:
        row = counts.get(cursor)
        if row is not None and float(row.sum().item()) > 0.0:
            return row
        cursor = cursor[1:]
    return counts[()]


def prior_counts_for_trie(
    prefix_ids: tuple[int, ...],
    trie: FlatTrie,
    train_counts: dict[tuple[int, ...], torch.Tensor],
    drop_min: int,
    drop_max: int,
    epoch: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    paths = trie_node_paths(trie)
    span = drop_max - drop_min + 1
    for node_id, path in enumerate(paths):
        full_context = prefix_ids + path
        requested_drop = drop_min + ((node_id + epoch - 1) % span)
        drop = min(requested_drop, len(full_context))
        rows.append(lookup_counts(full_context[drop:], train_counts))
    return torch.stack(rows, dim=0)


def log_prior_from_counts(prior_counts: torch.Tensor, unigram_probs: torch.Tensor, alpha: float, device, dtype) -> torch.Tensor:
    counts = prior_counts.to(device=device, dtype=dtype)
    unigram = unigram_probs.to(device=device, dtype=dtype)
    probs = (counts + alpha * unigram.unsqueeze(0)) / (counts.sum(dim=1, keepdim=True) + alpha).clamp_min(1e-12)
    return probs.clamp_min(1e-12).log()


def losses_for_subtree(
    model: TinyCharRNN,
    vocab: CharVocab,
    prefix: str,
    trie: FlatTrie,
    prior_counts: torch.Tensor,
    unigram_probs: torch.Tensor,
    alpha: float,
    residual_scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    hidden = compute_flat_hidden_states(model, trie, initial_hidden=prefix_hidden_state(model, vocab, prefix))
    device = hidden.device
    dtype = hidden.dtype
    counts = trie.transition_counts.to(device=device, dtype=dtype)
    totals = counts.sum(dim=1)
    active = torch.nonzero(totals > 0, as_tuple=False).flatten()
    log_prior = log_prior_from_counts(prior_counts, unigram_probs, alpha, device, dtype)
    prior_loss = hidden.new_tensor(0.0)
    residual_loss = hidden.new_tensor(0.0)
    for offset in range(0, active.numel(), chunk_size):
        node_ids = active[offset : offset + chunk_size]
        node_counts = counts[node_ids]
        prior_lp = log_prior[node_ids]
        logits = prior_lp + residual_scale * model.head(hidden[node_ids])
        prior_loss = prior_loss - (node_counts * prior_lp).sum()
        residual_loss = residual_loss - (node_counts * F.log_softmax(logits, dim=1)).sum()
    return prior_loss, residual_loss, int(counts.sum().item())


def maybe_filter_ranges(ranges: list[tuple[str, int, int]], limit_prefixes: str | None, max_subtrees: int | None) -> list[tuple[str, int, int]]:
    if limit_prefixes is not None:
        allowed = set(limit_prefixes)
        ranges = [item for item in ranges if item[0] in allowed]
    if max_subtrees is not None:
        ranges = ranges[:max_subtrees]
    return ranges


def main() -> None:
    args = parse_args()
    if args.forced_drop_min < 0 or args.forced_drop_max < args.forced_drop_min:
        raise ValueError("invalid forced drop range")
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    if args.checkpoint_output is None:
        args.checkpoint_output = args.output.with_suffix(".pt")
    else:
        args.checkpoint_output = prefixed_path(args.checkpoint_output, args.run_id)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    hidden_size = int(checkpoint_arg(checkpoint, "hidden_size", args.hidden_size))
    embedding_size = int(checkpoint_arg(checkpoint, "embedding_size", args.embedding_size))
    text = read_text(args.input)
    split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    if checkpoint is not None and tuple(checkpoint["vocab_chars"]) != vocab.chars:
        raise ValueError("checkpoint vocab does not match input text vocab")

    train_samples = make_circular_samples(split.train_text, block_size=args.block_size, stride=args.stride)
    val_samples = make_samples(split.val_text, block_size=args.block_size, stride=args.stride)
    if args.max_val_samples is not None:
        val_samples = val_samples[: args.max_val_samples]
    train_samples.sort()
    val_samples.sort()
    train_ranges = maybe_filter_ranges(prefix_ranges(train_samples, args.prefix_length), args.limit_prefixes, args.max_subtrees)
    val_ranges = prefix_ranges(val_samples, args.prefix_length)

    selected_train = [sample for _, lo, hi in train_ranges for sample in train_samples[lo:hi]]
    needed = collect_needed_contexts(
        [vocab.encode(sample) for sample in selected_train],
        args.prefix_length,
        args.forced_drop_min,
        args.forced_drop_max,
    )
    needed.update(
        collect_needed_contexts(
            [vocab.encode(sample) for sample in val_samples],
            args.prefix_length,
            args.eval_drop_min,
            args.eval_drop_max,
        )
    )
    count_started = time.perf_counter()
    train_counts = build_train_context_counts_from_corpus(
        vocab.encode(split.train_text),
        block_size=args.block_size,
        stride=args.stride,
        needed_contexts=needed,
        vocab_size=vocab.size,
    )
    count_sec = time.perf_counter() - count_started
    unigram_probs = train_counts[()] / train_counts[()].sum().clamp_min(1.0)

    model = TinyCharRNN(vocab.size, n_embd=embedding_size, n_hidden=hidden_size)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.head_init == "checkpoint_residual":
            model.head.load_state_dict(checkpoint["residual_head_state_dict"])
        elif args.head_init == "zero":
            torch.nn.init.zeros_(model.head.weight)
            torch.nn.init.zeros_(model.head.bias)
    elif args.head_init == "zero":
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    if not args.train_body:
        for param in model.embed.parameters():
            param.requires_grad_(False)
        for param in model.cell.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"train_prefixes={len(train_ranges)} val_prefixes={len(val_ranges)} train_samples={len(selected_train)} "
        f"val_samples={len(val_samples)} needed_contexts={len(needed)} train_count_rows={len(train_counts)} "
        f"context_count_sec={count_sec:.2f} head_init={args.head_init} train_body={args.train_body} "
        f"forced_drop={args.forced_drop_min}..{args.forced_drop_max} output={args.output}",
        flush=True,
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(DirectPriorResidualRow(0, "", "", 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_prior = 0.0
            total_residual = 0.0
            total_tokens = 0
            epoch_started = time.perf_counter()
            for prefix, lo, hi in train_ranges:
                started = time.perf_counter()
                suffix_samples = [sample[args.prefix_length :] for sample in train_samples[lo:hi]]
                trie = samples_to_flat_trie(suffix_samples, vocab)
                prior_counts = prior_counts_for_trie(
                    tuple(vocab.encode(prefix)),
                    trie,
                    train_counts,
                    args.forced_drop_min,
                    args.forced_drop_max,
                    epoch,
                )
                prior_loss, residual_loss, tokens = losses_for_subtree(
                    model,
                    vocab,
                    prefix,
                    trie,
                    prior_counts,
                    unigram_probs,
                    args.alpha,
                    args.residual_scale,
                    args.chunk_size,
                )
                optimizer.zero_grad(set_to_none=True)
                residual_loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                row = DirectPriorResidualRow(
                    epoch=epoch,
                    split="train",
                    prefix=prefix,
                    sample_count=hi - lo,
                    node_count=trie.node_count,
                    tokens=tokens,
                    prior_ppl=math.exp(float(prior_loss.item()) / tokens),
                    residual_ppl=math.exp(float(residual_loss.item()) / tokens),
                    loss=float(residual_loss.item()),
                    runtime_sec=time.perf_counter() - started,
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                writer.writerow(asdict(row))
                handle.flush()
                total_prior += float(prior_loss.item())
                total_residual += float(residual_loss.item())
                total_tokens += tokens
                if args.progress:
                    print(
                        f"epoch={epoch:03d} train prefix={prefix!r} prior={row.prior_ppl:.3f} "
                        f"residual={row.residual_ppl:.3f} nodes={trie.node_count} sec={row.runtime_sec:.2f}",
                        flush=True,
                    )
            print(
                f"epoch={epoch:03d} train_summary prior={math.exp(total_prior / total_tokens):.3f} "
                f"residual={math.exp(total_residual / total_tokens):.3f} sec={time.perf_counter() - epoch_started:.2f}",
                flush=True,
            )

        model.eval()
        total_prior = 0.0
        total_residual = 0.0
        total_tokens = 0
        with torch.no_grad():
            for prefix, lo, hi in val_ranges:
                started = time.perf_counter()
                suffix_samples = [sample[args.prefix_length :] for sample in val_samples[lo:hi]]
                trie = samples_to_flat_trie(suffix_samples, vocab)
                prior_counts = prior_counts_for_trie(
                    tuple(vocab.encode(prefix)),
                    trie,
                    train_counts,
                    args.eval_drop_min,
                    args.eval_drop_max,
                    1,
                )
                prior_loss, residual_loss, tokens = losses_for_subtree(
                    model,
                    vocab,
                    prefix,
                    trie,
                    prior_counts,
                    unigram_probs,
                    args.alpha,
                    args.residual_scale,
                    args.chunk_size,
                )
                row = DirectPriorResidualRow(
                    epoch=args.epochs,
                    split="val",
                    prefix=prefix,
                    sample_count=hi - lo,
                    node_count=trie.node_count,
                    tokens=tokens,
                    prior_ppl=math.exp(float(prior_loss.item()) / tokens),
                    residual_ppl=math.exp(float(residual_loss.item()) / tokens),
                    loss=float(residual_loss.item()),
                    runtime_sec=time.perf_counter() - started,
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                writer.writerow(asdict(row))
                total_prior += float(prior_loss.item())
                total_residual += float(residual_loss.item())
                total_tokens += tokens
        summary = DirectPriorResidualRow(
            epoch=args.epochs,
            split="val",
            prefix="",
            sample_count=len(val_samples),
            node_count=0,
            tokens=total_tokens,
            prior_ppl=math.exp(total_prior / total_tokens),
            residual_ppl=math.exp(total_residual / total_tokens),
            loss=total_residual,
            runtime_sec=0.0,
            peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        writer.writerow(asdict(summary))
        print(f"val_summary prior={summary.prior_ppl:.3f} residual={summary.residual_ppl:.3f}", flush=True)

    torch.save(
        {
            "epoch": args.epochs,
            "args": vars(args),
            "vocab_chars": vocab.chars,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args.checkpoint_output,
    )
    print(f"checkpoint={args.checkpoint_output}", flush=True)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
