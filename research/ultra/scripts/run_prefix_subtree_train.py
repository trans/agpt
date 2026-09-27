from __future__ import annotations

import argparse
import csv
import gc
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, make_circular_samples, make_samples, read_text
from agpt_ultra.embedding_fisher import EmbeddingFisherPreconditioner
from agpt_ultra.eval import evaluate_sequential_text_loss, split_text
from agpt_ultra.flat_ops import samples_to_flat_trie
from agpt_ultra.hybrid import freeze_embeddings, hybrid_epoch
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id


@dataclass(frozen=True)
class PrefixSubtreeRow:
    seed: int
    epoch: int
    step: int
    prefix: str
    sample_count: int
    node_count: int
    transition_mass: int
    train_loss: float | None
    after_body_loss: float | None
    seq_val_nll: float | None
    seq_val_ppl: float | None
    seq_val_bpc: float | None
    body_lr: float
    build_sec: float | None
    update_sec: float | None
    eval_sec: float | None
    body_sec: float | None
    cache_sec: float | None
    head_sec: float | None
    step_scale: float | None
    eta_quad: float | None
    eta_trust: float | None
    body_fisher_cg_iters: int | None
    body_fisher_residual_norm: float | None
    body_fisher_grad_norm: float | None
    body_fisher_step_norm: float | None
    body_fisher_predicted_improvement: float | None
    body_fisher_linear_term: float | None
    body_fisher_quadratic: float | None
    body_fisher_step_scale: float | None
    body_fisher_eta_quad: float | None
    body_fisher_eta_trust: float | None
    embedding_fisher_active_tokens: int | None
    embedding_fisher_max_update_norm: float | None
    embedding_fisher_mean_update_norm: float | None
    peak_rss_kb: int


def parse_step_scale(value: str) -> float | str:
    if value == "auto":
        return value
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train by sequential prefix subtrees over circular full-corpus windows.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/prefix_subtree_train.csv"))
    parser.add_argument("--run-id", type=str, default=None, help="Prefix output artifacts for a new run. Generated unless --resume is used.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--update-embeddings", action="store_true")
    parser.add_argument("--embedding-fisher", action="store_true", help="Update embeddings with per-token empirical Fisher blocks instead of AdamW.")
    parser.add_argument("--embedding-fisher-damping", type=float, default=10.0)
    parser.add_argument("--embedding-fisher-step-scale", type=float, default=1.0)
    parser.add_argument("--embedding-fisher-decay", type=float, default=1.0)
    parser.add_argument("--body-lr", type=float, default=3e-3)
    parser.add_argument("--body-optimizer", choices=["adam", "fisher"], default="adam")
    parser.add_argument("--body-fisher-damping", type=float, default=100.0)
    parser.add_argument("--body-fisher-step-scale", default=1.0)
    parser.add_argument("--body-fisher-max-step-scale", type=float, default=1.0)
    parser.add_argument("--body-fisher-trust-radius", type=float, default=None)
    parser.add_argument("--body-fisher-curvature", choices=["model", "empirical"], default="model")
    parser.add_argument("--head-damping", type=float, default=50.0)
    parser.add_argument("--head-step-scale", default="auto")
    parser.add_argument("--max-head-step-scale", type=float, default=1.0)
    parser.add_argument("--head-trust-radius", type=float, default=200.0)
    parser.add_argument("--line-search-steps", type=int, default=0)
    parser.add_argument("--max-cg-iter", type=int, default=8)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--shuffle-prefixes", action="store_true")
    parser.add_argument("--limit-prefixes", type=str, default=None, help="Optional literal prefix list for prefix length 1 smoke runs.")
    parser.add_argument("--max-subtrees", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--sequential-val-chunk-size", type=int, default=1024)
    parser.add_argument("--no-circular", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0, help="Optional intra-epoch checkpoint cadence. Zero means checkpoint only at epoch boundaries.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_header(path: Path, append: bool = False) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(asdict(PrefixSubtreeRow(0, 0, 0, "", 0, 0, 0, None, None, None, None, None, 0.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)).keys()))
    if not append:
        writer.writeheader()
    handle.flush()
    return writer, handle


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


def make_model(seed: int, vocab_size: int, embedding_size: int, hidden_size: int, update_embeddings: bool) -> TinyCharRNN:
    torch.manual_seed(seed)
    model = TinyCharRNN(vocab_size, n_embd=embedding_size, n_hidden=hidden_size)
    if not update_embeddings:
        freeze_embeddings(model)
    return model


def evaluate(model: TinyCharRNN, vocab: CharVocab, val_text: str, chunk_size: int) -> tuple[float, float, float, float]:
    started = time.perf_counter()
    metrics = evaluate_sequential_text_loss(model, vocab, val_text, chunk_size=chunk_size)
    return metrics.nll_per_token, metrics.perplexity, metrics.bits_per_char, time.perf_counter() - started


def print_row(row: PrefixSubtreeRow) -> None:
    if row.prefix:
        print(
            f"epoch={row.epoch:03d} step={row.step:04d} prefix={row.prefix!r} "
            f"samples={row.sample_count} nodes={row.node_count} mass={row.transition_mass} "
            f"train_loss={row.train_loss} update_sec={row.update_sec} "
            f"body_sec={row.body_sec} cache_sec={row.cache_sec} head_sec={row.head_sec} "
            f"step_scale={row.step_scale} peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
            flush=True,
        )
    else:
        print(
            f"epoch={row.epoch:03d} step={row.step:04d} "
            f"seq_val_ppl={row.seq_val_ppl:.2f} seq_val_bpc={row.seq_val_bpc:.3f} "
            f"eval_sec={row.eval_sec:.2f}",
            flush=True,
        )


def save_checkpoint(
    path: Path | None,
    model: TinyCharRNN,
    optimizer: torch.optim.Optimizer,
    embedding_fisher: EmbeddingFisherPreconditioner | None,
    epoch: int,
    prefix_index: int,
    step: int,
    prefix_order: list[str],
    rng: random.Random,
    args: argparse.Namespace,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "embedding_fisher": embedding_fisher.state_dict() if embedding_fisher is not None else None,
        "epoch": epoch,
        "prefix_index": prefix_index,
        "step": step,
        "prefix_order": prefix_order,
        "rng_state": rng.getstate(),
        "config": {
            "seed": args.seed,
            "block_size": args.block_size,
            "stride": args.stride,
            "prefix_length": args.prefix_length,
            "train_fraction": args.train_fraction,
            "hidden_size": args.hidden_size,
            "embedding_size": args.embedding_size,
            "update_embeddings": args.update_embeddings,
            "embedding_fisher": args.embedding_fisher,
            "embedding_fisher_damping": args.embedding_fisher_damping,
            "embedding_fisher_step_scale": args.embedding_fisher_step_scale,
            "embedding_fisher_decay": args.embedding_fisher_decay,
            "body_lr": args.body_lr,
            "body_optimizer": args.body_optimizer,
            "body_fisher_damping": args.body_fisher_damping,
            "body_fisher_step_scale": args.body_fisher_step_scale,
            "body_fisher_max_step_scale": args.body_fisher_max_step_scale,
            "body_fisher_trust_radius": args.body_fisher_trust_radius,
            "body_fisher_curvature": args.body_fisher_curvature,
            "circular": not args.no_circular,
        },
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path | None) -> dict | None:
    if path is None:
        return None
    return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    if args.prefix_length < 1:
        raise ValueError("prefix_length must be at least 1")
    if args.prefix_length >= args.block_size:
        raise ValueError("prefix_length must be smaller than block_size")
    if args.resume and args.checkpoint_path is None:
        raise ValueError("--resume requires --checkpoint-path")
    if args.embedding_fisher and not args.update_embeddings:
        raise ValueError("--embedding-fisher requires --update-embeddings")
    if args.embedding_fisher and args.body_optimizer == "fisher":
        raise ValueError("--embedding-fisher cannot currently be combined with --body-optimizer fisher")
    args.run_id = resolve_run_id(args.run_id, resume=args.resume)
    args.output = prefixed_path(args.output, args.run_id)
    args.checkpoint_path = prefixed_path(args.checkpoint_path, args.run_id) if args.checkpoint_path is not None else None
    args.head_step_scale = parse_step_scale(args.head_step_scale)
    args.body_fisher_step_scale = parse_step_scale(str(args.body_fisher_step_scale))

    text = read_text(args.input)
    text_split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    make = make_samples if args.no_circular else make_circular_samples

    started = time.perf_counter()
    samples = make(text_split.train_text, block_size=args.block_size, stride=args.stride)
    sample_sec = time.perf_counter() - started
    started = time.perf_counter()
    samples.sort()
    sort_sec = time.perf_counter() - started
    ranges = prefix_ranges(samples, args.prefix_length)
    if args.limit_prefixes is not None:
        if args.prefix_length != 1:
            raise ValueError("limit_prefixes currently supports prefix_length=1 only")
        allowed = set(args.limit_prefixes)
        ranges = [item for item in ranges if item[0] in allowed]
    if args.max_subtrees is not None:
        ranges = ranges[: args.max_subtrees]
    ranges_by_prefix = {prefix: (prefix, lo, hi) for prefix, lo, hi in ranges}

    print(
        f"train_chars={len(text_split.train_text)} val_chars={len(text_split.val_text)} "
        f"samples={len(samples)} prefixes={len(ranges)} block_size={args.block_size} "
        f"prefix_length={args.prefix_length} circular={not args.no_circular} "
        f"run_id={args.run_id} output={args.output} checkpoint={args.checkpoint_path} "
        f"sample_sec={sample_sec:.4f} sort_sec={sort_sec:.4f}",
        flush=True,
    )

    model = make_model(args.seed, vocab.size, args.embedding_size, args.hidden_size, args.update_embeddings)
    body_params = [*model.cell.parameters()]
    embedding_fisher = None
    if args.embedding_fisher:
        embedding_fisher = EmbeddingFisherPreconditioner(
            vocab_size=vocab.size,
            embedding_size=args.embedding_size,
            damping=args.embedding_fisher_damping,
            step_scale=args.embedding_fisher_step_scale,
            decay=args.embedding_fisher_decay,
        )
    if args.update_embeddings and embedding_fisher is None:
        body_params.extend(model.embed.parameters())
    optimizer = torch.optim.AdamW(body_params, lr=args.body_lr)
    checkpoint = load_checkpoint(args.checkpoint_path) if args.resume else None
    start_epoch = 1
    start_prefix_index = 0
    step = 0
    rng = random.Random(args.seed)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if embedding_fisher is not None and checkpoint.get("embedding_fisher") is not None:
            embedding_fisher.load_state_dict(checkpoint["embedding_fisher"])
        start_epoch = int(checkpoint["epoch"])
        start_prefix_index = int(checkpoint["prefix_index"])
        step = int(checkpoint["step"])
        rng.setstate(checkpoint["rng_state"])
        print(
            f"resumed checkpoint={args.checkpoint_path} epoch={start_epoch} "
            f"prefix_index={start_prefix_index} step={step}",
            flush=True,
        )
    writer, handle = write_header(args.output, append=args.resume and args.output.exists())

    try:
        if checkpoint is None:
            nll, ppl, bpc, eval_sec = evaluate(model, vocab, text_split.val_text, args.sequential_val_chunk_size)
            row = PrefixSubtreeRow(args.seed, 0, step, "", 0, 0, 0, None, None, nll, ppl, bpc, args.body_lr, None, None, eval_sec, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            writer.writerow(asdict(row))
            handle.flush()
            print_row(row)
            save_checkpoint(args.checkpoint_path, model, optimizer, embedding_fisher, 1, 0, step, [], rng, args)

        for epoch in range(start_epoch, args.epochs + 1):
            if checkpoint is not None and epoch == start_epoch and checkpoint.get("prefix_order"):
                epoch_ranges = [ranges_by_prefix[prefix] for prefix in checkpoint["prefix_order"]]
            else:
                epoch_ranges = list(ranges)
                if args.shuffle_prefixes:
                    rng.shuffle(epoch_ranges)
            prefix_order = [prefix for prefix, _, _ in epoch_ranges]
            range_start = start_prefix_index if epoch == start_epoch else 0
            for prefix_index, (prefix, lo, hi) in enumerate(epoch_ranges[range_start:], start=range_start):
                step += 1
                prefix_samples = samples[lo:hi]
                suffix_samples = [sample[args.prefix_length :] for sample in prefix_samples]
                build_started = time.perf_counter()
                trie = samples_to_flat_trie(suffix_samples, vocab)
                build_sec = time.perf_counter() - build_started
                update_started = time.perf_counter()
                result = hybrid_epoch(
                    model,
                    vocab,
                    suffix_samples,
                    body_optimizer=optimizer,
                    head_damping=args.head_damping,
                    head_step_scale=args.head_step_scale,
                    max_grad_norm=args.max_grad_norm,
                    max_cg_iter=args.max_cg_iter,
                    cg_tolerance=args.cg_tolerance,
                    flat_trie=trie,
                    flat_prefix=prefix,
                    max_head_step_scale=args.max_head_step_scale,
                    line_search_steps=args.line_search_steps,
                    trust_radius=args.head_trust_radius,
                    compute_before_loss=False,
                    update_embeddings=args.update_embeddings,
                    embedding_fisher=embedding_fisher,
                    body_optimizer_kind=args.body_optimizer,
                    body_fisher_damping=args.body_fisher_damping,
                    body_fisher_step_scale=args.body_fisher_step_scale,
                    body_fisher_max_step_scale=args.body_fisher_max_step_scale,
                    body_fisher_trust_radius=args.body_fisher_trust_radius,
                    body_fisher_curvature=args.body_fisher_curvature,
                )
                update_sec = time.perf_counter() - update_started
                head = result.head_result
                embedding_stats = result.embedding_fisher_stats
                body_fisher_stats = result.body_fisher_stats
                row = PrefixSubtreeRow(
                    args.seed,
                    epoch,
                    step,
                    prefix,
                    len(prefix_samples),
                    trie.node_count,
                    int(trie.transition_counts.sum().item()),
                    result.after_loss,
                    result.after_body_loss,
                    None,
                    None,
                    None,
                    args.body_lr,
                    build_sec,
                    update_sec,
                    None,
                    result.body_runtime_sec,
                    result.hidden_cache_runtime_sec,
                    result.head_runtime_sec,
                    head.step_scale,
                    head.eta_quad,
                    head.eta_trust,
                    body_fisher_stats.cg_iterations if body_fisher_stats is not None else None,
                    body_fisher_stats.residual_norm if body_fisher_stats is not None else None,
                    body_fisher_stats.grad_norm if body_fisher_stats is not None else None,
                    body_fisher_stats.step_norm if body_fisher_stats is not None else None,
                    body_fisher_stats.predicted_improvement if body_fisher_stats is not None else None,
                    body_fisher_stats.linear_term if body_fisher_stats is not None else None,
                    body_fisher_stats.fisher_quadratic if body_fisher_stats is not None else None,
                    body_fisher_stats.step_scale if body_fisher_stats is not None else None,
                    body_fisher_stats.eta_quad if body_fisher_stats is not None else None,
                    body_fisher_stats.eta_trust if body_fisher_stats is not None else None,
                    embedding_stats.active_tokens if embedding_stats is not None else None,
                    embedding_stats.max_update_norm if embedding_stats is not None else None,
                    embedding_stats.mean_update_norm if embedding_stats is not None else None,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                writer.writerow(asdict(row))
                handle.flush()
                print_row(row)
                del trie, prefix_samples, suffix_samples
                gc.collect()
                if args.checkpoint_every_steps > 0 and step % args.checkpoint_every_steps == 0:
                    save_checkpoint(
                        args.checkpoint_path,
                        model,
                        optimizer,
                        embedding_fisher,
                        epoch,
                        prefix_index + 1,
                        step,
                        prefix_order,
                        rng,
                        args,
                    )

                if args.eval_every_steps and step % args.eval_every_steps == 0:
                    nll, ppl, bpc, eval_sec = evaluate(model, vocab, text_split.val_text, args.sequential_val_chunk_size)
                    eval_row = PrefixSubtreeRow(args.seed, epoch, step, "", 0, 0, 0, None, None, nll, ppl, bpc, args.body_lr, None, None, eval_sec, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    writer.writerow(asdict(eval_row))
                    handle.flush()
                    print_row(eval_row)

            nll, ppl, bpc, eval_sec = evaluate(model, vocab, text_split.val_text, args.sequential_val_chunk_size)
            row = PrefixSubtreeRow(args.seed, epoch, step, "", 0, 0, 0, None, None, nll, ppl, bpc, args.body_lr, None, None, eval_sec, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            writer.writerow(asdict(row))
            handle.flush()
            print_row(row)
            save_checkpoint(args.checkpoint_path, model, optimizer, embedding_fisher, epoch + 1, 0, step, [], rng, args)
            checkpoint = None
    finally:
        handle.close()
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
