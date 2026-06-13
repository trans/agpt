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
from agpt_ultra.hybrid import prefix_hidden_state
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id
from agpt_ultra.state_fisher import parse_step_scale, state_fisher_correct_hidden


@dataclass(frozen=True)
class FisherResidualRow:
    epoch: int
    prefix: str
    sample_count: int
    node_count: int
    transition_mass: int
    fisher_prior_ppl: float
    residual_ppl: float
    residual_loss: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural residual logits on top of detached state-Fisher priors.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/fisher_residual.csv"))
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fisher-head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-steps-per-subtree", type=int, default=1)
    parser.add_argument("--train-fisher-head", action="store_true")
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--step-scale", default="auto-node")
    parser.add_argument("--max-auto-eta", type=float, default=32.0)
    parser.add_argument("--auto-newton-steps", type=int, default=8)
    parser.add_argument("--state-iterations", type=int, default=3)
    parser.add_argument("--curvature", choices=["model", "empirical"], default="empirical")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--limit-prefixes", type=str, default=None)
    parser.add_argument("--max-subtrees", type=int, default=None)
    parser.add_argument("--no-circular", action="store_true")
    return parser.parse_args()


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


def residual_loss(
    model: TinyCharRNN,
    residual_head: nn.Linear,
    hidden: torch.Tensor,
    counts: torch.Tensor,
    corrected_hidden: torch.Tensor,
    residual_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    device = hidden.device
    dtype = hidden.dtype
    counts = counts.to(device=device, dtype=dtype)
    corrected_hidden = corrected_hidden.to(device=device, dtype=dtype)
    totals = counts.sum(dim=1)
    active = torch.nonzero(totals > 0, as_tuple=False).flatten()
    loss = hidden.new_tensor(0.0)
    for offset in range(0, active.numel(), chunk_size):
        node_ids = active[offset : offset + chunk_size]
        prior_logits = model.head(corrected_hidden[node_ids])
        logits = prior_logits + residual_scale * residual_head(hidden[node_ids])
        log_probs = F.log_softmax(logits, dim=1)
        loss = loss - (counts[node_ids] * log_probs).sum()
    return loss


def main() -> None:
    args = parse_args()
    if args.prefix_length < 1:
        raise ValueError("prefix_length must be at least 1")
    if args.prefix_length >= args.block_size:
        raise ValueError("prefix_length must be smaller than block_size")
    if args.residual_steps_per_subtree < 1:
        raise ValueError("residual_steps_per_subtree must be at least 1")
    args.step_scale = parse_step_scale(str(args.step_scale))
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    if args.checkpoint_output is None:
        args.checkpoint_output = args.output.with_suffix(".pt")
    else:
        args.checkpoint_output = prefixed_path(args.checkpoint_output, args.run_id)
    torch.manual_seed(args.seed)

    text = read_text(args.input)
    split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    make = make_samples if args.no_circular else make_circular_samples
    samples = make(split.train_text, block_size=args.block_size, stride=args.stride)
    samples.sort()
    ranges = prefix_ranges(samples, args.prefix_length)
    if args.limit_prefixes is not None:
        if args.prefix_length != 1:
            raise ValueError("--limit-prefixes currently supports prefix length 1 only")
        allowed = set(args.limit_prefixes)
        ranges = [item for item in ranges if item[0] in allowed]
    if args.max_subtrees is not None:
        ranges = ranges[: args.max_subtrees]

    model = TinyCharRNN(vocab.size, n_embd=args.embedding_size, n_hidden=args.hidden_size)
    residual_head = nn.Linear(args.hidden_size, vocab.size)
    nn.init.zeros_(residual_head.weight)
    nn.init.zeros_(residual_head.bias)
    fisher_head_params = list(model.head.parameters()) if args.train_fisher_head else []
    for param in model.head.parameters():
        param.requires_grad_(args.train_fisher_head)
    optimizer_groups = [
        {"params": [*model.embed.parameters(), *model.cell.parameters(), *residual_head.parameters()], "lr": args.lr},
    ]
    if fisher_head_params:
        optimizer_groups.append({"params": fisher_head_params, "lr": args.fisher_head_lr if args.fisher_head_lr is not None else args.lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    best_residual_ppl = float("inf")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(FisherResidualRow(0, "", 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        print(
            f"samples={len(samples)} prefixes={len(ranges)} epochs={args.epochs} "
            f"block_size={args.block_size} prefix_length={args.prefix_length} "
            f"hidden_size={args.hidden_size} state_iterations={args.state_iterations} "
            f"train_fisher_head={args.train_fisher_head} residual_scale={args.residual_scale} "
            f"lr={args.lr} fisher_head_lr={args.fisher_head_lr if args.fisher_head_lr is not None else args.lr} "
            f"residual_steps_per_subtree={args.residual_steps_per_subtree} "
            f"run_id={args.run_id} output={args.output}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            total_prior_loss = 0.0
            total_residual_loss = 0.0
            total_mass = 0
            for prefix, lo, hi in ranges:
                started = time.perf_counter()
                prefix_samples = samples[lo:hi]
                suffix_samples = [sample[args.prefix_length :] for sample in prefix_samples]
                trie = samples_to_flat_trie(suffix_samples, vocab)
                mass = int(trie.transition_counts.sum().item())
                initial_hidden = prefix_hidden_state(model, vocab, prefix)
                with torch.no_grad():
                    teacher_hidden = compute_flat_hidden_states(model, trie, initial_hidden=initial_hidden)
                    correction = state_fisher_correct_hidden(
                        model,
                        teacher_hidden,
                        trie.transition_counts,
                        damping=args.damping,
                        step_scale=args.step_scale,
                        max_auto_eta=args.max_auto_eta,
                        auto_newton_steps=args.auto_newton_steps,
                        state_iterations=args.state_iterations,
                        curvature=args.curvature,
                        chunk_size=args.chunk_size,
                    )
                    corrected_hidden = correction.corrected_hidden.detach()

                loss = None
                for _ in range(args.residual_steps_per_subtree):
                    student_hidden = compute_flat_hidden_states(
                        model,
                        trie,
                        initial_hidden=prefix_hidden_state(model, vocab, prefix),
                    )
                    loss = residual_loss(
                        model,
                        residual_head,
                        student_hidden,
                        trie.transition_counts,
                        corrected_hidden,
                        residual_scale=args.residual_scale,
                        chunk_size=args.chunk_size,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [param for group in optimizer.param_groups for param in group["params"]],
                            args.max_grad_norm,
                        )
                    optimizer.step()
                if loss is None:
                    raise RuntimeError("residual optimization loop did not run")
                residual_loss_value = float(loss.item())
                row = FisherResidualRow(
                    epoch=epoch,
                    prefix=prefix,
                    sample_count=len(prefix_samples),
                    node_count=trie.node_count,
                    transition_mass=mass,
                    fisher_prior_ppl=math.exp(correction.after_loss / mass) if mass else float("nan"),
                    residual_ppl=math.exp(residual_loss_value / mass) if mass else float("nan"),
                    residual_loss=residual_loss_value,
                    runtime_sec=time.perf_counter() - started,
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                writer.writerow(asdict(row))
                handle.flush()
                total_prior_loss += correction.after_loss
                total_residual_loss += residual_loss_value
                total_mass += mass

            summary = FisherResidualRow(
                epoch=epoch,
                prefix="",
                sample_count=len(samples),
                node_count=0,
                transition_mass=total_mass,
                fisher_prior_ppl=math.exp(total_prior_loss / total_mass),
                residual_ppl=math.exp(total_residual_loss / total_mass),
                residual_loss=total_residual_loss,
                runtime_sec=time.perf_counter() - epoch_started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(summary))
            handle.flush()
            print(
                f"epoch={epoch:03d} train_ppl={summary.fisher_prior_ppl:.3f}->{summary.residual_ppl:.3f} "
                f"runtime_sec={summary.runtime_sec:.2f} peak_rss_mb={summary.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
            is_best = summary.residual_ppl < best_residual_ppl
            if is_best:
                best_residual_ppl = summary.residual_ppl
            checkpoint = {
                "epoch": epoch,
                "best_residual_ppl": best_residual_ppl,
                "is_best": is_best,
                "args": vars(args),
                "vocab_chars": vocab.chars,
                "model_state_dict": model.state_dict(),
                "residual_head_state_dict": residual_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, args.checkpoint_output)
            if is_best:
                best_path = args.checkpoint_output.with_name(
                    f"{args.checkpoint_output.stem}_best{args.checkpoint_output.suffix}"
                )
                torch.save(checkpoint, best_path)
                print(f"checkpoint={args.checkpoint_output} best_checkpoint={best_path}", flush=True)
            else:
                print(f"checkpoint={args.checkpoint_output}", flush=True)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
