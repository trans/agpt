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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, make_circular_samples, make_samples, read_text
from agpt_ultra.eval import evaluate_sequential_text_loss, split_text
from agpt_ultra.flat_ops import collect_batched_flat_head_evidence_from_hidden, compute_flat_hidden_states, samples_to_flat_trie
from agpt_ultra.head_only import (
    batched_head_fisher_matvec,
    load_head_theta,
    model_head_theta,
    natural_head_step_matrix_free,
    suggested_quadratic_step_scale,
)
from agpt_ultra.hybrid import prefix_hidden_state
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id
from agpt_ultra.state_fisher import parse_step_scale, state_fisher_correct_hidden
from agpt_ultra.state_projection import state_delta_projection_step


@dataclass(frozen=True)
class StateDeltaProjectRow:
    epoch: int
    prefix: str
    sample_count: int
    node_count: int
    transition_mass: int
    fisher_before_ppl: float
    fisher_after_ppl: float
    seq_val_nll: float | None
    seq_val_ppl: float | None
    seq_val_bpc: float | None
    cg_iterations: int
    projection_mse_before: float
    projection_mse_after: float
    projection_ratio: float
    target_delta_norm: float
    projected_delta_norm: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project direct state-Fisher hidden deltas into TinyCharRNN parameters.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/state_delta_project.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--projection-damping", type=float, default=1.0)
    parser.add_argument("--projection-step-scale", type=float, default=1.0)
    parser.add_argument("--max-cg-iter", type=int, default=8)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--head-fisher-after-projection", action="store_true")
    parser.add_argument("--head-damping", type=float, default=50.0)
    parser.add_argument("--max-head-cg-iter", type=int, default=8)
    parser.add_argument("--max-head-step-scale", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--step-scale", default="auto-node")
    parser.add_argument("--max-auto-eta", type=float, default=32.0)
    parser.add_argument("--auto-newton-steps", type=int, default=8)
    parser.add_argument("--state-iterations", type=int, default=3)
    parser.add_argument("--curvature", choices=["model", "empirical"], default="empirical")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--limit-prefixes", type=str, default=None)
    parser.add_argument("--max-subtrees", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-chunk-size", type=int, default=2048)
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


def main() -> None:
    args = parse_args()
    if args.prefix_length < 1:
        raise ValueError("prefix_length must be at least 1")
    if args.prefix_length >= args.block_size:
        raise ValueError("prefix_length must be smaller than block_size")
    args.step_scale = parse_step_scale(str(args.step_scale))
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(StateDeltaProjectRow(0, "", 0, 0, 0, 0.0, 0.0, None, None, None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        print(
            f"samples={len(samples)} prefixes={len(ranges)} epochs={args.epochs} "
            f"block_size={args.block_size} prefix_length={args.prefix_length} "
            f"hidden_size={args.hidden_size} state_iterations={args.state_iterations} "
            f"projection_damping={args.projection_damping} projection_step_scale={args.projection_step_scale} "
            f"max_cg_iter={args.max_cg_iter} update_embeddings={not args.freeze_embeddings} "
            f"head_fisher_after_projection={args.head_fisher_after_projection} "
            f"run_id={args.run_id} output={args.output}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            total_mass = 0
            total_fisher_before = 0.0
            total_fisher_after = 0.0
            total_mse_before = 0.0
            total_mse_after = 0.0
            weighted_target_norm = 0.0
            weighted_projected_norm = 0.0
            total_cg = 0
            for prefix, lo, hi in ranges:
                started = time.perf_counter()
                prefix_samples = samples[lo:hi]
                suffix_samples = [sample[args.prefix_length :] for sample in prefix_samples]
                trie = samples_to_flat_trie(suffix_samples, vocab)
                mass = int(trie.transition_counts.sum().item())
                initial_hidden = prefix_hidden_state(model, vocab, prefix)
                with torch.no_grad():
                    hidden = compute_flat_hidden_states(model, trie, initial_hidden=initial_hidden)
                    correction = state_fisher_correct_hidden(
                        model,
                        hidden,
                        trie.transition_counts,
                        damping=args.damping,
                        step_scale=args.step_scale,
                        max_auto_eta=args.max_auto_eta,
                        auto_newton_steps=args.auto_newton_steps,
                        state_iterations=args.state_iterations,
                        curvature=args.curvature,
                        chunk_size=args.chunk_size,
                    )
                    target_delta = correction.corrected_hidden - hidden

                projection = state_delta_projection_step(
                    model,
                    trie,
                    target_delta,
                    prefix_token_ids=tuple(vocab.encode(prefix)),
                    update_embeddings=not args.freeze_embeddings,
                    damping=args.projection_damping,
                    step_scale=args.projection_step_scale,
                    max_cg_iter=args.max_cg_iter,
                    cg_tolerance=args.cg_tolerance,
                )
                if args.head_fisher_after_projection:
                    with torch.no_grad():
                        projected_hidden = compute_flat_hidden_states(
                            model,
                            trie,
                            initial_hidden=prefix_hidden_state(model, vocab, prefix),
                        )
                        theta = model_head_theta(model)
                        evidence = collect_batched_flat_head_evidence_from_hidden(
                            model,
                            trie,
                            projected_hidden,
                            theta,
                        )
                        head_cg = natural_head_step_matrix_free(
                            evidence,
                            damping=args.head_damping,
                            max_cg_iter=args.max_head_cg_iter,
                            cg_tolerance=args.cg_tolerance,
                        )
                        fisher_delta = batched_head_fisher_matvec(head_cg.solution, evidence)
                        head_step_scale = suggested_quadratic_step_scale(
                            evidence.gradient,
                            head_cg.solution,
                            fisher_delta,
                            max_step_scale=args.max_head_step_scale,
                        )
                        load_head_theta(model, theta + head_step_scale * head_cg.solution)
                ratio = projection.weighted_mse_after / max(projection.weighted_mse_before, 1e-30)
                row = StateDeltaProjectRow(
                    epoch=epoch,
                    prefix=prefix,
                    sample_count=len(prefix_samples),
                    node_count=trie.node_count,
                    transition_mass=mass,
                    fisher_before_ppl=math.exp(correction.before_loss / mass) if mass else float("nan"),
                    fisher_after_ppl=math.exp(correction.after_loss / mass) if mass else float("nan"),
                    seq_val_nll=None,
                    seq_val_ppl=None,
                    seq_val_bpc=None,
                    cg_iterations=projection.cg_iterations,
                    projection_mse_before=projection.weighted_mse_before,
                    projection_mse_after=projection.weighted_mse_after,
                    projection_ratio=ratio,
                    target_delta_norm=projection.target_delta_norm,
                    projected_delta_norm=projection.projected_delta_norm,
                    runtime_sec=time.perf_counter() - started,
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                writer.writerow(asdict(row))
                handle.flush()

                total_mass += mass
                total_fisher_before += correction.before_loss
                total_fisher_after += correction.after_loss
                total_mse_before += projection.weighted_mse_before * mass
                total_mse_after += projection.weighted_mse_after * mass
                weighted_target_norm += projection.target_delta_norm * mass
                weighted_projected_norm += projection.projected_delta_norm * mass
                total_cg += projection.cg_iterations

            should_eval = epoch == 1 or epoch == args.epochs or (args.eval_every > 0 and epoch % args.eval_every == 0)
            if should_eval:
                metrics = evaluate_sequential_text_loss(model, vocab, split.val_text, chunk_size=args.eval_chunk_size)
                seq_val_nll = metrics.nll_per_token
                seq_val_ppl = metrics.perplexity
                seq_val_bpc = metrics.bits_per_char
            else:
                seq_val_nll = seq_val_ppl = seq_val_bpc = None
            summary = StateDeltaProjectRow(
                epoch=epoch,
                prefix="",
                sample_count=len(samples),
                node_count=0,
                transition_mass=total_mass,
                fisher_before_ppl=math.exp(total_fisher_before / total_mass),
                fisher_after_ppl=math.exp(total_fisher_after / total_mass),
                seq_val_nll=seq_val_nll,
                seq_val_ppl=seq_val_ppl,
                seq_val_bpc=seq_val_bpc,
                cg_iterations=total_cg,
                projection_mse_before=total_mse_before / total_mass,
                projection_mse_after=total_mse_after / total_mass,
                projection_ratio=total_mse_after / max(total_mse_before, 1e-30),
                target_delta_norm=weighted_target_norm / total_mass,
                projected_delta_norm=weighted_projected_norm / total_mass,
                runtime_sec=time.perf_counter() - epoch_started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(summary))
            handle.flush()
            print(
                f"epoch={epoch:03d} fisher_ppl={summary.fisher_before_ppl:.3f}->{summary.fisher_after_ppl:.3f} "
                f"seq_val_ppl={summary.seq_val_ppl} projection_ratio={summary.projection_ratio:.4f} "
                f"cg_iters={summary.cg_iterations} runtime_sec={summary.runtime_sec:.2f} "
                f"peak_rss_mb={summary.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
