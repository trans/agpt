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
from agpt_ultra.eval import split_text
from agpt_ultra.flat_ops import compute_flat_hidden_states, samples_to_flat_trie
from agpt_ultra.hybrid import prefix_hidden_state
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id
from agpt_ultra.state_fisher import parse_step_scale, state_fisher_correct_hidden


@dataclass(frozen=True)
class StateFisherRow:
    seed: int
    prefix: str
    sample_count: int
    node_count: int
    transition_mass: int
    before_loss: float
    after_loss: float
    improvement: float
    before_ppl: float
    after_ppl: float
    mean_step_norm: float
    max_step_norm: float
    mean_eta: float | None
    min_eta: float | None
    max_eta: float | None
    solve_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct hidden-state Fisher diagnostic over prefix subtrees.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/state_fisher_diagnostic.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--step-scale", default="1.0")
    parser.add_argument("--max-auto-eta", type=float, default=8.0)
    parser.add_argument("--auto-newton-steps", type=int, default=8)
    parser.add_argument("--state-iterations", type=int, default=1)
    parser.add_argument("--curvature", choices=["model", "empirical"], default="model")
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


def make_model(seed: int, vocab_size: int, embedding_size: int, hidden_size: int) -> TinyCharRNN:
    torch.manual_seed(seed)
    model = TinyCharRNN(vocab_size, n_embd=embedding_size, n_hidden=hidden_size)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    args.step_scale = parse_step_scale(str(args.step_scale))
    if args.prefix_length < 1:
        raise ValueError("prefix_length must be at least 1")
    if args.prefix_length >= args.block_size:
        raise ValueError("prefix_length must be smaller than block_size")
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)

    text = read_text(args.input)
    text_split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    make = make_samples if args.no_circular else make_circular_samples
    samples = make(text_split.train_text, block_size=args.block_size, stride=args.stride)
    samples.sort()
    ranges = prefix_ranges(samples, args.prefix_length)
    if args.limit_prefixes is not None:
        if args.prefix_length != 1:
            raise ValueError("--limit-prefixes currently supports prefix length 1 only")
        allowed = set(args.limit_prefixes)
        ranges = [item for item in ranges if item[0] in allowed]
    if args.max_subtrees is not None:
        ranges = ranges[: args.max_subtrees]

    model = make_model(args.seed, vocab.size, args.embedding_size, args.hidden_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"samples={len(samples)} prefixes={len(ranges)} block_size={args.block_size} "
        f"damping={args.damping} step_scale={args.step_scale} curvature={args.curvature} "
        f"run_id={args.run_id} output={args.output}",
        flush=True,
    )

    totals = {
        "before_loss": 0.0,
        "after_loss": 0.0,
        "mass": 0,
        "weighted_step_norm": 0.0,
        "max_step_norm": 0.0,
    }
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(StateFisherRow(0, "", 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, 0.0, 0)).keys()))
        writer.writeheader()
        for prefix, lo, hi in ranges:
            prefix_samples = samples[lo:hi]
            suffix_samples = [sample[args.prefix_length :] for sample in prefix_samples]
            trie = samples_to_flat_trie(suffix_samples, vocab)
            started = time.perf_counter()
            hidden = compute_flat_hidden_states(
                model,
                trie,
                initial_hidden=prefix_hidden_state(model, vocab, prefix),
            )
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
            before_loss = correction.before_loss
            after_loss = correction.after_loss
            mean_step_norm = correction.mean_step_norm
            max_step_norm = correction.max_step_norm
            mean_eta = correction.mean_eta
            min_eta = correction.min_eta
            max_eta = correction.max_eta
            solve_sec = time.perf_counter() - started
            mass = int(trie.transition_counts.sum().item())
            before_ppl = math.exp(before_loss / mass) if mass else float("nan")
            after_ppl = math.exp(after_loss / mass) if mass else float("nan")
            row = StateFisherRow(
                seed=args.seed,
                prefix=prefix,
                sample_count=len(prefix_samples),
                node_count=trie.node_count,
                transition_mass=mass,
                before_loss=before_loss,
                after_loss=after_loss,
                improvement=before_loss - after_loss,
                before_ppl=before_ppl,
                after_ppl=after_ppl,
                mean_step_norm=mean_step_norm,
                max_step_norm=max_step_norm,
                mean_eta=mean_eta,
                min_eta=min_eta,
                max_eta=max_eta,
                solve_sec=solve_sec,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(row))
            handle.flush()
            totals["before_loss"] += before_loss
            totals["after_loss"] += after_loss
            totals["mass"] += mass
            totals["weighted_step_norm"] += mean_step_norm * mass
            totals["max_step_norm"] = max(totals["max_step_norm"], max_step_norm)
            print(
                f"prefix={prefix!r} samples={len(prefix_samples)} nodes={trie.node_count} mass={mass} "
                f"ppl={before_ppl:.3f}->{after_ppl:.3f} improvement={before_loss - after_loss:.2f} "
                f"mean_step={mean_step_norm:.4f} max_step={max_step_norm:.4f} "
                f"mean_eta={mean_eta} sec={solve_sec:.2f} "
                f"peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )

        mass = int(totals["mass"])
        before_ppl = math.exp(totals["before_loss"] / mass)
        after_ppl = math.exp(totals["after_loss"] / mass)
        summary = StateFisherRow(
            seed=args.seed,
            prefix="",
            sample_count=len(samples),
            node_count=0,
            transition_mass=mass,
            before_loss=totals["before_loss"],
            after_loss=totals["after_loss"],
            improvement=totals["before_loss"] - totals["after_loss"],
            before_ppl=before_ppl,
            after_ppl=after_ppl,
            mean_step_norm=totals["weighted_step_norm"] / mass,
            max_step_norm=totals["max_step_norm"],
            mean_eta=None,
            min_eta=None,
            max_eta=None,
            solve_sec=0.0,
            peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        writer.writerow(asdict(summary))
        print(
            f"summary mass={mass} ppl={before_ppl:.3f}->{after_ppl:.3f} "
            f"improvement={summary.improvement:.2f} peak_rss_mb={summary.peak_rss_kb / 1024:.1f}",
            flush=True,
        )
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
