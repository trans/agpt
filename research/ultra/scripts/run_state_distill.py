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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, make_circular_samples, make_samples, read_text
from agpt_ultra.eval import evaluate_sequential_text_loss, split_text
from agpt_ultra.flat_ops import collect_batched_flat_head_evidence_from_hidden, compute_flat_hidden_states, samples_to_flat_trie
from agpt_ultra.head_only import (
    batched_head_fisher_matvec,
    chunk_batched_head_evidence,
    chunked_batched_head_fisher_matvec,
    load_head_theta,
    model_head_theta,
    natural_head_step_matrix_free,
    suggested_quadratic_step_scale,
)
from agpt_ultra.hybrid import prefix_hidden_state
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id
from agpt_ultra.state_fisher import parse_step_scale, state_fisher_correct_hidden


@dataclass(frozen=True)
class StateDistillRow:
    epoch: int
    prefix: str
    sample_count: int
    node_count: int
    transition_mass: int
    fisher_before_ppl: float
    fisher_after_ppl: float
    distill_loss: float
    seq_val_nll: float | None
    seq_val_ppl: float | None
    seq_val_bpc: float | None
    mean_step_norm: float
    mean_eta: float | None
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill direct state-Fisher corrections into TinyCharRNN.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/state_distill.csv"))
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
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--distill-steps-per-subtree", type=int, default=1)
    parser.add_argument("--distill-target", choices=["probs", "hidden"], default="probs")
    parser.add_argument("--train-head", action="store_true")
    parser.add_argument("--head-fisher-after-distill", action="store_true")
    parser.add_argument("--head-fisher-after-epoch", action="store_true")
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
    if prefix_length == 0:
        return [("", 0, len(sorted_samples))]
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


def active_weighted_kl_loss(
    model: TinyCharRNN,
    hidden: torch.Tensor,
    counts: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    counts = counts.to(device=hidden.device, dtype=hidden.dtype)
    totals = counts.sum(dim=1)
    active = totals > 0
    logits = model.head(hidden[active])
    log_probs = F.log_softmax(logits, dim=1)
    target = target_probs[active].to(device=hidden.device, dtype=hidden.dtype)
    return -(totals[active, None] * target * log_probs).sum() / totals[active].sum().clamp_min(1.0)


def active_weighted_hidden_mse(
    hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    counts = counts.to(device=hidden.device, dtype=hidden.dtype)
    totals = counts.sum(dim=1)
    active = totals > 0
    weights = totals[active] / totals[active].sum().clamp_min(1.0)
    residual = hidden[active] - target_hidden[active].to(device=hidden.device, dtype=hidden.dtype)
    return (weights[:, None] * residual.square()).sum()


def safe_ppl_from_nll(nll: float, mass: int) -> float:
    if mass <= 0:
        return float("nan")
    value = nll / mass
    if value > 700.0:
        return float("inf")
    return math.exp(value)


@torch.no_grad()
def apply_epoch_head_fisher(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    ranges: list[tuple[str, int, int]],
    prefix_length: int,
    damping: float,
    max_cg_iter: int,
    max_step_scale: float,
) -> float:
    theta = model_head_theta(model)
    chunks = []
    for prefix, lo, hi in ranges:
        prefix_samples = samples[lo:hi]
        suffix_samples = [sample[prefix_length:] for sample in prefix_samples]
        trie = samples_to_flat_trie(suffix_samples, vocab)
        hidden = compute_flat_hidden_states(
            model,
            trie,
            initial_hidden=prefix_hidden_state(model, vocab, prefix),
        )
        chunks.append(collect_batched_flat_head_evidence_from_hidden(model, trie, hidden, theta))

    evidence = chunk_batched_head_evidence(chunks)
    head_cg = natural_head_step_matrix_free(
        evidence,
        damping=damping,
        max_cg_iter=max_cg_iter,
    )
    fisher_delta = chunked_batched_head_fisher_matvec(head_cg.solution, evidence)
    step_scale = suggested_quadratic_step_scale(
        evidence.gradient,
        head_cg.solution,
        fisher_delta,
        max_step_scale=max_step_scale,
    )
    load_head_theta(model, theta + step_scale * head_cg.solution)
    return step_scale


def main() -> None:
    args = parse_args()
    if args.prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")
    if args.prefix_length >= args.block_size:
        raise ValueError("prefix_length must be smaller than block_size")
    if args.distill_steps_per_subtree < 1:
        raise ValueError("distill_steps_per_subtree must be at least 1")
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
    if not args.train_head:
        for param in model.head.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(StateDistillRow(0, "", 0, 0, 0, 0.0, 0.0, 0.0, None, None, None, 0.0, None, 0.0, 0)).keys()),
        )
        writer.writeheader()
        print(
            f"samples={len(samples)} prefixes={len(ranges)} epochs={args.epochs} "
            f"block_size={args.block_size} prefix_length={args.prefix_length} "
            f"state_iterations={args.state_iterations} damping={args.damping} "
            f"step_scale={args.step_scale} train_head={args.train_head} "
            f"distill_target={args.distill_target} "
            f"distill_steps_per_subtree={args.distill_steps_per_subtree} "
            f"head_fisher_after_distill={args.head_fisher_after_distill} "
            f"head_fisher_after_epoch={args.head_fisher_after_epoch} "
            f"run_id={args.run_id} output={args.output}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            total_distill_loss = 0.0
            total_mass = 0
            total_fisher_before = 0.0
            total_fisher_after = 0.0
            weighted_step_norm = 0.0
            weighted_eta = 0.0
            eta_mass = 0
            for prefix, lo, hi in ranges:
                started = time.perf_counter()
                prefix_samples = samples[lo:hi]
                suffix_samples = [sample[args.prefix_length :] for sample in prefix_samples]
                trie = samples_to_flat_trie(suffix_samples, vocab)
                mass = int(trie.transition_counts.sum().item())

                model.eval()
                with torch.no_grad():
                    teacher_hidden = compute_flat_hidden_states(
                        model,
                        trie,
                        initial_hidden=prefix_hidden_state(model, vocab, prefix),
                    )
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
                    target_hidden = correction.corrected_hidden.detach()
                    target_probs = torch.softmax(model.head(target_hidden), dim=1).detach()

                model.train()
                loss = None
                for _ in range(args.distill_steps_per_subtree):
                    student_hidden = compute_flat_hidden_states(
                        model,
                        trie,
                        initial_hidden=prefix_hidden_state(model, vocab, prefix),
                    )
                    if args.distill_target == "hidden":
                        loss = active_weighted_hidden_mse(student_hidden, target_hidden, trie.transition_counts)
                    else:
                        loss = active_weighted_kl_loss(model, student_hidden, trie.transition_counts, target_probs)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            args.max_grad_norm,
                        )
                    optimizer.step()
                if loss is None:
                    raise RuntimeError("distillation loop did not run")
                if args.head_fisher_after_distill:
                    model.eval()
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
                        )
                        fisher_delta = batched_head_fisher_matvec(head_cg.solution, evidence)
                        head_step_scale = suggested_quadratic_step_scale(
                            evidence.gradient,
                            head_cg.solution,
                            fisher_delta,
                            max_step_scale=args.max_head_step_scale,
                        )
                        load_head_theta(model, theta + head_step_scale * head_cg.solution)

                total_distill_loss += float(loss.item()) * mass
                total_mass += mass
                total_fisher_before += correction.before_loss
                total_fisher_after += correction.after_loss
                weighted_step_norm += correction.mean_step_norm * mass
                if correction.mean_eta is not None:
                    weighted_eta += correction.mean_eta * mass
                    eta_mass += mass
                row = StateDistillRow(
                    epoch=epoch,
                    prefix=prefix,
                    sample_count=len(prefix_samples),
                    node_count=trie.node_count,
                    transition_mass=mass,
                    fisher_before_ppl=safe_ppl_from_nll(correction.before_loss, mass),
                    fisher_after_ppl=safe_ppl_from_nll(correction.after_loss, mass),
                    distill_loss=float(loss.item()),
                    seq_val_nll=None,
                    seq_val_ppl=None,
                    seq_val_bpc=None,
                    mean_step_norm=correction.mean_step_norm,
                    mean_eta=correction.mean_eta,
                    runtime_sec=time.perf_counter() - started,
                    peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                writer.writerow(asdict(row))
                handle.flush()

            if args.head_fisher_after_epoch:
                model.eval()
                head_step_scale = apply_epoch_head_fisher(
                    model,
                    vocab,
                    samples,
                    ranges,
                    args.prefix_length,
                    damping=args.head_damping,
                    max_cg_iter=args.max_head_cg_iter,
                    max_step_scale=args.max_head_step_scale,
                )
                print(f"epoch={epoch:03d} merged_head_step_scale={head_step_scale:.6g}", flush=True)

            should_eval = epoch == 1 or epoch == args.epochs or (args.eval_every > 0 and epoch % args.eval_every == 0)
            if should_eval:
                model.eval()
                metrics = evaluate_sequential_text_loss(model, vocab, split.val_text, chunk_size=args.eval_chunk_size)
                seq_val_nll = metrics.nll_per_token
                seq_val_ppl = metrics.perplexity
                seq_val_bpc = metrics.bits_per_char
            else:
                seq_val_nll = seq_val_ppl = seq_val_bpc = None
            summary = StateDistillRow(
                epoch=epoch,
                prefix="",
                sample_count=len(samples),
                node_count=0,
                transition_mass=total_mass,
                fisher_before_ppl=safe_ppl_from_nll(total_fisher_before, total_mass),
                fisher_after_ppl=safe_ppl_from_nll(total_fisher_after, total_mass),
                distill_loss=total_distill_loss / total_mass,
                seq_val_nll=seq_val_nll,
                seq_val_ppl=seq_val_ppl,
                seq_val_bpc=seq_val_bpc,
                mean_step_norm=weighted_step_norm / total_mass,
                mean_eta=(weighted_eta / eta_mass) if eta_mass else None,
                runtime_sec=time.perf_counter() - epoch_started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(summary))
            handle.flush()
            print(
                f"epoch={epoch:03d} fisher_ppl={summary.fisher_before_ppl:.3f}->{summary.fisher_after_ppl:.3f} "
                f"distill_loss={summary.distill_loss:.4f} seq_val_ppl={summary.seq_val_ppl} "
                f"runtime_sec={summary.runtime_sec:.2f} peak_rss_mb={summary.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
