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

from agpt_ultra.data import CharVocab, make_circular_samples, read_text
from agpt_ultra.eval import split_text
from agpt_ultra.flat_ops import samples_to_flat_trie
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.run_ids import prefixed_path, resolve_run_id
from scripts.run_book_state_model import (
    conjugate_gradient,
    evaluate_heldout,
    head_fisher_update,
    safe_exp,
    trie_children,
)


@dataclass(frozen=True)
class ReconciledTransitionRow:
    epoch: int
    train_ppl_before: float
    train_ppl_after_state: float
    train_ppl_after_head: float
    heldout_ppl: float
    heldout_bpc: float
    root_step_norm: float
    token_step_mean_norm: float
    head_step_scale: float
    head_cg_iters: int
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tree-reconciled token-transition state model.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/reconciled_transition.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--state-damping", type=float, default=50.0)
    parser.add_argument("--state-step-scale", type=float, default=1.0)
    parser.add_argument("--state-line-search-steps", type=int, default=8)
    parser.add_argument("--head-damping", type=float, default=50.0)
    parser.add_argument("--head-max-step-scale", type=float, default=1.0)
    parser.add_argument("--head-max-cg-iter", type=int, default=16)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--eval-max-tokens", type=int, default=20000)
    return parser.parse_args()


def compute_transition_states(trie: FlatTrie, root: torch.Tensor, token_delta: torch.Tensor) -> torch.Tensor:
    states = root.new_zeros((trie.node_count, root.numel()))
    states[0] = root
    parents = trie.parents
    tokens = trie.tokens
    depths = trie.depths
    max_depth = int(depths.max().item())
    for depth in range(1, max_depth + 1):
        node_ids = torch.nonzero(depths == depth, as_tuple=False).flatten()
        states[node_ids] = states[parents[node_ids]] + token_delta[tokens[node_ids]]
    return states


def train_loss(states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return -(counts * F.log_softmax(states @ weight.T + bias, dim=1)).sum()


@torch.no_grad()
def local_diag_state_evidence(
    states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    totals = counts.sum(dim=1)
    logits = states @ weight.T + bias
    predicted = torch.softmax(logits, dim=1)
    target = torch.zeros_like(predicted)
    mask = totals > 0
    target[mask] = counts[mask] / totals[mask, None]
    residual = predicted - target
    gradient = totals[:, None] * (residual @ weight)

    # Diagonal of W^T Cov_q W.
    weighted_w2 = target @ weight.square()
    mean_w = target @ weight
    fisher_diag = totals[:, None] * (weighted_w2 - mean_w.square()).clamp_min(0.0)
    return gradient, fisher_diag


@torch.no_grad()
def reconcile_state_updates(
    trie: FlatTrie,
    local_gradient: torch.Tensor,
    local_fisher_diag: torch.Tensor,
    state_damping: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    gradient = local_gradient.clone()
    fisher_diag = local_fisher_diag.clone()
    parents = trie.parents
    tokens = trie.tokens

    for node_id in range(trie.node_count - 1, 0, -1):
        parent = int(parents[node_id].item())
        gradient[parent] += gradient[node_id]
        fisher_diag[parent] += fisher_diag[node_id]

    root_delta = -gradient[0] / (fisher_diag[0] + state_damping)
    token_gradient = torch.zeros(
        (trie.vocab_size, gradient.shape[1]),
        dtype=gradient.dtype,
        device=gradient.device,
    )
    token_fisher = torch.zeros_like(token_gradient)
    for node_id in range(1, trie.node_count):
        token = int(tokens[node_id].item())
        token_gradient[token] += gradient[node_id]
        token_fisher[token] += fisher_diag[node_id]
    token_delta_update = -token_gradient / (token_fisher + state_damping)
    token_step_norms = token_delta_update.norm(dim=1)
    return (
        root_delta,
        token_delta_update,
        float(root_delta.norm().item()),
        float(token_step_norms.mean().item()),
    )


def main() -> None:
    args = parse_args()
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    torch.manual_seed(args.seed)

    text = read_text(args.input)
    split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    samples = make_circular_samples(split.train_text, block_size=args.block_size, stride=args.stride)
    samples.sort()
    trie = samples_to_flat_trie(samples, vocab)
    children = trie_children(trie)
    counts = trie.transition_counts

    root = 0.02 * torch.randn(args.dim)
    token_delta = 0.02 * torch.randn(vocab.size, args.dim)
    weight = 0.02 * torch.randn(vocab.size, args.dim)
    bias = torch.zeros(vocab.size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(ReconciledTransitionRow(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        mass = float(counts.sum().item())
        print(
            f"samples={len(samples)} nodes={trie.node_count} mass={int(mass)} "
            f"epochs={args.epochs} block_size={args.block_size} dim={args.dim} "
            f"state_damping={args.state_damping} state_step_scale={args.state_step_scale} "
            f"run_id={args.run_id} output={args.output}",
            flush=True,
        )

        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            states = compute_transition_states(trie, root, token_delta)
            before_loss = float(train_loss(states, weight, bias, counts).item())
            local_gradient, local_fisher_diag = local_diag_state_evidence(states, weight, bias, counts)
            root_step, token_step, root_norm, token_mean_norm = reconcile_state_updates(
                trie,
                local_gradient,
                local_fisher_diag,
                state_damping=args.state_damping,
            )
            accepted_scale = 0.0
            after_state_loss = before_loss
            accepted_states = states
            trial_scale = args.state_step_scale
            for _ in range(args.state_line_search_steps + 1):
                candidate_root = root + trial_scale * root_step
                candidate_token_delta = token_delta + trial_scale * token_step
                candidate_states = compute_transition_states(trie, candidate_root, candidate_token_delta)
                candidate_loss = float(train_loss(candidate_states, weight, bias, counts).item())
                if candidate_loss < before_loss:
                    accepted_scale = trial_scale
                    after_state_loss = candidate_loss
                    accepted_states = candidate_states
                    root = candidate_root
                    token_delta = candidate_token_delta
                    break
                trial_scale *= 0.5
            states = accepted_states
            weight, bias, head_step_scale, head_cg_iters = head_fisher_update(
                states,
                weight,
                bias,
                counts,
                damping=args.head_damping,
                max_step_scale=args.head_max_step_scale,
                max_cg_iter=args.head_max_cg_iter,
                cg_tolerance=args.cg_tolerance,
            )
            after_head_loss = float(train_loss(states, weight, bias, counts).item())
            heldout_ppl, heldout_bpc = evaluate_heldout(
                states,
                weight,
                bias,
                split.val_text,
                vocab,
                children,
                max_context=args.block_size,
                max_tokens=args.eval_max_tokens,
            )
            row = ReconciledTransitionRow(
                epoch=epoch,
                train_ppl_before=safe_exp(before_loss / mass),
                train_ppl_after_state=safe_exp(after_state_loss / mass),
                train_ppl_after_head=safe_exp(after_head_loss / mass),
                heldout_ppl=heldout_ppl,
                heldout_bpc=heldout_bpc,
                root_step_norm=root_norm,
                token_step_mean_norm=token_mean_norm,
                head_step_scale=head_step_scale,
                head_cg_iters=head_cg_iters,
                runtime_sec=time.perf_counter() - started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(row))
            handle.flush()
            print(
                f"epoch={epoch:03d} train_ppl={row.train_ppl_before:.3f}->{row.train_ppl_after_state:.3f}"
                f"->{row.train_ppl_after_head:.3f} heldout_ppl={row.heldout_ppl:.3f} "
                f"state_scale={accepted_scale:.3g} root_step={row.root_step_norm:.3g} "
                f"token_step={row.token_step_mean_norm:.3g} "
                f"runtime_sec={row.runtime_sec:.2f} peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
