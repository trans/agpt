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
from agpt_ultra.state_fisher import optimal_node_eta, parse_step_scale


@dataclass(frozen=True)
class BookStateRow:
    epoch: int
    train_ppl_before: float
    train_ppl_after_state: float
    train_ppl_after_head: float
    heldout_ppl: float
    heldout_bpc: float
    state_mean_step_norm: float
    state_mean_eta: float | None
    head_step_scale: float
    head_cg_iters: int
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure book trie-state model: materialized node states plus one global output head.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/book_state_model.csv"))
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--state-damping", type=float, default=10.0)
    parser.add_argument("--state-step-scale", default="auto-node")
    parser.add_argument("--state-max-auto-eta", type=float, default=32.0)
    parser.add_argument("--state-auto-newton-steps", type=int, default=8)
    parser.add_argument("--state-iterations", type=int, default=1)
    parser.add_argument("--state-curvature", choices=["model", "empirical"], default="empirical")
    parser.add_argument("--head-damping", type=float, default=50.0)
    parser.add_argument("--head-max-step-scale", type=float, default=1.0)
    parser.add_argument("--head-max-cg-iter", type=int, default=16)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--eval-max-tokens", type=int, default=None)
    return parser.parse_args()


def safe_exp(value: float) -> float:
    if value > 700.0:
        return float("inf")
    return math.exp(value)


def trie_children(trie: FlatTrie) -> list[dict[int, int]]:
    children: list[dict[int, int]] = [dict() for _ in range(trie.node_count)]
    for node_id in range(1, trie.node_count):
        parent = int(trie.parents[node_id].item())
        token = int(trie.tokens[node_id].item())
        children[parent][token] = node_id
    return children


def longest_suffix_node(context: list[int], children: list[dict[int, int]]) -> int:
    for start in range(len(context)):
        node_id = 0
        ok = True
        for token in context[start:]:
            child = children[node_id].get(token)
            if child is None:
                ok = False
                break
            node_id = child
        if ok:
            return node_id
    return 0


def train_loss(states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    logits = states @ weight.T + bias
    return -(counts * F.log_softmax(logits, dim=1)).sum()


@torch.no_grad()
def evaluate_heldout(
    states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    text: str,
    vocab: CharVocab,
    children: list[dict[int, int]],
    max_context: int,
    max_tokens: int | None,
) -> tuple[float, float]:
    ids = vocab.encode(text)
    total_loss = 0.0
    total_tokens = max(0, len(ids) - 1)
    if max_tokens is not None:
        total_tokens = min(total_tokens, max_tokens)
    if total_tokens == 0:
        raise ValueError("heldout text is too short")
    log_probs = F.log_softmax(states @ weight.T + bias, dim=1)
    for index in range(total_tokens):
        context = ids[max(0, index + 1 - max_context) : index + 1]
        node_id = longest_suffix_node(context, children)
        target = ids[index + 1]
        total_loss -= float(log_probs[node_id, target].item())
    nll = total_loss / total_tokens
    return safe_exp(nll), nll / math.log(2.0)


@torch.no_grad()
def state_fisher_update(
    states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    counts: torch.Tensor,
    damping: float,
    step_scale: float | str,
    max_auto_eta: float,
    auto_newton_steps: int,
    iterations: int,
    curvature: str,
    chunk_size: int,
) -> tuple[float, float | None]:
    totals = counts.sum(dim=1)
    active = torch.nonzero(totals > 0, as_tuple=False).flatten()
    identity = torch.eye(states.shape[1], dtype=states.dtype, device=states.device)
    step_norms: list[torch.Tensor] = []
    etas: list[torch.Tensor] = []

    for _ in range(iterations):
        logits = states @ weight.T + bias
        for offset in range(0, active.numel(), chunk_size):
            node_ids = active[offset : offset + chunk_size]
            node_states = states[node_ids]
            node_counts = counts[node_ids]
            node_totals = totals[node_ids]
            node_logits = logits[node_ids]
            predicted = torch.softmax(node_logits, dim=1)
            target = node_counts / node_totals[:, None]
            residual = predicted - target
            gradient = node_totals[:, None] * (residual @ weight)
            distribution = predicted if curvature == "model" else target
            weighted_gram = torch.einsum("nv,vh,vk->nhk", distribution, weight, weight)
            mean_direction = distribution @ weight
            fisher = node_totals[:, None, None] * (
                weighted_gram - torch.einsum("nh,nk->nhk", mean_direction, mean_direction)
            )
            fisher = fisher + damping * identity
            delta = torch.linalg.solve(fisher, -gradient.unsqueeze(-1)).squeeze(-1)
            if step_scale == "auto-node":
                direction_logits = delta @ weight.T
                eta = optimal_node_eta(
                    node_logits,
                    target,
                    direction_logits,
                    max_eta=max_auto_eta,
                    newton_steps=auto_newton_steps,
                )
                states[node_ids] = node_states + eta[:, None] * delta
                etas.append(eta)
            else:
                states[node_ids] = node_states + float(step_scale) * delta
            step_norms.append(delta.norm(dim=1))

    mean_step_norm = float(torch.cat(step_norms).mean().item()) if step_norms else 0.0
    mean_eta = float(torch.cat(etas).mean().item()) if etas else None
    return mean_step_norm, mean_eta


def flatten_head(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.cat([weight.reshape(-1), bias])


def unflatten_head(theta: torch.Tensor, vocab_size: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight_size = vocab_size * dim
    return theta[:weight_size].reshape(vocab_size, dim), theta[weight_size:]


def conjugate_gradient(matvec, rhs: torch.Tensor, max_iter: int, tolerance: float) -> tuple[torch.Tensor, int]:
    x = torch.zeros_like(rhs)
    residual = rhs - matvec(x)
    direction = residual.clone()
    residual_sq = residual.dot(residual)
    tolerance_sq = tolerance * tolerance
    iterations = 0
    for iterations in range(1, max_iter + 1):
        mat_direction = matvec(direction)
        denom = direction.dot(mat_direction)
        if abs(float(denom.item())) < 1e-30:
            break
        alpha = residual_sq / denom
        x = x + alpha * direction
        residual = residual - alpha * mat_direction
        next_residual_sq = residual.dot(residual)
        if float(next_residual_sq.item()) <= tolerance_sq:
            residual_sq = next_residual_sq
            break
        beta = next_residual_sq / residual_sq.clamp_min(1e-30)
        direction = residual + beta * direction
        residual_sq = next_residual_sq
    return x, iterations


@torch.no_grad()
def head_fisher_update(
    states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    counts: torch.Tensor,
    damping: float,
    max_step_scale: float,
    max_cg_iter: int,
    cg_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    totals = counts.sum(dim=1)
    mask = totals > 0
    active_states = states[mask]
    active_counts = counts[mask]
    active_totals = totals[mask]
    vocab_size, dim = weight.shape
    logits = active_states @ weight.T + bias
    predicted = torch.softmax(logits, dim=1)
    target = active_counts / active_totals[:, None]
    residual = predicted - target
    weighted_residual = active_totals[:, None] * residual
    grad_w = weighted_residual.T @ active_states
    grad_b = weighted_residual.sum(dim=0)
    gradient = flatten_head(grad_w, grad_b)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        vector_w, vector_b = unflatten_head(vector, vocab_size, dim)
        logits_v = active_states @ vector_w.T + vector_b
        centered = predicted * (logits_v - (predicted * logits_v).sum(dim=1, keepdim=True))
        weighted = active_totals[:, None] * centered
        out_w = weighted.T @ active_states
        out_b = weighted.sum(dim=0)
        return flatten_head(out_w, out_b) + damping * vector

    delta, cg_iters = conjugate_gradient(matvec, -gradient, max_iter=max_cg_iter, tolerance=cg_tolerance)
    fisher_delta = matvec(delta) - damping * delta
    numerator = -gradient.dot(delta).item()
    denominator = delta.dot(fisher_delta).item()
    if numerator <= 0.0 or denominator <= 0.0:
        step_scale = 0.0
    else:
        step_scale = min(max_step_scale, numerator / denominator)
    next_weight, next_bias = unflatten_head(flatten_head(weight, bias) + step_scale * delta, vocab_size, dim)
    return next_weight, next_bias, float(step_scale), cg_iters


def main() -> None:
    args = parse_args()
    if args.block_size < 2:
        raise ValueError("block_size must be at least 2")
    args.state_step_scale = parse_step_scale(str(args.state_step_scale))
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
    samples = make_circular_samples(split.train_text, block_size=args.block_size, stride=args.stride)
    samples.sort()
    trie = samples_to_flat_trie(samples, vocab)
    children = trie_children(trie)
    counts = trie.transition_counts

    states = 0.02 * torch.randn(trie.node_count, args.dim)
    weight = 0.02 * torch.randn(vocab.size, args.dim)
    bias = torch.zeros(vocab.size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(BookStateRow(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        print(
            f"samples={len(samples)} nodes={trie.node_count} mass={int(counts.sum().item())} "
            f"epochs={args.epochs} block_size={args.block_size} dim={args.dim} "
            f"state_step_scale={args.state_step_scale} state_iterations={args.state_iterations} "
            f"head_max_step_scale={args.head_max_step_scale} run_id={args.run_id} "
            f"output={args.output} checkpoint={args.checkpoint_output}",
            flush=True,
        )

        mass = float(counts.sum().item())
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            before_loss = float(train_loss(states, weight, bias, counts).item())
            mean_step_norm, mean_eta = state_fisher_update(
                states,
                weight,
                bias,
                counts,
                damping=args.state_damping,
                step_scale=args.state_step_scale,
                max_auto_eta=args.state_max_auto_eta,
                auto_newton_steps=args.state_auto_newton_steps,
                iterations=args.state_iterations,
                curvature=args.state_curvature,
                chunk_size=args.chunk_size,
            )
            after_state_loss = float(train_loss(states, weight, bias, counts).item())
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
            row = BookStateRow(
                epoch=epoch,
                train_ppl_before=safe_exp(before_loss / mass),
                train_ppl_after_state=safe_exp(after_state_loss / mass),
                train_ppl_after_head=safe_exp(after_head_loss / mass),
                heldout_ppl=heldout_ppl,
                heldout_bpc=heldout_bpc,
                state_mean_step_norm=mean_step_norm,
                state_mean_eta=mean_eta,
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
                f"head_step={row.head_step_scale:.4g} runtime_sec={row.runtime_sec:.2f} "
                f"peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
    torch.save(
        {
            "states": states.detach().cpu(),
            "weight": weight.detach().cpu(),
            "bias": bias.detach().cpu(),
            "vocab_chars": vocab.chars,
            "trie": {
                "vocab_size": trie.vocab_size,
                "parents": trie.parents.cpu(),
                "tokens": trie.tokens.cpu(),
                "depths": trie.depths.cpu(),
                "counts": trie.counts.cpu(),
                "transition_counts": trie.transition_counts.cpu(),
            },
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        args.checkpoint_output,
    )
    print(f"checkpoint={args.checkpoint_output}", flush=True)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
