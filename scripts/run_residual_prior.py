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
from agpt_ultra.eval import split_text
from agpt_ultra.flat_ops import compute_flat_hidden_states, samples_to_flat_trie
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.run_ids import prefixed_path, resolve_run_id


@dataclass(frozen=True)
class ResidualPriorRow:
    epoch: int
    train_prior_ppl: float
    train_residual_ppl: float
    val_prior_ppl: float
    val_residual_ppl: float
    train_loss: float
    train_tokens: int
    val_tokens: int
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural residual on top of a trie/backoff prior.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/residual_prior.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prior-mode", choices=["backoff", "exact"], default="backoff")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--weight-mode", choices=["power", "bandpass"], default="power")
    parser.add_argument("--mass-beta", type=float, default=1.0)
    parser.add_argument("--entropy-gamma", type=float, default=0.0)
    parser.add_argument("--entropy-floor", type=float, default=0.0)
    parser.add_argument("--mass-trust-k", type=float, default=16.0)
    parser.add_argument("--normalize-weights", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--no-circular", action="store_true")
    return parser.parse_args()


def trie_node_paths(trie: FlatTrie) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = [()]
    for node_id in range(1, trie.node_count):
        parent = int(trie.parents[node_id].item())
        paths.append(paths[parent] + (int(trie.tokens[node_id].item()),))
    return paths


def count_map_from_trie(trie: FlatTrie, paths: list[tuple[int, ...]]) -> dict[tuple[int, ...], torch.Tensor]:
    return {
        path: trie.transition_counts[node_id].detach().clone()
        for node_id, path in enumerate(paths)
    }


def smoothed_distribution(counts: torch.Tensor, unigram: torch.Tensor, alpha: float) -> torch.Tensor:
    total = counts.sum()
    if float(total.item()) <= 0.0:
        return unigram
    return (counts + alpha * unigram) / (total + alpha)


def prior_context(path: tuple[int, ...], mode: str) -> tuple[int, ...]:
    if mode == "exact":
        return path
    if not path:
        return path
    return path[1:]


def prior_probs_for_paths(
    paths: list[tuple[int, ...]],
    count_map: dict[tuple[int, ...], torch.Tensor],
    unigram: torch.Tensor,
    alpha: float,
    mode: str,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for path in paths:
        lookup = prior_context(path, mode)
        while lookup and lookup not in count_map:
            lookup = lookup[1:]
        counts = count_map.get(lookup)
        if counts is None:
            rows.append(unigram)
        else:
            rows.append(smoothed_distribution(counts, unigram, alpha))
    return torch.stack(rows, dim=0)


def nll_from_prior(trie: FlatTrie, prior_probs: torch.Tensor) -> tuple[float, int, float]:
    counts = trie.transition_counts.to(dtype=prior_probs.dtype, device=prior_probs.device)
    log_prior = prior_probs.clamp_min(1e-12).log()
    loss = -(counts * log_prior).sum()
    tokens = int(counts.sum().item())
    return float(loss.item()), tokens, math.exp(float(loss.item()) / tokens)


def residual_loss_from_hidden(
    model: TinyCharRNN,
    trie: FlatTrie,
    hidden: torch.Tensor,
    prior_probs: torch.Tensor,
    residual_scale: float,
    chunk_size: int,
    mass_beta: float,
    entropy_gamma: float,
    entropy_floor: float,
    weight_mode: str,
    mass_trust_k: float,
    normalize_weights: bool,
) -> torch.Tensor:
    device = hidden.device
    dtype = hidden.dtype
    counts = trie.transition_counts.to(device=device, dtype=dtype)
    totals = counts.sum(dim=1)
    active = torch.nonzero(totals > 0, as_tuple=False).flatten()
    prior_log = prior_probs.to(device=device, dtype=dtype).clamp_min(1e-12).log()
    loss = hidden.new_tensor(0.0)
    weighted_tokens = hidden.new_tensor(0.0)
    for offset in range(0, active.numel(), chunk_size):
        node_ids = active[offset : offset + chunk_size]
        node_counts = counts[node_ids]
        node_totals = totals[node_ids]
        target = node_counts / node_totals[:, None]
        entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=1) / math.log(counts.shape[1])
        if weight_mode == "bandpass":
            mass_weight = node_totals / (node_totals + mass_trust_k)
            entropy_weight = 4.0 * entropy * (1.0 - entropy)
            entropy_weight = entropy_weight.clamp_min(entropy_floor)
            row_weight = (mass_weight * entropy_weight / node_totals.clamp_min(1.0))[:, None]
        else:
            entropy_weight = entropy.clamp_min(entropy_floor).pow(entropy_gamma)
            mass_weight = node_totals.clamp_min(1.0).pow(mass_beta - 1.0)
            row_weight = (mass_weight * entropy_weight)[:, None]
        logits = prior_log[node_ids] + residual_scale * model.head(hidden[node_ids])
        log_probs = F.log_softmax(logits, dim=1)
        loss = loss - (row_weight * node_counts * log_probs).sum()
        weighted_tokens = weighted_tokens + (row_weight.squeeze(1) * node_totals).sum()
    if normalize_weights:
        return loss * (totals.sum() / weighted_tokens.clamp_min(1e-12))
    return loss


@torch.no_grad()
def evaluate_residual(
    model: TinyCharRNN,
    trie: FlatTrie,
    prior_probs: torch.Tensor,
    residual_scale: float,
    chunk_size: int,
    mass_beta: float,
    entropy_gamma: float,
    entropy_floor: float,
    weight_mode: str,
    mass_trust_k: float,
    normalize_weights: bool,
) -> tuple[float, int, float]:
    hidden = compute_flat_hidden_states(model, trie)
    loss = residual_loss_from_hidden(
        model,
        trie,
        hidden,
        prior_probs,
        residual_scale,
        chunk_size,
        mass_beta=mass_beta,
        entropy_gamma=entropy_gamma,
        entropy_floor=entropy_floor,
        weight_mode=weight_mode,
        mass_trust_k=mass_trust_k,
        normalize_weights=normalize_weights,
    )
    tokens = int(trie.transition_counts.sum().item())
    return float(loss.item()), tokens, math.exp(float(loss.item()) / tokens)


def main() -> None:
    args = parse_args()
    if args.block_size < 2:
        raise ValueError("block_size must be at least 2")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    torch.manual_seed(args.seed)

    text = read_text(args.input)
    split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    make = make_samples if args.no_circular else make_circular_samples
    train_samples = make(split.train_text, block_size=args.block_size, stride=args.stride)
    val_samples = make(split.val_text, block_size=args.block_size, stride=args.stride)
    train_samples.sort()
    val_samples.sort()
    train_trie = samples_to_flat_trie(train_samples, vocab)
    val_trie = samples_to_flat_trie(val_samples, vocab)
    train_paths = trie_node_paths(train_trie)
    val_paths = trie_node_paths(val_trie)
    train_count_map = count_map_from_trie(train_trie, train_paths)
    unigram_counts = train_count_map[()]
    unigram = unigram_counts / unigram_counts.sum().clamp_min(1.0)
    train_prior = prior_probs_for_paths(train_paths, train_count_map, unigram, args.alpha, args.prior_mode)
    val_prior = prior_probs_for_paths(val_paths, train_count_map, unigram, args.alpha, args.prior_mode)
    train_prior_loss, train_tokens, train_prior_ppl = nll_from_prior(train_trie, train_prior)
    val_prior_loss, val_tokens, val_prior_ppl = nll_from_prior(val_trie, val_prior)

    model = TinyCharRNN(vocab.size, n_embd=args.embedding_size, n_hidden=args.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(ResidualPriorRow(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)).keys()),
        )
        writer.writeheader()
        print(
            f"train_samples={len(train_samples)} val_samples={len(val_samples)} "
            f"train_nodes={train_trie.node_count} val_nodes={val_trie.node_count} "
            f"epochs={args.epochs} block_size={args.block_size} prior_mode={args.prior_mode} "
            f"alpha={args.alpha} residual_scale={args.residual_scale} "
            f"weight_mode={args.weight_mode} mass_beta={args.mass_beta} "
            f"entropy_gamma={args.entropy_gamma} entropy_floor={args.entropy_floor} "
            f"mass_trust_k={args.mass_trust_k} "
            f"normalize_weights={args.normalize_weights} "
            f"hidden_size={args.hidden_size} embedding_size={args.embedding_size} "
            f"train_prior_ppl={train_prior_ppl:.3f} val_prior_ppl={val_prior_ppl:.3f} "
            f"run_id={args.run_id} output={args.output}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            hidden = compute_flat_hidden_states(model, train_trie)
            loss = residual_loss_from_hidden(
                model,
                train_trie,
                hidden,
                train_prior,
                residual_scale=args.residual_scale,
                chunk_size=args.chunk_size,
                mass_beta=args.mass_beta,
                entropy_gamma=args.entropy_gamma,
                entropy_floor=args.entropy_floor,
                weight_mode=args.weight_mode,
                mass_trust_k=args.mass_trust_k,
                normalize_weights=args.normalize_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            should_eval = epoch == 1 or epoch == args.epochs or (args.eval_every > 0 and epoch % args.eval_every == 0)
            if should_eval:
                model.eval()
                _, _, train_residual_ppl = evaluate_residual(
                    model,
                    train_trie,
                    train_prior,
                    residual_scale=args.residual_scale,
                    chunk_size=args.chunk_size,
                    mass_beta=1.0,
                    entropy_gamma=0.0,
                    entropy_floor=0.0,
                    weight_mode="power",
                    mass_trust_k=args.mass_trust_k,
                    normalize_weights=False,
                )
                _, _, val_residual_ppl = evaluate_residual(
                    model,
                    val_trie,
                    val_prior,
                    residual_scale=args.residual_scale,
                    chunk_size=args.chunk_size,
                    mass_beta=1.0,
                    entropy_gamma=0.0,
                    entropy_floor=0.0,
                    weight_mode="power",
                    mass_trust_k=args.mass_trust_k,
                    normalize_weights=False,
                )
            else:
                train_residual_ppl = float("nan")
                val_residual_ppl = float("nan")
            row = ResidualPriorRow(
                epoch=epoch,
                train_prior_ppl=train_prior_ppl,
                train_residual_ppl=train_residual_ppl,
                val_prior_ppl=val_prior_ppl,
                val_residual_ppl=val_residual_ppl,
                train_loss=float(loss.item()),
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                runtime_sec=time.perf_counter() - started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(row))
            handle.flush()
            print(
                f"epoch={epoch:03d} train_ppl={train_prior_ppl:.3f}->{train_residual_ppl:.3f} "
                f"val_ppl={val_prior_ppl:.3f}->{val_residual_ppl:.3f} "
                f"runtime_sec={row.runtime_sec:.2f} peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
