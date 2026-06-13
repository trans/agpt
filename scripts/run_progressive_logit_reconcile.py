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
from scripts.run_book_state_model import longest_suffix_node, safe_exp, trie_children


@dataclass(frozen=True)
class ProgressiveLogitRow:
    train_base_ppl: float
    train_progressive_ppl: float
    heldout_base_ppl: float
    heldout_progressive_ppl: float
    heldout_progressive_bpc: float
    mean_suffix_depth: float
    root_child_mass: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressive trie model reconciliation with local logit models.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/progressive_logit_reconcile.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--local-lr", type=float, default=1.0)
    parser.add_argument("--local-damping", type=float, default=1.0)
    parser.add_argument("--child-weight", type=float, default=0.1)
    parser.add_argument("--prior-weight", type=float, default=1e-3)
    parser.add_argument("--precision-floor", type=float, default=1e-3)
    parser.add_argument("--eval-max-tokens", type=int, default=None)
    return parser.parse_args()


def base_unigram_logits(counts: torch.Tensor, floor: float) -> torch.Tensor:
    unigram = counts.sum(dim=0)
    probs = (unigram + floor) / (unigram.sum() + floor * counts.shape[1])
    return probs.log()


def local_train_logits(
    logits: torch.Tensor,
    row_counts: torch.Tensor,
    local_steps: int,
    local_lr: float,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = row_counts.sum()
    if total <= 0:
        return logits, torch.zeros_like(logits)
    target = row_counts / total
    trained = logits.clone()
    fisher_diag = total * (target * (1.0 - target)).clamp_min(0.0)
    for _ in range(local_steps):
        predicted = torch.softmax(trained, dim=0)
        gradient = total * (predicted - target)
        trained = trained - local_lr * gradient / (fisher_diag + damping)
    return trained, fisher_diag


def progressive_reconcile_logits(
    trie: FlatTrie,
    counts: torch.Tensor,
    local_steps: int,
    local_lr: float,
    local_damping: float,
    child_weight: float,
    prior_weight: float,
    precision_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_count, vocab_size = counts.shape
    base_logits = base_unigram_logits(counts, precision_floor)
    child_precision = torch.zeros((node_count, vocab_size), dtype=counts.dtype)
    child_information = torch.zeros((node_count, vocab_size), dtype=counts.dtype)
    reconciled_logits = torch.empty((node_count, vocab_size), dtype=counts.dtype)
    reconciled_precision = torch.empty((node_count, vocab_size), dtype=counts.dtype)

    prior_precision = torch.full((vocab_size,), prior_weight, dtype=counts.dtype)
    prior_information = prior_precision * base_logits

    for node_id in range(node_count - 1, -1, -1):
        precision = prior_precision + child_weight * child_precision[node_id]
        information = prior_information + child_weight * child_information[node_id]
        incoming_logits = information / precision.clamp_min(1e-30)
        trained_logits, local_precision = local_train_logits(
            incoming_logits,
            counts[node_id],
            local_steps=local_steps,
            local_lr=local_lr,
            damping=local_damping,
        )
        precision = precision + local_precision + precision_floor
        information = precision * trained_logits
        reconciled_logits[node_id] = trained_logits
        reconciled_precision[node_id] = precision
        if node_id > 0:
            parent = int(trie.parents[node_id].item())
            child_precision[parent] += precision
            child_information[parent] += information

    return reconciled_logits, reconciled_precision


def trie_nll_from_logits(counts: torch.Tensor, logits: torch.Tensor) -> float:
    return float(-(counts * F.log_softmax(logits, dim=1)).sum().item())


@torch.no_grad()
def evaluate_heldout(
    logits: torch.Tensor,
    trie: FlatTrie,
    text: str,
    vocab: CharVocab,
    children: list[dict[int, int]],
    max_context: int,
    max_tokens: int | None,
) -> tuple[float, float, float]:
    ids = vocab.encode(text)
    total_tokens = max(0, len(ids) - 1)
    if max_tokens is not None:
        total_tokens = min(total_tokens, max_tokens)
    if total_tokens == 0:
        raise ValueError("heldout text is too short")
    log_probs = F.log_softmax(logits, dim=1)
    loss = 0.0
    depth_sum = 0
    for index in range(total_tokens):
        context = ids[max(0, index + 1 - max_context) : index + 1]
        node_id = longest_suffix_node(context, children)
        target = ids[index + 1]
        loss -= float(log_probs[node_id, target].item())
        depth_sum += int(trie.depths[node_id].item())
    nll = loss / total_tokens
    return safe_exp(nll), nll / math.log(2.0), depth_sum / total_tokens


def main() -> None:
    args = parse_args()
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    started = time.perf_counter()

    text = read_text(args.input)
    split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    samples = make_circular_samples(split.train_text, block_size=args.block_size, stride=args.stride)
    samples.sort()
    trie = samples_to_flat_trie(samples, vocab)
    children = trie_children(trie)
    counts = trie.transition_counts.float()
    mass = float(counts.sum().item())

    base_logits = base_unigram_logits(counts, args.precision_floor).expand(counts.shape[0], counts.shape[1])
    progressive_logits, progressive_precision = progressive_reconcile_logits(
        trie,
        counts,
        local_steps=args.local_steps,
        local_lr=args.local_lr,
        local_damping=args.local_damping,
        child_weight=args.child_weight,
        prior_weight=args.prior_weight,
        precision_floor=args.precision_floor,
    )
    train_base_nll = trie_nll_from_logits(counts, base_logits)
    train_progressive_nll = trie_nll_from_logits(counts, progressive_logits)
    heldout_base_ppl, _, _ = evaluate_heldout(
        base_logits,
        trie,
        split.val_text,
        vocab,
        children,
        max_context=args.block_size,
        max_tokens=args.eval_max_tokens,
    )
    heldout_progressive_ppl, heldout_bpc, mean_suffix_depth = evaluate_heldout(
        progressive_logits,
        trie,
        split.val_text,
        vocab,
        children,
        max_context=args.block_size,
        max_tokens=args.eval_max_tokens,
    )

    row = ProgressiveLogitRow(
        train_base_ppl=safe_exp(train_base_nll / mass),
        train_progressive_ppl=safe_exp(train_progressive_nll / mass),
        heldout_base_ppl=heldout_base_ppl,
        heldout_progressive_ppl=heldout_progressive_ppl,
        heldout_progressive_bpc=heldout_bpc,
        mean_suffix_depth=mean_suffix_depth,
        root_child_mass=float(progressive_precision[0].mean().item()),
        runtime_sec=time.perf_counter() - started,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))

    print(
        f"samples={len(samples)} nodes={trie.node_count} mass={int(mass)} "
        f"train_ppl base={row.train_base_ppl:.3f} progressive={row.train_progressive_ppl:.3f} "
        f"heldout_ppl base={row.heldout_base_ppl:.3f} progressive={row.heldout_progressive_ppl:.3f} "
        f"mean_suffix_depth={row.mean_suffix_depth:.2f} runtime_sec={row.runtime_sec:.2f} "
        f"peak_rss_mb={row.peak_rss_kb / 1024:.1f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
