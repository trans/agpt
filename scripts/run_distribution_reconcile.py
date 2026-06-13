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
class DistributionReconcileRow:
    train_local_ppl: float
    train_reconciled_ppl: float
    heldout_local_ppl: float
    heldout_reconciled_ppl: float
    heldout_reconciled_bpc: float
    mean_suffix_depth: float
    root_entropy: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distribution-only trie model reconciliation.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/distribution_reconcile.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--child-scale", type=float, default=1.0)
    parser.add_argument("--local-scale", type=float, default=1.0)
    parser.add_argument("--eval-max-tokens", type=int, default=None)
    return parser.parse_args()


def normalize_counts(counts: torch.Tensor, smoothing: float) -> torch.Tensor:
    smoothed = counts + smoothing
    return smoothed / smoothed.sum(dim=1, keepdim=True).clamp_min(1e-30)


def trie_nll_from_probs(counts: torch.Tensor, probs: torch.Tensor) -> float:
    return float(-(counts * probs.clamp_min(1e-30).log()).sum().item())


def reconcile_pseudo_counts(
    transition_counts: torch.Tensor,
    parents: torch.Tensor,
    local_scale: float,
    child_scale: float,
) -> torch.Tensor:
    pseudo = local_scale * transition_counts.clone()
    for node_id in range(parents.numel() - 1, 0, -1):
        parent = int(parents[node_id].item())
        pseudo[parent] += child_scale * pseudo[node_id]
    return pseudo


@torch.no_grad()
def evaluate_heldout(
    probs: torch.Tensor,
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
    loss = 0.0
    depth_sum = 0
    for index in range(total_tokens):
        context = ids[max(0, index + 1 - max_context) : index + 1]
        node_id = longest_suffix_node(context, children)
        target = ids[index + 1]
        loss -= float(probs[node_id, target].clamp_min(1e-30).log().item())
        depth_sum += int(trie.depths[node_id].item())
    nll = loss / total_tokens
    return safe_exp(nll), nll / math.log(2.0), depth_sum / total_tokens


def entropy(probs: torch.Tensor) -> float:
    return float(-(probs * probs.clamp_min(1e-30).log()).sum().item())


def main() -> None:
    args = parse_args()
    if args.block_size < 2:
        raise ValueError("block_size must be at least 2")
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

    local_probs = normalize_counts(counts, args.smoothing)
    pseudo_counts = reconcile_pseudo_counts(
        counts,
        trie.parents,
        local_scale=args.local_scale,
        child_scale=args.child_scale,
    )
    reconciled_probs = normalize_counts(pseudo_counts, args.smoothing)

    train_local_nll = trie_nll_from_probs(counts, local_probs)
    train_reconciled_nll = trie_nll_from_probs(counts, reconciled_probs)
    heldout_local_ppl, _, _ = evaluate_heldout(
        local_probs,
        trie,
        split.val_text,
        vocab,
        children,
        max_context=args.block_size,
        max_tokens=args.eval_max_tokens,
    )
    heldout_reconciled_ppl, heldout_reconciled_bpc, mean_suffix_depth = evaluate_heldout(
        reconciled_probs,
        trie,
        split.val_text,
        vocab,
        children,
        max_context=args.block_size,
        max_tokens=args.eval_max_tokens,
    )

    row = DistributionReconcileRow(
        train_local_ppl=safe_exp(train_local_nll / mass),
        train_reconciled_ppl=safe_exp(train_reconciled_nll / mass),
        heldout_local_ppl=heldout_local_ppl,
        heldout_reconciled_ppl=heldout_reconciled_ppl,
        heldout_reconciled_bpc=heldout_reconciled_bpc,
        mean_suffix_depth=mean_suffix_depth,
        root_entropy=entropy(reconciled_probs[0]),
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
        f"train_ppl local={row.train_local_ppl:.3f} reconciled={row.train_reconciled_ppl:.3f} "
        f"heldout_ppl local={row.heldout_local_ppl:.3f} reconciled={row.heldout_reconciled_ppl:.3f} "
        f"mean_suffix_depth={row.mean_suffix_depth:.2f} runtime_sec={row.runtime_sec:.2f} "
        f"peak_rss_mb={row.peak_rss_kb / 1024:.1f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
