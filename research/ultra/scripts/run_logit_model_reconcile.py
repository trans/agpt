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
class LogitReconcileRow:
    train_local_ppl: float
    train_reconciled_ppl: float
    heldout_local_ppl: float
    heldout_reconciled_ppl: float
    heldout_reconciled_bpc: float
    mean_suffix_depth: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal logit-model reconciliation through a trie.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/logit_model_reconcile.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--local-lr", type=float, default=1.0)
    parser.add_argument("--local-damping", type=float, default=1.0)
    parser.add_argument("--parent-local-weight", type=float, default=1.0)
    parser.add_argument("--child-weight", type=float, default=0.1)
    parser.add_argument("--precision-floor", type=float, default=1e-3)
    parser.add_argument("--eval-max-tokens", type=int, default=None)
    return parser.parse_args()


def local_logit_models(
    counts: torch.Tensor,
    local_steps: int,
    local_lr: float,
    damping: float,
    precision_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    totals = counts.sum(dim=1)
    vocab_size = counts.shape[1]
    unigram = counts.sum(dim=0)
    root_prior = (unigram + precision_floor) / (unigram.sum() + precision_floor * vocab_size)
    logits = root_prior.log().expand(counts.shape[0], vocab_size).clone()
    active = totals > 0
    for _ in range(local_steps):
        predicted = torch.softmax(logits[active], dim=1)
        target = counts[active] / totals[active, None]
        gradient = totals[active, None] * (predicted - target)
        fisher_diag = totals[active, None] * (target * (1.0 - target)).clamp_min(0.0)
        logits[active] = logits[active] - local_lr * gradient / (fisher_diag + damping)

    predicted = torch.softmax(logits, dim=1)
    target = torch.zeros_like(predicted)
    target[active] = counts[active] / totals[active, None]
    precision = totals[:, None] * (target * (1.0 - target)).clamp_min(0.0) + precision_floor
    return logits, precision


def reconcile_logits(
    local_logits: torch.Tensor,
    local_precision: torch.Tensor,
    parents: torch.Tensor,
    parent_local_weight: float,
    child_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconciled_logits = local_logits.clone()
    reconciled_precision = parent_local_weight * local_precision.clone()

    for node_id in range(parents.numel() - 1, 0, -1):
        parent = int(parents[node_id].item())
        child_precision = child_weight * reconciled_precision[node_id]
        total_precision = reconciled_precision[parent] + child_precision
        reconciled_logits[parent] = (
            reconciled_precision[parent] * reconciled_logits[parent]
            + child_precision * reconciled_logits[node_id]
        ) / total_precision.clamp_min(1e-30)
        reconciled_precision[parent] = total_precision

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

    local_logits, local_precision = local_logit_models(
        counts,
        local_steps=args.local_steps,
        local_lr=args.local_lr,
        damping=args.local_damping,
        precision_floor=args.precision_floor,
    )
    reconciled_logits, _ = reconcile_logits(
        local_logits,
        local_precision,
        trie.parents,
        parent_local_weight=args.parent_local_weight,
        child_weight=args.child_weight,
    )

    train_local_nll = trie_nll_from_logits(counts, local_logits)
    train_reconciled_nll = trie_nll_from_logits(counts, reconciled_logits)
    heldout_local_ppl, _, _ = evaluate_heldout(
        local_logits,
        trie,
        split.val_text,
        vocab,
        children,
        max_context=args.block_size,
        max_tokens=args.eval_max_tokens,
    )
    heldout_reconciled_ppl, heldout_bpc, mean_suffix_depth = evaluate_heldout(
        reconciled_logits,
        trie,
        split.val_text,
        vocab,
        children,
        max_context=args.block_size,
        max_tokens=args.eval_max_tokens,
    )

    row = LogitReconcileRow(
        train_local_ppl=safe_exp(train_local_nll / mass),
        train_reconciled_ppl=safe_exp(train_reconciled_nll / mass),
        heldout_local_ppl=heldout_local_ppl,
        heldout_reconciled_ppl=heldout_reconciled_ppl,
        heldout_reconciled_bpc=heldout_bpc,
        mean_suffix_depth=mean_suffix_depth,
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
