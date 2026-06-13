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
class SharedBasisAdapterRow:
    epoch: int
    train_base_ppl: float
    train_adapter_ppl: float
    heldout_base_ppl: float
    heldout_adapter_ppl: float
    heldout_adapter_bpc: float
    mean_suffix_depth: float
    runtime_sec: float
    peak_rss_kb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressive reconciliation of shared-basis head adapters.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/shared_basis_adapter_reconcile.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--local-lr", type=float, default=1.0)
    parser.add_argument("--local-damping", type=float, default=1.0)
    parser.add_argument("--child-weight", type=float, default=0.001)
    parser.add_argument("--prior-weight", type=float, default=1e-3)
    parser.add_argument("--precision-floor", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--eval-max-tokens", type=int, default=None)
    return parser.parse_args()


def compute_additive_states(trie: FlatTrie, token_embed: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    states = root.new_zeros((trie.node_count, root.numel()))
    states[0] = root
    parents = trie.parents
    tokens = trie.tokens
    depths = trie.depths
    for depth in range(1, int(depths.max().item()) + 1):
        node_ids = torch.nonzero(depths == depth, as_tuple=False).flatten()
        states[node_ids] = states[parents[node_ids]] + token_embed[tokens[node_ids]]
    return states


def adapter_logits(base_logits: torch.Tensor, adapter: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return base_logits + (adapter * features[None, :]).sum(dim=1)


def progressive_adapter_reconcile(
    trie: FlatTrie,
    counts: torch.Tensor,
    base_logits: torch.Tensor,
    features: torch.Tensor,
    local_steps: int,
    local_lr: float,
    local_damping: float,
    child_weight: float,
    prior_weight: float,
    precision_floor: float,
    prior_adapter: torch.Tensor | None = None,
) -> torch.Tensor:
    node_count, vocab_size = counts.shape
    rank = features.shape[1]
    child_precision = torch.zeros((node_count, vocab_size, rank), dtype=counts.dtype)
    child_information = torch.zeros_like(child_precision)
    adapters = torch.empty_like(child_precision)
    prior_precision = torch.full((vocab_size, rank), prior_weight, dtype=counts.dtype)
    base_prior_adapter = torch.zeros((vocab_size, rank), dtype=counts.dtype)
    if prior_adapter is not None:
        base_prior_adapter = prior_adapter.to(dtype=counts.dtype)
    prior_information = prior_precision * base_prior_adapter

    for node_id in range(node_count - 1, -1, -1):
        precision = prior_precision + child_weight * child_precision[node_id]
        information = prior_information + child_weight * child_information[node_id]
        adapter = information / precision.clamp_min(1e-30)
        row_counts = counts[node_id]
        total = row_counts.sum()
        if total > 0:
            target = row_counts / total
            phi = features[node_id]
            fisher_diag = total * (target * (1.0 - target)).clamp_min(0.0)[:, None] * phi.square()[None, :]
            for _ in range(local_steps):
                logits = adapter_logits(base_logits, adapter, phi)
                predicted = torch.softmax(logits, dim=0)
                gradient = total * (predicted - target)[:, None] * phi[None, :]
                adapter = adapter - local_lr * gradient / (fisher_diag + local_damping)
            precision = precision + fisher_diag + precision_floor
        adapters[node_id] = adapter
        if node_id > 0:
            parent = int(trie.parents[node_id].item())
            child_precision[parent] += precision
            child_information[parent] += precision * adapter
    return adapters


def nll_from_adapter(
    counts: torch.Tensor,
    base_logits: torch.Tensor,
    adapters: torch.Tensor,
    features: torch.Tensor,
    chunk_size: int,
) -> float:
    total = 0.0
    for start in range(0, counts.shape[0], chunk_size):
        end = min(start + chunk_size, counts.shape[0])
        logits = base_logits[None, :] + (adapters[start:end] * features[start:end, None, :]).sum(dim=2)
        total += float(-(counts[start:end] * F.log_softmax(logits, dim=1)).sum().item())
    return total


@torch.no_grad()
def evaluate_heldout(
    base_logits: torch.Tensor,
    adapters: torch.Tensor,
    features: torch.Tensor,
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
        logits = adapter_logits(base_logits, adapters[node_id], features[node_id])
        target = ids[index + 1]
        loss -= float(F.log_softmax(logits, dim=0)[target].item())
        depth_sum += int(trie.depths[node_id].item())
    nll = loss / total_tokens
    return safe_exp(nll), nll / math.log(2.0), depth_sum / total_tokens


def main() -> None:
    args = parse_args()
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    torch.manual_seed(args.seed)
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

    unigram = counts.sum(dim=0)
    base_probs = (unigram + args.precision_floor) / (unigram.sum() + args.precision_floor * vocab.size)
    base_logits = base_probs.log()
    token_embed = 0.02 * torch.randn(vocab.size, args.dim)
    root = torch.zeros(args.dim)
    basis = torch.randn(args.rank, args.dim) / math.sqrt(args.dim)
    states = compute_additive_states(trie, token_embed, root)
    features = states @ basis.T

    base_logits_all = base_logits.expand(counts.shape[0], vocab.size)
    train_base_nll = float(-(counts * F.log_softmax(base_logits_all, dim=1)).sum().item())
    zero_adapters = torch.zeros((counts.shape[0], vocab.size, args.rank), dtype=counts.dtype)
    heldout_base_ppl, _, mean_suffix_depth = evaluate_heldout(
        base_logits,
        zero_adapters,
        features,
        trie,
        split.val_text,
        vocab,
        children,
        max_context=args.block_size,
        max_tokens=args.eval_max_tokens,
    )
    rows: list[SharedBasisAdapterRow] = []
    global_adapter = torch.zeros((vocab.size, args.rank), dtype=counts.dtype)
    for epoch in range(1, args.epochs + 1):
        adapters = progressive_adapter_reconcile(
            trie,
            counts,
            base_logits,
            features,
            local_steps=args.local_steps,
            local_lr=args.local_lr,
            local_damping=args.local_damping,
            child_weight=args.child_weight,
            prior_weight=args.prior_weight,
            precision_floor=args.precision_floor,
            prior_adapter=global_adapter,
        )
        global_adapter = adapters[0].detach().clone()
        train_adapter_nll = nll_from_adapter(counts, base_logits, adapters, features, args.chunk_size)
        heldout_adapter_ppl, heldout_bpc, mean_suffix_depth = evaluate_heldout(
            base_logits,
            adapters,
            features,
            trie,
            split.val_text,
            vocab,
            children,
            max_context=args.block_size,
            max_tokens=args.eval_max_tokens,
        )
        row = SharedBasisAdapterRow(
            epoch=epoch,
            train_base_ppl=safe_exp(train_base_nll / mass),
            train_adapter_ppl=safe_exp(train_adapter_nll / mass),
            heldout_base_ppl=heldout_base_ppl,
            heldout_adapter_ppl=heldout_adapter_ppl,
            heldout_adapter_bpc=heldout_bpc,
            mean_suffix_depth=mean_suffix_depth,
            runtime_sec=time.perf_counter() - started,
            peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        rows.append(row)
        print(
            f"epoch={epoch:03d} train_ppl base={row.train_base_ppl:.3f} adapter={row.train_adapter_ppl:.3f} "
            f"heldout_ppl base={row.heldout_base_ppl:.3f} adapter={row.heldout_adapter_ppl:.3f} "
            f"runtime_sec={row.runtime_sec:.2f} peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
            flush=True,
        )

    row = rows[-1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for item in rows:
            writer.writerow(asdict(item))

    print(
        f"samples={len(samples)} nodes={trie.node_count} mass={int(mass)} "
        f"train_ppl base={row.train_base_ppl:.3f} adapter={row.train_adapter_ppl:.3f} "
        f"heldout_ppl base={row.heldout_base_ppl:.3f} adapter={row.heldout_adapter_ppl:.3f} "
        f"mean_suffix_depth={row.mean_suffix_depth:.2f} runtime_sec={row.runtime_sec:.2f} "
        f"peak_rss_mb={row.peak_rss_kb / 1024:.1f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
