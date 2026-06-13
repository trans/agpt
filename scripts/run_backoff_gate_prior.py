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
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, read_text
from agpt_ultra.eval import split_text
from agpt_ultra.run_ids import prefixed_path, resolve_run_id


@dataclass(frozen=True)
class PriorRow:
    epoch: int
    train_ppl: float | None
    val_ppl: float
    root_ppl: float | None
    deepest_ppl: float | None
    uniform_ppl: float | None
    mean_gate: float | None
    theta_r: float | None
    theta_g: float | None
    theta_h: float | None
    theta_l: float | None
    bias: float | None
    runtime_sec: float
    peak_rss_kb: int


@dataclass(frozen=True)
class PriorData:
    node_ids: dict[tuple[int, ...], int]
    log_dists: torch.Tensor
    features: torch.Tensor
    suffix_links: list[int]
    vocab_size: int
    max_depth: int


def encode_text(text: str, vocab: CharVocab, max_chars: int | None) -> list[int]:
    if max_chars is not None:
        text = text[:max_chars]
    return vocab.encode(text)


def build_counts(ids: list[int], vocab_size: int, max_depth: int) -> dict[tuple[int, ...], list[int]]:
    counts: dict[tuple[int, ...], list[int]] = {}
    root = [0] * vocab_size
    counts[()] = root
    for target_pos, token in enumerate(ids):
        root[token] += 1
        start = max(0, target_pos - max_depth)
        context: list[int] = []
        for pos in range(target_pos - 1, start - 1, -1):
            context.append(ids[pos])
            key = tuple(reversed(context))
            row = counts.get(key)
            if row is None:
                row = [0] * vocab_size
                counts[key] = row
            row[token] += 1
    return counts


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(probs.dtype).tiny
    return -(probs * probs.clamp_min(eps).log()).sum()


def build_prior_data(
    ids: list[int],
    vocab_size: int,
    max_depth: int,
    smoothing: float,
    device: torch.device,
) -> PriorData:
    counts_by_context = build_counts(ids, vocab_size, max_depth)
    contexts = sorted(counts_by_context.keys(), key=lambda item: (len(item), item))
    node_ids = {context: index for index, context in enumerate(contexts)}
    count_rows = torch.tensor([counts_by_context[context] for context in contexts], dtype=torch.float32)
    probs = (count_rows + smoothing) / (count_rows.sum(dim=1, keepdim=True) + smoothing * vocab_size)
    log_dists = probs.log()
    suffix_links: list[int] = []
    features = torch.zeros((len(contexts), 4), dtype=torch.float32)
    log_vocab = math.log(vocab_size)
    for context, node_id in node_ids.items():
        suffix = context[1:]
        while suffix and suffix not in node_ids:
            suffix = suffix[1:]
        suffix_id = node_ids.get(suffix, 0)
        suffix_links.append(suffix_id)
        counts = count_rows[node_id]
        mass = float(counts.sum().item())
        branching = float((counts > 0).sum().item())
        reliability = mass / (mass + branching) if mass + branching > 0 else 0.0
        entropy = float(entropy_from_probs(probs[node_id]).item()) / log_vocab
        back_probs = probs[suffix_id]
        kl = float((probs[node_id] * (log_dists[node_id] - back_probs.log())).sum().item())
        depth = len(context) / max(1, max_depth)
        features[node_id] = torch.tensor(
            [reliability, math.log1p(max(0.0, kl)), entropy, depth],
            dtype=torch.float32,
        )
    return PriorData(
        node_ids=node_ids,
        log_dists=log_dists.to(device),
        features=features.to(device),
        suffix_links=suffix_links,
        vocab_size=vocab_size,
        max_depth=max_depth,
    )


def longest_context_node(ids: list[int], pos: int, prior: PriorData) -> int:
    max_depth = min(prior.max_depth, pos)
    for depth in range(max_depth, 0, -1):
        key = tuple(ids[pos - depth : pos])
        node_id = prior.node_ids.get(key)
        if node_id is not None:
            return node_id
    return 0


def node_chain(node_id: int, prior: PriorData) -> list[int]:
    chain: list[int] = []
    seen = set()
    while node_id != 0 and node_id not in seen:
        seen.add(node_id)
        chain.append(node_id)
        node_id = prior.suffix_links[node_id]
    return chain


def make_examples(ids: list[int], prior: PriorData, max_examples: int | None) -> tuple[list[list[int]], torch.Tensor]:
    limit = len(ids) if max_examples is None else min(len(ids), max_examples)
    chains = [node_chain(longest_context_node(ids, pos, prior), prior) for pos in range(limit)]
    targets = torch.tensor(ids[:limit], dtype=torch.long, device=prior.log_dists.device)
    return chains, targets


class BackoffGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(features @ self.weight + self.bias)


def entropy_damper(features: torch.Tensor, mode: str, alpha: float) -> torch.Tensor:
    entropy = features[:, 2].clamp(0.0, 1.0)
    if mode == "none":
        return torch.ones_like(entropy)
    if mode == "entropy":
        return entropy.clamp_min(1e-6).pow(alpha)
    if mode == "middle":
        middle = (4.0 * entropy * (1.0 - entropy)).clamp_min(1e-6)
        return middle.pow(alpha)
    raise ValueError(f"unknown entropy_damper: {mode}")


def logits_for_chain(
    chain: list[int],
    prior: PriorData,
    gate: BackoffGate | None,
    mode: str,
    damper_mode: str,
    damper_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    logits = prior.log_dists[0].clone()
    if not chain:
        return logits, None
    node_tensor = torch.tensor(chain, dtype=torch.long, device=prior.log_dists.device)
    log_dists = prior.log_dists[node_tensor]
    damping = entropy_damper(prior.features[node_tensor], damper_mode, damper_alpha)
    if mode == "deepest":
        logits = logits + damping[:1].squeeze(0) * log_dists[:1].squeeze(0)
        return logits, None
    if mode == "uniform":
        logits = logits + (damping[:, None] * log_dists).sum(dim=0)
        return logits, None
    if mode == "gate":
        if gate is None:
            raise ValueError("gate mode requires gate")
        weights = gate(prior.features[node_tensor]) * damping
        logits = logits + (weights[:, None] * log_dists).sum(dim=0)
        return logits, weights.mean()
    raise ValueError(f"unknown mode: {mode}")


def evaluate_mode(
    chains: list[list[int]],
    targets: torch.Tensor,
    prior: PriorData,
    mode: str,
    gate: BackoffGate | None = None,
    damper_mode: str = "none",
    damper_alpha: float = 1.0,
) -> tuple[float, float | None]:
    total_loss = 0.0
    total_gate = 0.0
    gate_count = 0
    with torch.no_grad():
        for chain, target in zip(chains, targets, strict=True):
            logits, gate_mean = logits_for_chain(chain, prior, gate, mode, damper_mode, damper_alpha)
            total_loss += float(F.cross_entropy(logits.view(1, -1), target.view(1), reduction="sum").item())
            if gate_mean is not None:
                total_gate += float(gate_mean.item())
                gate_count += 1
    nll = total_loss / max(1, len(chains))
    mean_gate = total_gate / gate_count if gate_count > 0 else None
    return math.exp(nll), mean_gate


def train_epoch(
    chains: list[list[int]],
    targets: torch.Tensor,
    prior: PriorData,
    gate: BackoffGate,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    damper_mode: str,
    damper_alpha: float,
) -> tuple[float, float]:
    total_loss = 0.0
    total_gate = 0.0
    gate_count = 0
    order = torch.randperm(len(chains)).tolist()
    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        gate_means: list[torch.Tensor] = []
        for index in batch_indices:
            logits, gate_mean = logits_for_chain(
                chains[index],
                prior,
                gate,
                "gate",
                damper_mode,
                damper_alpha,
            )
            losses.append(F.cross_entropy(logits.view(1, -1), targets[index].view(1), reduction="sum"))
            if gate_mean is not None:
                gate_means.append(gate_mean)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch_indices)
        if gate_means:
            total_gate += float(torch.stack(gate_means).mean().detach().item()) * len(batch_indices)
            gate_count += len(batch_indices)
    nll = total_loss / max(1, len(chains))
    return math.exp(nll), total_gate / gate_count if gate_count > 0 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny gated product-of-experts backoff prior.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/backoff_gate_prior.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-train-chars", type=int, default=200000)
    parser.add_argument("--eval-max-chars", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--entropy-damper", choices=["none", "entropy", "middle"], default="none")
    parser.add_argument("--damper-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    text = read_text(args.input)
    split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    train_ids = encode_text(split.train_text, vocab, args.max_train_chars)
    val_ids = encode_text(split.val_text, vocab, args.eval_max_chars)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = build_prior_data(train_ids, vocab.size, args.max_depth, args.smoothing, device)
    train_chains, train_targets = make_examples(train_ids, prior, None)
    val_chains, val_targets = make_examples(val_ids, prior, None)
    gate = BackoffGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(PriorRow(0, None, 0.0, None, None, None, None, None, None, None, None, None, 0.0, 0)).keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        root_ppl, _ = evaluate_mode(val_chains, val_targets, prior, "uniform", None)
        # Root baseline is just empty-chain gate mode by evaluating root logits directly.
        root_loss = F.cross_entropy(
            prior.log_dists[0].expand(val_targets.numel(), -1),
            val_targets,
            reduction="mean",
        )
        root_ppl = float(torch.exp(root_loss).item())
        deepest_ppl, _ = evaluate_mode(
            val_chains,
            val_targets,
            prior,
            "deepest",
            None,
            args.entropy_damper,
            args.damper_alpha,
        )
        uniform_ppl, _ = evaluate_mode(
            val_chains,
            val_targets,
            prior,
            "uniform",
            None,
            args.entropy_damper,
            args.damper_alpha,
        )
        val_ppl, mean_gate = evaluate_mode(
            val_chains,
            val_targets,
            prior,
            "gate",
            gate,
            args.entropy_damper,
            args.damper_alpha,
        )
        row = PriorRow(
            epoch=0,
            train_ppl=None,
            val_ppl=val_ppl,
            root_ppl=root_ppl,
            deepest_ppl=deepest_ppl,
            uniform_ppl=uniform_ppl,
            mean_gate=mean_gate,
            theta_r=float(gate.weight[0].detach().item()),
            theta_g=float(gate.weight[1].detach().item()),
            theta_h=float(gate.weight[2].detach().item()),
            theta_l=float(gate.weight[3].detach().item()),
            bias=float(gate.bias.detach().item()),
            runtime_sec=time.perf_counter() - started,
            peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        writer.writerow(asdict(row))
        handle.flush()
        print(
            f"nodes={len(prior.node_ids)} train_tokens={len(train_targets)} val_tokens={len(val_targets)} "
            f"damper={args.entropy_damper} alpha={args.damper_alpha} "
            f"root_ppl={root_ppl:.3f} deepest_ppl={deepest_ppl:.3f} uniform_ppl={uniform_ppl:.3f} "
            f"init_gate_ppl={val_ppl:.3f} output={args.output}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            train_ppl, train_gate = train_epoch(
                train_chains,
                train_targets,
                prior,
                gate,
                optimizer,
                args.batch_size,
                args.entropy_damper,
                args.damper_alpha,
            )
            val_ppl, mean_gate = evaluate_mode(
                val_chains,
                val_targets,
                prior,
                "gate",
                gate,
                args.entropy_damper,
                args.damper_alpha,
            )
            row = PriorRow(
                epoch=epoch,
                train_ppl=train_ppl,
                val_ppl=val_ppl,
                root_ppl=root_ppl,
                deepest_ppl=deepest_ppl,
                uniform_ppl=uniform_ppl,
                mean_gate=mean_gate,
                theta_r=float(gate.weight[0].detach().item()),
                theta_g=float(gate.weight[1].detach().item()),
                theta_h=float(gate.weight[2].detach().item()),
                theta_l=float(gate.weight[3].detach().item()),
                bias=float(gate.bias.detach().item()),
                runtime_sec=time.perf_counter() - started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            writer.writerow(asdict(row))
            handle.flush()
            print(
                f"epoch={epoch} train_ppl={train_ppl:.3f} val_ppl={val_ppl:.3f} "
                f"mean_gate={mean_gate if mean_gate is not None else float('nan'):.4f} "
                f"theta={[round(float(v), 4) for v in gate.weight.detach().cpu().tolist()]} "
                f"bias={float(gate.bias.detach().item()):.4f} runtime_sec={row.runtime_sec:.2f} "
                f"peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
