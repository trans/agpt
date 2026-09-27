from __future__ import annotations

from dataclasses import dataclass

import torch

from agpt_ultra.data import CharVocab
from agpt_ultra.fisher import empirical_distribution, entropy, transition_counts
from agpt_ultra.trie import PrefixTrie


@dataclass(frozen=True)
class TrieStats:
    samples: int
    block_size: int
    nodes: int
    predictive_nodes: int
    singleton_nodes: int
    branching_nodes: int
    max_depth: int
    mean_count: float
    mean_entropy: float
    weighted_mean_entropy: float
    singleton_fraction: float
    branching_fraction: float


def trie_stats(samples: list[str], vocab: CharVocab) -> TrieStats:
    if not samples:
        raise ValueError("cannot compute stats for empty samples")
    block_size = len(samples[0])
    if any(len(sample) != block_size for sample in samples):
        raise ValueError("all samples must have the same block size")

    trie = PrefixTrie.from_samples(samples)
    nodes = trie.nodes()
    predictive = [node for node in nodes if node.children]
    singleton_nodes = sum(1 for node in predictive if node.count == 1)
    branching_nodes = sum(1 for node in predictive if len(node.children) > 1)
    max_depth = max(node.depth for node in nodes)
    counts = torch.tensor([float(node.count) for node in predictive])

    entropies = []
    weights = []
    for node in predictive:
        row = transition_counts(node, vocab)
        q = empirical_distribution(row)
        entropies.append(entropy(q))
        weights.append(row.sum())
    entropy_tensor = torch.stack(entropies) if entropies else torch.zeros(1)
    weight_tensor = torch.stack(weights) if weights else torch.ones(1)

    predictive_count = len(predictive)
    return TrieStats(
        samples=len(samples),
        block_size=block_size,
        nodes=len(nodes),
        predictive_nodes=predictive_count,
        singleton_nodes=singleton_nodes,
        branching_nodes=branching_nodes,
        max_depth=max_depth,
        mean_count=counts.mean().item() if predictive_count else 0.0,
        mean_entropy=entropy_tensor.mean().item(),
        weighted_mean_entropy=((entropy_tensor * weight_tensor).sum() / weight_tensor.sum()).item(),
        singleton_fraction=singleton_nodes / predictive_count if predictive_count else 0.0,
        branching_fraction=branching_nodes / predictive_count if predictive_count else 0.0,
    )
