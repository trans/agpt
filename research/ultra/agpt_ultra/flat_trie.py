from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from agpt_ultra.data import CharVocab


@dataclass(frozen=True)
class FlatTrie:
    vocab_size: int
    parents: Tensor
    tokens: Tensor
    depths: Tensor
    counts: Tensor
    transition_counts: Tensor

    @property
    def node_count(self) -> int:
        return int(self.parents.numel())

    @property
    def predictive_node_mask(self) -> Tensor:
        return self.transition_counts.sum(dim=1) > 0


def build_flat_trie(encoded_samples: list[list[int]], vocab_size: int) -> FlatTrie:
    if not encoded_samples:
        raise ValueError("cannot build a flat trie from no samples")
    block_size = len(encoded_samples[0])
    if block_size < 1:
        raise ValueError("samples must not be empty")
    if any(len(sample) != block_size for sample in encoded_samples):
        raise ValueError("all samples must have the same length")

    parents = [-1]
    tokens = [-1]
    depths = [0]
    counts = [0]
    transition_rows = [[0 for _ in range(vocab_size)]]
    children: list[dict[int, int]] = [{}]

    for sample in encoded_samples:
        node_id = 0
        counts[node_id] += 1
        for token in sample:
            transition_rows[node_id][token] += 1
            child_id = children[node_id].get(token)
            if child_id is None:
                child_id = len(parents)
                children[node_id][token] = child_id
                parents.append(node_id)
                tokens.append(token)
                depths.append(depths[node_id] + 1)
                counts.append(0)
                transition_rows.append([0 for _ in range(vocab_size)])
                children.append({})
            node_id = child_id
            counts[node_id] += 1

    return FlatTrie(
        vocab_size=vocab_size,
        parents=torch.tensor(parents, dtype=torch.long),
        tokens=torch.tensor(tokens, dtype=torch.long),
        depths=torch.tensor(depths, dtype=torch.long),
        counts=torch.tensor(counts, dtype=torch.float32),
        transition_counts=torch.tensor(transition_rows, dtype=torch.float32),
    )


def flat_trie_from_samples(samples: list[str], vocab: CharVocab) -> FlatTrie:
    encoded = [vocab.encode(sample) for sample in sorted(samples)]
    return build_flat_trie(encoded, vocab.size)
