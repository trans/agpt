from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from agpt_ultra.data import CharVocab
from agpt_ultra.fisher import transition_counts
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.trie import PrefixNode, PrefixTrie


@dataclass(frozen=True)
class NodeObjective:
    prefix: str
    count: int
    counts: Tensor
    hidden: Tensor
    logits: Tensor
    loss: Tensor


def immutable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: param.detach().clone() for name, param in model.state_dict().items()}


def assert_state_unchanged(before: dict[str, Tensor], model: nn.Module) -> None:
    after = model.state_dict()
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected)


def node_transition_loss(logits: Tensor, counts: Tensor) -> Tensor:
    total = counts.sum()
    if total <= 0:
        return logits.new_zeros(())
    log_probs = F.log_softmax(logits, dim=0)
    return -(counts * log_probs).sum()


def trie_node_objectives(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
) -> list[NodeObjective]:
    trie = PrefixTrie.from_samples(samples)
    device = next(model.parameters()).device
    objectives: list[NodeObjective] = []

    def visit(node: PrefixNode, prefix: str, hidden: Tensor) -> None:
        counts = transition_counts(node, vocab).to(device)
        logits = model.head(hidden.squeeze(0))
        if counts.sum() > 0:
            objectives.append(
                NodeObjective(
                    prefix=prefix,
                    count=node.count,
                    counts=counts,
                    hidden=hidden.squeeze(0).clone(),
                    logits=logits.clone(),
                    loss=node_transition_loss(logits, counts),
                )
            )

        for token in sorted(node.children):
            child = node.children[token]
            token_id = torch.tensor([vocab.stoi[token]], dtype=torch.long, device=device)
            _, child_hidden = model.step(token_id, hidden)
            visit(child, prefix + token, child_hidden)

    visit(trie.root, "", model.initial_state(1, device))
    return objectives


def trie_negative_log_likelihood(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
) -> Tensor:
    objectives = trie_node_objectives(model, vocab, samples)
    if not objectives:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack([objective.loss for objective in objectives]).sum()


def naive_transition_negative_log_likelihood(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
) -> Tensor:
    if not samples:
        return torch.zeros((), device=next(model.parameters()).device)
    ids = torch.tensor([vocab.encode(sample) for sample in samples], dtype=torch.long)
    batch = ids.shape[0]
    root_hidden = model.initial_state(batch, ids.device)
    root_logits = model.head(root_hidden).unsqueeze(1)
    suffix_logits = model(ids[:, :-1])
    logits = torch.cat([root_logits, suffix_logits], dim=1)
    targets = ids
    return F.cross_entropy(
        logits.reshape(-1, vocab.size),
        targets.reshape(-1),
        reduction="sum",
    )
