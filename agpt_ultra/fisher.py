from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from agpt_ultra.data import CharVocab
from agpt_ultra.trie import PrefixNode


def transition_counts(node: PrefixNode, vocab: CharVocab) -> Tensor:
    counts = torch.zeros(vocab.size, dtype=torch.float32)
    for token, child in node.children.items():
        counts[vocab.stoi[token]] = float(child.count)
    return counts


def empirical_distribution(counts: Tensor) -> Tensor:
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot form a distribution from zero counts")
    return counts / total


def categorical_covariance(probs: Tensor) -> Tensor:
    if probs.ndim != 1:
        raise ValueError("probs must be a rank-1 tensor")
    return torch.diag(probs) - torch.outer(probs, probs)


def empirical_logit_fisher(counts: Tensor) -> Tensor:
    q = empirical_distribution(counts)
    return categorical_covariance(q)


def empirical_embedding_fisher(counts: Tensor, output_weight: Tensor) -> Tensor:
    if output_weight.ndim != 2:
        raise ValueError("output_weight must have shape [vocab_size, hidden_size]")
    if output_weight.shape[0] != counts.shape[0]:
        raise ValueError("output_weight vocab dimension must match counts")
    cov = empirical_logit_fisher(counts)
    return output_weight.T @ cov @ output_weight


def node_empirical_fisher(node: PrefixNode, vocab: CharVocab, output_weight: Tensor) -> Tensor:
    return empirical_embedding_fisher(transition_counts(node, vocab), output_weight)


def entropy(probs: Tensor) -> Tensor:
    nonzero = probs[probs > 0]
    return -(nonzero * nonzero.log()).sum()


@dataclass(frozen=True)
class StateEvidence:
    gradient: Tensor
    fisher: Tensor

    def __add__(self, other: "StateEvidence") -> "StateEvidence":
        return StateEvidence(
            gradient=self.gradient + other.gradient,
            fisher=self.fisher + other.fisher,
        )


def logits_from_hidden(hidden: Tensor, output_weight: Tensor, output_bias: Tensor | None = None) -> Tensor:
    logits = output_weight @ hidden
    if output_bias is not None:
        logits = logits + output_bias
    return logits


def predicted_distribution(
    hidden: Tensor,
    output_weight: Tensor,
    output_bias: Tensor | None = None,
) -> Tensor:
    return torch.softmax(logits_from_hidden(hidden, output_weight, output_bias), dim=0)


def local_state_gradient(predicted: Tensor, target: Tensor, output_weight: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target distributions must have the same shape")
    return output_weight.T @ (predicted - target)


def local_state_fisher(predicted: Tensor, output_weight: Tensor) -> Tensor:
    return output_weight.T @ categorical_covariance(predicted) @ output_weight


def local_state_evidence(
    hidden: Tensor,
    target: Tensor,
    output_weight: Tensor,
    output_bias: Tensor | None = None,
) -> StateEvidence:
    predicted = predicted_distribution(hidden, output_weight, output_bias)
    return StateEvidence(
        gradient=local_state_gradient(predicted, target, output_weight),
        fisher=local_state_fisher(predicted, output_weight),
    )


def node_local_state_evidence(
    node: PrefixNode,
    vocab: CharVocab,
    hidden: Tensor,
    output_weight: Tensor,
    output_bias: Tensor | None = None,
) -> StateEvidence:
    target = empirical_distribution(transition_counts(node, vocab))
    return local_state_evidence(hidden, target, output_weight, output_bias)


def natural_state_step(evidence: StateEvidence, damping: float = 1e-3) -> Tensor:
    if damping < 0:
        raise ValueError("damping must be non-negative")
    hidden_size = evidence.gradient.shape[0]
    damped = evidence.fisher + damping * torch.eye(
        hidden_size,
        dtype=evidence.fisher.dtype,
        device=evidence.fisher.device,
    )
    return -torch.linalg.solve(damped, evidence.gradient)


def reconcile_evidence(local: StateEvidence, children: list[StateEvidence]) -> StateEvidence:
    evidence = local
    for child in children:
        evidence = evidence + child
    return evidence
