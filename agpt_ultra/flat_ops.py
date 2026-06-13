from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from agpt_ultra.data import CharVocab
from agpt_ultra.flat_trie import FlatTrie, flat_trie_from_samples
from agpt_ultra.head_only import (
    BatchedHeadEvidence,
    HeadFisherFactor,
    MatrixFreeHeadEvidence,
    flatten_head,
    head_shape,
    model_head_theta,
    node_head_factor,
)
from agpt_ultra.model import TinyCharRNN


def _coerce_initial_hidden(model: TinyCharRNN, node_count: int, device: torch.device, dtype: torch.dtype, initial_hidden: Tensor | None) -> Tensor:
    if initial_hidden is None:
        return model.initial_state(node_count, device).to(dtype=dtype)
    root_hidden = initial_hidden.to(device=device, dtype=dtype)
    if root_hidden.dim() == 2:
        if root_hidden.shape[0] != 1:
            raise ValueError("initial_hidden batch dimension must be 1")
        root_hidden = root_hidden.squeeze(0)
    if root_hidden.dim() != 1:
        raise ValueError("initial_hidden must be a hidden vector or a single-row hidden batch")
    hidden = root_hidden.new_zeros((node_count, root_hidden.numel()))
    hidden[0] = root_hidden
    return hidden


def compute_flat_hidden_states_sequential(model: TinyCharRNN, trie: FlatTrie, initial_hidden: Tensor | None = None) -> Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden = _coerce_initial_hidden(model, trie.node_count, device, dtype, initial_hidden)
    hidden_states = [hidden[0]]
    for node_id in range(1, trie.node_count):
        parent_id = int(trie.parents[node_id].item())
        token_id = torch.tensor([int(trie.tokens[node_id].item())], dtype=torch.long, device=device)
        parent_hidden = hidden_states[parent_id].unsqueeze(0)
        _, child_hidden = model.step(token_id, parent_hidden)
        hidden_states.append(child_hidden.squeeze(0))
    return torch.stack(hidden_states, dim=0)


def compute_flat_hidden_states(model: TinyCharRNN, trie: FlatTrie, initial_hidden: Tensor | None = None) -> Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden = _coerce_initial_hidden(model, trie.node_count, device, dtype, initial_hidden)
    parents = trie.parents.to(device)
    tokens = trie.tokens.to(device)
    depths = trie.depths.to(device)
    max_depth = int(trie.depths.max().item())

    for depth in range(1, max_depth + 1):
        node_ids = torch.nonzero(depths == depth, as_tuple=False).flatten()
        if node_ids.numel() == 0:
            continue
        parent_hidden = hidden[parents[node_ids]]
        token_embeddings = model.embed(tokens[node_ids])
        hidden[node_ids] = model.cell(token_embeddings, parent_hidden)

    return hidden


def flat_trie_negative_log_likelihood_from_hidden(model: TinyCharRNN, trie: FlatTrie, hidden: Tensor) -> Tensor:
    device = hidden.device
    counts = trie.transition_counts.to(device=device, dtype=hidden.dtype)
    logits = model.head(hidden)
    log_probs = F.log_softmax(logits, dim=1)
    return -(counts * log_probs).sum()


def flat_trie_negative_log_likelihood(model: TinyCharRNN, trie: FlatTrie, initial_hidden: Tensor | None = None) -> Tensor:
    hidden = compute_flat_hidden_states(model, trie, initial_hidden=initial_hidden)
    return flat_trie_negative_log_likelihood_from_hidden(model, trie, hidden)


@torch.no_grad()
def collect_flat_matrix_free_head_evidence(
    model: TinyCharRNN,
    trie: FlatTrie,
    theta: Tensor | None = None,
) -> MatrixFreeHeadEvidence:
    if theta is None:
        theta = model_head_theta(model)
    theta = theta.detach().clone()
    shape = head_shape(model)
    device = next(model.parameters()).device
    hidden = compute_flat_hidden_states(model, trie)
    counts = trie.transition_counts.to(device=device, dtype=theta.dtype)

    gradient = torch.zeros_like(theta)
    factors: list[HeadFisherFactor] = []
    for node_id in range(trie.node_count):
        node_gradient, factor = node_head_factor(theta, shape, hidden[node_id], counts[node_id])
        gradient = gradient + node_gradient
        if factor is not None:
            factors.append(factor)

    return MatrixFreeHeadEvidence(gradient=gradient, factors=factors, shape=shape)


@torch.no_grad()
def collect_batched_flat_head_evidence(
    model: TinyCharRNN,
    trie: FlatTrie,
    theta: Tensor | None = None,
    initial_hidden: Tensor | None = None,
) -> BatchedHeadEvidence:
    if theta is None:
        theta = model_head_theta(model)
    theta = theta.detach().clone()
    shape = head_shape(model)
    hidden = compute_flat_hidden_states(model, trie, initial_hidden=initial_hidden).to(dtype=theta.dtype)
    return collect_batched_flat_head_evidence_from_hidden(model, trie, hidden, theta)


@torch.no_grad()
def collect_batched_flat_head_evidence_from_hidden(
    model: TinyCharRNN,
    trie: FlatTrie,
    hidden: Tensor,
    theta: Tensor | None = None,
) -> BatchedHeadEvidence:
    if theta is None:
        theta = model_head_theta(model)
    theta = theta.detach().clone()
    shape = head_shape(model)
    device = hidden.device
    hidden = hidden.to(dtype=theta.dtype, device=device)
    counts = trie.transition_counts.to(device=device, dtype=theta.dtype)
    totals = counts.sum(dim=1)
    mask = totals > 0
    hidden = hidden[mask]
    counts = counts[mask]
    totals = totals[mask]

    weight, bias = theta[: shape.vocab_size * shape.hidden_size].reshape(
        shape.vocab_size,
        shape.hidden_size,
    ), theta[shape.vocab_size * shape.hidden_size :]
    logits = hidden @ weight.T + bias
    predicted = torch.softmax(logits, dim=1)
    target = counts / totals[:, None]
    residual = predicted - target
    weight_gradient = (totals[:, None] * residual).T @ hidden
    bias_gradient = (totals[:, None] * residual).sum(dim=0)
    features = torch.cat([hidden, hidden.new_ones((hidden.shape[0], 1))], dim=1)
    return BatchedHeadEvidence(
        gradient=flatten_head(weight_gradient, bias_gradient),
        features=features.detach().clone(),
        predicted=predicted.detach().clone(),
        counts=totals.detach().clone(),
        shape=shape,
    )


@torch.no_grad()
def evaluate_flat_trie_loss(model: TinyCharRNN, trie: FlatTrie):
    loss = flat_trie_negative_log_likelihood(model, trie).item()
    tokens = int(trie.transition_counts.sum().item())
    nll = loss / tokens
    return loss, tokens, nll, float(torch.exp(torch.tensor(nll)).item())


def samples_to_flat_trie(samples: list[str], vocab: CharVocab) -> FlatTrie:
    return flat_trie_from_samples(samples, vocab)
