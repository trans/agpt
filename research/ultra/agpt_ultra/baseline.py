from __future__ import annotations

from dataclasses import dataclass

import torch

from agpt_ultra.data import CharVocab
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.hybrid import freeze_embeddings
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.objective import trie_negative_log_likelihood


@dataclass(frozen=True)
class BaselineEpochResult:
    before_loss: float
    after_loss: float


def baseline_parameters(model: TinyCharRNN):
    freeze_embeddings(model)
    return [*model.cell.parameters(), *model.head.parameters()]


def baseline_epoch(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float | None = 1.0,
    flat_trie: FlatTrie | None = None,
) -> BaselineEpochResult:
    freeze_embeddings(model)
    if flat_trie is None:
        before_loss = trie_negative_log_likelihood(model, vocab, samples).item()
    else:
        from agpt_ultra.flat_ops import flat_trie_negative_log_likelihood

        before_loss = flat_trie_negative_log_likelihood(model, flat_trie).item()
    optimizer.zero_grad(set_to_none=True)
    if flat_trie is None:
        loss = trie_negative_log_likelihood(model, vocab, samples)
    else:
        loss = flat_trie_negative_log_likelihood(model, flat_trie)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(baseline_parameters(model), max_grad_norm)
    optimizer.step()
    if flat_trie is None:
        after_loss = trie_negative_log_likelihood(model, vocab, samples).item()
    else:
        after_loss = flat_trie_negative_log_likelihood(model, flat_trie).item()
    return BaselineEpochResult(before_loss=before_loss, after_loss=after_loss)


def train_baseline(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    epochs: int,
    lr: float = 1e-3,
    max_grad_norm: float | None = 1.0,
    flat_trie: FlatTrie | None = None,
) -> list[BaselineEpochResult]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    optimizer = torch.optim.AdamW(baseline_parameters(model), lr=lr)
    return [
        baseline_epoch(
            model,
            vocab,
            samples,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            flat_trie=flat_trie,
        )
        for _ in range(epochs)
    ]
