from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from agpt_ultra.data import sorted_samples
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.flat_ops import flat_trie_negative_log_likelihood
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.objective import trie_negative_log_likelihood


@dataclass(frozen=True)
class TextSplit:
    train_text: str
    val_text: str


@dataclass(frozen=True)
class SampleSplit:
    train_samples: list[str]
    val_samples: list[str]


@dataclass(frozen=True)
class LossMetrics:
    loss: float
    tokens: int
    nll_per_token: float
    perplexity: float

    @property
    def bits_per_char(self) -> float:
        return self.nll_per_token / math.log(2.0)


def split_text(text: str, train_fraction: float = 0.9) -> TextSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = int(len(text) * train_fraction)
    return TextSplit(train_text=text[:split_at], val_text=text[split_at:])


def make_sample_split(
    text: str,
    block_size: int,
    stride: int,
    train_fraction: float = 0.9,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> SampleSplit:
    split = split_text(text, train_fraction=train_fraction)
    train_samples = sorted_samples(split.train_text, block_size=block_size, stride=stride)
    val_samples = sorted_samples(split.val_text, block_size=block_size, stride=stride)
    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]
    if max_val_samples is not None:
        val_samples = val_samples[:max_val_samples]
    return SampleSplit(
        train_samples=train_samples,
        val_samples=val_samples,
    )


def count_objective_tokens(samples: list[str] | FlatTrie) -> int:
    if isinstance(samples, FlatTrie):
        return int(samples.transition_counts.sum().item())
    return sum(len(sample) for sample in samples)


@torch.no_grad()
def evaluate_trie_loss(model: TinyCharRNN, vocab, samples: list[str] | FlatTrie) -> LossMetrics:
    if isinstance(samples, FlatTrie):
        loss = flat_trie_negative_log_likelihood(model, samples).item()
    else:
        loss = trie_negative_log_likelihood(model, vocab, samples).item()
    tokens = count_objective_tokens(samples)
    if tokens == 0:
        raise ValueError("cannot evaluate empty sample set")
    nll = loss / tokens
    return LossMetrics(
        loss=loss,
        tokens=tokens,
        nll_per_token=nll,
        perplexity=float(torch.exp(torch.tensor(nll)).item()),
    )


@torch.no_grad()
def evaluate_sequential_text_loss(model: TinyCharRNN, vocab, text: str, chunk_size: int = 1024) -> LossMetrics:
    """Evaluate next-character loss over raw text with continuous recurrent state."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if len(text) < 2:
        raise ValueError("cannot evaluate fewer than two characters")

    device = next(model.parameters()).device
    ids = torch.tensor(vocab.encode(text), dtype=torch.long, device=device)
    state = model.initial_state(1, device)
    total_loss = 0.0
    total_tokens = ids.numel() - 1

    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        logits = []
        for token_id in ids[start:end]:
            step_logits, state = model.step(token_id.view(1), state)
            logits.append(step_logits)
        chunk_logits = torch.cat(logits, dim=0)
        targets = ids[start + 1 : end + 1]
        total_loss += float(F.cross_entropy(chunk_logits, targets, reduction="sum").item())

    nll = total_loss / total_tokens
    return LossMetrics(
        loss=total_loss,
        tokens=int(total_tokens),
        nll_per_token=nll,
        perplexity=float(torch.exp(torch.tensor(nll)).item()),
    )
