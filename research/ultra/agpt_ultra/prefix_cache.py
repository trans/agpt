from __future__ import annotations

import torch
from torch import Tensor

from agpt_ultra.data import CharVocab
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.trie import PrefixNode, PrefixTrie, longest_common_prefix


@torch.no_grad()
def logits_naive(model: TinyCharRNN, vocab: CharVocab, samples: list[str]) -> Tensor:
    ids = torch.tensor([vocab.encode(sample[:-1]) for sample in samples], dtype=torch.long)
    return model(ids)


@torch.no_grad()
def logits_with_prefix_stack(model: TinyCharRNN, vocab: CharVocab, samples: list[str]) -> Tensor:
    if samples != sorted(samples):
        raise ValueError("samples must be sorted lexicographically")

    device = next(model.parameters()).device
    previous = ""
    state_stack = [model.initial_state(1, device)]
    sample_logits: list[Tensor] = []

    for sample in samples:
        input_text = sample[:-1]
        common = longest_common_prefix(previous[:-1], input_text)
        state_stack = state_stack[: common + 1]
        logits = []

        for depth in range(common):
            token_id = torch.tensor([vocab.stoi[input_text[depth]]], dtype=torch.long, device=device)
            logits.append(model.head(state_stack[depth + 1]).squeeze(0))

        state = state_stack[-1]
        for token in input_text[common:]:
            token_id = torch.tensor([vocab.stoi[token]], dtype=torch.long, device=device)
            step_logits, state = model.step(token_id, state)
            state_stack.append(state)
            logits.append(step_logits.squeeze(0))

        sample_logits.append(torch.stack(logits, dim=0))
        previous = sample

    return torch.stack(sample_logits, dim=0)


@torch.no_grad()
def logits_with_prefix_trie(model: TinyCharRNN, vocab: CharVocab, samples: list[str]) -> Tensor:
    if samples != sorted(samples):
        raise ValueError("samples must be sorted lexicographically")

    device = next(model.parameters()).device
    input_texts = [sample[:-1] for sample in samples]
    trie = PrefixTrie.from_samples(input_texts)
    outputs: list[Tensor] = []

    def visit(node: PrefixNode, state: Tensor, logits: list[Tensor]) -> None:
        for _ in range(node.terminal_count):
            outputs.append(torch.stack(logits, dim=0))

        for token in sorted(node.children):
            child = node.children[token]
            token_id = torch.tensor([vocab.stoi[token]], dtype=torch.long, device=device)
            step_logits, next_state = model.step(token_id, state)
            visit(child, next_state, [*logits, step_logits.squeeze(0)])

    visit(trie.root, model.initial_state(1, device), [])
    return torch.stack(outputs, dim=0)
