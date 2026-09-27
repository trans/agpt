from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import torch

from agpt_ultra.body_fisher import BodyCurvature, BodyFisherStepStats, body_fisher_step
from agpt_ultra.data import CharVocab
from agpt_ultra.embedding_fisher import EmbeddingFisherPreconditioner, EmbeddingFisherStepStats
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.head_only import HeadOnlyEpochResult, head_only_epoch
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.objective import trie_negative_log_likelihood


@dataclass(frozen=True)
class HybridEpochResult:
    before_loss: float | None
    after_body_loss: float
    after_loss: float
    head_result: HeadOnlyEpochResult
    before_loss_runtime_sec: float = 0.0
    body_runtime_sec: float = 0.0
    hidden_cache_runtime_sec: float = 0.0
    head_runtime_sec: float = 0.0
    embedding_fisher_stats: EmbeddingFisherStepStats | None = None
    body_fisher_stats: BodyFisherStepStats | None = None


def freeze_embeddings(model: TinyCharRNN) -> None:
    for param in model.embed.parameters():
        param.requires_grad_(False)


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> list[bool]:
    previous = [param.requires_grad for param in module.parameters()]
    for param in module.parameters():
        param.requires_grad_(requires_grad)
    return previous


def _restore_requires_grad(module: torch.nn.Module, previous: list[bool]) -> None:
    for param, requires_grad in zip(module.parameters(), previous, strict=True):
        param.requires_grad_(requires_grad)


def prefix_hidden_state(model: TinyCharRNN, vocab: CharVocab, prefix: str) -> torch.Tensor | None:
    if not prefix:
        return None
    device = next(model.parameters()).device
    state = model.initial_state(1, device)
    for token_id in vocab.encode(prefix):
        token = torch.tensor([token_id], dtype=torch.long, device=device)
        _, state = model.step(token, state)
    return state.squeeze(0)


def body_gradient_step(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float | None = 1.0,
    flat_trie: FlatTrie | None = None,
    flat_prefix: str = "",
    update_embeddings: bool = False,
    embedding_fisher: EmbeddingFisherPreconditioner | None = None,
) -> tuple[float, EmbeddingFisherStepStats | None]:
    if not update_embeddings:
        freeze_embeddings(model)
    head_requires_grad = _set_requires_grad(model.head, False)
    try:
        optimizer.zero_grad(set_to_none=True)
        if flat_trie is None:
            loss = trie_negative_log_likelihood(model, vocab, samples)
        else:
            from agpt_ultra.flat_ops import flat_trie_negative_log_likelihood

            loss = flat_trie_negative_log_likelihood(
                model,
                flat_trie,
                initial_hidden=prefix_hidden_state(model, vocab, flat_prefix),
            )
        loss.backward()
        if max_grad_norm is not None:
            params = [*model.cell.parameters()]
            if update_embeddings and embedding_fisher is None:
                params.extend(model.embed.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        embedding_fisher_stats = None
        if embedding_fisher is not None:
            embedding_fisher_stats = embedding_fisher.step(model.embed)
        optimizer.step()
        return loss.item(), embedding_fisher_stats
    finally:
        _restore_requires_grad(model.head, head_requires_grad)


def body_natural_step(
    model: TinyCharRNN,
    vocab: CharVocab,
    flat_trie: FlatTrie,
    flat_prefix: str = "",
    update_embeddings: bool = False,
    damping: float = 100.0,
    step_scale: float | Literal["auto"] = 1.0,
    max_step_scale: float = 1.0,
    trust_radius: float | None = None,
    max_cg_iter: int = 8,
    cg_tolerance: float = 1e-6,
    curvature: BodyCurvature = "model",
) -> BodyFisherStepStats:
    if not update_embeddings:
        freeze_embeddings(model)
    prefix_token_ids = tuple(vocab.encode(flat_prefix))
    return body_fisher_step(
        model,
        flat_trie,
        prefix_token_ids=prefix_token_ids,
        update_embeddings=update_embeddings,
        damping=damping,
        step_scale=step_scale,
        max_step_scale=max_step_scale,
        trust_radius=trust_radius,
        max_cg_iter=max_cg_iter,
        cg_tolerance=cg_tolerance,
        curvature=curvature,
    )


def hybrid_epoch(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    body_optimizer: torch.optim.Optimizer,
    head_damping: float = 1.0,
    head_step_scale: float | Literal["auto"] = 0.5,
    max_grad_norm: float | None = 1.0,
    max_cg_iter: int | None = None,
    cg_tolerance: float = 1e-6,
    flat_trie: FlatTrie | None = None,
    flat_prefix: str = "",
    max_head_step_scale: float = 1.0,
    line_search_steps: int = 0,
    line_search_flat_trie: FlatTrie | None = None,
    calibration_flat_trie: FlatTrie | None = None,
    trust_radius: float | None = None,
    compute_before_loss: bool = True,
    update_embeddings: bool = False,
    embedding_fisher: EmbeddingFisherPreconditioner | None = None,
    body_optimizer_kind: Literal["adam", "fisher"] = "adam",
    body_fisher_damping: float = 100.0,
    body_fisher_step_scale: float | Literal["auto"] = 1.0,
    body_fisher_max_step_scale: float = 1.0,
    body_fisher_trust_radius: float | None = None,
    body_fisher_curvature: BodyCurvature = "model",
) -> HybridEpochResult:
    started = time.perf_counter()
    if not compute_before_loss:
        before_loss = None
    elif flat_trie is None:
        before_loss = trie_negative_log_likelihood(model, vocab, samples).item()
    else:
        from agpt_ultra.flat_ops import flat_trie_negative_log_likelihood

        before_loss = flat_trie_negative_log_likelihood(
            model,
            flat_trie,
            initial_hidden=prefix_hidden_state(model, vocab, flat_prefix),
        ).item()
    before_loss_runtime_sec = time.perf_counter() - started
    started = time.perf_counter()
    body_fisher_stats = None
    embedding_fisher_stats = None
    if body_optimizer_kind == "fisher":
        if flat_trie is None:
            raise ValueError("body_optimizer_kind='fisher' requires flat_trie")
        if embedding_fisher is not None:
            raise ValueError("embedding_fisher cannot be combined with body_optimizer_kind='fisher'")
        body_fisher_stats = body_natural_step(
            model,
            vocab,
            flat_trie,
            flat_prefix=flat_prefix,
            update_embeddings=update_embeddings,
            damping=body_fisher_damping,
            step_scale=body_fisher_step_scale,
            max_step_scale=body_fisher_max_step_scale,
            trust_radius=body_fisher_trust_radius,
            max_cg_iter=max_cg_iter or 8,
            cg_tolerance=cg_tolerance,
            curvature=body_fisher_curvature,
        )
    else:
        _, embedding_fisher_stats = body_gradient_step(
            model,
            vocab,
            samples,
            optimizer=body_optimizer,
            max_grad_norm=max_grad_norm,
            flat_trie=flat_trie,
            flat_prefix=flat_prefix,
            update_embeddings=update_embeddings,
            embedding_fisher=embedding_fisher,
        )
    body_runtime_sec = time.perf_counter() - started
    started = time.perf_counter()
    if flat_trie is None:
        after_body_loss = trie_negative_log_likelihood(model, vocab, samples).item()
        flat_hidden = None
    else:
        from agpt_ultra.flat_ops import compute_flat_hidden_states, flat_trie_negative_log_likelihood_from_hidden

        flat_hidden = compute_flat_hidden_states(
            model,
            flat_trie,
            initial_hidden=prefix_hidden_state(model, vocab, flat_prefix),
        )
        after_body_loss = flat_trie_negative_log_likelihood_from_hidden(model, flat_trie, flat_hidden).item()
    hidden_cache_runtime_sec = time.perf_counter() - started
    started = time.perf_counter()
    head_result = head_only_epoch(
        model,
        vocab,
        samples,
        damping=head_damping,
        step_scale=head_step_scale,
        use_matrix_free=True,
        max_cg_iter=max_cg_iter,
        cg_tolerance=cg_tolerance,
        flat_trie=flat_trie,
        max_step_scale=max_head_step_scale,
        line_search_steps=line_search_steps,
        line_search_flat_trie=line_search_flat_trie,
        calibration_flat_trie=calibration_flat_trie,
        trust_radius=trust_radius,
        flat_hidden=flat_hidden,
        before_loss_override=after_body_loss,
    )
    head_runtime_sec = time.perf_counter() - started
    return HybridEpochResult(
        before_loss=before_loss,
        after_body_loss=after_body_loss,
        after_loss=head_result.after_loss,
        head_result=head_result,
        before_loss_runtime_sec=before_loss_runtime_sec,
        body_runtime_sec=body_runtime_sec,
        hidden_cache_runtime_sec=hidden_cache_runtime_sec,
        head_runtime_sec=head_runtime_sec,
        embedding_fisher_stats=embedding_fisher_stats,
        body_fisher_stats=body_fisher_stats,
    )


def train_hybrid(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    epochs: int,
    body_lr: float = 1e-3,
    head_damping: float = 1.0,
    head_step_scale: float | Literal["auto"] = 0.5,
    max_grad_norm: float | None = 1.0,
    max_cg_iter: int | None = None,
    cg_tolerance: float = 1e-6,
    flat_trie: FlatTrie | None = None,
    max_head_step_scale: float = 1.0,
    line_search_steps: int = 0,
    line_search_flat_trie: FlatTrie | None = None,
    calibration_flat_trie: FlatTrie | None = None,
    trust_radius: float | None = None,
    compute_before_loss: bool = True,
    update_embeddings: bool = False,
    flat_prefix: str = "",
) -> list[HybridEpochResult]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not update_embeddings:
        freeze_embeddings(model)
    body_params = [*model.cell.parameters()]
    if update_embeddings:
        body_params.extend(model.embed.parameters())
    optimizer = torch.optim.AdamW(body_params, lr=body_lr)
    return [
        hybrid_epoch(
            model,
            vocab,
            samples,
            body_optimizer=optimizer,
            head_damping=head_damping,
            head_step_scale=head_step_scale,
            max_grad_norm=max_grad_norm,
            max_cg_iter=max_cg_iter,
            cg_tolerance=cg_tolerance,
            flat_trie=flat_trie,
            flat_prefix=flat_prefix,
            max_head_step_scale=max_head_step_scale,
            line_search_steps=line_search_steps,
            line_search_flat_trie=line_search_flat_trie,
            calibration_flat_trie=calibration_flat_trie,
            trust_radius=trust_radius,
            compute_before_loss=compute_before_loss,
            update_embeddings=update_embeddings,
        )
        for _ in range(epochs)
    ]
