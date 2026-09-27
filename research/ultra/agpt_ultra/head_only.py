from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from agpt_ultra.data import CharVocab
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.fisher import categorical_covariance, empirical_distribution, transition_counts
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.objective import trie_negative_log_likelihood
from agpt_ultra.reconcile import GradientModelEvidence, apply_gradient_evidence, reconcile_gradient_evidence
from agpt_ultra.trie import PrefixNode, PrefixTrie


@dataclass(frozen=True)
class HeadShape:
    vocab_size: int
    hidden_size: int

    @property
    def theta_size(self) -> int:
        return self.vocab_size * self.hidden_size + self.vocab_size


@dataclass(frozen=True)
class HeadOnlyEpochResult:
    before_loss: float
    after_loss: float
    theta: Tensor
    evidence: GradientModelEvidence
    cg_iterations: int | None = None
    step_scale: float | None = None
    suggested_step_scale: float | None = None
    predicted_improvement: float | None = None
    actual_improvement: float | None = None
    improvement_ratio: float | None = None
    train_actual_improvement: float | None = None
    train_improvement_ratio: float | None = None
    val_actual_improvement: float | None = None
    val_improvement_ratio: float | None = None
    linear_term: float | None = None
    fisher_quadratic: float | None = None
    eta_quad: float | None = None
    eta_trust: float | None = None
    trust_radius: float | None = None


@dataclass(frozen=True)
class HeadFisherFactor:
    features: Tensor
    predicted: Tensor
    count: Tensor


@dataclass(frozen=True)
class MatrixFreeHeadEvidence:
    gradient: Tensor
    factors: list[HeadFisherFactor]
    shape: HeadShape


@dataclass(frozen=True)
class BatchedHeadEvidence:
    gradient: Tensor
    features: Tensor
    predicted: Tensor
    counts: Tensor
    shape: HeadShape


@dataclass(frozen=True)
class ChunkedBatchedHeadEvidence:
    gradient: Tensor
    chunks: tuple[BatchedHeadEvidence, ...]
    shape: HeadShape


@dataclass(frozen=True)
class ConjugateGradientResult:
    solution: Tensor
    iterations: int
    residual_norm: float


def head_shape(model: TinyCharRNN) -> HeadShape:
    return HeadShape(vocab_size=model.head.out_features, hidden_size=model.head.in_features)


def flatten_head(weight: Tensor, bias: Tensor) -> Tensor:
    return torch.cat([weight.reshape(-1), bias.reshape(-1)])


def unflatten_head(theta: Tensor, shape: HeadShape) -> tuple[Tensor, Tensor]:
    expected = shape.theta_size
    if theta.numel() != expected:
        raise ValueError(f"expected theta with {expected} values, got {theta.numel()}")
    weight_size = shape.vocab_size * shape.hidden_size
    weight = theta[:weight_size].reshape(shape.vocab_size, shape.hidden_size)
    bias = theta[weight_size:].reshape(shape.vocab_size)
    return weight, bias


def model_head_theta(model: TinyCharRNN) -> Tensor:
    return flatten_head(model.head.weight.detach(), model.head.bias.detach())


def load_head_theta(model: TinyCharRNN, theta: Tensor) -> None:
    weight, bias = unflatten_head(theta, head_shape(model))
    with torch.no_grad():
        model.head.weight.copy_(weight)
        model.head.bias.copy_(bias)


def head_logits(theta: Tensor, shape: HeadShape, hidden: Tensor) -> Tensor:
    weight, bias = unflatten_head(theta, shape)
    return weight @ hidden + bias


def node_head_evidence(theta: Tensor, shape: HeadShape, hidden: Tensor, counts: Tensor) -> GradientModelEvidence:
    total = counts.sum()
    gradient = torch.zeros_like(theta)
    fisher = torch.zeros((theta.numel(), theta.numel()), dtype=theta.dtype, device=theta.device)
    if total <= 0:
        return GradientModelEvidence(gradient=gradient, fisher=fisher)

    counts = counts.to(dtype=theta.dtype, device=theta.device)
    hidden = hidden.to(dtype=theta.dtype, device=theta.device)
    target = empirical_distribution(counts)
    logits = head_logits(theta, shape, hidden)
    predicted = torch.softmax(logits, dim=0)
    residual = predicted - target

    logit_jacobian = torch.zeros(
        shape.vocab_size,
        shape.theta_size,
        dtype=theta.dtype,
        device=theta.device,
    )
    for token_id in range(shape.vocab_size):
        start = token_id * shape.hidden_size
        logit_jacobian[token_id, start : start + shape.hidden_size] = hidden
        logit_jacobian[token_id, shape.vocab_size * shape.hidden_size + token_id] = 1.0

    gradient = total * (logit_jacobian.T @ residual)
    logit_cov = categorical_covariance(predicted)
    fisher = total * (logit_jacobian.T @ logit_cov @ logit_jacobian)
    return GradientModelEvidence(gradient=gradient, fisher=fisher)


def node_head_factor(theta: Tensor, shape: HeadShape, hidden: Tensor, counts: Tensor) -> tuple[Tensor, HeadFisherFactor | None]:
    total = counts.sum()
    gradient = torch.zeros_like(theta)
    if total <= 0:
        return gradient, None

    counts = counts.to(dtype=theta.dtype, device=theta.device)
    hidden = hidden.to(dtype=theta.dtype, device=theta.device)
    target = empirical_distribution(counts)
    logits = head_logits(theta, shape, hidden)
    predicted = torch.softmax(logits, dim=0)
    residual = predicted - target
    features = torch.cat([hidden, hidden.new_ones(1)])
    weight_gradient = total * torch.outer(residual, hidden)
    bias_gradient = total * residual
    gradient = flatten_head(weight_gradient, bias_gradient)
    return gradient, HeadFisherFactor(
        features=features.detach().clone(),
        predicted=predicted.detach().clone(),
        count=total.detach().clone(),
    )


def head_fisher_matvec(v: Tensor, evidence: MatrixFreeHeadEvidence) -> Tensor:
    weight_v, bias_v = unflatten_head(v, evidence.shape)
    out_weight = torch.zeros_like(weight_v)
    out_bias = torch.zeros_like(bias_v)

    for factor in evidence.factors:
        hidden_v = factor.features[:-1]
        bias_feature = factor.features[-1]
        logits_v = weight_v @ hidden_v + bias_v * bias_feature
        centered = factor.predicted * (logits_v - factor.predicted.dot(logits_v))
        contribution = factor.count * centered
        out_weight = out_weight + torch.outer(contribution, hidden_v)
        out_bias = out_bias + contribution * bias_feature

    return flatten_head(out_weight, out_bias)


def batched_head_fisher_matvec(v: Tensor, evidence: BatchedHeadEvidence) -> Tensor:
    weight_v, bias_v = unflatten_head(v, evidence.shape)
    hidden = evidence.features[:, :-1]
    bias_feature = evidence.features[:, -1]
    logits_v = hidden @ weight_v.T + bias_feature[:, None] * bias_v[None, :]
    centered = evidence.predicted * (logits_v - (evidence.predicted * logits_v).sum(dim=1, keepdim=True))
    weighted = evidence.counts[:, None] * centered
    out_weight = weighted.T @ hidden
    out_bias = (weighted * bias_feature[:, None]).sum(dim=0)
    return flatten_head(out_weight, out_bias)


def merge_batched_head_evidence(evidence: list[BatchedHeadEvidence]) -> BatchedHeadEvidence:
    if not evidence:
        raise ValueError("cannot merge no evidence")
    shape = evidence[0].shape
    if any(item.shape != shape for item in evidence):
        raise ValueError("all evidence must have the same head shape")
    return BatchedHeadEvidence(
        gradient=sum((item.gradient for item in evidence), torch.zeros_like(evidence[0].gradient)),
        features=torch.cat([item.features for item in evidence], dim=0),
        predicted=torch.cat([item.predicted for item in evidence], dim=0),
        counts=torch.cat([item.counts for item in evidence], dim=0),
        shape=shape,
    )


def chunk_batched_head_evidence(evidence: list[BatchedHeadEvidence]) -> ChunkedBatchedHeadEvidence:
    if not evidence:
        raise ValueError("cannot chunk no evidence")
    shape = evidence[0].shape
    if any(item.shape != shape for item in evidence):
        raise ValueError("all evidence must have the same head shape")
    return ChunkedBatchedHeadEvidence(
        gradient=sum((item.gradient for item in evidence), torch.zeros_like(evidence[0].gradient)),
        chunks=tuple(evidence),
        shape=shape,
    )


def chunked_batched_head_fisher_matvec(v: Tensor, evidence: ChunkedBatchedHeadEvidence) -> Tensor:
    out = torch.zeros_like(v)
    for chunk in evidence.chunks:
        out = out + batched_head_fisher_matvec(v, chunk)
    return out


def conjugate_gradient(
    matvec,
    rhs: Tensor,
    max_iter: int | None = None,
    tolerance: float = 1e-6,
) -> ConjugateGradientResult:
    if max_iter is None:
        max_iter = min(rhs.numel(), 256)
    x = torch.zeros_like(rhs)
    residual = rhs - matvec(x)
    direction = residual.clone()
    residual_sq = residual.dot(residual)
    tolerance_sq = tolerance * tolerance

    if residual_sq.item() <= tolerance_sq:
        return ConjugateGradientResult(solution=x, iterations=0, residual_norm=residual_sq.sqrt().item())

    iterations = 0
    for iterations in range(1, max_iter + 1):
        mat_direction = matvec(direction)
        denom = direction.dot(mat_direction)
        if abs(denom.item()) < 1e-30:
            break
        alpha = residual_sq / denom
        x = x + alpha * direction
        residual = residual - alpha * mat_direction
        next_residual_sq = residual.dot(residual)
        if next_residual_sq.item() <= tolerance_sq:
            residual_sq = next_residual_sq
            break
        beta = next_residual_sq / residual_sq
        direction = residual + beta * direction
        residual_sq = next_residual_sq

    return ConjugateGradientResult(
        solution=x,
        iterations=iterations,
        residual_norm=residual_sq.sqrt().item(),
    )


def natural_head_step_matrix_free(
    evidence: MatrixFreeHeadEvidence | BatchedHeadEvidence | ChunkedBatchedHeadEvidence,
    damping: float = 1e-2,
    max_cg_iter: int | None = None,
    cg_tolerance: float = 1e-6,
) -> ConjugateGradientResult:
    if damping < 0:
        raise ValueError("damping must be non-negative")

    def damped_matvec(v: Tensor) -> Tensor:
        if isinstance(evidence, ChunkedBatchedHeadEvidence):
            return chunked_batched_head_fisher_matvec(v, evidence) + damping * v
        if isinstance(evidence, BatchedHeadEvidence):
            return batched_head_fisher_matvec(v, evidence) + damping * v
        return head_fisher_matvec(v, evidence) + damping * v

    return conjugate_gradient(
        damped_matvec,
        -evidence.gradient,
        max_iter=max_cg_iter,
        tolerance=cg_tolerance,
    )


def suggested_quadratic_step_scale(
    gradient: Tensor,
    delta: Tensor,
    fisher_delta: Tensor,
    max_step_scale: float = 1.0,
) -> float:
    numerator = -gradient.dot(delta).item()
    denominator = delta.dot(fisher_delta).item()
    if numerator <= 0.0 or denominator <= 0.0:
        return 0.0
    return max(0.0, min(max_step_scale, numerator / denominator))


def trust_radius_step_scale(
    fisher_quadratic: float,
    trust_radius: float | None,
    eps: float = 1e-12,
) -> float | None:
    if trust_radius is None:
        return None
    if trust_radius < 0.0:
        raise ValueError("trust_radius must be non-negative")
    if fisher_quadratic <= 0.0:
        return 0.0
    return trust_radius / ((fisher_quadratic + eps) ** 0.5)


def quadratic_predicted_improvement(
    gradient: Tensor,
    delta: Tensor,
    fisher_delta: Tensor,
    step_scale: float,
) -> float:
    linear = step_scale * gradient.dot(delta).item()
    quadratic = 0.5 * step_scale * step_scale * delta.dot(fisher_delta).item()
    return max(0.0, -(linear + quadratic))


@torch.no_grad()
def collect_head_evidence(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    theta: Tensor | None = None,
) -> GradientModelEvidence:
    if theta is None:
        theta = model_head_theta(model)
    theta = theta.detach().clone()
    shape = head_shape(model)
    trie = PrefixTrie.from_samples(samples)
    device = next(model.parameters()).device

    def visit(node: PrefixNode, hidden: Tensor) -> list[GradientModelEvidence]:
        counts = transition_counts(node, vocab).to(device=device, dtype=theta.dtype)
        evidence = [node_head_evidence(theta, shape, hidden.squeeze(0), counts)]
        for token, child in sorted(node.children.items()):
            token_id = torch.tensor([vocab.stoi[token]], dtype=torch.long, device=device)
            _, child_hidden = model.step(token_id, hidden)
            evidence.extend(visit(child, child_hidden))
        return evidence

    node_evidence = visit(trie.root, model.initial_state(1, device))
    return reconcile_gradient_evidence(node_evidence)


@torch.no_grad()
def collect_matrix_free_head_evidence(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    theta: Tensor | None = None,
) -> MatrixFreeHeadEvidence:
    if theta is None:
        theta = model_head_theta(model)
    theta = theta.detach().clone()
    shape = head_shape(model)
    trie = PrefixTrie.from_samples(samples)
    device = next(model.parameters()).device
    gradient = torch.zeros_like(theta)
    factors: list[HeadFisherFactor] = []

    def visit(node: PrefixNode, hidden: Tensor) -> None:
        nonlocal gradient
        counts = transition_counts(node, vocab).to(device=device, dtype=theta.dtype)
        node_gradient, factor = node_head_factor(theta, shape, hidden.squeeze(0), counts)
        gradient = gradient + node_gradient
        if factor is not None:
            factors.append(factor)
        for token, child in sorted(node.children.items()):
            token_id = torch.tensor([vocab.stoi[token]], dtype=torch.long, device=device)
            _, child_hidden = model.step(token_id, hidden)
            visit(child, child_hidden)

    visit(trie.root, model.initial_state(1, device))
    return MatrixFreeHeadEvidence(gradient=gradient, factors=factors, shape=shape)


def head_only_epoch(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    damping: float = 1e-2,
    step_scale: float | Literal["auto"] = 1.0,
    use_matrix_free: bool = True,
    max_cg_iter: int | None = None,
    cg_tolerance: float = 1e-6,
    flat_trie: FlatTrie | None = None,
    max_step_scale: float = 1.0,
    line_search_steps: int = 0,
    line_search_flat_trie: FlatTrie | None = None,
    calibration_flat_trie: FlatTrie | None = None,
    trust_radius: float | None = None,
    flat_hidden: Tensor | None = None,
    before_loss_override: float | None = None,
) -> HeadOnlyEpochResult:
    if flat_hidden is not None:
        from agpt_ultra.flat_ops import flat_trie_negative_log_likelihood_from_hidden
    if before_loss_override is not None:
        before_loss = before_loss_override
    elif flat_trie is None:
        before_loss = trie_negative_log_likelihood(model, vocab, samples).item()
    else:
        from agpt_ultra.flat_ops import flat_trie_negative_log_likelihood, flat_trie_negative_log_likelihood_from_hidden

        if flat_hidden is None:
            before_loss = flat_trie_negative_log_likelihood(model, flat_trie).item()
        else:
            before_loss = flat_trie_negative_log_likelihood_from_hidden(model, flat_trie, flat_hidden).item()
    if line_search_flat_trie is not None or calibration_flat_trie is not None:
        from agpt_ultra.flat_ops import flat_trie_negative_log_likelihood
    theta = model_head_theta(model)
    cg_iterations = None
    accepted_step_scale: float | None = None
    suggested_step_scale: float | None = None
    predicted_improvement: float | None = None
    actual_improvement: float | None = None
    improvement_ratio: float | None = None
    train_actual_improvement: float | None = None
    train_improvement_ratio: float | None = None
    val_actual_improvement: float | None = None
    val_improvement_ratio: float | None = None
    linear_term: float | None = None
    fisher_quadratic: float | None = None
    eta_quad: float | None = None
    eta_trust: float | None = None
    search_before_loss: float | None = None
    search_after_loss: float | None = None
    calibration_before_loss: float | None = None
    fisher_delta: Tensor | None = None
    if use_matrix_free:
        if flat_trie is None:
            matrix_free_evidence = collect_matrix_free_head_evidence(model, vocab, samples, theta)
        else:
            from agpt_ultra.flat_ops import collect_batched_flat_head_evidence, collect_batched_flat_head_evidence_from_hidden

            if flat_hidden is None:
                matrix_free_evidence = collect_batched_flat_head_evidence(model, flat_trie, theta)
            else:
                matrix_free_evidence = collect_batched_flat_head_evidence_from_hidden(model, flat_trie, flat_hidden, theta)
        cg_result = natural_head_step_matrix_free(
            matrix_free_evidence,
            damping=damping,
            max_cg_iter=max_cg_iter,
            cg_tolerance=cg_tolerance,
        )
        delta = cg_result.solution
        cg_iterations = cg_result.iterations
        if isinstance(matrix_free_evidence, BatchedHeadEvidence):
            fisher_delta = batched_head_fisher_matvec(delta, matrix_free_evidence)
        else:
            fisher_delta = head_fisher_matvec(delta, matrix_free_evidence)
        linear_term = matrix_free_evidence.gradient.dot(delta).item()
        fisher_quadratic = delta.dot(fisher_delta).item()
        eta_quad = suggested_quadratic_step_scale(
            matrix_free_evidence.gradient,
            delta,
            fisher_delta,
            max_step_scale=max_step_scale,
        )
        eta_trust = trust_radius_step_scale(fisher_quadratic, trust_radius)

        if step_scale == "auto":
            suggested_step_scale = eta_quad
            if eta_trust is not None:
                suggested_step_scale = min(suggested_step_scale, eta_trust)
            accepted_step_scale = suggested_step_scale
        else:
            accepted_step_scale = float(step_scale)

        def loss_for(candidate_theta: Tensor) -> float:
            load_head_theta(model, candidate_theta)
            if flat_trie is None:
                return trie_negative_log_likelihood(model, vocab, samples).item()
            if flat_hidden is None:
                return flat_trie_negative_log_likelihood(model, flat_trie).item()
            return flat_trie_negative_log_likelihood_from_hidden(model, flat_trie, flat_hidden).item()

        if calibration_flat_trie is not None:
            calibration_before_loss = flat_trie_negative_log_likelihood(model, calibration_flat_trie).item()

        if line_search_steps > 0:
            search_before_loss = before_loss
            if line_search_flat_trie is not None:
                search_before_loss = flat_trie_negative_log_likelihood(model, line_search_flat_trie).item()

            trial_scale = accepted_step_scale
            for _ in range(line_search_steps + 1):
                candidate = theta + trial_scale * delta
                if line_search_flat_trie is None:
                    trial_loss = loss_for(candidate)
                else:
                    load_head_theta(model, candidate)
                    trial_loss = flat_trie_negative_log_likelihood(model, line_search_flat_trie).item()
                if trial_loss < search_before_loss:
                    accepted_step_scale = trial_scale
                    search_after_loss = trial_loss
                    break
                trial_scale *= 0.5
            else:
                accepted_step_scale = 0.0
                search_after_loss = search_before_loss
        next_theta = theta + accepted_step_scale * delta
        if fisher_delta is not None:
            predicted_improvement = quadratic_predicted_improvement(
                matrix_free_evidence.gradient,
                delta,
                fisher_delta,
                accepted_step_scale,
            )
        evidence = GradientModelEvidence(
            gradient=matrix_free_evidence.gradient,
            fisher=torch.empty(0, dtype=theta.dtype, device=theta.device),
        )
    else:
        evidence = collect_head_evidence(model, vocab, samples, theta)
        accepted_step_scale = 1.0 if step_scale == "auto" else float(step_scale)
        next_theta = theta + accepted_step_scale * (apply_gradient_evidence(theta, [evidence], damping=damping) - theta)
    load_head_theta(model, next_theta)
    if flat_trie is None:
        after_loss = trie_negative_log_likelihood(model, vocab, samples).item()
    else:
        if flat_hidden is None:
            after_loss = flat_trie_negative_log_likelihood(model, flat_trie).item()
        else:
            after_loss = flat_trie_negative_log_likelihood_from_hidden(model, flat_trie, flat_hidden).item()
    if use_matrix_free:
        if search_before_loss is None:
            search_before_loss = before_loss
            search_after_loss = after_loss
        elif search_after_loss is None:
            if line_search_flat_trie is None:
                search_after_loss = after_loss
            else:
                search_after_loss = flat_trie_negative_log_likelihood(model, line_search_flat_trie).item()
        actual_improvement = search_before_loss - search_after_loss
        if predicted_improvement is not None and predicted_improvement > 0.0:
            improvement_ratio = actual_improvement / predicted_improvement
        train_actual_improvement = before_loss - after_loss
        if predicted_improvement is not None and predicted_improvement > 0.0:
            train_improvement_ratio = train_actual_improvement / predicted_improvement
        if calibration_flat_trie is not None and calibration_before_loss is not None:
            calibration_after_loss = flat_trie_negative_log_likelihood(model, calibration_flat_trie).item()
            val_actual_improvement = calibration_before_loss - calibration_after_loss
            if predicted_improvement is not None and predicted_improvement > 0.0:
                val_improvement_ratio = val_actual_improvement / predicted_improvement
    return HeadOnlyEpochResult(
        before_loss=before_loss,
        after_loss=after_loss,
        theta=next_theta.detach().clone(),
        evidence=evidence,
        cg_iterations=cg_iterations,
        step_scale=accepted_step_scale,
        suggested_step_scale=suggested_step_scale,
        predicted_improvement=predicted_improvement,
        actual_improvement=actual_improvement,
        improvement_ratio=improvement_ratio,
        train_actual_improvement=train_actual_improvement,
        train_improvement_ratio=train_improvement_ratio,
        val_actual_improvement=val_actual_improvement,
        val_improvement_ratio=val_improvement_ratio,
        linear_term=linear_term,
        fisher_quadratic=fisher_quadratic,
        eta_quad=eta_quad,
        eta_trust=eta_trust,
        trust_radius=trust_radius,
    )


def train_head_only(
    model: TinyCharRNN,
    vocab: CharVocab,
    samples: list[str],
    epochs: int,
    damping: float = 1e-2,
    step_scale: float | Literal["auto"] = 1.0,
    use_matrix_free: bool = True,
    max_cg_iter: int | None = None,
    cg_tolerance: float = 1e-6,
    flat_trie: FlatTrie | None = None,
    max_step_scale: float = 1.0,
    line_search_steps: int = 0,
    line_search_flat_trie: FlatTrie | None = None,
    calibration_flat_trie: FlatTrie | None = None,
    trust_radius: float | None = None,
    flat_hidden: Tensor | None = None,
) -> list[HeadOnlyEpochResult]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    return [
        head_only_epoch(
            model,
            vocab,
            samples,
            damping=damping,
            step_scale=step_scale,
            use_matrix_free=use_matrix_free,
            max_cg_iter=max_cg_iter,
            cg_tolerance=cg_tolerance,
            flat_trie=flat_trie,
            max_step_scale=max_step_scale,
            line_search_steps=line_search_steps,
            line_search_flat_trie=line_search_flat_trie,
            calibration_flat_trie=calibration_flat_trie,
            trust_radius=trust_radius,
            flat_hidden=flat_hidden,
        )
        for _ in range(epochs)
    ]
