from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.head_only import ConjugateGradientResult
from agpt_ultra.model import TinyCharRNN


BodyCurvature = Literal["model", "empirical"]


@dataclass(frozen=True)
class BodyFisherStepStats:
    loss: float
    cg_iterations: int
    residual_norm: float
    grad_norm: float
    step_norm: float
    predicted_improvement: float
    linear_term: float
    fisher_quadratic: float
    step_scale: float
    eta_quad: float | None
    eta_trust: float | None


def _body_parameter_items(model: TinyCharRNN, update_embeddings: bool) -> list[tuple[str, Tensor]]:
    items: list[tuple[str, Tensor]] = []
    if update_embeddings:
        items.append(("embed.weight", model.embed.weight))
    for name, param in model.cell.named_parameters():
        items.append((f"cell.{name}", param))
    return items


def _tuple_dot(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> Tensor:
    return sum((a * b).sum() for a, b in zip(left, right, strict=True))


def _tuple_add(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _tuple_sub(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _tuple_mul(values: tuple[Tensor, ...], scalar: Tensor | float) -> tuple[Tensor, ...]:
    return tuple(value * scalar for value in values)


def _tuple_zeros_like(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(torch.zeros_like(value) for value in values)


def _tuple_norm(values: tuple[Tensor, ...]) -> Tensor:
    return torch.sqrt(torch.clamp(_tuple_dot(values, values), min=0.0))


def _gru_cell(input: Tensor, state: Tensor, params: dict[str, Tensor]) -> Tensor:
    gi = F.linear(input, params["cell.weight_ih"], params["cell.bias_ih"])
    gh = F.linear(state, params["cell.weight_hh"], params["cell.bias_hh"])
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    reset = torch.sigmoid(i_r + h_r)
    update = torch.sigmoid(i_z + h_z)
    new = torch.tanh(i_n + reset * h_n)
    return new + update * (state - new)


def _body_logits_function(
    model: TinyCharRNN,
    trie: FlatTrie,
    parameter_names: tuple[str, ...],
    update_embeddings: bool,
    prefix_token_ids: tuple[int, ...],
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    parents = trie.parents.to(device)
    tokens = trie.tokens.to(device)
    depths = trie.depths.to(device)
    max_depth = int(trie.depths.max().item())
    head_weight = model.head.weight.detach()
    head_bias = model.head.bias.detach()
    frozen_embed_weight = model.embed.weight.detach()

    def embed(token_ids: Tensor, params: dict[str, Tensor]) -> Tensor:
        weight = params["embed.weight"] if update_embeddings else frozen_embed_weight
        return F.embedding(token_ids, weight)

    def fn(parameters: tuple[Tensor, ...]) -> Tensor:
        params = dict(zip(parameter_names, parameters, strict=True))
        root = torch.zeros(model.n_hidden, device=device, dtype=dtype)
        for token_id in prefix_token_ids:
            token = torch.tensor([token_id], dtype=torch.long, device=device)
            root = _gru_cell(embed(token, params), root.unsqueeze(0), params).squeeze(0)

        hidden = root.new_zeros((trie.node_count, model.n_hidden))
        hidden[0] = root
        for depth in range(1, max_depth + 1):
            node_ids = torch.nonzero(depths == depth, as_tuple=False).flatten()
            if node_ids.numel() == 0:
                continue
            parent_hidden = hidden[parents[node_ids]]
            token_embeddings = embed(tokens[node_ids], params)
            hidden[node_ids] = _gru_cell(token_embeddings, parent_hidden, params)
        return hidden @ head_weight.T + head_bias

    return fn


def _loss_from_logits(logits: Tensor, counts: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(counts * log_probs).sum()


def _body_conjugate_gradient(
    matvec,
    rhs: tuple[Tensor, ...],
    max_iter: int,
    tolerance: float,
) -> ConjugateGradientResult:
    x = _tuple_zeros_like(rhs)
    r = rhs
    p = r
    rs_old = _tuple_dot(r, r)
    residual_norm = torch.sqrt(torch.clamp(rs_old, min=0.0))
    if float(residual_norm.item()) <= tolerance:
        return ConjugateGradientResult(solution=x, iterations=0, residual_norm=float(residual_norm.item()))

    iterations = 0
    for iterations in range(1, max_iter + 1):
        ap = matvec(p)
        denom = _tuple_dot(p, ap).clamp_min(1e-30)
        alpha = rs_old / denom
        x = _tuple_add(x, _tuple_mul(p, alpha))
        r = _tuple_sub(r, _tuple_mul(ap, alpha))
        rs_new = _tuple_dot(r, r)
        residual_norm = torch.sqrt(torch.clamp(rs_new, min=0.0))
        if float(residual_norm.item()) <= tolerance:
            break
        beta = rs_new / rs_old.clamp_min(1e-30)
        p = _tuple_add(r, _tuple_mul(p, beta))
        rs_old = rs_new

    return ConjugateGradientResult(solution=x, iterations=iterations, residual_norm=float(residual_norm.item()))


def body_fisher_step(
    model: TinyCharRNN,
    trie: FlatTrie,
    prefix_token_ids: tuple[int, ...] = (),
    update_embeddings: bool = False,
    damping: float = 100.0,
    step_scale: float | Literal["auto"] = 1.0,
    max_step_scale: float = 1.0,
    trust_radius: float | None = None,
    max_cg_iter: int = 8,
    cg_tolerance: float = 1e-6,
    curvature: BodyCurvature = "model",
) -> BodyFisherStepStats:
    items = _body_parameter_items(model, update_embeddings=update_embeddings)
    names = tuple(name for name, _ in items)
    parameters = tuple(param.detach().clone().requires_grad_(True) for _, param in items)
    counts = trie.transition_counts.to(device=parameters[0].device, dtype=parameters[0].dtype)
    totals = counts.sum(dim=1)
    mask = totals > 0
    logits_fn = _body_logits_function(
        model,
        trie,
        names,
        update_embeddings=update_embeddings,
        prefix_token_ids=prefix_token_ids,
    )

    def loss_fn(params: tuple[Tensor, ...]) -> Tensor:
        return _loss_from_logits(logits_fn(params), counts)

    loss = loss_fn(parameters)
    gradient = torch.func.grad(loss_fn)(parameters)
    rhs = _tuple_mul(gradient, -1.0)

    with torch.no_grad():
        base_logits = logits_fn(parameters)
        predicted = torch.softmax(base_logits, dim=1)
        target = torch.zeros_like(predicted)
        target[mask] = counts[mask] / totals[mask, None]
        covariance_distribution = predicted if curvature == "model" else target

    def fisher_only_matvec(vector: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        _, logits_tangent = torch.func.jvp(logits_fn, (parameters,), (vector,))
        distribution = covariance_distribution
        centered = distribution * (logits_tangent - (distribution * logits_tangent).sum(dim=1, keepdim=True))
        weighted = totals[:, None] * centered
        _, vjp_fn = torch.func.vjp(logits_fn, parameters)
        return vjp_fn(weighted)[0]

    def damped_matvec(vector: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        fisher_vector = fisher_only_matvec(vector)
        return tuple(
            fisher_item + damping * vector_item
            for fisher_item, vector_item in zip(fisher_vector, vector, strict=True)
        )

    cg = _body_conjugate_gradient(
        damped_matvec,
        rhs,
        max_iter=max_cg_iter,
        tolerance=cg_tolerance,
    )
    step = cg.solution
    fisher_step = fisher_only_matvec(step)
    linear_term = _tuple_dot(gradient, step)
    fisher_quadratic = _tuple_dot(step, fisher_step)
    eta_quad = None
    eta_trust = None
    if float(fisher_quadratic.detach().item()) > 0:
        eta_quad_tensor = -linear_term / fisher_quadratic.clamp_min(1e-30)
        eta_quad = float(eta_quad_tensor.detach().item())
    if trust_radius is not None:
        eta_trust_tensor = torch.as_tensor(trust_radius, device=fisher_quadratic.device, dtype=fisher_quadratic.dtype) / torch.sqrt(
            fisher_quadratic.clamp_min(1e-30)
        )
        eta_trust = float(eta_trust_tensor.detach().item())
    if step_scale == "auto":
        candidates = [max_step_scale]
        if eta_quad is not None:
            candidates.append(max(0.0, eta_quad))
        if eta_trust is not None:
            candidates.append(max(0.0, eta_trust))
        applied_step_scale = min(candidates)
    else:
        applied_step_scale = float(step_scale)
    predicted_improvement = -(
        applied_step_scale * linear_term
        + 0.5 * applied_step_scale * applied_step_scale * fisher_quadratic
    )

    with torch.no_grad():
        for (_, param), delta in zip(items, step, strict=True):
            param.add_(applied_step_scale * delta)

    return BodyFisherStepStats(
        loss=float(loss.detach().item()),
        cg_iterations=cg.iterations,
        residual_norm=cg.residual_norm,
        grad_norm=float(_tuple_norm(gradient).detach().item()),
        step_norm=float(_tuple_norm(step).detach().item()),
        predicted_improvement=float(predicted_improvement.detach().item()),
        linear_term=float(linear_term.detach().item()),
        fisher_quadratic=float(fisher_quadratic.detach().item()),
        step_scale=applied_step_scale,
        eta_quad=eta_quad,
        eta_trust=eta_trust,
    )
