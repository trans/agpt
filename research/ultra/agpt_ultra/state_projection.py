from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from agpt_ultra.body_fisher import (
    _body_conjugate_gradient,
    _body_parameter_items,
    _gru_cell,
    _tuple_dot,
    _tuple_norm,
)
from agpt_ultra.flat_trie import FlatTrie
from agpt_ultra.model import TinyCharRNN


@dataclass(frozen=True)
class StateProjectionStats:
    cg_iterations: int
    residual_norm: float
    rhs_norm: float
    step_norm: float
    projected_delta_norm: float
    target_delta_norm: float
    weighted_mse_before: float
    weighted_mse_after: float
    step_scale: float


def _body_hidden_function(
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
        return hidden

    return fn


def state_delta_projection_step(
    model: TinyCharRNN,
    trie: FlatTrie,
    target_delta_hidden: Tensor,
    prefix_token_ids: tuple[int, ...] = (),
    update_embeddings: bool = True,
    damping: float = 1.0,
    step_scale: float = 1.0,
    max_cg_iter: int = 8,
    cg_tolerance: float = 1e-6,
) -> StateProjectionStats:
    items = _body_parameter_items(model, update_embeddings=update_embeddings)
    names = tuple(name for name, _ in items)
    parameters = tuple(param.detach().clone().requires_grad_(True) for _, param in items)
    device = parameters[0].device
    dtype = parameters[0].dtype
    counts = trie.transition_counts.to(device=device, dtype=dtype)
    totals = counts.sum(dim=1)
    weights = totals / totals.sum().clamp_min(1.0)
    target_delta = target_delta_hidden.detach().to(device=device, dtype=dtype)
    weighted_target = weights[:, None] * target_delta
    hidden_fn = _body_hidden_function(
        model,
        trie,
        names,
        update_embeddings=update_embeddings,
        prefix_token_ids=prefix_token_ids,
    )

    _, vjp_fn = torch.func.vjp(hidden_fn, parameters)
    rhs = vjp_fn(weighted_target)[0]

    def jt_weighted_j(vector: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        _, hidden_tangent = torch.func.jvp(hidden_fn, (parameters,), (vector,))
        _, local_vjp = torch.func.vjp(hidden_fn, parameters)
        return local_vjp(weights[:, None] * hidden_tangent)[0]

    def damped_matvec(vector: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        jtj_vector = jt_weighted_j(vector)
        return tuple(
            item + damping * vector_item
            for item, vector_item in zip(jtj_vector, vector, strict=True)
        )

    cg = _body_conjugate_gradient(
        damped_matvec,
        rhs,
        max_iter=max_cg_iter,
        tolerance=cg_tolerance,
    )
    step = cg.solution
    _, projected_delta = torch.func.jvp(hidden_fn, (parameters,), (step,))
    weighted_mse_before = (weights[:, None] * target_delta.square()).sum()
    residual_after = step_scale * projected_delta - target_delta
    weighted_mse_after = (weights[:, None] * residual_after.square()).sum()

    with torch.no_grad():
        for (_, param), delta in zip(items, step, strict=True):
            param.add_(step_scale * delta)

    return StateProjectionStats(
        cg_iterations=cg.iterations,
        residual_norm=cg.residual_norm,
        rhs_norm=float(_tuple_norm(rhs).detach().item()),
        step_norm=float(_tuple_norm(step).detach().item()),
        projected_delta_norm=float(torch.sqrt((weights[:, None] * projected_delta.square()).sum()).detach().item()),
        target_delta_norm=float(torch.sqrt(weighted_mse_before).detach().item()),
        weighted_mse_before=float(weighted_mse_before.detach().item()),
        weighted_mse_after=float(weighted_mse_after.detach().item()),
        step_scale=step_scale,
    )
