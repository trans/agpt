from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from agpt_ultra.model import TinyCharRNN


@dataclass(frozen=True)
class StateFisherCorrection:
    corrected_hidden: torch.Tensor
    before_loss: float
    after_loss: float
    mean_step_norm: float
    max_step_norm: float
    mean_eta: float | None
    min_eta: float | None
    max_eta: float | None


def parse_step_scale(value: str) -> float | str:
    if value == "auto-node":
        return value
    return float(value)


def optimal_node_eta(
    logits: torch.Tensor,
    target: torch.Tensor,
    direction_logits: torch.Tensor,
    max_eta: float,
    newton_steps: int,
) -> torch.Tensor:
    eta = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    upper = torch.full_like(eta, max_eta)

    for _ in range(newton_steps):
        stepped_logits = logits + eta[:, None] * direction_logits
        predicted = torch.softmax(stepped_logits, dim=1)
        derivative = (direction_logits * (predicted - target)).sum(dim=1)
        curvature = (
            predicted * direction_logits.square()
        ).sum(dim=1) - (predicted * direction_logits).sum(dim=1).square()

        upper = torch.where(derivative > 0, torch.minimum(upper, eta), upper)
        lower_candidate = torch.where(derivative <= 0, eta, torch.zeros_like(eta))
        fallback = 0.5 * (lower_candidate + upper)
        eta_newton = eta - derivative / curvature.clamp_min(1e-12)
        eta = torch.where(
            (eta_newton >= 0) & (eta_newton <= upper),
            eta_newton,
            fallback,
        ).clamp(0.0, max_eta)

    return eta


@torch.no_grad()
def state_fisher_correct_hidden(
    model: TinyCharRNN,
    hidden: torch.Tensor,
    counts: torch.Tensor,
    damping: float,
    step_scale: float | str,
    max_auto_eta: float,
    auto_newton_steps: int,
    state_iterations: int,
    curvature: str,
    chunk_size: int,
) -> StateFisherCorrection:
    dtype = hidden.dtype
    device = hidden.device
    counts = counts.to(device=device, dtype=dtype)
    totals = counts.sum(dim=1)
    corrected = hidden.detach().clone()
    step_norms: list[torch.Tensor] = []
    etas: list[torch.Tensor] = []
    weight = model.head.weight.detach().to(device=device, dtype=dtype)
    identity = torch.eye(hidden.shape[1], dtype=dtype, device=device)
    logits = model.head(corrected)
    before_loss = -(counts * F.log_softmax(logits, dim=1)).sum()

    active = torch.nonzero(totals > 0, as_tuple=False).flatten()
    for _ in range(state_iterations):
        logits = model.head(corrected)
        for offset in range(0, active.numel(), chunk_size):
            node_ids = active[offset : offset + chunk_size]
            node_hidden = corrected[node_ids]
            node_counts = counts[node_ids]
            node_totals = totals[node_ids]
            node_logits = logits[node_ids]
            predicted = torch.softmax(node_logits, dim=1)
            target = node_counts / node_totals[:, None]
            residual = predicted - target
            gradient = node_totals[:, None] * (residual @ weight)
            distribution = predicted if curvature == "model" else target

            weighted_gram = torch.einsum("nv,vh,vk->nhk", distribution, weight, weight)
            mean_direction = distribution @ weight
            covariance_term = torch.einsum("nh,nk->nhk", mean_direction, mean_direction)
            fisher = node_totals[:, None, None] * (weighted_gram - covariance_term)
            fisher = fisher + damping * identity
            delta = torch.linalg.solve(fisher, -gradient.unsqueeze(-1)).squeeze(-1)
            if step_scale == "auto-node":
                direction_logits = delta @ weight.T
                eta = optimal_node_eta(
                    node_logits,
                    target,
                    direction_logits,
                    max_eta=max_auto_eta,
                    newton_steps=auto_newton_steps,
                )
                corrected[node_ids] = node_hidden + eta[:, None] * delta
                etas.append(eta)
            else:
                corrected[node_ids] = node_hidden + float(step_scale) * delta
            step_norms.append(delta.norm(dim=1))

    after_logits = model.head(corrected)
    after_loss = -(counts * F.log_softmax(after_logits, dim=1)).sum()
    if step_norms:
        norms = torch.cat(step_norms)
        mean_step_norm = float(norms.mean().item())
        max_step_norm = float(norms.max().item())
    else:
        mean_step_norm = 0.0
        max_step_norm = 0.0
    if etas:
        eta_values = torch.cat(etas)
        mean_eta = float(eta_values.mean().item())
        min_eta = float(eta_values.min().item())
        max_eta_value = float(eta_values.max().item())
    else:
        mean_eta = None
        min_eta = None
        max_eta_value = None
    return StateFisherCorrection(
        corrected_hidden=corrected,
        before_loss=float(before_loss.item()),
        after_loss=float(after_loss.item()),
        mean_step_norm=mean_step_norm,
        max_step_norm=max_step_norm,
        mean_eta=mean_eta,
        min_eta=min_eta,
        max_eta=max_eta_value,
    )
