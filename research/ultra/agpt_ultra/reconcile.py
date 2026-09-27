from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from agpt_ultra.trie import PrefixNode


@dataclass(frozen=True)
class LocalModelEvidence:
    theta: Tensor
    fisher: Tensor


@dataclass(frozen=True)
class GradientModelEvidence:
    gradient: Tensor
    fisher: Tensor


@dataclass(frozen=True)
class TreeReconcileResult:
    global_theta: Tensor
    root_evidence: LocalModelEvidence
    node_evidence: dict[str, LocalModelEvidence]


def reconcile_child_models(
    children: list[LocalModelEvidence],
    damping: float = 1e-3,
    prior_theta: Tensor | None = None,
) -> Tensor:
    if not children:
        raise ValueError("cannot reconcile an empty child set")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    theta_shape = children[0].theta.shape
    theta_size = children[0].theta.numel()
    dtype = children[0].theta.dtype
    device = children[0].theta.device

    precision = torch.zeros((theta_size, theta_size), dtype=dtype, device=device)
    rhs = torch.zeros(theta_size, dtype=dtype, device=device)

    for child in children:
        theta = child.theta.reshape(-1)
        if theta.numel() != theta_size:
            raise ValueError("all child theta tensors must have the same size")
        if child.fisher.shape != (theta_size, theta_size):
            raise ValueError("child fisher must have shape [theta_size, theta_size]")
        precision = precision + child.fisher
        rhs = rhs + child.fisher @ theta

    identity = torch.eye(theta_size, dtype=dtype, device=device)
    if prior_theta is not None:
        if prior_theta.shape != theta_shape:
            raise ValueError("prior_theta must have the same shape as child theta")
        rhs = rhs + damping * prior_theta.reshape(-1)
    solution = torch.linalg.solve(precision + damping * identity, rhs)
    return solution.reshape(theta_shape)


def natural_parameter_step(evidence: GradientModelEvidence, damping: float = 1e-3) -> Tensor:
    if damping < 0:
        raise ValueError("damping must be non-negative")
    theta_size = evidence.gradient.numel()
    fisher = evidence.fisher.reshape(theta_size, theta_size)
    identity = torch.eye(theta_size, dtype=evidence.gradient.dtype, device=evidence.gradient.device)
    return -torch.linalg.solve(fisher + damping * identity, evidence.gradient.reshape(-1)).reshape(
        evidence.gradient.shape
    )


def reconcile_gradient_evidence(children: list[GradientModelEvidence]) -> GradientModelEvidence:
    if not children:
        raise ValueError("cannot reconcile an empty child set")

    gradient = torch.zeros_like(children[0].gradient)
    fisher = torch.zeros_like(children[0].fisher)
    for child in children:
        if child.gradient.shape != gradient.shape:
            raise ValueError("all child gradients must have the same shape")
        if child.fisher.shape != fisher.shape:
            raise ValueError("all child fishers must have the same shape")
        gradient = gradient + child.gradient
        fisher = fisher + child.fisher
    return GradientModelEvidence(gradient=gradient, fisher=fisher)


def apply_gradient_evidence(
    theta: Tensor,
    children: list[GradientModelEvidence],
    damping: float = 1e-3,
) -> Tensor:
    evidence = reconcile_gradient_evidence(children)
    return theta + natural_parameter_step(evidence, damping=damping)


def quadratic_reconciliation_loss(theta: Tensor, children: list[LocalModelEvidence]) -> Tensor:
    flat_theta = theta.reshape(-1)
    loss = flat_theta.new_zeros(())
    for child in children:
        delta = flat_theta - child.theta.reshape(-1)
        loss = loss + 0.5 * (delta @ child.fisher @ delta)
    return loss


def sum_fishers(evidence: list[LocalModelEvidence]) -> Tensor:
    if not evidence:
        raise ValueError("cannot sum an empty evidence set")
    total = torch.zeros_like(evidence[0].fisher)
    for item in evidence:
        total = total + item.fisher
    return total


def reconcile_tree_epoch(
    root: PrefixNode,
    global_theta: Tensor,
    local_update: Callable[[str, PrefixNode, Tensor], LocalModelEvidence],
    damping: float = 1e-3,
) -> TreeReconcileResult:
    node_evidence: dict[str, LocalModelEvidence] = {}

    def visit(node: PrefixNode, prefix: str) -> LocalModelEvidence:
        local = local_update(prefix, node, global_theta.detach().clone())
        child_evidence = [
            visit(child, prefix + token)
            for token, child in sorted(node.children.items())
        ]
        evidence_set = [local, *child_evidence]
        theta = reconcile_child_models(evidence_set, damping=damping, prior_theta=global_theta)
        reconciled = LocalModelEvidence(theta=theta, fisher=sum_fishers(evidence_set))
        node_evidence[prefix] = reconciled
        return reconciled

    root_evidence = visit(root, "")
    return TreeReconcileResult(
        global_theta=root_evidence.theta,
        root_evidence=root_evidence,
        node_evidence=node_evidence,
    )


def run_reconciliation_epochs(
    root: PrefixNode,
    initial_theta: Tensor,
    local_update: Callable[[int, str, PrefixNode, Tensor], LocalModelEvidence],
    epochs: int,
    damping: float = 1e-3,
) -> list[TreeReconcileResult]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    theta = initial_theta.detach().clone()
    results: list[TreeReconcileResult] = []
    for epoch in range(epochs):
        result = reconcile_tree_epoch(
            root,
            theta,
            local_update=lambda prefix, node, base_theta: local_update(epoch, prefix, node, base_theta),
            damping=damping,
        )
        results.append(result)
        theta = result.global_theta.detach().clone()
    return results
