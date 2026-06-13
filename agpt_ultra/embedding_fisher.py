from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EmbeddingFisherStepStats:
    active_tokens: int
    max_update_norm: float
    mean_update_norm: float


class EmbeddingFisherPreconditioner:
    """Per-token empirical Fisher blocks for embedding gradients."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        damping: float = 10.0,
        step_scale: float = 1.0,
        decay: float = 1.0,
    ) -> None:
        if damping <= 0:
            raise ValueError("damping must be positive")
        if step_scale < 0:
            raise ValueError("step_scale must be non-negative")
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        self.damping = float(damping)
        self.step_scale = float(step_scale)
        self.decay = float(decay)
        self.blocks = torch.zeros(vocab_size, embedding_size, embedding_size)

    def state_dict(self) -> dict[str, object]:
        return {
            "damping": self.damping,
            "step_scale": self.step_scale,
            "decay": self.decay,
            "blocks": self.blocks,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.damping = float(state["damping"])
        self.step_scale = float(state["step_scale"])
        self.decay = float(state["decay"])
        self.blocks = torch.as_tensor(state["blocks"]).clone()

    @torch.no_grad()
    def step(self, embedding: nn.Embedding) -> EmbeddingFisherStepStats:
        grad = embedding.weight.grad
        if grad is None:
            return EmbeddingFisherStepStats(0, 0.0, 0.0)
        grad = grad.detach()
        active = torch.nonzero(grad.norm(dim=1) > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            embedding.weight.grad = None
            return EmbeddingFisherStepStats(0, 0.0, 0.0)

        blocks = self.blocks.to(device=grad.device, dtype=grad.dtype)
        eye = torch.eye(grad.shape[1], device=grad.device, dtype=grad.dtype)
        update_norms = []
        for token_id in active.tolist():
            token_grad = grad[token_id]
            blocks[token_id].mul_(self.decay).add_(torch.outer(token_grad, token_grad))
            system = blocks[token_id] + self.damping * eye
            delta = -torch.linalg.solve(system, token_grad)
            update = self.step_scale * delta
            embedding.weight[token_id].add_(update)
            update_norms.append(float(update.norm().item()))

        self.blocks = blocks.detach().cpu()
        embedding.weight.grad = None
        update_norm_tensor = torch.tensor(update_norms)
        return EmbeddingFisherStepStats(
            active_tokens=int(active.numel()),
            max_update_norm=float(update_norm_tensor.max().item()),
            mean_update_norm=float(update_norm_tensor.mean().item()),
        )
