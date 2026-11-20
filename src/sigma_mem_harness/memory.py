"""The additive Sigma-memory statistic used by the harness.

This is the small, dependency-light part of the implementation from
``delta-mem-miles``. The writer supplies normalized features and the harness
only needs exact additive ``add`` operations while turns move from native KV
into readable long-term memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SigmaMemory:
    """An additive second moment ``H = sum_t h_t h_t^T``."""

    cov: torch.Tensor
    count: torch.Tensor
    cross: torch.Tensor | None = None

    @classmethod
    def empty(cls, dim: int, *, device=None, dtype=torch.float32) -> "SigmaMemory":
        if dim < 1:
            raise ValueError("memory dimension must be positive")
        return cls(
            cov=torch.zeros(dim, dim, device=device, dtype=dtype),
            count=torch.zeros((), device=device, dtype=dtype),
        )

    @classmethod
    def from_features(
        cls,
        features: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        successors: torch.Tensor | None = None,
    ) -> "SigmaMemory":
        if features.ndim != 2:
            raise ValueError("features must have shape [tokens, dimension]")
        x = features.float()
        if weights is None:
            wx = x
            count = torch.tensor(float(x.shape[0]), device=x.device, dtype=x.dtype)
        else:
            weights = weights.to(device=x.device, dtype=x.dtype).reshape(-1, 1)
            if weights.shape[0] != x.shape[0]:
                raise ValueError("weights and features must have the same token count")
            wx = weights * x
            count = weights.sum()
        cross = None
        if successors is not None:
            if successors.shape != features.shape:
                raise ValueError("successors must have the same shape as features")
            cross = successors.float().transpose(0, 1) @ wx
        return cls(cov=x.transpose(0, 1) @ wx, count=count, cross=cross)

    @property
    def dim(self) -> int:
        return int(self.cov.shape[-1])

    def add(self, other: "SigmaMemory") -> "SigmaMemory":
        if self.dim != other.dim:
            raise ValueError(f"cannot add dimensions {self.dim} and {other.dim}")
        if (self.cross is None) != (other.cross is None):
            raise ValueError("cannot mix memories with and without successor moments")
        cross = None if self.cross is None else self.cross + other.cross.to(self.cross)
        return SigmaMemory(
            cov=self.cov + other.cov.to(self.cov),
            count=self.count + other.count.to(self.count),
            cross=cross,
        )

    def subtract(self, other: "SigmaMemory") -> "SigmaMemory":
        if self.dim != other.dim:
            raise ValueError(f"cannot subtract dimensions {self.dim} and {other.dim}")
        if (self.cross is None) != (other.cross is None):
            raise ValueError("cannot mix memories with and without successor moments")
        cross = None if self.cross is None else self.cross - other.cross.to(self.cross)
        return SigmaMemory(
            cov=self.cov - other.cov.to(self.cov),
            count=self.count - other.count.to(self.count),
            cross=cross,
        )

    def to(self, *args, **kwargs) -> "SigmaMemory":
        return SigmaMemory(
            cov=self.cov.to(*args, **kwargs),
            count=self.count.to(*args, **kwargs),
            cross=None if self.cross is None else self.cross.to(*args, **kwargs),
        )


def rms_normalize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match the parameter-free writer normalization in delta-mem-miles."""

    features = features.float()
    return features * torch.rsqrt(features.square().mean(dim=-1, keepdim=True) + eps)


__all__ = ["SigmaMemory", "rms_normalize"]
