"""Differential privacy for node features using Laplace mechanism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DPResult:
    """Result of differential privacy application."""
    X_dp: torch.Tensor
    epsilon: float
    delta: float
    scale: float


def sample_laplace(
    shape,
    *,
    scale: float,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Sample from Laplace distribution."""
    U = torch.rand(shape, device=device, dtype=dtype) - 0.5
    noise = -scale * torch.sign(U) * torch.log1p(-2.0 * torch.abs(U))
    return noise


def dp_features_laplace(
    X: torch.Tensor,
    *,
    epsilon: float,
    delta: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    seed: Optional[int] = None,
) -> DPResult:
    """
    Apply differential privacy to features using Laplace mechanism.

    Args:
        X: Feature matrix [N, F]
        epsilon: Privacy budget (larger = less noise, inf = no noise)
        delta: Sensitivity parameter
        clip_min: Minimum feature value for clipping
        clip_max: Maximum feature value for clipping
        seed: Random seed for reproducibility

    Returns:
        DPResult with DP-protected features
    """
    if epsilon == float("inf"):
        X_clipped = torch.clamp(X, min=clip_min, max=clip_max)
        return DPResult(X_dp=X_clipped, epsilon=epsilon, delta=delta, scale=0.0)

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 (or inf).")

    if seed is not None:
        torch.manual_seed(seed)

    scale = float(delta) / float(epsilon)
    X_clipped = torch.clamp(X, min=clip_min, max=clip_max)
    noise = sample_laplace(
        X_clipped.shape,
        scale=scale,
        device=X_clipped.device,
        dtype=X_clipped.dtype
    )
    X_noisy = X_clipped + noise
    X_dp = torch.clamp(X_noisy, min=clip_min, max=clip_max)

    return DPResult(X_dp=X_dp, epsilon=epsilon, delta=delta, scale=scale)
