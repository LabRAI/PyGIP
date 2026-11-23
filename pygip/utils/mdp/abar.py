"""Build normalized adjacency matrix for MDP defense."""

from __future__ import annotations

from dataclasses import dataclass
import time
import torch


@dataclass
class AbarResult:
    """Result of normalized adjacency matrix construction."""
    Abar: torch.Tensor
    build_time_sec: float


def build_abar_dense(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device
) -> AbarResult:
    """
    Build normalized dense adjacency matrix.

    Computes: Abar = I + D^(-1/2) A D^(-1/2)

    Args:
        edge_index: Edge index tensor [2, E]
        num_nodes: Number of nodes
        device: Torch device

    Returns:
        AbarResult containing the normalized adjacency matrix
    """
    t0 = time.time()

    n = int(num_nodes)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")

    row = edge_index[0].to(device)
    col = edge_index[1].to(device)

    A = torch.zeros((n, n), device=device, dtype=torch.float32)
    A[row, col] = 1.0
    A[col, row] = 1.0
    A.fill_diagonal_(0.0)

    deg = A.sum(dim=1)
    inv_sqrt = torch.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = deg[nz].pow(-0.5)

    A_norm = inv_sqrt.view(n, 1) * A * inv_sqrt.view(1, n)
    Abar = torch.eye(n, device=device, dtype=torch.float32) + A_norm

    return AbarResult(Abar=Abar, build_time_sec=time.time() - t0)
