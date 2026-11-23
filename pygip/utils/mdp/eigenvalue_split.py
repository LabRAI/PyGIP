"""Eigenvalue-based matrix splitting for MDP defense."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class ESNCResult:
    """Result of eigenvalue splitting."""
    shares: List[torch.Tensor]
    eigvals: torch.Tensor
    assignments: torch.Tensor


def es_split_into_nc(
    A: torch.Tensor,
    *,
    nc: int,
    seed: Optional[int] = None,
    assume_symmetric: bool = True,
) -> ESNCResult:
    """
    Split matrix A into nc shares via eigendecomposition.

    Each eigenvalue is randomly assigned to one of nc shares.
    The shares sum to the original matrix: A = sum(shares)

    Args:
        A: Square matrix to split [n, n]
        nc: Number of shares to create
        seed: Random seed for reproducible assignments
        assume_symmetric: If True, uses efficient symmetric eigendecomposition

    Returns:
        ESNCResult with list of matrix shares
    """
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (n,n). Got shape {tuple(A.shape)}")

    if nc < 2:
        raise ValueError("nc must be >= 2")

    if seed is not None:
        torch.manual_seed(seed)

    device = A.device
    n = A.shape[0]

    if assume_symmetric:
        eigvals, U = torch.linalg.eigh(A)
        UT = U.t()
    else:
        eigvals_complex, U_complex = torch.linalg.eig(A)
        eigvals = eigvals_complex.real
        U = U_complex.real
        UT = torch.linalg.pinv(U)

    assignments = torch.randint(low=0, high=nc, size=(n,), device=device)

    shares: List[torch.Tensor] = []
    for k in range(nc):
        mask = (assignments == k)
        lam_k = torch.zeros_like(eigvals)
        lam_k[mask] = eigvals[mask]
        A_k = U @ torch.diag(lam_k) @ UT
        shares.append(A_k)

    shares = [s.to(dtype=torch.float32) for s in shares]
    eigvals = eigvals.to(dtype=torch.float32)

    return ESNCResult(shares=shares, eigvals=eigvals, assignments=assignments)
