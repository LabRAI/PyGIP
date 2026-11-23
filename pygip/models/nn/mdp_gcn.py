"""
ManualGCN: A GCN implementation that accepts dense adjacency matrices.

Unlike standard PyG/DGL GCN layers that expect edge_index or DGLGraph,
this model performs message passing using dense matrix multiplication.
This is required for MDP defense which operates on eigendecomposed
adjacency matrix shares.
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


AdjType = Union[torch.Tensor, List[torch.Tensor]]


class ManualGCN(nn.Module):
    """
    Two-layer GCN using dense adjacency matrix multiplication.

    Message passing is computed as: H' = A @ H @ W
    where A can be a single matrix or list of shares that sum to the full matrix.

    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension (number of classes)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.dropout_p = float(dropout)

    def _message_passing(
        self,
        A_or_shares: AdjType,
        H: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform message passing via dense matrix multiplication.

        If A_or_shares is a list, sums the results of A_i @ H for each share.
        This allows training on individual shares while inference uses the full sum.
        """
        if isinstance(A_or_shares, list):
            out = None
            for A in A_or_shares:
                A = A.to(dtype=torch.float32, device=H.device)
                part = A @ H
                out = part if out is None else (out + part)
            return out
        A = A_or_shares.to(dtype=torch.float32, device=H.device)
        return A @ H

    def forward(
        self,
        A_or_shares: AdjType,
        X: torch.Tensor,
        dropout: float = None
    ) -> torch.Tensor:
        """
        Forward pass through 2-layer GCN.

        Args:
            A_or_shares: Dense adjacency matrix or list of shares
            X: Node feature matrix [N, in_dim]
            dropout: Override dropout probability (optional)

        Returns:
            Logits tensor [N, out_dim]
        """
        X = X.to(dtype=torch.float32)
        p = self.dropout_p if dropout is None else float(dropout)

        # Layer 1: Message passing -> Linear -> ReLU -> Dropout
        H = self._message_passing(A_or_shares, X)
        H = self.lin1(H)
        H = F.relu(H)
        H = F.dropout(H, p=p, training=self.training)

        # Layer 2: Message passing -> Linear
        H = self._message_passing(A_or_shares, H)
        H = self.lin2(H)

        return H
