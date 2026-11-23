"""Generate overlapping training masks for federated calculators."""

from __future__ import annotations

from typing import List
import torch


def make_overlapping_train_masks(
    train_mask: torch.Tensor,
    nc: int,
    seed: int,
    keep_ratio: float = 0.8
) -> List[torch.Tensor]:
    """
    Create overlapping training masks for nc calculators.

    Each calculator sees keep_ratio fraction of the training nodes,
    with different random subsets to ensure diversity.

    Args:
        train_mask: Boolean mask of training nodes
        nc: Number of calculators
        seed: Random seed
        keep_ratio: Fraction of training nodes each calculator sees

    Returns:
        List of nc training masks
    """
    if train_mask.dtype != torch.bool:
        raise ValueError("train_mask must be a boolean tensor")
    if nc < 1:
        raise ValueError("nc must be >= 1")
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1]")

    num_nodes = train_mask.numel()
    device = train_mask.device
    train_idx = torch.where(train_mask)[0]
    n_train = train_idx.numel()
    keep_n = max(1, int(round(n_train * keep_ratio)))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    masks: List[torch.Tensor] = []
    used_signatures = set()

    for _i in range(nc):
        for _attempt in range(50):
            perm = torch.randperm(n_train, generator=g)
            chosen = train_idx[perm[:keep_n]].to(device)
            m = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            m[chosen] = True
            sig = tuple(sorted(chosen.tolist()))
            if sig not in used_signatures:
                used_signatures.add(sig)
                masks.append(m)
                break
        else:
            masks.append(m)

    return masks
