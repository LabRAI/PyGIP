import os
import random
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.nn import ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import ExplanationType, ModelMode, ModelTaskLevel
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_embeddings, subgraph


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def safe_auc(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(np.unique(y_true)) == 1:
        return 0.5
    if len(np.unique(y_pred)) == 1:
        return 0.5
    return roc_auc_score(y_true, y_pred)


class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_scores, true_scores, batch_ids):
        unique_batch = torch.unique(batch_ids)
        total_loss = 0.0
        count = 0

        for b in unique_batch:
            mask = batch_ids == b
            p = pred_scores[mask]
            t = true_scores[mask]
            num_nodes = p.size(0)
            if num_nodes < 2:
                continue

            indices_i, indices_j = torch.triu_indices(
                num_nodes, num_nodes, offset=1, device=p.device
            )

            s_i = t[indices_i]
            s_j = t[indices_j]
            p_i = p[indices_i]
            p_j = p[indices_j]

            y_ij = torch.zeros_like(s_i, dtype=torch.float, device=p.device)
            y_ij[s_i > s_j] = 1.0
            y_ij[s_i == s_j] = 0.5

            sigmoid_diff = torch.sigmoid(p_i - p_j)
            loss = F.binary_cross_entropy(sigmoid_diff, y_ij, reduction="mean")

            total_loss += loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)
        return total_loss / count


class DataAugmentor:
    def drop_node(self, sample: Data, drop_ratio=0.1) -> Optional[Data]:
        node_mask = (
            sample.node_mask
            if hasattr(sample, "node_mask")
            else torch.ones(sample.num_nodes, dtype=torch.float, device=sample.x.device)
        )
        num_nodes = sample.num_nodes
        num_drop = int(drop_ratio * num_nodes)
        if num_drop == 0 or num_drop >= num_nodes:
            return None

        _, drop_indices = torch.topk(node_mask, k=num_drop, largest=False)
        keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=sample.x.device)
        keep_mask[drop_indices] = False
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze()

        new_edge_index, edge_attr, _ = subgraph(
            keep_indices,
            sample.edge_index,
            edge_attr=sample.edge_attr if hasattr(sample, "edge_attr") else None,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=True,
        )

        new_x = sample.x[keep_indices]
        new_node_mask = node_mask[keep_indices]

        return Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=sample.y,
            target_pred=sample.target_pred if hasattr(sample, "target_pred") else None,
            node_mask=new_node_mask,
        )

    def drop_edge(self, sample: Data, drop_ratio=0.1) -> Optional[Data]:
        node_mask = (
            sample.node_mask
            if hasattr(sample, "node_mask")
            else torch.ones(sample.num_nodes, dtype=torch.float, device=sample.x.device)
        )
        num_nodes = sample.num_nodes
        num_low = int(drop_ratio * num_nodes)
        if num_low < 2:
            return None

        _, indices = torch.topk(node_mask, k=num_low, largest=False)
        low_nodes = set(indices.cpu().tolist())

        src_nodes = sample.edge_index[0]
        dst_nodes = sample.edge_index[1]
        edge_keep = torch.ones(sample.edge_index.size(1), dtype=torch.bool, device=sample.x.device)

        for node in low_nodes:
            mask = (src_nodes == node) | (dst_nodes == node)
            edge_keep = edge_keep & ~mask

        new_edge_index = sample.edge_index[:, edge_keep]
        edge_attr = sample.edge_attr[edge_keep] if hasattr(sample, "edge_attr") and sample.edge_attr is not None else None

        return Data(
            x=sample.x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=sample.y,
            target_pred=sample.target_pred if hasattr(sample, "target_pred") else None,
            node_mask=node_mask,
        )

    def add_edge(self, sample: Data, add_ratio=0.1) -> Optional[Data]:
        node_mask = (
            sample.node_mask
            if hasattr(sample, "node_mask")
            else torch.ones(sample.num_nodes, dtype=torch.float, device=sample.x.device)
        )
        num_nodes = sample.num_nodes
        num_low = int(add_ratio * num_nodes)
        if num_low < 2:
            return None

        _, indices = torch.topk(node_mask, k=num_low, largest=False)
        low_nodes = indices.tolist()

        new_edges = []
        for i in range(len(low_nodes)):
            for j in range(i + 1, len(low_nodes)):
                u, v = low_nodes[i], low_nodes[j]
                new_edges.append([u, v])
                new_edges.append([v, u])

        if not new_edges:
            return None

        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long, device=sample.x.device).t()
        new_edge_index = torch.cat([sample.edge_index, new_edges_tensor], dim=1)

        unique_edges, unique_idx = torch.unique(new_edge_index.t(), dim=0, return_inverse=True)
        new_edge_index = unique_edges.t()

        if hasattr(sample, "edge_attr") and sample.edge_attr is not None:
            num_new_edges = new_edge_index.size(1) - sample.edge_index.size(1)
            if num_new_edges > 0:
                default_attr = torch.ones((num_new_edges, sample.edge_attr.size(1)),
                                          dtype=sample.edge_attr.dtype, device=sample.x.device)
                edge_attr = torch.cat([sample.edge_attr, default_attr], dim=0)
            else:
                edge_attr = sample.edge_attr
            edge_attr = edge_attr[unique_idx]
        else:
            edge_attr = None

        return Data(
            x=sample.x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=sample.y,
            target_pred=sample.target_pred if hasattr(sample, "target_pred") else None,
            node_mask=node_mask,
        )

    def combined_augmentation(self, sample: Data,
                              drop_node_ratio=0.1, drop_edge_ratio=0.1, add_edge_ratio=0.1) -> Optional[Data]:
        chosen = random.choice(["drop_node", "drop_edge", "add_edge"])
        try:
            if chosen == "drop_node":
                return self.drop_node(sample, drop_ratio=drop_node_ratio)
            if chosen == "drop_edge":
                return self.drop_edge(sample, drop_ratio=drop_edge_ratio)
            return self.add_edge(sample, add_ratio=add_edge_ratio)
        except Exception:
            return None


class PGExplainer(ExplainerAlgorithm):
    coeffs = {
        "edge_size": 0.05,
        "edge_ent": 1.0,
        "temp": [5.0, 2.0],
        "bias": 0.01,
    }

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1

    def reset_parameters(self):
        reset(self.mlp)

    def train(self, epoch: int, model: torch.nn.Module, x: Tensor, edge_index: Tensor, *,
              target: Tensor, index: Optional[Union[int, Tensor]] = None, **kwargs):
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError("Heterogeneous graphs not supported")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError("index required for node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError("index must be scalar")

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]
        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)
        edge_mask = self._concrete_sample(logits, temperature)
        set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index, num_nodes=x.size(0))
            edge_mask = edge_mask[hard_edge_mask]

        y_hat, y = model(x, edge_index, **kwargs), target
        if index is not None:
            y_hat, y = y_hat[index], y[index]

        loss = self._loss(y_hat, y, edge_mask)
        loss.backward()
        self.optimizer.step()

        clear_masks(model)
        self._curr_epoch = epoch
        return float(loss)

    def forward(self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, *,
                target: Tensor, index: Optional[Union[int, Tensor]] = None, **kwargs) -> Explanation:
        if self._curr_epoch < self.epochs - 1:
            raise ValueError("PGExplainer not fully trained yet")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError("index required for node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError("index must be scalar")
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index, num_nodes=x.size(0))

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]
        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)
        edge_mask = self._post_process_mask(logits, hard_edge_mask, apply_sigmoid=True)
        return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        if self.explainer_config.explanation_type != ExplanationType.phenomenon:
            logging.error("PGExplainer only supports phenomenon explanations")
            return False
        if self.model_config.task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error("PGExplainer supports node-level or graph-level only")
            return False
        if self.explainer_config.node_mask_type is not None:
            logging.error("PGExplainer does not support feature masks")
            return False
        return True

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor, index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs["temp"]
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
        bias = self.coeffs["bias"]
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        else:
            loss = self._loss_regression(y_hat, y)

        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs["edge_size"]
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs["edge_ent"]
        return loss + size_loss + mask_ent_loss

