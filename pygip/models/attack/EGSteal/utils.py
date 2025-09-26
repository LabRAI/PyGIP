# utils.py

import numpy as np
import random
import os
from typing import Optional
from sklearn.metrics import roc_auc_score
import logging
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import ExplanationType, ModelMode, ModelTaskLevel
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_embeddings
from torch_geometric.data import Batch, Data
from pygip.models.nn import CAM
from scipy.stats import kendalltau
import math
from scipy.stats import kendalltau
from torch_geometric.data import Batch, Data
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def safe_auc(y_true, y_pred):
    """
    Safely compute AUC, avoiding NaN values.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities

    Returns:
        float: AUC value, returns 0.5 if data does not meet computation conditions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(np.unique(y_true)) == 1:
        return 0.5
        
    if len(np.unique(y_pred)) == 1:
        return 0.5
        
    return roc_auc_score(y_true, y_pred)

def train_loop_target_model(model, dataloader, optimizer, device, args):
    """Train the model for one epoch, returning average loss, accuracy, and AUC."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    criterion = torch.nn.CrossEntropyLoss()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.explanation_mode in ['GNNExplainer', 'PGExplainer']:
            out = model(data.x, data.edge_index, data.batch)
        else:
            logits, out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs

        all_labels.extend(data.y.cpu().numpy())
        probs = F.softmax(out, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total

    binary_labels = np.array(all_labels)
    binary_probs = np.array([prob[1] for prob in all_probs])
    auc = safe_auc(binary_labels, binary_probs)

    return avg_loss, accuracy, auc


def evaluate_loop_target_model(model, dataloader, device, args):
    """Evaluate the model on the validation or test set, returning average loss, accuracy, and AUC."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            if args.explanation_mode in ['GNNExplainer','PGExplainer']:
                out = model(data.x, data.edge_index, data.batch)
            else:
                _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs

            all_labels.extend(data.y.cpu().numpy())
            probs = F.softmax(out, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total

    binary_labels = np.array(all_labels)
    binary_probs = np.array([prob[1] for prob in all_probs])
    auc = safe_auc(binary_labels, binary_probs)

    return avg_loss, accuracy, auc

def convert_edge_scores_to_node_scores(edge_mask, edge_index, num_nodes):
    """
    Convert edge importance scores to node importance scores.
    
    Parameters:
    - edge_mask (Tensor): Edge importance scores, shape [num_edges].
    - edge_index (Tensor): Edge connections, shape [2, num_edges], indicating the two nodes connected by each edge.
    - num_nodes (int): Number of nodes in the graph.
    
    Returns:
    - node_scores (Tensor): Node importance scores, shape [num_nodes].
    """
    # Initialize node importance scores tensor
    node_scores = torch.zeros(num_nodes, device=edge_mask.device) # shape: [num_nodes]

    # Initialize node degrees
    node_degrees = torch.zeros(num_nodes, device=edge_mask.device)  # shape: [num_nodes]

    # Iterate through each edge to calculate the contribution of edge importance to node importance
    for i in range(edge_index.shape[1]):
        node1, node2 = edge_index[:, i]  # Get the two nodes connected by the edge
        importance = edge_mask[i]  # Get the importance score of the edge

        node_scores[node1] += importance
        node_scores[node2] += importance

        node_degrees[node1] += 1
        node_degrees[node2] += 1
    
    node_degrees[node_degrees == 0] = 1  # Avoid division by zero

    # Calculate node importance, normalized by node degree
    node_scores = node_scores / node_degrees

    return node_scores

def custom_collate(batch):
    """
    Custom collate function to batch PyTorch Geometric Data objects.

    Args:
        batch: List[Data]

    Returns:
        Batch object
    """
    return Batch.from_data_list(batch)


def process_query_dataset(query_dataset):
    """
    Process query_dataset uniformly, setting the target model prediction as target_pred,
    and retaining the original label y and node_mask.

    Parameters:
    - query_dataset: List[dict], containing 'original_data', 'pred', and 'node_mask' fields

    Returns:
    - processed_data_list: List[Data], each Data object contains original features, y, target_pred, and node_mask
    """
    processed_data_list = []
    for sample in query_dataset:
        original_data = sample['original_data']
        pred = sample['pred']
        node_mask = sample['node_mask']

        # Ensure pred is an integer
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        elif isinstance(pred, (list, np.ndarray)):
            pred = pred[0]

        # Create new Data object
        new_data = Data(
            x=original_data.x,
            edge_index=original_data.edge_index,
            edge_attr=getattr(original_data, 'edge_attr', None),
            y=original_data.y,
            target_pred=torch.tensor(pred, dtype=torch.long, device=original_data.x.device),
            node_mask=node_mask
        )

        if hasattr(original_data, 'batch'):
            new_data.batch = original_data.batch

        processed_data_list.append(new_data)
    return processed_data_list

def augment(dataset, augmentor, augmentation_ratio, operation_ratio=0.1, augmentation_type='combined'):
    """
    Generate augmented samples based on the chosen augmentation strategy.

    Parameters:
    - dataset: Original training dataset (list of Data objects)
    - augmentor: DataAugmentor instance
    - augmentation_ratio: Augmentation ratio (e.g., 0.2 means generating augmented samples equal to 20% of original data)
    - operation_ratio: Ratio for adding or removing operations
    - augmentation_type: Augmentation type, 'drop_node', 'drop_edge', 'add_edge', or 'combined'

    Returns:
    - augmented_data_list: List of augmented Data objects
    """
    augmented_data_list = []
    num_original_samples = len(dataset)
    num_augmented_samples = int(num_original_samples * augmentation_ratio)

    if num_augmented_samples == 0:
        return augmented_data_list

    # Group samples by label
    label_to_samples = defaultdict(list)
    for sample in dataset:
        label = sample.y.item()
        label_to_samples[label].append(sample)

    # Calculate inverse frequency and normalize
    label_weights = {}
    total_inverse = 0
    for label, samples in label_to_samples.items():
        Ni = len(samples)
        if Ni == 0:
            continue
        inverse_freq = 1.0 / Ni
        label_weights[label] = inverse_freq
        total_inverse += inverse_freq

    for label in label_weights:
        label_weights[label] /= total_inverse

    # Calculate number of augmented samples per label
    num_augmented_samples_per_label = {}
    for label, weight in label_weights.items():
        num_aug = int(num_augmented_samples * weight)
        num_augmented_samples_per_label[label] = num_aug

    # Allocate remaining samples
    remaining = num_augmented_samples - sum(num_augmented_samples_per_label.values())
    labels = list(label_weights.keys())
    for i in range(remaining):
        label = labels[i % len(labels)]
        num_augmented_samples_per_label[label] += 1

    # Generate augmented samples
    for label, num_aug in num_augmented_samples_per_label.items():
        samples = label_to_samples[label]
        if len(samples) < 1:
            print(f"Label {label} has insufficient samples to generate augmented samples (requires at least 1 sample).")
            continue

        for _ in range(num_aug):
            try:
                sample = random.choice(samples)

                if augmentation_type == 'drop_node':
                    augmented_data = augmentor.drop_node(sample, drop_ratio=operation_ratio)
                elif augmentation_type == 'drop_edge':
                    augmented_data = augmentor.drop_edge(sample, drop_ratio=operation_ratio)
                elif augmentation_type == 'add_edge':
                    augmented_data = augmentor.add_edge(sample, add_ratio=operation_ratio)
                elif augmentation_type == 'combined':
                    augmented_data = augmentor.combined_augmentation(
                        sample,
                        drop_node_ratio=operation_ratio,
                        drop_edge_ratio=operation_ratio,
                        add_edge_ratio=operation_ratio
                    )
                else:
                    print(f"Unknown augmentation type: {augmentation_type}")
                    continue

                if augmented_data is None:
                    continue

                # Skip single-node graphs
                if augmented_data.x.size(0) <= 1:
                    continue

                augmented_data_list.append(augmented_data)

            except Exception as e:
                print(f"Error generating augmented sample for label {label}: {e}")
                continue

    return augmented_data_list


def train(model, dataloader, optimizer, device, align_weight=1.0, criterion=None, ranknet_loss_fn=None):
    """
    Training function, using target_pred as labels and incorporating RankNet loss.
    If align_weight is 0, skip the calculation of RankNet loss.
    """
    model.train()
    total_loss_pred = 0.0
    total_ranknet_loss = 0.0
    total_samples = 0

    ex = CAM(model)

    for batch_samples in dataloader:
        optimizer.zero_grad()

        all_data = batch_samples.to(device)

        node_emb, out_surr = model(all_data.x, all_data.edge_index, all_data.batch)

        loss_pred = criterion(out_surr, all_data.target_pred)

        # Calculate CAM scores
        cam_scores = ex.get_cam_scores(all_data.target_pred, all_data.batch)

        node_masks = all_data.node_mask

        batch_ids = all_data.batch  # [total_num_nodes]

        ranknet_loss = torch.tensor(0.0, device=device)
        if align_weight != 0:
            ranknet_loss = ranknet_loss_fn(cam_scores, node_masks, batch_ids)

        # Total loss
        total_batch_loss = loss_pred + align_weight * ranknet_loss

        total_batch_loss.backward()
        optimizer.step()

        batch_size = len(batch_samples)
        total_loss_pred += loss_pred.item() * batch_size
        total_ranknet_loss += ranknet_loss.item() * batch_size
        total_samples += batch_size

    # Calculate average loss
    avg_loss_pred = total_loss_pred / total_samples
    avg_ranknet_loss = total_ranknet_loss / total_samples

    return avg_loss_pred, avg_ranknet_loss


def eval(model, dataloader, device):
    """
    Evaluate the model on the validation set for accuracy and AUC, using target_pred as labels.

    Returns: validation accuracy, validation AUC
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_samples in dataloader:
            batch = batch_samples.to(device)
            target_preds_tensor = batch.target_pred

            node_emb, out_surr = model(batch.x, batch.edge_index, batch.batch)
            pred = out_surr.argmax(dim=1)

            total_correct += pred.eq(target_preds_tensor).sum().item()
            total_samples += len(batch_samples)

            all_targets.extend(target_preds_tensor.cpu().numpy())
            if out_surr.size(1) > 1:
                all_probs.extend(F.softmax(out_surr, dim=1)[:, 1].cpu().numpy())
            else:
                all_probs.extend(torch.sigmoid(out_surr).cpu().numpy())

    accuracy = total_correct / total_samples

    try:
        auc = safe_auc(all_targets, all_probs)
    except ValueError:
        auc = float('nan')

    return accuracy, auc


def calculate_rank_correlation(pred_scores, true_scores, batch_ids):
    correlations = []
    for b in torch.unique(batch_ids):
        mask = (batch_ids == b)
        p = pred_scores[mask].cpu().numpy()
        t = true_scores[mask].cpu().numpy()
        corr, _ = kendalltau(p, t)
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations)


def calculate_order_accuracy(pred_scores, true_scores, batch_ids):
    """
    Parameters:
    - pred_scores: [total_num_nodes], predicted node importance scores
    - true_scores: [total_num_nodes], true node importance scores
    - batch_ids: [total_num_nodes], graph index for each node

    Returns:
    - order_accuracy: Mean order accuracy across all graphs
    """
    unique_batch = torch.unique(batch_ids)
    per_graph_accuracies = []

    for b in unique_batch:
        mask = (batch_ids == b)
        p = pred_scores[mask]
        t = true_scores[mask]
        num_nodes = p.size(0)

        if num_nodes < 2:
            continue

        # Generate all possible node pairs (i, j) where i < j
        indices_i, indices_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=p.device)

        s_i = t[indices_i]
        s_j = t[indices_j]

        p_i = p[indices_i]
        p_j = p[indices_j]

        # Calculate true relation
        # 1 for s_i > s_j, -1 for s_i < s_j, 0 for s_i == s_j
        true_relation = torch.where(s_i > s_j, torch.ones_like(s_i),
                                    torch.where(s_i < s_j, torch.ones_like(s_i) * -1, torch.zeros_like(s_i)))

        pred_relation = torch.where(p_i > p_j, torch.ones_like(p_i),
                                    torch.where(p_i < p_j, torch.ones_like(p_i) * -1, torch.zeros_like(p_i)))

        correct = true_relation.eq(pred_relation).float()
        correct_total = correct.sum().item()
        total = correct.numel()

        if total > 0:
            graph_accuracy = correct_total / total
            per_graph_accuracies.append(graph_accuracy)

    if len(per_graph_accuracies) == 0:
        return float('nan')

    mean_accuracy = sum(per_graph_accuracies) / len(per_graph_accuracies)

    return mean_accuracy


def test(model, dataloader, device):
    """
    Evaluate the model on the test set for accuracy, AUC, fidelity, order accuracy, and rank correlation.
    """
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_probs = []
    all_predictions = []
    all_target_preds = []
    all_pred_scores = []
    all_true_scores = []
    all_batch_ids = []
    graph_offset = 0

    ex = CAM(model)

    with torch.no_grad():
        for batch_samples in dataloader:
            batch = batch_samples.to(device)
            true_labels_tensor = batch.y
            target_preds_tensor = batch.target_pred

            node_emb, out_surr = model(batch.x, batch.edge_index, batch.batch)
            pred = out_surr.argmax(dim=1)

            correct += pred.eq(true_labels_tensor).sum().item()
            total += len(batch_samples)

            all_predictions.append(pred)
            all_target_preds.append(target_preds_tensor)

            all_targets.extend(true_labels_tensor.cpu().numpy())
            if out_surr.size(1) > 1:
                all_probs.extend(F.softmax(out_surr, dim=1)[:, 1].cpu().numpy())
            else:
                all_probs.extend(torch.sigmoid(out_surr).cpu().numpy())

            cam_scores = ex.get_cam_scores(true_labels_tensor, batch.batch)
            all_pred_scores.append(cam_scores.cpu())
            all_true_scores.append(batch.node_mask.cpu())
            all_batch_ids.append(batch.batch.cpu() + graph_offset)

            graph_offset += batch.num_graphs

    # Calculate accuracy
    accuracy = correct / total

    # Calculate AUC
    try:
        auc = safe_auc(all_targets, all_probs)
    except ValueError:
        auc = float('nan')

    # Calculate fidelity
    all_predictions = torch.cat(all_predictions)
    all_target_preds = torch.cat(all_target_preds)
    total_fidelity = all_target_preds.size(0)
    fidelity_score = float('nan') if total_fidelity == 0 else \
                     all_predictions.eq(all_target_preds).sum().item() / total_fidelity

    # Calculate rank correlation and order accuracy
    all_pred_scores = torch.cat(all_pred_scores)
    all_true_scores = torch.cat(all_true_scores)
    all_batch_ids = torch.cat(all_batch_ids)
    order_accuracy = calculate_order_accuracy(all_pred_scores, all_true_scores, all_batch_ids)
    rank_correlation = calculate_rank_correlation(all_pred_scores, all_true_scores, all_batch_ids)

    return accuracy, auc, fidelity_score, order_accuracy, rank_correlation

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()
        pass

    def forward(self, pred_scores, true_scores, batch_ids):
        """
        Compute RankNet loss for batched data.

        Parameters:
        - pred_scores: [total_num_nodes], predicted node importance scores (CAM)
        - true_scores: [total_num_nodes], true node importance scores (node_mask)
        - batch_ids: [total_num_nodes], graph index for each node

        Returns:
        - loss: Average RankNet loss
        """
        unique_batch = torch.unique(batch_ids)
        total_loss = 0.0
        count = 0

        for b in unique_batch:
            mask = (batch_ids == b)
            p = pred_scores[mask]
            t = true_scores[mask]
            num_nodes = p.size(0)

            if num_nodes < 2:
                continue

            indices_i, indices_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=p.device)

            s_i = t[indices_i]
            s_j = t[indices_j]

            p_i = p[indices_i]
            p_j = p[indices_j]

            # Label y_ij = 1 if s_i > s_j, else 0, if s_i == s_j then y_ij = 0.5
            y_ij = torch.zeros_like(s_i, dtype=torch.float, device=p.device)
            y_ij[s_i > s_j] = 1.0
            y_ij[s_i == s_j] = 0.5

            surr_diff = p_i - p_j
            sigmoid_diff = torch.sigmoid(surr_diff)
            loss = F.binary_cross_entropy(sigmoid_diff, y_ij, reduction='mean')

            total_loss += loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)

        return total_loss / count

class DataAugmentor:
    def __init__(self):
        pass

    def drop_node(self, sample, drop_ratio=0.1) -> Optional[Data]:
        """
        Drop nodes with the lowest importance and their associated edges.

        Args:
            sample: Input graph data (torch_geometric.data.Data)
            drop_ratio: Proportion of nodes to drop

        Returns:
            Augmented graph data, or None if invalid
        """
        node_mask = sample.node_mask if hasattr(sample, 'node_mask') else torch.ones(sample.num_nodes, dtype=torch.float, device=sample.x.device)

        num_nodes = sample.num_nodes
        num_drop = int(drop_ratio * num_nodes)
        if num_drop == 0 or num_drop >= num_nodes:
            return None

        _, drop_indices = torch.topk(node_mask, k=num_drop, largest=False)

        keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=sample.x.device)
        keep_mask[drop_indices] = False
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze()

        new_edge_index, edge_attr, edge_mask = subgraph(
            keep_indices,
            sample.edge_index,
            edge_attr=sample.edge_attr if hasattr(sample, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=True
        )

        new_x = sample.x[keep_indices]
        new_node_mask = node_mask[keep_indices]

        # Create augmented graph data
        augmented_data = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=sample.y,
            target_pred=sample.target_pred if hasattr(sample, 'target_pred') else None,
            node_mask=new_node_mask
        )

        return augmented_data

    def drop_edge(self, sample, drop_ratio=0.1) -> Optional[Data]:
        """
        Drop edges between low-importance nodes.

        Args:
            sample: Input graph data (torch_geometric.data.Data)
            drop_ratio: Proportion of nodes considered low-importance

        Returns:
            Augmented graph data, or None if invalid
        """
        node_mask = sample.node_mask if hasattr(sample, 'node_mask') else torch.ones(sample.num_nodes, dtype=torch.float, device=sample.x.device)

        num_nodes = sample.num_nodes
        num_low_importance = int(drop_ratio * num_nodes)
        if num_low_importance < 2:
            return None

        _, indices = torch.topk(node_mask, k=num_low_importance, largest=False)
        low_importance_nodes = set(indices.cpu().tolist())
        src_nodes = sample.edge_index[0]
        dst_nodes = sample.edge_index[1]

        edge_mask = torch.ones(sample.edge_index.size(1), dtype=torch.bool, device=sample.x.device)
        for node in low_importance_nodes:
            mask = (src_nodes == node) | (dst_nodes == node)
            edge_mask = edge_mask & ~mask

        new_edge_index = sample.edge_index[:, edge_mask]
        edge_attr = sample.edge_attr[edge_mask] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None

        # Create augmented graph data
        augmented_data = Data(
            x=sample.x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=sample.y,
            target_pred=sample.target_pred if hasattr(sample, 'target_pred') else None,
            node_mask=node_mask
        )

        return augmented_data

    def add_edge(self, sample, add_ratio=0.1) -> Optional[Data]:
        """
        Add edges between low-importance nodes.

        Args:
            sample: Input graph data (torch_geometric.data.Data)
            add_ratio: Proportion of nodes considered low-importance

        Returns:
            Augmented graph data, or None if invalid
        """
        node_mask = sample.node_mask if hasattr(sample, 'node_mask') else torch.ones(sample.num_nodes, dtype=torch.float, device=sample.x.device)
        num_nodes = sample.num_nodes
        num_low_importance = int(add_ratio * num_nodes)
        if num_low_importance < 2:
            return None

        _, indices = torch.topk(node_mask, k=num_low_importance, largest=False)
        low_importance_nodes = indices.tolist()

        new_edges = []
        for i in range(len(low_importance_nodes)):
            for j in range(i + 1, len(low_importance_nodes)):
                u, v = low_importance_nodes[i], low_importance_nodes[j]
                new_edges.append([u, v])
                new_edges.append([v, u])

        if not new_edges:
            return None

        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long, device=sample.x.device).t()
        new_edge_index = torch.cat([sample.edge_index, new_edges_tensor], dim=1)

        # Remove duplicate edges
        unique_edges, unique_idx = torch.unique(new_edge_index.t(), dim=0, return_inverse=True)
        new_edge_index = unique_edges.t()

        # Handle edge attributes
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            num_new_edges = new_edge_index.size(1) - sample.edge_index.size(1)
            if num_new_edges > 0:
                default_attr = torch.ones((num_new_edges, sample.edge_attr.size(1)),
                                         dtype=sample.edge_attr.dtype,
                                         device=sample.x.device)
                edge_attr = torch.cat([sample.edge_attr, default_attr], dim=0)
            else:
                edge_attr = sample.edge_attr
            edge_attr = edge_attr[unique_idx]
        else:
            edge_attr = None

        # Create augmented graph data
        augmented_data = Data(
            x=sample.x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=sample.y,
            target_pred=sample.target_pred if hasattr(sample, 'target_pred') else None,
            node_mask=node_mask
        )

        return augmented_data

    def combined_augmentation(self, sample, drop_node_ratio=0.1, drop_edge_ratio=0.1, add_edge_ratio=0.1) -> Optional[Data]:
        """
        Randomly apply one of the graph augmentation methods (drop node, drop edge, add edge).

        Args:
            sample: Input graph data (torch_geometric.data.Data)
            drop_node_ratio: Proportion of nodes to drop
            drop_edge_ratio: Proportion of nodes for edge dropping
            add_edge_ratio: Proportion of nodes for edge addition

        Returns:
            Augmented graph data or None (if augmentation fails)
        """
        chosen_method = random.choice(['drop_node', 'drop_edge', 'add_edge'])

        try:
            if chosen_method == 'drop_node':
                return self.drop_node(sample, drop_ratio=drop_node_ratio)
            elif chosen_method == 'drop_edge':
                return self.drop_edge(sample, drop_ratio=drop_edge_ratio)
            elif chosen_method == 'add_edge':
                return self.add_edge(sample, add_ratio=add_edge_ratio)

        except Exception as e:
            print(f"Error occurred during data augmentation: {str(e)}")
            return None




class PGExplainer(ExplainerAlgorithm):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.

    Internally, it utilizes a neural network to identify subgraph structures
    that play a crucial role in the predictions made by a GNN.
    Importantly, the :class:`PGExplainer` needs to be trained via
    :meth:`~PGExplainer.train` before being able to generate explanations:

    .. code-block:: python

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(...),
        )

        # Train against a variety of node-level or graph-level predictions:
        for epoch in range(30):
            for index in [...]:  # Indices to train against.
                loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                                 target=target, index=index)

        # Get the final explanations:
        explanation = explainer(x, edge_index, target=target, index=0)

    Args:
        epochs (int): The number of epochs to train.
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.01,
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
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features of a
                homogeneous graph.
            edge_index (torch.Tensor): The input edge indices of a homogeneous
                graph.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)
        edge_mask = self._concrete_sample(logits, temperature)
        set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))
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

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self._curr_epoch < self.epochs - 1:  # Safety check:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fully "
                             f"trained (got {self._curr_epoch + 1} epochs "
                             f"from {self.epochs} epochs). Please first train "
                             f"the underlying explainer model by running "
                             f"`explainer.algorithm.train(...)`.")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)

        edge_mask = self._post_process_mask(logits, hard_edge_mask,
                                            apply_sigmoid=True)

        return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:

        if self.model_config.mode == ModelMode.binary_classification:
            # loss = self._loss_binary_classification(y_hat, y)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss