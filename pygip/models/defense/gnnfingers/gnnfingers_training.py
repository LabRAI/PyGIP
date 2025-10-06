# utils/training.py

import torch
import torch.nn.functional as F

# ========== NODE CLASSIFICATION TRAINING ==========
def train_node(model, data, optimizer, device="cpu"):
    """
    Train a node classification model for one epoch.
    """
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_node(model, data, mask, device="cpu"):
    """
    Evaluate a node classification model on a given mask.
    """
    model.eval()
    data = data.to(device)

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
    return acc


# ========== GRAPH CLASSIFICATION TRAINING ==========
def train_graph(model, loader, optimizer, device="cpu"):
    """
    Train a graph classification model for one epoch.
    """
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def test_node(model, data, mask, device):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
    return acc
import torch

@torch.no_grad()
def test_graph(model, loader, device):
    """
    Evaluate a graph classification model on a dataset.
    
    Args:
        model: PyTorch Geometric model (e.g., GCN, GIN, GraphSAGE).
        loader: DataLoader containing batches of graphs.
        device: torch.device (cpu or cuda).
    
    Returns:
        accuracy (float): classification accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    acc = correct / total if total > 0 else 0
    return acc
