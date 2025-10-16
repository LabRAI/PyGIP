"""
run_bboxve.py — Backdoor-based Ownership Verification (BBoxVe) in PyG.

This script:
- Injects a backdoor watermark trigger into node features.
- Trains a target model and an extracted surrogate model.
- Evaluates clean and backdoor performance (TCA, TBA, ECA, EBA).
- Loops over datasets and models automatically.
- Saves all results to results/BboxVe_results.csv
"""

import os, sys
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygip.models.nn.pyg_backbones import GCN, GAT, GraphSAGE, GIN, SGC

# from torch_geometric.nn import GINConv, SGConv
import torch.nn as nn





# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inject_backdoor(data, node_indices, num_features, fixed_val=10, trigger_size=35):
    """Inject backdoor trigger on selected nodes."""
    poisoned_x = data.x.clone()
    poisoned_y = data.y.clone()
    least_class = torch.bincount(data.y).argmin()

    for idx in node_indices:
        feat_ids = torch.randperm(num_features)[:trigger_size]
        poisoned_x[idx, feat_ids] = fixed_val
        poisoned_y[idx] = least_class

    return poisoned_x, poisoned_y


def train_model(model, data, train_idx, epochs=50, lr=0.01, device="cpu"):
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        opt.step()

    return model


def evaluate(model, data, clean_idx, backdoor_idx):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)

    clean_acc = (preds[clean_idx] == data.y[clean_idx]).float().mean().item()
    backdoor_acc = (preds[backdoor_idx] == data.y[backdoor_idx]).float().mean().item()

    return clean_acc * 100, backdoor_acc * 100


# ----------------------------
# Main Experiment
# ----------------------------
def run_experiment(dataset_name, model_type, with_backdoor=True, device="cpu"):
    dataset = Planetoid(root=f"data/{dataset_name}", name=dataset_name)
    data = dataset[0].to(device)
    num_nodes = data.num_nodes

    idx = torch.randperm(num_nodes)
    train_idx = idx[: int(0.2 * num_nodes)]
    surr_idx = idx[int(0.2 * num_nodes): int(0.6 * num_nodes)]
    test_idx = idx[int(0.6 * num_nodes):]

    bd_train_idx = train_idx[torch.randperm(len(train_idx))[: int(0.15 * len(train_idx))]]
    bd_test_idx = test_idx[torch.randperm(len(test_idx))[: int(0.10 * len(test_idx))]]

    if with_backdoor:
        data.x, data.y = inject_backdoor(data, bd_train_idx, dataset.num_features)
        data.x, data.y = inject_backdoor(data, bd_test_idx, dataset.num_features)

    # Select model
    if model_type == "GCN":
        model_fn = lambda: GCN(dataset.num_features, 64, dataset.num_classes)
    elif model_type == "GAT":
        model_fn = lambda: GAT(dataset.num_features, 64, dataset.num_classes)
    elif model_type == "GraphSAGE":
        model_fn = lambda: GraphSAGE(dataset.num_features, 64, dataset.num_classes)
    elif model_type == "GIN":
        model_fn = lambda: GIN(dataset.num_features, 64, dataset.num_classes)
    elif model_type == "SGC":
        model_fn = lambda: SGC(dataset.num_features, dataset.num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    target = train_model(model_fn(), data, train_idx, device=device)

    surr_data = data if with_backdoor else dataset[0].clone()
    surrogate = train_model(model_fn(), surr_data, surr_idx, device=device)

    clean_idx = torch.tensor(list(set(test_idx.tolist()) - set(bd_test_idx.tolist())), dtype=torch.long)
    TCA, TBA = evaluate(target, data, clean_idx, bd_test_idx)
    ECA, EBA = evaluate(surrogate, data, clean_idx, bd_test_idx)

    return {
        "Dataset": dataset_name,
        "Model": model_type,
        "Setting": "With Backdoor" if with_backdoor else "Without Backdoor",
        "TCA": TCA,
        "ECA": ECA,
        "TBA": TBA,
        "EBA": EBA
    }


# ----------------------------
# Runner
# ----------------------------
if __name__ == "__main__":
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    out_file = "results/BboxVe_results.csv"

    datasets = ["Cora", "CiteSeer", "PubMed"]
    models = ["GCN", "GAT", "GraphSAGE", "GIN", "SGC"]

    all_results = []

    for dataset in datasets:
        for model_type in models:
            print(f"\n=== Running {dataset} | {model_type} | With Backdoor ===")
            res = run_experiment(dataset, model_type, with_backdoor=True, device=device)
            all_results.append(res)

    df = pd.DataFrame(all_results)
    if os.path.exists(out_file):
        df.to_csv(out_file, mode="a", header=False, index=False)
    else:
        df.to_csv(out_file, index=False)

    print("\n=== All Table 3 Rows Added ===")
    print(df)
