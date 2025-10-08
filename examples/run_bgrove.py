"""
examples/run_bgrove.py

Integration of BGrOVe experiment (Table 4 reproduction) using PyGIP datasets and models.
- Preserves original evaluation: FPR, FNR, ACC
- Uses same dataset/model structure as main framework
"""

import os, sys
import random
import numpy as np
import pandas as pd
import torch
import dgl
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# integrate with PyGIP
from pygip.datasets.pyg_datasets import Cora, CiteSeer, PubMed, DBLP, Amazon
from pygip.models.nn.pyg_backbones import GCN, GAT, GraphSAGE, GIN, SGC


# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, data, train_mask, epochs=50, lr=0.01, device="cpu"):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
    return model


def get_posteriors(model, data, nodes):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[nodes]
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def compute_metrics(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    FP = np.sum((pred_labels == 1) & (true_labels == 0))
    FN = np.sum((pred_labels == 0) & (true_labels == 1))
    TN = np.sum((pred_labels == 0) & (true_labels == 0))
    TP = np.sum((pred_labels == 1) & (true_labels == 1))
    FPR = FP / (FP + TN + 1e-8) * 100
    FNR = FN / (FN + TP + 1e-8) * 100
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8) * 100
    return FPR, FNR, ACC


# ----------------------------
# Model Builder
# ----------------------------
def build_model(model_type, in_dim, out_dim, layers=2):
    if model_type == "GCN":
        return GCN(in_dim, 16, out_dim)
    elif model_type == "GraphSAGE":
        return GraphSAGE(in_dim, 16, out_dim)
    elif model_type == "GAT":
        return GAT(in_dim, 16, out_dim)
    elif model_type == "GIN":
        return GIN(in_dim, 16, out_dim)
    elif model_type == "SGC":
        return SGC(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ----------------------------
# Threshold tuning
# ----------------------------
def tune_threshold(Fs_star, Fs, Find, data, query_nodes):
    scores, labels = [], []
    for star in Fs_star:
        probs_star = get_posteriors(star, data, query_nodes)
        for surrogate in Fs:
            sim = cosine_similarity(probs_star, get_posteriors(surrogate, data, query_nodes)).mean()
            scores.append(sim)
            labels.append(1)
        for ind in Find:
            sim = cosine_similarity(probs_star, get_posteriors(ind, data, query_nodes)).mean()
            scores.append(sim)
            labels.append(0)
    best_thr, best_acc = 0.5, 0
    for thr in np.linspace(0.1, 0.99, 50):
        preds = [1 if s > thr else 0 for s in scores]
        _, _, acc = compute_metrics(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr


# ----------------------------
# Main Experiment
# ----------------------------
def run_bgrove_experiment(dataset_cls, condition="CondA ✓", setting="I", device="cpu"):
    ds = dataset_cls(path="./data")
    data = ds.graph_data.to(device)
    in_dim, out_dim = ds.num_features, ds.num_classes
    train_mask = data.train_mask

    overlapping = ["GCN", "GAT", "GraphSAGE"]
    disjoint = ["GIN", "SGC"]
    layers_same, layers_diff = 2, 3

    if setting == "I":
        arch_Fs, arch_Find = overlapping, overlapping
        nFs, nFind = layers_same, layers_same
    elif setting == "II":
        arch_Fs, arch_Find = overlapping, overlapping
        nFs, nFind = layers_diff, layers_same
    elif setting == "III":
        arch_Fs, arch_Find = disjoint, overlapping
        nFs, nFind = layers_same, layers_same
    elif setting == "IV":
        arch_Fs, arch_Find = disjoint, overlapping
        nFs, nFind = layers_diff, layers_same
    else:
        raise ValueError("Invalid setting")

    target = train_model(build_model("GCN", in_dim, out_dim, 2), data, train_mask, device=device)

    Fs = [train_model(build_model(a, in_dim, out_dim, nFs), data, train_mask, device=device)
          for a in arch_Fs]
    set_seed(123 if condition != "CondA ✓" else 0)
    Fs_star = [train_model(build_model(a, in_dim, out_dim, nFs), data, train_mask, device=device)
               for a in arch_Fs]
    Find = [train_model(build_model(a, in_dim, out_dim, nFind), data, train_mask, device=device)
            for a in arch_Find]

    num_queries = max(1, int(0.1 * data.num_nodes))
    query_nodes = torch.randperm(data.num_nodes)[:num_queries]
    thr = tune_threshold(Fs_star, Fs, Find, data, query_nodes)

    true_labels, pred_labels = [], []
    for model in Fs + Find:
        for star in Fs_star:
            sim = cosine_similarity(
                get_posteriors(model, data, query_nodes),
                get_posteriors(star, data, query_nodes)
            ).mean()
            true_labels.append(1 if model in Fs else 0)
            pred_labels.append(1 if sim > thr else 0)
    return compute_metrics(true_labels, pred_labels)


# ----------------------------
# Multi-seed Runner
# ----------------------------
def run_multi(dataset_cls, condition, setting, device="cpu", seeds=[0, 1, 2, 3, 4]):
    all_fpr, all_fnr, all_acc = [], [], []
    for seed in seeds:
        set_seed(seed)
        FPR, FNR, ACC = run_bgrove_experiment(dataset_cls, condition, setting, device)
        all_fpr.append(FPR)
        all_fnr.append(FNR)
        all_acc.append(ACC)
    fmt = lambda arr: f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"
    return fmt(all_fpr), fmt(all_fnr), fmt(all_acc)


# ----------------------------
# Main Entry
# ----------------------------
if __name__ == "__main__":
    datasets = [Cora, CiteSeer, PubMed, DBLP, Amazon]
    conditions = ["CondA ✓", "CondA ✗"]
    settings = ["I", "II", "III", "IV"]

    total = len(datasets) * len(conditions) * len(settings)
    results = {}
    count = 0

    for DatasetClass in datasets:
        for cond in conditions:
            for setting in settings:
                count += 1
                print(f"\n=== [{count}/{total}] {DatasetClass.__name__}, {cond}, Setting {setting} ===")
                FPR, FNR, ACC = run_multi(DatasetClass, cond, setting)
                results[(DatasetClass.__name__, cond, setting)] = [FPR, FNR, ACC]

    df = pd.DataFrame.from_dict(results, orient="index", columns=["FPR (%)", "FNR (%)", "ACC (%)"])
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Dataset", "Condition", "Setting"])

    print("\n=== Table 4: BGrOVe Results (mean ± std) ===")
    print(df)
    os.makedirs("results", exist_ok=True)
    path = "results/BGrOVe_table4.csv"
    df.to_csv(path)
    print(f"\n✅ Results saved to {path}")
