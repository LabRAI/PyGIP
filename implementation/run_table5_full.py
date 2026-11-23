# run_table5_full.py
# Rewritten to reproduce Figure 3 & Table 5 from Zhou et al. (2024) with aggregation + stability fixes

import os, random, numpy as np, pandas as pd, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, CitationFull
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pygip.models.nn.pyg_backbones import GCN, GAT, GraphSAGE, GIN, SGC

# ----------------------------
# Config
# ----------------------------
SEEDS = [0, 1]        
NUM_INDEP = 3         # fewer independent models
NUM_SURR = 3          # fewer surrogates
MODEL_TRAIN_EPOCHS = 40
SURR_TRAIN_EPOCHS = 40
COWN_TRAIN_EPOCHS = 20
MASK_RATIOS = [0.0, 0.1, 0.2, 0.4]   


# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(name, device="cpu"):
    lname = name.lower()
    if lname in [ "pubmed","cora","citeseer"]:
        dataset = Planetoid(root=f"data/{name}", name=name)
        data = dataset[0].to(device)
    elif "amazon" in lname:
        sub = "Photo" if "photo" in lname else "Computers"
        dataset = Amazon(root=f"data/{lname}", name=sub)
        data = dataset[0].to(device)
    elif lname in ["dblp","db_lp","db-lp"]:
        dataset = CitationFull(root="data/dblp", name="dblp")
        data = dataset[0].to(device)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return data, dataset

def split_nodes(num_nodes, ratios=(0.3,0.3,0.3,0.1), seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_nodes)
    sizes = [int(r*num_nodes) for r in ratios]
    sizes[-1] = num_nodes - sum(sizes[:-1])
    splits, names, start = {}, ["train","dshadow","dsurr","dtest"], 0
    for name, sz in zip(names, sizes):
        idx = perm[start:start+sz]
        mask = torch.zeros(num_nodes, dtype=torch.bool); mask[idx] = True
        splits[name] = mask; start += sz
    return splits

def filter_edges_to_mask(data, mask):
    ei = data.edge_index; mask = mask.to(ei.device)
    keep = ((mask[ei[0]] == True) & (mask[ei[1]] == True))
    return ei[:, keep]

def mask_features_global(data, mask_ratio=0.1, seed=0):
    x = data.x.clone(); num_feats = x.size(1)
    k = max(1, int(mask_ratio * num_feats))
    rng = np.random.RandomState(seed)
    feat_idx = rng.choice(num_feats, k, replace=False)
    x[:, feat_idx] = 0.0
    data2 = Data(x=x, edge_index=data.edge_index.clone(), y=data.y.clone())
    return data2, feat_idx

# ----------------------------
# Models & Training
# ----------------------------
def build_model(model_type, in_dim, out_dim, layers=2):
    cls_map = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "SGC": SGC}
    cls = cls_map[model_type]
    try:
        return cls(in_channels=in_dim, out_channels=out_dim, num_layers=layers)
    except TypeError:
        return cls(in_dim, out_dim, layers)

def train_model(model, data, train_mask, epochs=200, lr=0.01, device="cpu"):
    model = model.to(device); data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward(); opt.step()
    return model

def train_with_soft_labels(model, data, train_mask, soft_targets, epochs=200, lr=0.01, device="cpu"):
    model = model.to(device); data = data.to(device)
    soft_targets = soft_targets.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = F.log_softmax(model(data.x, data.edge_index), dim=1)
        loss = F.kl_div(out[train_mask], soft_targets[train_mask], reduction='batchmean')
        loss.backward(); opt.step()
    return model

def compute_accuracy(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        return (pred[mask] == data.y[mask]).float().mean().item() * 100

def compute_fidelity(model, target, data, mask):
    model.eval(); target.eval()
    with torch.no_grad():
        pred_m = model(data.x, data.edge_index).argmax(dim=1)
        pred_t = target(data.x, data.edge_index).argmax(dim=1)
        return (pred_m[mask] == pred_t[mask]).float().mean().item() * 100

# ----------------------------
# Holistic vectors & C_own
# ----------------------------
def model_to_vector_probs(model, data, node_order=None):
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index), dim=1).cpu()
    if node_order is None:
        node_order = torch.arange(probs.size(0))
    return probs[node_order].reshape(-1).numpy()

class COwn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# ----------------------------
# Settings mapping (I–IV)
# ----------------------------
def get_setting_architectures(setting):
    overlapping, disjoint = ["GCN","GAT","GraphSAGE"], ["GIN","SGC"]
    l_same, l_diff = 2, 3
    if setting == "I": Fs, Find, lFs, lFind = overlapping, overlapping, l_same, l_same
    elif setting == "II": Fs, Find, lFs, lFind = overlapping, overlapping, l_diff, l_same
    elif setting == "III": Fs, Find, lFs, lFind = disjoint, overlapping, l_same, l_same
    elif setting == "IV": Fs, Find, lFs, lFind = disjoint, overlapping, l_diff, l_same
    else: raise ValueError("Invalid setting")
    return Fs, Find, lFs, lFind

# ----------------------------
# Main experiment (Table 5 / Fig 3)
# ----------------------------
def run_table5_full(dataset_name, setting="I", inductive=False, device="cpu"):
    data_orig, dataset = load_dataset(dataset_name, device=device)
    in_dim, out_dim = dataset.num_features, dataset.num_classes
    Fs, Find, lFs, lFind = get_setting_architectures(setting)

    results = []
    for seed in SEEDS:
        set_seed(seed)
        splits = split_nodes(data_orig.num_nodes, seed=seed)
        node_order = torch.where(splits["train"])[0]

        # baseline target
        base_model = build_model("GCN", in_dim, out_dim, 2)
        base_model = train_model(base_model, data_orig, splits["train"], 
                                 epochs=MODEL_TRAIN_EPOCHS, device=device)
        base_acc = compute_accuracy(base_model, data_orig, splits["dtest"])

        for mask_ratio in MASK_RATIOS:
            data_masked, _ = mask_features_global(data_orig, mask_ratio, seed=seed)

            # train masked target
            tgt = build_model("GCN", in_dim, out_dim, 2)
            tgt = train_model(tgt, data_masked, splits["train"], epochs=MODEL_TRAIN_EPOCHS, device=device)
            tgt_acc = compute_accuracy(tgt, data_masked, splits["dtest"])
            drop = base_acc - tgt_acc
            print(f"[{dataset_name}-{setting}-seed{seed}] Mask={mask_ratio:.2f}, acc={tgt_acc:.2f}, drop={drop:.2f}")

            # Independents
            indep_vecs, indep_accs = [], []
            for arch in Find:
                for j in range(NUM_INDEP):
                    m = build_model(arch, in_dim, out_dim, lFind)
                    m = train_model(m, data_masked, splits["train"], epochs=MODEL_TRAIN_EPOCHS, device=device)
                    indep_accs.append(compute_accuracy(m, data_masked, splits["dtest"]))
                    indep_vecs.append(model_to_vector_probs(m, data_masked, node_order))

            # Surrogates
            with torch.no_grad():
                soft_all = F.softmax(tgt(data_masked.x, data_masked.edge_index), dim=1).cpu()

            surr_vecs, surr_accs, surr_fids = [], [], []
            for arch in Fs:
                for j in range(NUM_SURR):
                    m = build_model(arch, in_dim, out_dim, lFs)
                    m = train_with_soft_labels(m, data_masked, splits["train"], soft_all, 
                                               epochs=SURR_TRAIN_EPOCHS, device=device)
                    surr_accs.append(compute_accuracy(m, data_masked, splits["dtest"]))
                    surr_fids.append(compute_fidelity(m, tgt, data_masked, splits["dtest"]))
                    surr_vecs.append(model_to_vector_probs(m, data_masked, node_order))

            # Ownership classifier (full batch training for stability)
            X = np.vstack(indep_vecs + surr_vecs)
            y = np.array([0]*len(indep_vecs) + [1]*len(surr_vecs))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed
            )
            cown = COwn(X.shape[1]).to(device)
            opt = torch.optim.Adam(cown.parameters(), lr=0.001, weight_decay=1e-4)
            X_train_t, y_train_t = torch.tensor(X_train,dtype=torch.float32,device=device), torch.tensor(y_train,dtype=torch.long,device=device)
            X_test_t, y_test_t = torch.tensor(X_test,dtype=torch.float32,device=device), torch.tensor(y_test,dtype=torch.long,device=device)

            for epoch in range(COWN_TRAIN_EPOCHS):
                cown.train()
                out = cown(X_train_t)
                loss = F.cross_entropy(out, y_train_t)
                opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                preds = cown(X_test_t).argmax(dim=1).cpu().numpy()
                c_acc = (preds == y_test).mean()*100
            print(f"[{dataset_name}-{setting}-seed{seed}] C_own acc={c_acc:.2f}")

            # save
            results.append({
                "dataset": dataset_name,
                "setting": setting,
                "mode": "Inductive" if inductive else "Transductive",
                "seed": seed,
                "mask_ratio": mask_ratio,
                "target_acc": tgt_acc,
                "indep_acc_mean": np.mean(indep_accs),
                "surr_acc_mean": np.mean(surr_accs),
                "surr_fid_mean": np.mean(surr_fids),
                "cown_acc": c_acc
            })

    return pd.DataFrame(results)

# ----------------------------
# Driver
# ----------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    datasets, settings = ["Cora","CiteSeer","PubMed","Amazon","dblp"], ["I","II","III","IV"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []

    for ds in datasets:
        for st in settings:
            for mode in [False, True]:  # transductive=False, inductive=True
                df = run_table5_full(dataset_name=ds, setting=st, inductive=mode, device=device)
                all_results.append(df)

    all_results = pd.concat(all_results, ignore_index=True)
    all_results.to_csv("results/all_results_per_seed.csv", index=False)

    # --- Aggregation for analyze_tables_extended.py ---
    agg = all_results.groupby(["dataset","setting","mode"]).agg({
        "target_acc": ["mean","std"],
        "indep_acc_mean": ["mean","std"],
        "surr_acc_mean": ["mean","std"],
        "surr_fid_mean": ["mean","std"],
        "cown_acc": ["mean","std"]
    }).reset_index()

    agg.columns = [
        "dataset","setting","mode",
        "target_acc_mean","target_acc_std",
        "indep_acc_mean","indep_acc_std",
        "surr_acc_mean","surr_acc_std",
        "surr_fid_mean","surr_fid_std",
        "cown_acc_mean","cown_acc_std"
    ]
    agg.to_csv("results/table5_all_results.csv", index=False)

    print("✅ Saved results/all_results_per_seed.csv and results/table5_all_results.csv (aggregated)")
