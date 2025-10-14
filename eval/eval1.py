import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import yaml
import warnings
warnings.filterwarnings("ignore")

# Ensure src/ is in PYTHONPATH
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from models.gnn import GCN, GAT, GraphSAGE
from models.ownership_classifier import OwnershipClassifier
from datasets.datareader import create_splits
from datasets.graph_operator import apply_masking

# -----------------------------
# Seed for reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# Config Loader
# -----------------------------
def load_config(config_path="config/global_cfg.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# -----------------------------
# GNN Evaluation
# -----------------------------
def evaluate_gnn(model, x, edge_index, mask, data):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        return accuracy_score(data.y[mask].cpu(), pred[mask].cpu())

# -----------------------------
# Extract model posteriors
# -----------------------------
def get_posteriors(model, x, edge_index, mask):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        return out[mask].detach().cpu().flatten()

# -----------------------------
# Ownership Classifier
# -----------------------------
def train_classifier(X, y, input_dim, hidden_dim=64, epochs=100, lr=0.01, device="cpu"):
    classifier = OwnershipClassifier(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    X, y = X.to(device), y.to(device)
    for _ in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        out = classifier(X).squeeze()
        loss = F.binary_cross_entropy(out, y)
        loss.backward()
        optimizer.step()
    return classifier

def evaluate_classifier(classifier, X, y, device="cpu"):
    classifier.eval()
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
        pred = (classifier(X) > 0.5).float()
        acc = accuracy_score(y.cpu(), pred.cpu())
        # False positive and false negative rates
        fp = ((pred == 1) & (y == 0)).sum().float() / max((y == 0).sum().float(), 1)
        fn = ((pred == 0) & (y == 1)).sum().float() / max((y == 1).sum().float(), 1)
    return acc, fp.item(), fn.item()

# -----------------------------
# Main Evaluation
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    cfg = load_config()
    dataset_name = cfg['dataset']['name']
    architectures = cfg['model']['architectures']
    mask_ratios = {
        'inductive': cfg['training']['mask_ratio_inductive'],
        'transductive': cfg['training']['mask_ratio_transductive']
    }

    # Load dataset
    dataset = Planetoid(root='./data', name=dataset_name)
    data = dataset[0].to(device)
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    results = []

    for mode in ['inductive', 'transductive']:
        mask_ratio = mask_ratios[mode]
        masks = create_splits(data, mode=mode)
        masked_x = apply_masking(data.x, mask_ratio=mask_ratio).to(device)

        for arch in architectures:
            model_class = {'gcn': GCN, 'gat': GAT, 'sage': GraphSAGE}[arch]

            for setting in ['I', 'II', 'III', 'IV']:
                target_path = f"temp_results/diff/model_states/{dataset_name}/{mode}/mask_models/random_mask/1.0_{mask_ratio}/{arch}_224_128.pt"
                if not os.path.exists(target_path):
                    print(f"Skipping {arch} {mode} Setting {setting}: Target model not found")
                    continue

                # Load target model
                target_model = model_class(num_features, 224, num_classes).to(device)
                target_model.load_state_dict(torch.load(target_path, map_location=device))
                target_acc = evaluate_gnn(target_model, masked_x, data.edge_index, masks['test'], data)

                # Collect posteriors
                posteriors = []
                labels = []

                # Independent models
                ind_dir = f"temp_results/diff/model_states/{dataset_name}/{mode}/independent_models"
                if os.path.exists(ind_dir):
                    for f in os.listdir(ind_dir):
                        if arch in f and f.endswith('.pt'):
                            model = model_class(num_features, 224, num_classes).to(device)
                            model.load_state_dict(torch.load(os.path.join(ind_dir, f), map_location=device))
                            post = get_posteriors(model, masked_x, data.edge_index, masks['train'])
                            posteriors.append(post)
                            labels.append(0)

                # Surrogate models
                surr_dir = f"temp_results/diff/model_states/{dataset_name}/{mode}/extraction_models/random_mask/{arch}_224_128/1.0_{mask_ratio}"
                if os.path.exists(surr_dir):
                    for f in os.listdir(surr_dir):
                        if arch in f and f.endswith('.pt'):
                            model = model_class(num_features, 224, num_classes).to(device)
                            model.load_state_dict(torch.load(os.path.join(surr_dir, f), map_location=device))
                            post = get_posteriors(model, masked_x, data.edge_index, masks['train'])
                            posteriors.append(post)
                            labels.append(1)

                if len(posteriors) < 2:
                    print(f"Skipping {arch} {mode} Setting {setting}: Insufficient models")
                    continue

                # Train and evaluate classifier
                X = torch.stack(posteriors).to(device)
                y = torch.tensor(labels, dtype=torch.float32).to(device)
                classifier = train_classifier(X, y, X.shape[1], device=device)
                ver_acc, fpr, fnr = evaluate_classifier(classifier, X, y, device=device)

                results.append({
                    'dataset': dataset_name,
                    'mode': mode,
                    'model': arch,
                    'setting': setting,
                    'target_acc': target_acc,
                    'ver_acc': ver_acc,
                    'fpr': fpr,
                    'fnr': fnr
                })

    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('experiments/results/results.csv', index=False)
    print("Results saved to experiments/results/results.csv")

if __name__ == "__main__":
    main()
