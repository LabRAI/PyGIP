# attacks/genie_model_extraction.py
from typing import Optional, Dict, Any
import torch
import os
import random
from sklearn.model_selection import train_test_split

# Imports in your repo: avoid referencing a missing top-level package
try:
    from pygip.core.base import BaseAttack
except Exception:
    # If your repo uses a different layout, BaseAttack may be elsewhere.
    # To keep the smoke path working we allow missing BaseAttack in tests; but
    # in your actual repo this should import the real BaseAttack.
    class BaseAttack:
        def __init__(self, dataset, attack_node_fraction=0.05, model_path=None):
            self.dataset = dataset
            self.attack_node_fraction = attack_node_fraction
            self.model_path = model_path
            self.graph_data = getattr(dataset, "graph_data", None)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch_geometric.utils import negative_sampling

# Our predictor (ensure the file exists in models/gcn_link_predictor.py)
from models.gcn_link_predictor import GCNLinkPredictor

class GenieModelExtraction(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets = set()

    def __init__(self, dataset, attack_node_fraction: float = 0.05, model_path: Optional[str] = None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.query_ratio = attack_node_fraction
        self.surrogate_epochs = 50
        self.surrogate_lr = 0.01
        self.hidden_dim = 64

    def attack(self) -> Dict[str, Any]:
        print(f"[GenieModelExtraction] Running on device {self.device}")
        data = self.graph_data
        if data is None:
            raise RuntimeError("No graph data attached to dataset.")

        num_nodes = int(getattr(data, "num_nodes", None) or data.x.size(0))

        # Ensure features present
        if getattr(data, "x", None) is None:
            print("[GenieModelExtraction] No node features found. Using random features.")
            data.x = torch.randn((num_nodes, 64))

        teacher_model = self._load_model()
        if teacher_model is None:
            print("[GenieModelExtraction] No teacher model loaded — using an untrained local predictor for smoke tests.")
            # deterministic small default teacher
            teacher_model = GCNLinkPredictor(in_channels=data.x.size(1), hidden_channels=self.hidden_dim).to(self.device)

        teacher_model.eval()
        device = self.device
        data = data.to(device)
        x = data.x.to(device)
        full_edge_index = data.edge_index.to(device)

        # choose positive edges to query
        num_pos = full_edge_index.size(1)
        sample_size = max(1, int(num_pos * float(self.query_ratio)))
        cols = random.sample(range(num_pos), sample_size)
        sampled_pos = full_edge_index[:, cols].to(device)

        # build train/val indices for surrogate
        pos_pairs = list(zip(sampled_pos[0].cpu().tolist(), sampled_pos[1].cpu().tolist()))
        if len(pos_pairs) < 2:
            train_pos_pairs = pos_pairs
            val_pos_pairs = pos_pairs
        else:
            train_pos_pairs, val_pos_pairs = train_test_split(pos_pairs, test_size=0.2, random_state=42)

        def to_edge_index(pairs):
            if len(pairs) == 0:
                return torch.empty((2,0), dtype=torch.long, device=device)
            t = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
            return t

        train_pos_index = to_edge_index(train_pos_pairs)
        val_pos_index = to_edge_index(val_pos_pairs)

        # query teacher
        teacher_logits_train = self._query_teacher(teacher_model, train_pos_index, x, full_edge_index)
        teacher_logits_val = self._query_teacher(teacher_model, val_pos_index, x, full_edge_index)

        # train_targets (binary)
        train_targets = (torch.sigmoid(teacher_logits_train) > 0.5).float() if train_pos_index.numel() else torch.tensor([], device=device)
        val_targets = (torch.sigmoid(teacher_logits_val) > 0.5).float() if val_pos_index.numel() else torch.tensor([], device=device)

        # surrogate training
        surrogate = self._train_surrogate(x, full_edge_index, train_pos_index, train_targets,
                                         val_pos_index, val_targets)

        # test using random negatives matched to sampled positives
        neg_edges = negative_sampling(edge_index=full_edge_index, num_nodes=num_nodes, num_neg_samples=sampled_pos.size(1)).to(device)
        test_auc = self._eval_surrogate_auc(surrogate, full_edge_index, sampled_pos.to(device), neg_edges, x)

        results = {
            "dataset": getattr(self.dataset, "dataset_name", "unknown"),
            "query_ratio": float(self.query_ratio),
            "surrogate_test_auc": float(test_auc)
        }
        return results

    def _load_model(self):
        # If caller passed model_path, try to load; else None
        if not self.model_path:
            return None
        if not os.path.exists(self.model_path):
            print(f"[GenieModelExtraction] model_path {self.model_path} does not exist")
            return None
        try:
            ckpt = torch.load(self.model_path, map_location=self.device)
            state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            # instantiate a predictor and try to load
            in_ch = getattr(self.dataset, "num_features", getattr(self.dataset, "num_features", 64))
            model = GCNLinkPredictor(in_channels=in_ch, hidden_channels=self.hidden_dim).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            print("[GenieModelExtraction] Loaded model checkpoint.")
            return model
        except Exception as e:
            print("[GenieModelExtraction] Failed to load checkpoint:", e)
            return None

    @torch.no_grad()
    def _query_teacher(self, teacher, edge_label_index, features, full_edge_index):
        if edge_label_index.numel() == 0:
            return torch.tensor([], device=self.device)
        teacher.eval()
        z = teacher.encode(features, full_edge_index)
        logits = teacher.decode(z, edge_label_index)
        return logits.view(-1)

    def _train_surrogate(self, x, full_edge_index, train_edge_index, train_targets, val_edge_index, val_targets):
        device = self.device
        model = GCNLinkPredictor(in_channels=x.size(1), hidden_channels=self.hidden_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.surrogate_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(1, self.surrogate_epochs + 1):
            model.train()
            opt.zero_grad()
            z = model.encode(x, full_edge_index)
            if train_edge_index.numel() == 0:
                # nothing to train on
                break
            logits = model.decode(z, train_edge_index).view(-1)
            loss = criterion(logits, train_targets.to(device))
            loss.backward()
            opt.step()
        return model

    @torch.no_grad()
    def _eval_surrogate_auc(self, model, full_edge_index, pos_edge_index, neg_edge_index, features):
        model.eval()
        if pos_edge_index.numel() == 0 or neg_edge_index.numel() == 0:
            return float("nan")
        z = model.encode(features, full_edge_index)
        pos_score = torch.sigmoid(model.decode(z, pos_edge_index)).view(-1).cpu().numpy()
        neg_score = torch.sigmoid(model.decode(z, neg_edge_index)).view(-1).cpu().numpy()
        import numpy as np
        from sklearn.metrics import roc_auc_score
        y_true = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])
        y_pred = np.concatenate([pos_score, neg_score])
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return float("nan")
