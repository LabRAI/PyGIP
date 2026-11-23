"""
pygip.models.attack.genie_model_extraction
Modified to infer teacher input channels from checkpoint and pad features
so that checkpoint loading doesn't fail when dataset feature dimension differs.
"""
from typing import Optional, Dict, Any
import torch
import os
import random
from sklearn.model_selection import train_test_split

try:
    from pygip.models.attack.base import BaseAttack
    from pygip.datasets.datasets import Dataset
except Exception:
    # best-effort fallbacks — adjust if your project exposes different paths
    from pygip.models.attack.base import BaseAttack
    from pygip.datasets.datasets import Dataset

class GenieModelExtraction(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets = set()

    def __init__(self, dataset: Dataset, attack_node_fraction: float = 0.05, model_path: Optional[str] = None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.query_ratio = attack_node_fraction
        # surrogate params
        self.surrogate_epochs = 50
        self.surrogate_lr = 0.01
        self.hidden_dim = 64
        # how many negative samples per positive to draw for surrogate training
        self.neg_ratio = 1
        # teacher expected input channels (set when loading checkpoint)
        self.teacher_in_ch: Optional[int] = None

    def attack(self) -> Dict[str, Any]:
        print(f"[GenieModelExtraction] Running on device {self.device}")
        data = self.graph_data
        num_nodes = data.num_nodes

        # Ensure features exist
        if getattr(data, "x", None) is None:
            print("[GenieModelExtraction] No node features found. Using random features.")
            data.x = torch.randn((num_nodes, 64))

        # Load teacher model
        teacher_model = self._load_model()
        if teacher_model is None:
            raise RuntimeError("Could not load teacher model for extraction")

        teacher_model.eval()
        device = self.device
        data = data.to(device)

        # pad features if teacher expects larger dimensionality
        x = data.x.to(device)
        if self.teacher_in_ch is not None and x.size(1) != self.teacher_in_ch:
            old = x
            new_ch = self.teacher_in_ch
            if old.size(1) < new_ch:
                pad = torch.zeros((old.size(0), new_ch - old.size(1)), device=device, dtype=old.dtype)
                x = torch.cat([old, pad], dim=1)
                print(f"[GenieModelExtraction] Padded node features {old.size(1)} -> {new_ch}")
            else:
                # if dataset has larger features than teacher expects, truncate
                x = old[:, :new_ch]
                print(f"[GenieModelExtraction] Truncated node features {old.size(1)} -> {new_ch}")

        full_edge_index = data.edge_index.to(device)

        # Sample a subset of existing edges as positives to query teacher
        num_pos_total = full_edge_index.size(1)
        sample_size = max(1, int(num_pos_total * self.query_ratio))
        cols = random.sample(range(num_pos_total), sample_size)
        sampled_pos = full_edge_index[:, cols].to(device)

        # Build pos list for train/val split
        pos_pairs = [(int(u.item()), int(v.item())) for u, v in zip(sampled_pos[0], sampled_pos[1])]
        if len(pos_pairs) < 2:
            train_pos, val_pos = pos_pairs, pos_pairs
        else:
            train_pos, val_pos = train_test_split(pos_pairs, test_size=0.2, random_state=42)

        # Create negatives with rejection sampling
        existing = set((int(u.item()), int(v.item())) for u, v in zip(full_edge_index[0], full_edge_index[1]))
        def sample_neg(n_samples):
            negs = []
            tries = 0
            while len(negs) < n_samples and tries < n_samples * 20:
                a = random.randrange(num_nodes)
                b = random.randrange(num_nodes)
                if a == b:
                    tries += 1; continue
                if (a, b) in existing:
                    tries += 1; continue
                negs.append((a, b))
            return negs

        train_neg = sample_neg(max(1, int(len(train_pos) * self.neg_ratio)))
        val_neg = sample_neg(max(1, int(len(val_pos) * self.neg_ratio)))

        def pairs_to_edge_index(pairs):
            if len(pairs) == 0:
                return torch.empty((2,0), dtype=torch.long, device=device)
            u = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
            v = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
            return torch.stack([u, v], dim=0)

        train_pos_ei = pairs_to_edge_index(train_pos)
        val_pos_ei = pairs_to_edge_index(val_pos)
        train_neg_ei = pairs_to_edge_index(train_neg)
        val_neg_ei = pairs_to_edge_index(val_neg)

        # Query teacher for logits
        teacher_logits_train_pos = self._query_teacher(teacher_model, train_pos_ei, x, full_edge_index)
        teacher_logits_val_pos = self._query_teacher(teacher_model, val_pos_ei, x, full_edge_index)
        teacher_logits_train_neg = self._query_teacher(teacher_model, train_neg_ei, x, full_edge_index) if train_neg_ei.size(1) > 0 else torch.tensor([], device=device)
        teacher_logits_val_neg = self._query_teacher(teacher_model, val_neg_ei, x, full_edge_index) if val_neg_ei.size(1) > 0 else torch.tensor([], device=device)

        # Build targets (binary) for surrogate using teacher's sigmoid threshold
        train_targets = torch.cat([
            (torch.sigmoid(teacher_logits_train_pos) > 0.5).float(),
            (torch.sigmoid(teacher_logits_train_neg) > 0.5).float()
        ], dim=0)
        val_targets = torch.cat([
            (torch.sigmoid(teacher_logits_val_pos) > 0.5).float(),
            (torch.sigmoid(teacher_logits_val_neg) > 0.5).float()
        ], dim=0)

        train_edge_index = torch.cat([train_pos_ei, train_neg_ei], dim=1)
        val_edge_index = torch.cat([val_pos_ei, val_neg_ei], dim=1)

        print(f"[GenieModelExtraction] Train pos {train_pos_ei.size(1)} neg {train_neg_ei.size(1)} total {train_edge_index.size(1)}")
        print(f"[GenieModelExtraction] Val pos {val_pos_ei.size(1)} neg {val_neg_ei.size(1)} total {val_edge_index.size(1)}")

        # Train surrogate
        surrogate = self._train_surrogate(x, full_edge_index, train_edge_index, train_targets, val_edge_index, val_targets)

        # Evaluate surrogate on test pos/neg
        from torch_geometric.utils import negative_sampling
        test_sample_size = max(100, int(num_pos_total * 0.02))
        test_cols = random.sample(range(num_pos_total), min(test_sample_size, num_pos_total))
        test_pos_ei = full_edge_index[:, test_cols].to(device)
        test_neg_ei = negative_sampling(edge_index=full_edge_index, num_nodes=num_nodes, num_neg_samples=test_pos_ei.size(1)).to(device)

        test_auc = self._eval_surrogate_auc(surrogate, full_edge_index, test_pos_ei, test_neg_ei, x)

        results = {
            "dataset": self.dataset.dataset_name if hasattr(self.dataset, "dataset_name") else "unknown",
            "query_ratio": self.query_ratio,
            "surrogate_test_auc": float(test_auc)
        }
        return results

    def _load_model(self):
        """Load teacher/watermarked model and robustly infer expected input channels from checkpoint weights.

        Strategy:
        - Try to infer input channels by scanning all 2-D tensors in the saved state dict.
        - Prefer candidates >= 8 (to avoid picking tiny feature dims like 1/2 which are likely dataset-specific).
        - If multiple candidates exist pick the largest (conservative) or the most common.
        - Fall back to dataset.num_features or 64 if nothing inferred.
    """
        if not self.model_path:
            print("[GenieModelExtraction] No model_path passed. Attempting dataset default (not implemented).")
            return None

        try:
            ckpt = torch.load(self.model_path, map_location=self.device)
        except Exception as e:
            print("[GenieModelExtraction] Error loading checkpoint:", e)
            return None

        state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

        # collect candidate input dims from 2-D tensors in the state dict
        cand = []
        try:
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    # v.shape == (out, in) for linear/convolution weight matrices
                    in_ch = int(v.size(1))
                    cand.append((k, in_ch))
        except Exception:
            cand = []

        # build a list of numeric candidates
        nums = [c[1] for c in cand]
        inferred_in_ch = None
        if nums:
            # prefer candidate dims >= 8 (heuristic), otherwise fallback to max
            big = [n for n in nums if n >= 8]
            if big:
                # pick the most common among big, else max
                from collections import Counter
                cnt = Counter(big)
                # most common; if tie choose the largest among tied
                most_common, _ = cnt.most_common(1)[0]
                # if there is tie in counts, we prefer the max
                tied = [n for n,c in cnt.items() if c == cnt[most_common]]
                inferred_in_ch = max(tied) if len(tied) > 1 else most_common
            else:
                # nothing 'big' — choose the largest candidate (safe)
                inferred_in_ch = max(nums)

        if inferred_in_ch is None:
            inferred_in_ch = getattr(self.dataset, "num_features", None) or 64

        # debug print
        print(f"[GenieModelExtraction] Inferred teacher in_channels from checkpoint candidates {nums} -> choosing {inferred_in_ch}")

        # construct model using inferred input channels and load checkpoint
        try:
            from pygip.models.gcn_link_predictor import GCNLinkPredictor
            model = GCNLinkPredictor(in_channels=inferred_in_ch, hidden_channels=self.hidden_dim).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            self.teacher_in_ch = inferred_in_ch
            print(f"[GenieModelExtraction] Loaded teacher checkpoint expecting in_channels={inferred_in_ch}")
            return model
        except Exception as e:
            print("[GenieModelExtraction] Failed to reconstruct teacher model:", e)
            return None

    @torch.no_grad()
    def _query_teacher(self, teacher, edge_label_index, features, full_edge_index):
        if edge_label_index is None or edge_label_index.size(1) == 0:
            return torch.tensor([], device=self.device)
        teacher.eval()
        z = teacher.encode(features, full_edge_index)
        logits = teacher.decode(z, edge_label_index)
        return logits.view(-1)

    def _train_surrogate(self, x, full_edge_index, train_edge_index, train_targets, val_edge_index, val_targets):
        from pygip.models.gcn_link_predictor import GCNLinkPredictor
        device = self.device
        model = GCNLinkPredictor(in_channels=x.size(1), hidden_channels=self.hidden_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.surrogate_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(1, self.surrogate_epochs + 1):
            model.train()
            opt.zero_grad()
            z = model.encode(x, full_edge_index)
            logits = model.decode(z, train_edge_index).view(-1)
            loss = criterion(logits, train_targets.to(device))
            loss.backward()
            opt.step()
        return model

    @torch.no_grad()
    def _eval_surrogate_auc(self, model, full_edge_index, pos_edge_index, neg_edge_index, features):
        model.eval()
        z = model.encode(features, full_edge_index)
        pos_score = torch.sigmoid(model.decode(z, pos_edge_index)).view(-1).cpu().numpy()
        neg_score = torch.sigmoid(model.decode(z, neg_edge_index)).view(-1).cpu().numpy()
        import numpy as np
        y_true = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])
        y_pred = np.concatenate([pos_score, neg_score])
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return float("nan")
