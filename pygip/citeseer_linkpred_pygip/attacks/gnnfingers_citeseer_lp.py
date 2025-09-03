# attacks/gnnfingers_citeseer_lp.py
# GNNFingersAttack for Citeseer (Link Prediction)

import copy, gc
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.utils import subgraph

from utils.common import get_device, set_seed
from datasets.citeseer import Dataset
from models.link_predictors import GCNLinkPredictor, GraphSAGELinkPredictor

class BaseAttack(nn.Module):
    supported_api_types = set()
    supported_datasets = set()
    def __init__(self, dataset: Dataset, attack_node_fraction: float = None,
                 model_path: Optional[str] = None, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.device = torch.device(device) if device else get_device()
        print(f"[Attack] Using device: {self.device}")
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.attack_node_fraction = attack_node_fraction
        self.model_path = model_path
        self._check_dataset_compatibility()
    def _check_dataset_compatibility(self):
        if self.supported_datasets and self.dataset.dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset {self.dataset.dataset_name} not supported")
    def attack(self): raise NotImplementedError
    def _load_model(self): raise NotImplementedError
    def _train_target_model(self): raise NotImplementedError
    def _train_attack_model(self): raise NotImplementedError

class GNNFingersAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets  = {"Citeseer"}
    def __init__(self, dataset: Dataset, attack_node_fraction: float = 0.1,
                 model_path: Optional[str] = None, device: Optional[Union[str, torch.device]] = None,
                 hidden=16, dropout=0.5, pos_train=50, pos_test=50, neg_train=50, neg_test=50,
                 use_ft_last=True, use_ft_all=True, use_pr_last=True, use_pr_all=True, use_distill=True,
                 distill_steps=250, seed=42):
        super().__init__(dataset, attack_node_fraction, model_path, device)
        self.hidden = hidden
        self.dropout = dropout
        self.pos_train = pos_train
        self.pos_test = pos_test
        self.neg_train = neg_train
        self.neg_test = neg_test
        self.use_ft_last = use_ft_last
        self.use_ft_all = use_ft_all
        self.use_pr_last = use_pr_last
        self.use_pr_all = use_pr_all
        self.use_distill = use_distill
        self.distill_steps = distill_steps
        self.seed = seed
        self.target_model = None
        self.positive_models = []
        self.negative_models = []

    def attack(self):
        self.target_model = self._train_target_model()
        self._generate_suspects()
        return {
            'target_model': self.target_model,
            'positive_models': self.positive_models,
            'negative_models': self.negative_models,
        }

    def _train_target_model(self, epochs=200, lr=0.005, wd=5e-4):
        model = GCNLinkPredictor(self.num_features, hidden=self.hidden, dropout=self.dropout).to(self.device)
        train_data = self.dataset.train_data.to(self.device)
        val_data = self.dataset.val_data.to(self.device)
        opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
        best = {'val': 0.0, 'state': None}
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            scores = model(train_data)
            edge_label = train_data.edge_label.float()
            loss = F.binary_cross_entropy(scores, edge_label)
            loss.backward(); opt.step()
            # validation
            model.eval()
            with torch.no_grad():
                val_scores = model(val_data)
                val_labels = val_data.edge_label.float()
                val_pred = (val_scores > 0.5).float()
                val_acc = (val_pred == val_labels).float().mean().item()
            if val_acc > best['val']:
                best['val'] = val_acc; best['state'] = copy.deepcopy(model.state_dict())
            if ep % 20 == 0:
                print(f"[Target] Epoch {ep:03d} | loss {loss.item():.4f} | val {val_acc:.3f}")
        if best['state'] is not None:
            model.load_state_dict(best['state'])
        # test
        test_data = self.dataset.test_data.to(self.device)
        model.eval()
        with torch.no_grad():
            test_scores = model(test_data)
            test_labels = test_data.edge_label.float()
            test_pred = (test_scores > 0.5).float()
            test_acc = (test_pred == test_labels).float().mean().item()
        print(f"[Target] Final test accuracy: {test_acc:.3f}")
        return model.eval()

    def _load_model(self):
        return torch.load(self.model_path, map_location=self.device)

    def _train_attack_model(self): pass

    @torch.no_grad()
    def reset_module(self, m):
        for layer in m.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def ft_model(self, base_model, last_only=True, epochs=10, lr=0.005, seed=123):
        set_seed(seed)
        m = copy.deepcopy(base_model).to(self.device)
        for p in m.parameters(): p.requires_grad_(not last_only)
        for p in m.conv3.parameters(): p.requires_grad_(True)
        opt = Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=lr)
        train_data = self.dataset.train_data.to(self.device)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            scores = m(train_data)
            edge_label = train_data.edge_label.float()
            loss = F.binary_cross_entropy(scores, edge_label)
            loss.backward(); opt.step()
        return m.eval()

    def pr_model(self, base_model, last_only=True, epochs=10, lr=0.005, seed=456):
        set_seed(seed)
        m = copy.deepcopy(base_model).to(self.device)
        if last_only: self.reset_module(m.conv3)
        else: self.reset_module(m)
        opt = Adam(m.parameters(), lr=lr)
        train_data = self.dataset.train_data.to(self.device)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            scores = m(train_data)
            edge_label = train_data.edge_label.float()
            loss = F.binary_cross_entropy(scores, edge_label)
            loss.backward(); opt.step()
        return m.eval()

    def make_student(self, arch='GCN', hidden=16):
        if arch == 'GCN':
            return GCNLinkPredictor(self.num_features, hidden, dropout=self.dropout).to(self.device)
        else:
            return GraphSAGELinkPredictor(self.num_features, hidden, dropout=self.dropout).to(self.device)

    def random_subgraph_idx(self, n, keep_ratio=0.6, seed=7):
        g = torch.Generator().manual_seed(seed)
        keep = int(n*keep_ratio)
        return torch.randperm(n, generator=g)[:keep]

    def distill_from_teacher(self, teacher, arch='GCN', steps=250, lr=0.01, seed=777):
        set_seed(seed)
        student = self.make_student(arch, hidden=self.hidden)
        opt = Adam(student.parameters(), lr=lr)
        mse = nn.MSELoss()
        data = self.dataset.graph_data.to(self.device)
        x_all = data.x; ei_all = data.edge_index
        for t in range(steps):
            keep_ratio = float(torch.empty(1).uniform_(0.5, 0.8))
            idx = self.random_subgraph_idx(data.num_nodes, keep_ratio=keep_ratio, seed=seed+t).to(self.device)
            ei_sub, _ = subgraph(idx, ei_all, relabel_nodes=True)
            x_sub = x_all[idx]
            with torch.no_grad():
                h_t = teacher.encode(x_sub, ei_sub)
            student.train(); opt.zero_grad()
            h_s = student.encode(x_sub, ei_sub)
            loss = mse(h_s, h_t)
            loss.backward(); opt.step()
        return student.eval()

    def _distribute_budget(self, total, keys):
        if not keys: return {}
        base = total // len(keys); rem = total - base*len(keys)
        out = {k: base for k in keys}
        for k in keys[:rem]: out[k] += 1
        return out

    def _generate_suspects(self):
        print("[Attack] Generating suspect models...")
        # Positives
        F_pos_all = []
        pos_total = self.pos_train + self.pos_test
        pos_keys = []
        if self.use_ft_last: pos_keys.append("FT_LAST")
        if self.use_ft_all: pos_keys.append("FT_ALL")
        if self.use_pr_last: pos_keys.append("PR_LAST")
        if self.use_pr_all: pos_keys.append("PR_ALL")
        if self.use_distill: pos_keys.append("DISTILL")
        pos_budget = self._distribute_budget(pos_total, pos_keys)
        seed_base = 10
        for key in pos_keys:
            cnt = pos_budget[key]
            if key == "FT_LAST":
                for s in range(seed_base, seed_base+cnt):
                    F_pos_all.append(self.ft_model(self.target_model, last_only=True, epochs=10, seed=s))
                seed_base += cnt
            elif key == "FT_ALL":
                for s in range(seed_base, seed_base+cnt):
                    F_pos_all.append(self.ft_model(self.target_model, last_only=False, epochs=10, seed=s))
                seed_base += cnt
            elif key == "PR_LAST":
                for s in range(seed_base, seed_base+cnt):
                    F_pos_all.append(self.pr_model(self.target_model, last_only=True, epochs=10, seed=s))
                seed_base += cnt
            elif key == "PR_ALL":
                for s in range(seed_base, seed_base+cnt):
                    F_pos_all.append(self.pr_model(self.target_model, last_only=False, epochs=10, seed=s))
                seed_base += cnt
            elif key == "DISTILL":
                arches = (['GCN'] * (cnt//2) + ['SAGE'] * (cnt - cnt//2))
                for i, arch in enumerate(arches, 400):
                    F_pos_all.append(self.distill_from_teacher(self.target_model, arch=arch,
                                                               steps=self.distill_steps, seed=1000+i))
                seed_base += cnt
        assert len(F_pos_all) == pos_total, f"Expected {pos_total} positives, got {len(F_pos_all)}"
        # Negatives
        F_neg_all = []
        neg_total = self.neg_train + self.neg_test
        neg_keys = ["GCN", "SAGE"]
        neg_budget = self._distribute_budget(neg_total, neg_keys)
        seed_base = 500
        def train_model(model, train_data, val_data, epochs=120, lr=0.005, wd=5e-4):
            model = model.to(self.device)
            train_data = train_data.to(self.device)
            val_data = val_data.to(self.device)
            opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
            best = {'val': 0.0, 'state': None}
            for ep in range(epochs):
                model.train(); opt.zero_grad()
                scores = model(train_data)
                edge_label = train_data.edge_label.float()
                loss = F.binary_cross_entropy(scores, edge_label)
                loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    val_scores = model(val_data)
                    val_labels = val_data.edge_label.float()
                    val_pred = (val_scores > 0.5).float()
                    val_acc = (val_pred == val_labels).float().mean().item()
                if val_acc > best['val']:
                    best['val'] = val_acc; best['state'] = copy.deepcopy(model.state_dict())
            if best['state'] is not None:
                model.load_state_dict(best['state'])
            return model.eval()
        # Train neg pools
        for s in range(seed_base, seed_base+neg_budget["GCN"]):
            set_seed(s)
            m = GCNLinkPredictor(self.num_features, self.hidden, dropout=self.dropout)
            m = train_model(m, self.dataset.train_data, self.dataset.val_data, epochs=120, lr=0.005, wd=5e-4)
            F_neg_all.append(m)
        seed_base += neg_budget["GCN"]
        for s in range(seed_base, seed_base+neg_budget["SAGE"]):
            set_seed(s)
            m = GraphSAGELinkPredictor(self.num_features, self.hidden, dropout=self.dropout)
            m = train_model(m, self.dataset.train_data, self.dataset.val_data, epochs=120, lr=0.005, wd=5e-4)
            F_neg_all.append(m)
        # Split into train/test
        def split_pool(pool, n_train, n_test, seed=999):
            set_seed(seed)
            idx = torch.randperm(len(pool)).tolist()
            train = [pool[i] for i in idx[:n_train]]
            test  = [pool[i] for i in idx[n_train:n_train+n_test]]
            return train, test
        self.positive_models = split_pool(F_pos_all, self.pos_train, self.pos_test)
        self.negative_models = split_pool(F_neg_all, self.neg_train, self.neg_test)
        print(f"[Attack] F+ train/test: {len(self.positive_models[0])}/{len(self.positive_models[1])} | "
              f"F- train/test: {len(self.negative_models[0])}/{len(self.negative_models[1])}")
