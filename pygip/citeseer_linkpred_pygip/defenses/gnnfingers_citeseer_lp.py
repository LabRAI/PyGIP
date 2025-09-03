# defenses/gnnfingers_citeseer_lp.py
# Fingerprints + Verifier + GNNFingersDefense for Citeseer (Link Prediction)

from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.common import get_device
from datasets.citeseer import Dataset
from attacks.gnnnfingers_citeseer_lp import BaseAttack  # for type parity

class BaseDefense(nn.Module):
    supported_api_types = set()
    supported_datasets = set()
    def __init__(self, dataset: Dataset, attack_node_fraction: float,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.device = torch.device(device) if device else get_device()
        print(f"[Defense] Using device: {self.device}")
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.attack_node_fraction = attack_node_fraction
        self._check_dataset_compatibility()
    def _check_dataset_compatibility(self):
        if self.supported_datasets and self.dataset.dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset {self.dataset.dataset_name} not supported")
    def defend(self): raise NotImplementedError
    def _load_model(self): raise NotImplementedError
    def _train_target_model(self): raise NotImplementedError
    def _train_defense_model(self): raise NotImplementedError
    def _train_surrogate_model(self): raise NotImplementedError

# Fingerprint graph that probes link-pred models by sampling edge-pair scores
class FingerprintGraph(nn.Module):
    def __init__(self, n_nodes, feat_dim, sample_m, edge_init_p=0.05, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.n = n_nodes; self.d = feat_dim
        self.m = min(sample_m, n_nodes*(n_nodes-1)//2)
        X = torch.empty(self.n, self.d).uniform_(-0.5, 0.5)
        self.X = nn.Parameter(X.to(self.device))
        A0 = (torch.rand(self.n, self.n, device=self.device) < edge_init_p).float()
        A0.fill_diagonal_(0.0)
        A0 = torch.maximum(A0, A0.T)
        self.A_logits = nn.Parameter(torch.logit(torch.clamp(A0, 1e-4, 1-1e-4)))
        all_pairs = torch.combinations(torch.arange(self.n, device=self.device), r=2)
        perm = torch.randperm(len(all_pairs), device=self.device)[:self.m]
        self.sample_pairs = all_pairs[perm].t()

    @torch.no_grad()
    def edge_index(self):
        A_prob = torch.sigmoid(self.A_logits)
        A_bin = (A_prob > 0.5).float()
        A_bin.fill_diagonal_(0.0)
        A_bin = torch.maximum(A_bin, A_bin.T)
        idx = A_bin.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        return idx.t().contiguous()

    @torch.no_grad()
    def flip_topk_by_grad(self, gradA, topk=64, step=2.5):
        g = gradA.abs()
        triu = torch.triu(torch.ones_like(g), diagonal=1)
        scores = (g * triu).flatten()
        k = min(topk, scores.numel())
        if k == 0: return
        _, idxs = torch.topk(scores, k=k)
        r = self.n
        pairs = torch.stack((idxs // r, idxs % r), dim=1)
        A_prob = torch.sigmoid(self.A_logits).detach()
        for (u, v) in pairs.tolist():
            guv = gradA[u, v].item()
            exist = A_prob[u, v] > 0.5
            if exist and guv <= 0:
                self.A_logits.data[u, v] -= step
                self.A_logits.data[v, u] -= step
            elif (not exist) and guv >= 0:
                self.A_logits.data[u, v] += step
                self.A_logits.data[v, u] += step
        self.A_logits.data.fill_diagonal_(-10.0)

class FingerprintSet(nn.Module):
    def __init__(self, P, n_nodes, feat_dim, sample_m, edge_init_p=0.05,
                 topk_edges=64, edge_step=2.5, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.P = P
        self.fps = nn.ModuleList([
            FingerprintGraph(n_nodes, feat_dim, sample_m, edge_init_p, self.device)
            for _ in range(P)
        ])
        self.topk_edges = topk_edges
        self.edge_step = edge_step

    def concat_outputs(self, model, *, require_grad: bool = False):
        outs = []
        model.eval()
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            for fp in self.fps:
                ei = fp.edge_index()
                h = model.encode(fp.X, ei)
                h_u = h[fp.sample_pairs[0]]
                h_v = h[fp.sample_pairs[1]]
                probs = torch.sigmoid((h_u * h_v).sum(dim=-1))
                outs.append(probs)
        return torch.cat(outs, dim=0)

    def flip_adj_by_grad(self, surrogate_grad_list):
        for fp, g in zip(self.fps, surrogate_grad_list):
            fp.flip_topk_by_grad(g, topk=self.topk_edges, step=self.edge_step)

class Univerifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)

class GNNFingersDefense(BaseDefense):
    supported_api_types = {"pyg"}
    supported_datasets  = {"Citeseer"}
    def __init__(self, dataset: Dataset, attack_node_fraction: float = 0.1,
                 device: Optional[Union[str, torch.device]] = None,
                 P=64, fp_nodes=32, sample_m=32, edge_init_p=0.05, topk_edges=64, edge_step=2.5,
                 outer_iters=20, fp_steps=5, v_steps=10, lr_v=1e-3, lr_x=1e-3):
        super().__init__(dataset, attack_node_fraction, device)
        self.P = P
        self.fp_nodes = fp_nodes
        self.sample_m = sample_m
        self.edge_init_p = edge_init_p
        self.topk_edges = topk_edges
        self.edge_step = edge_step
        self.outer_iters = outer_iters
        self.fp_steps = fp_steps
        self.v_steps = v_steps
        self.lr_v = lr_v
        self.lr_x = lr_x
        self.fp_set = FingerprintSet(P=P, n_nodes=fp_nodes, feat_dim=dataset.num_features,
                                     sample_m=sample_m, edge_init_p=edge_init_p,
                                     topk_edges=topk_edges, edge_step=edge_step, device=self.device)
        input_dim = P * sample_m
        self.verifier = Univerifier(input_dim).to(self.device)
        self.opt_v = Adam(self.verifier.parameters(), lr=lr_v)
        self.target_model = None

    def defend(self, attack_results=None):
        from attacks.gnnnfingers_citeseer_lp import GNNFingersAttack
        if attack_results is None:
            attack = GNNFingersAttack(self.dataset, self.attack_node_fraction, device=self.device)
            attack_results = attack.attack()
        target_model = attack_results['target_model']
        pos_models = attack_results['positive_models']
        neg_models = attack_results['negative_models']
        self.target_model = target_model
        # Flatten train/test pools
        pos_train = pos_models[0]; pos_test = pos_models[1]
        neg_train = neg_models[0]; neg_test = neg_models[1]
        self._joint_train(target_model, pos_train, neg_train)
        return {'fingerprint_set': self.fp_set, 'verifier': self.verifier, 'target_model': target_model,
                'pos_test': pos_test, 'neg_test': neg_test}

    def batch_from_pool(self, pos_models, neg_models, *, require_grad: bool):
        X = []; y = []
        for m in pos_models:
            X.append(self.fp_set.concat_outputs(m, require_grad=require_grad)); y.append(1)
        for m in neg_models:
            X.append(self.fp_set.concat_outputs(m, require_grad=require_grad)); y.append(0)
        return torch.stack(X, dim=0), torch.tensor(y, device=self.device)

    def surrogate_grad_A_for_fp(self, fp, model):
        with torch.no_grad():
            ei = fp.edge_index()
            # simple surrogate based on first conv layer
            h = model.conv1(fp.X, ei) if hasattr(model, 'conv1') else model.encode(fp.X, ei)
            h = F.relu(h)
            hn = F.normalize(h, dim=-1)
            sim = hn @ hn.t()
            gradA = sim - 0.5
        return gradA.detach().cpu()

    def update_features(self, pos_models, neg_models, steps):
        all_models = pos_models + neg_models
        for m in all_models:
            for p in m.parameters(): p.requires_grad_(False)
        for fp in self.fp_set.fps:
            fp.X.requires_grad_(True)
        for _ in range(steps):
            Xb, yb = self.batch_from_pool(pos_models, neg_models, require_grad=True)
            self.verifier.eval()
            for p in self.verifier.parameters(): p.requires_grad_(False)
            logits = self.verifier(Xb.to(self.device))
            loss = F.cross_entropy(logits, yb)
            for fp in self.fp_set.fps:
                if fp.X.grad is not None: fp.X.grad.zero_()
            loss.backward()
            with torch.no_grad():
                for fp in self.fp_set.fps:
                    fp.X.add_(self.lr_x * fp.X.grad)
                    fp.X.grad.zero_()
            for p in self.verifier.parameters(): p.requires_grad_(True)
        grads = [self.surrogate_grad_A_for_fp(fp, pos_models[0]) for fp in self.fp_set.fps]
        self.fp_set.flip_adj_by_grad(grads)

    def update_verifier(self, pos_models, neg_models, steps):
        for _ in range(steps):
            self.verifier.train()
            Xb, yb = self.batch_from_pool(pos_models, neg_models, require_grad=False)
            logits = self.verifier(Xb.to(self.device))
            loss = F.cross_entropy(logits, yb)
            self.opt_v.zero_grad(); loss.backward(); self.opt_v.step()

    def _joint_train(self, target_model, pos_train, neg_train):
        models_pos_tr = [target_model] + pos_train
        models_neg_tr = neg_train
        print(f"[Defense] Train pools -> Pos: {len(models_pos_tr)} | Neg: {len(models_neg_tr)}")
        for it in range(1, self.outer_iters + 1):
            self.update_features(models_pos_tr, models_neg_tr, steps=self.fp_steps)
            self.update_verifier(models_pos_tr, models_neg_tr, steps=self.v_steps)
            self.verifier.eval()
            Xb, yb = self.batch_from_pool(models_pos_tr, models_neg_tr, require_grad=False)
            with torch.no_grad():
                pred = self.verifier(Xb).argmax(dim=1)
                acc = (pred.cpu() == yb.cpu()).float().mean().item()
                pos_acc = (pred[:len(models_pos_tr)].cpu() == 1).float().mean().item()
                neg_acc = (pred[len(models_pos_tr):].cpu() == 0).float().mean().item()
            print(f"[Defense] Iter {it:02d}/{self.outer_iters} | train all {acc:.3f} | pos {pos_acc:.3f} | neg {neg_acc:.3f}")

    @torch.no_grad()
    def verify_scores(self, models):
        Xs = [self.fp_set.concat_outputs(m, require_grad=False) for m in models]
        logits = self.verifier(torch.stack(Xs, dim=0).to(self.device))
        probs = F.softmax(logits, dim=-1)[:, 1]
        return probs.detach().cpu().numpy()

    def evaluate(self, pos_test, neg_test):
        print("[Defense] Evaluating on test set...")
        models_pos_te = [self.target_model] + pos_test
        models_neg_te = neg_test
        p_pos = self.verify_scores(models_pos_te)
        p_neg = self.verify_scores(models_neg_te)
        ths = np.linspace(0.0, 1.0, num=301)
        R = []; U = []; A = []
        for t in ths:
            tp = (p_pos >= t).mean()
            tn = (p_neg < t).mean()
            R.append(tp); U.append(tn)
            A.append((tp + tn) / 2.0)
        best_idx = np.array(A).argmax()
        mean_acc = np.array(A).mean()
        ARUC = np.trapezoid(np.minimum(R, U), ths) if hasattr(np, "trapezoid") else np.trapz(np.minimum(R, U), ths)
        print(f"[Defense] Best @ λ={ths[best_idx]:.3f} | Robustness={R[best_idx]:.3f} | Uniqueness={U[best_idx]:.3f} | MeanAcc={A[best_idx]:.3f}")
        print(f"[Defense] Mean Test Accuracy: {mean_acc:.3f}")
        print(f"[Defense] ARUC: {ARUC:.3f}")
        return {'robustness': R[best_idx], 'uniqueness': U[best_idx], 'mean_accuracy': A[best_idx], 'aruc': ARUC}
