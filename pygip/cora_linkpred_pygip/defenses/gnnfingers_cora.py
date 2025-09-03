# defenses/gnnfingers_cora.py
# GNNFingersDefense + Fingerprint machinery for Cora (Link Prediction)

import random
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from datasets.cora import Dataset
from attacks.gnnfingers_cora import get_device, BaseAttack  # get_device reuse
# BaseDefense local copy to avoid external deps

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
    def defend(self):
        raise NotImplementedError
    def _load_model(self):
        raise NotImplementedError
    def _train_target_model(self):
        raise NotImplementedError
    def _train_defense_model(self):
        raise NotImplementedError
    def _train_surrogate_model(self):
        raise NotImplementedError

class FingerprintGraph(nn.Module):
    def __init__(self, num_nodes: int, feat_dim: int, edge_init_p: float=0.10, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.n = num_nodes
        self.d = feat_dim
        # Initialize node features
        X = torch.empty(self.n, self.d, device=self.device).uniform_(-0.5, 0.5)
        self.X = nn.Parameter(X)
        # Initialize adjacency matrix
        self._init_adj(edge_init_p)

    def _init_adj(self, p):
        A = (torch.rand(self.n, self.n, device=self.device) < p).float()
        A.fill_diagonal_(0.0)
        A.copy_(torch.maximum(A, A.T))
        eps = 1e-4
        self.A_logits = nn.Parameter(torch.logit(A.clamp(eps, 1-eps)))

    def edge_index(self, thresh=0.5):
        A = torch.sigmoid(self.A_logits)
        A = (A > thresh).float()
        A.fill_diagonal_(0.0)
        A = torch.maximum(A, A.T)
        idx = A.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        return idx.t().contiguous()

class FingerprintSet(nn.Module):
    def __init__(self, n_nodes: int, feat_dim: int, m_sampled_edges: int=32, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.m = m_sampled_edges  # Number of edges to sample
        self.fp = FingerprintGraph(n_nodes, feat_dim, device=self.device)

        # Sampled edge indices (fixed during training)
        self.num_possible_edges = n_nodes * (n_nodes - 1) // 2
        self.sampled_indices = torch.randperm(self.num_possible_edges, device=self.device)[:m_sampled_edges]

    def _idx_to_edge(self, idx):
        # Convert linear index from upper triangular to (i, j) with i < j
        n = self.fp.n
        # Derived mapping
        i = int((2 * n - 1 - torch.sqrt(torch.tensor((2 * n - 1)**2 - 8 * idx, dtype=torch.float32))) / 2)
        num_before_i = i * n - i * (i + 1) // 2
        j = i + 1 + (idx - num_before_i)
        return i, int(j)

    @torch.no_grad()
    def dim_out(self):
        # Each sampled edge contributes a concat of two node embeddings of size 'backbone_out'
        # We'll multiply by the backbone_out externally when building the verifier.
        return self.m * 2

    def model_response(self, model, require_grad=False):
        model.eval()
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            ei = self.fp.edge_index()
            data = Data(x=self.fp.X, edge_index=ei).to(self.device)

            # Get node embeddings
            z = model.encode(data.x, data.edge_index)

            # Sample edges and get their embeddings
            edge_embeddings = []
            for idx in self.sampled_indices:
                i, j = self._idx_to_edge(idx.item())
                edge_emb = torch.cat([z[i], z[j]])  # [2 * out_dim]
                edge_embeddings.append(edge_emb)

            # Concatenate embeddings from all sampled edges
            if len(edge_embeddings) > 0:
                return torch.cat(edge_embeddings)
            else:
                # Handle case with no sampled edges
                return torch.empty(0, z.size(1) * 2, device=self.device)

class Verifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64),  nn.LeakyReLU(0.01),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class GNNFingersDefense(BaseDefense):
    supported_api_types = {"pyg"}
    supported_datasets  = {"Cora"}
    def __init__(self, dataset: Dataset, attack_node_fraction: float=0.1,
                 device: Optional[Union[str, torch.device]]=None,
                 fp_nodes=100, backbone_out=64, joint_iters=150,
                 verif_lr=5e-4, fp_lr=2e-3):
        super().__init__(dataset, attack_node_fraction, device)
        self.fp_nodes = fp_nodes
        self.backbone_out = backbone_out
        self.joint_iters = joint_iters
        self.verif_lr = verif_lr
        self.fp_lr = fp_lr
        self.fp_set = FingerprintSet(
            n_nodes=self.dataset.num_nodes,
            feat_dim=self.num_features,
            m_sampled_edges=32,
            device=self.device
        )
        in_dim = self.fp_set.dim_out() * backbone_out
        self.verifier = Verifier(in_dim).to(self.device)
        self.target_model = None  # for eval convenience

    def defend(self, attack_results=None):
        from attacks.gnnfingers_cora import GNNFingersAttack  # local import to avoid cycle at import time
        if attack_results is None:
            attack = GNNFingersAttack(self.dataset, self.attack_node_fraction, device=self.device,
                                      victim_hidden=64, victim_out=self.backbone_out)
            attack_results = attack.attack()
        tgt = attack_results['target_model']
        pos = attack_results['positive_models']
        neg = attack_results['negative_models']

        self.target_model = tgt
        # Joint training
        self._joint_train(tgt, pos, neg)
        return {'fingerprint_set': self.fp_set, 'verifier': self.verifier}

    def _joint_train(self, target_model, pos_models, neg_models):
        opt_ver = Adam(self.verifier.parameters(), lr=self.verif_lr)
        fp_params = [p for p in self.fp_set.fp.parameters()]
        opt_fp = Adam(fp_params, lr=self.fp_lr)

        import random
        for it in range(1, self.joint_iters+1):
            # Randomly sample positive and negative models
            p_m = random.choice([target_model] + pos_models).to(self.device)
            n_m = random.choice(neg_models).to(self.device)

            # Phase A: Update verifier (freeze fingerprints)
            for p in fp_params: p.requires_grad_(False)
            self.verifier.train()
            opt_ver.zero_grad()

            X_pos = self.fp_set.model_response(p_m, require_grad=False)
            X_neg = self.fp_set.model_response(n_m, require_grad=False)
            X_batch = torch.stack([X_pos, X_neg], dim=0)

            logits = self.verifier(X_batch)
            labels = torch.tensor([1, 0], device=self.device, dtype=torch.long)
            loss_ver = F.cross_entropy(logits, labels)
            loss_ver.backward()
            opt_ver.step()

            # Phase B: Update fingerprints (freeze verifier)
            for p in fp_params: p.requires_grad_(True)
            self.verifier.eval()
            opt_fp.zero_grad()

            X_pos = self.fp_set.model_response(p_m, require_grad=True)
            X_neg = self.fp_set.model_response(n_m, require_grad=True)
            X_batch = torch.stack([X_pos, X_neg], dim=0)

            logits = self.verifier(X_batch)
            labels = torch.tensor([1, 0], device=self.device, dtype=torch.long)
            loss_fp = F.cross_entropy(logits, labels)
            loss_fp.backward()
            opt_fp.step()

            if it % 10 == 0:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    print(f"[Joint] iter {it:03d} | L_ver {loss_ver.item():.4f} | L_fp {loss_fp.item():.4f} | pos_p~{probs[0].item():.3f} | neg_p~{probs[1].item():.3f}")

    @torch.no_grad()
    def verify_ownership(self, suspect_model):
        suspect_model.eval()
        x = self.fp_set.model_response(suspect_model, require_grad=False)
        x = x.unsqueeze(0)
        logits = self.verifier(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        return {
            'is_stolen': int(prob > 0.5),
            'confidence': prob
        }
