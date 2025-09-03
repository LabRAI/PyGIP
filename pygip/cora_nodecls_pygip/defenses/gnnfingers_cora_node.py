# defenses/gnnfingers_cora_node.py
# GNNFingersDefense + Fingerprint machinery for Cora (Node Classification)

from typing import Optional, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from datasets.cora import Dataset
from attacks.gnnnfingers_cora_node import get_device, BaseAttack  # re-use get_device

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

# Fingerprint pair (two small graphs per fingerprint)
class FingerprintGraphPair(nn.Module):
    def __init__(self, n_nodes: int, feat_dim: int, edge_init_p: float = 0.10, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.n = n_nodes
        self.d = feat_dim
        X1 = torch.empty(self.n, self.d, device=self.device).uniform_(-0.5, 0.5)
        X2 = torch.empty(self.n, self.d, device=self.device).uniform_(-0.5, 0.5)
        self.X1 = nn.Parameter(X1)
        self.X2 = nn.Parameter(X2)
        self._init_adj(edge_init_p)

    def _init_adj(self, p):
        def make_A():
            A = (torch.rand(self.n, self.n, device=self.device) < p).float()
            A.fill_diagonal_(0.0)
            A.copy_(torch.maximum(A, A.T))
            eps = 1e-4
            return nn.Parameter(torch.logit(A.clamp(eps, 1-eps)))
        self.A1_logits = make_A()
        self.A2_logits = make_A()

    def _binarize(self, logits, thresh=0.5):
        A = torch.sigmoid(logits)
        A = (A > thresh).float()
        A.fill_diagonal_(0.0)
        A = torch.maximum(A, A.T)
        idx = A.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        return idx.t().contiguous()

    def edge_index_pair(self):
        return self._binarize(self.A1_logits), self._binarize(self.A2_logits)

class FingerprintSet(nn.Module):
    def __init__(self, P: int, n_nodes: int, feat_dim: int, num_classes: int, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.P = P
        self.num_classes = num_classes
        self.fps = nn.ModuleList([FingerprintGraphPair(n_nodes, feat_dim, device=self.device) for _ in range(P)])

    @torch.no_grad()
    def dim_out(self):
        # per fingerprint: z1 (C) || z2 (C)  => 2C ; total P => 2PC
        return self.P * (2 * self.num_classes)

    def model_response(self, model, require_grad=False):
        outs = []
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        model.eval()
        with ctx:
            for fp in self.fps:
                ei1, ei2 = fp.edge_index_pair()
                g1 = Data(x=fp.X1, edge_index=ei1).to(self.device)
                g2 = Data(x=fp.X2, edge_index=ei2).to(self.device)
                z1 = model(g1)[0]  # logits of node 0
                z2 = model(g2)[0]
                outs.append(torch.cat([z1, z2], dim=-1))
        return torch.cat(outs, dim=-1)

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
                 P=64, fp_nodes=32, backbone_out=64, joint_iters=150,
                 verif_lr=5e-4, fp_lr=2e-3):
        super().__init__(dataset, attack_node_fraction, device if device is not None else get_device())
        self.P = P
        self.fp_nodes = fp_nodes
        self.backbone_out = backbone_out
        self.joint_iters = joint_iters
        self.verif_lr = verif_lr
        self.fp_lr = fp_lr
        self.fp_set = FingerprintSet(P=self.P, n_nodes=self.fp_nodes,
                                     feat_dim=self.num_features, num_classes=self.num_classes, device=self.device)
        in_dim = self.fp_set.dim_out()
        self.verifier = Verifier(in_dim).to(self.device)
        self.target_model = None

    def defend(self, attack_results=None):
        from attacks.gnnnfingers_cora_node import GNNFingersAttack  # safe local import
        if attack_results is None:
            attack = GNNFingersAttack(self.dataset, self.attack_node_fraction, device=get_device(),
                                      victim_hidden=256, victim_out=self.backbone_out)
            attack_results = attack.attack()
        self.target_model = attack_results['target_model']
        pos = attack_results['positive_models']
        neg = attack_results['negative_models']
        self._joint_train(self.target_model, pos, neg)
        return {'fingerprint_set': self.fp_set, 'verifier': self.verifier, 'target_model': self.target_model}

    def _joint_train(self, target_model, pos_models, neg_models):
        opt_ver = Adam(self.verifier.parameters(), lr=self.verif_lr)
        fp_params = [p for fp in self.fp_set.fps for p in fp.parameters()]
        opt_fp = Adam(fp_params, lr=self.fp_lr)
        for it in range(1, self.joint_iters+1):
            p_m = random.choice([target_model] + pos_models).to(self.device)
            n_m = random.choice(neg_models).to(self.device)
            # A) update verifier
            for p in fp_params: p.requires_grad_(False)
            self.verifier.train(); opt_ver.zero_grad()
            x_pos = self.fp_set.model_response(p_m, require_grad=False)
            x_neg = self.fp_set.model_response(n_m, require_grad=False)
            X = torch.stack([x_pos, x_neg], dim=0)
            y = torch.tensor([1,0], device=self.device, dtype=torch.long)
            logits = self.verifier(X)
            loss_v = F.cross_entropy(logits, y)
            loss_v.backward(); opt_ver.step()
            # B) update fingerprints
            for p in fp_params: p.requires_grad_(True)
            self.verifier.eval(); opt_fp.zero_grad()
            x_pos = self.fp_set.model_response(p_m, require_grad=True)
            x_neg = self.fp_set.model_response(n_m, require_grad=True)
            X = torch.stack([x_pos, x_neg], dim=0)
            logits = self.verifier(X)
            y = torch.tensor([1,0], device=self.device, dtype=torch.long)
            loss_fp = F.cross_entropy(logits, y)
            loss_fp.backward(); opt_fp.step()
            if it % 10 == 0:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)[:,1]
                print(f"[Joint] {it:03d} | L_ver={loss_v.item():.4f} | L_fp={loss_fp.item():.4f} | p(pos)~{probs[0]:.3f} | p(neg)~{probs[1]:.3f}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    @torch.no_grad()
    def verify_ownership(self, suspect_model):
        suspect_model.eval()
        x = self.fp_set.model_response(suspect_model, require_grad=False).unsqueeze(0)
        logits = self.verifier(x)
        prob = torch.softmax(logits, dim=1)[0,1].item()
        return {'is_stolen': int(prob>0.5), 'confidence': prob}
