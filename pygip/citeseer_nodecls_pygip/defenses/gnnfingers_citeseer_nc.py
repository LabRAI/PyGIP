from typing import Optional, Union, List
import random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.common import get_device
from datasets.citeseer import Dataset
from attacks.gnnfingers_citeseer_nc import BaseAttack, GNNFingersAttack

# ---- Fingerprints ----
from torch_geometric.data import Data

class FingerprintGraph(nn.Module):
    def __init__(self, num_nodes: int, feat_dim: int, edge_init_p: float=0.10, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.n = num_nodes; self.d = feat_dim
        X = torch.empty(self.n, self.d, device=self.device).uniform_(-0.5, 0.5)
        self.X = nn.Parameter(X)
        self._init_adj(edge_init_p)

    def _init_adj(self, p):
        A = (torch.rand(self.n, self.n, device=self.device) < p).float()
        A.fill_diagonal_(0.0); A.copy_(torch.maximum(A, A.T))
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
    def __init__(self, n_nodes: int, feat_dim: int, num_classes: int, m_sampled_nodes: int=64, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.m = m_sampled_nodes
        self.fp = FingerprintGraph(n_nodes, feat_dim, device=self.device)
        self.num_classes = num_classes
        self.sampled_indices = torch.randperm(n_nodes, device=self.device)[:m_sampled_nodes]

    @torch.no_grad()
    def dim_out(self):
        return self.m * self.num_classes

    def model_response(self, model, require_grad=False):
        model.eval()
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            ei = self.fp.edge_index()
            data = Data(x=self.fp.X, edge_index=ei).to(self.device)
            logits, _ = model(data)
            probs = F.softmax(logits, dim=1)
            sampled_probs = probs[self.sampled_indices]
            return sampled_probs.flatten()

class Verifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64),  nn.LeakyReLU(0.01),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# ---- BaseDefense + GNNFingersDefense ----
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

class GNNFingersDefense(BaseDefense):
    supported_api_types = {"pyg"}
    supported_datasets  = {"Citeseer"}

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
        self.fp_set = FingerprintSet(n_nodes=self.fp_nodes,
                                     feat_dim=self.num_features,
                                     num_classes=self.num_classes,
                                     m_sampled_nodes=64,
                                     device=self.device)
        in_dim = self.fp_set.dim_out()
        self.verifier = Verifier(in_dim).to(self.device)
        self.target_model = None

    def defend(self, attack_results=None):
        if attack_results is None:
            attack = GNNFingersAttack(self.dataset, self.attack_node_fraction, device=self.device,
                                      victim_hidden=64, victim_out=self.backbone_out)
            attack_results = attack.attack()
        self.target_model = attack_results['target_model']
        pos = attack_results['positive_models']
        neg = attack_results['negative_models']
        self._joint_train(self.target_model, pos, neg)
        return {'fingerprint_set': self.fp_set, 'verifier': self.verifier, 'target_model': self.target_model}

    def _joint_train(self, target_model, pos_models, neg_models):
        opt_ver = Adam(self.verifier.parameters(), lr=self.verif_lr)
        fp_params = [p for p in self.fp_set.fp.parameters()]
        opt_fp = Adam(fp_params, lr=self.fp_lr)
        for it in range(1, self.joint_iters+1):
            p_m = random.choice([target_model] + pos_models).to(self.device)
            n_m = random.choice(neg_models).to(self.device)
            # Phase A: verifier
            for p in fp_params: p.requires_grad_(False)
            self.verifier.train(); opt_ver.zero_grad()
            X_pos = self.fp_set.model_response(p_m, require_grad=False)
            X_neg = self.fp_set.model_response(n_m, require_grad=False)
            X = torch.stack([X_pos, X_neg], dim=0)
            logits = self.verifier(X)
            labels = torch.tensor([1, 0], device=self.device, dtype=torch.long)
            loss_ver = F.cross_entropy(logits, labels)
            loss_ver.backward(); opt_ver.step()
            # Phase B: fingerprints
            for p in fp_params: p.requires_grad_(True)
            self.verifier.eval(); opt_fp.zero_grad()
            X_pos = self.fp_set.model_response(p_m, require_grad=True)
            X_neg = self.fp_set.model_response(n_m, require_grad=True)
            X = torch.stack([X_pos, X_neg], dim=0)
            logits = self.verifier(X)
            labels = torch.tensor([1, 0], device=self.device, dtype=torch.long)
            loss_fp = F.cross_entropy(logits, labels)
            loss_fp.backward(); opt_fp.step()
            if it % 10 == 0:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    print(f"[Joint] iter {it:03d} | L_ver {loss_ver.item():.4f} | L_fp {loss_fp.item():.4f} | pos_p~{probs[0].item():.3f} | neg_p~{probs[1].item():.3f}")

    @torch.no_grad()
    def verify_ownership(self, suspect_model):
        x = self.fp_set.model_response(suspect_model, require_grad=False).unsqueeze(0)
        logits = self.verifier(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        return {'is_stolen': int(prob > 0.5), 'confidence': prob}

def evaluate(defense: GNNFingersDefense, pos_models, neg_models):
    print("Evaluating...")
    tpr_list, tnr_list = [], []
    for m in pos_models:
        res = defense.verify_ownership(m); tpr_list.append(res['is_stolen'])
    for m in neg_models:
        res = defense.verify_ownership(m); tnr_list.append(1 - res['is_stolen'])
    tpr = sum(tpr_list)/len(tpr_list) if tpr_list else 0.0
    tnr = sum(tnr_list)/len(tnr_list) if tnr_list else 0.0
    aruc = (tpr + tnr) / 2.0
    # test accuracy on target
    target = defense.target_model
    target.eval()
    with torch.no_grad():
        data = defense.graph_data.to(defense.device)
        logits, _ = target(data)
        pred = logits.argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return {'robustness_TPR': tpr, 'uniqueness_TNR': tnr, 'ARUC': aruc, 'mean_test_accuracy': test_acc}
