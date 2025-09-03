
import torch, random
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.optim import Adam
from torch_geometric.data import Data
from core.base import BaseDefense
from utils.common import get_device
# ---- Fingerprints ----
class FingerprintGraph(nn.Module):
    def __init__(self, n_nodes: int, feat_dim: int, edge_init_p: float = 0.10, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.n = n_nodes
        self.d = feat_dim
        X = torch.empty(self.n, self.d, device=self.device).uniform_(-0.5, 0.5)
        self.X = nn.Parameter(X)
        self._init_adj(edge_init_p)
    def _init_adj(self, p):
        A = (torch.rand(self.n, self.n, device=self.device) < p).float()
        A.fill_diagonal_(0.0)
        A.copy_(torch.maximum(A, A.T))
        eps = 1e-4
        self.A_logits = nn.Parameter(torch.logit(A.clamp(eps, 1-eps)))
    def edge_index(self, thresh=0.5, update=False):
        A = torch.sigmoid(self.A_logits)
        if update and self.A_logits.grad is not None:
            grad = self.A_logits.grad
            mask = torch.sign(grad)
            new_A = A.clone()
            new_A = torch.where((mask > 0) & (A < 0.5), torch.ones_like(new_A), new_A)
            new_A = torch.where((mask < 0) & (A >= 0.5), torch.zeros_like(new_A), new_A)
            new_A.fill_diagonal_(0.0)
            new_A = torch.maximum(new_A, new_A.T)
            idx = new_A.nonzero(as_tuple=False)
            if idx.numel() == 0:
                return torch.empty(2, 0, dtype=torch.long, device=self.device)
            return idx.t().contiguous()
        else:
            A = (A > thresh).float()
            A.fill_diagonal_(0.0)
            A = torch.maximum(A, A.T)
            idx = A.nonzero(as_tuple=False)
            if idx.numel() == 0:
                return torch.empty(2, 0, dtype=torch.long, device=self.device)
            return idx.t().contiguous()

class FingerprintSet(nn.Module):
    def __init__(self, P: int, n_nodes: int, feat_dim: int, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.P = P
        self.fps = nn.ModuleList([FingerprintGraph(n_nodes, feat_dim, device=self.device) for _ in range(P)])
    @torch.no_grad()
    def dim_out(self, backbone_out_dim: int, num_classes: int):
        return self.P * num_classes
    def model_response(self, model, require_grad=False):
        outs = []
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        model.eval()
        with ctx:
            for fp in self.fps:
                ei = fp.edge_index()
                batch = torch.zeros(fp.n, dtype=torch.long, device=self.device)
                data = Data(x=fp.X, edge_index=ei, batch=batch).to(self.device)
                logits, _ = model(data)
                probs = F.softmax(logits, dim=1)  # shape [1, num_classes]
                outs.append(probs)
        return torch.cat(outs, dim=1)  # shape [1, P*num_classes]

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
    supported_datasets  = {"PROTEINS"}
    def __init__(self, dataset, attack_node_fraction: float = 0.1,
                 device=None, P=64, fp_nodes=32, backbone_out=64, joint_iters=150,
                 verif_lr=5e-4, fp_lr=2e-3):
        super().__init__(dataset, attack_node_fraction, device)
        self.P = P
        self.fp_nodes = fp_nodes
        self.backbone_out = backbone_out
        self.joint_iters = joint_iters
        self.verif_lr = verif_lr
        self.fp_lr = fp_lr
        self.fp_set = FingerprintSet(P=self.P, n_nodes=self.fp_nodes, feat_dim=self.num_features, device=self.device)
        in_dim = self.fp_set.dim_out(self.backbone_out, self.num_classes)
        self.verifier = Verifier(in_dim).to(self.device)

    def defend(self, attack_results: Dict = None):
        if attack_results is None:
            from attacks.gnnfingers_proteins_gc import GNNFingersAttack
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
        fp_params = [p for fp in self.fp_set.fps for p in fp.parameters()]
        opt_fp = Adam(fp_params, lr=self.fp_lr)
        for it in range(1, self.joint_iters+1):
            p_m = random.choice([target_model] + pos_models).to(self.device)
            n_m = random.choice(neg_models).to(self.device)
            # Phase A
            for p in fp_params: p.requires_grad_(False)
            self.verifier.train(); opt_ver.zero_grad()
            X_pos = self.fp_set.model_response(p_m, require_grad=False)  # [1, D]
            X_neg = self.fp_set.model_response(n_m, require_grad=False)  # [1, D]
            X_batch = torch.cat([X_pos, X_neg], dim=0)                  # [2, D]
            logits = self.verifier(X_batch)                              # [2, 2]
            labels = torch.tensor([1,0], dtype=torch.long, device=self.device)
            loss_ver = F.cross_entropy(logits, labels)
            loss_ver.backward(); opt_ver.step()
            # Phase B
            for p in fp_params: p.requires_grad_(True)
            self.verifier.eval(); opt_fp.zero_grad()
            X_pos = self.fp_set.model_response(p_m, require_grad=True)
            X_neg = self.fp_set.model_response(n_m, require_grad=True)
            X_batch = torch.cat([X_pos, X_neg], dim=0)
            logits = self.verifier(X_batch)
            labels = torch.tensor([1,0], dtype=torch.long, device=self.device)
            loss_fp = F.cross_entropy(logits, labels)
            loss_fp.backward(); opt_fp.step()
            if it % 10 == 0:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)[:,1]
                    print(f"[Joint] iter {it:03d} | L_ver {loss_ver.item():.4f} | L_fp {loss_fp.item():.4f} | pos~{probs[0].item():.3f} | neg~{probs[1].item():.3f}")

    @torch.no_grad()
    def verify_ownership(self, suspect_model):
        suspect_model.eval()
        x = self.fp_set.model_response(suspect_model, require_grad=False)  # [1, D]
        logits = self.verifier(x)                                          # [1, 2]
        prob = torch.softmax(logits, dim=1)[0,1].item()
        return {'is_stolen': int(prob > 0.5), 'confidence': prob}

def evaluate(defense, pos_models, neg_models):
    print("Evaluating...")
    tpr = sum(defense.verify_ownership(m)['is_stolen'] for m in pos_models) / max(1, len(pos_models))
    tnr = sum(1 - defense.verify_ownership(m)['is_stolen'] for m in neg_models) / max(1, len(neg_models))
    aruc = 0.5 * (tpr + tnr)
    # Mean test accuracy of target on PROTEINS test split
    target = getattr(defense, 'target_model', None) or (pos_models[0] if pos_models else None)
    correct = total = 0
    if target is not None:
        target.eval()
        with torch.no_grad():
            for batch in defense.dataset.test_data:
                batch = batch.to(defense.device)
                logits, _ = target(batch)
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total   += batch.y.size(0)
    mean_acc = correct / max(1, total)
    return {'robustness_TPR': tpr, 'uniqueness_TNR': tnr, 'ARUC': aruc, 'mean_test_accuracy': mean_acc}
