# attacks/gnnfingers_cora_node.py
# GNNFingersAttack implementation for Cora (Node Classification)

import copy, gc
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from datasets.cora import Dataset
from models.node_classifiers import GCN, GraphSAGE

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.s, self.n = 0.0, 0
    def add(self, v, k=1): self.s += float(v)*k; self.n += k
    @property
    def avg(self): return self.s/max(1,self.n)

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
    supported_datasets  = {"Cora"}
    def __init__(self, dataset: Dataset, attack_node_fraction: float=0.1,
                 model_path: Optional[str]=None, device: Optional[Union[str, torch.device]]=None,
                 victim_hidden=256, victim_out=64):
        super().__init__(dataset, attack_node_fraction, model_path, device)
        self.victim_hidden = victim_hidden
        self.victim_out = victim_out
        self.target_model = None
        self.positive_models: List[nn.Module] = []
        self.negative_models: List[nn.Module] = []
        # masks
        self.train_mask = self.graph_data.train_mask
        self.val_mask   = self.graph_data.val_mask
        self.test_mask  = self.graph_data.test_mask

    def attack(self):
        self.target_model = self._load_model() if self.model_path else self._train_target_model()
        self._generate_suspects()
        return {
            'target_model': self.target_model,
            'positive_models': self.positive_models,
            'negative_models': self.negative_models,
        }

    def _train_target_model(self, epochs=200, lr=1e-3, patience=20):
        model = GCN(self.num_features, hidden=self.victim_hidden,
                    out_dim=self.victim_out, num_classes=self.num_classes).to(self.device)
        opt = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        acc = AvgMeter()
        best_val = float('inf'); best_model = None; patience_ctr = 0
        data = self.graph_data.to(self.device)
        for ep in range(1, epochs+1):
            acc.reset()
            model.train(); opt.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits[self.train_mask], data.y[self.train_mask])
            loss.backward(); opt.step()
            pred = logits[self.train_mask].argmax(dim=1)
            acc.add((pred == data.y[self.train_mask]).float().mean().item())
            with torch.no_grad():
                val_logits = model(data)
                val_loss = F.cross_entropy(val_logits[self.val_mask], data.y[self.val_mask]).item()
            if ep % 20 == 0:
                print(f"[Target] epoch {ep:03d}: Train Acc={acc.avg:.4f} | Val Loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss; best_model = copy.deepcopy(model); patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"[Target] Early stopping at epoch {ep}")
                    break
        return best_model.eval()

    def _load_model(self):
        return torch.load(self.model_path, map_location=self.device)

    def _train_attack_model(self): pass

    def _generate_suspects(self, k_pos=20, k_neg=20):
        print(f"[Suspects] Generating: F+={k_pos}, F-={k_neg}")
        # F+: fine-tune with distillation
        for i in range(k_pos):
            m = copy.deepcopy(self.target_model).to(self.device)
            self._fine_tune(m, epochs=10, lr=1e-3, seed=7+i)
            self.positive_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # F-: train independently with shuffled labels
        y_shuf = self.graph_data.y.clone()
        torch.manual_seed(7)
        y_shuf = y_shuf[torch.randperm(len(y_shuf))]
        for i in range(k_neg):
            m = (GCN if i%2==0 else GraphSAGE)(self.num_features, hidden=self.victim_hidden,
                                               out_dim=self.victim_out, num_classes=self.num_classes).to(self.device)
            self._train_with_labels(m, labels=y_shuf, epochs=100, lr=1e-3, seed=100+i)
            self.negative_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("[Suspects] Done.")

    def _fine_tune(self, model, epochs=10, lr=1e-3, seed=0, patience=5):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        best_val = float('inf'); best_model = None; pat=0
        data = self.graph_data.to(self.device)
        model.train(); self.target_model.eval()
        for ep in range(epochs):
            opt.zero_grad()
            with torch.no_grad():
                soft = F.softmax(self.target_model(data), dim=1)
            logits = model(data)
            loss = F.kl_div(F.log_softmax(logits[self.train_mask], dim=1),
                            soft[self.train_mask], reduction='batchmean')
            loss.backward(); opt.step()
            with torch.no_grad():
                val_logits = model(data)
                val_loss = F.cross_entropy(val_logits[self.val_mask], data.y[self.val_mask]).item()
            if val_loss < best_val:
                best_val = val_loss; best_model = copy.deepcopy(model); pat=0
            else:
                pat += 1
                if pat >= patience:
                    print(f"[F+] Early stop @ {ep+1}"); break
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return best_model.eval()

    def _train_with_labels(self, model, labels, epochs=100, lr=1e-3, seed=0, patience=20):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        best_val = float('inf'); best_model=None; pat=0
        data = self.graph_data.to(self.device)
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits[self.train_mask], labels[self.train_mask])
            loss.backward(); opt.step()
            with torch.no_grad():
                val_logits = model(data)
                val_loss = F.cross_entropy(val_logits[self.val_mask], labels[self.val_mask]).item()
            if ep % 20 == 0:
                print(f"[F-] epoch {ep+1}: Val Loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss; best_model = copy.deepcopy(model); pat=0
            else:
                pat += 1
                if pat >= patience:
                    print(f"[F-] Early stop @ {ep+1}"); break
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return best_model.eval()
