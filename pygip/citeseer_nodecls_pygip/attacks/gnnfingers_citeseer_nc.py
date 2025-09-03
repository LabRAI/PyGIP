from typing import Optional, Union, List
import copy, gc, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.common import get_device, AvgMeter, set_seed, SEED_DEFAULT
from datasets.citeseer import Dataset
from models.node_classifiers import GCNNodeClassifier, SAGENodeClassifier

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

    def __init__(self, dataset: Dataset, attack_node_fraction: float=0.1,
                 model_path: Optional[str]=None, device: Optional[Union[str, torch.device]]=None,
                 victim_hidden=64, victim_out=64):
        super().__init__(dataset, attack_node_fraction, model_path, device)
        self.victim_hidden = victim_hidden
        self.victim_out = victim_out
        self.target_model = None
        self.positive_models: List[nn.Module] = []
        self.negative_models: List[nn.Module] = []

    def attack(self):
        self.target_model = self._load_model() if self.model_path else self._train_target_model()
        self._generate_suspects()
        return {
            'target_model': self.target_model,
            'positive_models': self.positive_models,
            'negative_models': self.negative_models,
        }

    def _train_target_model(self, epochs=200, lr=1e-3, early_stop_patience=20):
        model = GCNNodeClassifier(self.num_features, hidden=self.victim_hidden,
                                  out_dim=self.victim_out, dropout=0.3,
                                  num_classes=self.num_classes).to(self.device)
        opt = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        best_acc = 0.0; patience = 0
        data = self.graph_data.to(self.device)
        for ep in range(1, epochs+1):
            model.train(); opt.zero_grad()
            logits, _ = model(data)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                logits, _ = model(data)
                pred = logits.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            if val_acc > best_acc:
                best_acc = val_acc; patience = 0
            else:
                patience += 1
            if patience >= early_stop_patience:
                print(f"[Target] Early stopping at epoch {ep}")
                break
            if ep % 20 == 0:
                print(f"[Target] epoch {ep:03d} | Val Acc={val_acc:.4f}")
        return model.eval()

    def _load_model(self):
        return torch.load(self.model_path, map_location=self.device)

    def _train_attack_model(self): pass

    def _generate_suspects(self, k_pos=20, k_neg=20):
        print(f"[Attack] Building suspects: F+={k_pos}, F-={k_neg}")
        # Positives: fine-tune with distillation from target
        for i in range(k_pos):
            m = copy.deepcopy(self.target_model).to(self.device)
            self._fine_tune(m, epochs=10, lr=1e-3, seed=SEED_DEFAULT+i)
            self.positive_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Negatives: independent GCN/SAGE
        for i in range(k_neg):
            if i % 2 == 0:
                m = GCNNodeClassifier(self.num_features, hidden=self.victim_hidden,
                                      out_dim=self.victim_out, dropout=0.3,
                                      num_classes=self.num_classes).to(self.device)
            else:
                m = SAGENodeClassifier(self.num_features, hidden=self.victim_hidden,
                                       out_dim=self.victim_out, dropout=0.3,
                                       num_classes=self.num_classes).to(self.device)
            self._train_independent(m, epochs=100, lr=1e-3, seed=100+i)
            self.negative_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("[Attack] Suspects ready.")

    def _fine_tune(self, model, epochs=10, lr=1e-3, seed=0, early_stop_patience=5):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        data = self.graph_data.to(self.device)
        best_val = float('inf'); patience = 0
        model.train(); self.target_model.eval()
        for ep in range(epochs):
            opt.zero_grad()
            with torch.no_grad():
                soft_logits, _ = self.target_model(data)
                soft_labels = F.softmax(soft_logits, dim=1)
            logits, _ = model(data)
            loss = F.kl_div(F.log_softmax(logits[data.train_mask], dim=1),
                            soft_labels[data.train_mask], reduction='batchmean')
            loss.backward(); opt.step()
            with torch.no_grad():
                logits_v, _ = model(data)
                val_loss = F.cross_entropy(logits_v[data.val_mask], data.y[data.val_mask]).item()
            if val_loss < best_val:
                best_val = val_loss; patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"[Fine-tune] Early stop at {ep+1}")
                    break

    def _train_independent(self, model, epochs=100, lr=1e-3, seed=0, early_stop_patience=20):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        data = self.graph_data.to(self.device)
        best_acc = 0.0; patience = 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            logits, _ = model(data)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                logits, _ = model(data)
                pred = logits.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            if val_acc > best_acc:
                best_acc = val_acc; patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"[Neg model] Early stop at {ep+1}")
                    break
            if torch.cuda.is_available(): torch.cuda.empty_cache()
