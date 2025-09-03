
import copy, gc, torch, torch.nn.functional as F
from torch.optim import Adam
from typing import List, Optional, Union
import torch.nn as nn
from ..core.base import BaseAttack
from ..models.backbones import GCN, GraphSAGE
from ..utils import AvgMeter, get_device

class GNNFingersAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets  = {"AIDS"}
    def __init__(self, dataset, attack_node_fraction: float=0.1,
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
        return {'target_model': self.target_model,'positive_models': self.positive_models,'negative_models': self.negative_models}

    def _train_target_model(self, epochs=200, lr=1e-3, max_pairs=800, early_stop_patience=20):
        model = GCN(self.num_features, hidden=self.victim_hidden, out_dim=self.victim_out, dropout=0.3).to(self.device)
        opt = Adam(model.parameters(), lr=lr)
        mse = AvgMeter()
        best_loss = float('inf'); patience_counter = 0
        model.train()
        for ep in range(1, epochs+1):
            mse.reset()
            for (g1,g2), sim in zip(self.dataset.train_pairs[:max_pairs], self.dataset.train_sims[:max_pairs]):
                g1 = g1.to(self.device); g2 = g2.to(self.device)
                opt.zero_grad()
                pred,_,_ = model.forward_pair(g1,g2)
                loss = F.mse_loss(pred, torch.tensor(sim, device=self.device).view(-1))
                loss.backward(); opt.step()
                mse.add(loss.item())
            current_loss = mse.avg
            if current_loss < best_loss:
                best_loss = current_loss; patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[Target] Early stopping at epoch {ep}")
                break
            if ep % 20 == 0:
                print(f"[Target] epoch {ep:03d}: MSE={current_loss:.4f}")
        return model.eval()

    def _load_model(self):
        return torch.load(self.model_path, map_location=self.device)

    def _train_attack_model(self): pass

    def _generate_suspects(self, k_pos=20, k_neg=20):
        print(f"[Debug] Starting suspect generation: {k_pos} positive, {k_neg} negative models")
        for i in range(k_pos):
            print(f"[Debug] Fine-tuning positive model {i+1}/{k_pos}")
            m = copy.deepcopy(self.target_model).to(self.device)
            self._fine_tune(m, epochs=10, lr=1e-3, max_pairs=800, seed=7+i)
            self.positive_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        opp = [1.0 - s for s in self.dataset.train_sims]
        for i in range(k_neg):
            print(f"[Debug] Training negative model {i+1}/{k_neg}")
            if i % 2 == 0:
                m = GCN(self.num_features, hidden=self.victim_hidden, out_dim=self.victim_out).to(self.device)
            else:
                m = GraphSAGE(self.num_features, hidden=self.victim_hidden, out_dim=self.victim_out).to(self.device)
            self._train_with_labels(m, labels=opp, epochs=100, lr=1e-3, max_pairs=800, seed=100+i)
            self.negative_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[Debug] Suspect generation completed")

    def _fine_tune(self, model, epochs=10, lr=1e-3, max_pairs=800, seed=0, early_stop_patience=5):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr)
        best_loss = float('inf'); patience_counter = 0
        model.train(); self.target_model.eval()
        for ep in range(epochs):
            mse = AvgMeter()
            for (g1,g2), _ in zip(self.dataset.train_pairs[:max_pairs], self.dataset.train_sims[:max_pairs]):
                g1 = g1.to(self.device); g2 = g2.to(self.device)
                opt.zero_grad()
                with torch.no_grad():
                    soft_label, _, _ = self.target_model.forward_pair(g1, g2)
                pred, _, _ = model.forward_pair(g1, g2)
                loss = F.mse_loss(pred, soft_label.view(-1))
                loss.backward(); opt.step()
                mse.add(loss.item())
            current_loss = mse.avg
            if current_loss < best_loss:
                best_loss = current_loss; patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[Fine-tune] Early stopping at epoch {ep+1}")
                break
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _train_with_labels(self, model, labels, epochs=100, lr=1e-3, max_pairs=800, seed=0, early_stop_patience=20):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr)
        best_loss = float('inf'); patience_counter = 0
        model.train()
        for ep in range(epochs):
            mse = AvgMeter()
            for (g1,g2), sim in zip(self.dataset.train_pairs[:max_pairs], labels[:max_pairs]):
                g1 = g1.to(self.device); g2 = g2.to(self.device)
                opt.zero_grad()
                pred, _, _ = model.forward_pair(g1, g2)
                loss = F.mse_loss(pred, torch.tensor(sim, device=self.device).view(-1))
                loss.backward(); opt.step()
                mse.add(loss.item())
            current_loss = mse.avg
            if current_loss < best_loss:
                best_loss = current_loss; patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[Neg model] Early stopping at epoch {ep+1}")
                break
            if torch.cuda.is_available(): torch.cuda.empty_cache()
