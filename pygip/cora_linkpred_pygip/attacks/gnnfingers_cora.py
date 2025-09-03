# attacks/gnnfingers_cora.py
# GNNFingersAttack implementation for Cora (Link Prediction)

import copy, gc, random
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import average_precision_score

from datasets.cora import Dataset
from models.link_predictors import GCNLinkPredictor, SAGELinkPredictor

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def attack(self):
        raise NotImplementedError
    def _load_model(self):
        raise NotImplementedError
    def _train_target_model(self):
        raise NotImplementedError
    def _train_attack_model(self):
        raise NotImplementedError

class GNNFingersAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets  = {"Cora"}
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
        model = GCNLinkPredictor(
            self.num_features,
            hidden=self.victim_hidden,
            out_dim=self.victim_out,
            dropout=0.3
        ).to(self.device)
        opt = Adam(model.parameters(), lr=lr)
        data = self.graph_data.to(self.device)

        best_auprc = 0.0
        patience_counter = 0
        model.train()

        for ep in range(1, epochs+1):
            # Training
            model.train()
            opt.zero_grad()
            pos_pred, neg_pred, _ = model(data)

            # Get predictions for train edges
            train_pos_pred = pos_pred[data.train_mask]
            train_neg_pred = neg_pred[data.neg_train_mask]

            # Create labels
            train_labels = torch.cat([
                torch.ones_like(train_pos_pred),
                torch.zeros_like(train_neg_pred)
            ])
            train_preds = torch.cat([train_pos_pred, train_neg_pred])

            loss = F.binary_cross_entropy_with_logits(train_preds, train_labels)
            loss.backward()
            opt.step()

            # Validation
            model.eval()
            with torch.no_grad():
                pos_pred, neg_pred, _ = model(data)
                val_pos_pred = pos_pred[data.val_mask]
                val_neg_pred = neg_pred[data.neg_val_mask]

                val_labels = torch.cat([
                    torch.ones_like(val_pos_pred),
                    torch.zeros_like(val_neg_pred)
                ])
                val_preds = torch.cat([val_pos_pred, val_neg_pred])
                val_probs = torch.sigmoid(val_preds)

                auprc = average_precision_score(val_labels.cpu(), val_probs.cpu())

            if auprc > best_auprc:
                best_auprc = auprc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"[Target] Early stopping at epoch {ep}")
                break

            if ep % 20 == 0:
                print(f"[Target] epoch {ep:03d}: Val AUPRC={auprc:.4f}")

        return model.eval()

    def _load_model(self):
        return torch.load(self.model_path, map_location=self.device)

    def _train_attack_model(self):
        pass

    def _generate_suspects(self, k_pos=20, k_neg=20):
        print(f"[Debug] Starting suspect generation: {k_pos} positive, {k_neg} negative models")

        # F+: Fine-tune target model (distillation-like)
        for i in range(k_pos):
            m = copy.deepcopy(self.target_model).to(self.device)
            self._fine_tune(m, epochs=10, lr=1e-3, seed=7+i)
            self.positive_models.append(m.eval())
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # F-: Train independent models (GCN/SAGE)
        for i in range(k_neg):
            if i % 2 == 0:
                m = GCNLinkPredictor(
                    self.num_features,
                    hidden=self.victim_hidden,
                    out_dim=self.victim_out,
                    dropout=0.3
                ).to(self.device)
            else:
                m = SAGELinkPredictor(
                    self.num_features,
                    hidden=self.victim_hidden,
                    out_dim=self.victim_out,
                    dropout=0.3
                ).to(self.device)
            self._train_independent(m, epochs=100, lr=1e-3, seed=100+i)
            self.negative_models.append(m.eval())
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[Debug] Suspect generation completed")

    def _fine_tune(self, model, epochs=10, lr=1e-3, seed=0, early_stop_patience=5):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr)
        data = self.graph_data.to(self.device)

        best_loss = float('inf')
        patience_counter = 0
        model.train()
        self.target_model.eval()

        for ep in range(epochs):
            # Training with distillation
            model.train()
            opt.zero_grad()

            # Get soft labels from target model
            with torch.no_grad():
                target_pos, target_neg, _ = self.target_model(data)
                target_pos_probs = torch.sigmoid(target_pos)
                target_neg_probs = torch.sigmoid(target_neg)

            # Get model predictions
            pos_pred, neg_pred, _ = model(data)
            pos_probs = torch.sigmoid(pos_pred)
            neg_probs = torch.sigmoid(neg_pred)

            # Distillation loss on training edges
            train_pos_probs = pos_probs[data.train_mask]
            train_neg_probs = neg_probs[data.neg_train_mask]
            target_train_pos_probs = target_pos_probs[data.train_mask]
            target_train_neg_probs = target_neg_probs[data.neg_train_mask]

            loss = F.mse_loss(train_pos_probs, target_train_pos_probs) +                    F.mse_loss(train_neg_probs, target_train_neg_probs)
            loss.backward()
            opt.step()

            # Validation
            model.eval()
            with torch.no_grad():
                target_pos, target_neg, _ = self.target_model(data)
                target_pos_probs = torch.sigmoid(target_pos)
                target_neg_probs = torch.sigmoid(target_neg)

                pos_pred, neg_pred, _ = model(data)
                pos_probs = torch.sigmoid(pos_pred)
                neg_probs = torch.sigmoid(neg_pred)

                val_pos_probs = pos_probs[data.val_mask]
                val_neg_probs = neg_probs[data.neg_val_mask]
                target_val_pos_probs = target_pos_probs[data.val_mask]
                target_val_neg_probs = target_neg_probs[data.neg_val_mask]

                val_loss = F.mse_loss(val_pos_probs, target_val_pos_probs) +                            F.mse_loss(val_neg_probs, target_val_neg_probs)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"[Fine-tune] Early stopping at epoch {ep+1}")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train_independent(self, model, epochs=100, lr=1e-3, seed=0, early_stop_patience=20):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr)
        data = self.graph_data.to(self.device)

        best_auprc = 0.0
        patience_counter = 0
        model.train()

        for ep in range(epochs):
            # Training
            model.train()
            opt.zero_grad()
            pos_pred, neg_pred, _ = model(data)

            # Get predictions for train edges
            train_pos_pred = pos_pred[data.train_mask]
            train_neg_pred = neg_pred[data.neg_train_mask]

            # Create labels
            train_labels = torch.cat([
                torch.ones_like(train_pos_pred),
                torch.zeros_like(train_neg_pred)
            ])
            train_preds = torch.cat([train_pos_pred, train_neg_pred])

            loss = torch.nn.functional.binary_cross_entropy_with_logits(train_preds, train_labels)
            loss.backward()
            opt.step()

            # Validation
            model.eval()
            with torch.no_grad():
                pos_pred, neg_pred, _ = model(data)
                val_pos_pred = pos_pred[data.val_mask]
                val_neg_pred = neg_pred[data.neg_val_mask]

                val_labels = torch.cat([
                    torch.ones_like(val_pos_pred),
                    torch.zeros_like(val_neg_pred)
                ])
                val_preds = torch.cat([val_pos_pred, val_neg_pred])
                val_probs = torch.sigmoid(val_preds)

                auprc = average_precision_score(val_labels.cpu(), val_probs.cpu())

            if auprc > best_auprc:
                best_auprc = auprc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"[Neg model] Early stopping at epoch {ep+1}")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
