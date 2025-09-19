# src/custom_attack.py
import os
import random
from typing import Optional, Union

import torch
import torch.nn.functional as F

from src.dataset import Dataset
from src.attacks import BaseAttack
from src.models import GraphSAGE, GCN
from src.train_target import train_masked_target


def evaluate_model(model: torch.nn.Module, data, device: torch.device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        mask = getattr(data, "test_mask", None) or getattr(data, "val_mask", None)
        if mask is None:
            mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        return (preds[mask] == data.y[mask]).float().mean().item()


class FeatureFlipAttack(BaseAttack):
    """
    Custom attack that perturbs node features for a fraction of nodes.
    Conforms to the PyGIP BaseAttack API:
      - attack()
      - _load_model()
      - _train_target_model()
      - _train_attack_model()
    """

    supported_api_types = {"pyg"}
    supported_datasets = set()

    def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None,
                 device: Optional[Union[str, torch.device]] = None):
        # must call super() so BaseAttack sets self.device and graph fields
        super().__init__(dataset, attack_node_fraction, model_path, device)

        if not (0.0 < self.attack_node_fraction <= 1.0):
            raise ValueError("attack_node_fraction must be in (0,1].")

    def _seed(self, seed: int = 0):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self):
        """
        Try to load a model checkpoint at self.model_path.
        Expected checkpoint formats:
          - state_dict or
          - dict with 'state_dict' / 'model_state' and optional 'model_type' metadata
        Returns a model instance moved to self.device or None if no model_path provided.
        """
        if not self.model_path:
            return None
        if not os.path.exists(self.model_path):
            print(f"[FeatureFlipAttack] model_path {self.model_path} not found.")
            return None

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model_type = checkpoint.get("model_type", None) if isinstance(checkpoint, dict) else None

        # minimal heuristic for model_type
        if model_type is None and isinstance(state, dict):
            keys = list(state.keys())
            if any("lin_l" in k or "lin_r" in k for k in keys):
                model_type = "GraphSAGE"
            else:
                model_type = "GCN"

        # instantiate appropriate model
        in_dim = self.num_features
        out_dim = self.num_classes
        hid = checkpoint.get("hidden", 64) if isinstance(checkpoint, dict) else 64

        if model_type == "GraphSAGE":
            model = GraphSAGE(in_dim, out_dim, hidden=hid).to(self.device)
        else:
            model = GCN(in_dim, out_dim, hidden=hid).to(self.device)

        try:
            # try multiple common keys
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = state
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"[FeatureFlipAttack] Failed to load weights: {e}")
            return None

    def _train_target_model(self, train_epochs: int = 100, lr: float = 1e-2, seed: int = 0):
        """
        Train a victim model (GraphSAGE) on the clean dataset and return the trained model.
        Uses self.graph_data and self.device.
        """
        self._seed(seed)
        data = self.graph_data.to(self.device)
        model = GraphSAGE(self.num_features, self.num_classes, hidden=64).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(train_epochs):
            opt.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            opt.step()

        model.eval()
        return model

    def _train_attack_model(self, dataset_name: str = "Cora", mask_ratio: float = 0.12,
                            hidden: int = 64, epochs: int = 200, seed: int = 0):
        """
        Train a surrogate/attack model using the existing train_masked_target routine.
        Returns path to checkpoint saved by train_masked_target.
        """
        # train_masked_target will handle device selection via its arguments
        ckpt_path = train_masked_target(dataset_name=dataset_name,
                                        mask_ratio=mask_ratio,
                                        hidden=hidden,
                                        epochs=epochs,
                                        seed=seed,
                                        device=self.device)
        return ckpt_path

    def _perturb_features(self, data, fraction: float):
        """Return a clone of data with fraction of node features replaced by Gaussian noise."""
        pert = data.clone()
        num_nodes = pert.num_nodes
        k = max(1, int(fraction * num_nodes))
        idx = torch.randperm(num_nodes)[:k]
        noise = torch.randn((k, pert.x.size(1)), device=pert.x.device) * 0.5
        pert.x[idx] = noise
        return pert, k

    def attack(self, retrain_target: bool = True, retrain_epochs: int = 50, seed: int = 0):
        """
        Main attack logic:
          1) Load or train a target model
          2) Evaluate baseline
          3) Perturb features on a fraction of nodes
          4) Optionally retrain a model on perturbed data and evaluate
        Returns dict with metrics.
        """
        self._seed(seed)

        if self.graph_data is None:
            raise RuntimeError("No graph_data available in dataset.")

        # Step 1: get target model
        model = self._load_model() or (self._train_target_model(train_epochs=retrain_epochs, seed=seed) if retrain_target else None)
        if model is None:
            model = self._train_target_model(train_epochs=retrain_epochs, seed=seed)

        # Step 2: eval before
        acc_before = evaluate_model(model, self.graph_data, self.device)

        # Step 3: perturb features
        perturbed_data, num_perturbed = self._perturb_features(self.graph_data, self.attack_node_fraction)

        # Step 4: retrain and evaluate on perturbed graph
        model_pert = GraphSAGE(self.num_features, self.num_classes, hidden=64).to(self.device)
        opt = torch.optim.Adam(model_pert.parameters(), lr=1e-2)
        perturbed_data = perturbed_data.to(self.device)

        model_pert.train()
        for _ in range(retrain_epochs):
            opt.zero_grad()
            logits = model_pert(perturbed_data.x, perturbed_data.edge_index)
            loss = F.cross_entropy(logits[perturbed_data.train_mask], perturbed_data.y[perturbed_data.train_mask])
            loss.backward()
            opt.step()

        acc_after = evaluate_model(model_pert, perturbed_data, self.device)

        results = {
            "attack_name": "FeatureFlipAttack",
            "attack_fraction": self.attack_node_fraction,
            "num_perturbed": num_perturbed,
            "acc_before": acc_before,
            "acc_after": acc_after,
        }
        return results
