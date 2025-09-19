# src/custom_defense.py
import torch
import torch.nn.functional as F

from src.dataset import Dataset
from src.defenses import BaseDefense
from src.models import GraphSAGE
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


class NeighborSmoothingDefense(BaseDefense):
    """
    Defense that smooths features by neighbor averaging and retrains a model.
    Implements required BaseDefense hooks:
      - defend()
      - _load_model()
      - _train_target_model()
      - _train_defense_model()
      - _train_surrogate_model()
    """

    supported_api_types = {"pyg"}
    supported_datasets = set()

    def __init__(self, dataset: Dataset, attack_node_fraction: float, device: Optional[torch.device] = None):
        super().__init__(dataset, attack_node_fraction, device)

    @staticmethod
    def smooth_features(data):
        row, col = data.edge_index
        acc = torch.zeros_like(data.x)
        deg = torch.zeros(data.num_nodes, device=data.x.device)
        acc.index_add_(0, row, data.x[col])
        deg.index_add_(0, row, torch.ones(col.size(0), device=data.x.device))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        return acc / deg

    def _load_model(self):
        """
        Attempt to load a model from self.model_path (optional).
        Mirrors _load_model style from attack.
        """
        if not getattr(self, "model_path", None):
            return None
        if not os.path.exists(self.model_path):
            print(f"[NeighborSmoothingDefense] model_path {self.model_path} not found.")
            return None

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model_type = checkpoint.get("model_type", None) if isinstance(checkpoint, dict) else None

        if model_type is None and isinstance(state, dict):
            keys = list(state.keys())
            if any("lin_l" in k or "lin_r" in k for k in keys):
                model_type = "GraphSAGE"
            else:
                model_type = "GCN"

        in_dim = self.num_features
        out_dim = self.num_classes
        hid = checkpoint.get("hidden", 64) if isinstance(checkpoint, dict) else 64

        if model_type == "GraphSAGE":
            model = GraphSAGE(in_dim, out_dim, hidden=hid).to(self.device)
        else:
            model = GraphSAGE(in_dim, out_dim, hidden=hid).to(self.device)

        try:
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
            print(f"[NeighborSmoothingDefense] load failed: {e}")
            return None

    def _train_target_model(self, data, epochs: int = 50, lr: float = 1e-2, seed: int = 0):
        """
        Train a standard target model on the provided data and return it.
        This matches the framework hook signature: accepts data and returns model.
        """
        torch.manual_seed(seed)
        data = data.to(self.device)
        model = GraphSAGE(self.num_features, self.num_classes, hidden=64).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            opt.step()

        model.eval()
        return model

    def _train_defense_model(self, data, epochs: int = 50, lr: float = 1e-2, seed: int = 0):
        """
        Train model using defense-prepared data (e.g., smoothed features).
        """
        # Behavior mirrors _train_target_model; kept separate for clarity
        return self._train_target_model(data, epochs=epochs, lr=lr, seed=seed)

    def _train_surrogate_model(self, dataset_name: str = "Cora", mask_ratio: float = 0.12,
                               hidden: int = 64, epochs: int = 200, seed: int = 0):
        """
        Train a surrogate (attacker's) model using the provided train_masked_target helper.
        Returns path to saved checkpoint.
        """
        ckpt_path = train_masked_target(dataset_name=dataset_name,
                                        mask_ratio=mask_ratio,
                                        hidden=hidden,
                                        epochs=epochs,
                                        seed=seed,
                                        device=self.device)
        return ckpt_path

    def defend(self, retrain_epochs: int = 50, seed: int = 0):
        """
        Defense workflow:
          1) Train baseline target
          2) Train surrogate (optional) - returns ckpt path for analysis
          3) Apply smoothing and train defense model
          4) Return metrics dictionary
        """
        if self.graph_data is None:
            raise RuntimeError("No graph_data available in dataset.")

        # Baseline target (trained on original data)
        baseline_model = self._train_target_model(self.graph_data, epochs=retrain_epochs, seed=seed)
        acc_baseline = evaluate_model(baseline_model, self.graph_data, self.device)

        # Optionally train surrogate for evaluation/debug (not used directly here)
        # surrogate_ckpt = self._train_surrogate_model(dataset_name=self.dataset.dataset_name,
        #                                              mask_ratio=0.12, hidden=64, epochs=200, seed=seed)

        # Apply smoothing to features
        smoothed = self.graph_data.clone()
        smoothed.x = self.smooth_features(smoothed)

        # Train defense model on smoothed features
        defense_model = self._train_defense_model(smoothed, epochs=max(10, retrain_epochs // 2), seed=seed)
        acc_defended = evaluate_model(defense_model, smoothed, self.device)

        results = {
            "defense_name": "NeighborSmoothingDefense",
            "acc_baseline": acc_baseline,
            "acc_defended": acc_defended,
            "attack_node_fraction": self.attack_node_fraction,
        }
        return results
