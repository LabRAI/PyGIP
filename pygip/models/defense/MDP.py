"""
MDP (Matrix Decomposition + Differential Privacy) Defense

A privacy-preserving defense mechanism for Graph Neural Networks that:
1. Builds normalized adjacency matrix: Abar = I + D^(-1/2)AD^(-1/2)
2. Splits Abar into nc shares via eigendecomposition
3. Applies Laplace noise to features for differential privacy
4. Trains multiple "calculators" on different shares with federated averaging

Reference:
    Privacy-Preserving GNN Based on Matrix Decomposition and Differential Privacy
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygip.models.defense.base import BaseDefense
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric
from pygip.utils.mdp.abar import build_abar_dense
from pygip.utils.mdp.eigenvalue_split import es_split_into_nc
from pygip.utils.mdp.dp_features import dp_features_laplace
from pygip.utils.mdp.splits import make_overlapping_train_masks
from pygip.models.nn.mdp_gcn import ManualGCN


class MDP(BaseDefense):
    """
    Matrix Decomposition + Differential Privacy defense for GNNs.

    This defense provides privacy preservation by:
    - Splitting the adjacency matrix into multiple shares via eigendecomposition
    - Adding calibrated Laplace noise to node features
    - Training multiple local models on different shares with parameter averaging

    Attributes:
        supported_api_types: Set of compatible API types (pyg only due to dense matrices)
        supported_datasets: Set of compatible dataset names (empty = all supported)
    """

    supported_api_types = {"pyg"}
    supported_datasets = set()

    def __init__(
        self,
        dataset,
        attack_node_fraction: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        nc: int = 4,
        es: int = 2,
        epsilon: float = 30.0,
        keep_ratio: float = 1.0,
        hidden_dim: int = 16,
        dropout: float = 0.5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        epochs: int = 200,
        patience: int = 50,
        seed: int = 42,
    ):
        """
        Initialize MDP defense.

        Args:
            dataset: PyGIP Dataset instance (must have api_type='pyg')
            attack_node_fraction: Fraction of nodes considered under attack
            device: Torch device (auto-detected if None)
            nc: Number of calculators for federated training
            es: Number of eigenvalue shares (must be >= 2)
            epsilon: Differential privacy budget (float('inf') for no noise)
            keep_ratio: Fraction of training nodes each calculator sees (0,1]
            hidden_dim: Hidden layer dimension for GCN
            dropout: Dropout probability
            lr: Learning rate
            weight_decay: L2 regularization weight
            epochs: Maximum training epochs
            patience: Early stopping patience
            seed: Random seed for reproducibility
        """
        super().__init__(dataset, attack_node_fraction, device)

        if nc < 2:
            raise ValueError("nc (number of calculators) must be >= 2")
        if es < 2:
            raise ValueError("es (number of eigenvalue shares) must be >= 2")
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1]")
        if epsilon <= 0 and epsilon != float('inf'):
            raise ValueError("epsilon must be > 0 or float('inf')")

        self.nc = nc
        self.es = es
        self.epsilon = epsilon
        self.keep_ratio = keep_ratio
        self.seed = seed

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience

        self.defense_model: Optional[nn.Module] = None
        self.Abar: Optional[torch.Tensor] = None
        self.Abar_shares: Optional[List[torch.Tensor]] = None
        self.X_noised: Optional[torch.Tensor] = None
        self._training_stats: Optional[Dict] = None

    def defend(self):
        """
        Execute the MDP defense.

        Returns:
            Tuple of (res, res_comp) where:
                - res: Dictionary from DefenseMetric.compute()
                - res_comp: Dictionary from DefenseCompMetric.compute()
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================MDP Defense====================")

        # Build adjacency and apply DP
        self._build_adjacency()
        self._apply_dp_noise()
        self._split_adjacency()

        # Train defense model
        defense_s = time.time()
        self.defense_model = self._train_defense_model()
        defense_e = time.time()
        metric_comp.update(defense_time=(defense_e - defense_s))

        # Evaluate
        inference_s = time.time()
        preds, labels = self._get_predictions()
        inference_e = time.time()

        # Compute metrics
        metric = DefenseMetric()
        metric.update(preds, labels)
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        res_comp = metric_comp.compute()

        return res, res_comp

    def _build_adjacency(self) -> None:
        """Build normalized adjacency matrix Abar = I + D^(-1/2)AD^(-1/2)."""
        data = self.graph_data
        edge_index = data.edge_index.to(self.device)
        num_nodes = self.num_nodes

        abar_result = build_abar_dense(
            edge_index=edge_index,
            num_nodes=num_nodes,
            device=self.device
        )
        self.Abar = abar_result.Abar.to(torch.float32)

    def _apply_dp_noise(self) -> None:
        """Apply Laplace noise to features for differential privacy."""
        data = self.graph_data
        X = data.x.to(self.device).to(torch.float32)

        dp_result = dp_features_laplace(
            X,
            epsilon=self.epsilon,
            delta=1.0,
            clip_min=0.0,
            clip_max=1.0,
            seed=self.seed
        )
        self.X_noised = dp_result.X_dp

    def _split_adjacency(self) -> None:
        """Split adjacency matrix into shares via eigendecomposition."""
        es_result = es_split_into_nc(
            self.Abar,
            nc=self.es,
            seed=self.seed,
            assume_symmetric=True
        )

        self.Abar_shares = [
            es_result.shares[i % self.es].to(torch.float32)
            for i in range(self.nc)
        ]

    def _train_target_model(self):
        """
        Train the target model (baseline without defense).

        Returns:
            torch.nn.Module: The trained target model
        """
        print("Training target model...")

        in_dim = self.num_features
        out_dim = self.num_classes

        model = ManualGCN(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=out_dim,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        data = self.graph_data
        X = data.x.to(self.device).to(torch.float32)
        y = data.y.to(self.device)
        train_mask = data.train_mask.to(self.device)
        val_mask = data.val_mask.to(self.device)

        best_val = -1.0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            model.train()
            logits = model(self.Abar, X)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(self.Abar, X)
                val_acc = self._accuracy(logits, y, val_mask)

            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state, strict=True)

        print(f"Target model trained. Val accuracy: {best_val:.4f}")
        return model

    def _train_defense_model(self):
        """
        Train the defense model using federated averaging across calculators.

        Returns:
            torch.nn.Module: The trained defense model
        """
        print("Training defense model with MDP...")

        data = self.graph_data

        train_masks_per_calc = None
        if self.nc > self.es:
            train_masks_per_calc = make_overlapping_train_masks(
                data.train_mask.to(self.device),
                nc=self.nc,
                seed=self.seed,
                keep_ratio=self.keep_ratio
            )

        in_dim = self.num_features
        out_dim = self.num_classes

        def model_ctor():
            return ManualGCN(
                in_dim=in_dim,
                hidden_dim=self.hidden_dim,
                out_dim=out_dim,
                dropout=self.dropout
            )

        stats, best_state = self._federated_train(
            model_ctor=model_ctor,
            Abar_full=self.Abar,
            Abar_shares=self.Abar_shares,
            X=self.X_noised,
            y=data.y.to(self.device),
            train_mask=data.train_mask.to(self.device),
            val_mask=data.val_mask.to(self.device),
            train_masks_per_calc=train_masks_per_calc,
        )

        self._training_stats = stats

        model = model_ctor().to(self.device)
        model.load_state_dict(best_state, strict=True)

        print(f"Defense model trained. Val accuracy: {stats['val_acc'][-1]:.4f}")
        return model

    def _train_surrogate_model(self):
        """
        Train surrogate model (for attack evaluation).

        Returns:
            torch.nn.Module: The trained surrogate model
        """
        return self._train_target_model()

    def _federated_train(
        self,
        model_ctor,
        Abar_full: torch.Tensor,
        Abar_shares: List[torch.Tensor],
        X: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        train_masks_per_calc: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict, Dict[str, torch.Tensor]]:
        """
        Federated training loop with parameter averaging.

        Each calculator trains on its assigned adjacency share.
        After each epoch, parameters are averaged across all calculators.
        """
        nc = len(Abar_shares)

        local_models = [model_ctor().to(self.device) for _ in range(nc)]
        optimizers = [
            torch.optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for m in local_models
        ]

        train_loss_hist = []
        val_acc_hist = []

        best_val = -1.0
        best_state = None
        epochs_since_improve = 0

        for epoch in range(1, self.epochs + 1):
            local_states = []
            local_losses = []

            for i in range(nc):
                model = local_models[i]
                opt = optimizers[i]
                model.train()

                A_share = Abar_shares[i].to(self.device)
                logits = model(A_share, X)

                tm = train_masks_per_calc[i] if train_masks_per_calc else train_mask
                loss = F.cross_entropy(logits[tm], y[tm])

                opt.zero_grad()
                loss.backward()
                opt.step()

                local_losses.append(loss.item())
                local_states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})

            avg_state = self._average_state_dicts(local_states)
            for model in local_models:
                model.load_state_dict(avg_state, strict=True)

            model_eval = local_models[0]
            model_eval.eval()
            with torch.no_grad():
                logits = model_eval(Abar_full.to(self.device), X)
                val_acc = self._accuracy(logits, y, val_mask)

            train_loss_hist.append(sum(local_losses) / len(local_losses))
            val_acc_hist.append(val_acc)

            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu() for k, v in model_eval.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if epochs_since_improve >= self.patience:
                break

        stats = {
            "train_loss": train_loss_hist,
            "val_acc": val_acc_hist,
            "epochs_trained": len(train_loss_hist)
        }

        return stats, best_state

    def _average_state_dicts(
        self,
        state_dicts: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Average parameters across multiple state dicts."""
        avg = {}
        for key in state_dicts[0].keys():
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            avg[key] = stacked.mean(dim=0)
        return avg

    def _accuracy(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor
    ) -> float:
        """Compute accuracy for masked nodes."""
        pred = logits.argmax(dim=1)
        return float((pred[mask] == y[mask]).float().mean().item())

    def _get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions and labels for test set."""
        data = self.graph_data
        test_mask = data.test_mask.to(self.device)
        y = data.y.to(self.device)

        self.defense_model.eval()
        with torch.no_grad():
            logits = self.defense_model(self.Abar.to(self.device), self.X_noised)
            preds = logits.argmax(dim=1)[test_mask]
            labels = y[test_mask]

        return preds.cpu(), labels.cpu()

    def _load_model(self):
        """Load pre-trained model (not implemented for MDP)."""
        pass

    def get_defended_features(self) -> torch.Tensor:
        """Return the DP-noised features."""
        return self.X_noised

    def get_adjacency_shares(self) -> List[torch.Tensor]:
        """Return the eigenvalue-split adjacency shares."""
        return self.Abar_shares

    def get_training_stats(self) -> Optional[Dict]:
        """Return training statistics."""
        return self._training_stats
