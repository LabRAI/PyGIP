from __future__ import annotations

from pygip.models.attack.base import BaseAttack
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelMode, ModelTaskLevel

# 🔧 If your grep showed BaseAttack in a different path, change this import:
from pygip.models.attack.base import BaseAttack

from .eg_models import (
    CAM,
    GAT,
    GCN,
    GIN,
    GraphSAGE,
    Classifier,
    GradientExplainer,
    GradCAM,
    SurrogateModel,
    TargetModel,
)
from .eg_utils import DataAugmentor, PGExplainer, RankNetLoss, safe_auc, set_seed


def custom_collate(batch: List[Data]) -> Batch:
    return Batch.from_data_list(batch)


def process_query_dataset(query_dataset: List[dict]) -> List[Data]:
    processed = []
    for sample in query_dataset:
        original_data = sample["original_data"]
        pred = sample["pred"]
        node_mask = sample["node_mask"]

        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        elif isinstance(pred, (list, np.ndarray)):
            pred = pred[0]

        new_data = Data(
            x=original_data.x,
            edge_index=original_data.edge_index,
            edge_attr=getattr(original_data, "edge_attr", None),
            y=original_data.y,
            target_pred=torch.tensor(pred, dtype=torch.long),
            node_mask=node_mask,
        )
        processed.append(new_data)
    return processed


def convert_edge_scores_to_node_scores(edge_mask: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    node_scores = torch.zeros(num_nodes, device=edge_mask.device)
    node_degrees = torch.zeros(num_nodes, device=edge_mask.device)

    for i in range(edge_index.shape[1]):
        n1, n2 = edge_index[:, i]
        imp = edge_mask[i]
        node_scores[n1] += imp
        node_scores[n2] += imp
        node_degrees[n1] += 1
        node_degrees[n2] += 1

    node_degrees[node_degrees == 0] = 1
    return node_scores / node_degrees


@dataclass
class EGStealConfig:
    seed: int = 43
    gnn_backbone: str = "GIN"          # GIN/GCN/GAT/GraphSAGE
    gnn_layers: int = 3
    hidden_dim: int = 128
    gat_heads: int = 4

    explanation_mode: str = "CAM"      # CAM/GradCAM/Grad/GNNExplainer/PGExplainer
    gnnexplainer_epochs: int = 100
    pgexplainer_epochs: int = 30

    epochs: int = 50                  # start smaller; set to 200 for full run
    lr: float = 1e-3
    batch_size: int = 64

    # query/training split ratios (match your data_preparation.py defaults) :contentReference[oaicite:2]{index=2}
    target_ratio: float = 0.4
    target_val_ratio: float = 0.2
    test_ratio: float = 0.2
    shadow_ratio: float = 0.4

    # surrogate alignment + augmentation (match your surrogate script defaults) :contentReference[oaicite:3]{index=3}
    align_weight: float = 1.0
    augmentation_ratio: float = 0.2
    operation_ratio: float = 0.05
    augmentation_type: str = "combined"   # drop_node/drop_edge/add_edge/combined


class EGStealAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets = set()  # leave empty unless you want to restrict

    def __init__(
        self,
        dataset,
        attack_node_fraction: float = None,
        model_path: str = None,
        device: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(dataset, attack_node_fraction=attack_node_fraction, model_path=model_path, device=device)
        self.cfg = EGStealConfig(**(config or {}))
        set_seed(self.cfg.seed)

    # -----------------------
    # Public API
    # -----------------------
    def attack(self) -> Dict[str, float]:
        device = self.device

        full_dataset = self._get_graph_dataset_list()
        target_train, target_val, test_ds, shadow_ds = self._split_dataset(full_dataset)

        target_model = self._train_target_model(target_train, target_val)
        query_dataset_shadow = self._query_target_model(target_model, shadow_ds)
        query_dataset_test = self._query_target_model(target_model, test_ds)

        processed_shadow = process_query_dataset(query_dataset_shadow)
        processed_test = process_query_dataset(query_dataset_test)

        surrogate_model = self._train_attack_model(processed_shadow, processed_test)

        # Evaluate
        test_acc, test_auc, fidelity, rank_corr = self._evaluate_surrogate(surrogate_model, processed_test, target_model)

        return {
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "fidelity_score": float(fidelity),
            "rank_correlation": float(rank_corr),
        }

    # -----------------------
    # Dataset helpers
    # -----------------------
    def _get_graph_dataset_list(self):
        # PyGIP’s Dataset stores data differently depending on dataset type.
        # For TU-style graph classification, graph_dataset is usually a list-like dataset.
        if getattr(self.dataset, "graph_dataset", None) is not None:
            return self.dataset.graph_dataset
        if getattr(self.dataset, "graph_data", None) is not None:
            # If it's a single Data object, wrap it
            gd = self.dataset.graph_data
            return [gd] if isinstance(gd, Data) else gd
        raise ValueError("Could not find dataset.graph_dataset or dataset.graph_data")

    def _split_dataset(self, dataset):
        n = len(dataset)
        target_num = int(n * self.cfg.target_ratio)
        test_num = int(n * self.cfg.test_ratio)
        shadow_num = n - target_num - test_num

        target_ds, test_ds, shadow_ds = random_split(dataset, [target_num, test_num, shadow_num])

        target_train_num = int(target_num * (1 - self.cfg.target_val_ratio))
        target_val_num = target_num - target_train_num
        target_train, target_val = random_split(target_ds, [target_train_num, target_val_num])

        return target_train, target_val, test_ds, shadow_ds

    # -----------------------
    # Model builders
    # -----------------------
    def _build_encoder(self, input_dim: int):
        if self.cfg.gnn_backbone == "GIN":
            return GIN(input_dim=input_dim, hidden_dim=self.cfg.hidden_dim, num_layers=self.cfg.gnn_layers)
        if self.cfg.gnn_backbone == "GCN":
            return GCN(input_dim=input_dim, hidden_dim=self.cfg.hidden_dim, num_layers=self.cfg.gnn_layers)
        if self.cfg.gnn_backbone == "GAT":
            return GAT(input_dim=input_dim, hidden_dim=self.cfg.hidden_dim, num_layers=self.cfg.gnn_layers, heads=self.cfg.gat_heads)
        if self.cfg.gnn_backbone == "GraphSAGE":
            return GraphSAGE(input_dim=input_dim, hidden_dim=self.cfg.hidden_dim, num_layers=self.cfg.gnn_layers)
        raise ValueError(f"Unknown backbone {self.cfg.gnn_backbone}")

    def _build_classifier(self, num_classes: int):
        return Classifier(input_dim=self.cfg.hidden_dim, output_dim=num_classes)

    # -----------------------
    # Target model training
    # -----------------------
    def _train_target_model(self, train_ds, val_ds):
        device = self.device
        num_features = self.num_features
        num_classes = self.num_classes

        encoder = self._build_encoder(num_features).to(device)
        predictor = self._build_classifier(num_classes).to(device)
        model = TargetModel(encoder=encoder, predictor=predictor, explanation_mode=self.cfg.explanation_mode).to(device)

        opt = Adam(model.parameters(), lr=self.cfg.lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        best_auc = -math.inf
        best_state = None

        for _ in range(self.cfg.epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                if batch.x is None:
                    batch.x = torch.ones((batch.num_nodes, 1), device=device)

                opt.zero_grad()
                if self.cfg.explanation_mode in ["GNNExplainer", "PGExplainer"]:
                    out = model(batch.x, batch.edge_index, batch.batch)
                else:
                    _, out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                opt.step()

            # val AUC
            acc, auc = self._eval_classifier(model, val_loader)
            if not math.isnan(auc) and auc >= best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def _eval_classifier(self, model, loader):
        device = self.device
        model.eval()
        y_true, y_prob, y_pred = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if batch.x is None:
                    batch.x = torch.ones((batch.num_nodes, 1), device=device)

                if self.cfg.explanation_mode in ["GNNExplainer", "PGExplainer"]:
                    out = model(batch.x, batch.edge_index, batch.batch)
                else:
                    _, out = model(batch.x, batch.edge_index, batch.batch)

                probs = F.softmax(out, dim=1)
                pred = out.argmax(dim=1)

                y_true.extend(batch.y.detach().cpu().tolist())
                y_pred.extend(pred.detach().cpu().tolist())

                # binary AUC uses prob of class 1
                if probs.size(1) == 2:
                    y_prob.extend(probs[:, 1].detach().cpu().tolist())
                else:
                    # fallback: max prob (not ideal for true multiclass AUC)
                    y_prob.extend(probs.max(dim=1).values.detach().cpu().tolist())

        acc = float(np.mean(np.array(y_pred) == np.array(y_true))) if len(y_true) else 0.0
        auc = float(safe_auc(y_true, y_prob)) if len(y_true) else 0.5
        return acc, auc

    # -----------------------
    # Querying + explanations
    # -----------------------
    def _query_target_model(self, target_model, dataset_split) -> List[dict]:
        device = self.device
        target_model.eval()
        loader = DataLoader(dataset_split, batch_size=min(self.cfg.batch_size, 256), shuffle=False)

        results = []

        # setup explainers
        cam = CAM(target_model) if self.cfg.explanation_mode == "CAM" else None
        gradcam = GradCAM(target_model) if self.cfg.explanation_mode == "GradCAM" else None
        grad = GradientExplainer(target_model) if self.cfg.explanation_mode == "Grad" else None

        gnnexplainer = None
        pgexplainer = None
        if self.cfg.explanation_mode == "GNNExplainer":
            gnnexplainer = Explainer(
                model=target_model,
                algorithm=GNNExplainer(epochs=self.cfg.gnnexplainer_epochs),
                explanation_type="phenomenon",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=ModelConfig(
                    mode=ModelMode.multiclass_classification,
                    task_level=ModelTaskLevel.graph,
                    return_type="raw",
                ),
            )

        if self.cfg.explanation_mode == "PGExplainer":
            pgexplainer = Explainer(
                model=target_model,
                algorithm=PGExplainer(epochs=self.cfg.pgexplainer_epochs),
                explanation_type="phenomenon",
                edge_mask_type="object",
                model_config=ModelConfig(
                    mode=ModelMode.multiclass_classification,
                    task_level=ModelTaskLevel.graph,
                    return_type="raw",
                ),
            )

        for batch_data in loader:
            batch_data = batch_data.to(device)
            if batch_data.x is None:
                batch_data.x = torch.ones((batch_data.num_nodes, 1), device=device)

            batch = batch_data.batch

            with torch.no_grad():
                if self.cfg.explanation_mode in ["GNNExplainer", "PGExplainer"]:
                    out = target_model(batch_data.x, batch_data.edge_index, batch)
                else:
                    _, out = target_model(batch_data.x, batch_data.edge_index, batch)

            preds = out.argmax(dim=1)

            # get node_mask aligned with nodes in the batch
            if self.cfg.explanation_mode == "CAM":
                # CAM needs a forward pass WITH hooks active (already happened above)
                node_mask = cam.get_cam_scores(preds, batch)
            elif self.cfg.explanation_mode == "GradCAM":
                node_mask = gradcam.get_gradcam_scores(batch_data, preds)
            elif self.cfg.explanation_mode == "Grad":
                node_mask = grad.get_gradient_scores(batch_data, preds)
            elif self.cfg.explanation_mode == "GNNExplainer":
                exp = gnnexplainer(batch_data.x, batch_data.edge_index, batch=batch)
                node_mask = exp.node_mask.view(-1)
            elif self.cfg.explanation_mode == "PGExplainer":
                # train pgexplainer quickly on this batch
                for epoch in range(self.cfg.pgexplainer_epochs):
                    pgexplainer.algorithm = pgexplainer.algorithm.to(device)
                    _ = pgexplainer.algorithm.train(epoch, target_model, batch_data.x, batch_data.edge_index, target=preds, batch=batch)
                exp = pgexplainer(batch_data.x, batch_data.edge_index, target=preds, batch=batch)
                node_mask = convert_edge_scores_to_node_scores(exp.edge_mask, exp.edge_index, batch_data.x.size(0))
            else:
                raise ValueError(f"Unknown explanation_mode {self.cfg.explanation_mode}")

            # split per-graph like your script :contentReference[oaicite:4]{index=4}
            original_graphs = batch_data.to_data_list()
            num_nodes_per_graph = batch_data.ptr[1:] - batch_data.ptr[:-1]
            node_masks_list = torch.split(node_mask.detach().cpu(), num_nodes_per_graph.tolist())
            preds_list = preds.detach().cpu().tolist()

            for original_data, pred, nm in zip(original_graphs, preds_list, node_masks_list):
                results.append(
                    {
                        "original_data": original_data.to("cpu"),
                        "pred": pred,
                        "node_mask": nm.to("cpu"),
                    }
                )

        return results

    # -----------------------
    # Surrogate training (attack model)
    # -----------------------
    def _train_attack_model(self, processed_shadow: List[Data], processed_test: List[Data]):
        device = self.device
        num_features = self.num_features
        num_classes = self.num_classes

        encoder = self._build_encoder(num_features).to(device)
        predictor = self._build_classifier(num_classes).to(device)
        model = SurrogateModel(encoder=encoder, predictor=predictor).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        ranknet = RankNetLoss().to(device)
        opt = Adam(model.parameters(), lr=self.cfg.lr)

        augmentor = DataAugmentor()

        best_fidelity = -math.inf
        best_state = None

        for _ in range(self.cfg.epochs):
            # augmentation like your script :contentReference[oaicite:5]{index=5}
            augmented = []
            if self.cfg.augmentation_ratio > 0:
                num_aug = int(len(processed_shadow) * self.cfg.augmentation_ratio)
                for i in np.random.choice(len(processed_shadow), size=num_aug, replace=False) if num_aug > 0 else []:
                    s = processed_shadow[int(i)]
                    if self.cfg.augmentation_type == "drop_node":
                        a = augmentor.drop_node(s, drop_ratio=self.cfg.operation_ratio)
                    elif self.cfg.augmentation_type == "drop_edge":
                        a = augmentor.drop_edge(s, drop_ratio=self.cfg.operation_ratio)
                    elif self.cfg.augmentation_type == "add_edge":
                        a = augmentor.add_edge(s, add_ratio=self.cfg.operation_ratio)
                    else:
                        a = augmentor.combined_augmentation(
                            s,
                            drop_node_ratio=self.cfg.operation_ratio,
                            drop_edge_ratio=self.cfg.operation_ratio,
                            add_edge_ratio=self.cfg.operation_ratio,
                        )
                    if a is not None:
                        augmented.append(a)

            train_data = processed_shadow + augmented
            train_loader = DataLoader(train_data, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=custom_collate)
            test_loader = DataLoader(processed_test, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=custom_collate)

            model.train()
            cam_surr = CAM(model)

            for batch in train_loader:
                batch = batch.to(device)
                if batch.x is None:
                    batch.x = torch.ones((batch.num_nodes, 1), device=device)

                opt.zero_grad()
                node_emb, out = model(batch.x, batch.edge_index, batch.batch)

                # prediction loss against target_pred
                loss_pred = criterion(out, batch.target_pred)

                # ranking alignment loss between surrogate CAM and true node_mask
                preds = out.argmax(dim=1)
                _ = out  # forward already done; cam_surr hooks are populated
                cam_scores = cam_surr.get_cam_scores(preds, batch.batch)

                true_mask = batch.node_mask.view(-1)
                align = ranknet(cam_scores.view(-1), true_mask.to(device), batch.batch)

                loss = loss_pred + self.cfg.align_weight * align
                loss.backward()
                opt.step()

            # track best by fidelity on test
            fidelity = self._fidelity(model, test_loader)
            if fidelity >= best_fidelity:
                best_fidelity = fidelity
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # -----------------------
    # Metrics
    # -----------------------
    def _fidelity(self, surrogate, loader):
        device = self.device
        surrogate.eval()
        hits = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if batch.x is None:
                    batch.x = torch.ones((batch.num_nodes, 1), device=device)
                _, out = surrogate(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                hits += int((pred == batch.target_pred).sum().item())
                total += int(batch.target_pred.size(0))
        return hits / total if total else 0.0

    def _evaluate_surrogate(self, surrogate, processed_test, target_model):
        device = self.device
        loader = DataLoader(processed_test, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=custom_collate)

        test_acc, test_auc = self._eval_classifier_like_surrogate(surrogate, loader)
        fidelity = self._fidelity(surrogate, loader)

        # rank correlation: per-graph kendall tau averaged (approximate, matches intent)
        try:
            from scipy.stats import kendalltau
        except Exception:
            kendalltau = None

        surrogate.eval()
        cam_surr = CAM(surrogate)

        taus = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if batch.x is None:
                    batch.x = torch.ones((batch.num_nodes, 1), device=device)

                _, out = surrogate(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(dim=1)
                cam_scores = cam_surr.get_cam_scores(preds, batch.batch).detach().cpu()
                true_scores = batch.node_mask.view(-1).detach().cpu()
                batch_ids = batch.batch.detach().cpu()

                if kendalltau is None:
                    continue

                for gid in torch.unique(batch_ids):
                    m = batch_ids == gid
                    p = cam_scores[m].numpy()
                    t = true_scores[m].numpy()
                    if len(p) < 2:
                        continue
                    tau = kendalltau(p, t).correlation
                    if tau is not None and not np.isnan(tau):
                        taus.append(float(tau))

        rank_corr = float(np.mean(taus)) if taus else 0.0
        return test_acc, test_auc, fidelity, rank_corr

    def _eval_classifier_like_surrogate(self, surrogate, loader):
        device = self.device
        surrogate.eval()
        y_true, y_prob, y_pred = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if batch.x is None:
                    batch.x = torch.ones((batch.num_nodes, 1), device=device)

                _, out = surrogate(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(out, dim=1)
                pred = out.argmax(dim=1)

                # Use original y for "accuracy/auc" like your surrogate script output
                y_true.extend(batch.y.detach().cpu().tolist())
                y_pred.extend(pred.detach().cpu().tolist())
                if probs.size(1) == 2:
                    y_prob.extend(probs[:, 1].detach().cpu().tolist())
                else:
                    y_prob.extend(probs.max(dim=1).values.detach().cpu().tolist())

        acc = float(np.mean(np.array(y_pred) == np.array(y_true))) if len(y_true) else 0.0
        auc = float(safe_auc(y_true, y_prob)) if len(y_true) else 0.5
        return acc, auc

