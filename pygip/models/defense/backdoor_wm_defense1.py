import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Dict, Any
import copy
import torch_geometric
import dgl

from pygip.models.defense.base import BaseDefense
from pygip.datasets.datasets import Dataset
from pygip.utils.hardware import get_device
from pygip.models.nn import GCN

class BackdoorWatermarkDefense(BaseDefense):
    """
    Implementation of a backdoor-based watermarking defense for GNNs.
    """
    supported_api_types = {"pyg"}
    supported_datasets = {"Cora", "CiteSeer", "PubMed"}

    def __init__(self,
                 dataset: Dataset,
                 device: Optional[Union[str, torch.device]] = None,
                 model_name: str = 'GCN',
                 lr: float = 0.01,
                 epochs: int = 200,
                 backdoor_node_fraction: float = 0.15,
                 backdoor_feature_manipulation_fraction: float = 0.35,
                 verification_thresholds: Dict[str, float] = None):

        super().__init__(dataset, attack_node_fraction=backdoor_node_fraction, device=device)

        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.backdoor_node_fraction = backdoor_node_fraction
        self.backdoor_feature_manipulation_fraction = backdoor_feature_manipulation_fraction

        if verification_thresholds is None:
            self.verification_thresholds = {'cora': 0.50, 'citeseer': 0.53, 'pubmed': 0.50}

        else:
            self.verification_thresholds = verification_thresholds

        self.original_data = copy.deepcopy(self.graph_data)

    def _inject_backdoors(self, data: 'torch_geometric.data.Data', node_mask: torch.Tensor) -> tuple:
        watermarked_data = data.clone()
        target_nodes = node_mask.nonzero(as_tuple=False).view(-1)

        if len(target_nodes) == 0:
            return watermarked_data, torch.tensor([], dtype=torch.long, device=self.device), -1

        num_backdoor_nodes = int(len(target_nodes) * self.backdoor_node_fraction)
        backdoor_node_indices = target_nodes[torch.randperm(len(target_nodes))[:num_backdoor_nodes]]
        labels = watermarked_data.y[watermarked_data.train_mask]
        label_counts = torch.bincount(labels)
        backdoor_target_label = torch.argmin(label_counts).item()
        num_features_to_manipulate = int(self.num_features * self.backdoor_feature_manipulation_fraction)

        for node_idx in backdoor_node_indices:
            feature_indices_to_flip = torch.randperm(self.num_features)[:num_features_to_manipulate]
            watermarked_data.x[node_idx, feature_indices_to_flip] = 1.0
            watermarked_data.y[node_idx] = backdoor_target_label

        print(f"Injected backdoors into {len(backdoor_node_indices)} nodes. Target label: {backdoor_target_label}")

        return watermarked_data, backdoor_node_indices, backdoor_target_label

    def _train_model(self, model, data, mask, epochs, lr):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()

        dgl_graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes).to(self.device)
        dgl_graph = dgl.add_self_loop(dgl_graph)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(dgl_graph, data.x)
            loss = F.cross_entropy(out[mask], data.y[mask])
            loss.backward()
            optimizer.step()

        return model

    def _evaluate_model(self, model, data, test_mask, backdoor_mask, backdoor_label, target_model_for_fidelity=None):
        model.eval()
        with torch.no_grad():
            dgl_graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes).to(self.device)

            dgl_graph = dgl.add_self_loop(dgl_graph)
            out = model(dgl_graph, data.x)

            if target_model_for_fidelity:
                target_model_for_fidelity.eval()
                target_preds = target_model_for_fidelity(dgl_graph, data.x).argmax(dim=1)

            preds = out.argmax(dim=1)
            clean_mask = test_mask & ~backdoor_mask
            clean_nodes_count = int(clean_mask.sum())
            clean_acc = int((preds[clean_mask] == data.y[clean_mask]).sum()) / clean_nodes_count if clean_nodes_count > 0 else 0.0
            backdoor_test_nodes_mask = test_mask & backdoor_mask
            backdoor_nodes_count = int(backdoor_test_nodes_mask.sum())
            backdoor_acc = int((preds[backdoor_test_nodes_mask] == backdoor_label).sum()) / backdoor_nodes_count if backdoor_nodes_count > 0 else 0.0
            fidelity = 0.0
            test_nodes_count = int(test_mask.sum())

            if target_model_for_fidelity and test_nodes_count > 0:
                fidelity = int((preds[test_mask] == target_preds[test_mask]).sum()) / test_nodes_count

        return {'clean_accuracy': clean_acc, 'backdoor_accuracy': backdoor_acc, 'fidelity': fidelity}

    def defend(self) -> Dict[str, Any]:
        print("--- Starting Backdoor Watermarking Defense ---")

        num_nodes = self.num_nodes
        indices = torch.randperm(num_nodes)
        train_end, surr_end = int(0.2 * num_nodes), int(0.6 * num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[indices[:train_end]] = True
        surr_mask = torch.zeros(num_nodes, dtype=torch.bool); surr_mask[indices[train_end:surr_end]] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool); test_mask[indices[surr_end:]] = True
        self.graph_data.train_mask, self.graph_data.surr_mask, self.graph_data.test_mask = train_mask, surr_mask, test_mask
        data_for_wm = self.graph_data.to(self.device)
        wm_data, train_backdoor_indices, backdoor_label = self._inject_backdoors(data_for_wm, data_for_wm.train_mask)
        wm_data, test_backdoor_indices, _ = self._inject_backdoors(wm_data, data_for_wm.test_mask)
        all_backdoor_indices = torch.cat([train_backdoor_indices.cpu(), test_backdoor_indices.cpu()]).unique()
        backdoor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        backdoor_mask[all_backdoor_indices] = True

        print("\n--- Training Watermarked Target Model (F_wm) ---")
        target_model = GCN(self.num_features, self.num_classes).to(self.device)
        self._train_model(target_model, wm_data, wm_data.train_mask, self.epochs, self.lr)
        target_metrics = self._evaluate_model(target_model, wm_data, wm_data.test_mask, backdoor_mask, backdoor_label)
        TCA, TBA = target_metrics['clean_accuracy'], target_metrics['backdoor_accuracy']

        print("\n--- Scenario 1: Simulating attack WITH attacker cooperation (uses watermarked graph) ---")

        surrogate_model_1 = GCN(self.num_features, self.num_classes).to(self.device)

        with torch.no_grad():
            dgl_graph_wm = dgl.graph((wm_data.edge_index[0], wm_data.edge_index[1]), num_nodes=wm_data.num_nodes).to(self.device)
            dgl_graph_wm = dgl.add_self_loop(dgl_graph_wm)
            surrogate_labels = target_model(dgl_graph_wm, wm_data.x).argmax(dim=1)

        data_for_surr_1 = wm_data.clone()
        data_for_surr_1.y = surrogate_labels
        self._train_model(surrogate_model_1, data_for_surr_1, data_for_surr_1.surr_mask, self.epochs, self.lr)
        surrogate_metrics_1 = self._evaluate_model(surrogate_model_1, wm_data, wm_data.test_mask, backdoor_mask, backdoor_label, target_model)
        ECA_1, EBA_1, Fidelity_1 = surrogate_metrics_1['clean_accuracy'], surrogate_metrics_1['backdoor_accuracy'], surrogate_metrics_1['fidelity']

        print("\n--- Scenario 2: Simulating attack WITHOUT attacker cooperation (uses original graph) ---")

        surrogate_model_2 = GCN(self.num_features, self.num_classes).to(self.device)
        original_data_device = self.original_data.to(self.device)

        with torch.no_grad():
            dgl_graph_orig = dgl.graph((original_data_device.edge_index[0], original_data_device.edge_index[1]), num_nodes=original_data_device.num_nodes).to(self.device)            
            dgl_graph_orig = dgl.add_self_loop(dgl_graph_orig)            
            surrogate_labels_2 = target_model(dgl_graph_orig, original_data_device.x).argmax(dim=1)
        
        data_for_surr_2 = original_data_device.clone()
        data_for_surr_2.y = surrogate_labels_2
        data_for_surr_2.surr_mask = data_for_wm.surr_mask
        
        self._train_model(surrogate_model_2, data_for_surr_2, data_for_surr_2.surr_mask, self.epochs, self.lr)
        
        surrogate_metrics_2 = self._evaluate_model(surrogate_model_2, wm_data, wm_data.test_mask, 
        
        backdoor_mask, backdoor_label, target_model)
        
        ECA_2, EBA_2, Fidelity_2 = surrogate_metrics_2['clean_accuracy'], surrogate_metrics_2['backdoor_accuracy'], surrogate_metrics_2['fidelity']

        threshold = self.verification_thresholds.get(self.dataset.dataset_name.lower(), 0.5)
        verification_1 = "Extracted" if EBA_1 > threshold else "Independent"
        verification_2 = "Extracted" if EBA_2 > threshold else "Independent"
        
        return {
            "target_model_metrics": {"TCA": TCA, "TBA": TBA},
            "scenario_with_cooperation": {"ECA": ECA_1, "EBA": EBA_1, "Fidelity": Fidelity_1, "verification_result": verification_1},
            "scenario_without_cooperation": {"ECA": ECA_2, "EBA": EBA_2, "Fidelity": Fidelity_2, "verification_result": verification_2},
            "verification_threshold": threshold
        }