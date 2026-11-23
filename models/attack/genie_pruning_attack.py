"""
pygip/models/attack/genie_pruning_attack.py

Global unstructured pruning attack compatible with PyGIP BaseAttack.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn.utils.prune as prune
from sklearn.metrics import roc_auc_score
from pygip.models.attack import BaseAttack
from pygip.data.dataset import Dataset
from pygip.models.nn.backbones import GCNLinkPredictor
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import negative_sampling


class GeniePruningAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets = set()

    def __init__(self, dataset: Dataset, attack_node_fraction: float = 0.1, model_path: Optional[str] = None,
                 prune_ratio: float = 0.2, save_pruned: bool = False):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.prune_ratio = prune_ratio
        self.save_pruned = save_pruned

    def attack(self) -> Dict[str, Any]:
        device = self.device
        data = self.graph_data.to(device)
        model = self._load_model()
        if model is None:
            raise RuntimeError("Could not load model for pruning attack")
        model.to(device)

        params_to_prune = [(module.lin, "weight")
                           for _, module in model.named_modules()
                           if isinstance(module, pyg_nn.GCNConv) and hasattr(module, "lin")]

        if not params_to_prune:
            raise RuntimeError("No GCNConv linear layers found to prune.")

        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=self.prune_ratio)

        test_auc, wm_auc = self._evaluate_model(model, data)

        results = {
            "dataset": getattr(self.dataset, "dataset_name", "unknown"),
            "prune_ratio": self.prune_ratio,
            "test_auc": float(test_auc),
            "watermark_auc": float(wm_auc) if wm_auc is not None else None
        }

        if self.save_pruned and self.model_path:
            out_path = self.model_path.replace(".pth", f"_pruned_{int(self.prune_ratio*100)}.pth")
            torch.save(model.state_dict(), out_path)
            results["pruned_model_path"] = out_path

        return results

    def _load_model(self):
        if not self.model_path:
            print("[GeniePruningAttack] No model path provided.")
            return None
        ckpt = torch.load(self.model_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            in_ch = getattr(self.dataset, "num_features", 64)
            model = GCNLinkPredictor(in_channels=in_ch, hidden_channels=64).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            print("[GeniePruningAttack] Failed to load model:", e)
            return None

    def _evaluate_model(self, model, data):
        model.eval()
        device = self.device
        train_pos = getattr(data, "train_pos_edge_index", None)
        test_pos = getattr(data, "test_pos_edge_index", None)
        if train_pos is None or test_pos is None:
            test_pos = data.edge_index

        z = model.encode(data.x.to(device), getattr(data, "train_pos_edge_index", data.edge_index).to(device))
        pos_logits = model.decode(z, test_pos.to(device)).view(-1).cpu().detach()
        neg = negative_sampling(edge_index=data.edge_index.to(device), num_nodes=data.num_nodes,
                                num_neg_samples=pos_logits.size(0)).to(device)
        neg_logits = model.decode(z, neg).view(-1).cpu().detach()

        import numpy as np
        y_true = np.concatenate([np.ones(pos_logits.size(0)), np.zeros(neg_logits.size(0))])
        y_pred = np.concatenate([pos_logits.numpy(), neg_logits.numpy()])
        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = float("nan")

        wm_auc = None
        if hasattr(self.dataset, "watermark_edges") and hasattr(self.dataset, "watermark_labels"):
            with torch.no_grad():
                z_wm = model.encode(data.x.to(device), getattr(data, "train_pos_edge_index", data.edge_index).to(device))
                wm_preds = model.decode(z_wm, self.dataset.watermark_edges.to(device)).view(-1).cpu().numpy()
                try:
                    wm_auc = roc_auc_score(self.dataset.watermark_labels.cpu().numpy(), wm_preds)
                except Exception:
                    wm_auc = float("nan")
        return auc, wm_auc
