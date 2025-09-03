
from typing import Optional, Union
import torch, torch.nn as nn
from ..utils import get_device

class BaseAttack(nn.Module):
    supported_api_types = set()
    supported_datasets = set()
    def __init__(self, dataset, attack_node_fraction: float = None,
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

class BaseDefense(nn.Module):
    supported_api_types = set()
    supported_datasets = set()
    def __init__(self, dataset, attack_node_fraction: float,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.device = torch.device(device) if device else get_device()
        print(f"[Defense] Using device: {self.device}")
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.attack_node_fraction = attack_node_fraction
        self._check_dataset_compatibility()
    def _check_dataset_compatibility(self):
        if self.supported_datasets and self.dataset.dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset {self.dataset.dataset_name} not supported")
    def defend(self): raise NotImplementedError
    def _load_model(self): raise NotImplementedError
    def _train_target_model(self): raise NotImplementedError
    def _train_defense_model(self): raise NotImplementedError
    def _train_surrogate_model(self): raise NotImplementedError
