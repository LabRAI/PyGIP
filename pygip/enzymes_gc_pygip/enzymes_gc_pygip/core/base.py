
from typing import Optional, Union
import torch
from torch import nn
from torch_geometric.data import Data

class Dataset(object):
    def __init__(self, api_type='pyg', path='./data'):
        assert api_type in {'dgl', 'pyg'}, 'API type must be dgl or pyg'
        self.api_type = api_type
        self.path = path
        self.dataset_name = self.get_name()
        self.graph_dataset = None
        self.graph_data: Optional[Data] = None
        self.num_nodes = 0
        self.num_features = 0
        self.num_classes = 0
    def get_name(self):
        raise NotImplementedError

class BaseAttack(nn.Module):
    supported_api_types = set()
    supported_datasets = set()
    def __init__(self, dataset: Dataset, attack_node_fraction: float = None,
                 model_path: Optional[str] = None, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
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

class BaseDefense(nn.Module):
    supported_api_types = set()
    supported_datasets = set()
    def __init__(self, dataset: Dataset, attack_node_fraction: float,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
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
    def defend(self):
        raise NotImplementedError
    def _load_model(self):
        raise NotImplementedError
    def _train_target_model(self):
        raise NotImplementedError
    def _train_defense_model(self):
        raise NotImplementedError
    def _train_surrogate_model(self):
        raise NotImplementedError
