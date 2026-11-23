#Base attack class - Base attack as per the implementation guide
import torch
from typing import Optional

class BaseAttack(object):
    supported_api_types = set()
    supported_datasets = set()

    def __init__(self, dataset, attack_node_fraction: float = None,
                 model_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"[BaseAttack] Using device: {self.device}")

        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data.to(self.device)

        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.attack_node_fraction = attack_node_fraction
        self.model_path =model_path

        self._check_dataset_compatibility()

    def _check_dataset_compatibility(self):
        if self.supported_api_types and (self.dataset.api_type not in self.supported_api_types):
            raise RuntimeError('Dataset api type not supported')
        if self.supported_datasets and (self.dataset.dataset_name not in self.supported_datasets):
            print('[BaseAttack] Warning: dataset name not listed')

    def attack(self):
        raise NotImplementedError