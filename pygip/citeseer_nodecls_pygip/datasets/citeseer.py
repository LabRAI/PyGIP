import os
import torch
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

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

class CiteseerDataset(Dataset):
    def __init__(self, api_type='pyg', path='./data', seed=7):
        super().__init__(api_type, path)
        assert self.api_type == 'pyg', 'This repro uses PyG.'
        self.dataset_name = 'Citeseer'
        self.graph_dataset = Planetoid(root=os.path.join(path, 'Citeseer'), name='Citeseer',
                                       transform=NormalizeFeatures())
        self.graph_data = self.graph_dataset[0]
        self.num_nodes = int(self.graph_data.num_nodes)
        self.num_features = int(self.graph_dataset.num_features)
        self.num_classes = int(self.graph_dataset.num_classes)  # 6 for Citeseer
        # Random 70/10/20 node split
        torch.manual_seed(seed)
        n = self.num_nodes
        idx = torch.randperm(n)
        n_tr = int(n * 0.7); n_va = int(n * 0.1)
        train_idx = idx[:n_tr]; val_idx = idx[n_tr:n_tr+n_va]; test_idx = idx[n_tr+n_va:]
        self.train_mask = torch.zeros(n, dtype=torch.bool)
        self.val_mask   = torch.zeros(n, dtype=torch.bool)
        self.test_mask  = torch.zeros(n, dtype=torch.bool)
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
        self.graph_data.train_mask = self.train_mask
        self.graph_data.val_mask = self.val_mask
        self.graph_data.test_mask = self.test_mask

    def get_name(self):
        return 'Citeseer'
