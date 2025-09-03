
import os
import torch
from typing import Optional
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from core.base import Dataset

class ProteinsDataset(Dataset):
    def __init__(self, api_type='pyg', path='./data', seed: int = 7):
        super().__init__(api_type, path)
        assert self.api_type == 'pyg', 'This implementation uses PyG.'
        self.dataset_name = 'PROTEINS'
        self.graph_dataset = TUDataset(root=os.path.join(path, 'PROTEINS'), name='PROTEINS')
        # satisfy api_type='pyg' requirement: a Data object lives at graph_data
        self.graph_data: Optional[Data] = self.graph_dataset[0]
        # meta
        self.num_features = int(self.graph_dataset.num_features)
        self.num_classes = int(self.graph_dataset.num_classes)
        self.num_nodes = int(max(g.num_nodes for g in self.graph_dataset))
        # split indices 70/10/20 over graphs
        torch.manual_seed(seed)
        n = len(self.graph_dataset)
        idx = torch.randperm(n)
        n_tr = int(0.7 * n)
        n_va = int(0.1 * n)
        self.train_data = [self.graph_dataset[i] for i in idx[:n_tr]]
        self.val_data   = [self.graph_dataset[i] for i in idx[n_tr:n_tr+n_va]]
        self.test_data  = [self.graph_dataset[i] for i in idx[n_tr+n_va:]]
    def get_name(self):
        return 'PROTEINS'
