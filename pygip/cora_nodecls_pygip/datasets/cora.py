# datasets/cora.py
# PyGIP-style Dataset wrapper for Cora (Node Classification)

import os
from typing import Optional
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

SEED = 7

class Dataset(object):
    def __init__(self, api_type='pyg', path='./data'):
        assert api_type in {'dgl','pyg'}, 'API type must be dgl or pyg'
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

class CoraDataset(Dataset):
    def __init__(self, api_type='pyg', path='./data', seed=SEED):
        super().__init__(api_type, path)
        assert self.api_type=='pyg', 'This repo uses PyG.'
        self.dataset_name = 'Cora'
        # Load Planetoid dataset (Cora is a single graph)
        self.graph_dataset = Planetoid(root=os.path.join(path, 'Cora'), name='Cora', transform=NormalizeFeatures())
        self.graph_data = self.graph_dataset[0]
        # meta
        self.num_nodes = int(self.graph_data.num_nodes)
        self.num_features = int(self.graph_dataset.num_features)
        self.num_classes = int(self.graph_dataset.num_classes)

        # Expose default masks
        self.train_mask = self.graph_data.train_mask
        self.val_mask   = self.graph_data.val_mask
        self.test_mask  = self.graph_data.test_mask

    def get_name(self):
        return 'Cora'
