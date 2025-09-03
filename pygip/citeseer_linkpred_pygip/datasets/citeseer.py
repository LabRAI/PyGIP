# datasets/citeseer.py
# PyGIP-style Dataset wrapper for Citeseer (Link Prediction)

import os
from typing import Optional
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import RandomLinkSplit

SEED = 42

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

class CiteseerDataset(Dataset):
    def __init__(self, api_type='pyg', path='./data', seed=SEED):
        super().__init__(api_type, path)
        assert self.api_type=='pyg', 'This implementation uses PyG.'
        self.dataset_name = 'Citeseer'
        # Load dataset (single graph)
        self.graph_dataset = Planetoid(root=os.path.join(path, 'Citeseer'), name='Citeseer')
        self.graph_data = self.graph_dataset[0]
        # Meta
        self.num_nodes = int(self.graph_data.num_nodes)
        self.num_features = int(self.graph_dataset.num_features)
        # Prepare link prediction splits
        torch.manual_seed(seed)
        data = self.graph_data
        data.edge_index = to_undirected(data.edge_index)
        transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, add_negative_train_samples=True)
        self.train_data, self.val_data, self.test_data = transform(data)

    def get_name(self):
        return 'Citeseer'
