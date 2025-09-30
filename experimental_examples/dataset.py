#Dataset class - load dataset

import os, random
from typing import Optional

import torch
from torch_geometric.datasets import Planetoid

class Dataset(object):
    def __init__(self, api_type: str ='pyg', path: str = './data', name: str = 'Cora'):
        assert api_type in {'dgl', 'pyg'}, 'API type must be dgl or pyg'
        self.api_type= api_type
        self.path = path
        self.dataset_name =name

        self.graph_dataset=None
        self.graph_data = None

        self.num_nodes = 0
        self.num_features= 0
        self.num_classes = 0

        self._load()

    def _load(self):
        if self.api_type == 'pyg':
            ds = Planetoid(root=os.path.join(self.path,'Planetoid'), name=self.dataset_name)
            self.graph_dataset = ds
            self.graph_data = ds[0]
            self.num_nodes = ds[0].num_nodes
            self.num_features = ds.num_node_features
            self.num_classes = ds.num_classes
        else:
            raise NotImplementedError('dgl api not implemented')

    def to(self, device: Optional[torch.device] = None):
        dev =device if device is not None else(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.graph_data= self.graph_data.to(dev)
        return self