# datasets/cora.py
# PyGIP-style Dataset wrapper for Cora (Link Prediction)

import os
from typing import Optional
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

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
        self.graph_dataset = Planetoid(root=os.path.join(path, 'Cora'), name='Cora')
        self.graph_data = self.graph_dataset[0]
        # meta
        self.num_nodes = self.graph_data.num_nodes
        self.num_features = self.graph_data.num_features
        # self.num_classes = self.graph_data.num_classes  # not needed for LP

        # Create train/val/test edge masks for link prediction
        torch.manual_seed(seed)
        edge_index = self.graph_data.edge_index
        num_edges = edge_index.size(1)

        # 70/10/20 split for edges
        perm = torch.randperm(num_edges)
        n_train = int(num_edges * 0.7)
        n_val = int(num_edges * 0.1)

        train_edges = perm[:n_train]
        val_edges = perm[n_train:n_train+n_val]
        test_edges = perm[n_train+n_val:]

        # Create masks
        self.train_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.val_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.test_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.train_mask[train_edges] = True
        self.val_mask[val_edges] = True
        self.test_mask[test_edges] = True

        # Store edge masks in graph_data
        self.graph_data.train_mask = self.train_mask
        self.graph_data.val_mask = self.val_mask
        self.graph_data.test_mask = self.test_mask

        # Also create negative edges for training
        self._create_negative_edges()

    def _create_negative_edges(self):
        # Generate negative edges (non-existent edges)
        num_neg_edges = self.graph_data.edge_index.size(1)
        adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        adj[self.graph_data.edge_index[0], self.graph_data.edge_index[1]] = True
        adj[self.graph_data.edge_index[1], self.graph_data.edge_index[0]] = True

        neg_edges = []
        while len(neg_edges) < num_neg_edges:
            i, j = torch.randint(0, self.num_nodes, (2,))
            if not adj[i, j] and i != j:
                neg_edges.append([i, j])
                adj[i, j] = True
                adj[j, i] = True

        self.neg_edge_index = torch.stack([torch.tensor(edge) for edge in neg_edges], dim=1)

        # Split negative edges
        n_train = int(num_neg_edges * 0.7)
        n_val = int(num_neg_edges * 0.1)

        self.neg_train_mask = torch.zeros(num_neg_edges, dtype=torch.bool)
        self.neg_val_mask = torch.zeros(num_neg_edges, dtype=torch.bool)
        self.neg_test_mask = torch.zeros(num_neg_edges, dtype=torch.bool)

        perm = torch.randperm(num_neg_edges)
        self.neg_train_mask[perm[:n_train]] = True
        self.neg_val_mask[perm[n_train:n_train+n_val]] = True
        self.neg_test_mask[perm[n_train+n_val:]] = True

        # Store in graph_data
        self.graph_data.neg_edge_index = self.neg_edge_index
        self.graph_data.neg_train_mask = self.neg_train_mask
        self.graph_data.neg_val_mask = self.neg_val_mask
        self.graph_data.neg_test_mask = self.neg_test_mask

    def get_name(self):
        return 'Cora'
