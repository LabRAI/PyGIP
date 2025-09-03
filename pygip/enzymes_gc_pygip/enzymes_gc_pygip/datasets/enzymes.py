
import os, torch
from torch_geometric.datasets import TUDataset
from ..core.base import Dataset

class EnzymesDataset(Dataset):
    def __init__(self, api_type='pyg', path='./data', seed=7):
        super().__init__(api_type, path)
        assert self.api_type == 'pyg', 'This repo uses PyG.'
        self.dataset_name = 'ENZYMES'
        self.graph_dataset = TUDataset(root=os.path.join(path, 'ENZYMES'), name='ENZYMES')
        self.graph_data = self.graph_dataset[0]
        self.num_features = int(self.graph_dataset.num_features)
        self.num_classes = int(self.graph_dataset.num_classes)
        self.num_nodes = int(max(g.num_nodes for g in self.graph_dataset))
        torch.manual_seed(seed)
        n = len(self.graph_dataset)
        idx = torch.randperm(n)
        n_tr = int(n * 0.7)
        n_va = int(n * 0.1)
        train_idx = idx[:n_tr]
        val_idx = idx[n_tr:n_tr+n_va]
        test_idx = idx[n_tr+n_va:]
        self.train_data = [self.graph_dataset[i] for i in train_idx]
        self.val_data   = [self.graph_dataset[i] for i in val_idx]
        self.test_data  = [self.graph_dataset[i] for i in test_idx]
    def get_name(self):
        return 'ENZYMES'
