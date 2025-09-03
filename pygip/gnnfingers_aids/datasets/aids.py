
from typing import Optional, Union, List, Tuple
import os, torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

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

class AIDSMatchingDataset(Dataset):
    def __init__(self, api_type='pyg', path='./data',
                 num_train_pairs=1200, num_val_pairs=200, num_test_pairs=200,
                 seed=7):
        super().__init__(api_type, path)
        assert self.api_type=='pyg', 'This repro uses PyG.'
        self.dataset_name = 'AIDS'
        self.graph_dataset = TUDataset(root=os.path.join(path, 'AIDS'), name='AIDS')
        self.graph_data = self.graph_dataset[0]
        self.num_features = int(self.graph_dataset.num_features)
        self.num_classes = int(self.graph_dataset.num_classes)
        self.num_nodes = int(max(g.num_nodes for g in self.graph_dataset))
        torch.manual_seed(seed)
        self.train_pairs, self.train_sims = self._create_graph_pairs(num_train_pairs, seed+1)
        self.val_pairs,   self.val_sims   = self._create_graph_pairs(num_val_pairs,   seed+2)
        self.test_pairs,  self.test_sims  = self._create_graph_pairs(num_test_pairs,  seed+3)

    def get_name(self):
        return 'AIDS'

    def _create_graph_pairs(self, num_pairs=1000, seed=42):
        torch.manual_seed(seed)
        pairs: List[Tuple[Data,Data]] = []
        sims:  List[float] = []
        n = len(self.graph_dataset)
        for _ in range(num_pairs):
            i1, i2 = torch.randint(0, n, (2,)).tolist()
            g1 = self.graph_dataset[i1]
            g2 = self.graph_dataset[i2]
            if int(g1.y) == int(g2.y):
                base = 0.7 + 0.3*torch.rand(1).item()
            else:
                base = 0.1 + 0.4*torch.rand(1).item()
            size_diff = abs(g1.num_nodes - g2.num_nodes) / max(g1.num_nodes, g2.num_nodes)
            sim = base * (1 - 0.3*size_diff)
            pairs.append((g1,g2))
            sims.append(float(sim))
        return pairs, sims
