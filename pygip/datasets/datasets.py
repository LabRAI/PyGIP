import dgl
import numpy as np
import torch
from dgl import DGLGraph
from dgl.data import AmazonCoBuyComputerDataset  # Amazon-Computer
from dgl.data import AmazonCoBuyPhotoDataset  # Amazon-Photo
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import FakeNewsDataset
from dgl.data import FlickrDataset
from dgl.data import GINDataset
from dgl.data import MUTAGDataset
from dgl.data import RedditDataset
from dgl.data import YelpDataset
from dgl.data import citation_graph  # Cora, CiteSeer, PubMed
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Data as PyGData
from torch_geometric.datasets import Amazon  # Amazon Computers, Photo
from torch_geometric.datasets import Coauthor  # cs, physics
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.datasets import Flickr as FlickrPyG
from torch_geometric.datasets import LastFMAsia
from torch_geometric.datasets import Planetoid  # Cora, CiteSeer, PubMed
from torch_geometric.datasets import PolBlogs as PolBlogsPyG
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import TUDataset  # ENZYMES

## Added for EGSteal
from torch_geometric.transforms import Constant

def dgl_to_tg(dgl_graph):
    edge_index = torch.stack(dgl_graph.edges())
    x = dgl_graph.ndata.get('feat')
    y = dgl_graph.ndata.get('label')

    train_mask = dgl_graph.ndata.get('train_mask')
    val_mask = dgl_graph.ndata.get('val_mask')
    test_mask = dgl_graph.ndata.get('test_mask')

    data = PyGData(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data


def tg_to_dgl(py_g_data):
    edge_index = py_g_data.edge_index
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]))

    if py_g_data.x is not None:
        dgl_graph.ndata['feat'] = py_g_data.x
    if py_g_data.y is not None:
        dgl_graph.ndata['label'] = py_g_data.y

    if hasattr(py_g_data, 'train_mask') and py_g_data.train_mask is not None:
        dgl_graph.ndata['train_mask'] = py_g_data.train_mask
    if hasattr(py_g_data, 'val_mask') and py_g_data.val_mask is not None:
        dgl_graph.ndata['val_mask'] = py_g_data.val_mask
    if hasattr(py_g_data, 'test_mask') and py_g_data.test_mask is not None:
        dgl_graph.ndata['test_mask'] = py_g_data.test_mask

    return dgl_graph


class Dataset(object):
    def __init__(self, api_type='dgl', path='./data'):
        assert api_type in {'dgl', 'pyg'}, 'API type must be dgl or pyg'
        self.api_type = api_type
        self.path = path
        self.dataset_name = self.get_name()

        # DGLGraph or PyGData
        self.graph_dataset = None
        self.graph_data = None

        # meta data
        self.num_nodes = 0
        self.num_features = 0
        self.num_classes = 0

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'pyg':
            self.load_pyg_data()
        else:
            raise ValueError("Unsupported api_type.")

        self._load_meta_data()

    def get_name(self):
        return self.__class__.__name__

    def load_dgl_data(self):
        raise NotImplementedError("load_dgl_data not implemented in subclasses.")

    def load_pyg_data(self):
        raise NotImplementedError("load_pyg_data not implemented in subclasses.")

    def _load_meta_data(self):
        if isinstance(self.graph_data, DGLGraph):
            self.num_nodes = self.graph_data.number_of_nodes()
            self.num_features = len(self.graph_data.ndata['feat'][0])
            self.num_classes = int(max(self.graph_data.ndata['label']) - min(self.graph_data.ndata['label'])) + 1
        elif isinstance(self.graph_data, PyGData):
            self.num_nodes = self.graph_data.num_nodes
            self.num_features = self.graph_dataset.num_node_features
            self.num_classes = self.graph_dataset.num_classes
        else:
            raise TypeError("graph_data must be either DGLGraph or torch_geometric.data.Data.")

    def _generate_masks_by_ratio(self, train_ratio=0.8):
        if self.graph_data is None:
            raise ValueError("graph_data is not loaded.")

        try:
            import dgl
        except ImportError:
            dgl = None

        try:
            from torch_geometric.data import Data
        except ImportError:
            Data = None

        is_dgl = dgl and isinstance(self.graph_data, dgl.DGLGraph)
        is_pyg = Data and isinstance(self.graph_data, Data)

        if not (is_dgl or is_pyg):
            raise TypeError("graph_data must be either DGLGraph or torch_geometric.data.Data.")

        # Check if masks already exist
        if is_dgl:
            if all(k in self.graph_data.ndata for k in ['train_mask', 'val_mask', 'test_mask']):
                print("Masks already exist in DGL graph. Skipping mask generation.")
                return
            num_nodes = self.graph_data.num_nodes()
        else:  # PyG
            if all(hasattr(self.graph_data, k) for k in ['train_mask', 'val_mask', 'test_mask']):
                print("Masks already exist in PyG data. Skipping mask generation.")
                return
            num_nodes = self.graph_data.num_nodes

        # Generate masks
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = (num_nodes - train_size) // 2

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        # Store masks
        if is_dgl:
            self.graph_data.ndata['train_mask'] = train_mask
            self.graph_data.ndata['val_mask'] = val_mask
            self.graph_data.ndata['test_mask'] = test_mask
        else:  # PyG
            self.graph_data.train_mask = train_mask
            self.graph_data.val_mask = val_mask
            self.graph_data.test_mask = test_mask

        print(f"Masks successfully generated and stored. (train_ratio={train_ratio})")

    def _generate_masks_by_classes(self, num_class_samples=100, val_count=500, test_count=1000, seed=42):
        """
        For Amazon and Coauthor datasets:
        - train: `num_class_samples` per class
        - val: `val_count` nodes from remaining
        - test: `test_count` nodes from remaining after val
        Works for both DGL and PyG graphs via self.graph_data
        """
        try:
            import dgl
        except ImportError:
            dgl = None
        try:
            from torch_geometric.data import Data as PyGData
        except ImportError:
            PyGData = None

        is_dgl = dgl is not None and isinstance(self.graph_data, dgl.DGLGraph)
        is_pyg = PyGData is not None and isinstance(self.graph_data, PyGData)

        if not (is_dgl or is_pyg):
            raise TypeError("graph_data must be either DGLGraph or torch_geometric.data.Data.")

        if is_dgl:
            if all(k in self.graph_data.ndata for k in ['train_mask', 'val_mask', 'test_mask']):
                print("Masks already exist in DGL graph. Skipping mask generation.")
                return
            num_nodes = self.graph_data.num_nodes()
            labels = self.graph_data.ndata['label']
        else:  # PyG
            if all(hasattr(self.graph_data, k) for k in ['train_mask', 'val_mask', 'test_mask']):
                print("Masks already exist in PyG data. Skipping mask generation.")
                return
            num_nodes = self.graph_data.num_nodes
            labels = self.graph_data.y

        num_classes = int(labels.max().item()) + 1

        used_mask = torch.zeros(num_nodes, dtype=torch.bool)
        generator = torch.Generator().manual_seed(seed)
        train_idx_parts = []

        # train set
        print("Training samples per class:")
        for c in range(num_classes):
            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            if class_idx.numel() == 0:
                print(f"  class {c}: no available samples")
                continue
            perm = class_idx[torch.randperm(class_idx.size(0), generator=generator)]
            n_select = min(num_class_samples, perm.size(0))
            selected = perm[:n_select]
            train_idx_parts.append(selected)
            used_mask[selected] = True
            print(f"  class {c}: select {n_select} samples")

        if len(train_idx_parts) == 0:
            raise ValueError("no training samples available.")

        train_idx = torch.cat(train_idx_parts, dim=0)

        # val set
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        if remaining_idx.numel() == 0:
            raise ValueError("no remaining samples available.")
        remaining_perm = remaining_idx[torch.randperm(remaining_idx.size(0), generator=generator)]

        val_take = min(val_count, remaining_perm.size(0))
        val_idx = remaining_perm[:val_take]
        used_mask[val_idx] = True

        # test set
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        test_take = min(test_count, remaining_idx.size(0))
        test_idx = remaining_idx[:test_take]

        train_mask = self._index_to_mask(train_idx, num_nodes)
        val_mask = self._index_to_mask(val_idx, num_nodes)
        test_mask = self._index_to_mask(test_idx, num_nodes)

        if is_pyg:
            self.graph_data.train_mask = train_mask
            self.graph_data.val_mask = val_mask
            self.graph_data.test_mask = test_mask
        else:
            self.graph_data.ndata["train_mask"] = train_mask
            self.graph_data.ndata["val_mask"] = val_mask
            self.graph_data.ndata["test_mask"] = test_mask

    def _index_to_mask(self, index: torch.Tensor, size: int):
        mask = torch.zeros(size, dtype=torch.bool, device=index.device if isinstance(index, torch.Tensor) else None)
        mask[index] = True
        return mask

    def __repr__(self):
        return (f"Dataset(name={self.dataset_name}, api_type={self.api_type}, "
                f"#Nodes={self.num_nodes}, #Features={self.num_features}, "
                f"#Classes={self.num_classes})")


class Cora(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = citation_graph.load_cora()
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

    def load_pyg_data(self):
        dataset = Planetoid(root=self.path, name='Cora')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data


class CiteSeer(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = citation_graph.load_citeseer()
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

    def load_pyg_data(self):
        dataset = Planetoid(root=self.path, name='Citeseer')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data


class PubMed(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = citation_graph.load_pubmed()
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

    def load_pyg_data(self):
        dataset = Planetoid(root=self.path, name='PubMed')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data


class Computers(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = AmazonCoBuyComputerDataset(raw_dir=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = dgl.add_self_loop(data)

        self._generate_masks_by_classes()

    def load_pyg_data(self):
        dataset = Amazon(root=self.path, name='Computers')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

        self._generate_masks_by_classes()


class Photo(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = AmazonCoBuyPhotoDataset(raw_dir=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = dgl.add_self_loop(data)

        self._generate_masks_by_classes()

    def load_pyg_data(self):
        dataset = Amazon(root=self.path, name='Photo')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

        self._generate_masks_by_classes()


class CoauthorCS(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = CoauthorCSDataset(raw_dir=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

        self._generate_masks_by_classes()

    def load_pyg_data(self):
        dataset = Coauthor(root=self.path, name='CS')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

        self._generate_masks_by_classes()


class CoauthorPhysics(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = CoauthorPhysicsDataset(raw_dir=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

        self._generate_masks_by_classes()

    def load_pyg_data(self):
        dataset = Coauthor(root=self.path, name='Physics')
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

        self._generate_masks_by_classes()


class ENZYMES(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_pyg_data(self):
        dataset = TUDataset(root=self.path, name='ENZYMES')
        data_list = [data for data in dataset]
        all_x = torch.cat([d.x for d in data_list], dim=0)
        mean, std = all_x.mean(0), all_x.std(0)
        for d in data_list:
            d.x = (d.x - mean) / (std + 1e-6)
        all_labels = np.array([int(d.y) for d in data_list])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))
        self.train_data = [data_list[i] for i in train_idx]
        self.test_data = [data_list[i] for i in test_idx]


class Facebook(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_pyg_data(self):
        dataset = FacebookPagePage(root=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data


class Flickr(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = FlickrDataset(raw_dir=self.path)
        self.graph_dataset = dataset
        self.graph_data = dataset[0]

    def load_pyg_data(self):
        dataset = FlickrPyG(root=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data


class PolBlogs(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_pyg_data(self):
        dataset = PolBlogsPyG(root=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data

        self._generate_masks_by_classes()


class LastFM(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_pyg_data(self):
        dataset = LastFMAsia(root=self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data


class Reddit(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = RedditDataset(raw_dir=self.path)
        self.graph_dataset = dataset
        self.graph_data = dataset[0]

    def load_pyg_data(self):
        dataset = Reddit(self.path)
        data = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = data


class Twitter(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = FakeNewsDataset('gossipcop', 'bert', raw_dir=self.path)
        graph, _ = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = dgl.add_self_loop(graph)


class MUTAG(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = MUTAGDataset(raw_dir=self.path)
        self.graph_dataset = dataset
        self.graph_data = dataset[0]


class PTC(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = GINDataset(name='PTC', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph_dataset = dataset
        self.graph_data = dgl.batch(graph)


class NCI1(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = GINDataset(name='NCI1', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph_dataset = dataset
        self.graph_data = dgl.batch(graph)


class PROTEINS(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = GINDataset(name='PROTEINS', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph_dataset = dataset
        self.graph_data = dgl.batch(graph)


class Collab(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = GINDataset(name='COLLAB', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph_dataset = dataset
        self.graph_data = dgl.batch(graph)


class IMDB(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = GINDataset(name='IMDB-BINARY', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph_dataset = dataset
        self.graph_data = dgl.batch(graph)


class YelpData(Dataset):
    def __init__(self, api_type='dgl', path='./data'):
        super().__init__(api_type, path)

    def load_dgl_data(self):
        dataset = YelpDataset(raw_dir=self.path)
        self.graph_dataset = dataset
        self.graph_data = dataset[0]

class MUTAGGraphClassification(Dataset):
    def __init__(self, api_type='pyg', path='./data'):
        super().__init__(api_type, path)
    
    def _load_meta_data(self):
        if self.api_type == 'pyg':
            ds = self.graph_dataset  
            self.num_features = int(ds.num_node_features)
            self.num_classes  = int(ds.num_classes)
            self.num_graphs    = int(len(ds))
            self.num_edge_features = int(ds.num_edge_features) if ds.num_edge_features is not None else 0
        else:
            super()._load_meta_data()

    def load_pyg_data(self):
        self.dataset_name = 'Mutagenicity'
        temp_dataset = TUDataset(root=self.path,name='Mutagenicity')
        data_transform = None
        if temp_dataset.num_node_features == 0:
            print("\nNo node features found. Adding constant node features (all ones).")
            data_transform = Constant(value=1, cat=False)
        self.graph_dataset = TUDataset(root=self.path,name='Mutagenicity',transform=data_transform)
        num_graphs = len(self.graph_dataset)
        print(f"\nTotal number of graphs in {self.dataset_name}: {num_graphs}")
        print(f"Node features dimension: {self.graph_dataset.num_node_features}")
        print(f"Edge features dimension: {self.graph_dataset.num_edge_features if hasattr(self.graph_dataset, 'num_edge_features') else 'N/A'}")
        print(f"Number of classes: {self.graph_dataset.num_classes if hasattr(self.graph_dataset, 'num_classes') else 'N/A'}")
        return self.graph_dataset
