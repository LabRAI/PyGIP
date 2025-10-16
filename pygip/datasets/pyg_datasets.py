import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # for GCN models
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    Flickr,
    Reddit,
    TUDataset,
    FacebookPagePage,
    LastFMAsia,
    PolBlogs as PolBlogsPyG,
    DBLP,
    
)

# ----------------------------
# Base Dataset Wrapper
# ----------------------------
class BasePyGDataset:
    def __init__(self, dataset, data):
        self.graph_dataset = dataset
        self.graph_data = data
        self.num_nodes = data.num_nodes
        self.num_features = dataset.num_node_features
        self.num_classes = dataset.num_classes

    def _generate_masks_by_classes(self, num_class_samples=100, val_count=500, test_count=1000, seed=42):
        """Generate train/val/test masks by selecting fixed number of nodes per class."""
        num_nodes = self.graph_data.num_nodes
        labels = self.graph_data.y
        num_classes = int(labels.max().item()) + 1

        used_mask = torch.zeros(num_nodes, dtype=torch.bool)
        generator = torch.Generator().manual_seed(seed)
        train_idx_parts = []

        # train set
        for c in range(num_classes):
            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            if class_idx.numel() == 0:
                continue
            perm = class_idx[torch.randperm(class_idx.size(0), generator=generator)]
            n_select = min(num_class_samples, perm.size(0))
            selected = perm[:n_select]
            train_idx_parts.append(selected)
            used_mask[selected] = True

        if len(train_idx_parts) == 0:
            raise ValueError("No training samples available.")

        train_idx = torch.cat(train_idx_parts, dim=0)

        # val set
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        remaining_perm = remaining_idx[torch.randperm(remaining_idx.size(0), generator=generator)]
        val_take = min(val_count, remaining_perm.size(0))
        val_idx = remaining_perm[:val_take]
        used_mask[val_idx] = True

        # test set
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        test_take = min(test_count, remaining_idx.size(0))
        test_idx = remaining_idx[:test_take]

        self.graph_data.train_mask = self._index_to_mask(train_idx, num_nodes)
        self.graph_data.val_mask = self._index_to_mask(val_idx, num_nodes)
        self.graph_data.test_mask = self._index_to_mask(test_idx, num_nodes)

    def _index_to_mask(self, index: torch.Tensor, size: int):
        mask = torch.zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

# ----------------------------
# Datasets
# ----------------------------
class Cora(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Planetoid(root=path, name="Cora")
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"  # required for CustomAttack


class CiteSeer(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Planetoid(root=path, name="CiteSeer")
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"



class PubMed(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Planetoid(root=path, name="PubMed")
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"
        
class DBLP(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = DBLP(root=path, name="DBLP")
        super().__init__(dataset, dataset[0])
        self._generate_masks_by_classes()
        self.api_type = "pyg"



class Amazon(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Amazon(root=path, name="Computers")
        super().__init__(dataset, dataset[0])
        self._generate_masks_by_classes()
        self.api_type = "pyg"


class Photo(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Amazon(root=path, name="Photo")
        super().__init__(dataset, dataset[0])
        self._generate_masks_by_classes()
        self.api_type = "pyg"


class CoauthorCS(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Coauthor(root=path, name="CS")
        super().__init__(dataset, dataset[0])
        self._generate_masks_by_classes()
        self.api_type = "pyg"


class CoauthorPhysics(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Coauthor(root=path, name="Physics")
        super().__init__(dataset, dataset[0])
        self._generate_masks_by_classes()
        self.api_type = "pyg"


class ENZYMES(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = TUDataset(root=path, name="ENZYMES")
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"


class Facebook(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = FacebookPagePage(root=path)
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"


class Flickr(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Flickr(root=path)
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"


class PolBlogs(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = PolBlogsPyG(root=path)
        super().__init__(dataset, dataset[0])
        self._generate_masks_by_classes()
        self.api_type = "pyg"


class LastFM(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = LastFMAsia(root=path)
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"


class Reddit(BasePyGDataset):
    def __init__(self, path="./data"):
        dataset = Reddit(root=path)
        super().__init__(dataset, dataset[0])
        self.api_type = "pyg"
