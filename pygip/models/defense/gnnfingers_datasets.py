# datasets.py
import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

def load_node_dataset(name):
    if name == "Cora":
        return Planetoid(root='data/Cora', name='Cora')[0]
    elif name == "CiteSeer":
        return Planetoid(root='data/CiteSeer', name='CiteSeer')[0]
    elif name == "PubMed":
        return Planetoid(root='data/PubMed', name='PubMed')[0]
    else:
        raise ValueError(f"Unsupported node dataset: {name}")

def load_graph_dataset(name, batch_size=32):
    if name == "PROTEINS":
        dataset = TUDataset(root='data/PROTEINS', name='PROTEINS')
    elif name == "ENZYMES":
        dataset = TUDataset(root='data/ENZYMES', name='ENZYMES')
    elif name == "NCI1":
        dataset = TUDataset(root='data/NCI1', name='NCI1')
    elif name == "IMDB-BINARY":
        dataset = TUDataset(root='data/IMDB-BINARY', name='IMDB-BINARY')
    else:
        raise ValueError(f"Unsupported graph dataset: {name}")

    dataset = dataset.shuffle()
    split_idx = int(0.8 * len(dataset))
    train_dataset, test_dataset = dataset[:split_idx], dataset[split_idx:]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
