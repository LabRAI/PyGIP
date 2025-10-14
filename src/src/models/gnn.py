import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    SGConv
)


# ------------------- Base Helper -------------------
def get_hidden_dims(hidden_dim):
    """Ensure hidden_dim is a list."""
    if isinstance(hidden_dim, int):
        return [hidden_dim]
    elif isinstance(hidden_dim, list):
        return hidden_dim
    else:
        raise ValueError("hidden_dim must be int or list of ints")


# ------------------- GCN -------------------
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
        super(GCN, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()

        self.layers.append(GCNConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNConv(hidden_dim[i], hidden_dim[i + 1]))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- GraphSAGE -------------------
class GraphSage(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
        super(GraphSage, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()

        self.layers.append(SAGEConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SAGEConv(hidden_dim[i], hidden_dim[i + 1]))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- GAT -------------------
class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], heads=8):
        super(GAT, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()

        self.layers.append(GATConv(in_dim, hidden_dim[0], heads=heads, concat=True))
        for i in range(len(hidden_dim) - 1):
            in_channels = hidden_dim[i] * heads if i == 0 else hidden_dim[i]
            self.layers.append(GATConv(in_channels, hidden_dim[i + 1], heads=1, concat=False))

        self.fc = Linear(hidden_dim[-1], out_dim)
        self.heads = heads

    def forward(self, data):
        x, edge_index = data
        for i, conv in enumerate(self.layers):
            x = F.relu(conv(x, edge_index))
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- GIN -------------------
class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
        super(GIN, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()

        self.layers.append(GINConv(
            Sequential(
                Linear(in_dim, in_dim),
                BatchNorm1d(in_dim),
                ReLU(),
                Linear(in_dim, hidden_dim[0]),
                ReLU()
            )
        ))

        for i in range(len(hidden_dim) - 1):
            self.layers.append(GINConv(
                Sequential(
                    Linear(hidden_dim[i], hidden_dim[i]),
                    BatchNorm1d(hidden_dim[i]),
                    ReLU(),
                    Linear(hidden_dim[i], hidden_dim[i + 1]),
                    ReLU()
                )
            ))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for conv in self.layers:
            x = conv(x, edge_index)
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- SGC -------------------
class SGC(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], K=2):
        super(SGC, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()

        self.layers.append(SGConv(in_dim, hidden_dim[0], K=K))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SGConv(hidden_dim[i], hidden_dim[i + 1], K=K))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- Model Factory -------------------
def get_gnn_model(model_name, in_dim, out_dim, hidden_dim=[64, 32], **kwargs):
    model_name = model_name.lower()
    if model_name == "gcn":
        return GCN(in_dim, out_dim, hidden_dim)
    elif model_name == "graphsage":
        return GraphSage(in_dim, out_dim, hidden_dim)
    elif model_name == "gat":
        return GAT(in_dim, out_dim, hidden_dim, **kwargs)
    elif model_name == "gin":
        return GIN(in_dim, out_dim, hidden_dim)
    elif model_name == "sgc":
        return SGC(in_dim, out_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")
