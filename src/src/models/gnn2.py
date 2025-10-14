import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    SGConv
)


# ------------------- Utility Functions -------------------
def get_hidden_dims(hidden_dim):
    """Ensure hidden_dim is a list."""
    if isinstance(hidden_dim, int):
        return [hidden_dim]
    elif isinstance(hidden_dim, list):
        return hidden_dim
    else:
        raise ValueError("hidden_dim must be int or list of ints")


def get_activation(name="relu"):
    """Return the chosen activation function."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.2)
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {name}")


# ------------------- GCN -------------------
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32],
                 dropout=0.5, activation="relu", use_bn=True):
        super(GCN, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        self.act = get_activation(activation)
        self.dropout = dropout
        self.use_bn = use_bn

        # Build GCN layers
        self.layers.append(GCNConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNConv(hidden_dim[i], hidden_dim[i + 1]))
        if use_bn:
            for h in hidden_dim:
                self.bns.append(BatchNorm1d(h))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- GraphSAGE -------------------
class GraphSage(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32],
                 dropout=0.5, activation="relu", use_bn=True):
        super(GraphSage, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        self.act = get_activation(activation)
        self.dropout = dropout
        self.use_bn = use_bn

        self.layers.append(SAGEConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SAGEConv(hidden_dim[i], hidden_dim[i + 1]))
        if use_bn:
            for h in hidden_dim:
                self.bns.append(BatchNorm1d(h))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- GAT -------------------
class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], heads=8,
                 dropout=0.5, activation="relu", use_bn=True):
        super(GAT, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        self.act = get_activation(activation)
        self.dropout = dropout
        self.use_bn = use_bn
        self.heads = heads

        self.layers.append(GATConv(in_dim, hidden_dim[0], heads=heads, concat=True))
        for i in range(len(hidden_dim) - 1):
            in_channels = hidden_dim[i] * heads if i == 0 else hidden_dim[i]
            self.layers.append(GATConv(in_channels, hidden_dim[i + 1], heads=1, concat=False))
        if use_bn:
            for h in hidden_dim:
                self.bns.append(BatchNorm1d(h))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- GIN -------------------
class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32],
                 dropout=0.5, activation="relu", use_bn=True):
        super(GIN, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        self.act = get_activation(activation)
        self.dropout = dropout
        self.use_bn = use_bn

        def mlp(in_dim, out_dim):
            layers = [
                Linear(in_dim, out_dim),
                self.act
            ]
            return Sequential(*layers)

        self.layers.append(GINConv(mlp(in_dim, hidden_dim[0])))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GINConv(mlp(hidden_dim[i], hidden_dim[i + 1])))
        if use_bn:
            for h in hidden_dim:
                self.bns.append(BatchNorm1d(h))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- SGC -------------------
class SGC(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], K=2,
                 dropout=0.5, activation="relu", use_bn=True):
        super(SGC, self).__init__()
        hidden_dim = get_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        self.act = get_activation(activation)
        self.dropout = dropout
        self.use_bn = use_bn

        self.layers.append(SGConv(in_dim, hidden_dim[0], K=K))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SGConv(hidden_dim[i], hidden_dim[i + 1], K=K))
        if use_bn:
            for h in hidden_dim:
                self.bns.append(BatchNorm1d(h))

        self.fc = Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        x = self.fc(x)
        return embedding, x


# ------------------- Model Factory -------------------
def get_gnn_model(model_name, in_dim, out_dim, hidden_dim=[64, 32], **kwargs):
    model_name = model_name.lower()
    if model_name == "gcn":
        return GCN(in_dim, out_dim, hidden_dim, **kwargs)
    elif model_name == "graphsage":
        return GraphSage(in_dim, out_dim, hidden_dim, **kwargs)
    elif model_name == "gat":
        return GAT(in_dim, out_dim, hidden_dim, **kwargs)
    elif model_name == "gin":
        return GIN(in_dim, out_dim, hidden_dim, **kwargs)
    elif model_name == "sgc":
        return SGC(in_dim, out_dim, hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")
