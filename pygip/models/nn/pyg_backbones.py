import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, SGConv



# ----------------------------
# GCN
# ----------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)

# ----------------------------
# GraphSAGE
# ----------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)

# ----------------------------
# GAT
# ----------------------------
class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, num_layers=2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden, heads=heads)])
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
        self.convs.append(GATConv(hidden * heads, out_channels, heads=1))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)

# ----------------------------
# GIN
# ----------------------------
class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, num_layers=2):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.convs = nn.ModuleList([GINConv(nn1)])
        for _ in range(num_layers - 2):
            nnk = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim)
            )
            self.convs.append(GINConv(nnk))
        nn_last = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        self.convs.append(GINConv(nn_last))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)

# ----------------------------
# SGC
# ----------------------------
class SGC(nn.Module):
    def __init__(self, in_dim, out_dim, K=2):
        super().__init__()
        self.conv = SGConv(in_dim, out_dim, K=K)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
