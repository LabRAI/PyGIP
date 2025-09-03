
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.proj  = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.sim_head = nn.Linear(out_dim*2, 1)
    def forward_graph(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)
        g = self.proj(g)
        return g
    def forward_pair(self, g1, g2):
        z1 = self.forward_graph(g1.x, g1.edge_index, getattr(g1,'batch', None))
        z2 = self.forward_graph(g2.x, g2.edge_index, getattr(g2,'batch', None))
        s  = self.sim_head(torch.cat([z1,z2], dim=1))
        return s.squeeze(-1), z1, z2

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.proj  = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.sim_head = nn.Linear(out_dim*2, 1)
    def forward_graph(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)
        g = self.proj(g)
        return g
    def forward_pair(self, g1, g2):
        z1 = self.forward_graph(g1.x, g1.edge_index, getattr(g1,'batch', None))
        z2 = self.forward_graph(g2.x, g2.edge_index, getattr(g2,'batch', None))
        s  = self.sim_head(torch.cat([z1,z2], dim=1))
        return s.squeeze(-1), z1, z2
