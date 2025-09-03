
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3, num_classes=6):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.proj  = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(out_dim, num_classes)
    def forward_graph(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)
        g = self.proj(g)
        return g
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z = self.forward_graph(x, edge_index, batch)
        logits = self.classifier(z)
        return logits, z

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3, num_classes=6):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.proj  = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(out_dim, num_classes)
    def forward_graph(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)
        g = self.proj(g)
        return g
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z = self.forward_graph(x, edge_index, batch)
        logits = self.classifier(z)
        return logits, z
