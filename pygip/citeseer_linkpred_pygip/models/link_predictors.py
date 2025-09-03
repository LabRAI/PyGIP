# models/link_predictors.py
# GCN/GraphSAGE encoders for link prediction (dot-product head)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden=16, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

    def forward(self, data):
        h = self.encode(data.x, data.edge_index)
        ei = data.edge_label_index
        h_s, h_d = h[ei[0]], h[ei[1]]
        scores = torch.sigmoid((h_s * h_d).sum(dim=-1))
        return scores

class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden=16, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

    def forward(self, data):
        h = self.encode(data.x, data.edge_index)
        ei = data.edge_label_index
        h_s, h_d = h[ei[0]], h[ei[1]]
        scores = torch.sigmoid((h_s * h_d).sum(dim=-1))
        return scores
