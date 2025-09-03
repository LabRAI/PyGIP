# models/link_predictors.py
# GCN/SAGE backbones for link prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(out_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def predict_link(self, z, edge_index):
        src, dst = edge_index
        h_src = z[src]
        h_dst = z[dst]
        h = torch.cat([h_src, h_dst], dim=1)
        return self.link_predictor(h).squeeze(-1)

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        pos_pred = self.predict_link(z, data.edge_index)
        neg_pred = self.predict_link(z, data.neg_edge_index)
        return pos_pred, neg_pred, z

class SAGELinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout
        self.link_predictor = nn.Sequential(
            nn.Linear(out_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def predict_link(self, z, edge_index):
        src, dst = edge_index
        h_src = z[src]
        h_dst = z[dst]
        h = torch.cat([h_src, h_dst], dim=1)
        return self.link_predictor(h).squeeze(-1)

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        pos_pred = self.predict_link(z, data.edge_index)
        neg_pred = self.predict_link(z, data.neg_edge_index)
        return pos_pred, neg_pred, z
