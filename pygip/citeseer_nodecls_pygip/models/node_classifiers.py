import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCNNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3, num_classes=6):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.proj  = nn.Linear(hidden, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.dropout = dropout

    def forward_graph(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        x = self.proj(x)
        x = F.relu(x)
        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.forward_graph(x, edge_index)
        logits = self.classifier(z)
        return logits, z

class SAGENodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=64, dropout=0.3, num_classes=6):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.proj  = nn.Linear(hidden, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.dropout = dropout

    def forward_graph(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        x = self.proj(x)
        x = F.relu(x)
        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.forward_graph(x, edge_index)
        logits = self.classifier(z)
        return logits, z
