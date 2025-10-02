import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv, SAGEConv, global_mean_pool


class GCN(nn.Module):
    """A simple GCN Network."""

    def __init__(self, feature_number, label_number):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = self.layers[0](g, features)
        x = F.relu(x)
        x = self.layers[1](g, x)
        return x


class GraphSAGE(nn.Module):
    """
    A GraphSAGE model implemented with PyG's SAGEConv module.

    It consists of two SAGEConv layers:
    - The first layer projects features to 'hidden_channels',
    - The second layer outputs 'out_channels'.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initializes the GraphSAGE model.

        Parameters
        ----------
        in_channels : int
            The dimensionality of the input features.
        hidden_channels : int
            The dimensionality of the hidden layer.
        out_channels : int
            The dimensionality of the output layer (or the number of classes).
        """
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggregator_type='mean')

    def forward(self, blocks, x):
        """
        Forward pass.

        Parameters
        ----------
        blocks : list of dgl.DGLGraph
            A list of subgraphs sampled for multiple layers.
        x : torch.Tensor
            The node features of shape (num_nodes, in_channels).

        Returns
        -------
        torch.Tensor
            The model outputs (logits) of shape (num_nodes, out_channels).
        """
        x = self.conv1(blocks[0], x)
        x = F.relu(x)
        x = self.conv2(blocks[1], x)
        return x


class ShadowNet(torch.nn.Module):
    """A shadow model GCN."""

    def __init__(self, feature_number, label_number):
        super(ShadowNet, self).__init__()
        self.layer1 = GraphConv(feature_number, 16)
        self.layer2 = GraphConv(16, label_number)

    def forward(self, g, features):
        x = torch.nn.functional.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


class AttackNet(nn.Module):
    """An attack model GCN."""

    def __init__(self, feature_number, label_number):
        super(AttackNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x



class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, out_channels, heads=1)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class GCN_PyG(nn.Module):  # Rename to avoid clash with existing DGL GCN
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)