import torch

class GCNLinkPredictor(torch.nn.Module):
    """Tiny fallback GCN-like link predictor used for CI/demo."""
    def __init__(self, in_channels=64, hidden_channels=64):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def encode(self, x, edge_index=None):
        h = self.lin1(x)
        h = torch.relu(h)
        return self.lin2(h)

    def decode(self, z, edge_index):
        # edge_index: [2, E] or tuple/list
        if isinstance(edge_index, (list, tuple)):
            src, dst = edge_index
        else:
            src, dst = edge_index[0], edge_index[1]
        return (z[src] * z[dst]).sum(dim=1)
