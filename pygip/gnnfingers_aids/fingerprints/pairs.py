
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from ..utils import get_device

class FingerprintGraphPair(nn.Module):
    def __init__(self, n_nodes: int, feat_dim: int, edge_init_p: float=0.10, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.n = n_nodes
        self.d = feat_dim
        X1 = torch.empty(self.n, self.d, device=self.device).uniform_(-0.5, 0.5)
        X2 = torch.empty(self.n, self.d, device=self.device).uniform_(-0.5, 0.5)
        self.X1 = nn.Parameter(X1)
        self.X2 = nn.Parameter(X2)
        self._init_adj(edge_init_p)

    def _init_adj(self, p):
        A1 = (torch.rand(self.n, self.n, device=self.device) < p).float()
        A2 = (torch.rand(self.n, self.n, device=self.device) < p).float()
        for A in (A1, A2):
            A.fill_diagonal_(0.0)
            A.copy_(torch.maximum(A, A.T))
        eps = 1e-4
        self.A1_logits = nn.Parameter(torch.logit(A1.clamp(eps, 1-eps)))
        self.A2_logits = nn.Parameter(torch.logit(A2.clamp(eps, 1-eps)))

    def edge_index_pair(self, thresh=0.5, update=False):
        def binarize(logits, update=False):
            A = torch.sigmoid(logits)
            if update and logits.grad is not None:
                grad = logits.grad
                mask = torch.sign(grad)
                new_A = A.clone()
                new_A = torch.where((mask > 0) & (A < 0.5), torch.ones_like(new_A), new_A)
                new_A = torch.where((mask < 0) & (A >= 0.5), torch.zeros_like(new_A), new_A)
                new_A.fill_diagonal_(0.0)
                new_A = torch.maximum(new_A, new_A.T)
                return new_A.nonzero(as_tuple=False).t().contiguous()
            else:
                A = (A > thresh).float()
                A.fill_diagonal_(0.0)
                A = torch.maximum(A, A.T)
                idx = A.nonzero(as_tuple=False)
                if idx.numel() == 0:
                    return torch.empty(2, 0, dtype=torch.long, device=self.device)
                return idx.t().contiguous()
        return binarize(self.A1_logits, update), binarize(self.A2_logits, update)

class FingerprintSet(nn.Module):
    def __init__(self, P: int, n_nodes: int, feat_dim: int, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.P = P
        self.fps = nn.ModuleList([FingerprintGraphPair(n_nodes, feat_dim, device=self.device) for _ in range(P)])

    @torch.no_grad()
    def dim_out(self, backbone_out_dim: int):
        return self.P * (2 * backbone_out_dim)

    def model_response(self, model, require_grad=False):
        outs = []
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        model.eval()
        with ctx:
            for fp in self.fps:
                ei1, ei2 = fp.edge_index_pair()
                g1 = Data(x=fp.X1, edge_index=ei1).to(self.device)
                g2 = Data(x=fp.X2, edge_index=ei2).to(self.device)
                z1 = model.forward_graph(g1.x, g1.edge_index)
                z2 = model.forward_graph(g2.x, g2.edge_index)
                outs.append(torch.cat([z1, z2], dim=1))
        return torch.cat(outs, dim=1)

class Verifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64),  nn.LeakyReLU(0.01),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)
