#Fingerprints class - learnable fingerprints and Univerifier

import torch
import torch.nn as nn

class LearnableFingerprint(nn.Module):
    def __init__(self, num_nodes, feat_dim, density=0.05, device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat = nn.Parameter(torch.randn(num_nodes, feat_dim) * 0.1)
        self.adj_param = nn.Parameter(torch.randn(num_nodes, num_nodes) * -3.0)

        with torch.no_grad():
            mask = torch.rand(num_nodes, num_nodes, device=device) < density
            mask.fill_diagonal_(0)
            self.adj_param[mask] = 3.0
            ap = (self.adj_param + self.adj_param.t()) / 2.0
            self.adj_param.copy_(ap)

        src, dst = torch.where(~torch.eye(num_nodes, dtype=torch.bool, device=device))
        self.register_buffer('edge_index_all',torch.stack([src,dst], dim=0))

    def current_edge_weight(self):
        w = torch.sigmoid(self.adj_param)
        ew = w[self.edge_index_all[0], self.edge_index_all[1]]
        return ew

    def harden_topk(self, edge_density: float):
        with torch.no_grad():
            w =torch.sigmoid(self.adj_param)
            iu = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1, device=w.device)
            wu= w[iu[0], iu[1]]
            k = max(1, int(edge_density * wu.numel()))
            vals, idx= torch.topk(wu, k)
            keep = torch.zeros_like(wu, dtype=torch.bool)
            keep[idx] =True
            bin_u = torch.zeros_like(wu)
            bin_u[keep] = 1.0
            w_new = torch.zeros_like(w)
            w_new[iu[0],iu[1]] =bin_u
            w_new = w_new + w_new.t()
            eps = 1e-3
            w_new.clamp_(0, 1)
            p_new = torch.log((w_new + eps) / (1-w_new + eps))
            self.adj_param.copy_((p_new + p_new.t()) / 2.0)

    def forward(self, model):
        ew = self.current_edge_weight()
        logits = model(self.feat, self.edge_index_all, edge_weight=ew)
        return logits

class Univerifier(nn.Module):
    def __init__(self, in_dim, hidden=[128, 64, 32], p=0.1):
        super().__init__()
        dims = [in_dim]+hidden+[2]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LeakyReLU(), nn.Dropout(p)]
        layers +=[nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

