import time
import numpy as np
import torch
import dgl
from typing import Optional, Dict, Any
from .base import BaseAttack
from pygip.models.nn import GCN

class MyCustomAttack(BaseAttack):
    supported_api_types = {"dgl"}
    supported_datasets = {"Cora"}

    def __init__(self,
                 dataset,
                 attack_node_fraction: float = 0.05,
                 samples_per_class: int = 10,
                 subgraph_size: int = 8,
                 fully_connect: bool = True,
                 connect_to_orig: bool = True,
                 seed: Optional[int] = 42,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None):
        super().__init__(dataset, attack_node_fraction, model_path, device)
        self.samples_per_class = int(samples_per_class)
        self.subgraph_size = int(subgraph_size)
        self.fully_connect = bool(fully_connect)
        self.connect_to_orig = bool(connect_to_orig)
        self.rng = np.random.RandomState(int(seed) if seed is not None else None)
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        self.graph = self.graph_data
        self.features = self.graph.ndata.get('feat')
        self.labels = self.graph.ndata.get('label')
        self.net_target = None
        self.net_attack = None

    def _build_priors(self):
        X = self.features.cpu().numpy()
        Y = self.labels.cpu().numpy()
        num_features = X.shape[1]
        Fd, Md = [], []
        for c in range(int(self.num_classes)):
            class_nodes = X[Y == c]
            if class_nodes.shape[0] == 0:
                Fd.append(np.ones(num_features, dtype=float) / num_features)
                Md.append(np.ones(self.subgraph_size + 1, dtype=float) / (self.subgraph_size + 1))
                continue
            class_nodes = np.maximum(class_nodes, 0.0)
            feature_counts = class_nodes.sum(axis=0)
            if feature_counts.sum() > 0:
                Fd.append(feature_counts / feature_counts.sum())
            else:
                Fd.append(np.ones(num_features, dtype=float) / num_features)
            per_node_counts = class_nodes.sum(axis=1).astype(int)
            per_node_counts = np.clip(per_node_counts, 0, self.subgraph_size)
            binc = np.bincount(per_node_counts, minlength=self.subgraph_size + 1).astype(float)
            if binc.sum() > 0:
                Md.append(binc / binc.sum())
            else:
                Md.append(np.ones(self.subgraph_size + 1, dtype=float) / (self.subgraph_size + 1))
        return Fd, Md

    def _sample_subgraph_numpy(self, Fd_c: np.ndarray, Md_c: np.ndarray):
        k = self.subgraph_size
        num_features = int(self.num_features)
        p = Md_c / (Md_c.sum() + 1e-12)
        counts = self.rng.choice(len(p), size=k, p=p)
        Xc = np.zeros((k, num_features), dtype=float)
        for i in range(k):
            feat_count = int(counts[i])
            feat_count = max(1, min(num_features, feat_count))
            feats = self.rng.choice(num_features, size=feat_count, replace=False, p=Fd_c)
            Xc[i, feats] = 1.0
        if self.fully_connect:
            A = np.ones((k, k), dtype=int)
            np.fill_diagonal(A, 0)
        else:
            p_edge = 0.25
            R = (self.rng.rand(k, k) < p_edge).astype(int)
            R = np.triu(R, 1)
            A = R + R.T
        src, dst = np.nonzero(A)
        return Xc, src, dst, k

    def attack(self) -> Dict[str, Any]:
        start = time.time()
        g_cpu = self.graph.to('cpu').clone()
        num_orig = g_cpu.number_of_nodes()
        if self.attack_node_fraction is not None:
            budget_nodes = max(1, int(self.attack_node_fraction * int(self.num_nodes)))
        else:
            budget_nodes = max(1, self.samples_per_class * self.subgraph_size)
        Fd, Md = self._build_priors()
        synth_feat_list, synth_label_list = [], []
        synth_edge_blocks = []
        nodes_generated = 0
        blocks_created = 0
        for c in range(int(self.num_classes)):
            for _ in range(self.samples_per_class):
                if nodes_generated >= budget_nodes:
                    break
                Xc, src, dst, k = self._sample_subgraph_numpy(Fd[c], Md[c])
                if nodes_generated + k > budget_nodes:
                    remaining = budget_nodes - nodes_generated
                    if remaining <= 0:
                        break
                    Xc = Xc[:remaining, :]
                    k = remaining
                    mask = (src < remaining) & (dst < remaining)
                    src = src[mask]
                    dst = dst[mask]
                synth_feat_list.append(torch.tensor(Xc, dtype=self.features.dtype))
                synth_label_list.append(torch.tensor([c] * k, dtype=self.labels.dtype))
                synth_edge_blocks.append((src.copy(), dst.copy(), k))
                nodes_generated += k
                blocks_created += 1
            if nodes_generated >= budget_nodes:
                break
        if nodes_generated == 0:
            return {
                'status': 'no_synthetic_nodes_created',
                'num_original_nodes': num_orig,
                'num_new_nodes': 0,
                'attack_node_fraction_requested': self.attack_node_fraction,
                'attack_node_fraction_used': 0.0,
                'time_seconds': time.time() - start
            }
        XG = torch.cat(synth_feat_list, dim=0)
        YG = torch.cat(synth_label_list, dim=0)
        if 'feat' not in g_cpu.ndata or 'label' not in g_cpu.ndata:
            raise RuntimeError("Original graph missing 'feat' or 'label' ndata key")
        g_cpu.add_nodes(nodes_generated)
        orig_feats = g_cpu.ndata['feat'][:num_orig]
        orig_labels = g_cpu.ndata['label'][:num_orig]
        total_nodes = num_orig + nodes_generated
        new_feats = torch.zeros((total_nodes, orig_feats.size(1)), dtype=orig_feats.dtype)
        new_labels = torch.zeros((total_nodes,), dtype=orig_labels.dtype)
        new_feats[:num_orig] = orig_feats
        new_labels[:num_orig] = orig_labels
        if XG.size(1) != orig_feats.size(1):
            raise RuntimeError(f"Feature dimension mismatch: {XG.size(1)} != {orig_feats.size(1)}")
        new_feats[num_orig:] = XG
        new_labels[num_orig:] = YG
        g_cpu.ndata['feat'] = new_feats
        g_cpu.ndata['label'] = new_labels
        cur_block_start = num_orig
        offset = 0
        for (src, dst, block_k) in synth_edge_blocks:
            block_shift = cur_block_start + offset
            if src.size > 0:
                src_shifted = (src + block_shift).tolist()
                dst_shifted = (dst + block_shift).tolist()
                g_cpu.add_edges(src_shifted, dst_shifted)
            offset += block_k
        if self.connect_to_orig:
            offset2 = 0
            for (_, _, block_k) in synth_edge_blocks:
                if block_k <= 0:
                    continue
                block_first_node = num_orig + offset2
                target_orig = int(self.rng.randint(0, num_orig))
                g_cpu.add_edges([block_first_node], [target_orig])
                g_cpu.add_edges([target_orig], [block_first_node])
                offset2 += block_k
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            if mask_name in g_cpu.ndata:
                old_mask = g_cpu.ndata[mask_name]
                ext = torch.zeros(nodes_generated, dtype=old_mask.dtype)
                g_cpu.ndata[mask_name] = torch.cat([old_mask[:num_orig], ext], dim=0)
        perturbed = g_cpu.to(self.device)
        actual_fraction_used = nodes_generated / float(self.num_nodes) if self.num_nodes > 0 else 0.0
        metrics = {'budget_nodes_requested': budget_nodes, 'nodes_generated': nodes_generated, 'blocks_created': blocks_created}
        return {
            'status': 'success',
            'perturbed_graph': perturbed,
            'num_original_nodes': num_orig,
            'num_new_nodes': nodes_generated,
            'attack_node_fraction_requested': self.attack_node_fraction,
            'attack_node_fraction_used': actual_fraction_used,
            'metrics': metrics,
            'time_seconds': time.time() - start
        }

    def _load_model(self, model_path: str):
        if model_path is None:
            return None
        net = GCN(self.num_features, self.num_classes).to(self.device)
        net.load_state_dict(torch.load(model_path, map_location=self.device))
        net.eval()
        self.net_target = net
        return net

    def _train_target_model(self, epochs: int = 200, lr: float = 0.01, weight_decay: float = 5e-4):
        net = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        g = self.graph.to(self.device)
        feats = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata.get('train_mask')
        if train_mask is None:
            raise RuntimeError("Graph is missing 'train_mask' ndata; cannot train target model")
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = net(g, feats)
            loss = torch.nn.functional.nll_loss(torch.log_softmax(logits, dim=1)[train_mask], labels[train_mask])
            loss.backward()
            opt.step()
        net.eval()
        self.net_target = net
        return net

    def _train_attack_model(self, *args, **kwargs):
        return None
