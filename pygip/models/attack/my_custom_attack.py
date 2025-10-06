import time
import logging
import numpy as np
import torch
import dgl
from typing import Optional, Dict, Any, List
from .base import BaseAttack
from pygip.models.nn import GCN

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

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
                 device: Optional[str] = None,
                 backdoor_enable: bool = False,
                 backdoor_fraction: float = 0.1,
                 backdoor_feature_indices: Optional[List[int]] = None,
                 backdoor_add_edge_to_node: Optional[int] = None,
                 backdoor_target_label: Optional[int] = None):
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
        self.seed = seed
        self.config = {
            'attack_node_fraction': attack_node_fraction,
            'samples_per_class': samples_per_class,
            'subgraph_size': subgraph_size,
            'fully_connect': fully_connect,
            'connect_to_orig': connect_to_orig,
            'seed': seed,
            'model_path': model_path,
            'device': str(self.device),
            'backdoor_enable': backdoor_enable,
            'backdoor_fraction': backdoor_fraction,
            'backdoor_feature_indices': backdoor_feature_indices,
            'backdoor_add_edge_to_node': backdoor_add_edge_to_node,
            'backdoor_target_label': backdoor_target_label
        }
        self.backdoor_enable = bool(backdoor_enable)
        self.backdoor_fraction = float(backdoor_fraction)
        self.backdoor_feature_indices = backdoor_feature_indices
        self.backdoor_add_edge_to_node = backdoor_add_edge_to_node
        self.backdoor_target_label = backdoor_target_label

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

    def _create_backdoored_copy(self):
        g_cpu = self.graph.to('cpu').clone()
        num_nodes = g_cpu.number_of_nodes()
        test_mask = g_cpu.ndata.get('test_mask')
        if test_mask is None:
            return g_cpu, torch.zeros(num_nodes, dtype=torch.bool)
        test_idxs = test_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        num_poison = max(1, int(len(test_idxs) * self.backdoor_fraction))
        poison_idxs = self.rng.choice(test_idxs, size=num_poison, replace=False).tolist()
        poison_mask = torch.zeros(num_nodes, dtype=torch.bool)
        poison_mask[poison_idxs] = True
        if self.backdoor_feature_indices is not None:
            fi = [int(i) for i in self.backdoor_feature_indices if 0 <= int(i) < g_cpu.ndata['feat'].shape[1]]
            feats = g_cpu.ndata['feat'].clone()
            for idx in poison_idxs:
                feats[idx, fi] = 1
            g_cpu.ndata['feat'] = feats
        if self.backdoor_add_edge_to_node is not None:
            target = int(self.backdoor_add_edge_to_node) % num_nodes
            for idx in poison_idxs:
                g_cpu.add_edges([idx], [target])
                g_cpu.add_edges([target], [idx])
        if self.backdoor_target_label is not None:
            g_cpu.ndata['poison_label'] = g_cpu.ndata.get('poison_label', torch.full((num_nodes,), -1, dtype=g_cpu.ndata['label'].dtype))
            g_cpu.ndata['poison_label'][poison_mask] = int(self.backdoor_target_label)
        g_cpu.ndata['poison_mask'] = poison_mask
        return g_cpu, poison_mask

    def attack(self) -> Dict[str, Any]:
        start = time.time()
        run_logs = []
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
        run_logs.append(f"generated_nodes={nodes_generated},blocks={blocks_created},budget={budget_nodes}")
        if nodes_generated == 0:
            return {
                'status': 'no_synthetic_nodes_created',
                'num_original_nodes': num_orig,
                'num_new_nodes': 0,
                'attack_node_fraction_requested': self.attack_node_fraction,
                'attack_node_fraction_used': 0.0,
                'time_seconds': time.time() - start,
                'config': self.config,
                'seed': self.seed,
                'logs': run_logs
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
        run_logs.append(f"graph_perturbed total_nodes={perturbed.number_of_nodes()} total_edges={perturbed.number_of_edges()}")
        if self.net_target is None:
            if self.model_path is not None:
                try:
                    self._load_model(self.model_path)
                    run_logs.append("loaded_target_model_from_path")
                except Exception as e:
                    run_logs.append(f"failed_load_target:{e}")
                    self._train_target_model()
                    run_logs.append("trained_target_model_fallback")
            else:
                self._train_target_model()
                run_logs.append("trained_target_model")
        else:
            run_logs.append("target_model_already_available")
        target = self.net_target
        perturbed = perturbed.to(self.device)
        surrogate = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        with torch.no_grad():
            target_logits = target(perturbed, perturbed.ndata['feat'])
            pseudo = target_logits.argmax(dim=1)
        labeled_mask = perturbed.ndata.get('train_mask')
        if labeled_mask is None or labeled_mask.sum().item() == 0:
            idxs = list(range(num_orig, num_orig + nodes_generated))
            train_idx = idxs
        else:
            train_idx = (labeled_mask == 1).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(train_idx) == 0:
                train_idx = list(range(num_orig, num_orig + nodes_generated))
        train_labels = pseudo[train_idx].to(self.device)
        epochs_attack = 200
        logs_attack = []
        for epoch in range(epochs_attack):
            surrogate.train()
            opt.zero_grad()
            logits = surrogate(perturbed, perturbed.ndata['feat'])
            loss = torch.nn.functional.cross_entropy(logits[train_idx], train_labels)
            loss.backward()
            opt.step()
            if epoch % 50 == 0:
                s = f"attack_train epoch={epoch} loss={loss.item():.6f}"
                logs_attack.append(s)
                run_logs.append(s)
        self.net_attack = surrogate
        self.net_attack.eval()
        self.net_target.eval()
        with torch.no_grad():
            logits_t_full = self.net_target(self.graph.to(self.device), self.graph.ndata['feat'].to(self.device))
            logits_s_full = self.net_attack(self.graph.to(self.device), self.graph.ndata['feat'].to(self.device))
        test_mask = self.graph.ndata.get('test_mask')
        if test_mask is None:
            raise RuntimeError("Original graph missing 'test_mask' ndata; cannot compute metrics")
        test_mask = test_mask.to(self.device)
        labels = self.graph.ndata['label'].to(self.device)
        tca = (logits_t_full.argmax(dim=1)[test_mask] == labels[test_mask]).float().mean().item()
        eca = (logits_s_full.argmax(dim=1)[test_mask] == labels[test_mask]).float().mean().item()
        fidelity = (logits_s_full.argmax(dim=1)[test_mask] == logits_t_full.argmax(dim=1)[test_mask]).float().mean().item()
        tba = None
        eba = None
        asr_t = None
        asr_s = None
        if self.backdoor_enable:
            backdoored_copy, poison_mask = self._create_backdoored_copy()
            backdoored_copy = backdoored_copy.to(self.device)
            with torch.no_grad():
                logits_t_bd = self.net_target(backdoored_copy, backdoored_copy.ndata['feat'])
                logits_s_bd = self.net_attack(backdoored_copy, backdoored_copy.ndata['feat'])
            if poison_mask.sum().item() > 0:
                mask = poison_mask.to(self.device)
                tba = (logits_t_bd.argmax(dim=1)[mask] == labels[mask]).float().mean().item()
                eba = (logits_s_bd.argmax(dim=1)[mask] == labels[mask]).float().mean().item()
                if self.backdoor_target_label is not None:
                    target_label = int(self.backdoor_target_label)
                    asr_t = (logits_t_bd.argmax(dim=1)[mask] == target_label).float().mean().item()
                    asr_s = (logits_s_bd.argmax(dim=1)[mask] == target_label).float().mean().item()
            run_logs.append(f"backdoor: poison_count={int(poison_mask.sum().item())} TBA={tba} EBA={eba} ASR_target_t={asr_t} ASR_target_s={asr_s}")
        result = {
            'status': 'success',
            'perturbed_graph': perturbed,
            'num_original_nodes': num_orig,
            'num_new_nodes': nodes_generated,
            'attack_node_fraction_requested': self.attack_node_fraction,
            'attack_node_fraction_used': actual_fraction_used,
            'metrics': {
                'TCA': tca,
                'ECA': eca,
                'TBA': tba,
                'EBA': eba,
                'fidelity': fidelity,
                'ASR_target_t': asr_t,
                'ASR_target_s': asr_s
            },
            'time_seconds': time.time() - start,
            'config': self.config,
            'seed': self.seed,
            'logs': run_logs + logs_attack
        }
        logger.info("MyCustomAttack finished: TCA=%.4f ECA=%.4f fidelity=%.4f", tca, eca, fidelity)
        return result

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
        logs = []
        for epoch in range(epochs):
            opt.zero_grad()
            logits = net(g, feats)
            loss = torch.nn.functional.nll_loss(torch.log_softmax(logits, dim=1)[train_mask], labels[train_mask])
            loss.backward()
            opt.step()
            if epoch % 50 == 0:
                s = f"target_train epoch={epoch} loss={loss.item():.6f}"
                logs.append(s)
                logger.info(s)
        net.eval()
        self.net_target = net
        return net

    def _train_attack_model(self, *args, **kwargs):
        return None
