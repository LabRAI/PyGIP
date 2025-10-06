import os
import random
import time
from typing import List, Tuple, Optional

import dgl
import torch
import torch.nn.functional as F
from torch import nn

from pygip.models.attack.base import BaseAttack
from pygip.models.nn.backbones import GCN
from pygip.utils.metrics import AttackMetric, AttackCompMetric


def _as_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)


def add_self_loops(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Return a copy of g with self-loops added to every node."""
    num_nodes = g.number_of_nodes()
    src = torch.arange(num_nodes)
    dst = src.clone()
    return dgl.add_edges(g, src, dst)


def subgraph_from_nodes(g: dgl.DGLGraph, node_idx: List[int]) -> dgl.DGLGraph:
    """Induce a subgraph that contains only edges whose endpoints are both in node_idx."""
    sg = dgl.node_subgraph(g, node_idx)
    sg = dgl.remove_self_loop(sg)
    sg = dgl.add_self_loop(sg)
    return sg


def erdos_renyi_graph(num_nodes: int, p: float = 0.05) -> dgl.DGLGraph:
    import networkx as nx
    g_nx = nx.erdos_renyi_graph(num_nodes, p)
    g = dgl.from_networkx(g_nx)
    g = add_self_loops(g)
    return g


def random_shadow_indices(g: dgl.DGLGraph, k: int, extra: int = 2) -> Tuple[List[int], List[int]]:
    """
    Heuristic shadow graph index generator.
    Returns two sets: target_nodes (size k) and potential_nodes (neighbors around target nodes).
    """
    num_nodes = g.number_of_nodes()
    k = max(1, min(k, num_nodes))
    target_nodes = random.sample(range(num_nodes), k)
    # collect neighbors up to 2 hops around the target nodes
    neigh = set(target_nodes)
    src, dst = g.edges()
    src = src.tolist()
    dst = dst.tolist()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)
    for u in list(target_nodes):
        for v in adj[u]:
            neigh.add(v)
            for w in adj[v]:
                neigh.add(w)
    # potential nodes are neighbors that are not target nodes
    potential_nodes = list(sorted(set(neigh) - set(target_nodes)))
    # if too many, sample a multiple of k
    max_size = min(num_nodes, extra * k if extra * k > k else k)
    if len(potential_nodes) > max_size:
        potential_nodes = random.sample(potential_nodes, max_size)
    return list(target_nodes), potential_nodes


def _safe_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_attack2_generated_graph(dataset_name: str, default_nodes: int) -> Tuple[
    dgl.DGLGraph, torch.Tensor, Optional[List[int]]]:
    """
    Try to load an attack-2 pre-generated graph. If files are missing, fall back to
    an on-the-fly Erdos–Rényi graph with random features. Returns (graph, features, selected_indices).
    """
    base = os.path.join(_safe_dir(), "data", "attack2_generated_graph", dataset_name)
    graph_label = os.path.join(base, "graph_label.txt")
    selected_idx = os.path.join(base, "selected_index.txt")
    if os.path.exists(graph_label):
        # we only need the number of nodes; reconstruct a random graph and random features
        try:
            with open(graph_label, "r") as f:
                n = sum(1 for _ in f)
            num_nodes = max(1, n)
        except Exception:
            num_nodes = max(1, default_nodes)
        g = erdos_renyi_graph(num_nodes, p=0.05)
        return g, None, None
    else:
        g = erdos_renyi_graph(default_nodes, p=0.05)
        return g, None, None


def load_attack3_shadow_indices(dataset_name: str, g: dgl.DGLGraph, k: int) -> Tuple[List[int], List[int]]:
    """
    Try to load shadow graph indices from disk; if not found, generate heuristically.
    """
    base = os.path.join(_safe_dir(), "data", "attack3_shadow_graph", dataset_name)
    target_path = os.path.join(base, "target_graph_index.txt")
    if os.path.exists(target_path):
        try:
            with open(target_path, "r") as f:
                target_nodes = [int(x.strip()) for x in f if len(x.strip()) > 0]
        except Exception:
            target_nodes = None
    else:
        target_nodes = None

    potential_nodes = None
    if os.path.isdir(base):
        for fn in os.listdir(base):
            if fn.startswith("protential") and fn.endswith(".txt"):
                try:
                    with open(os.path.join(base, fn), "r") as f:
                        potential_nodes = [int(x.strip()) for x in f if len(x.strip()) > 0]
                except Exception:
                    potential_nodes = None
                break

    if target_nodes is None or potential_nodes is None:
        t, p = random_shadow_indices(g, k)
        return t, p
    return target_nodes, potential_nodes


class _MEABase(BaseAttack):
    """
    Base class for MEA family attacks. This class handles the target model training,
    metric bookkeeping, and utility helpers. Subclasses only need to decide which
    training indices and which graph to use for the surrogate.
    """
    supported_api_types = {"dgl"}

    def __init__(self, dataset, attack_x_ratio: float, attack_a_ratio: float, model_path: Optional[str] = None):
        super().__init__(dataset, attack_x_ratio, model_path)

        self.dataset = dataset
        self.graph: dgl.DGLGraph = dataset.graph_data.to(self.device)
        self.features: torch.Tensor = self.graph.ndata['feat']
        self.labels: torch.Tensor = dataset.graph_data.ndata['label']
        self.train_mask: torch.Tensor = dataset.graph_data.ndata['train_mask']
        self.test_mask: torch.Tensor = dataset.graph_data.ndata['test_mask']

        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # budget based on the availability of features X and adjacency A
        self.attack_x_ratio = float(attack_x_ratio)
        self.attack_a_ratio = float(attack_a_ratio)
        self.attack_node_num = max(1, int(self.num_nodes * max(self.attack_x_ratio, self.attack_a_ratio)))

        # target model
        if model_path is None:
            self._train_target_model()
        else:
            self._load_model(model_path)

    def _train_target_model(self):
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)

        self.net1.train()
        for _ in range(200):
            opt.zero_grad()
            logits = self.net1(self.graph, self.features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            opt.step()
        self.net1.eval()

    def _load_model(self, model_path: str):
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.net1.load_state_dict(state)
        self.net1.eval()

    # ---------- core utilities ----------
    def _query_target(self, g: dgl.DGLGraph, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        start = time.time()
        with torch.no_grad():
            logits = self.net1(g, x)
        return logits, time.time() - start

    def _train_surrogate(self, g: dgl.DGLGraph, x: torch.Tensor, train_idx: torch.Tensor, y_train: torch.Tensor,
                         epochs: int = 200, lr: float = 0.01) -> Tuple[nn.Module, float]:
        model = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # boolean mask
        if y_train.dim() > 1:
            y_train = y_train.argmax(dim=1)
        mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=self.device)
        mask[train_idx] = True

        start = time.time()
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(g, x)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[mask], y_train[mask])
            loss.backward()
            opt.step()
        train_time = time.time() - start
        model.eval()
        return model, train_time

    def _compute_metrics(self, surrogate: nn.Module, metric: AttackMetric, metric_comp: AttackCompMetric):
        # target inference
        g = self.graph
        x = self.features
        y = self.labels
        mask = self.test_mask

        t0 = time.time()
        with torch.no_grad():
            logits_v = self.net1(g, x)
        metric_comp.update(inference_target_time=time.time() - t0)
        y_target = logits_v.argmax(dim=1)

        # surrogate inference
        t0 = time.time()
        with torch.no_grad():
            logits_s = surrogate(g, x)
        metric_comp.update(inference_surrogate_time=time.time() - t0)
        y_pred = logits_s.argmax(dim=1)

        metric.update(y_pred[mask], y[mask], y_target[mask])

    # ---------- template method ----------
    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        """
        Subclasses must implement this method to
        1) build a graph g_att and features x_att for training,
        2) pick a list of training indices idx_train of length attack_node_num,
        3) query the target for labels on idx_train and train a surrogate,
        and then return filled metrics objects.
        """
        raise NotImplementedError

    def attack(self, *args, **kwargs):
        metric = AttackMetric()
        metric_comp = AttackCompMetric()
        start_all = time.time()

        # delegate to subclass implementation
        metric, metric_comp = self._attack_impl()

        # finalize
        metric_comp.update(attack_time=time.time() - start_all)
        metric_comp.end()

        return metric.compute(), metric_comp.compute()


# ----------------------- concrete attacks -----------------------

class ModelExtractionAttack0(_MEABase):
    """
    Attack-0: Random-node label-only extraction on the original graph.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        # sample nodes to query
        idx_train = random.sample(range(self.num_nodes), self.attack_node_num)
        idx_train_t = torch.tensor(idx_train, device=self.device)

        # query target
        logits_v, q_time = self._query_target(self.graph, self.features)
        y_pseudo = logits_v.argmax(dim=1)
        metric_comp.update(query_target_time=q_time)

        # train surrogate on original graph but only using queried nodes
        surrogate, t_train = self._train_surrogate(self.graph, self.features, idx_train_t, y_pseudo)
        metric_comp.update(train_surrogate_time=t_train)

        # evaluate on real test set
        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack1(_MEABase):
    """
    Attack-1: Degree-based sampling of query nodes on the original graph.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        g = self.graph
        deg = g.in_degrees().cpu().tolist()
        order = sorted(range(self.num_nodes), key=lambda i: deg[i], reverse=True)
        idx_train = order[:self.attack_node_num]
        idx_train_t = torch.tensor(idx_train, device=self.device)

        logits_v, q_time = self._query_target(self.graph, self.features)
        y_pseudo = logits_v.argmax(dim=1)
        metric_comp.update(query_target_time=q_time)

        surrogate, t_train = self._train_surrogate(self.graph, self.features, idx_train_t, y_pseudo)
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack2(_MEABase):
    """
    Attack-2: Data-free extraction on a synthetic graph with random features.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        # build synthetic graph
        syn_g, _, _ = load_attack2_generated_graph(
            getattr(self.dataset, 'dataset_name', getattr(self.dataset, 'name', 'default')),
            default_nodes=self.attack_node_num
        )
        syn_g = syn_g.to(self.device)
        syn_x = torch.randn(syn_g.number_of_nodes(), self.num_features, device=self.device)

        # query target on synthetic inputs
        logits_v, q_time = self._query_target(syn_g, syn_x)
        y_pseudo = logits_v.argmax(dim=1)
        metric_comp.update(query_target_time=q_time)

        # use all nodes in synthetic graph for training
        idx_train_t = torch.arange(syn_g.number_of_nodes(), device=self.device)
        surrogate, t_train = self._train_surrogate(syn_g, syn_x, idx_train_t, y_pseudo)
        metric_comp.update(train_surrogate_time=t_train)

        # evaluate on real test set
        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack3(_MEABase):
    """
    Attack-3: Shadow-graph extraction. Train on a subgraph induced by a
    set of target nodes and their neighbors (potential nodes).
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        dataset_name = getattr(self.dataset, 'dataset_name', getattr(self.dataset, 'name', 'default'))
        target_nodes, potential_nodes = load_attack3_shadow_indices(dataset_name, self.graph, self.attack_node_num)

        # training nodes are the union
        idx_train = list(sorted(set(target_nodes) | set(potential_nodes)))
        idx_train_t = torch.tensor(idx_train, device=self.device)

        sg = subgraph_from_nodes(self.graph, idx_train)
        x_sg = self.features[idx_train_t]

        # map back to subgraph index for labels
        logits_v_full, q_time = self._query_target(self.graph, self.features)
        metric_comp.update(query_target_time=q_time)
        y_pseudo_full = logits_v_full.argmax(dim=1)
        y_pseudo = y_pseudo_full[idx_train_t]

        # train on the shadow subgraph
        surrogate, t_train = self._train_surrogate(sg, x_sg, torch.arange(sg.number_of_nodes(), device=self.device),
                                                   y_pseudo)
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack4(_MEABase):
    """
    Attack-4: Cosine-similarity neighbor expansion. Start from random seeds and
    expand candidates by feature similarity to form the training subgraph.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        seeds = random.sample(range(self.num_nodes), max(1, self.attack_node_num // 4))
        # compute cosine similarity on CPU to save GPU memory
        x = self.features.detach().cpu()
        norm = x.norm(dim=1, keepdim=True) + 1e-12
        x_n = x / norm
        sims = torch.mm(x_n, x_n.t())
        # choose top-k neighbors for each seed
        cand = set(seeds)
        for s in seeds:
            topk = torch.topk(sims[s], k=min(self.num_nodes, self.attack_node_num)).indices.tolist()
            cand.update(topk)
        idx_train = list(sorted(cand))[:self.attack_node_num]
        idx_train_t = torch.tensor(idx_train, device=self.device)

        # query target on original graph to get labels for these nodes
        logits_v, q_time = self._query_target(self.graph, self.features)
        metric_comp.update(query_target_time=q_time)
        y_pseudo = logits_v.argmax(dim=1)

        sg = subgraph_from_nodes(self.graph, idx_train)
        x_sg = self.features[idx_train_t]
        surrogate, t_train = self._train_surrogate(sg, x_sg, torch.arange(sg.number_of_nodes(), device=self.device),
                                                   y_pseudo[idx_train_t])
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack5(_MEABase):
    """
    Attack-5: Variant of the shadow-graph attack that samples two candidate lists and
    trains on their union. If attack_6 index files are present (historical name),
    they will be used; otherwise we fall back to generated indices.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        dataset_name = getattr(self.dataset, 'dataset_name', getattr(self.dataset, 'name', 'default'))
        base = os.path.join(_safe_dir(), "data", "attack3_shadow_graph", dataset_name)
        f_a = os.path.join(base, "attack_6_sub_shadow_graph_index_attack_2.txt")
        f_b = os.path.join(base, "attack_6_sub_shadow_graph_index_attack_3.txt")

        a_idx, b_idx = None, None
        if os.path.exists(f_a):
            try:
                with open(f_a, "r") as f:
                    a_idx = [int(x.strip()) for x in f if len(x.strip()) > 0]
            except Exception:
                a_idx = None
        if os.path.exists(f_b):
            try:
                with open(f_b, "r") as f:
                    b_idx = [int(x.strip()) for x in f if len(x.strip()) > 0]
            except Exception:
                b_idx = None

        if a_idx is None or b_idx is None:
            t, p = random_shadow_indices(self.graph, self.attack_node_num, extra=3)
            a_idx = t
            b_idx = p

        idx_train = list(sorted(set(a_idx) | set(b_idx)))
        idx_train = idx_train[:max(self.attack_node_num, len(idx_train))]
        idx_train_t = torch.tensor(idx_train, device=self.device)

        logits_v, q_time = self._query_target(self.graph, self.features)
        metric_comp.update(query_target_time=q_time)
        y_pseudo = logits_v.argmax(dim=1)

        sg = subgraph_from_nodes(self.graph, idx_train)
        x_sg = self.features[idx_train_t]
        surrogate, t_train = self._train_surrogate(sg, x_sg, torch.arange(sg.number_of_nodes(), device=self.device),
                                                   y_pseudo[idx_train_t])
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp
