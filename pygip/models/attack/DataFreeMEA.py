from abc import abstractmethod
import time

import dgl
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pygip.models.attack.base import BaseAttack
from pygip.models.nn import GCN, GraphSAGE  # Backbone architectures
from pygip.utils.metrics import AttackMetric, AttackCompMetric  # Consistent with AdvMEA


class GraphGenerator:
    def __init__(self, node_number, feature_number, label_number):
        self.node_number = node_number
        self.feature_number = feature_number
        self.label_number = label_number

    def generate(self):
        # Generate a random Erdős–Rényi graph and convert it to DGL
        g_nx = nx.erdos_renyi_graph(n=self.node_number, p=0.05)
        g_dgl = dgl.from_networkx(g_nx)
        # Random node features
        features = torch.randn((self.node_number, self.feature_number))
        return g_dgl, features


class DFEAAttack(BaseAttack):
    supported_api_types = {"dgl"}

    # Use unified attack_x_ratio and attack_a_ratio
    def __init__(self, dataset, attack_x_ratio, attack_a_ratio, model_path=None):
        super().__init__(dataset, attack_x_ratio, model_path)
        # Load graph data
        self.graph = dataset.graph_data.to(self.device)
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata.get('val_mask', None)
        self.test_mask = self.graph.ndata['test_mask']
        # Meta data
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes

        # Use the maximum of the two visibility ratios as the budget; 
        # if both are 0, fallback to a small default value to avoid zero-size graph
        ratio_budget = max(float(attack_x_ratio), float(attack_a_ratio))
        if ratio_budget <= 0.0:
            ratio_budget = 0.05
        self.attack_node_number = max(1, int(dataset.num_nodes * ratio_budget))

        # Generate synthetic graph and features for surrogate training (data-free)
        self.generator = GraphGenerator(
            node_number=self.attack_node_number,
            feature_number=self.feature_number,
            label_number=self.label_number
        )
        self.synthetic_graph, self.synthetic_features = self.generator.generate()
        self.synthetic_graph = self.synthetic_graph.to(self.device)
        self.synthetic_features = self.synthetic_features.to(self.device)

        if model_path is None:
            self._train_target_model()
        else:
            self._load_model(model_path)

    def _train_target_model(self):
        # Train the victim GCN model on real data
        model = GCN(self.feature_number, self.label_number).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4
        )
        model.train()
        # Dataset-specific label shaping
        name = getattr(self.dataset, 'dataset_name', None) or getattr(self.dataset, 'name', None)
        epochs = 200
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(self.graph, self.features)
            labels = self.labels.squeeze() if name == 'ogb-arxiv' else self.labels
            loss = F.nll_loss(
                F.log_softmax(logits[self.train_mask], dim=1),
                labels[self.train_mask]
            )
            loss.backward()
            optimizer.step()

        model.eval()
        self.model = model

    def _load_model(self, model_path):
        # Load a pretrained victim model
        model = GCN(self.feature_number, self.label_number)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def _forward(self, model, graph, features):
        # Forward wrapper for GCN and GraphSAGE
        if isinstance(model, GraphSAGE):
            # GraphSAGE expects a two-block input list
            return model([graph, graph], features)
        return model(graph, features)

    def _evaluate_on_real_test(self, surrogate, metric: AttackMetric, metric_comp: AttackCompMetric):
        """Evaluate the surrogate on the real test set and update metrics"""
        g = self.graph
        x = self.features
        y = self.labels
        mask = self.test_mask

        # Victim inference time
        t0 = time.time()
        with torch.no_grad():
            logits_v = self._forward(self.model, g, x)
        metric_comp.update(inference_target_time=(time.time() - t0))
        labels_query = logits_v.argmax(dim=1)

        # Surrogate inference time
        t0 = time.time()
        with torch.no_grad():
            logits_s = self._forward(surrogate, g, x)
        metric_comp.update(inference_surrogate_time=(time.time() - t0))
        preds_s = logits_s.argmax(dim=1)

        # Update performance metrics: accuracy and fidelity
        metric.update(preds_s[mask], y[mask], labels_query[mask])

    @abstractmethod
    def attack(self):
        pass


class DFEATypeI(DFEAAttack):
    """
    Type I: Uses victim outputs + gradients for surrogate training.
    """

    def attack(self):
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        attack_start = time.time()
        surrogate = GCN(self.feature_number, self.label_number).to(self.device)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01)

        # Surrogate training time
        train_surrogate_start = time.time()
        # Victim query time
        total_query_time = 0.0

        for _ in tqdm(range(200)):
            surrogate.train()
            optimizer.zero_grad()
            # Victim logits (no gradient), count query time
            t_q = time.time()
            with torch.no_grad():
                logits_v = self._forward(
                    self.model, self.synthetic_graph, self.synthetic_features
                )
            total_query_time += (time.time() - t_q)

            logits_s = self._forward(
                surrogate, self.synthetic_graph, self.synthetic_features
            )
            loss = F.kl_div(
                F.log_softmax(logits_s, dim=1),
                F.softmax(logits_v, dim=1),
                reduction='batchmean'
            )
            loss.backward()
            optimizer.step()

        train_surrogate_end = time.time()

        surrogate.eval()
        self._evaluate_on_real_test(surrogate, metric, metric_comp)

        metric_comp.end()
        metric_comp.update(
            attack_time=(time.time() - attack_start),
            query_target_time=total_query_time,
            train_surrogate_time=(train_surrogate_end - train_surrogate_start),
        )
        res = metric.compute()
        res_comp = metric_comp.compute()
        return res, res_comp


class DFEATypeII(DFEAAttack):
    """
    Type II: Uses victim outputs only (hard labels).
    """

    def attack(self):
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        attack_start = time.time()
        surrogate = GraphSAGE(self.feature_number, 16, self.label_number).to(self.device)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01)

        train_surrogate_start = time.time()
        total_query_time = 0.0

        for _ in tqdm(range(200)):
            surrogate.train()
            optimizer.zero_grad()
            # Victim pseudo labels
            t_q = time.time()
            with torch.no_grad():
                logits_v = self._forward(
                    self.model, self.synthetic_graph, self.synthetic_features
                )
            total_query_time += (time.time() - t_q)
            pseudo = logits_v.argmax(dim=1)

            logits_s = self._forward(
                surrogate, self.synthetic_graph, self.synthetic_features
            )
            loss = F.cross_entropy(logits_s, pseudo)
            loss.backward()
            optimizer.step()

        train_surrogate_end = time.time()

        surrogate.eval()
        self._evaluate_on_real_test(surrogate, metric, metric_comp)

        metric_comp.end()
        metric_comp.update(
            attack_time=(time.time() - attack_start),
            query_target_time=total_query_time,
            train_surrogate_time=(train_surrogate_end - train_surrogate_start),
        )
        res = metric.compute()
        res_comp = metric_comp.compute()
        return res, res_comp


class DFEATypeIII(DFEAAttack):
    """
    Type III: Two surrogates with victim supervision + consistency.
    """

    def attack(self):
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        attack_start = time.time()
        s1 = GCN(self.feature_number, self.label_number).to(self.device)
        s2 = GraphSAGE(self.feature_number, 16, self.label_number).to(self.device)
        opt1 = torch.optim.Adam(s1.parameters(), lr=0.01)
        opt2 = torch.optim.Adam(s2.parameters(), lr=0.01)

        train_surrogate_start = time.time()
        total_query_time = 0.0

        for _ in tqdm(range(200)):
            s1.train()
            s2.train()
            opt1.zero_grad()
            opt2.zero_grad()
            # Victim pseudo-labels
            t_q = time.time()
            with torch.no_grad():
                logits_v = self._forward(
                    self.model, self.synthetic_graph, self.synthetic_features
                )
            total_query_time += (time.time() - t_q)
            pseudo_v = logits_v.argmax(dim=1)
            # Surrogate predictions
            l1 = self._forward(s1, self.synthetic_graph, self.synthetic_features)
            l2 = self._forward(s2, self.synthetic_graph, self.synthetic_features)
            # Loss: supervised + consistency
            loss1 = F.cross_entropy(l1, pseudo_v)
            loss2 = F.cross_entropy(l2, pseudo_v)
            cons = F.mse_loss(l1, l2)
            total = loss1 + loss2 + 0.5 * cons
            total.backward()
            opt1.step()
            opt2.step()

        train_surrogate_end = time.time()

        # Use s1 as the final surrogate for evaluation
        s1.eval()
        self._evaluate_on_real_test(s1, metric, metric_comp)

        metric_comp.end()
        metric_comp.update(
            attack_time=(time.time() - attack_start),
            query_target_time=total_query_time,
            train_surrogate_time=(train_surrogate_end - train_surrogate_start),
        )
        res = metric.compute()
        res_comp = metric_comp.compute()
        return res, res_comp


# Factory mapping of attack names to classes
ATTACK_FACTORY = {
    "ModelExtractionAttack0": DFEATypeI,
    "ModelExtractionAttack1": DFEATypeI,
    "ModelExtractionAttack2": DFEATypeII,
    "ModelExtractionAttack3": DFEATypeIII
}
