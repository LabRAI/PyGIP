import random
import warnings
import time

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

from pygip.models.nn.backbones import GCN
from .base import BaseAttack
from pygip.utils.metrics import AttackMetric, AttackCompMetric  # align with AdvMEA

warnings.filterwarnings('ignore')


class DGLEdgePredictor(nn.Module):
    """DGL version of edge prediction module."""

    def __init__(self, input_dim, hidden_dim, num_classes, device):
        super(DGLEdgePredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device

        # Use the same GCN backbone as the target model to obtain node embeddings.
        self.gnn = GCN(input_dim, hidden_dim)
        self.node_classifier = nn.Linear(hidden_dim, num_classes)

        # Edge prediction head.
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, graph, features):
        # Compute node embeddings then logits for node classification.
        node_embeddings = self.gnn(graph, features)
        node_logits = self.node_classifier(node_embeddings)
        return node_embeddings, node_logits

    def predict_edges(self, node_embeddings, node_pairs):
        """Predict edge existence probability for a list of node index pairs."""
        if len(node_pairs) == 0:
            return torch.tensor([], device=self.device)

        node_pairs = torch.tensor(node_pairs, device=self.device)
        src_embeddings = node_embeddings[node_pairs[:, 0]]
        dst_embeddings = node_embeddings[node_pairs[:, 1]]

        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        edge_probs = self.edge_predictor(edge_features).squeeze()
        return edge_probs


class DGLSurrogateModel(nn.Module):
    """DGL version of surrogate model."""

    def __init__(self, input_dim, num_classes, model_type='GCN'):
        super(DGLSurrogateModel, self).__init__()
        self.model_type = model_type
        if model_type == 'GCN':
            self.gnn = GCN(input_dim, num_classes)
        else:
            # Default to GCN; can be extended to other backbones.
            self.gnn = GCN(input_dim, num_classes)

    def forward(self, graph, features):
        return self.gnn(graph, features)


class RealisticAttack(BaseAttack):
    """DGL-based GNN model extraction attack with updated metrics API."""
    supported_api_types = {"dgl"}
    supported_datasets = {}

    def __init__(self, dataset, attack_x_ratio: float, attack_a_ratio: float, model_path: str = None,
                 hidden_dim: int = 64, threshold_s: float = 0.7, threshold_a: float = 0.5):
        # Keep BaseAttack initialization contract; store ratios for this attack.
        super().__init__(dataset, attack_x_ratio, model_path)

        self.attack_x_ratio = float(attack_x_ratio)
        self.attack_a_ratio = float(attack_a_ratio)

        self.hidden_dim = hidden_dim
        self.threshold_s = threshold_s  # Cosine similarity threshold
        self.threshold_a = threshold_a  # Edge prediction threshold

        # Determine the number of queried nodes by the availability ratios.
        ratio_budget = max(self.attack_x_ratio, self.attack_a_ratio)
        if ratio_budget <= 0.0:
            ratio_budget = 0.05  # small default to avoid zero queries
        self.attack_node_number = max(1, int(self.num_nodes * ratio_budget))

        # Graph tensors
        self.graph_data = self.graph_data.to(self.device)
        self.graph = self.graph_data
        self.features = self.graph.ndata['feat']

        # Initialize edge predictor and surrogate model.
        self.edge_predictor = DGLEdgePredictor(
            self.num_features, hidden_dim, self.num_classes, self.device
        ).to(self.device)
        self.surrogate_model = DGLSurrogateModel(
            self.num_features, self.num_classes
        ).to(self.device)

        # Target model used to simulate black-box responses.
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)

        # Optimizers
        self.optimizer_edge = optim.Adam(self.edge_predictor.parameters(), lr=0.01, weight_decay=5e-4)
        self.optimizer_surrogate = optim.Adam(self.surrogate_model.parameters(), lr=0.01, weight_decay=5e-4)

        print(f"Initialized attack on {dataset.dataset_name} dataset")
        print(f"Nodes: {self.num_nodes}, Features: {self.num_features}, Classes: {self.num_classes}")
        print(f"Attack nodes: {self.attack_node_number} (x_ratio={self.attack_x_ratio:.2f}, a_ratio={self.attack_a_ratio:.2f})")

    def simulate_target_model_queries(self, query_nodes, error_rate=0.15):
        """Query the target model for labels on query_nodes and introduce a small error rate."""
        self.net1.eval()
        with torch.no_grad():
            logits = self.net1(self.graph, self.features)
            predictions = F.log_softmax(logits, dim=1).argmax(dim=1)

        predicted_labels = predictions[query_nodes].clone()

        # Flip a portion of labels to simulate noise in responses.
        num_errors = int(len(predicted_labels) * error_rate)
        if num_errors > 0:
            error_indices = random.sample(range(len(predicted_labels)), num_errors)
            for idx in error_indices:
                wrong_label = random.randint(0, self.num_classes - 1)
                predicted_labels[idx] = wrong_label
        return predicted_labels

    def compute_cosine_similarity(self, features):
        """Compute cosine similarity of node features."""
        features_np = features.cpu().detach().numpy()
        similarity_matrix = cosine_similarity(features_np)
        return torch.tensor(similarity_matrix, dtype=torch.float32, device=self.device)

    def generate_candidate_edges(self, labeled_nodes, unlabeled_nodes):
        """Generate candidate edges based on feature cosine similarity threshold."""
        similarity_matrix = self.compute_cosine_similarity(self.features)
        candidate_edges = []
        for u_node in unlabeled_nodes:
            for l_node in labeled_nodes:
                if similarity_matrix[u_node, l_node] > self.threshold_s:
                    candidate_edges.append([u_node, l_node])
        print(f"Generated {len(candidate_edges)} candidate edges based on cosine similarity")
        return candidate_edges

    def train_edge_predictor(self, labeled_nodes, predicted_labels, epochs=100):
        """Train the auxiliary edge prediction model."""
        print("Training edge predictor...")
        self.edge_predictor.train()

        # Create node labels tensor; only queried nodes are labeled.
        train_labels = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        train_labels[labeled_nodes] = predicted_labels

        for epoch in range(epochs):
            self.optimizer_edge.zero_grad()

            # Forward pass through edge predictor
            node_embeddings, node_logits = self.edge_predictor(self.graph, self.features)

            # Node classification loss (supervised on labeled nodes)
            labeled_mask = train_labels != -1
            if labeled_mask.sum() > 0:
                node_loss = F.cross_entropy(node_logits[labeled_mask], train_labels[labeled_mask])
            else:
                node_loss = torch.tensor(0.0, device=self.device)

            # Positive and negative edge samples
            src_nodes, dst_nodes = self.graph.edges()
            positive_pairs = list(zip(src_nodes.cpu().numpy(), dst_nodes.cpu().numpy()))

            pos_edge_probs = self.edge_predictor.predict_edges(node_embeddings, positive_pairs)
            pos_loss = -torch.log(pos_edge_probs + 1e-15).mean()

            negative_pairs = []
            num_neg_samples = min(len(positive_pairs), 1000)
            for _ in range(num_neg_samples):
                src = random.randint(0, self.num_nodes - 1)
                dst = random.randint(0, self.num_nodes - 1)
                if src != dst and not self.graph_data.has_edges_between(src, dst):
                    negative_pairs.append([src, dst])

            if negative_pairs:
                neg_edge_probs = self.edge_predictor.predict_edges(node_embeddings, negative_pairs)
                neg_loss = -torch.log(1 - neg_edge_probs + 1e-15).mean()
            else:
                neg_loss = torch.tensor(0.0, device=self.device)

            total_loss = node_loss + 0.5 * (pos_loss + neg_loss)
            total_loss.backward()
            self.optimizer_edge.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: total={total_loss.item():.4f}, "
                      f"node={node_loss.item():.4f}, edge={(pos_loss + neg_loss).item():.4f}")

    def add_potential_edges(self, candidate_edges, labeled_nodes):
        """Add potential edges whose predicted probability exceeds the threshold."""
        if not candidate_edges:
            return self.graph

        print("Predicting edge weights and adding potential edges...")
        self.edge_predictor.eval()
        with torch.no_grad():
            node_embeddings, _ = self.edge_predictor(self.graph, self.features)
            edge_probs = self.edge_predictor.predict_edges(node_embeddings, candidate_edges)

        selected_edges = []
        for i, (src, dst) in enumerate(candidate_edges):
            if edge_probs[i] > self.threshold_a:
                selected_edges.extend([(src, dst), (dst, src)])  # undirected

        print(f"Selected {len(selected_edges) // 2} potential edges to add")
        if selected_edges:
            enhanced_graph = dgl.add_edges(
                self.graph,
                [e[0] for e in selected_edges],
                [e[1] for e in selected_edges]
            )
            return enhanced_graph
        else:
            return self.graph

    def train_surrogate_model(self, enhanced_graph, labeled_nodes, predicted_labels, epochs=200):
        """Train the surrogate model on queried nodes and pseudo labels."""
        print("Training surrogate model...")
        self.surrogate_model.train()

        # Build training labels for queried nodes.
        train_labels = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        train_labels[labeled_nodes] = predicted_labels
        labeled_mask = train_labels != -1

        for epoch in range(epochs):
            self.optimizer_surrogate.zero_grad()
            logits = self.surrogate_model(enhanced_graph, self.features)

            if labeled_mask.sum() > 0:
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[labeled_mask], train_labels[labeled_mask])
                loss.backward()
                self.optimizer_surrogate.step()

                if epoch % 50 == 0:
                    print(f"Surrogate epoch {epoch:3d}, loss={loss.item():.4f}")

    def _evaluate_and_update_metrics(self, enhanced_graph, metric: AttackMetric, metric_comp: AttackCompMetric):
        """Evaluate surrogate against target on the real test set and update metric containers."""
        # Target inference
        t0 = time.time()
        with torch.no_grad():
            logits_target = self.net1(self.graph, self.features)
        metric_comp.update(inference_target_time=(time.time() - t0))
        target_preds = F.log_softmax(logits_target, dim=1).argmax(dim=1)

        # Surrogate inference
        t0 = time.time()
        with torch.no_grad():
            logits_surrogate = self.surrogate_model(enhanced_graph, self.features)
        metric_comp.update(inference_surrogate_time=(time.time() - t0))
        surrogate_preds = F.log_softmax(logits_surrogate, dim=1).argmax(dim=1)

        # Update performance metrics with ground truth and target predictions on test split.
        mask = self.graph_data.ndata['test_mask']
        labels = self.graph_data.ndata['label']
        metric.update(surrogate_preds[mask], labels[mask], target_preds[mask])

    def attack(self):
        """Execute the attack and return two JSON-like dicts: performance and computation metrics."""
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        print("=" * 60)
        print("Starting GNN Model Extraction Attack (Realistic)")
        print("=" * 60)

        attack_start = time.time()

        # Step 1: Randomly select query nodes according to the budget.
        all_nodes = list(range(self.num_nodes))
        labeled_nodes = random.sample(all_nodes, self.attack_node_number)
        unlabeled_nodes = [n for n in all_nodes if n not in labeled_nodes]
        print(f"Selected {len(labeled_nodes)} nodes for querying")

        # Step 2: Query the target model once for pseudo labels.
        t_q = time.time()
        predicted_labels = self.simulate_target_model_queries(labeled_nodes)
        query_time = time.time() - t_q
        metric_comp.update(query_target_time=query_time)
        print("Finished querying the target model")

        # Step 3: Generate candidate edges (feature similarity).
        candidate_edges = self.generate_candidate_edges(labeled_nodes, unlabeled_nodes)

        # Step 4: Train the auxiliary edge predictor (included in total attack time).
        self.train_edge_predictor(labeled_nodes, predicted_labels)

        # Step 5: Add potential edges to obtain an enhanced graph.
        enhanced_graph = self.add_potential_edges(candidate_edges, labeled_nodes)
        original_edges = self.graph_data.num_edges()
        enhanced_edges = enhanced_graph.num_edges()
        print(f"Enhanced graph: {original_edges} -> {enhanced_edges} edges (+{enhanced_edges - original_edges})")

        # Step 6: Train the surrogate model and record its training time.
        t_train_surr = time.time()
        self.train_surrogate_model(enhanced_graph, labeled_nodes, predicted_labels)
        train_surrogate_time = time.time() - t_train_surr
        metric_comp.update(train_surrogate_time=train_surrogate_time)

        # Step 7: One-shot evaluation and metrics update.
        self._evaluate_and_update_metrics(enhanced_graph, metric, metric_comp)

        # Finalize computation stats.
        metric_comp.end()
        metric_comp.update(attack_time=(time.time() - attack_start))

        # Return two JSON-like dicts as required by the new API.
        res = metric.compute()
        res_comp = metric_comp.compute()
        return res, res_comp
