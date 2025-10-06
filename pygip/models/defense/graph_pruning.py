import torch
import torch.nn.functional as F
import dgl
import numpy as np
from dgl.nn.pytorch import GraphConv
from .base import BaseDefense

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GraphPruningDefense(BaseDefense):
    supported_api_types = {"dgl"}

    def __init__(self, dataset, attack_node_fraction, pruning_ratio=0.1, **kwargs):
        super().__init__(dataset, attack_node_fraction, **kwargs)
        self.pruning_ratio = pruning_ratio


    # A helper function to train a SimpleGCN model on a given graph.
    def _train_gcn_model(self, graph, epochs=100, lr=0.01, return_pred=False):
        graph = graph.to(self.device)
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        test_mask = graph.ndata['test_mask']
        
        model = SimpleGCN(self.num_features, 16, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            logits = model(graph, features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            pred = logits.argmax(1).cpu()
            test_acc = (pred[test_mask].to(labels.device) == labels[test_mask]).float().mean().item()

        if return_pred:
            return test_acc, pred
        return test_acc


    # Main method to execute the defense and calculate evaluation metrics.
    def defend(self):
        """
        Run baseline and pruned models and return ECA and Fidelity.
        """
        original_graph = dgl.add_self_loop(self.graph_data)
        baseline_acc, baseline_pred = self._train_gcn_model(original_graph, return_pred=True)

        num_edges_to_remove = int(original_graph.number_of_edges() * self.pruning_ratio)
        edges_removed = 0
        if num_edges_to_remove > 0:
            all_edge_ids = np.arange(original_graph.number_of_edges())
            edges_to_remove_ids = np.random.choice(all_edge_ids, num_edges_to_remove, replace=False)
            pruned_graph = dgl.remove_edges(original_graph, torch.tensor(edges_to_remove_ids, dtype=torch.int64))
            edges_removed = len(edges_to_remove_ids)
        else:
            pruned_graph = original_graph

        # Also add self-loops to the pruned graph for consistency.
        pruned_graph = dgl.add_self_loop(pruned_graph)

        # Train a new GCN model from scratch on the pruned graph.
        pruned_acc, pruned_pred = self._train_gcn_model(pruned_graph, return_pred=True)

        test_mask = pruned_graph.ndata['test_mask']
        labels = pruned_graph.ndata['label']

        eca = pruned_acc
        baseline_on_test = baseline_pred[test_mask]
        pruned_on_test = pruned_pred[test_mask]
        fidelity = float((baseline_on_test == pruned_on_test).float().mean().item())

        return {
            'ECA': eca,
            'Fidelity': fidelity,
            'baseline_accuracy': baseline_acc,
            'pruning_ratio': self.pruning_ratio,
            'edges_removed': edges_removed
        }

    def _load_model(self):
        pass

    def _train_surrogate_model(self):
        pass

