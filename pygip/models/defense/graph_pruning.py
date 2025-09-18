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

    def __init__(self, dataset, pruning_ratio=0.1, **kwargs):
        super().__init__(dataset, **kwargs)
        self.pruning_ratio = pruning_ratio

    def defend(self):
        """
        Main public method to execute the defense.
        It trains the defended model and returns its performance metrics.
        """
        defended_acc = self._train_defense_model()
        
        return {
            'defended_accuracy': defended_acc,
            'pruning_ratio': self.pruning_ratio
        }

    def _train_gcn_model(self, graph, epochs=100, lr=0.01):
        """Private helper to train a SimpleGCN model on a given graph."""
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
            pred = logits.argmax(1)
            accuracy = (pred[test_mask] == labels[test_mask]).float().mean()
        return accuracy.item()

    def _train_target_model(self):
        """Trains a baseline model on the original, unmodified graph."""
        print("   - Training baseline target model...")
        original_graph = dgl.add_self_loop(self.graph_data)
        accuracy = self._train_gcn_model(original_graph)
        return accuracy

    def _train_defense_model(self):
        """Prunes the graph and then trains a new model on the pruned graph."""
        print(f"   - Training defense model (Pruning ratio: {self.pruning_ratio*100:.1f}%)...")
        original_graph = self.graph_data
        num_edges_to_remove = int(original_graph.number_of_edges() * self.pruning_ratio)
        
        if num_edges_to_remove > 0:
            all_edge_ids = np.arange(original_graph.number_of_edges())
            edges_to_remove_ids = np.random.choice(all_edge_ids, num_edges_to_remove, replace=False)
            pruned_graph = dgl.remove_edges(original_graph, torch.tensor(edges_to_remove_ids, dtype=torch.int64))
        else:
            pruned_graph = original_graph

        pruned_graph = dgl.add_self_loop(pruned_graph)
        
        accuracy = self._train_gcn_model(pruned_graph)
        return accuracy

    def _load_model(self):
        pass

    def _train_surrogate_model(self):
        pass