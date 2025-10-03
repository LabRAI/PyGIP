import time

import dgl
import torch
import networkx as nx
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from pygip.models.defense.base import BaseDefense
from pygip.models.nn import GCN
from pygip.utils.metrics import GraphNeuralNetworkMetric, DefenseCompMetric, DefenseMetric


class SurviveWM(BaseDefense):
    supported_api_types = {"dgl"}

    def __init__(self, dataset, defense_ratio: float = 0.1, model_path=None):
        super().__init__(dataset, defense_ratio)
        # load graph data
        self.dataset = dataset
        self.graph_dataset = dataset.graph_data
        self.graph_data = dataset.graph_data.to(device=self.device)
        self.model_path = model_path
        self.graph = self.graph_data
        self.features = self.graph_data.ndata['feat']
        self.labels = self.graph_data.ndata['label']
        self.train_mask = self.graph_data.ndata['train_mask']
        self.test_mask = self.graph_data.ndata['test_mask']

        # load meta data
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes

        # params
        self.defense_ratio = defense_ratio

    def _load_model(self):
        """
        Load a pre-trained model.
        """
        assert self.model_path is not None, "Please provide a pre-trained model"

        # Create the model
        self.net1 = GCN(self.feature_number, self.label_number).to(self.device)

        # Load the saved state dict
        self.net1.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # Set to evaluation mode
        self.net1.eval()

    def _to_cpu(self, tensor):
        """
        Safely move tensor to CPU for NumPy operations
        """
        if tensor.is_cuda:
            return tensor.cpu()
        return tensor

    # === Soft Nearest Neighbor Loss ===
    def snn_loss(self, x, y, T=0.5):
        x = F.normalize(x, p=2, dim=1)
        dist_matrix = torch.cdist(x, x, p=2) ** 2
        eye = torch.eye(len(x), device=self.device).bool()
        sim = torch.exp(-dist_matrix / T)
        mask_same = y.unsqueeze(1) == y.unsqueeze(0)
        sim = sim.masked_fill(eye, 0)
        denom = sim.sum(1)
        nom = (sim * mask_same.float()).sum(1)
        loss = -torch.log(nom / (denom + 1e-10) + 1e-10).mean()
        return loss

    # === Trigger Graph Generator ===
    def generate_key_graph(self, num_nodes=None, edge_prob=None):
        if num_nodes is None:
            num_nodes = max(5, int(self.dataset.num_nodes * self.defense_ratio))
        if edge_prob is None:
            edge_prob = min(0.5, self.defense_ratio * 3)

        trigger = nx.erdos_renyi_graph(num_nodes, edge_prob)
        edge_index = torch.tensor(list(trigger.edges), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        x = torch.randn((num_nodes, self.feature_number), device=self.device) * 0.1
        label = torch.randint(0, self.label_number, (num_nodes,), device=self.device)
        return Data(x=x, edge_index=edge_index.to(self.device), y=label)

    # === Combine base and trigger ===
    def combine_with_trigger(self, base_graph, base_features, base_labels, trigger_data):
        # Convert DGL graph to edge_index format
        src, dst = base_graph.edges()
        base_edge_index = torch.stack([src, dst], dim=0)

        x = torch.cat([base_features, trigger_data.x], dim=0)
        edge_index = torch.cat([base_edge_index, trigger_data.edge_index + base_features.size(0)], dim=1)
        y = torch.cat([base_labels, trigger_data.y], dim=0)

        # Create DGL graph from combined data
        src_combined, dst_combined = edge_index[0], edge_index[1]
        combined_graph = dgl.graph((src_combined, dst_combined), num_nodes=x.size(0)).to(self.device)

        # **FIX: Add self-loops to handle zero in-degree nodes**
        combined_graph = dgl.add_self_loop(combined_graph)

        combined_graph.ndata['feat'] = x.to(self.device)

        return combined_graph, y.to(self.device)

    def train_with_snnl(self, model, graph, features, labels, train_mask, optimizer, T=0.5, alpha=0.1):
        model.train()
        optimizer.zero_grad()
        out = model(graph, features)
        loss_nll = F.nll_loss(F.log_softmax(out, dim=1)[train_mask], labels[train_mask])
        snnl = self.snn_loss(out[train_mask], labels[train_mask], T=T)
        loss = loss_nll - alpha * snnl
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def verify_watermark(self, model, trigger_graph, trigger_labels):
        model.eval()
        out = model(trigger_graph, trigger_graph.ndata['feat'])
        pred = out.argmax(dim=1)
        return (pred == trigger_labels).float().mean().item()

    def compute_metrics(self, y_true, y_pred, y_score=None):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'auroc': roc_auc_score(y_true, y_score, multi_class='ovo') if y_score is not None else None
        }

    def defend(self):
        """Execute the SurviveWM defense."""
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================SurviveWM Defense====================")

        # If model wasn't trained yet, train it
        if not hasattr(self, 'model_trained'):
            self.train_target_model(metric_comp)

        # Evaluate the defended model
        preds = self.evaluate_model(self.watermarked_model)
        inference_s = time.time()
        wm_preds = self.verify_watermark_model(self.watermarked_model)
        inference_e = time.time()

        # metric
        metric = DefenseMetric()
        metric.update(preds, self.labels[self.test_mask])
        metric.update_wm(wm_preds, self.trigger_data.y)
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        res_comp = metric_comp.compute()

        return res, res_comp

    def train_target_model(self, metric_comp: DefenseCompMetric):
        """Train the target model with watermark injection."""
        defense_s = time.time()

        # Generate trigger and train watermarked model (defense mechanism)
        self.trigger_data = self.generate_key_graph().to(self.device)
        self.watermarked_model = self._train_watermarked_model()
        self.model_trained = True

        defense_e = time.time()
        metric_comp.update(defense_time=(defense_e - defense_s))
        return self.watermarked_model

    def evaluate_model(self, model):
        """Evaluate model performance on downstream task"""
        model.eval()
        with torch.no_grad():
            test_graph = dgl.add_self_loop(self.graph)
            logits = model(test_graph, self.features)
            pred = logits.argmax(dim=1)
            preds = pred[self.test_mask].cpu()
        return preds

    def verify_watermark_model(self, model):
        """Verify watermark success rate"""
        model.eval()
        with torch.no_grad():
            # Use the stored trigger graph if available
            if hasattr(self, 'trigger_graph'):
                trigger_graph = self.trigger_graph
            else:
                # Create trigger graph for verification
                trigger_src, trigger_dst = self.trigger_data.edge_index[0], self.trigger_data.edge_index[1]
                trigger_graph = dgl.graph((trigger_src, trigger_dst), num_nodes=self.trigger_data.num_nodes).to(
                    self.device)
                trigger_graph = dgl.add_self_loop(trigger_graph)
                trigger_graph.ndata['feat'] = self.trigger_data.x.to(self.device)

            out = model(trigger_graph, trigger_graph.ndata['feat'])
            wm_preds = out.argmax(dim=1).cpu()
        return wm_preds

    def _train_watermarked_model(self):
        """Helper function to train the watermarked model"""
        print("Training watermarked model...")

        # Combine base graph with trigger
        combined_graph, combined_labels = self.combine_with_trigger(
            self.graph, self.features, self.labels, self.trigger_data)

        # Create train mask for combined data (include trigger nodes in training)
        base_train_mask = self.train_mask
        trigger_train_mask = torch.ones(self.trigger_data.num_nodes, dtype=torch.bool, device=self.device)
        combined_train_mask = torch.cat([base_train_mask, trigger_train_mask])

        # Create test mask for combined data (exclude trigger nodes from testing)
        base_test_mask = self.test_mask
        trigger_test_mask = torch.zeros(self.trigger_data.num_nodes, dtype=torch.bool, device=self.device)
        combined_test_mask = torch.cat([base_test_mask, trigger_test_mask])

        # Create watermarked model
        watermarked_model = GCN(self.feature_number, self.label_number).to(self.device)
        optimizer = torch.optim.Adam(watermarked_model.parameters(), lr=0.01, weight_decay=5e-4)

        # Create trigger graph for watermark verification
        trigger_src, trigger_dst = self.trigger_data.edge_index[0], self.trigger_data.edge_index[1]
        trigger_graph = dgl.graph((trigger_src, trigger_dst), num_nodes=self.trigger_data.num_nodes).to(self.device)
        trigger_graph = dgl.add_self_loop(trigger_graph)
        trigger_graph.ndata['feat'] = self.trigger_data.x.to(self.device)

        # Store trigger graph for later use
        self.trigger_graph = trigger_graph

        dur = []
        best_performance_metrics = GraphNeuralNetworkMetric()

        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            # Train with SNNL
            loss = self.train_with_snnl(
                watermarked_model, combined_graph, combined_graph.ndata['feat'],
                combined_labels, combined_train_mask, optimizer)

            if epoch >= 3:
                dur.append(time.time() - t0)

            # Evaluate periodically
            if epoch % 20 == 0:
                watermarked_model.eval()
                with torch.no_grad():
                    # Test on original graph (ensure it has self-loops)
                    test_graph = dgl.add_self_loop(self.graph)

                    logits = watermarked_model(test_graph, self.features)
                    pred = logits.argmax(dim=1)
                    test_acc = (pred[self.test_mask] == self.labels[self.test_mask]).float().mean()

                    # Verify watermark
                    wm_acc = self.verify_watermark(watermarked_model, trigger_graph, self.trigger_data.y)

                    print(f"Epoch {epoch}, Test Acc: {test_acc.item():.4f}, Watermark Acc: {wm_acc:.4f}")

        # Final evaluation
        watermarked_model.eval()
        with torch.no_grad():
            # Evaluate on test set (ensure graph has self-loops)
            test_graph = dgl.add_self_loop(self.graph)

            logits = watermarked_model(test_graph, self.features)
            pred = logits.argmax(dim=1)
            probs = F.softmax(logits, dim=1)

            test_metrics = self.compute_metrics(
                self._to_cpu(self.labels[self.test_mask]).numpy(),
                self._to_cpu(pred[self.test_mask]).numpy(),
                self._to_cpu(probs[self.test_mask]).numpy()
            )

            # Verify watermark
            wm_acc = self.verify_watermark(watermarked_model, trigger_graph, self.trigger_data.y)

            print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Final Test F1: {test_metrics['f1']:.4f}")
            print(f"Final Test Precision: {test_metrics['precision']:.4f}")
            print(f"Final Test Recall: {test_metrics['recall']:.4f}")
            print(f"Final Watermark Accuracy: {wm_acc:.4f}")

        # Store final metrics for later use
        self.final_test_metrics = test_metrics
        self.final_wm_acc = wm_acc

        return watermarked_model
