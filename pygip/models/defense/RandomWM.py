import importlib
import time

import dgl
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.utils import erdos_renyi_graph
from dgl.dataloading import NeighborSampler, NodeCollator

from pygip.models.nn import GraphSAGE
from pygip.models.defense.base import BaseDefense
from pygip.utils.metrics import DefenseCompMetric, DefenseMetric


class RandomWM(BaseDefense):
    """
    A flexible defense implementation using watermarking to protect against
    model extraction attacks on graph neural networks.
    
    This class combines the functionalities from the original watermark.py:
    - Generating watermark graphs
    - Training models on original and watermark graphs
    - Merging graphs for testing
    - Evaluating effectiveness against attacks
    - Dynamic selection of attack methods
    """
    supported_api_types = {"dgl"}

    def __init__(self, dataset, defense_ratio=0.1, wm_node=50, pr=0.2, pg=0.2, attack_name=None):
        """
        Initialize the custom defense.
        
        Parameters
        ----------
        defense_ratio : float
        Defense strength (0-1): used to determine the number of watermark nodes and the attack node sampling scale
        wm_node : Optional[int]
        If specified, a fixed number of watermark nodes is used; otherwise, a dynamic calculation is performed based on defense_ratio * num_nodes
        pr : float
        Bernoulli probability of the watermark feature being 1
        pg : float
        Edge probability of the watermark graph
        attack_name : Optional[str]
        Attack class name (from models.attack)
        """
        super().__init__(dataset, defense_ratio)
        self.defense_ratio = defense_ratio
        self.attack_name = attack_name or "ModelExtractionAttack0"
        self.dataset = dataset
        self.graph = dataset.graph_data

        # Extract dataset properties
        self.node_number = dataset.num_nodes
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes
        self.attack_node_number = int(self.node_number * defense_ratio)

        # Watermark parameters
        self.wm_node = int(wm_node) if wm_node is not None else max(10, int(dataset.num_nodes * defense_ratio))
        self.pr = pr
        self.pg = pg

        # Extract features and labels
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        # Extract masks
        self.train_mask = self.graph.ndata['train_mask']
        self.test_mask = self.graph.ndata['test_mask']

        # Move tensors to self.device
        if self.device != 'cpu':
            self.graph = self.graph.to(self.device)
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
            self.train_mask = self.train_mask.to(self.device)
            self.test_mask = self.test_mask.to(self.device)

    def _get_attack_class(self, attack_name):
        """
        Dynamically import and return the specified attack class.
        
        Parameters
        ----------
        attack_name : str
            Name of the attack class to import
            
        Returns
        -------
        class
            The requested attack class
        """
        try:
            # Try to import from models.attack module
            attack_module = importlib.import_module('models.attack')
            attack_class = getattr(attack_module, attack_name)
            return attack_class
        except (ImportError, AttributeError) as e:
            print(f"Error loading attack class '{attack_name}': {e}")
            print("Falling back to ModelExtractionAttack0")
            # Fallback to ModelExtractionAttack0
            attack_module = importlib.import_module('models.attack')
            return getattr(attack_module, "ModelExtractionAttack0")

    def defend(self, attack_name=None):
        """
        Execute the random watermark defense.
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================Random Watermark Defense====================")

        # If model wasn't trained yet, train it
        if not hasattr(self, 'model_trained'):
            self.train_target_model(metric_comp)

        # Evaluate the defended model
        preds = self.evaluate_model(self.defense_model, self.dataset)
        inference_s = time.time()
        wm_preds = self.verify_watermark(self.defense_model)
        inference_e = time.time()

        # metric
        metric = DefenseMetric()
        metric.update(preds, self.labels[self.test_mask])
        wm_labels = self.watermark_graph.ndata['label']
        metric.update_wm(wm_preds, wm_labels)
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        res_comp = metric_comp.compute()

        return res, res_comp

    def train_target_model(self, metric_comp: DefenseCompMetric):
        """Train the target model with watermark injection."""
        defense_s = time.time()

        # Training and watermark generation (defense mechanism)
        self.defense_model = self._train_defense_model()
        self.model_trained = True

        defense_e = time.time()
        metric_comp.update(defense_time=(defense_e - defense_s))
        return self.defense_model

    def evaluate_model(self, model, dataset):
        """Evaluate model performance on downstream task"""
        model.eval()

        # Setup data loading
        sampler = NeighborSampler([5, 5])
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(self.device)
        test_collator = NodeCollator(self.graph, test_nids, sampler)
        test_dataloader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False
        )

        all_preds = []
        with torch.no_grad():
            for _, _, blocks in test_dataloader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_predictions = model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                all_preds.append(pred)

        preds = torch.cat(all_preds, dim=0).cpu()
        return preds

    def verify_watermark(self, model):
        """Verify watermark success rate"""
        model.eval()

        # Setup data loading for watermark graph
        sampler = NeighborSampler([5, 5])
        wm_nids = torch.arange(self.watermark_graph.number_of_nodes(), device=self.device)
        wm_collator = NodeCollator(self.watermark_graph, wm_nids, sampler)
        wm_dataloader = DataLoader(
            wm_collator.dataset,
            batch_size=self.wm_node,
            shuffle=False,
            collate_fn=wm_collator.collate,
            drop_last=False
        )

        all_preds = []
        with torch.no_grad():
            for _, _, blocks in wm_dataloader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_predictions = model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                all_preds.append(pred)

        wm_preds = torch.cat(all_preds, dim=0).cpu()
        return wm_preds

    def _train_target_model(self):
        """
        Helper function for training the target model on the original graph.
        
        Returns
        -------
        torch.nn.Module
            The trained target model
        """
        print("Training target model...")

        # Initialize model
        model = GraphSAGE(in_channels=self.feature_number,
                          hidden_channels=128,
                          out_channels=self.label_number)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        # Setup data loading
        sampler = NeighborSampler([5, 5])
        train_nids = self.train_mask.nonzero(as_tuple=True)[0].to(self.device)
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(self.device)

        train_collator = NodeCollator(self.graph, train_nids, sampler)
        test_collator = NodeCollator(self.graph, test_nids, sampler)

        train_dataloader = DataLoader(
            train_collator.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=train_collator.collate,
            drop_last=False
        )

        test_dataloader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False
        )

        # Training loop
        best_acc = 0
        for epoch in tqdm(range(1, 51), desc="Target model training"):
            # Train
            model.train()
            total_loss = 0
            for _, _, blocks in train_dataloader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']

                optimizer.zero_grad()
                output_predictions = model(blocks, input_features)
                loss = F.cross_entropy(output_predictions, output_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Test
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, _, blocks in test_dataloader:
                    blocks = [b.to(self.device) for b in blocks]
                    input_features = blocks[0].srcdata['feat']
                    output_labels = blocks[-1].dstdata['label']
                    output_predictions = model(blocks, input_features)
                    pred = output_predictions.argmax(dim=1)
                    correct += (pred == output_labels).sum().item()
                    total += len(output_labels)

            acc = correct / total
            if acc > best_acc:
                best_acc = acc

        print(f"Target model trained. Test accuracy: {best_acc:.4f}")
        return model

    def _train_defense_model(self):
        """
        Helper function for training a defense model with watermarking.
        
        Returns
        -------
        torch.nn.Module
            The trained defense model with embedded watermark
        """
        print("Training defense model with watermarking...")

        # Generate watermark graph
        wm_graph = self._generate_watermark_graph()

        # Initialize model
        model = GraphSAGE(in_channels=self.feature_number,
                          hidden_channels=128,
                          out_channels=self.label_number)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        # Setup data loading for original graph
        sampler = NeighborSampler([5, 5])
        train_nids = self.train_mask.nonzero(as_tuple=True)[0].to(self.device)
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(self.device)

        train_collator = NodeCollator(self.graph, train_nids, sampler)
        test_collator = NodeCollator(self.graph, test_nids, sampler)

        train_dataloader = DataLoader(
            train_collator.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=train_collator.collate,
            drop_last=False
        )

        test_dataloader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False
        )

        # Setup data loading for watermark graph
        wm_nids = torch.arange(wm_graph.number_of_nodes(), device=self.device)
        wm_collator = NodeCollator(wm_graph, wm_nids, sampler)

        wm_dataloader = DataLoader(
            wm_collator.dataset,
            batch_size=self.wm_node,
            shuffle=True,
            collate_fn=wm_collator.collate,
            drop_last=False
        )

        # First stage: Train on original graph
        best_acc = 0
        for epoch in tqdm(range(1, 51), desc="Defense model - stage 1"):
            # Train
            model.train()
            total_loss = 0
            for _, _, blocks in train_dataloader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']

                optimizer.zero_grad()
                output_predictions = model(blocks, input_features)
                loss = F.cross_entropy(output_predictions, output_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Test
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, _, blocks in test_dataloader:
                    blocks = [b.to(self.device) for b in blocks]
                    input_features = blocks[0].srcdata['feat']
                    output_labels = blocks[-1].dstdata['label']
                    output_predictions = model(blocks, input_features)
                    pred = output_predictions.argmax(dim=1)
                    correct += (pred == output_labels).sum().item()
                    total += len(output_labels)

            acc = correct / total
            if acc > best_acc:
                best_acc = acc

        # Second stage: Fine-tune on watermark graph
        for epoch in tqdm(range(1, 11), desc="Defense model - stage 2"):
            # Train on watermark
            model.train()
            total_loss = 0
            for _, _, blocks in wm_dataloader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']

                optimizer.zero_grad()
                output_predictions = model(blocks, input_features)
                loss = F.cross_entropy(output_predictions, output_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Final evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, _, blocks in test_dataloader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                output_predictions = model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                correct += (pred == output_labels).sum().item()
                total += len(output_labels)

        final_acc = correct / total

        # Watermark accuracy
        wm_acc = self._test_on_watermark(model, wm_dataloader)

        print(f"Defense model trained.")
        print(f"Test accuracy on original data: {final_acc:.4f}")
        print(f"Test accuracy on watermark: {wm_acc:.4f}")

        # Store watermark graph for later verification
        self.watermark_graph = wm_graph

        return model

    def _generate_watermark_graph(self):
        """
        Generate a watermark graph using Erdos-Renyi random graph model.
        
        Returns
        -------
        dgl.DGLGraph
            The generated watermark graph
        """
        # Generate random edges using Erdos-Renyi model
        wm_edge_index = erdos_renyi_graph(self.wm_node, self.pg, directed=False)

        # Generate random features with binomial distribution
        wm_features = torch.tensor(np.random.binomial(
            1, self.pr, size=(self.wm_node, self.feature_number)),
            dtype=torch.float32).to(self.device)

        # Generate random labels
        wm_labels = torch.tensor(np.random.randint(
            low=0, high=self.label_number, size=self.wm_node),
            dtype=torch.long).to(self.device)

        # Create DGL graph
        wm_graph = dgl.graph((wm_edge_index[0], wm_edge_index[1]), num_nodes=self.wm_node)
        wm_graph = wm_graph.to(self.device)

        # Add node features and labels
        wm_graph.ndata['feat'] = wm_features
        wm_graph.ndata['label'] = wm_labels

        # Add train and test masks (all True for simplicity)
        wm_graph.ndata['train_mask'] = torch.ones(self.wm_node, dtype=torch.bool, device=self.device)
        wm_graph.ndata['test_mask'] = torch.ones(self.wm_node, dtype=torch.bool, device=self.device)

        # Add self-loops
        wm_graph = dgl.add_self_loop(wm_graph)

        return wm_graph

    def _test_on_watermark(self, model, wm_dataloader):
        """
        Test a model's accuracy on the watermark graph.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to test
        wm_dataloader : DataLoader
            DataLoader for the watermark graph
            
        Returns
        -------
        float
            Accuracy on the watermark graph
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, _, blocks in wm_dataloader:
                blocks = [b.to(self.device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                output_predictions = model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)
                correct += (pred == output_labels).sum().item()
                total += len(output_labels)

        return correct / total

    def _evaluate_watermark(self, model):
        """
        Evaluate watermark detection effectiveness.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate
            
        Returns
        -------
        float
            Watermark detection accuracy
        """
        if not hasattr(self, 'watermark_graph'):
            print("Warning: No watermark graph found. Generate one first.")
            return 0.0

        # Setup data loading for watermark graph
        sampler = NeighborSampler([5, 5])
        wm_nids = torch.arange(self.watermark_graph.number_of_nodes(), device=self.device)
        wm_collator = NodeCollator(self.watermark_graph, wm_nids, sampler)

        wm_dataloader = DataLoader(
            wm_collator.dataset,
            batch_size=self.wm_node,
            shuffle=False,
            collate_fn=wm_collator.collate,
            drop_last=False
        )

        return self._test_on_watermark(model, wm_dataloader)

    def _evaluate_attack_on_watermark(self, attack_model):
        """
        Evaluate how well the attack model performs on the watermark graph.
        
        Parameters
        ----------
        attack_model : torch.nn.Module
            The model obtained from the attack
            
        Returns
        -------
        float
            Attack model's accuracy on the watermark graph
        """
        if not hasattr(self, 'watermark_graph'):
            print("Warning: No watermark graph found. Generate one first.")
            return 0.0

        # Check the model type to determine the correct evaluation approach
        model_name = attack_model.__class__.__name__

        # For GCN models that expect (g, features) input format
        if model_name == 'GCN':
            # Evaluate using the whole graph at once
            attack_model.eval()
            with torch.no_grad():
                # Pass the entire graph and features at once
                output_predictions = attack_model(self.watermark_graph, self.watermark_graph.ndata['feat'])
                pred = output_predictions.argmax(dim=1)
                correct = (pred == self.watermark_graph.ndata['label']).sum().item()
                total = self.watermark_graph.number_of_nodes()

            return correct / total

        # For GraphSAGE models that expect blocks input format
        elif model_name == 'GraphSAGE':
            # Setup data loading for watermark graph
            sampler = NeighborSampler([5, 5])
            wm_nids = torch.arange(self.watermark_graph.number_of_nodes(), device=self.device)
            wm_collator = NodeCollator(self.watermark_graph, wm_nids, sampler)

            wm_dataloader = DataLoader(
                wm_collator.dataset,
                batch_size=self.wm_node,
                shuffle=False,
                collate_fn=wm_collator.collate,
                drop_last=False
            )

            # Evaluate attack model on watermark
            attack_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, _, blocks in wm_dataloader:
                    blocks = [b.to(self.device) for b in blocks]
                    input_features = blocks[0].srcdata['feat']
                    output_labels = blocks[-1].dstdata['label']
                    output_predictions = attack_model(blocks, input_features)
                    pred = output_predictions.argmax(dim=1)
                    correct += (pred == output_labels).sum().item()
                    total += len(output_labels)

            return correct / total

        # For any other model type, print a warning and return 0
        else:
            print(f"Warning: Unsupported model type '{model_name}' for watermark evaluation")
            return 0.0
