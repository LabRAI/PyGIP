import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.data import Data as PyGData, Batch
from models.defense.base import BaseDefense


class BaseGraphFingerprint(nn.Module):
    """Base class for all graph fingerprint types"""
    def __init__(self, task_type, num_nodes, feature_dim):
        super(BaseGraphFingerprint, self).__init__()
        self.task_type = task_type
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
    def forward(self):
        raise NotImplementedError
        
    def to_pyg_data(self):
        raise NotImplementedError
        
    def get_sampled_outputs(self, model_output):
        """Sample outputs based on task type"""
        raise NotImplementedError

class NodeLevelFingerprint(BaseGraphFingerprint):
    """Fingerprint for node-level tasks (node classification)"""
    def __init__(self, num_nodes, feature_dim):
        super(NodeLevelFingerprint, self).__init__('node_level', num_nodes, feature_dim)
        
        # Initialize node features and adjacency
        self.x = nn.Parameter(torch.randn(num_nodes, feature_dim))
        self.adj_matrix = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        
        # For node-level tasks, sample outputs from m nodes
        self.sample_indices = nn.Parameter(
            torch.randint(0, num_nodes, (min(10, num_nodes),)), 
            requires_grad=False
        )
    
    def forward(self, return_pyg_data=True):
        # Use straight-through estimator for discrete adjacency
        adj_binary = (self.adj_matrix > 0.5).float()
        adj_binary_st = adj_binary + (self.adj_matrix - self.adj_matrix.detach())
        edge_index, edge_attr = dense_to_sparse(adj_binary_st)
        
        if return_pyg_data:
            return PyGData(x=self.x, edge_index=edge_index, edge_attr=edge_attr)
        return self.x, edge_index, adj_binary
    
    def to_pyg_data(self):
        """Convert to PyG Data object without gradient tracking"""
        with torch.no_grad():
            adj_binary = (self.adj_matrix > 0.5).float()
            edge_index, edge_attr = dense_to_sparse(adj_binary)
            return PyGData(x=self.x.detach(), edge_index=edge_index, edge_attr=edge_attr)
    
    def get_sampled_outputs(self, model_output):
        """Sample outputs from specific nodes for verification"""
        return model_output[self.sample_indices]

class EdgeLevelFingerprint(BaseGraphFingerprint):
    """Fingerprint for edge-level tasks (link prediction)"""
    def __init__(self, num_nodes, feature_dim):
        super(EdgeLevelFingerprint, self).__init__('edge_level', num_nodes, feature_dim)
        
        self.x = nn.Parameter(torch.randn(num_nodes, feature_dim))
        self.adj_matrix = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        
        # For edge-level tasks, sample m node pairs
        self.sample_pairs = self._initialize_sample_pairs(num_nodes, 8)
    
    def _initialize_sample_pairs(self, num_nodes, num_pairs):
        """Initialize node pairs to sample for edge outputs"""
        pairs = []
        for _ in range(num_pairs):
            u, v = random.sample(range(num_nodes), 2)
            pairs.append([u, v])
        return nn.Parameter(torch.tensor(pairs), requires_grad=False)
    
    def forward(self, return_pyg_data=True):
        adj_binary = (self.adj_matrix > 0.5).float()
        adj_binary_st = adj_binary + (self.adj_matrix - self.adj_matrix.detach())
        edge_index, edge_attr = dense_to_sparse(adj_binary_st)
        
        if return_pyg_data:
            return PyGData(x=self.x, edge_index=edge_index, edge_attr=edge_attr)
        return self.x, edge_index, adj_binary
    
    def to_pyg_data(self):
        """Convert to PyG Data object without gradient tracking"""
        with torch.no_grad():
            adj_binary = (self.adj_matrix > 0.5).float()
            edge_index, edge_attr = dense_to_sparse(adj_binary)
            return PyGData(x=self.x.detach(), edge_index=edge_index, edge_attr=edge_attr)
    
    def get_sampled_outputs(self, model_output):
        """Sample edge outputs for verification"""
        # For link prediction, model_output is an adjacency probability matrix
        sampled_outputs = []
        for u, v in self.sample_pairs:
            sampled_outputs.append(model_output[u, v])
        return torch.stack(sampled_outputs)

class GraphLevelFingerprint(BaseGraphFingerprint):
    """Fingerprint for graph-level tasks (graph classification, matching)"""
    def __init__(self, num_nodes, feature_dim, num_graphs=64):
        super(GraphLevelFingerprint, self).__init__('graph_level', num_nodes, feature_dim)
        
        # We have Multiple Independent Graphs for Graph Level Task
        self.graphs = nn.ModuleList([
            SingleGraphFingerprint(num_nodes, feature_dim) for _ in range(num_graphs)
        ])
    
    def forward(self, return_pyg_data=True):
        return [graph(return_pyg_data) for graph in self.graphs]
    
    def to_pyg_data(self):
        """Convert all graphs to PyG Data objects"""
        return [graph.to_pyg_data() for graph in self.graphs]
    
    def get_sampled_outputs(self, model_outputs):
        """Return graph-level outputs directly"""
        # For graph-level tasks, outputs are already at the graph level
        return torch.cat([output.unsqueeze(0) for output in model_outputs])

class SingleGraphFingerprint(nn.Module):
    """Single graph component for graph-level fingerprints"""
    def __init__(self, num_nodes, feature_dim):
        super(SingleGraphFingerprint, self).__init__()
        self.x = nn.Parameter(torch.randn(num_nodes, feature_dim))
        self.adj_matrix = nn.Parameter(torch.zeros(num_nodes, num_nodes))
    
    def forward(self, return_pyg_data=True):
        adj_binary = (self.adj_matrix > 0.5).float()
        adj_binary_st = adj_binary + (self.adj_matrix - self.adj_matrix.detach())
        edge_index, edge_attr = dense_to_sparse(adj_binary_st)
        
        if return_pyg_data:
            return PyGData(x=self.x, edge_index=edge_index, edge_attr=edge_attr)
        return self.x, edge_index, adj_binary
    
    def to_pyg_data(self):
        """Convert to PyG Data object without gradient tracking"""
        with torch.no_grad():
            adj_binary = (self.adj_matrix > 0.5).float()
            edge_index, edge_attr = dense_to_sparse(adj_binary)
            return PyGData(x=self.x.detach(), edge_index=edge_index, edge_attr=edge_attr)


class Univerifier(nn.Module):
    """
    Unified Verification Mechanism - Binary classifier that takes concatenated outputs
    from suspect models and predicts whether they are pirated or irrelevant.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16, 8, 4]):
        super(Univerifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.softmax(self.classifier(x), dim=1)


class GNNFingers(BaseDefense):
    """
    GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks
    with multi-task support for node-level, edge-level, and graph-level tasks
    """
    supported_api_types = {"dgl"}
    
    def __init__(self, dataset, attack_node_fraction=0.2, device=None, attack_name=None,
                 num_fingerprints=64, fingerprint_nodes=32, lambda_threshold=0.7,  
                 fingerprint_update_epochs=5, univerifier_update_epochs=3, 
                 fingerprint_lr=0.01, univerifier_lr=0.001, top_k_ratio=0.1, 
                 epochs=100, batch_size=32, num_neighbors=[10, 5], 
                 task_type='node_level'): 
        """
        Initialize GNNFingers defense framework with multi-task support
        
        Parameters
        ----------
        dataset : Dataset
            The original dataset containing the graph to defend
        attack_node_fraction : float
            Fraction of nodes to consider for attack
        device : torch.device
            Device to run computations on
        attack_name : str
            Name of the attack class to use
        num_fingerprints : int
            Number of graph fingerprints to generate
        fingerprint_nodes : int
            Number of nodes in each fingerprint graph
        lambda_threshold : float
            Threshold for Univerifier classification
        fingerprint_update_epochs: int
            Number of Epochs to update Fingerprint
        univerifier_update_epochs: int
            Number of Epochs to update Univerifier
        fingerprint_lr: float
            Learning rate for fingerprint update
        univerifier_lr: float
            Learning rate for Univerifier update
        top_k_ratio: float
            top k gradients of fingerprint adjacency matrix to select
        epochs: int
            total number of epochs to run experiment
        batch_size : int
            Batch size for training
        num_neighbors : list
            Number of neighbors to sample at each layer
        task_type : str
            Type of GNN task: 'node_level', 'edge_level', or 'graph_level'
        """
        super().__init__(dataset, attack_node_fraction, device)
        self.attack_name = attack_name or "ModelExtractionAttack0"
        self.dataset = dataset
        self.graph = dataset.graph_data
        
        # Extract dataset properties
        self.node_number = dataset.num_nodes
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes
        self.attack_node_number = int(self.node_number * attack_node_fraction)
        
        # Training parameters
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors

        #Task type
        self.task_type = task_type

        # Convert DGL to PyG data
        self.pyg_data = self._dgl_to_pyg(self.graph)
        
        # Extract features and labels
        self.features = self.pyg_data.x
        self.labels = self.pyg_data.y
        
        # Extract masks
        self.train_mask = self.pyg_data.train_mask
        self.test_mask = self.pyg_data.test_mask
        
        # GNNFingers parameters
        self.num_fingerprints = num_fingerprints
        self.fingerprint_nodes = fingerprint_nodes
        self.lambda_threshold = lambda_threshold
        
        # Initialize components
        self.target_gnn = None
        self.positive_gnns = []  # Pirated GNNs
        self.negative_gnns = []  # Irrelevant GNNs
        self.graph_fingerprints = None
        self.univerifier = None

        self.fingerprint_lr = fingerprint_lr
        self.fingerprint_update_epochs = fingerprint_update_epochs
        self.univerifier_update_epochs = univerifier_update_epochs
        self.univerifier_lr = univerifier_lr
        self.top_k_ratio = top_k_ratio
        self.epochs = epochs
        
        # Move tensors to device
        if self.device != 'cpu':
            self.graph = self.graph.to(self.device)
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
            self.train_mask = self.train_mask.to(self.device)
            self.test_mask = self.test_mask.to(self.device)
    
    def _dgl_to_pyg(self, dgl_graph):
        """Convert DGL graph to PyTorch Geometric Data object"""
        # Extract edge indices
        edge_index = torch.stack(dgl_graph.edges())
        x = dgl_graph.ndata.get('feat')
        y = dgl_graph.ndata.get('label')

        train_mask = dgl_graph.ndata.get('train_mask')
        val_mask = dgl_graph.ndata.get('val_mask')
        test_mask = dgl_graph.ndata.get('test_mask')

        data = PyGData(x=x, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        return data
   
    def _create_dataloaders(self, graph_data, batch_size=None, num_neighbors=None):
        """
        Create train and test dataloaders for PyG data
        
        Parameters
        ----------
        graph_data : PyG Data
            The graph data to create loaders for
        batch_size : int, optional
            Batch size (defaults to self.batch_size)
        num_neighbors : list, optional
            Number of neighbors to sample (defaults to self.num_neighbors)
        
        Returns
        -------
        train_loader : NeighborLoader
            Training dataloader
        test_loader : NeighborLoader
            Test dataloader
        """
        batch_size = batch_size or self.batch_size
        num_neighbors = num_neighbors or self.num_neighbors
        
        #Different dataloader for graph-level tasks
        if self.task_type == 'graph_level':
            # For graph-level tasks, use standard DataLoader
            train_loader = DataLoader([graph_data], batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader([graph_data], batch_size=self.batch_size, shuffle=False)
        else:
            # For node/edge level tasks, use NeighborLoader
            train_loader = NeighborLoader(
                graph_data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=True,
                input_nodes=graph_data.train_mask,
            )
            
            test_loader = NeighborLoader(
                graph_data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=False,
                input_nodes=graph_data.test_mask,
            )
        
        return train_loader, test_loader

    def defend(self, attack_name=None):
        """
        Main defense workflow for GNNFingers with multi-task support
        """
        #Validate task-dataset compatibility before starting
        #self._validate_task_dataset_compatibility()
        attack_name = attack_name or self.attack_name
        AttackClass = self._get_attack_class(attack_name)
        print(f"Using attack method: {attack_name}")
        print(f"Task type: {self.task_type}")
        
        # Step 1: Train target model
        print("Training target GNN...")
        self.target_gnn = self._train_gnn_model(self.pyg_data, "Target GNN")
        
        # Step 2: Prepare positive and negative GNNs
        print("Preparing positive (pirated) GNNs...")
        self.positive_gnns = self._prepare_positive_gnns(self.target_gnn, num_models=50)
        
        print("Preparing negative (irrelevant) GNNs...")
        self.negative_gnns = self._prepare_negative_gnns(num_models=50)
        
        # Step 3: Initialize graph fingerprints
        print("Initializing graph fingerprints...")
        self.graph_fingerprints = self._initialize_graph_fingerprints()
        
        # Step 4: Initialize Univerifier
        output_dim = self._get_output_dimension(self.target_gnn, self.graph_fingerprints[0])
        self.univerifier = Univerifier(input_dim=output_dim * self.num_fingerprints)
        self.univerifier = self.univerifier.to(self.device)
        
        # Step 5: Joint learning of fingerprints and Univerifier
        print("Joint learning of fingerprints and Univerifier...")
        self._joint_learning_alternating()
        
        # Step 6: Attack target model
        print("Attacking target model...")
        attack = AttackClass(self.dataset, attack_node_fraction=0.2)
        attack_results = attack.attack()
        suspect_model = attack.net2 if hasattr(attack, 'net2') else None
        
        # Step 7: Verify ownership
        if suspect_model is not None:
            print("Verifying ownership of suspect model...")
            verification_result = self._verify_ownership(suspect_model)
            print(f"Ownership verification result: {verification_result}")
            
            return {
                "attack_results": attack_results,
                "verification_result": verification_result,
                "target_accuracy": self._evaluate_model(self.target_gnn, self.pyg_data),
                "suspect_accuracy": self._evaluate_model(suspect_model, self.pyg_data)
            }
        
        return {"attack_results": attack_results, "verification_result": "No suspect model found"}


    def _validate_task_dataset_compatibility(self):
        """
        Validate that the selected task type is compatible with the dataset
        """
        print(f"Validating task type '{self.task_type}' with dataset {type(self.dataset).__name__}...")
        
        if self.task_type == 'graph_level':
            # For graph-level tasks, we need to check if this is a graph dataset
            if not hasattr(self.pyg_data, 'graph_y') or self.pyg_data.graph_y is None:
                raise ValueError(
                    f"Graph-level task selected but dataset {type(self.dataset).__name__} "
                    f"appears to be a single-graph dataset. Use node-level or edge-level task instead, "
                    f"or use a multi-graph dataset like TUDataset for graph classification."
                )
            print(" Graph-level task compatible with dataset")
        
        elif self.task_type == 'edge_level':
            # For edge-level tasks, check if we have sufficient edge information
            if not hasattr(self.pyg_data, 'edge_index') or self.pyg_data.edge_index.size(1) == 0:
                print("⚠ Warning: Edge-level task selected but dataset has limited edge information")
            else:
                print(f"Edge-level task compatible with dataset ({self.pyg_data.edge_index.size(1)} edges)")
        
        elif self.task_type == 'node_level':
            # For node-level tasks, ensure we have node labels
            if not hasattr(self.pyg_data, 'y') or self.pyg_data.y is None:
                raise ValueError(
                    f"Node-level task selected but dataset {type(self.dataset).__name__} "
                    f"does not have node labels."
                )
            print(f"Node-level task compatible with dataset ({self.pyg_data.y.unique().size(0)} classes)")
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}. Use 'node_level', 'edge_level', or 'graph_level'")
        
        return True
        
    def _train_gnn_model(self, data, model_name="GNN", epochs=100):
        """Train a GNN model on the given data using batched training"""
        # NEW: Different model architectures for different tasks
        if self.task_type == 'graph_level':
            model = GraphLevelGNN(
                in_channels=data.x.size(1),
                hidden_channels=128,
                out_channels=self.label_number,
                num_layers=3
            ).to(self.device)
        else:
            model = GCNConvGNN(
                in_channels=data.x.size(1),
                hidden_channels=128,
                out_channels=self.label_number,
                num_layers=3
            ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Create dataloaders using the helper function
        train_loader, test_loader = self._create_dataloaders(data)
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass with task-specific handling
                if self.task_type == 'graph_level':
                    out = model(batch.x, batch.edge_index, batch.batch)
                else:
                    out = model(batch.x, batch.edge_index)
                
                # Loss calculation with task-specific masking
                if self.task_type == 'graph_level':
                    loss = criterion(out, batch.y)
                else:
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluate on test set
            if epoch % 10 == 0:
                test_acc = self._evaluate_model_with_loader(model, test_loader)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = copy.deepcopy(model)
                
                print(f"{model_name} Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={test_acc:.4f}")
        
        print(f"{model_name} trained with best accuracy: {best_acc:.4f}")
        return best_model
    
    def _evaluate_model_with_loader(self, model, test_loader):
        """Evaluate model accuracy using a test dataloader"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                if self.task_type == 'graph_level':
                    out = model(batch.x, batch.edge_index, batch.batch)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
                else:
                    out = model(batch.x, batch.edge_index)
                    pred = out.argmax(dim=1)
                    
                    # Only count test nodes
                    test_mask = batch.test_mask if hasattr(batch, 'test_mask') else torch.ones(batch.num_nodes, dtype=bool)
                    correct += (pred[test_mask] == batch.y[test_mask]).sum().item()
                    total += test_mask.sum().item()
        
        return correct / total if total > 0 else 0
    
    def _prepare_positive_gnns(self, target_model, num_models=50):
        """Prepare pirated GNNs using obfuscation techniques"""
        positive_models = []
        
        for i in range(num_models):
            # Apply different obfuscation techniques
            if i % 3 == 0:
                # Fine-tuning with batched training
                layers_to_finetune = random.randint(1, 3)
                model = self._fine_tune_model(copy.deepcopy(target_model), self.pyg_data, 
                                            epochs=10, num_layers_to_finetune=layers_to_finetune)
            elif i % 3 == 1:
                # Partial retraining with batched training
                layers_to_retrain = random.randint(1, 3)
                model = self._partial_retrain_model(copy.deepcopy(target_model), self.pyg_data, 
                                                epochs=15, num_layers_to_retrain=layers_to_retrain)
            else:
                # Distillation with batched training
                temperature = random.choice([1.5, 2.0, 3.0, 4.0])
                model = self._distill_model(target_model, self.pyg_data, 
                                        epochs=30, temperature=temperature)
            
            positive_models.append(model)
        
        return positive_models
    
    def _prepare_negative_gnns(self, num_models=50):
        """Prepare irrelevant GNNs"""
        negative_models = []
        
        for i in range(num_models):
            # Train from scratch with different architectures or data
            if i % 2 == 0:
                # Different architecture
                model = self._train_different_architecture(self.pyg_data)
            else:
                # Different training data (subset)
                model = self._train_on_subset(self.pyg_data)
            
            negative_models.append(model)
        
        return negative_models
    
    def _fine_tune_model(self, model, data, epochs=10, num_layers_to_finetune=1):
        """Fine-tune a model using batched training"""
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last K layers for fine-tuning
        if hasattr(model, 'convs'):
            total_layers = len(model.convs)
            layers_to_finetune = min(num_layers_to_finetune, total_layers)
            
            for i in range(total_layers - layers_to_finetune, total_layers):
                for param in model.convs[i].parameters():
                    param.requires_grad = True
        
        # Create dataloader using helper function
        train_loader, _ = self._create_dataloaders(data)
        
        # Only optimize parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Task-specific forward pass
                if self.task_type == 'graph_level':
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                else:
                    out = model(batch.x, batch.edge_index)
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Unfreeze all parameters for future use
        for param in model.parameters():
            param.requires_grad = True
        
        return model
    
    def _partial_retrain_model(self, model, data, epochs=10, num_layers_to_retrain=2):
        """Partially retrain a model with random initialization of K layers before resuming training"""
        # Randomly initialize selected K layers
        if hasattr(model, 'convs'):
            # For models with convs attribute (like GCNConvGNN, GATConvGNN)
            total_layers = len(model.convs)
            layers_to_retrain = min(num_layers_to_retrain, total_layers)
            
            # Randomly select K layers to retrain
            layer_indices = random.sample(range(total_layers), layers_to_retrain)
            
            print(f"Partially retraining layers: {layer_indices}")
            
            for idx in layer_indices:
                model.convs[idx].reset_parameters()  # Random reinitialization
        
        # Train the entire model (both retrained and original layers)
        train_loader, test_loader = self._create_dataloaders(data)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Task-specific forward pass
                if self.task_type == 'graph_level':
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                else:
                    out = model(batch.x, batch.edge_index)
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Track best model
            if epoch % 5 == 0:
                acc = self._evaluate_model_with_loader(model, test_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model)
        
        print(f"Partial retraining completed. Best accuracy: {best_acc:.4f}")
        return best_model if best_model is not None else model
    
    def _distill_model(self, teacher_model, data, epochs=30, temperature=2.0):
        """Distill knowledge using batched training"""
        # Create student model with different architecture
        if isinstance(teacher_model, (GCNConvGNN, GraphLevelGNN)):
            # If teacher is GCN or GraphLevel, use GAT as student
            if self.task_type == 'graph_level':
                student_model = GraphLevelGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=96,
                    out_channels=self.label_number,
                    num_layers=2
                ).to(self.device)
            else:
                student_model = GATConvGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=96,
                    out_channels=self.label_number,
                    num_layers=2,
                    heads=3
                ).to(self.device)
        else:
            # If teacher is GAT or other, use GCN as student
            if self.task_type == 'graph_level':
                student_model = GraphLevelGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=3
                ).to(self.device)
            else:
                student_model = GCNConvGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=3
                ).to(self.device)
        
        # Create dataloader using helper function
        train_loader, test_loader = self._create_dataloaders(data)
        
        optimizer = optim.Adam(student_model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # Combined loss: KL divergence for distillation + cross entropy for ground truth
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss()
        
        teacher_model.eval()
        
        best_acc = 0
        best_student = None
        
        for epoch in range(epochs):
            student_model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Get teacher predictions (with temperature scaling)
                with torch.no_grad():
                    if self.task_type == 'graph_level':
                        teacher_logits = teacher_model(batch.x, batch.edge_index, batch.batch)
                    else:
                        teacher_logits = teacher_model(batch.x, batch.edge_index)
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                
                # Get student predictions
                if self.task_type == 'graph_level':
                    student_logits = student_model(batch.x, batch.edge_index, batch.batch)
                else:
                    student_logits = student_model(batch.x, batch.edge_index)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                
                # Distillation loss (KL divergence between teacher and student)
                if self.task_type == 'graph_level':
                    distill_loss = kl_loss(student_log_probs, teacher_probs) * (temperature ** 2)
                    class_loss = ce_loss(student_logits, batch.y)
                else:
                    distill_loss = kl_loss(student_log_probs[batch.train_mask], 
                                        teacher_probs[batch.train_mask]) * (temperature ** 2)
                    class_loss = ce_loss(student_logits[batch.train_mask], 
                                        batch.y[batch.train_mask])
                
                # Combined loss (weighted sum)
                loss = 0.7 * distill_loss + 0.3 * class_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Track best student model
            if epoch % 5 == 0:
                student_model.eval()
                test_acc = self._evaluate_model_with_loader(student_model, test_loader)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_student = copy.deepcopy(student_model)
                
                print(f"Distillation Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={test_acc:.4f}")
        
        print(f"Distillation completed. Best student accuracy: {best_acc:.4f}")
        return best_student if best_student is not None else student_model
    
    def _train_different_architecture(self, data):
        """Train a model with different architecture for negative GNNs"""
        # Use opposite architecture of target model
        if isinstance(self.target_gnn, (GCNConvGNN, GraphLevelGNN)):
            if self.task_type == 'graph_level':
                model = GraphLevelGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=2
                ).to(self.device)
            else:
                model = GATConvGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=2,
                    heads=4
                ).to(self.device)
        else:
            if self.task_type == 'graph_level':
                model = GraphLevelGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=3
                ).to(self.device)
            else:
                model = GCNConvGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=3
                ).to(self.device)
    
        return self._train_gnn_model_with_data(model, data, epochs=50)
    
    def _train_on_subset(self, data, subset_ratio=0.7):
        """Train on a subset of the data"""
        # Create subset mask
        if self.task_type == 'graph_level':
            # For graph-level, we can't easily create subset, so use different architecture
            return self._train_different_architecture(data)
        else:
            # For node/edge level, create subset of training nodes
            num_train = int(data.train_mask.sum().item() * subset_ratio)
            subset_mask = torch.zeros_like(data.train_mask)
            train_indices = data.train_mask.nonzero(as_tuple=True)[0]
            selected_indices = random.sample(range(len(train_indices)), min(num_train, len(train_indices)))
            subset_mask[train_indices[selected_indices]] = True
            
            # Create subset data
            subset_data = PyGData(
                x=data.x, 
                edge_index=data.edge_index, 
                y=data.y,
                train_mask=subset_mask,
                test_mask=data.test_mask
            )
            
            # Use opposite architecture of target model
            if isinstance(self.target_gnn, (GCNConvGNN, GraphLevelGNN)):
                model = GCNConvGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=2
                ).to(self.device)
            else:
                model = GATConvGNN(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=self.label_number,
                    num_layers=3,
                    heads=4
                ).to(self.device)
            
            return self._train_gnn_model_with_data(model, subset_data, epochs=50)
    
    def _train_gnn_model_with_data(self, model, data, epochs=100):
        """Train a specific model on specific data using batched training"""
        # Create dataloaders using helper function
        train_loader, test_loader = self._create_dataloaders(data)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Task-specific forward pass
                if self.task_type == 'graph_level':
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                else:
                    out = model(batch.x, batch.edge_index)
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                acc = self._evaluate_model_with_loader(model, test_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model)
        
        return best_model
    
    #Task-Specific Fingerprint Initialization
    def _initialize_graph_fingerprints(self):
        """Initialize task-specific graph fingerprints"""
        fingerprints = nn.ModuleList()
        feature_dim = self.features.size(1) if self.features is not None else 16
        
        for _ in range(self.num_fingerprints):
            if self.task_type == 'node_level':
                fingerprint = NodeLevelFingerprint(
                    self.fingerprint_nodes, feature_dim
                ).to(self.device)
                
            elif self.task_type == 'edge_level':
                fingerprint = EdgeLevelFingerprint(
                    self.fingerprint_nodes, feature_dim
                ).to(self.device)
                
            elif self.task_type == 'graph_level':
                # For graph-level tasks, use multiple graphs per fingerprint
                fingerprint = GraphLevelFingerprint(
                    self.fingerprint_nodes, feature_dim, num_graphs=3
                ).to(self.device)
            
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    #Task-Specific Output Handling
    def _get_model_outputs(self, model, fingerprint):
        """Get model outputs based on task type"""
        model.eval()
        
        if self.task_type == 'node_level':
            fingerprint_data = fingerprint.to_pyg_data()
            output = model(fingerprint_data.x.to(self.device), 
                          fingerprint_data.edge_index.to(self.device))
            return fingerprint.get_sampled_outputs(output)
            
        elif self.task_type == 'edge_level':
            fingerprint_data = fingerprint.to_pyg_data()
            # For link prediction, assume model outputs adjacency probabilities
            node_embeddings = model(fingerprint_data.x.to(self.device), 
                                  fingerprint_data.edge_index.to(self.device))
            # Simulate edge prediction by dot product of node embeddings
            adj_probs = torch.sigmoid(torch.mm(node_embeddings, node_embeddings.t()))
            return fingerprint.get_sampled_outputs(adj_probs)
            
        elif self.task_type == 'graph_level':
            # For graph-level tasks, process each graph in the fingerprint
            graph_outputs = []
            for graph_data in fingerprint.to_pyg_data():
                # Create batch dimension for single graph
                batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device)
                output = model(graph_data.x.to(self.device), 
                             graph_data.edge_index.to(self.device), 
                             batch)
                graph_outputs.append(output)
            return fingerprint.get_sampled_outputs(graph_outputs)
    
    def _get_output_dimension(self, model, fingerprint):
        """Get the output dimension for a given fingerprint"""
        model.eval()
        with torch.no_grad():
            output = self._get_model_outputs(model, fingerprint)
            return output.numel()  # Total number of elements
    
    def _verify_ownership(self, suspect_model):
        """Verify if a suspect model is pirated from the target model"""
        target_outputs = []
        suspect_outputs = []
        
        for fingerprint in self.graph_fingerprints:
            self.target_gnn.eval()
            suspect_model.eval()
            
            with torch.no_grad():
                target_out = self._get_model_outputs(self.target_gnn, fingerprint)
                suspect_out = self._get_model_outputs(suspect_model, fingerprint)
            
            target_outputs.append(target_out)
            suspect_outputs.append(suspect_out)
        
        # Concatenate all outputs
        target_concat = torch.cat(target_outputs, dim=0).view(1, -1)
        suspect_concat = torch.cat(suspect_outputs, dim=0).view(1, -1)
        
        # Get Univerifier prediction
        self.univerifier.eval()
        with torch.no_grad():
            prediction = self.univerifier(suspect_concat)
            confidence = prediction[0, 1].item()  # Probability of being pirated
        
        return confidence > self.lambda_threshold, confidence
    
    def _evaluate_model(self, model, data):
        """Evaluate model accuracy"""
        model.eval()
        with torch.no_grad():
            if self.task_type == 'graph_level':
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
                out = model(data.x.to(self.device), data.edge_index.to(self.device), batch)
                pred = out.argmax(dim=1)
                correct = (pred == data.y).sum().item()
                total = data.y.size(0)
            else:
                out = model(data.x.to(self.device), data.edge_index.to(self.device))
                pred = out.argmax(dim=1)
                correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
                total = data.test_mask.sum().item()
            return correct / total if total > 0 else 0

    def _update_adjacency_discrete(self, fingerprint, grad_adj):
        """
        Update discrete adjacency matrix based on gradients
        """
        # Get current discrete adjacency
        if self.task_type == 'graph_level':
            # For graph-level fingerprints, update each graph's adjacency
            for graph_idx, graph in enumerate(fingerprint.graphs):
                current_adj = (graph.adj_matrix > 0.5).float()
                self._update_single_adjacency(graph, grad_adj[graph_idx] if grad_adj.dim() > 2 else grad_adj)
        else:
            # For node/edge level fingerprints
            current_adj = (fingerprint.adj_matrix > 0.5).float()
            self._update_single_adjacency(fingerprint, grad_adj)
    
    def _update_single_adjacency(self, fingerprint, grad_adj):
        """Update adjacency for a single graph"""
        current_adj = (fingerprint.adj_matrix > 0.5).float()
        
        # Get absolute gradient values and flatten
        grad_abs = torch.abs(grad_adj)
        grad_abs_flat = grad_abs.view(-1)
        
        # Determine top-K edges to consider for flipping
        k = int(self.top_k_ratio * self.fingerprint_nodes * self.fingerprint_nodes)
        topk_values, topk_indices = torch.topk(grad_abs_flat, k)
        
        # Convert flat indices to row, col indices
        rows = topk_indices // self.fingerprint_nodes
        cols = topk_indices % self.fingerprint_nodes
        
        # Update edges based on gradient signs
        with torch.no_grad():
            for idx in range(k):
                row, col = rows[idx], cols[idx]
                grad_val = grad_adj[row, col]
                
                # Current edge existence (0 or 1)
                current_edge = current_adj[row, col]
                
                # Apply update rules:
                if current_edge > 0.5 and grad_val <= 0:
                    # Edge exists and gradient is negative → remove edge
                    fingerprint.adj_matrix.data[row, col] = 0.0
                elif current_edge < 0.5 and grad_val >= 0:
                    # Edge doesn't exist and gradient is positive → add edge
                    fingerprint.adj_matrix.data[row, col] = 1.0

    def _update_fingerprints_discrete(self, loss, top_k_ratio=0.1):
        """
        Update graph fingerprints using gradients
        """
        # Compute gradients for all fingerprints
        gradients_adj = []
        gradients_x = []
        
        for fingerprint in self.graph_fingerprints:
            if self.task_type == 'graph_level':
                # For graph-level, we need to handle multiple graphs
                grad_adj_list = []
                grad_x_list = []
                for graph in fingerprint.graphs:
                    grad_adj = torch.autograd.grad(
                        loss, graph.adj_matrix, 
                        retain_graph=True, create_graph=False
                    )[0]
                    grad_x = torch.autograd.grad(
                        loss, graph.x,
                        retain_graph=True, create_graph=False
                    )[0]
                    grad_adj_list.append(grad_adj)
                    grad_x_list.append(grad_x)
                gradients_adj.append(torch.stack(grad_adj_list))
                gradients_x.append(torch.stack(grad_x_list))
            else:
                # For node/edge level
                grad_adj = torch.autograd.grad(
                    loss, fingerprint.adj_matrix, 
                    retain_graph=True, create_graph=False
                )[0]
                grad_x = torch.autograd.grad(
                    loss, fingerprint.x,
                    retain_graph=True, create_graph=False
                )[0]
                gradients_adj.append(grad_adj)
                gradients_x.append(grad_x)
        
        # Update each fingerprint
        for i, fingerprint in enumerate(self.graph_fingerprints):
            grad_adj = gradients_adj[i]
            grad_x = gradients_x[i]
            
            if self.task_type == 'graph_level':
                # Update each graph in the fingerprint
                for graph_idx, graph in enumerate(fingerprint.graphs):
                    with torch.no_grad():
                        graph.x.data += self.fingerprint_lr * grad_x[graph_idx]
                        # Clip node features
                        if self.features is not None:
                            min_val = self.features.min().item()
                            max_val = self.features.max().item()
                            graph.x.data = torch.clamp(graph.x.data, min_val, max_val)
                        else:
                            graph.x.data = torch.clamp(graph.x.data, -3, 3)
                
                # Update adjacency
                self._update_adjacency_discrete(fingerprint, grad_adj)
            else:
                # Update node features with clipping
                with torch.no_grad():
                    fingerprint.x.data += self.fingerprint_lr * grad_x
                    
                    # Clip node features to reasonable range
                    if self.features is not None:
                        min_val = self.features.min().item()
                        max_val = self.features.max().item()
                        fingerprint.x.data = torch.clamp(fingerprint.x.data, min_val, max_val)
                    else:
                        fingerprint.x.data = torch.clamp(fingerprint.x.data, -3, 3)
                
                # Update adjacency matrix using discrete strategy
                self._update_adjacency_discrete(fingerprint, grad_adj)

    def visualize_fingerprint_evolution(self, epoch):
        """Visualize how fingerprints evolve during training"""
        if epoch % 20 == 0:  # Visualize every 20 epochs
            print(f"\n=== Fingerprint Evolution at Epoch {epoch} ===")
            
            for i, fingerprint in enumerate(self.graph_fingerprints[:2]):  # First 2 only
                if self.task_type == 'graph_level':
                    print(f"Graph-Level Fingerprint {i}: {len(fingerprint.graphs)} graphs")
                    for graph_idx, graph in enumerate(fingerprint.graphs):
                        current_adj = (graph.adj_matrix > 0.5).float()
                        num_edges = current_adj.sum().item()
                        sparsity = 1 - (num_edges / (self.fingerprint_nodes * self.fingerprint_nodes))
                        print(f"  Graph {graph_idx}: {num_edges} edges, sparsity: {sparsity:.3f}")
                else:
                    if self.task_type == 'node_level':
                        current_adj = (fingerprint.adj_matrix > 0.5).float()
                        num_edges = current_adj.sum().item()
                        sparsity = 1 - (num_edges / (self.fingerprint_nodes * self.fingerprint_nodes))
                        print(f"Node-Level Fingerprint {i}: {num_edges} edges, sparsity: {sparsity:.3f}")
                    else:
                        current_adj = (fingerprint.adj_matrix > 0.5).float()
                        num_edges = current_adj.sum().item()
                        sparsity = 1 - (num_edges / (self.fingerprint_nodes * self.fingerprint_nodes))
                        print(f"Edge-Level Fingerprint {i}: {num_edges} edges, sparsity: {sparsity:.3f}")

    def _joint_learning_alternating(self):
        """
        Joint learning with alternating optimization algorithm
        """
        
        # Prepare all models and labels
        all_models = [self.target_gnn] + self.positive_gnns + self.negative_gnns
        labels = torch.cat([
            torch.ones(len(self.positive_gnns) + 1),  # Target + positive models
            torch.zeros(len(self.negative_gnns))      # Negative models
        ]).long().to(self.device)
        
        # Flag to alternate between fingerprint and univerifier updates
        update_fingerprints = True
        
        for epoch in range(self.epochs):
            # Forward pass through all models
            all_outputs = []
            for model in all_models:
                model_outputs = []
                for fingerprint in self.graph_fingerprints:
                    model.eval()
                    
                    # Get model outputs with task-specific handling
                    output = self._get_model_outputs(model, fingerprint)
                    model_outputs.append(output)
                
                # Concatenate all fingerprint outputs
                concatenated = torch.cat(model_outputs, dim=0).view(1, -1)
                all_outputs.append(concatenated)
            
            # Stack all outputs
            all_outputs = torch.cat(all_outputs, dim=0)
            
            # Univerifier prediction
            univerifier_out = self.univerifier(all_outputs)
            
            # Calculate joint loss
            loss = 0
            for i, model in enumerate(all_models):
                if i < len(self.positive_gnns) + 1:  # Target + positive models
                    # log o_+(F) and log o_+(F_+) terms
                    loss += torch.log(univerifier_out[i, 1] + 1e-10)
                else:  # Negative models
                    # log o_-(F_-) term
                    loss += torch.log(1 - univerifier_out[i, 1] + 1e-10)
            
            loss = -loss  # Negative log likelihood
            
            # Alternating optimization
            if update_fingerprints:
                # Phase 1: Update fingerprints for e1 epochs
                for e in range(self.fingerprint_update_epochs):
                    self._update_fingerprints_discrete(loss, self.top_k_ratio)
                
                update_fingerprints = False
                print(f"Epoch {epoch}: Updated fingerprints, Loss: {loss.item():.4f}")
                
            else:
                # Phase 2: Update Univerifier for e2 epochs
                univerifier_optimizer = optim.Adam(self.univerifier.parameters(), lr=self.univerifier_lr)
                
                for e in range(self.univerifier_update_epochs):
                    univerifier_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    univerifier_optimizer.step()
                
                update_fingerprints = True
                print(f"Epoch {epoch}: Updated Univerifier, Loss: {loss.item():.4f}")
            
            # Calculate accuracy every 10 epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    preds = univerifier_out.argmax(dim=1)
                    acc = (preds == labels).float().mean().item()
                    
                    # Calculate true positive and true negative rates
                    tp_mask = (preds == 1) & (labels == 1)
                    tn_mask = (preds == 0) & (labels == 0)
                    
                    tp_rate = tp_mask.float().mean().item() if (labels == 1).sum() > 0 else 0
                    tn_rate = tn_mask.float().mean().item() if (labels == 0).sum() > 0 else 0
                    
                    print(f"Epoch {epoch}, Acc: {acc:.4f}, TP: {tp_rate:.4f}, TN: {tn_rate:.4f}")
            
            # Visualize fingerprint evolution
            if epoch % 20 == 0:
                self.visualize_fingerprint_evolution(epoch)


#Graph-Level GNN Model
class GraphLevelGNN(nn.Module):
    """GNN model for graph-level tasks with global pooling"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, pool_type='mean'):
        super(GraphLevelGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.pool_type = pool_type
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training, p=0.5)
        
        # Global pooling
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        else:  # sum pooling
            x = global_add_pool(x, batch)
        
        # Final classification
        return self.classifier(x)

class GCNConvGNN(nn.Module):
    """GCN-based GNN model"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GCNConvGNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training, p=0.5)
        return x

class GATConvGNN(nn.Module):
    """GAT-based GNN model"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4):
        super(GATConvGNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, training=self.training, p=0.6)
        return x
    