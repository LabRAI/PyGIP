import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
from .utils import *
from pygip.models.attack.base import BaseAttack
from pygip.models.nn import SurrogateModelGraphClassification,TargetModelGraphClassification,GCNGraphClassification,GraphSAGEGraphClassification,GATGraphClassification,GINGraphClassification,Classifier,CAM,GradCAM,GradientExplainer
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer
import os.path as osp
import os
import random
import math
from scipy.stats import kendalltau
from torch_geometric.data import Batch, Data
from collections import defaultdict

class EGSteal(BaseAttack):
    supported_api_types = {"pyg"}
    
    def __init__(self, dataset, query_shadow_ratio=0.3,gnn_backbone = 'GIN',explanation_mode = 'CAM'):
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data
        self.query_shadow_ratio = query_shadow_ratio
        
        self.num_graphs = self.dataset.num_graphs
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self.num_edge_features = self.dataset.num_edge_features if hasattr(dataset, 'num_edge_features') else 0
        self.gnn_backbone = gnn_backbone
        self.explanation_mode = explanation_mode
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fixed parameters
        self.seed = 42
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 200
        self.gat_heads = 4
        self.gnn_layer = 3
        self.gnn_hidden_dim = 128
        self.gnnexplainer_epochs = 100
        self.pgexplainer_epochs = 100
        self.augmentation_ratio = 0.2
        self.operation_ratio = 0.05
        self.align_weight = 1.0
        self.augmentation_type = 'combined' # ['drop_node', 'drop_edge', 'add_edge', 'combined']
        self.shadow_val_ratio = 0.2
    
    def prepare_data(self):
        set_seed(self.seed)  # For reproducibility
        # Define split ratios
        target_ratio = 0.4
        target_val_ratio = 0.2
        test_ratio = 0.2
        shadow_ratio = 0.4
        target_num = int(self.num_graphs * target_ratio)
        test_num = int(self.num_graphs * test_ratio)
        shadow_num = self.num_graphs - target_num - test_num  # Ensure total consistency

        # Randomly split dataset
        target_dataset, test_dataset, shadow_dataset = random_split(
            self.graph_dataset,
            [target_num, test_num, shadow_num]
        )

        # Further split target_dataset into train and val
        target_train_num = int(target_num * (1 - target_val_ratio))
        target_val_num = target_num - target_train_num

        target_train_dataset, target_val_dataset = random_split(
            target_dataset,
            [target_train_num, target_val_num]
        )
        
        print("\nDataset split sizes:")
        print(f"Target train set size: {len(target_train_dataset)} ({len(target_train_dataset) / self.num_graphs:.1%})")
        print(f"Target val set size: {len(target_val_dataset)} ({len(target_val_dataset) / self.num_graphs:.1%})")
        print(f"Test set size: {len(test_dataset)} ({len(test_dataset) / self.num_graphs:.1%})")
        print(f"Shadow dataset size: {len(shadow_dataset)} ({len(shadow_dataset) / self.num_graphs:.1%})")
        
        return target_train_dataset, target_val_dataset, shadow_dataset, test_dataset
    
    def _train_target_model(self,target_train_dataset,target_val_dataset,test_dataset):
        # build paths
        save_root = './saved_models/EGSteal'
        os.makedirs(save_root, exist_ok=True)
        model_path = osp.join(
            save_root,
            f"{self.dataset.__class__.__name__}_{self.gnn_backbone}_{self.explanation_mode}_model.pth"
        )
        target_train_loader = DataLoader(target_train_dataset, batch_size=self.batch_size, shuffle=True)
        target_val_loader = DataLoader(target_val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        #Initialize model
        if self.gnn_backbone == 'GIN':
            encoder = GINGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer
            ).to(self.device)
        elif self.gnn_backbone == 'GCN':
            encoder = GCNGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer
            ).to(self.device)
        elif self.gnn_backbone == 'GAT':
            encoder = GATGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer,
                heads=self.gat_heads
            ).to(self.device)
        elif self.gnn_backbone == 'GraphSAGE':
            encoder = GraphSAGEGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer
            ).to(self.device)
        else:
            raise ValueError(f"Invalid GNN backbone specified: {self.gnn_backbone}. Expected 'GIN', 'GCN', or 'GAT', or 'GraphSAGE'.")

        predictor = Classifier(
            input_dim=self.gnn_hidden_dim,
            output_dim=self.num_classes
        ).to(self.device)

        model = TargetModelGraphClassification(encoder=encoder, predictor=predictor, explanation_mode=self.explanation_mode).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        epochs = self.epochs
        best_val_auc = 0.0
        best_model_state = None
        
        # ---- LOAD if exists (state_dict) ----
        if osp.exists(model_path):
            print(f"Loading pre-trained target model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        
        with tqdm(total=epochs, desc='Epochs') as epoch_pbar:
            for epoch in range(1, epochs + 1):
                train_loss, train_acc, train_auc = train_loop_target_model(model, target_train_loader, optimizer, self.device,self)
                val_loss, val_acc, val_auc = evaluate_loop_target_model(model, target_val_loader, self.device,self)

                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Train Acc': f'{train_acc:.4f}',
                    'Train AUC': f'{train_auc:.4f}',
                    'Val Loss': f'{val_loss:.4f}',
                    'Val Acc': f'{val_acc:.4f}',
                    'Val AUC': f'{val_auc:.4f}'
                })
                epoch_pbar.update(1)

                # Update best model if validation AUC is higher
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict()
        # Evaluate the best model on the test set
        model.load_state_dict(best_model_state)
        test_loss, test_acc, test_auc = evaluate_loop_target_model(model, test_loader, self.device, self)
        print(f"Test Accuracy of the best model: {test_acc:.4f}")
        print(f"Test AUC of the best model: {test_auc:.4f}")
        
        # ---- SAVE (ensure directory exists; save file, not directory) ----
        # (we already created save_root above)
        torch.save(best_model_state, model_path)
        print(f"Saved best target state_dict to {model_path}")
        
        return model
    
    def prepare_shadow_data(self,model,shadow_dataset,test_dataset):
        if self.explanation_mode == 'GNNExplainer':
            gnnexplainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=self.gnnexplainer_epochs),
                explanation_type='model',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw'
                ),
                node_mask_type='object',
                edge_mask_type=None
            )

        if self.explanation_mode == 'PGExplainer':
            pgexplainer = Explainer(
                model=model,
                algorithm=PGExplainer(epochs=self.pgexplainer_epochs, lr=0.003),
                explanation_type='phenomenon',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw'
                ),
                node_mask_type=None,
                edge_mask_type='object',
            )

        if self.explanation_mode == 'GradCAM':
            gradcam = GradCAM(model=model)

        if self.explanation_mode == 'CAM':
            cam = CAM(model=model)    

        if self.explanation_mode == 'Grad':
            grad = GradientExplainer(model=model)
        for i in range(2):
            if i == 0:
                dataset = shadow_dataset
            else:
                dataset = test_dataset
            dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
            
            results = []
            total_graph_idx = 0

            # Iterate through the dataset
            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(self.device)
                if batch_data.x is None:
                    batch_data.x = torch.ones((batch_data.num_nodes, 1)).to(self.device)

                batch = batch_data.batch

                # Get model predictions
                with torch.no_grad():
                    if self.explanation_mode in ['GNNExplainer','PGExplainer']:
                        out = model(batch_data.x, batch_data.edge_index, batch)
                    else:
                        _, out = model(batch_data.x, batch_data.edge_index, batch)
                
                preds = out.argmax(dim=1)

                # Generate explanations
                if self.explanation_mode == 'GNNExplainer':
                    explanations = gnnexplainer(batch_data.x, batch_data.edge_index, batch=batch)
                    node_mask = explanations.node_mask.view(-1)

                if self.explanation_mode == 'PGExplainer':
                    for epoch in range(self.pgexplainer_epochs):
                        pgexplainer.algorithm = pgexplainer.algorithm.to(self.device)
                        loss = pgexplainer.algorithm.train(epoch, model, batch_data.x, batch_data.edge_index, target=preds, batch=batch)
                    explanations = pgexplainer(batch_data.x, batch_data.edge_index, target=preds, batch=batch)

                    edge_mask = explanations.edge_mask
                    edge_index = explanations.edge_index
                    num_nodes = batch_data.x.shape[0]

                    # edge score -> node score
                    node_mask = convert_edge_scores_to_node_scores(edge_mask, edge_index, num_nodes)

                if self.explanation_mode == 'GradCAM':
                    explanations = gradcam.get_gradcam_scores(batch_data, preds)
                    node_mask = explanations

                if self.explanation_mode == 'CAM':
                    explanations = cam.get_cam_scores(preds, batch)
                    node_mask = explanations

                if self.explanation_mode == 'Grad':
                    explanations = grad.get_gradient_scores(batch_data, preds)
                    node_mask = explanations

                # Split batch data
                original_graphs = batch_data.to_data_list()
                batch_preds = preds.tolist()

                # Get the number of nodes per graph
                num_nodes_per_graph = batch_data.ptr[1:] - batch_data.ptr[:-1]
                node_masks_list = torch.split(node_mask, num_nodes_per_graph.tolist())

                # Process each graph
                for idx_in_batch, (original_data, pred, node_m) in enumerate(zip(original_graphs, batch_preds, node_masks_list)):
                    # Move data back to CPU
                    original_data = original_data.to('cpu')
                    node_m = node_m.to('cpu')

                    results.append({
                        'original_data': original_data,
                        'pred': pred,
                        'node_mask': node_m
                    })
                    total_graph_idx += 1

                print(f"Processed {total_graph_idx}/{len(dataset)} graphs.")

            if i == 0:
                queried_shadow_dataset = results
            else:
                queried_test_dataset = results
        return queried_shadow_dataset, queried_test_dataset
    
    def _train_attack_model(self,queried_shadow,queried_dataset_test):
        n = len(queried_shadow)
        n_query = max(1, int(round(n * self.query_shadow_ratio)))

        idx = list(range(n))
        rng = random.Random(self.seed)
        rng.shuffle(idx)

        qidx = idx[:n_query]                         # the queried subset
        n_val = 0
        if n_query > 1:
            n_val = min(max(1, int(round(n_query * self.shadow_val_ratio))), n_query - 1)

        val_idx = set(qidx[:n_val])
        train_idx = qidx[n_val:]

        # If edge case leaves train empty, move one from val → train
        if len(train_idx) == 0:
            train_idx = [qidx[-1]]
            val_idx = set(qidx[:-1])

        queried_dataset_val   = [queried_shadow[i] for i in val_idx]
        queried_dataset_train = [queried_shadow[i] for i in train_idx]
        
        print(f"Shadow Train Dataset Size: {len(queried_dataset_train)}")
        print(f"Shadow Val Dataset Size: {len(queried_dataset_val)}")
        print(f"Shadow Test Dataset Size: {len(queried_dataset_test)}")
        
        if self.gnn_backbone == 'GIN':
            encoder = GINGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer
            ).to(self.device)
        elif self.gnn_backbone == 'GCN':
            encoder = GCNGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer
            ).to(self.device)
        elif self.gnn_backbone == 'GAT':
            encoder = GATGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer,
                heads=self.gat_heads
            ).to(self.device)
        elif self.gnn_backbone == 'GraphSAGE':
            encoder = GraphSAGEGraphClassification(
                input_dim=self.num_features,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_layer
            ).to(self.device)
        else:
            raise ValueError(f"Invalid GNN backbone specified: {self.gnn_backbone}. Expected 'GIN', 'GCN', or 'GAT', or 'GraphSAGE'.")

        predictor = Classifier(
            input_dim=self.gnn_hidden_dim,
            output_dim=self.num_classes
        ).to(self.device)

        model = SurrogateModelGraphClassification(encoder=encoder, predictor=predictor).to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        ranknet_loss_fn = RankNetLoss().to(self.device)

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)
        
        processed_train_dataset = process_query_dataset(queried_dataset_train)
        processed_val_dataset = process_query_dataset(queried_dataset_val)
        processed_test_dataset = process_query_dataset(queried_dataset_test)

        augmentor = DataAugmentor()

        val_loader = DataLoader(processed_val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate)
        test_loader = DataLoader(processed_test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate)

        best_val_auc = -math.inf
        best_model_state = None
        
        print("Starting Surrogate Model Training...")
        with tqdm(total=self.epochs, desc=f'Training (seed={self.seed})') as epoch_pbar:
            for epoch_num in range(1, self.epochs + 1):
                # --- 1. Data augmentation ---
                augmented_data = augment(
                    dataset=processed_train_dataset,
                    augmentor=augmentor,
                    augmentation_ratio=self.augmentation_ratio,
                    operation_ratio=self.operation_ratio,
                    augmentation_type=self.augmentation_type
                )

                combined_train_dataset = processed_train_dataset + augmented_data

                combined_train_loader = DataLoader(combined_train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                collate_fn=custom_collate)

                # --- 2. Training ---
                train_loss_pred, train_ranknet_loss = train(
                    model, combined_train_loader, optimizer, self.device,
                    align_weight=self.align_weight,
                    criterion=criterion,
                    ranknet_loss_fn=ranknet_loss_fn
                )

                # --- 3. Validation ---
                val_acc, val_auc = eval(
                    model, val_loader, self.device
                )

                epoch_pbar.set_postfix({
                    'Train Pred Loss': f'{train_loss_pred:.4f}',
                    'Train RankNet Loss': f'{train_ranknet_loss:.4f}',
                    'Val Acc': f'{val_acc:.4f}',
                    'Val AUC': f'{val_auc:.4f}'
                })
                epoch_pbar.update(1)

                # --- 4. Save best model (based on validation AUC) ---
                if not math.isnan(val_auc) and val_auc >= best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict()

        # Prepare defaults so we always record something
        run_metrics = {
            'seed': self.seed,
            'best_val_auc': float(best_val_auc) if best_val_auc != -math.inf else float('nan'),
            'test_acc': float('nan'),
            'test_auc': float('nan'),
            'fidelity_score': float('nan'),
            'order_accuracy': float('nan'),
            'rank_correlation': float('nan'),
        }

        # Evaluate best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

            test_acc, test_auc, fidelity_score, order_accuracy, rank_correlation = test(model, test_loader, self.device)

            print(f"\n[seed={self.seed}] Best Validation AUC: {best_val_auc:.4f}")
            print(f"[seed={self.seed}] Test Accuracy: {test_acc:.4f}")
            print(f"[seed={self.seed}] Test AUC: {test_auc:.4f}")
            print(f"[seed={self.seed}] Fidelity Score: {fidelity_score:.4f}")
            print(f"[seed={self.seed}] Order Accuracy: {order_accuracy:.4f}")
            print(f"[seed={self.seed}] Rank Correlation: {rank_correlation:.4f}")

            run_metrics.update({
                'test_acc': float(test_acc),
                'test_auc': float(test_auc),
                'fidelity_score': float(fidelity_score),
                'order_accuracy': float(order_accuracy),
                'rank_correlation': float(rank_correlation),
            })
        else:
            print(f"[seed={self.seed}] No improvement in validation AUC during training.")
        return model
    
    def attack(self):
        target_train_dataset,target_val_dataset,shadow_dataset,test_dataset = self.prepare_data()
        target_model = self._train_target_model(target_train_dataset,target_val_dataset,test_dataset)
        queried_shadow_dataset,queried_test_dataset = self.prepare_shadow_data(target_model,shadow_dataset,test_dataset)
        surrogate_model = self._train_attack_model(queried_shadow_dataset,queried_test_dataset)
        
        pass
        