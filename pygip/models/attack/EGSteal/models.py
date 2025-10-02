import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv, SAGEConv, global_mean_pool

## Graph Classification Models
class SurrogateModelGraphClassification(nn.Module):
    def __init__(self, encoder, predictor):
        super(SurrogateModelGraphClassification, self).__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x, edge_index, batch):
        node_embeddings, graph_embeddings = self.encoder(x, edge_index, batch)
        out = self.predictor(graph_embeddings)
        return node_embeddings, out
    


class TargetModelGraphClassification(nn.Module):
    def __init__(self, encoder, predictor, explanation_mode):
        super(TargetModelGraphClassification, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.explanation_mode = explanation_mode

    def forward(self, x, edge_index, batch):
        node_embeddings, graph_embeddings = self.encoder(x, edge_index, batch)
        out = self.predictor(graph_embeddings)
        if self.explanation_mode in ['GNNExplainer','PGExplainer']:
            return out
        return node_embeddings, out
    


class GCNGraphClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNGraphClassification, self).__init__()
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(input_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)
        
        return node_embeddings, graph_embeddings



class GraphSAGEGraphClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GraphSAGEGraphClassification, self).__init__()
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)

        return node_embeddings, graph_embeddings
    


class GATGraphClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, heads=4):
        super(GATGraphClassification, self).__init__()
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)
        
        return node_embeddings, graph_embeddings
    


class GINGraphClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GINGraphClassification, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                nn_seq = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                nn_seq = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            self.convs.append(GINConv(nn_seq))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)

        return node_embeddings, graph_embeddings



class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)



class CAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.classifier_weights = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        classifier = self.model.predictor

        if isinstance(classifier, nn.Linear):
            self.classifier_weights = classifier.weight  # [num_classes, hidden_dim]
        elif hasattr(classifier, 'fc') and isinstance(classifier.fc, nn.Linear):
            self.classifier_weights = classifier.fc.weight  # [num_classes, hidden_dim]
        else:
            raise ValueError("Classifier should be an instance of nn.Linear or have a linear layer named 'fc'")

        last_conv = self.model.encoder.convs[-1]
        last_conv.register_forward_hook(forward_hook)

    def get_cam_scores(self, target_classes, batch_ids):
        """
        Generate CAM scores based on captured activations and classifier weights.

        Parameters:
        - target_classes: [batch_size] tensor, each element is a class index
        - batch_ids: [total_num_nodes] tensor, indicating the graph index for each node

        Returns:
        - cam_scores: list, each element is the CAM score for the corresponding graph [num_nodes_in_graph]
        """
        if self.activations is None:
            raise ValueError("No activations recorded. Ensure a forward pass has been done before generating CAM.")

        cam_scores = []

        num_graphs = batch_ids.max().item() + 1
        for graph_id in range(num_graphs):
            cls = target_classes[graph_id].item()
            weight = self.classifier_weights[cls]  # [hidden_dim]   w^c_k

            # Get node indices belonging to the current graph
            node_indices = (batch_ids == graph_id).nonzero(as_tuple=False).squeeze()
            if node_indices.numel() == 0:
                cam = torch.tensor([], device=self.activations.device)
            else:
                activation = self.activations[node_indices]  # [num_nodes_in_graph, hidden_dim]   F^l_k(X, A) = σ(V F^(l-1)(X, A)W^l_k)
                cam = torch.matmul(activation, weight)  # [num_nodes_in_graph]   L^c_CAM[n] = ReLU(∑_k w^c_k F^L_{k,n}(X, A))
                
            if cam.dim() == 0:  # single node graph
                cam = cam.unsqueeze(0)

            cam_scores.append(cam)


        cam_scores = torch.cat(cam_scores, dim=0)

        return cam_scores
    


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        last_conv = self.model.encoder.convs[-1]
        last_conv.register_forward_hook(forward_hook)
    
    def get_gradcam_scores(self, input, target_class):
        self.model.zero_grad()
        
        node_embeddings, output = self.model(input.x, input.edge_index, input.batch)
        
        batch_size = len(target_class)
        scores = output[range(batch_size), target_class]
        
        gradients = torch.autograd.grad(
            scores, 
            self.activations,
            grad_outputs=torch.ones_like(scores),
            retain_graph=True
        )[0]

        if self.activations is None:
            raise ValueError("Activations were not captured")
        
        weights = self.compute_weights(gradients, input.batch)
        
        # Calculate Grad-CAM scores
        batch = input.batch
        num_graphs = batch.max().item() + 1
        gradcam_scores = []
        
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            if mask.any():
                curr_activations = self.activations[mask]  # [num_nodes, hidden_dim]
                curr_weights = weights[graph_id]  # [hidden_dim]
                
                gradcam = torch.matmul(curr_activations, curr_weights)  # [num_nodes]
                gradcam = F.relu(gradcam)
                    
                gradcam_scores.append(gradcam)
        
        return torch.cat(gradcam_scores)

    def compute_weights(self, gradients, batch):
        num_graphs = batch.max().item() + 1
        weights = []
        
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            graph_grads = gradients[mask]
            alpha = graph_grads.mean(dim=0)
            weights.append(alpha)
            
        return torch.stack(weights)



class GradientExplainer:
    def __init__(self, model):
        self.model = model
        
    def get_gradient_scores(self, input_data, target_classes):
        input_data.x.requires_grad = True
        self.model.zero_grad()
        
        # Forward
        _, output = self.model(input_data.x, input_data.edge_index, input_data.batch)
        
        # Calculate gradients for each graph
        batch = input_data.batch
        num_graphs = batch.max().item() + 1
        normalized_scores = []
        
        # Calculate gradients for each graph's target class
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            target_class = target_classes[graph_id]
            score = output[graph_id, target_class]
            
            grad = torch.autograd.grad(score, input_data.x, 
                                     retain_graph=True, 
                                     create_graph=False)[0]  # [num_nodes, feature_dim]
            
            curr_grad = grad[mask]
            relu_grad = F.relu(curr_grad)
            scores = torch.norm(relu_grad, p=2, dim=1)
            
            normalized_scores.append(scores)
            
        return torch.cat(normalized_scores)
