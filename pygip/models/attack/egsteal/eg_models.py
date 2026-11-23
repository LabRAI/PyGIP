import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool


class SurrogateModel(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x, edge_index, batch):
        node_embeddings, graph_embeddings = self.encoder(x, edge_index, batch)
        out = self.predictor(graph_embeddings)
        return node_embeddings, out


class TargetModel(nn.Module):
    def __init__(self, encoder, predictor, explanation_mode: str):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.explanation_mode = explanation_mode

    def forward(self, x, edge_index, batch):
        node_embeddings, graph_embeddings = self.encoder(x, edge_index, batch)
        out = self.predictor(graph_embeddings)
        if self.explanation_mode in ["GNNExplainer", "PGExplainer"]:
            return out
        return node_embeddings, out


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(input_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)
        return node_embeddings, graph_embeddings


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)
        return node_embeddings, graph_embeddings


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)
        return node_embeddings, graph_embeddings


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                nn_seq = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                nn_seq = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            self.convs.append(GINConv(nn_seq))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        node_embeddings = x
        graph_embeddings = global_mean_pool(x, batch)
        return node_embeddings, graph_embeddings


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
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
            self.classifier_weights = classifier.weight
        elif hasattr(classifier, "fc") and isinstance(classifier.fc, nn.Linear):
            self.classifier_weights = classifier.fc.weight
        else:
            raise ValueError("Classifier must be nn.Linear or have .fc as nn.Linear")

        last_conv = self.model.encoder.convs[-1]
        last_conv.register_forward_hook(forward_hook)

    def get_cam_scores(self, target_classes, batch_ids):
        if self.activations is None:
            raise ValueError("No activations recorded. Run a forward pass first.")

        num_graphs = batch_ids.max().item() + 1
        cam_scores = []
        for graph_id in range(num_graphs):
            cls = target_classes[graph_id].item()
            weight = self.classifier_weights[cls]
            node_indices = (batch_ids == graph_id).nonzero(as_tuple=False).squeeze()
            if node_indices.numel() == 0:
                cam = torch.tensor([], device=self.activations.device)
            else:
                activation = self.activations[node_indices]
                cam = torch.matmul(activation, weight)
            if cam.dim() == 0:
                cam = cam.unsqueeze(0)
            cam_scores.append(cam)

        return torch.cat(cam_scores, dim=0)


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

    def compute_weights(self, gradients, batch):
        num_graphs = batch.max().item() + 1
        weights = []
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            graph_grads = gradients[mask]
            alpha = graph_grads.mean(dim=0)
            weights.append(alpha)
        return torch.stack(weights)

    def get_gradcam_scores(self, input_data, target_classes):
        self.model.zero_grad()
        _, output = self.model(input_data.x, input_data.edge_index, input_data.batch)

        batch_size = len(target_classes)
        scores = output[range(batch_size), target_classes]

        gradients = torch.autograd.grad(
            scores,
            self.activations,
            grad_outputs=torch.ones_like(scores),
            retain_graph=True,
        )[0]

        if self.activations is None:
            raise ValueError("Activations were not captured")

        weights = self.compute_weights(gradients, input_data.batch)

        batch = input_data.batch
        num_graphs = batch.max().item() + 1
        gradcam_scores = []
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            if mask.any():
                curr_activations = self.activations[mask]
                curr_weights = weights[graph_id]
                gradcam = torch.matmul(curr_activations, curr_weights)
                gradcam = F.relu(gradcam)
                gradcam_scores.append(gradcam)

        return torch.cat(gradcam_scores)


class GradientExplainer:
    def __init__(self, model):
        self.model = model

    def get_gradient_scores(self, input_data, target_classes):
        input_data.x.requires_grad = True
        self.model.zero_grad()

        _, output = self.model(input_data.x, input_data.edge_index, input_data.batch)

        batch = input_data.batch
        num_graphs = batch.max().item() + 1
        scores_all = []

        for graph_id in range(num_graphs):
            mask = batch == graph_id
            target_class = target_classes[graph_id]
            score = output[graph_id, target_class]

            grad = torch.autograd.grad(
                score,
                input_data.x,
                retain_graph=True,
                create_graph=False,
            )[0]

            curr_grad = grad[mask]
            relu_grad = F.relu(curr_grad)
            scores = torch.norm(relu_grad, p=2, dim=1)
            scores_all.append(scores)

        return torch.cat(scores_all)

