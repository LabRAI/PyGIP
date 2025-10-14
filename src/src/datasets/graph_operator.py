import copy
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import src.models.gnn as gnn
from torch_geometric.utils import add_self_loops, subgraph


class Graph_self():
    def __init__(self, features, edge_index, labels):
        self.features = copy.deepcopy(features)
        self.adjacency = copy.deepcopy(edge_index)
        self.labels = copy.deepcopy(labels)
        self.node_num = len(self.labels)
        self.feat_dim = len(self.features[0])
        self.get_class_num()
        self.set_adj_mat()

    def get_class_num(self):
        labels = self.labels.tolist()
        labels = set(labels)
        self.class_num = len(labels)

    def set_adj_mat(self):
        self.adj_matrix = torch.zeros([self.node_num, self.node_num])
        for i in range(self.node_num):
            source_node = self.adjacency[0][i]
            target_node = self.adjacency[1][i]
            self.adj_matrix[source_node, target_node] = 1

    def __len__(self):
        return self.node_num
    
    def __getitem__(self, index):
        return [self.features[index], self.adj_matrix[index], self.labels[index]]
    

def sort_features(args, feat_num, graph_data, original_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Get the predictions of nodes from the original model
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    loss_fn = F.cross_entropy

    original_model.eval()
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    labels = graph_data.labels.to(device)
    _, output = original_model(input_data)
    original_predictions = predict_fn(output)

    chosen_feat = list()
    candidate_feat = copy.deepcopy(graph_data.features)

    for iter in range(feat_num):

        feat_fidelity = dict()
        for feat_index in range(graph_data.feat_dim):
            if feat_index in chosen_feat:
                continue
            print(feat_index)
            selection_model = None
            if args.benign_model == 'gcn':
                selection_model = gnn.GCN(iter+1, graph_data.class_num, hidden_dim=args.benign_hidden_dim)
            elif args.benign_model == 'sage':
                selection_model = gnn.GraphSAGE(iter+1, graph_data.class_num, hidden_dim=args.benign_hidden_dim)
            elif args.benign_model == 'gat':
                selection_model = gnn.GAT(iter+1, graph_data.class_num, hidden_dim=args.benign_hidden_dim)
            selection_model.to(device)
            optimizer = torch.optim.Adam(selection_model.parameters(), lr=args.benign_lr, weight_decay=args.benign_weight_decay, betas=(0.5, 0.999))
            scheduler = lr_scheduler.MultiStepLR(optimizer, args.benign_lr_decay_steps, gamma=0.1)

            this_loop_feat = copy.deepcopy(chosen_feat)
            this_loop_feat.append(feat_index)
            selected_feat = candidate_feat[:, this_loop_feat]
            this_loop_input_data = selected_feat.to(device), graph_data.adjacency.to(device)
            this_loop_labels = graph_data.labels.to(device)
            selection_model.train()
            for epoch in range(args.benign_train_epochs):
                optimizer.zero_grad()
                _, output = selection_model(this_loop_input_data)
                loss = loss_fn(output[graph_data.benign_train_mask], this_loop_labels[graph_data.benign_train_mask])
                loss.backward()
                optimizer.step()
                scheduler.step()

            selection_model.eval()
            _, output = selection_model(this_loop_input_data)
            pred = predict_fn(output)
            final_pred = pred[graph_data.benign_train_mask]
            original_pred = original_predictions[graph_data.benign_train_mask]
            correct_num = 0
            for i in range(final_pred.shape[0]):
                if final_pred[i, 0] == original_pred[i, 0]:
                    correct_num += 1
                test_acc = correct_num / final_pred.shape[0] * 100
            feat_fidelity.update({feat_index: test_acc})

        feat_fidelity = sorted(feat_fidelity.items(), key=lambda x:x[1], reverse=True)
        most_important_feat = feat_fidelity[0][0]
        chosen_feat.append(most_important_feat)

    print(chosen_feat)
    return chosen_feat


def split_subgraph(graph):
    temp_edge_index = add_self_loops(graph.adjacency)[0]
    target_edge_index = subgraph(torch.as_tensor(graph.target_nodes_index), temp_edge_index, relabel_nodes=True)[0]
    shadow_edge_index = subgraph(torch.as_tensor(graph.shadow_nodes_index), temp_edge_index, relabel_nodes=True)[0]
    attacker_edge_index = subgraph(torch.as_tensor(graph.attacker_nodes_index), temp_edge_index, relabel_nodes=True)[0]
    test_edge_index = subgraph(torch.as_tensor(graph.test_nodes_index), temp_edge_index, relabel_nodes=True)[0]

    target_features = graph.features[graph.target_nodes_index]
    shadow_features = graph.features[graph.shadow_nodes_index]
    attacker_features = graph.features[graph.attacker_nodes_index]
    test_features = graph.features[graph.test_nodes_index]

    target_labels = graph.labels[graph.target_nodes_index]
    shadow_labels = graph.labels[graph.shadow_nodes_index]
    attacker_labels = graph.labels[graph.attacker_nodes_index]
    test_labels = graph.labels[graph.test_nodes_index]

    target_subgraph = Graph_self(target_features, target_edge_index, target_labels)
    shadow_subgraph = Graph_self(shadow_features, shadow_edge_index, shadow_labels)
    attacker_subgraph = Graph_self(attacker_features, attacker_edge_index, attacker_labels)
    test_subgraph = Graph_self(test_features, test_edge_index, test_labels)

    return target_subgraph, shadow_subgraph, attacker_subgraph, test_subgraph