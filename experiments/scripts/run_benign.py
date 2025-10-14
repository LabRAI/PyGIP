import torch
import copy
from src.utils.config import parse_args
import src.datasets.datareader
import src.models.gnn
import torch.nn.functional as F
from src.datasets.graph_operator import split_subgraph
from pathlib import Path
import os
import random


def transductive_train(args, model_save_path, graph_data, process):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if graph_data is None:
        data = src.datasets.datareader.get_data(args)
        gdata = src.datasets.datareader.GraphData(data, args)
    else:
        gdata = graph_data
    
    path = Path(model_save_path)
    os.makedirs(path.parent, exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]

    # Load existing model or create new
    if path.is_file():
        gnn_model = torch.load(model_save_path)
    else:
        if args.benign_model == 'gcn':
            gnn_model = src.models.gnn.GCN(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sage':
            gnn_model = src.models.gnn.GraphSage(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gat':
            gnn_model = src.models.gnn.GAT(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gin':
            gnn_model = src.models.gnn.GIN(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sgc':
            gnn_model = src.models.gnn.SGC(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)

        gnn_model.to(device)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr)
        last_train_acc = 0.0

        if process == 'test':
            train_nodes_index = [i for i in range(gdata.node_num)]
            random.shuffle(train_nodes_index)
            train_nodes_index = train_nodes_index[:len(gdata.target_nodes_index)]

        for epoch in range(args.benign_train_epochs):
            gnn_model.train()
            optimizer.zero_grad()
            input_data = gdata.features.to(device), gdata.adjacency.to(device)
            labels = gdata.labels.to(device)
            _, output = gnn_model(input_data)
            loss = loss_fn(output[train_nodes_index] if process=='test' else output[gdata.target_nodes_index],
                           labels[train_nodes_index] if process=='test' else labels[gdata.target_nodes_index])
            loss.backward()
            optimizer.step()

            # Early stopping check every 50 epochs
            if (epoch + 1) % 50 == 0:
                gnn_model.eval()
                _, output = gnn_model(input_data)
                pred = predict_fn(output)
                train_pred = pred[train_nodes_index] if process=='test' else pred[gdata.target_nodes_index]
                train_labels = labels[train_nodes_index] if process=='test' else labels[gdata.target_nodes_index]
                correct = (train_pred.squeeze() == train_labels).sum().item()
                train_acc = correct / train_pred.shape[0] * 100

                if last_train_acc == 0.0:
                    last_train_acc = train_acc
                else:
                    if abs(train_acc - last_train_acc) / last_train_acc * 100 <= 0.5:
                        break
                    last_train_acc = train_acc

        torch.save(gnn_model, model_save_path)

    # Test accuracy
    gnn_model.eval()
    input_data = gdata.features.to(device), gdata.adjacency.to(device)
    _, output = gnn_model(input_data)
    pred = predict_fn(output)
    test_pred = pred[gdata.test_nodes_index]
    test_labels = gdata.labels[gdata.test_nodes_index]
    test_acc = (test_pred.squeeze() == test_labels).sum().item() / test_pred.shape[0] * 100

    return gdata, gnn_model, round(test_acc, 3)


def inductive_train(args, model_save_path, graph_data, process):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if graph_data is None:
        data = src.datasets.datareader.get_data(args)
        gdata = src.datasets.datareader.GraphData(data, args)
        target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data = split_subgraph(gdata)
    else:
        target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data = graph_data

    path = Path(model_save_path)
    os.makedirs(path.parent, exist_ok=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]

    if path.is_file():
        gnn_model = torch.load(model_save_path)
    else:
        if args.benign_model == 'gcn':
            gnn_model = src.models.gnn.GCN(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sage':
            gnn_model = src.models.gnn.GraphSage(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gat':
            gnn_model = src.models.gnn.GAT(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gin':
            gnn_model = src.models.gnn.GIN(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sgc':
            gnn_model = src.models.gnn.SGC(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)

        gnn_model.to(device)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr)
        last_train_acc = 0.0

        for epoch in range(args.benign_train_epochs):
            gnn_model.train()
            optimizer.zero_grad()
            input_data = target_graph_data.features.to(device), target_graph_data.adjacency.to(device)
            labels = target_graph_data.labels.to(device)
            _, output = gnn_model(input_data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                gnn_model.eval()
                _, output = gnn_model(input_data)
                pred = predict_fn(output)
                correct = (pred.squeeze() == labels).sum().item()
                train_acc = correct / labels.shape[0] * 100

                if last_train_acc == 0.0:
                    last_train_acc = train_acc
                else:
                    if abs(train_acc - last_train_acc) / last_train_acc * 100 <= 0.5:
                        break
                    last_train_acc = train_acc

        torch.save(gnn_model, model_save_path)

    gnn_model.eval()
    input_data = test_graph_data.features.to(device), test_graph_data.adjacency.to(device)
    _, output = gnn_model(input_data)
    pred = predict_fn(output)
    correct = (pred.squeeze() == test_graph_data.labels).sum().item()
    test_acc = correct / test_graph_data.labels.shape[0] * 100

    return [target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data], gnn_model, round(test_acc, 3)


def run(args, model_save_path, given_graph_data=None, process=None):
    if args.task_type == 'transductive':
        return transductive_train(args, model_save_path, given_graph_data, process)
    elif args.task_type == 'inductive':
        return inductive_train(args, model_save_path, given_graph_data, process)


if __name__ == '__main__':
    args = parse_args()
    
    # Example: auto-create dataset folders and save models where run_robustness expects
    dataset = args.dataset
    for task in ['transductive', 'inductive']:
        folder = f"../temp_results/diff/model_states/{dataset}/{task}/extraction_models/random_mask/"
        os.makedirs(folder, exist_ok=True)
        model_file = os.path.join(folder, f"{args.benign_model}_model.pt")
        print(f"Training and saving model to: {model_file}")
        graph_data, gnn_model, test_acc = run(args, model_file, process='train')
        print(f"{task.capitalize()} training finished. Test accuracy: {test_acc}")
