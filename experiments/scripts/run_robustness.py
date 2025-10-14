import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from utils.config import parse_args
import src.datasets.datareader
import src.datasets.graph_operator
from verifier.verification_cfg import multiple_experiments
import extraction
import yaml
import torch
import random
import math

def fine_tune(args, load_root, specific_mask_mag):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    substring_path = load_root.split('/')
    substring_path.remove('..')
    substring_path.remove('')
    substring_path.remove('temp_results')

    load_folder_root, save_folder_root = list(), list()
    save_root = '../robustness_results/fine_tune'
    for i in substring_path:
        save_root = os.path.join(save_root, i)
    with os.scandir(load_root) as itr_0:
        for target_model_folder in itr_0:
            sub_load_root = os.path.join(load_root, target_model_folder.name)
            sub_save_root = os.path.join(save_root, target_model_folder.name)
            with os.scandir(sub_load_root) as itr_1:
                for mask_mag in itr_1:
                    if mask_mag.name != specific_mask_mag:
                        continue
                    final_load_root = os.path.join(sub_load_root, mask_mag.name)
                    final_save_root = os.path.join(sub_save_root, mask_mag.name)
                    load_folder_root.append(final_load_root)
                    save_folder_root.append(final_save_root)

                    if not os.path.exists(final_save_root):
                        os.makedirs(final_save_root)

    data = src.datasets.datareader.get_data(args)
    graph_data = src.datasets.datareader.GraphData(data, args)
    if args.task_type == 'inductive':
        _, _, graph_data, _ = src.datasets.graph_operator.split_subgraph(graph_data)
    loss_fn = torch.nn.CrossEntropyLoss()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]

    for folder_index in range(len(load_folder_root)):
        models_folder_path = load_folder_root[folder_index]
        with os.scandir(models_folder_path) as itr:
            for entry in itr:
                if 'train' in entry.name:
                    continue

                original_model_load_path = os.path.join(models_folder_path, entry.name)
                fine_tune_model_save_path = os.path.join(save_folder_root[folder_index], entry.name)

                gnn_model = torch.load(original_model_load_path)
                gnn_model.to(device)
                gnn_model.train()
                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr)

                # use testing dataset to fine-tune extraction models
                last_train_acc = 0.0
                if args.task_type == 'transductive':
                    for epoch in range(args.benign_train_epochs):
                        optimizer.zero_grad()
                        input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
                        labels = graph_data.labels.to(device)
                        _, output = gnn_model(input_data)
                        loss = loss_fn(output[graph_data.attacker_nodes_index], labels[graph_data.attacker_nodes_index])
                        loss.backward()
                        optimizer.step()

                        train_correct_num = 0
                        if (epoch + 1) % 100 == 0:
                            _, output = gnn_model(input_data)
                            pred = predict_fn(output)
                            train_pred = pred[graph_data.attacker_nodes_index]
                            train_labels = graph_data.labels[graph_data.attacker_nodes_index]
                            for i in range(train_pred.shape[0]):
                                if train_pred[i, 0] == train_labels[i]:
                                    train_correct_num += 1
                            train_acc = train_correct_num / train_pred.shape[0] * 100
                            if last_train_acc == 0.0:
                                last_train_acc = train_acc
                            else:
                                train_acc_diff = (train_acc - last_train_acc) / last_train_acc * 100
                                if train_acc_diff <= 0.5: #0.5%
                                    break
                                else:
                                    last_train_acc = train_acc
                elif args.task_type == 'inductive':
                    for epoch in range(args.benign_train_epochs):
                        optimizer.zero_grad()
                        input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
                        labels = graph_data.labels.to(device)
                        _, output = gnn_model(input_data)
                        loss = loss_fn(output, labels)
                        loss.backward()
                        optimizer.step()

                        train_correct_num = 0
                        if (epoch + 1) % 100 == 0:
                            _, output = gnn_model(input_data)
                            predictions = predict_fn(output)
                    
                            for i in range(predictions.shape[0]):
                                if predictions[i, 0] == labels[i]:
                                    train_correct_num += 1
                            train_acc = train_correct_num / predictions.shape[0] * 100

                            if last_train_acc == 0.0:
                                last_train_acc = train_acc
                            else:
                                train_acc_diff = (train_acc - last_train_acc) / last_train_acc * 100
                                if train_acc_diff <= 0.5: #0.5%
                                    break
                                else:
                                    last_train_acc = train_acc

                torch.save(gnn_model, fine_tune_model_save_path)


def prune(args, load_root, specific_mask_mag):
    substring_path = load_root.split('/')
    substring_path.remove('..')
    substring_path.remove('')
    substring_path.remove('temp_results')

    load_folder_root, save_folder_root = list(), list()
    save_root = '../robustness_results/prune'
    save_root = os.path.join(save_root, str(args.prune_weight_ratio))

    for i in substring_path:
        save_root = os.path.join(save_root, i)
    with os.scandir(load_root) as itr_0:
        for target_model_folder in itr_0:
            sub_load_root = os.path.join(load_root, target_model_folder.name)
            sub_save_root = os.path.join(save_root, target_model_folder.name)
            with os.scandir(sub_load_root) as itr_1:
                for mask_mag in itr_1:
                    if mask_mag.name != specific_mask_mag:
                        continue
                    final_load_root = os.path.join(sub_load_root, mask_mag.name)
                    final_save_root = os.path.join(sub_save_root, mask_mag.name)
                    load_folder_root.append(final_load_root)
                    save_folder_root.append(final_save_root)

                    if not os.path.exists(final_save_root):
                        os.makedirs(final_save_root)

    for folder_index in range(len(load_folder_root)):
        models_folder_path = load_folder_root[folder_index]
        with os.scandir(models_folder_path) as itr:
            for entry in itr:
                if 'train' in entry.name:
                    continue
                original_model_load_path = os.path.join(models_folder_path, entry.name)
                prune_model_save_path = os.path.join(save_folder_root[folder_index], entry.name)

                gnn_model = torch.load(original_model_load_path)

                for name, param in gnn_model.named_parameters():
                    if 'fc' in name:
                        continue
                    if 'bias' in name:
                        continue
                    
                    original_param_shape = param.data.shape
                    temp_param = torch.flatten(param.data)
                    prune_num = math.floor(temp_param.shape[0] * args.prune_weight_ratio)
                    prune_index = [i for i in range(temp_param.shape[0])]
                    random.shuffle(prune_index)
                    prune_index = prune_index[:prune_num]
                    temp_param[prune_index] = 0
                    param.data = temp_param.reshape(original_param_shape)
                    
                torch.save(gnn_model, prune_model_save_path)


def double_extraction(args, load_root, specific_mask_mag):
    substring_path = load_root.split('/')
    substring_path.remove('..')
    substring_path.remove('')
    substring_path.remove('temp_results')

    load_folder_root, save_folder_root = list(), list()
    save_root = '../robustness_results/double_extraction'
    for i in substring_path:
        save_root = os.path.join(save_root, i)
    with os.scandir(load_root) as itr_0:
        for target_model_folder in itr_0:
            sub_load_root = os.path.join(load_root, target_model_folder.name)
            sub_save_root = os.path.join(save_root, target_model_folder.name)
            with os.scandir(sub_load_root) as itr_1:
                for mask_mag in itr_1:
                    if mask_mag.name != specific_mask_mag:
                        continue
                    final_load_root = os.path.join(sub_load_root, mask_mag.name)
                    final_save_root = os.path.join(sub_save_root, mask_mag.name)
                    load_folder_root.append(final_load_root)
                    save_folder_root.append(final_save_root)

                    if not os.path.exists(final_save_root):
                        os.makedirs(final_save_root)

    data = src.datasets.datareader.get_data(args)
    graph_data = src.datasets.datareader.GraphData(data, args)
    if args.task_type == 'inductive':
        target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data = src.datasets.graph_operator.split_subgraph(graph_data)
        graph_data = [target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data]

    for folder_index in range(len(load_folder_root)):
        models_folder_path = load_folder_root[folder_index]
        with os.scandir(models_folder_path) as itr:
            for entry in itr:
                if 'train' in entry.name:
                    continue

                original_model_load_path = os.path.join(models_folder_path, entry.name)
                double_extraction_model_save_path = os.path.join(save_folder_root[folder_index], entry.name)
                gnn_model = torch.load(original_model_load_path)
                
                arch_and_layers = entry.name.split('_')
                arch_and_layers[-1] = arch_and_layers[-1].strip('.pt')
                args.extraction_model = arch_and_layers[1]
                layers = list()
                for i in arch_and_layers[2:]:
                    layers.append(int(i))
                args.extraction_hidden_dim = layers

                _, _, _ = extraction.run(args, double_extraction_model_save_path, graph_data, gnn_model, 'test')


if __name__ == '__main__':
    args = parse_args()

    # Get absolute path to the config folder
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))

    # Build the path to global_cfg.yaml
    global_cfg_file = os.path.join(config_path, "global_cfg.yaml")

    if not os.path.exists(global_cfg_file):
        raise FileNotFoundError(f"Global config file not found: {global_cfg_file}")

    with open(global_cfg_file, "r") as file:
        global_cfg = yaml.safe_load(file)

    # transductive
    args.task_type = 'transductive'
    args.mask_feat_ratio = 0.1
    path = '../temp_results/diff/model_states/{}/transductive/extraction_models/random_mask/'.format(global_cfg['dataset'])
    transductive_mask_mag = '1.0_{}'.format(args.mask_feat_ratio)
    # fine_tune(args, path, transductive_mask_mag)

    global_cfg["test_save_root"] = "../robustness_results/fine_tune/diff/model_states/"
    global_cfg["res_path"] = "../robustness_results/res/fine_tune"
    # multiple_experiments(args, global_cfg)
    

    for prune_ratio in [0.6, 0.7]:
        args.prune_weight_ratio = prune_ratio
        prune(args, path, transductive_mask_mag)
        global_cfg["test_save_root"] = "../robustness_results/prune/{}/diff/model_states/".format(prune_ratio)
        global_cfg["res_path"] = "../robustness_results/res/prune/{}".format(prune_ratio)
        multiple_experiments(args, global_cfg)
        

    # double_extraction(args, path, transductive_mask_mag)
    global_cfg["test_save_root"] = "../robustness_results/double_extraction/diff/model_states/"
    global_cfg["res_path"] = "../robustness_results/res/double_extraction"
    # multiple_experiments(args, global_cfg)

    
    # inductive
    args.task_type = 'inductive'
    args.mask_feat_ratio = 0.05
    path = '../temp_results/diff/model_states/{}/inductive/extraction_models/random_mask/'.format(global_cfg['dataset'])
    inductive_mask_mag = '1.0_{}'.format(args.mask_feat_ratio)
    # fine_tune(args, path, inductive_mask_mag)

    global_cfg["test_save_root"] = "../robustness_results/fine_tune/diff/model_states/"
    global_cfg["res_path"] = "../robustness_results/res/fine_tune"
    # multiple_experiments(args, global_cfg)

    
    for prune_ratio in [0.6, 0.7]:
        args.prune_weight_ratio = prune_ratio
        prune(args, path, inductive_mask_mag)
        global_cfg["test_save_root"] = "../robustness_results/prune/{}/diff/model_states/".format(prune_ratio)
        global_cfg["res_path"] = "../robustness_results/res/prune/{}".format(prune_ratio)
        multiple_experiments(args, global_cfg)

    
    # double_extraction(args, path, inductive_mask_mag)
    global_cfg["test_save_root"] = "../robustness_results/double_extraction/diff/model_states/"
    global_cfg["res_path"] = "../robustness_results/res/double_extraction"
    # multiple_experiments(args, global_cfg)
    