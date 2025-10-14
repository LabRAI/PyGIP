import json
import torch
import experiments.scripts.run_benign as benign
import extraction
import random
import math
import src.datasets.datareader
from src.extraction import extraction_runner
# import src.models.gnn
# import src.models.extraction
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.ownership_classifier import mlp_nn
import src.masking.boundary as boundary
from statistics import mean
import time
import os
import yaml
import itertools
from datetime import timedelta
import copy

def extract_outputs(graph_data, specific_nodes, independent_model, surrogate_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    independent_model.eval()
    surrogate_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    independent_embedding, independent_logits = independent_model(input_data)
    surrogate_embedding, surrogate_logits = surrogate_model(input_data)

    softmax = torch.nn.Softmax(dim=1)
    independent_prob = softmax(independent_logits)
    surrogate_prob = softmax(surrogate_logits)

    if specific_nodes != None:
        independent_prob = independent_prob[specific_nodes].detach().cpu()
        surrogate_prob = surrogate_prob[specific_nodes].detach().cpu()
        independent_embedding = independent_embedding[specific_nodes].detach().cpu()
        surrogate_embedding = surrogate_embedding[specific_nodes].detach().cpu()
        independent_logits = independent_logits[specific_nodes].detach().cpu()
        surrogate_logits = surrogate_logits[specific_nodes].detach().cpu()

    probability = {'independent': independent_prob, 'surrogate': surrogate_prob}
    embedding = {'independent': independent_embedding, 'surrogate': surrogate_embedding}
    logits = {'independent': independent_logits, 'surrogate': surrogate_logits}

    return probability, logits, embedding


def preprocess_data_flatten(distance_pairs:list):
    total_label0, total_label1 = list(), list()

    for pair_index in range(len(distance_pairs)):
        label0_distance = torch.flatten(distance_pairs[pair_index]['independent']).view(1, -1)
        label1_distance = torch.flatten(distance_pairs[pair_index]['surrogate']).view(1, -1)
        
        total_label0.append(label0_distance)
        total_label1.append(label1_distance)
    
    processed_data = {'independent': total_label0, 'surrogate': total_label1}

    return processed_data


def pair_to_dataloader(distance_pairs, batch_size=5):
    processed_data = preprocess_data_flatten(distance_pairs)
    dataset = src.datasets.datareader.VarianceData(processed_data['independent'], processed_data['surrogate'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_original_classifier(distance_pairs: list):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    processed_data = preprocess_data_flatten(distance_pairs)
    dataset = src.datasets.datareader.VarianceData(processed_data['independent'], processed_data['surrogate'])

    hidden_layers = [128, 64]
    model = mlp_nn(dataset.data.shape[1], hidden_layers)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 1000

    best_model, best_acc = None, 0
    for i in range(10):
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        model.to(device)
        acc = 0
        for epoch_index in range(epoch_num):
            model.train()
            for _, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if (epoch_index + 1) % 100 == 0:
                model.eval()
                correct = 0
                for _, (inputs, labels) in enumerate(dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    correct += (predictions == labels).sum().item()

                acc = correct / len(dataset) * 100


                if acc == 100:
                    break


        if acc > best_acc:
            best_model = model
            best_acc = acc

        if best_acc == 100:
            break
    print("best acc:{}".format(best_acc))
    return best_model


def owner_verify(suspicious_logits, verifier_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    distance = torch.flatten(suspicious_logits).view(1, -1)

    verifier_model.to(device)
    verifier_model.eval()

    outputs = verifier_model(distance.to(device))
    _, predictions = torch.max(outputs.data, 1)
    
    return predictions


def join_path(*save_path):
    original_model_save_root = os.path.join(*save_path)
    if not os.path.exists(original_model_save_root):
        os.makedirs(original_model_save_root)
    return original_model_save_root


def join_name(hidden_dims):
    str_dims = [str(n) for n in hidden_dims]
    return "_".join(str_dims)


def random_generate_arch(layer_dims, num_hidden_layers, seed):

    # first generate all possible arches, then shuffle, sample
    def _generate_combinations(layer_dims, num_hidden_layer):
        combinations = list(itertools.product(layer_dims, repeat=num_hidden_layer))
        return [sorted(list(combination), reverse=True) for combination in combinations]

    all_hidden_dims = []
    for num_hidden_layer in num_hidden_layers:

        _hidden_dims = _generate_combinations(layer_dims, num_hidden_layer)
        random.seed(seed)
        random.shuffle(_hidden_dims)
        all_hidden_dims.append(_hidden_dims)

    # TODO not deduplicate
    res = []

    for i in range(len(all_hidden_dims[-1])):
        for j in range(len(num_hidden_layers)):
            if i < len(all_hidden_dims[j]):
                res.append(all_hidden_dims[j][i])
    return res


class GNNVerification():
    def __init__(self, args, global_cfg, train_setting_cfg, test_setting_cfg):
        self.global_cfg = global_cfg

        self.test_setting_cfg = test_setting_cfg
        self.train_setting_cfg = train_setting_cfg
        self.args = args
        self.train_save_root = os.path.join(global_cfg["train_save_root"], args.dataset, args.task_type)
        self.test_save_root = os.path.join(global_cfg["test_save_root"], args.dataset, args.task_type)
        # one experimental setting

        self.mask_model_save_name = "{}_{}".format(self.global_cfg["target_model"], join_name(self.global_cfg["target_hidden_dims"]))

    def train_original_model(self):
        # save original model
        original_model_save_root = join_path(self.train_save_root, 'original_models')
        original_model_save_path = os.path.join(original_model_save_root,
                                                "{}_{}.pt".format(self.args.benign_model, join_name(self.args.benign_hidden_dim)))
        return benign.run(self.args, original_model_save_path)

    def geneate_mask_model(self):

        if self.args.task_type == "inductive":
            extract_logits_data = self.original_graph_data[0]
        else:
            extract_logits_data = self.original_graph_data

        # generate mask model
        mask_graph_data, mask_nodes = boundary.mask_graph_data(self.args, extract_logits_data, self.original_model)
        mask_model_save_root = join_path(self.train_save_root, "mask_models", self.args.mask_feat_type,
                                         "{}_{}".format(self.args.mask_node_ratio, self.args.mask_feat_ratio))

        mask_model_save_path = os.path.join(mask_model_save_root, "{}.pt".format(self.mask_model_save_name))

        if self.args.task_type == "inductive":
            mask_graph_data = [mask_graph_data, self.original_graph_data[1], self.original_graph_data[2], self.original_graph_data[3]]

        if self.args.mask_feat_ratio == 0.0:
            mask_model = copy.deepcopy(self.original_model)
            torch.save(mask_model, mask_model_save_path)
            mask_model_acc = self.original_model_acc
        else:
            _, mask_model, mask_model_acc = benign.run(self.args, mask_model_save_path, mask_graph_data)

        measure_nodes = []
        for each_class_nodes in mask_nodes:
            measure_nodes += each_class_nodes

        return mask_model, mask_model_acc, measure_nodes

    # all model generate by this function will automaticly add a final layer for grove
    def train_models_by_arch(self, setting_cfg, model_arch, model_save_root, seed,
                             mask_model_save_name=None, mask_model=None, stage="train", process="train", classifier=None):

        hidden_dims_generator = random_generate_arch(setting_cfg["layer_dims"], setting_cfg["num_hidden_layers"],
                                                     seed=seed)

        if len(hidden_dims_generator) < setting_cfg["num_model_per_arch"]:
            raise Exception("Can not generate enough unique model hidden dims, please reduce num_model_per_arch")

        model_list, acc_list, fidelity_list = [], [], []
        # generate num_model_per_arch models
        for hidden_dims, _ in zip(hidden_dims_generator, list(range(setting_cfg["num_model_per_arch"]))):

            # Important! add a fixed layer
            hidden_dims.append(self.global_cfg["embedding_dim"])
            if mask_model is None:
                # layer_dim, num_hidden_layers
                self.args.benign_hidden_dim = hidden_dims
                self.args.benign_model = model_arch
                # train independent model
                independent_model_save_root = join_path(model_save_root, 'independent_models')
                independent_model_save_path = os.path.join(independent_model_save_root,
                                                           "{}_{}_{}.pt".format(stage, self.args.benign_model,
                                                                                join_name(hidden_dims)))
                _, model, model_acc = benign.run(self.args, independent_model_save_path, self.original_graph_data, process)

            else:
                self.args.extraction_hidden_dim = hidden_dims
                self.args.extraction_model = model_arch
                extraction_model_save_root = join_path(model_save_root, 'extraction_models', self.args.mask_feat_type,
                                                       mask_model_save_name,
                                                       "{}_{}".format(self.args.mask_node_ratio, self.args.mask_feat_ratio))
                extraction_model_save_path = os.path.join(extraction_model_save_root,
                                                          "{}_{}_{}.pt".format(stage, self.args.extraction_model,
                                                                               join_name(hidden_dims)))
                model, model_acc, fidelity = extraction_runner.run(self.args, extraction_model_save_path,
                                                            self.original_graph_data, mask_model, process, classifier)
                fidelity_list.append(fidelity)

            model_list.append(model)
            acc_list.append(model_acc)

        return model_list, acc_list, fidelity_list
    # This function train all models accroding to setting config
    def train_models_by_setting(self, setting_cfg,  model_save_root, mask_model_save_name=None, mask_model=None, stage="train", process="train", classifier=None):
        all_model_list, all_acc_list, all_fidelity_list = [], [], []
        for seed, model_arch in enumerate(setting_cfg["model_arches"]):
            model_list, acc_list, fidelity_list = self.train_models_by_arch(setting_cfg, model_arch, model_save_root, seed, mask_model_save_name,
                                                                       mask_model=mask_model, stage=stage, process=process, classifier=classifier)
            all_model_list += model_list
            all_acc_list.append(acc_list)

            if mask_model is not None:
                all_fidelity_list.append(fidelity_list)

        return all_model_list, all_acc_list, all_fidelity_list


    def run_single_experiment(self, n_run):
        save_json = {}

        start = time.time()
        # train original model
        self.original_graph_data, self.original_model, self.original_model_acc = self.train_original_model()
        if self.args.task_type == "inductive":
            extract_logits_data = self.original_graph_data[0]
        else:
            extract_logits_data = self.original_graph_data

        # generate mask model
        mask_start = time.time()
        self.mask_model, self.mask_model_acc, self.measure_nodes = self.geneate_mask_model()
        mask_run_time = time.time() - mask_start

        mask_outputs, mask_logits, mask_embedding = extract_outputs(extract_logits_data, self.measure_nodes, self.mask_model, self.mask_model)

        # train independent model
        train_inde_model_list, train_inde_acc_list, _ = self.train_models_by_setting(self.train_setting_cfg, self.train_save_root,
                                                                          mask_model=None, stage="train", process='train')
        # train surrogate model
        train_surr_model_list, train_surr_acc_list, train_surr_fidelity_list = self.train_models_by_setting(self.train_setting_cfg,
                                                                                                 self.train_save_root, self.mask_model_save_name,
                                                                                                            self.mask_model, stage="train", process=self.global_cfg["train_process"])

        # TODO
        train_prob_list, train_logits_list, train_embedding_list = [], [],[]
        for independent_model, extraction_model in zip(train_inde_model_list, train_surr_model_list):
            outputs, logits, embedding = extract_outputs(extract_logits_data, self.measure_nodes, independent_model, extraction_model)
            train_prob_list.append(outputs)
            train_logits_list.append([mask_logits["independent"], logits["independent"], logits["surrogate"]])
            train_embedding_list.append([mask_embedding["independent"], embedding["independent"], embedding["surrogate"]])
        train_clf_start = time.time()
        classifier_model = train_original_classifier(train_prob_list)
        # classifier_model = train_k_fold(pair_list)
        train_clf_time = time.time()-train_clf_start

        # train independent  model
        test_inde_model_list, test_inde_acc_list, _ = self.train_models_by_setting(self.test_setting_cfg,  self.train_save_root,
                                                                      mask_model=None, stage="test", process='train')
        # train surrogate model
        test_surr_model_list, test_surr_acc_list, test_surr_fidelity_list = self.train_models_by_setting(
                                                                    self.test_setting_cfg, self.test_save_root,
                                                                    self.mask_model_save_name, self.mask_model, stage="test", process=self.global_cfg["test_process"]) # classifier=classifier_model
        test_logits_list, test_embedding_list = [], []
        TN, FP, FN, TP = 0, 0, 0, 0
        for test_independent_model, test_extraction_model in zip(test_inde_model_list, test_surr_model_list):
            independent__outputs, test_inde_logits, test_inde_embedding = extract_outputs(extract_logits_data, self.measure_nodes, test_independent_model, test_independent_model)
            surrogate_outputs, test_surr_logits, test_surr_embedding = extract_outputs(extract_logits_data, self.measure_nodes, test_extraction_model, test_extraction_model)

            ind_pred = owner_verify(independent__outputs["independent"], classifier_model)
            ext_pred = owner_verify(surrogate_outputs["surrogate"], classifier_model)

            test_embedding_list.append([mask_embedding["independent"], test_inde_embedding["independent"], test_surr_embedding["surrogate"]])
            test_logits_list.append([mask_logits["independent"], test_inde_logits["independent"], test_surr_logits["surrogate"]])

            if ind_pred == 0:
                TN += 1
            else:
                FP += 1

            if ext_pred == 0:
                FN += 1
            else:
                TP += 1


        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)
        Accuracy = (TP + TN) / (TN + FP + TP + FN)

        # save to a
        save_json["TN"], save_json["TP"] = TN,  TP
        save_json["FN"], save_json["FP"] = FN, FP
        save_json["FPR"], save_json["FNR"] = FPR, FNR

        save_json["Accuracy"] = Accuracy
        save_json["original_model_acc"] = self.original_model_acc
        save_json["mask_model_acc"] = self.mask_model_acc
        save_json["train_inde_acc_list"] = train_inde_acc_list
        save_json["train_surr_acc_list"] = train_surr_acc_list
        save_json["train_surr_fidelity_list"] = train_surr_fidelity_list

        save_json["test_inde_acc_list"] = test_inde_acc_list
        save_json["test_surr_acc_list"] = test_surr_acc_list
        save_json["test_surr_fidelity_list"] = test_surr_fidelity_list

        save_json["total_time"] = time.time()-start
        save_json["mask_run_time"] = mask_run_time

        json_save_root = join_path(self.global_cfg["res_path"], self.args.dataset, self.args.task_type, self.args.mask_feat_type,
                                               "{}_{}".format(self.args.mask_node_ratio, self.args.mask_feat_ratio))
        json_save_root = join_path(json_save_root,"train_setting{}".format(self.global_cfg["train_setting"]), "test_setting{}".format(self.global_cfg["test_setting"]))

        with open("{}/{}_{}.json".format(json_save_root, self.mask_model_save_name, n_run), "w") as f:
            f.write(json.dumps(save_json))

        with open("{}/train_setting.yaml".format(json_save_root), "w") as f:
            yaml.dump(self.train_setting_cfg, f, default_flow_style=False)
        with open("{}/test_setting.yaml".format(json_save_root), "w") as f:
            yaml.dump(self.test_setting_cfg, f, default_flow_style=False)

        # if n_run == 0:
        #     torch.save(train_embedding_list,
        #                os.path.join(json_save_root, "{}_train_embedding.pkl".format(self.mask_model_save_name)))
        #     torch.save(test_embedding_list,
        #                os.path.join(json_save_root, "{}_test_embedding.pkl".format(self.mask_model_save_name)))

        #     torch.save(train_logits_list,
        #                os.path.join(json_save_root, "{}_train_logits.pkl".format(self.mask_model_save_name)))
        #     torch.save(test_logits_list,
        #                os.path.join(json_save_root, "{}_test_logits.pkl".format(self.mask_model_save_name)))

        # print("Total Time:{}",save_json["total_time"])
        # print("Train classifier time:{}, Total time:{}, ratio:{}".format(train_clf_time, save_json["total_time"], train_clf_time/save_json["total_time"]))

        return TP, FN, TN, FP


def multiple_experiments(args, global_cfg, config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
    target_arch_list = ["gat", "gcn", "sage"]
    # target_arch_list = ["gat"]
    target_hidden_dim_list = [[352, 128],[288, 128],[224, 128]]
    # target_hidden_dim_list = [[224, 128]]
    attack_setting_list = [1, 2, 3, 4]

    # load setting
    with open(os.path.join(config_path,'train_setting{}.yaml'.format(global_cfg["train_setting"])), 'r') as file:
        train_setting_cfg = yaml.safe_load(file)

    # obtain experimental parameters
    grid_params = []
    for dataset in [global_cfg["dataset"]]:
        for test_setting in attack_setting_list:
            for target_arch in target_arch_list:
                for target_hidden_dims in target_hidden_dim_list:
                    grid_params.append([dataset, test_setting, target_arch, target_hidden_dims])


    for dataset, test_setting, target_arch, target_hidden_dims in grid_params:

        # load test setting
        with open(os.path.join(config_path, 'test_setting{}.yaml'.format(test_setting)), 'r') as file:
            test_setting_cfg = yaml.safe_load(file)
        for n_run in range(global_cfg["n_run"]):
            args.dataset = dataset
            args.benign_hidden_dim = target_hidden_dims
            args.benign_model = target_arch
            global_cfg['test_setting'] = test_setting
            global_cfg['target_model'] = target_arch
            global_cfg['target_hidden_dims'] = target_hidden_dims

            gnn_verification = GNNVerification(args, global_cfg, train_setting_cfg, test_setting_cfg)
            gnn_verification.run_single_experiment(n_run)


if __name__ == '__main__':
    from utils.config import parse_args
    # from verification_cfg import multiple_experiments

    args = parse_args()
    # ownver(args)

    with open(os.path.join("../config", "global_cfg.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)

    multiple_experiments(args, global_cfg)