import os
import torch
import torch_geometric.datasets as dt
import random
import math
import copy

def get_data(args):
    if args.dataset == 'Citeseer' or args.dataset == 'DBLP' or args.dataset == 'PubMed':
        dataset = dt.CitationFull(args.data_path, args.dataset)
        data_path = args.data_path + '/' + (args.dataset).lower() + '/processed/data.pt'
    elif args.dataset == 'Coauthor':
        dataset = dt.Coauthor(args.data_path, 'Physics')
        data_path = args.data_path + '/' + 'Physics' + '/processed/data.pt'
    elif args.dataset == 'Amazon':
        dataset = dt.Amazon(args.data_path, 'Photo')
        data_path = args.data_path + '/' + 'Photo' + '/processed/data.pt'
    elif args.dataset == 'Cora':
        dataset = dt.Planetoid(args.data_path, args.dataset)
        data_path = args.data_path + '/' + args.dataset + '/processed/data.pt'
    
    data = torch.load(data_path)
    
    return data

def load_dataset(dataset_name, data_path='./data'):
    """
    Wrapper function for compatibility with evaluate.py
    """
    class Args:
        def __init__(self, dataset, path):
            self.dataset = dataset
            self.data_path = path
            self.dataset_random_seed = 42
            self.split_dataset_ratio = [0.4, 0.3, 0.2, 0.1]
    
    args = Args(dataset_name, data_path)
    return get_data(args)


class GraphData(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.features = data[0]['x']
        self.adjacency = data[0]['edge_index']
        self.labels = data[0]['y']
        self.node_num = len(self.labels)
        self.feat_dim = len(self.features[0])

        self.set_adj_mat()
        self.get_class_num()
        self.split_dataset(args)
    
    def set_adj_mat(self):
        self.adj_matrix = torch.zeros([self.node_num, self.node_num])
        for i in range(self.node_num):
            source_node = self.adjacency[0][i]
            target_node = self.adjacency[1][i]
            self.adj_matrix[source_node, target_node] = 1
            self.adj_matrix[target_node, source_node] = 1
    
    def get_class_num(self):
        labels = self.labels.tolist()
        labels = set(labels)
        self.class_num = len(labels)

    def split_dataset(self, args):
        all_nodes_index = list(i for i in range(self.node_num))
        self.target_nodes_index, self.shadow_nodes_index, self.attacker_nodes_index, self.test_nodes_index = list(), list(), list(), list()
        
        each_class_nodes_index = [list() for _ in range(self.class_num)]
        for i in range(self.node_num):
            each_class_nodes_index[self.labels[i]].append(i)
        
        node_index = [i for i in range(self.node_num)]
        random.seed(args.dataset_random_seed)
        random.shuffle(node_index)
        target_nodes_size = math.floor(self.node_num * args.split_dataset_ratio[0])
        shadow_nodes_size = math.floor(self.node_num * args.split_dataset_ratio[1])
        attacker_nodes_size = math.floor(self.node_num * args.split_dataset_ratio[2])
        test_nodes_size = self.node_num - target_nodes_size - shadow_nodes_size - attacker_nodes_size
        self.target_nodes_index += node_index[:target_nodes_size]
        self.shadow_nodes_index += node_index[target_nodes_size:(target_nodes_size + shadow_nodes_size)]
        self.attacker_nodes_index += node_index[(target_nodes_size + shadow_nodes_size):(target_nodes_size + shadow_nodes_size + attacker_nodes_size)]
        self.test_nodes_index += node_index[(target_nodes_size + shadow_nodes_size + attacker_nodes_size):]

        self.target_nodes_index.sort()
        self.shadow_nodes_index.sort()
        self.attacker_nodes_index.sort()
        self.test_nodes_index.sort()

    def __len__(self):
        return self.node_num
    
    def __getitem__(self, index):
        return 0


class VarianceData(torch.utils.data.Dataset):
    def __init__(self, label0_data_list, label1_data_list):
        self.label0_data_list = copy.deepcopy(label0_data_list)
        self.label1_data_list = copy.deepcopy(label1_data_list)
        self.concat_data()
        label0_data_label = [0 for _ in range(self.label0_data.shape[0])]
        label1_data_label = [1 for _ in range(self.label1_data.shape[0])]
        self.label = label0_data_label + label1_data_label
        self.label = torch.as_tensor(self.label)
    
    def concat_data(self):
        self.label0_data = None
        self.label1_data = None
        for data_index in range(len(self.label0_data_list)):
            if data_index == 0:
                self.label0_data = self.label0_data_list[data_index]
                self.label1_data = self.label1_data_list[data_index]
            else:
                self.label0_data = torch.cat((self.label0_data, self.label0_data_list[data_index]), 0)
                self.label1_data = torch.cat((self.label1_data, self.label1_data_list[data_index]), 0)
        
        self.data = torch.cat((self.label0_data, self.label1_data), 0)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index, :], self.label[index]


class DistanceData(torch.utils.data.Dataset):
    def __init__(self, label0_data_list, label1_data_list):
        self.label0_data_list = copy.deepcopy(label0_data_list)
        self.label1_data_list = copy.deepcopy(label1_data_list)
        self.concat_data()
        label0_data_label = [0 for _ in range(self.label0_data.shape[0])]
        label1_data_label = [1 for _ in range(self.label1_data.shape[0])]
        self.label = label0_data_label + label1_data_label
        self.label = torch.as_tensor(self.label)
    
    def concat_data(self):
        self.label0_data = None
        self.label1_data = None
        for data_index in range(len(self.label0_data_list)):
            if data_index == 0:
                self.label0_data = self.label0_data_list[data_index]
                self.label1_data = self.label1_data_list[data_index]
            else:
                self.label0_data = torch.cat((self.label0_data, self.label0_data_list[data_index]), 0)
                self.label1_data = torch.cat((self.label1_data, self.label1_data_list[data_index]), 0)
        
        self.data = torch.cat((self.label0_data, self.label1_data), 0)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index, :], self.label[index]


if __name__ == '__main__':
    pass