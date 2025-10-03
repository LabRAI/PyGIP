import copy
import json
import math
import os
import pickle as pkl
import random
import sys
import time

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.sparse as sp
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset, \
    RedditDataset, WikiCSDataset, AmazonRatingsDataset, QuestionsDataset, RomanEmpireDataset, FlickrDataset, \
    CoraFullDataset
from dgl.data import citation_graph as citegrh
from dgl.nn.pytorch import GraphConv
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch_geometric.datasets import CitationFull
from tqdm import tqdm

time_limit = 300


def get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score, adj_matrix2):
    receptive_vector = ((cur_neighbors + adj_matrix2[selected_node]) != 0) + 0
    count = weighted_score.dot(receptive_vector)
    return count


def get_current_neighbors_dense(cur_nodes, adj_matrix2):
    if np.array(cur_nodes).shape[0] == 0:
        return 0
    neighbors = (adj_matrix2[list(cur_nodes)].sum(axis=0) != 0) + 0
    return neighbors


def get_current_neighbors_1(cur_nodes, adj_matrix):
    if np.array(cur_nodes).shape[0] == 0:
        return 0
    neighbors = (adj_matrix[list(cur_nodes)].sum(axis=0) != 0) + 0
    return neighbors


def get_entropy_contribute(npy_m1, npy_m2):
    entro1 = 0
    entro2 = 0
    for i in range(npy_m1.shape[0]):
        entro1 -= np.sum(npy_m1[i] * np.log2(npy_m1[i]))
        entro2 -= np.sum(npy_m2[i] * np.log2(npy_m2[i]))
    return entro1 - entro2


def get_max_info_entropy_node_set(idx_used,
                                  high_score_nodes,
                                  labels,
                                  batch_size,
                                  adj_matrix2,
                                  num_class,
                                  model_prediction):
    max_info_node_set = []
    high_score_nodes_ = copy.deepcopy(high_score_nodes)
    labels_ = copy.deepcopy(labels)
    for k in range(batch_size):
        score_list = np.zeros(len(high_score_nodes_))
        for i in range(len(high_score_nodes_)):
            labels_tmp = copy.deepcopy(labels_)
            node = high_score_nodes_[i]
            node_neighbors = get_current_neighbors_dense([node], adj_matrix2)
            adj_neigh = adj_matrix2[list(node_neighbors)]
            aay = np.matmul(adj_neigh, labels_)
            total_score = 0
            for j in range(num_class):
                if model_prediction[node][j] != 0:
                    labels_tmp[node] = 0
                    labels_tmp[node][j] = 1
                    aay_ = np.matmul(adj_neigh, labels_tmp)
                    total_score += model_prediction[node][j] * get_entropy_contribute(aay, aay_)
            score_list[i] = total_score
        idx = np.argmax(score_list)
        max_node = high_score_nodes_[idx]
        max_info_node_set.append(max_node)
        labels_[max_node] = model_prediction[max_node]
        high_score_nodes_.remove(max_node)
    return max_info_node_set


def get_max_nnd_node_dense(idx_used,
                           high_score_nodes,
                           min_distance,
                           distance_aax,
                           num_ones,
                           num_node,
                           adj_matrix2,
                           gamma=1):
    dmax = np.ones(num_node)

    max_receptive_node = 0
    max_total_score = 0
    cur_neighbors = get_current_neighbors_dense(idx_used, adj_matrix2)
    for node in high_score_nodes:
        receptive_field = get_receptive_fields_dense(cur_neighbors, node, num_ones, adj_matrix2)
        node_distance = distance_aax[node, :]
        node_distance = np.where(node_distance < min_distance, node_distance, min_distance)
        node_distance = dmax - node_distance
        distance_score = node_distance.dot(num_ones)
        total_score = receptive_field / num_node + gamma * distance_score / num_node
        if total_score > max_total_score:
            max_total_score = total_score
            max_receptive_node = node
    return max_receptive_node


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def compute_distance(_i, _j, features_aax):
    return la.norm(features_aax[_i, :] - features_aax[_j, :])


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data_from_grain(path="./data", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum() / 2))

    features = normalize(features)
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_normalized_adjacency(adj):
    adj = adj  # + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


class GCN_drop(nn.Module):
    def __init__(self, feature_number, label_number, dropout=0.85, nhid=128):
        super(GCN_drop, self).__init__()

        self.gc1 = GraphConv(feature_number, nhid, bias=True)
        self.gc2 = GraphConv(nhid, label_number, bias=True)
        self.dropout = dropout

    def forward(self, g, features):
        x = F.dropout(features, self.dropout, training=self.training)
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        return x


def convert_pyg_to_dgl(pyg_data):
    """
    Converts a PyTorch Geometric Data object into a DGLGraph.

    Args:
        pyg_data (torch_geometric.data.Data): PyTorch Geometric Data object.

    Returns:
        dgl.DGLGraph: The converted DGL graph.
    """
    edge_index = pyg_data.edge_index
    num_nodes = pyg_data.num_nodes

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

    if hasattr(pyg_data, 'x') and pyg_data.x is not None:
        g.ndata['feat'] = pyg_data.x

    if hasattr(pyg_data, 'y') and pyg_data.y is not None:
        g.ndata['label'] = pyg_data.y

    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(pyg_data, mask_name) and getattr(pyg_data, mask_name) is not None:
            g.ndata[mask_name] = getattr(pyg_data, mask_name)

    return g


def load_data(dataset_name):
    if dataset_name == 'cora':
        data = citegrh.load_cora()
    if dataset_name == 'citeseer':
        data = citegrh.load_citeseer()
    if dataset_name == 'pubmed':
        data = citegrh.load_pubmed()
    if dataset_name == 'amazoncomputer':
        data = AmazonCoBuyComputerDataset()
    if dataset_name == 'amazonphoto':
        data = AmazonCoBuyPhotoDataset()
    if dataset_name == 'coauthorCS':
        data = CoauthorCSDataset()
    if dataset_name == 'coauthorphysics':
        data = CoauthorPhysicsDataset()
    if dataset_name == 'reddit':
        data = RedditDataset()
    if dataset_name == 'wiki':
        data = WikiCSDataset()
    if dataset_name == 'amazonrating':
        data = AmazonRatingsDataset()
    if dataset_name == 'question':
        data = QuestionsDataset()
    if dataset_name == 'roman':
        data = RomanEmpireDataset()
    if dataset_name == 'flickr':
        data = FlickrDataset()
    if dataset_name == 'cora_full':
        data = CoraFullDataset()
    if dataset_name == 'dblp':
        data = CitationFull(root='./data/', name='DBLP')
        data = data[0]

    if dataset_name == 'dblp':
        g = convert_pyg_to_dgl(data)
    else:
        g = data[0]

    isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
    g.remove_nodes(isolated_nodes)

    if dataset_name in ['cora', 'citeseer', 'pubmed', 'reddit', 'flickr']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        test_mask = g.ndata['test_mask']
        num_nodes = g.num_nodes()
    elif dataset_name in ['wiki']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        test_mask = g.ndata['test_mask'].bool()
        train_mask = (1 - g.ndata['test_mask']).bool()  #
        num_nodes = g.num_nodes()
    elif dataset_name in ['amazoncomputer', 'amazonphoto', 'coauthorCS', 'coauthorphysics', 'cora_full', 'dblp']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        num_nodes = g.num_nodes()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        torch.manual_seed(42)
        indices = torch.randperm(num_nodes)
        num_train = int(num_nodes * 0.6)
        train_mask[indices[:num_train]] = True
        test_mask[indices[num_train:]] = True
        assert train_mask.sum() + test_mask.sum() == num_nodes
    elif dataset_name in ['amazonrating', 'question', 'roman']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        num_nodes = g.num_nodes()
        train_mask = g.ndata['train_mask'][:, 0]
        test_mask = g.ndata['test_mask'][:, 0]
    return g, features, labels, num_nodes, train_mask, test_mask


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        f1score = f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average='macro')
        return correct.item() * 1.0 / len(labels), f1score


class GcnNet(nn.Module):
    def __init__(self, feature_number, label_number):
        super(GcnNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x


# Initialization
def init_mask(C, sub_train_mask, sub_labels):
    # print(f"=========Initialization with {2 * C} Nodes==========================")
    initial_set = []
    for label in range(C):
        label_nodes = []
        for i, l in enumerate(sub_labels):
            if sub_train_mask[i] == True and l == label:
                label_nodes.append(i)
        selected_nodes = random.sample(label_nodes, k=2)  # initial pool for each class
        initial_set.extend(selected_nodes)

    # print(initial_set)
    return initial_set

    # node pool
    ## center_rank = rank_centrality(sub_g, sub_train_mask, sub_train_init, num_center, return_rank=True)
    ## selected_indices_center = center_rank[:num_center]
    ## sub_train_init[selected_indices_center] = True
    # Randomly select the rest of the initial nodes
    ## full_true_indices = th.nonzero(sub_train_mask & ~sub_train_init).squeeze()
    ## selected_indices_random = random.sample(full_true_indices.tolist(), num_random)
    ## sub_train_init[selected_indices_random] = True

    # Transform the formality and return the outcome; note the output are indicators
    # sub_train_init = th.zeros(len(sub_train_mask), dtype=th.bool)
    # sub_train_init[initial_set] = True
    # print(sub_labels[initial_set])
    # sub_train_init = th.tensor(initial_set)
    # return sub_train_init


def update_sub_train_mask(num_each, sub_train_mask, sub_train_mask_new):
    full_true_indices = th.nonzero(sub_train_mask).squeeze()
    current_true_indices = th.nonzero(sub_train_mask_new).squeeze()
    missing_indices = set(full_true_indices.tolist()) - set(current_true_indices.tolist())
    if len(missing_indices) >= num_each:
        # print(f"=========Update Random Querying Label with {num_each} Nodes==========================")
        selected_indices = random.sample(list(missing_indices), num_each)
        ## sub_train_mask_new[selected_indices] = True

    return selected_indices


# Calculate the entropy
def calculate_entropy(probs):
    return -th.sum(probs * th.log(probs + 1e-9), dim=-1)


def rank_entropy(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                 num_each, return_rank=True):
    logits = net(sub_g, sub_features)
    prob = F.softmax(logits, dim=-1)
    nodes_interest = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()
    probs_interest = prob[nodes_interest]
    entropy_interest = calculate_entropy(probs_interest)
    nodes_rank = nodes_interest[th.argsort(entropy_interest, descending=True)]
    if len(nodes_rank) >= num_each:
        if return_rank:
            return nodes_rank
        else:
            print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


def rank_density(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                 num_each, num_clusters, return_rank=True):
    full_true_indices = th.nonzero(sub_train_mask).squeeze()
    current_true_indices = th.nonzero(sub_train_mask_new).squeeze()
    missing_indices = set(full_true_indices.tolist()) - set(current_true_indices.tolist())
    ## Get the embeddings that we need
    ## Under numpy formality
    embedding_all = net(sub_g, sub_features, return_hidden=True).detach().numpy()
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embedding_all)
    ## Set up cluster_centers
    cluster_centers = kmeans.cluster_centers_

    # Calculate the Euclidean distance
    dist = np.linalg.norm(embedding_all - cluster_centers[kmeans.labels_], axis=1)
    density_scores = th.from_numpy(1 / (1 + dist))

    # pull back to the node coefficients
    list_missing_indices = torch.tensor(list(missing_indices))
    shuffle_order = th.argsort(density_scores, descending=True)
    positions = [th.where(shuffle_order == temp)[0].item() for temp in list_missing_indices]
    sorted_positions = th.argsort(th.tensor(positions))
    list_output = list_missing_indices[sorted_positions]

    if len(list_output) >= num_each:
        if return_rank:
            return list_output
        else:
            print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = list_output[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


def rank_centrality(sub_g, sub_train_mask,
                    sub_train_mask_new, num_each, return_rank=True):
    nodes_interest = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()
    page_rank_score = page_rank(sub_g)[nodes_interest]
    nodes_centrality = nodes_interest[th.argsort(page_rank_score, descending=True)]

    if len(nodes_centrality) >= num_each:
        if return_rank:
            return nodes_centrality
        else:
            print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_centrality[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


# Hand-written pagerank score
def page_rank(graph, damping_factor=0.85, max_iter=100, tol=1e-8):
    num_nodes = graph.number_of_nodes()

    # Initialize the PageRank score for all nodes to be uniform
    pagerank_scores = torch.ones(num_nodes) / num_nodes
    graph.ndata['pagerank'] = pagerank_scores

    # Degree normalization factor
    # with graph.local_scope():
    graph.ndata['deg'] = graph.out_degrees().float().clamp(min=1)  # Avoid dividing by 0

    for _ in range(max_iter):
        # Perform message passing (send normalized pagerank score)
        # print("Iteration ", _)
        prev_scores = pagerank_scores.clone()
        graph.ndata['h'] = pagerank_scores / graph.ndata['deg']
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_new'))
        # Apply PageRank formula
        pagerank_scores = damping_factor * graph.ndata['h_new'] + (1 - damping_factor) / num_nodes
        # pagerank_scores_new = (1 - damping_factor) / num_nodes + damping_factor * graph.ndata['pagerank_sum'] / \
        #                       graph.ndata['deg']

        # Check for convergence
        delta = torch.abs(pagerank_scores - prev_scores).sum().item()
        if delta < tol:
            break
        # Update pagerank scores
        graph.ndata['pagerank'] = pagerank_scores

    return graph.ndata['pagerank']


# ECE and Perturbation
def perturb_features(sub_features, noise_level=0.05):
    noise = th.randn_like(sub_features) * noise_level
    perturbed_features = sub_features + noise
    return perturbed_features


# Take the perturbation and count the average
def perturb_avg(net, sub_g, sub_features, num_perturbations, noise_level):
    original_logits = net(sub_g, sub_features)
    # Number of classes
    num_classes = original_logits.size(-1)
    # Initialization
    cumulative_probs = th.zeros(sub_features.size(0), num_classes,
                                device=original_logits.device)
    # Perturbation
    for _ in range(num_perturbations):
        features_p = perturb_features(sub_features, noise_level=noise_level)
        logits_p = net(sub_g, features_p)
        probs_p = F.softmax(logits_p, dim=-1)
        cumulative_probs += probs_p
    # get a fair estimation for the distribution on existing label
    avg_probs = cumulative_probs / num_perturbations

    return avg_probs


# Try the traditional way: count the number of perturbed labels for each node
def rank_perturb(net, sub_g, sub_features, num_perturbations,
                 sub_train_mask, sub_train_mask_new, noise_level,
                 num_each, return_rank=True):
    original_logits = net(sub_g, sub_features)
    nodes_interest = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()
    original_pred = th.argmax(original_logits[nodes_interest], dim=-1)
    ## Store the outcome
    # unchanged_counts = th.zeros_like(original_pred, dtype = th.float)
    unchanged_counts = th.zeros_like(original_pred)
    # Perturbation
    for _ in range(num_perturbations):
        features_p = perturb_features(sub_features, noise_level=noise_level)
        logits_p = net(sub_g, features_p)
        labels_p = th.argmax(logits_p[nodes_interest], dim=-1)
        unchanged = labels_p.eq(original_pred)
        unchanged_counts += unchanged.int()

    # unchanged_counts_float = unchanged_counts.float()
    # unchanged_counts_float.mean()
    _, change_indices = torch.sort(unchanged_counts)
    nodes_rank_label = nodes_interest[change_indices]

    if len(nodes_rank_label) >= num_each:
        if return_rank:
            return nodes_rank_label
        else:
            print(f"=========Update Perturbation Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank_label[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


# Consider items in the embedding space
def rank_cluster(net, sub_g, sub_features, labels, total_sub_nodes,
                 sub_train_mask, sub_train_mask_new, num_clusters,
                 num_each, return_rank=True):
    # Work on missing indices
    full_true_indices = th.nonzero(sub_train_mask).squeeze()
    current_true_indices = th.nonzero(sub_train_mask_new).squeeze()
    missing_indices = set(full_true_indices.tolist()) - set(current_true_indices.tolist())
    # Work on prep of embedding
    labels_true = labels[total_sub_nodes]
    logits = net(sub_g, sub_features)
    prob = F.softmax(logits, dim=-1)
    labels_pred = th.argmax(prob, dim=-1)
    embedding_all = net(sub_g, sub_features, return_hidden=True)
    mismatches_queried = (labels_true != labels_pred) & sub_train_mask_new
    selected_embeddings = embedding_all[mismatches_queried].detach().numpy()
    # Try kmeans
    num_clusters_used = min(num_clusters, th.sum(mismatches_queried).item())
    # print(selected_embeddings)
    print("mismatches_queried:" + str(th.sum(mismatches_queried).item()))
    print("num_clusters_used:" + str(num_clusters_used))
    if num_clusters_used >= 1:
        kmeans = KMeans(n_clusters=num_clusters_used, random_state=0)
        kmeans.fit(selected_embeddings)
        cluster_centers = th.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        # Get back to the original field: Try to use a separate function for remaining functions
        list_missing_indices = list(missing_indices)
        embedding_pool = embedding_all[list_missing_indices]
        min_distances = find_short_dist(embedding_pool, cluster_centers)
        shuffle_order = th.argsort(min_distances)
        output_order = [list_missing_indices[i] for i in shuffle_order]
        nodes_rank_distance = torch.tensor(output_order)
    else:
        print("All nodes give the same label.")
        nodes_rank_distance = torch.tensor(list(missing_indices))

    if len(nodes_rank_distance) >= num_each:
        if return_rank:
            return nodes_rank_distance
        else:
            print(f"=========Update Cluster Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank_distance[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


# Use a separate function to write out the calculation of distance
def find_short_dist(embedding_pool, cluster_centers):
    distances = torch.cdist(embedding_pool, cluster_centers)
    min_distances, _ = torch.min(distances, dim=1)
    return min_distances


# Consider Diversity; see what we can do from here.
def rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new, num_each, num_clusters, rho,
                   return_rank=True):
    full_indices = th.nonzero(sub_train_mask).squeeze()
    queried_indices = th.nonzero(sub_train_mask_new).squeeze()
    candidate_indices = set(full_indices.tolist()) - set(queried_indices.tolist())
    # Get the embeddings
    embedding_all = net(sub_g, sub_features, return_hidden=True).detach().numpy()
    embedding_queried = embedding_all[queried_indices]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embedding_queried)
    cluster_centers = kmeans.cluster_centers_

    node_embeddings = th.tensor(embedding_all, dtype=th.float32)
    centroids = th.tensor(cluster_centers, dtype=th.float32)
    kmeans_labels = th.tensor(kmeans.labels_, dtype=th.int32)

    minimal_distance = th.min(th.cdist(node_embeddings, centroids, p=2), dim=1).values
    proposed_labels = th.min(th.cdist(node_embeddings, centroids, p=2), dim=1).indices

    # Closeness Scores (Distance to assigned centroid)
    close_temp = 1 / (1 + minimal_distance)
    close_normalized = (close_temp - close_temp.min()) / (close_temp.max() - close_temp.min() + 1e-10)

    # Rarity Scores (How rare as shown in )
    queried_bincount = th.bincount(kmeans_labels)
    rarity_temp = 1 / (1 + queried_bincount[proposed_labels])
    rarity_normalized = (rarity_temp - rarity_temp.min()) / (rarity_temp.max() - rarity_temp.min() + 1e-10)

    # Assemble the scores; rho is subject to tuning
    composite_scores = rho * close_normalized + (1 - rho) * rarity_normalized
    composite_scores_candidate = composite_scores[list(candidate_indices)]
    candidate_tensor = th.tensor(list(candidate_indices))
    nodes_rank_diversity = candidate_tensor[th.argsort(composite_scores_candidate, descending=True)]

    if len(nodes_rank_diversity) >= num_each:
        if return_rank:
            return nodes_rank_diversity
        else:
            print(f"=========Update Cluster Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank_diversity[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


def quantile_selection(A, B, C, index_1, index_2, index_3, sub_train_mask, sub_train_mask_new, num_each):
    elements = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()

    ranks_A = [compute_rank(A, el) for el in elements]
    ranks_B = [compute_rank(B, el) for el in elements]
    ranks_C = [compute_rank(C, el) for el in elements]

    weighted_ranks = []
    for i in range(len(elements)):
        weighted_rank = index_1 * ranks_A[i] + index_2 * ranks_B[i] + index_3 * ranks_C[i]
        weighted_ranks.append(weighted_rank)

    # Sort elements based on weighted ranks
    sorted_indices = np.argsort(weighted_ranks)
    sorted_elements = th.stack([elements[i] for i in sorted_indices])
    # sorted_weighted_ranks = [weighted_ranks[i] for i in sorted_indices]

    # print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
    # selected_indices = random.sample(list(missing_indices), num_each)
    selected_indices = sorted_elements[:num_each]
    # sub_train_mask_new[selected_indices] = True

    return selected_indices


def compute_rank(tensor, element):
    return np.where(tensor == element)[0][0]


class GcnNet(nn.Module):
    def __init__(self, feature_number, label_number):
        super(GcnNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features, return_hidden=False):
        relu = nn.ReLU()
        x = F.relu(self.layers[0](g, features))
        if return_hidden:
            return x
        x = self.layers[1](g, x)
        return x


## Main Function
def attack0(dataset_name, seed, cuda, attack_node_arg=0.25, file_path='', LR=1e-3, TGT_LR=1e-2,
            EVAL_EPOCH=1000, TGT_EPOCH=1000, WARMUP_EPOCH=400, dropout=False, model_performance=True, **kwargs):
    # Initialization
    device = th.device(cuda)
    set_seed(seed)
    metrics_df = pd.DataFrame(columns=['Num Attack Nodes', 'Method', 'Test Accuracy', 'Test Fidelity'])

    g, features, labels, node_number, train_mask, test_mask = load_data(dataset_name)
    attack_node_number = int(node_number * attack_node_arg)
    feature_number = features.shape[1]
    label_number = len(labels.unique())
    C_var = label_number

    print('The attack node number is: ', attack_node_number)

    g = g.to(device)
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    if cuda != None:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    if dropout == True:
        gcn_Net = GCN_drop(feature_number, label_number)
    else:
        gcn_Net = GcnNet(feature_number, label_number)
    optimizer = th.optim.Adam(gcn_Net.parameters(), lr=TGT_LR, weight_decay=5e-4)
    dur = []

    ## Send the training to cuda
    features = features.to(device)
    gcn_Net = gcn_Net.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    labels = labels.to(device)
    target_performance = {
        'acc': 0,
        'f1score': 0
    }

    print("=========Target Model Generating==========================")
    for epoch in range(TGT_EPOCH):
        if epoch >= 3:
            t0 = time.time()

        gcn_Net.train()
        logits = gcn_Net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc, f1score = evaluate(gcn_Net, g, features, labels, test_mask)
        if acc > target_performance['acc']:
            target_performance['acc'] = acc
        if f1score > target_performance['f1score']:
            target_performance['f1score'] = f1score

        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 macro {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, f1score, np.mean(dur)))

    ## Get the cuda-trained data back
    g = g.cpu()
    features = features.cpu()
    gcn_Net = gcn_Net.cpu()
    train_mask = train_mask.cpu()
    test_mask = test_mask.cpu()
    labels = labels.cpu()

    # Generate sub-graph index
    alpha = 0.8
    sub_graph_node_index = []
    for i in range(attack_node_number):
        sub_graph_node_index.append(random.randint(0, node_number - 1))

    sub_labels = labels[sub_graph_node_index]

    syn_nodes = []
    g_matrix = np.asmatrix(g.adjacency_matrix().to_dense())

    for node_index in sub_graph_node_index:
        # get nodes
        one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
        two_step_node_index = []
        for first_order_node_index in one_step_node_index:
            syn_nodes.append(first_order_node_index)
            two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()

    sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))

    # Generate features for SubGraph attack
    np_features_query = features.clone()

    for node_index in sub_graph_syn_node_index:
        # initialized as zero
        np_features_query[node_index] = np_features_query[node_index] * 0
        # get one step and two steps nodes
        one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
        one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))

        total_two_step_node_index = []
        num_one_step = len(one_step_node_index)
        for first_order_node_index in one_step_node_index:
            # caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
            # new_array = features[first_order_node_index]*0.8/num_one_step
            this_node_degree = len(g_matrix[first_order_node_index, :].nonzero()[1].tolist())
            np_features_query[node_index] = torch.from_numpy(np.sum(
                [np_features_query[node_index],
                 features[first_order_node_index] * alpha / math.sqrt(num_one_step * this_node_degree)],
                axis=0))

            two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()
            total_two_step_node_index = list(
                set(total_two_step_node_index + two_step_node_index) - set(one_step_node_index))
        total_two_step_node_index = list(set(total_two_step_node_index).intersection(set(sub_graph_node_index)))

        num_two_step = len(total_two_step_node_index)
        for second_order_node_index in total_two_step_node_index:

            # caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
            this_node_second_step_nodes = []
            this_node_first_step_nodes = g_matrix[second_order_node_index, :].nonzero()[1].tolist()
            for nodes_in_this_node in this_node_first_step_nodes:
                this_node_second_step_nodes = list(
                    set(this_node_second_step_nodes + g_matrix[nodes_in_this_node, :].nonzero()[1].tolist()))
            this_node_second_step_nodes = list(set(this_node_second_step_nodes) - set(this_node_first_step_nodes))

            this_node_second_degree = len(this_node_second_step_nodes)
            np_features_query[node_index] = torch.from_numpy(np.sum(
                [np_features_query[node_index],
                 features[second_order_node_index] * (1 - alpha) / math.sqrt(num_two_step * this_node_second_degree)],
                axis=0))

    features_query = th.FloatTensor(np_features_query)

    # generate sub-graph adj-matrix, features, labels
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))
    sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
    for sub_index in range(len(total_sub_nodes)):
        sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

    for i in range(node_number):
        if i in sub_graph_node_index:
            test_mask[i] = 0
            train_mask[i] = 1
            continue
        if i in sub_graph_syn_node_index:
            test_mask[i] = 1
            train_mask[i] = 0
        else:
            test_mask[i] = 1
            train_mask[i] = 0

    sub_train_mask = train_mask[total_sub_nodes]

    sub_features = features_query[total_sub_nodes]
    sub_labels = labels[total_sub_nodes]

    sub_features = th.FloatTensor(sub_features)
    sub_labels = th.LongTensor(sub_labels)
    sub_train_mask = sub_train_mask
    sub_test_mask = test_mask
    # sub_g = DGLGraph(nx.from_numpy_matrix(sub_g))

    # features = th.FloatTensor(data.features)
    # labels = th.LongTensor(data.labels)
    # train_mask = th.ByteTensor(data.train_mask)
    # test_mask = th.ByteTensor(data.test_mask)
    # g = DGLGraph(data.graph)

    gcn_Net.eval()

    # =================Generate Label===================================================
    logits_query = gcn_Net(g, features)
    _, labels_query = th.max(logits_query, dim=1)

    sub_labels_query = labels_query[total_sub_nodes]
    sub_g = nx.from_numpy_array(sub_g)

    sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
    sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))

    sub_g = dgl.from_networkx(sub_g)  # sub_g = DGLGraph(sub_g)
    n_edges = sub_g.number_of_edges()
    # normalization
    degs = sub_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0

    sub_g.ndata['norm'] = norm.unsqueeze(1)

    print("=========Model Extracting==========================")

    # hyperparameters get from kwargs
    # no need to change these default for now
    num_perturbations = kwargs.get('num_perturbations', 100)
    noise_level = kwargs.get('noise_level', 0.05)
    rho = kwargs.get('rho', 0.8)
    num_each = kwargs.get('num_each', 1)
    epochs_per_cycle = kwargs.get('epochs_per_cycle', 1)
    setup = kwargs.get('setup', "experiment")
    # This need to be relatively bigger to allow for more accurate classification
    if_warmup = kwargs.get('if_warmup', False)
    LR_CEGA = kwargs.get('LR_CEGA', 1e-2)
    # Tuning parameters for adaptive weight in each of the CEGA iteration
    # Default works for cora and amazonphoto and coauthorCS
    # Need specific modification for citeseer and pubmed
    curve = kwargs.get('curve', 0.3)
    init_1 = kwargs.get('init_1', 0.2)
    init_2 = kwargs.get('init_2', 0.2)
    init_3 = kwargs.get('init_3', 0.2)
    gap = kwargs.get('gap', 0.6)

    # Derivative parameters
    num_node = sub_features.shape[0]
    total_epochs = epochs_per_cycle * 18 * C_var
    total_num = 20 * C_var
    num_cycles = total_epochs // epochs_per_cycle

    # Set up adaptive weights: set the numbers then reweight them
    # For citeseer, try k = 0.5, init_1 = 0.3. The other parameters seem to be working fine
    cycles = np.linspace(0, 1, num_cycles)
    index_1 = init_1 + gap * np.exp(-1 * curve * cycles)
    index_2 = init_2 + gap * (1 - np.exp(-1 * curve * cycles))
    index_3 = init_3 * (1 - np.exp(-1 * cycles))
    total = index_1 + index_2 + index_3
    index_1 /= total
    index_2 /= total
    index_3 /= total

    # Set up output data formality
    # data_output = pd.DataFrame(columns=['Num Attack Nodes', 'Method', 'Test Accuracy', 'Test Fidelity'])

    # create GCN model
    max_acc1 = 0
    max_acc2 = 0
    max_f1 = 0
    dur = []

    if dropout == True:
        net = GCN_drop(feature_number, label_number)
    else:
        net = GcnNet(feature_number, label_number)
    optimizer = th.optim.Adam(net.parameters(), lr=LR_CEGA, weight_decay=5e-4)

    ## Set up initial set which is iteratively progressive
    train_inits = init_mask(C_var, sub_train_mask, sub_labels)
    train_inits_tensor = th.tensor(train_inits)
    sub_train_mask_new = th.zeros(len(sub_train_mask), dtype=th.bool)
    sub_train_mask_new[train_inits] = True

    ## Record the initial nodes in torch object
    nodes_queried = th.tensor([], dtype=th.long)
    nodes_queried = th.cat((nodes_queried, train_inits_tensor))

    ## Do warm up if that is ever an option
    if if_warmup == True:
        sub_train_mask_warmup = th.zeros(len(sub_train_mask), dtype=th.bool)
        sub_train_mask_warmup[train_inits] = True
        net.train()

        for epoch in range(WARMUP_EPOCH):
            logits = net(sub_g, sub_features)
            logp = F.log_softmax(logits, dim=1)

            loss = F.nll_loss(logp[sub_train_mask_warmup], sub_labels_query[sub_train_mask_warmup])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc, f1score = evaluate(net, g, features, labels, test_mask)
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 score {:.4f}".format(
                epoch + 1, loss.item(), acc, f1score))

        net.eval()

    ## Now start timing when the real cycles begin
    start_time = time.time()
    log_dir = f"{file_path}/timelogs/{dataset_name}/logtime_cega_{seed}"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Learn a node in each cycle
    for cycle in range(10):
        # print(f"=========Cycle {cycle + 1}==========================")
        # print(f"========={int(sub_train_mask_new.sum())} Selected Nodes==========================")

        # Train some epochs:
        net.train()

        for epoch in range(epochs_per_cycle):
            logits = net(sub_g, sub_features)

            ## Need to get new sub_train_mask
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[sub_train_mask_new], sub_labels_query[sub_train_mask_new])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #    if epoch >= 3:
            #        dur.append(time.time() - t0)
            # dur.append(time.time() - t0)

            acc1, _ = evaluate(net, g, features, labels_query, test_mask)
            acc2, f1score = evaluate(net, g, features, labels, test_mask)
            if acc1 > max_acc1:
                max_acc1 = acc1
            if acc2 > max_acc2:
                max_acc2 = acc2
            if f1score > max_f1:
                max_f1 = f1score
            # Add f1 in output
            print(
                "Cycle {:05d} | Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} ".format(
                    cycle + 1, epoch + 1 + cycle * epochs_per_cycle, loss.item(), acc2, acc1, max_f1))

        net.eval()

        ## Not realized here!
        # new_row = {"Epoch": epoch + 1 + cycle * epochs_per_cycle, "Loss": loss.item(), "Accuracy": acc2, "Fidelity": acc1}
        # data_output = data_output.append(new_row, ignore_index = True)
        # data_output.append(new_row)

        # Update the sub_train_mask using your specially-designed algorithm
        if sub_train_mask_new.sum() < total_num:
            # Random
            if setup == "random":
                print("Setup: Random")
                # Add the entry to the node pool nodes_queried on the supposed order
                node_queried = update_sub_train_mask(num_each, sub_train_mask, sub_train_mask_new)
                node_queried_tensor = th.tensor(node_queried)
                # node_queried_tensor = th.tensor(node_queried, dtype = th.long)
                nodes_queried = th.cat((nodes_queried, node_queried_tensor))
                sub_train_mask_new[node_queried] = True

            elif setup == "experiment":
                print("Setup: Experiment")
                ## First: Representativeness
                ## Can be replaced by other centrality measurement
                Rank1 = rank_centrality(sub_g, sub_train_mask, sub_train_mask_new, num_each, return_rank=True)
                ## Second: Uncertainty
                Rank2 = rank_entropy(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                     num_each, return_rank=True)
                ## Third: Diversity
                Rank3 = rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                       num_each, C_var, rho, return_rank=True)

                if Rank1 is None:
                    print("Completed!")
                selected_indices = quantile_selection(Rank1, Rank2, Rank3, index_1[cycle], index_2[cycle],
                                                      index_3[cycle],
                                                      sub_train_mask, sub_train_mask_new, num_each)
                selected_indices_tensor = selected_indices.clone().detach()
                # th.tensor(, dtype = th.long)
                nodes_queried = th.cat((nodes_queried, selected_indices_tensor))
                sub_train_mask_new[selected_indices] = True

            elif setup == "perturbation":
                print("Setup: Experiment with Perturbation")
                Rank1 = rank_centrality(sub_g, sub_train_mask, sub_train_mask_new, num_each, return_rank=True)
                Rank2 = rank_perturb(net, sub_g, sub_features, num_perturbations,
                                     sub_train_mask, sub_train_mask_new, noise_level,
                                     num_each, return_rank=True)
                Rank3 = rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                       num_each, C_var, rho, return_rank=True)

                if Rank1 is None:
                    print("Completed!")
                selected_indices = quantile_selection(Rank1, Rank2, Rank3, index_1[cycle], index_2[cycle],
                                                      index_3[cycle],
                                                      sub_train_mask, sub_train_mask_new, num_each)
                selected_indices_tensor = selected_indices.clone().detach()
                nodes_queried = th.cat((nodes_queried, selected_indices_tensor))
                sub_train_mask_new[selected_indices] = True
            else:
                print("Wrong Setup!")
                return 1
        else:
            print("Move on with designated nodes!")
            sub_train_mask_new = sub_train_mask_new

    ## Record time for all these cycles when the loop is complete
    node_selection_time = time.time() - start_time
    with open(log_dir, 'a') as log_file:
        log_file.write(f"CEGA {dataset_name} {seed} ")
        log_file.write(f"{node_selection_time:.4f}s\n")

    idx_train = nodes_queried.tolist()

    output_data = {
        'total_sub_nodes': total_sub_nodes,
        'idx_train': idx_train
    }

    ## Assertation and printing
    assert len(idx_train) == 20 * C_var
    print('node selection finished')
    with open(f'./node_selection/CEGA_{setup}_{dataset_name}_selected_nodes_{(20 * label_number)}_{seed}.json',
              'w') as f:
        json.dump(output_data, f)

    sub_g = sub_g.to(device)
    sub_features = sub_features.to(device)
    sub_labels_query = sub_labels_query.to(device)
    labels_query = labels_query.to(device)
    g = g.to(device)
    features = features.to(device)
    test_mask = test_mask.to(device)
    labels = labels.to(device)

    print('=========Model Evaluation==========================')
    if model_performance:
        for iter in range(2 * C_var, 21 * C_var, C_var):
            set_seed(seed)

            ## Create net from scratch
            if dropout == True:
                net_scratch = GCN_drop(feature_number, label_number)
            else:
                net_scratch = GcnNet(feature_number, label_number)
            optimizer = th.optim.Adam(net_scratch.parameters(), lr=LR, weight_decay=5e-4)

            ## set up training nodes and send them to device
            sub_train_scratch = th.zeros(sub_features.size()[0], dtype=th.bool)
            sub_train_scratch[idx_train[:iter]] = True
            sub_train_scratch = sub_train_scratch.to(device)
            net_scratch = net_scratch.to(device)

            ## Reset data
            max_acc1 = 0
            max_acc2 = 0
            max_f1 = 0
            dur = []

            for epoch in range(EVAL_EPOCH):
                if epoch >= 3:
                    t0 = time.time()

                net_scratch.train()
                logits = net_scratch(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_scratch], sub_labels_query[sub_train_scratch])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                acc1, _ = evaluate(net_scratch, g, features, labels_query, test_mask)
                acc2, f1score = evaluate(net_scratch, g, features, labels, test_mask)
                if acc1 > max_acc1:
                    max_acc1 = acc1
                if acc2 > max_acc2:
                    max_acc2 = acc2
                if f1score > max_f1:
                    max_f1 = f1score

            # Output Epoch Scores
            epoch_metrics = pd.DataFrame({
                'Num Attack Nodes': [iter],
                'Method': ['CEGA'],
                'Test Accuracy': [max_acc2],
                'Test Fidelity': [max_acc1],
                'Test F1score': [max_f1],
            })
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

            print("Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} | Time(s) {:.4f}".format(
                acc2, acc1, max_f1, np.mean(dur)))

        ## Should this be 'f1score'?
        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [int(th.sum(train_mask))],
            'Method': ['CEGA'],
            'Test Accuracy': [target_performance['acc']],
            'Test Fidelity': [1],
            'Test F1score': [target_performance['f1score']],
        })
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        log_file_path = f"{file_path}/{dataset_name}/log_cega_{seed}.csv"
        metrics_df.to_csv(log_file_path, mode='w', header=False, index=False)

    # Set net_full for the next graph to be taken care of, which is expected to include all nodes
    if True:
        set_seed(seed)
        log_file_path = f"{file_path}/{dataset_name}/log_cega_{seed}.csv"
        if dropout == True:
            net_full = GCN_drop(feature_number, label_number)
        else:
            net_full = GcnNet(feature_number, label_number)
        optimizer_full = th.optim.Adam(net_full.parameters(), lr=LR, weight_decay=5e-4)

        net_full = net_full.to(device)
        net = net.to(device)

        perfm_attack = {
            'acc': 0,
            'fid': 0,
            'f1score': 0
        }

        print('========================== Model Evaluation ==========================')
        progress_bar = tqdm(range(EVAL_EPOCH), desc="Generating model with ALL attack nodes", ncols=100)
        for epoch in progress_bar:
            if epoch >= 3:
                t0 = time.time()

            net_full.train()
            logits = net_full(sub_g, sub_features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, sub_labels_query)  # [sub_train_mask]

            optimizer_full.zero_grad()
            loss.backward()
            optimizer_full.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc, f1score = evaluate(net_full, g, features, labels, test_mask)
            fid, _ = evaluate(net_full, g, features, labels_query, test_mask)
            if acc > perfm_attack['acc']:
                perfm_attack['acc'] = acc
            if fid > perfm_attack['fid']:
                perfm_attack['fid'] = fid
            if f1score > perfm_attack['f1score']:
                perfm_attack['f1score'] = f1score

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Test Acc": f"{acc:.4f}",
            "Test F1": f"{f1score:.4f}",
            # "Processed %": f"{(epoch + 1) / TGT_EPOCH * 100:.2f}",
            # "Time(s)": f"{np.mean(dur) if dur else 0:.4f}"
        })
        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [sub_train_mask.sum().item()],
            'Method': ['cega'],
            'Test Accuracy': [perfm_attack['acc']],
            'Test Fidelity': [perfm_attack['fid']],
            'Test F1score': [perfm_attack['f1score']],
        })
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)
        log_file_path = f"{file_path}/{dataset_name}/log_cega_{seed}.csv"
        metrics_df.to_csv(log_file_path, mode='w', header=False, index=False)


from pygip.models.attack.base import BaseAttack
from pygip.datasets import Dataset
from pygip.utils.metrics import AttackMetric, AttackCompMetric

class CEGA(BaseAttack):
    supported_api_types = {"dgl"}

    # ====== only signature and stored params are changed here ======
    def __init__(
        self,
        dataset: Dataset,
        attack_node_fraction: float,
        model_path: str = None,
        attack_x_ratio: float = 1.0,
        attack_a_ratio: float = 1.0,
    ):
        super(CEGA, self).__init__(dataset, attack_node_fraction, model_path)
        # graph data
        self.dataset = dataset
        self.graph = dataset.graph_data.to(self.device)
        self.features = dataset.graph_data.ndata['feat']
        self.labels = dataset.graph_data.ndata['label']
        self.train_mask = dataset.graph_data.ndata['train_mask']
        self.test_mask = dataset.graph_data.ndata['test_mask']
        # meta data
        self.node_number = dataset.num_nodes
        self.feature_number = dataset.num_features
        self.label_number = dataset.num_classes
        self.attack_node_number = int(dataset.num_nodes * attack_node_fraction)
        self.attack_node_fraction = attack_node_fraction
        # new visibility knobs for inputs (kept for consistency across attacks)
        self.attack_x_ratio = float(attack_x_ratio)
        self.attack_a_ratio = float(attack_a_ratio)

    def attack(
        self,
        seed=1,
        cuda=None,
        LR=1e-3,
        TGT_LR=1e-2,
        EVAL_EPOCH=10,
        TGT_EPOCH=10,
        WARMUP_EPOCH=4,
        dropout=False,
        model_performance=True,
        **kwargs
    ):
        """
        Returns
        -------
        perf_json : dict
            Performance metrics (JSON-serialisable): accuracy/fidelity/F1 of the surrogate,
            and optionally target accuracy/F1 for reference.
        comp_json : dict
            Computation metrics (JSON-serialisable): attack_time, query_target_time, train_surrogate_time, etc.
        """
        # ===== metrics collection (computation) =====
        attack_time_start = time.time()
        query_target_time = 0.0
        train_surrogate_time = 0.0

        # Initialization
        set_seed(seed)
        metrics_df = pd.DataFrame(columns=['Num Attack Nodes', 'Method', 'Test Accuracy', 'Test Fidelity'])

        # data handles
        g = self.graph
        features = self.features
        labels = self.labels
        node_number = self.node_number
        train_mask = self.train_mask
        test_mask = self.test_mask

        attack_node_arg = self.attack_node_fraction
        attack_node_number = int(node_number * attack_node_arg)
        feature_number = features.shape[1]
        label_number = len(labels.unique())
        C_var = label_number

        print('The attack node number is: ', attack_node_number)

        g = g.to(self.device)
        degs = g.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        if cuda is not None:
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        if dropout:
            gcn_Net = GCN_drop(feature_number, label_number)
        else:
            gcn_Net = GcnNet(feature_number, label_number)
        optimizer = th.optim.Adam(gcn_Net.parameters(), lr=TGT_LR, weight_decay=5e-4)
        dur = []

        # Send the training to device
        features = features.to(self.device)
        gcn_Net = gcn_Net.to(self.device)
        train_mask = train_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        labels = labels.to(self.device)
        target_performance = {'acc': 0, 'f1score': 0}

        print("=========Target Model Generating==========================")
        # train target model
        tgt_train_t0 = time.time()
        for epoch in range(TGT_EPOCH):
            if epoch >= 3:
                t0 = time.time()
            gcn_Net.train()
            logits = gcn_Net(g, features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch >= 3:
                dur.append(time.time() - t0)
            acc, f1score = evaluate(gcn_Net, g, features, labels, test_mask)
            target_performance['acc'] = max(target_performance['acc'], acc)
            target_performance['f1score'] = max(target_performance['f1score'], f1score)
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 macro {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, f1score, np.mean(dur)))
        train_surrogate_time += (time.time() - tgt_train_t0)

        # move tensors back to cpu for subgraph building
        g = g.cpu()
        features = features.cpu()
        gcn_Net = gcn_Net.cpu()
        train_mask = train_mask.cpu()
        test_mask = test_mask.cpu()
        labels = labels.cpu()

        # Generate sub-graph index
        alpha = 0.8
        sub_graph_node_index = [random.randint(0, node_number - 1) for _ in range(attack_node_number)]
        sub_labels = labels[sub_graph_node_index]

        syn_nodes = []
        g_matrix = np.asmatrix(g.adjacency_matrix().to_dense())
        for node_index in sub_graph_node_index:
            # get nodes
            one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
            two_step_node_index = []
            for first_order_node_index in one_step_node_index:
                syn_nodes.append(first_order_node_index)
                two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()

        sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
        total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))

        # Generate features for SubGraph attack
        np_features_query = features.clone()
        for node_index in sub_graph_syn_node_index:
            # initialized as zero
            np_features_query[node_index] = np_features_query[node_index] * 0
            # get one step and two steps nodes
            one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
            one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))

            total_two_step_node_index = []
            num_one_step = len(one_step_node_index)
            for first_order_node_index in one_step_node_index:
                # features = 0.8 * average_one / sqrt(num_one_step * deg)
                this_node_degree = len(g_matrix[first_order_node_index, :].nonzero()[1].tolist())
                x1 = np_features_query[node_index]
                x2 = features[first_order_node_index] * alpha / math.sqrt(max(1, num_one_step * this_node_degree))
                np_features_query[node_index] = x1 + x2

                two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()
                total_two_step_node_index = list(
                    set(total_two_step_node_index + two_step_node_index) - set(one_step_node_index)
                )
            total_two_step_node_index = list(set(total_two_step_node_index).intersection(set(sub_graph_node_index)))

            num_two_step = len(total_two_step_node_index)
            for second_order_node_index in total_two_step_node_index:
                this_node_second_step_nodes = []
                this_node_first_step_nodes = g_matrix[second_order_node_index, :].nonzero()[1].tolist()
                for nodes_in_this_node in this_node_first_step_nodes:
                    this_node_second_step_nodes = list(
                        set(this_node_second_step_nodes + g_matrix[nodes_in_this_node, :].nonzero()[1].tolist())
                    )
                this_node_second_step_nodes = list(set(this_node_second_step_nodes) - set(this_node_first_step_nodes))
                this_node_second_degree = len(this_node_second_step_nodes)
                x1 = np_features_query[node_index]
                x2 = features[first_order_node_index] * alpha / math.sqrt(max(1, num_one_step * this_node_degree))
                np_features_query[node_index] = x1 + x2

        features_query = th.FloatTensor(np_features_query)

        # generate sub-graph adj-matrix, features, labels
        total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))
        sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
        for sub_index in range(len(total_sub_nodes)):
            sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

        for i in range(node_number):
            if i in sub_graph_node_index:
                test_mask[i] = 0
                train_mask[i] = 1
                continue
            if i in sub_graph_syn_node_index:
                test_mask[i] = 1
                train_mask[i] = 0
            else:
                test_mask[i] = 1
                train_mask[i] = 0

        sub_train_mask = train_mask[total_sub_nodes]
        sub_features = features_query[total_sub_nodes]
        sub_labels = labels[total_sub_nodes]

        sub_features = th.FloatTensor(sub_features)
        sub_labels = th.LongTensor(sub_labels)
        sub_train_mask = sub_train_mask
        sub_test_mask = test_mask

        gcn_Net.eval()

        # =================Generate Label========================
        # timing: query the target once for labels on query set
        qt0 = time.time()
        logits_query = gcn_Net(g, features)
        _, labels_query = th.max(logits_query, dim=1)
        query_target_time += (time.time() - qt0)

        sub_labels_query = labels_query[total_sub_nodes]
        sub_g = nx.from_numpy_array(sub_g)
        sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
        sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
        sub_g = dgl.from_networkx(sub_g)
        # normalization
        degs = sub_g.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        sub_g.ndata['norm'] = norm.unsqueeze(1)

        print("=========Model Extracting==========================")

        # hyperparameters from kwargs
        num_perturbations = kwargs.get('num_perturbations', 100)
        noise_level = kwargs.get('noise_level', 0.05)
        rho = kwargs.get('rho', 0.8)
        num_each = kwargs.get('num_each', 1)
        epochs_per_cycle = kwargs.get('epochs_per_cycle', 1)
        setup = kwargs.get('setup', "experiment")
        if_warmup = kwargs.get('if_warmup', False)
        LR_CEGA = kwargs.get('LR_CEGA', 1e-2)
        curve = kwargs.get('curve', 0.3)
        init_1 = kwargs.get('init_1', 0.2)
        init_2 = kwargs.get('init_2', 0.2)
        init_3 = kwargs.get('init_3', 0.2)
        gap = kwargs.get('gap', 0.6)

        # derivative params
        num_node = sub_features.shape[0]
        total_epochs = epochs_per_cycle * 18 * C_var
        total_num = 20 * C_var
        num_cycles = total_epochs // epochs_per_cycle

        cycles = np.linspace(0, 1, num_cycles)
        index_1 = init_1 + gap * np.exp(-1 * curve * cycles)
        index_2 = init_2 + gap * (1 - np.exp(-1 * curve * cycles))
        index_3 = init_3 * (1 - np.exp(-1 * cycles))
        total = index_1 + index_2 + index_3
        index_1 /= total
        index_2 /= total
        index_3 /= total

        # create surrogate model
        max_acc1 = 0
        max_acc2 = 0
        max_f1 = 0
        dur = []

        net = GCN_drop(feature_number, label_number) if dropout else GcnNet(feature_number, label_number)
        optimizer = th.optim.Adam(net.parameters(), lr=LR_CEGA, weight_decay=5e-4)

        # initial training set
        train_inits = init_mask(C_var, sub_train_mask, sub_labels)
        train_inits_tensor = th.tensor(train_inits)
        sub_train_mask_new = th.zeros(len(sub_train_mask), dtype=th.bool)
        sub_train_mask_new[train_inits] = True
        nodes_queried = th.tensor([], dtype=th.long)
        nodes_queried = th.cat((nodes_queried, train_inits_tensor))

        # warmup
        if if_warmup:
            sub_train_mask_warmup = th.zeros(len(sub_train_mask), dtype=th.bool)
            sub_train_mask_warmup[train_inits] = True
            net.train()
            warm_s = time.time()
            for epoch in range(WARMUP_EPOCH):
                logits = net(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_mask_warmup], sub_labels_query[sub_train_mask_warmup])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc, f1score = evaluate(net, g, features, labels, test_mask)
                print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 score {:.4f}".format(
                    epoch + 1, loss.item(), acc, f1score))
            train_surrogate_time += (time.time() - warm_s)
            net.eval()

        # cycles
        print("=========Learn a node in each cycle==========================")
        cycle_train_s = time.time()
        for cycle in range(num_cycles):
            net.train()
            for epoch in range(epochs_per_cycle):
                logits = net(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_mask_new], sub_labels_query[sub_train_mask_new])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc1, _ = evaluate(net, g, features, labels_query, test_mask)  # fidelity proxy
                acc2, f1score = evaluate(net, g, features, labels, test_mask)
                max_acc1 = max(max_acc1, acc1)
                max_acc2 = max(max_acc2, acc2)
                max_f1 = max(max_f1, f1score)
                print(
                    "Cycle {:05d} | Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} ".format(
                        cycle + 1, epoch + 1 + cycle * epochs_per_cycle, loss.item(), acc2, acc1, max_f1))

            net.eval()
            if sub_train_mask_new.sum() < total_num:
                if setup == "random":
                    print("Setup: Random")
                    node_queried = update_sub_train_mask(num_each, sub_train_mask, sub_train_mask_new)
                    node_queried_tensor = th.tensor(node_queried)
                    nodes_queried = th.cat((nodes_queried, node_queried_tensor))
                    sub_train_mask_new[node_queried] = True
                elif setup == "experiment":
                    print("Setup: Experiment")
                    Rank1 = rank_centrality(sub_g, sub_train_mask, sub_train_mask_new, num_each, return_rank=True)
                    Rank2 = rank_entropy(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                         num_each, return_rank=True)
                    Rank3 = rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                           num_each, C_var, rho, return_rank=True)
                    if Rank1 is None:
                        print("Completed!")
                    selected_indices = quantile_selection(
                        Rank1, Rank2, Rank3, index_1[cycle], index_2[cycle], index_3[cycle],
                        sub_train_mask, sub_train_mask_new, num_each
                    )
                    selected_indices_tensor = selected_indices.clone().detach()
                    nodes_queried = th.cat((nodes_queried, selected_indices_tensor))
                    sub_train_mask_new[selected_indices] = True
                elif setup == "perturbation":
                    print("Setup: Experiment with Perturbation")
                    Rank1 = rank_centrality(sub_g, sub_train_mask, sub_train_mask_new, num_each, return_rank=True)
                    Rank2 = rank_perturb(net, sub_g, sub_features, num_perturbations,
                                         sub_train_mask, sub_train_mask_new, noise_level,
                                         num_each, return_rank=True)
                    Rank3 = rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                           num_each, C_var, rho, return_rank=True)
                    if Rank1 is None:
                        print("Completed!")
                    selected_indices = quantile_selection(
                        Rank1, Rank2, Rank3, index_1[cycle], index_2[cycle], index_3[cycle],
                        sub_train_mask, sub_train_mask_new, num_each
                    )
                    selected_indices_tensor = selected_indices.clone().detach()
                    nodes_queried = th.cat((nodes_queried, selected_indices_tensor))
                    sub_train_mask_new[selected_indices] = True
                else:
                    print("Wrong Setup!")
                    return 1
            else:
                print("Move on with designated nodes!")
        train_surrogate_time += (time.time() - cycle_train_s)

        idx_train = nodes_queried.tolist()
        output_data = {'total_sub_nodes': total_sub_nodes, 'idx_train': idx_train}
        print('node selection finished')

        # move to device for evaluation/training-from-scratch
        sub_g = sub_g.to(self.device)
        sub_features = sub_features.to(self.device)
        sub_labels_query = sub_labels_query.to(self.device)
        labels_query = labels_query.to(self.device)
        g = g.to(self.device)
        features = features.to(self.device)
        test_mask = test_mask.to(self.device)
        labels = labels.to(self.device)

        print('=========Model Evaluation==========================')
        if model_performance:
            for iter in range(2 * C_var, 21 * C_var, C_var):
                set_seed(seed)
                net_scratch = GCN_drop(feature_number, label_number) if dropout else GcnNet(feature_number, label_number)
                optimizer = th.optim.Adam(net_scratch.parameters(), lr=LR, weight_decay=5e-4)
                sub_train_scratch = th.zeros(sub_features.size()[0], dtype=th.bool)
                sub_train_scratch[idx_train[:iter]] = True
                sub_train_scratch = sub_train_scratch.to(self.device)
                net_scratch = net_scratch.to(self.device)
                max_acc1 = 0
                max_acc2 = 0
                max_f1 = 0
                dur = []
                eval_train_s = time.time()
                for epoch in range(EVAL_EPOCH):
                    if epoch >= 3:
                        t0 = time.time()
                    net_scratch.train()
                    logits = net_scratch(sub_g, sub_features)
                    logp = F.log_softmax(logits, dim=1)
                    loss = F.nll_loss(logp[sub_train_scratch], sub_labels_query[sub_train_scratch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch >= 3:
                        dur.append(time.time() - t0)
                    acc1, _ = evaluate(net_scratch, g, features, labels_query, test_mask)
                    acc2, f1score = evaluate(net_scratch, g, features, labels, test_mask)
                    max_acc1 = max(max_acc1, acc1)
                    max_acc2 = max(max_acc2, acc2)
                    max_f1 = max(max_f1, f1score)
                train_surrogate_time += (time.time() - eval_train_s)
                epoch_metrics = pd.DataFrame({
                    'Num Attack Nodes': [iter],
                    'Method': ['CEGA'],
                    'Test Accuracy': [max_acc2],
                    'Test Fidelity': [max_acc1],
                    'Test F1score': [max_f1],
                })
                metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)
                print("Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} | Time(s) {:.4f}".format(
                    acc2, acc1, max_f1, np.mean(dur)))
            epoch_metrics = pd.DataFrame({
                'Num Attack Nodes': [int(th.sum(train_mask))],
                'Method': ['CEGA'],
                'Test Accuracy': [target_performance['acc']],
                'Test Fidelity': [1],
                'Test F1score': [target_performance['f1score']],
            })
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        # train a surrogate on all selected nodes for final report
        set_seed(seed)
        net_full = GCN_drop(feature_number, label_number) if dropout else GcnNet(feature_number, label_number)
        optimizer_full = th.optim.Adam(net_full.parameters(), lr=LR, weight_decay=5e-4)
        net_full = net_full.to(self.device)
        net = net.to(self.device)
        perfm_attack = {'acc': 0, 'fid': 0, 'f1score': 0}

        print('========================== Model Evaluation ==========================')
        final_train_s = time.time()
        progress_bar = tqdm(range(EVAL_EPOCH), desc="Generating model with ALL attack nodes", ncols=100)
        for epoch in progress_bar:
            if epoch >= 3:
                t0 = time.time()
            net_full.train()
            logits = net_full(sub_g, sub_features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, sub_labels_query)
            optimizer_full.zero_grad()
            loss.backward()
            optimizer_full.step()
            if epoch >= 3:
                dur.append(time.time() - t0)

            acc, f1score = evaluate(net_full, g, features, labels, test_mask)
            fid, _ = evaluate(net_full, g, features, labels_query, test_mask)
            perfm_attack['acc'] = max(perfm_attack['acc'], acc)
            perfm_attack['fid'] = max(perfm_attack['fid'], fid)
            perfm_attack['f1score'] = max(perfm_attack['f1score'], f1score)
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Test Acc": f"{acc:.4f}",
                "Test F1": f"{f1score:.4f}",
            })
        train_surrogate_time += (time.time() - final_train_s)

        print("Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} | Time(s) {:.4f}".format(
            perfm_attack['acc'], perfm_attack['fid'], perfm_attack['f1score'], np.mean(dur)))

        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [sub_train_mask.sum().item()],
            'Method': ['cega'],
            'Test Accuracy': [perfm_attack['acc']],
            'Test Fidelity': [perfm_attack['fid']],
            'Test F1score': [perfm_attack['f1score']],
        })
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        # ===== assemble JSON outputs required by the new metric API =====
        attack_total_time = time.time() - attack_time_start

        perf_json = {
            "attack": "CEGA",
            "num_attack_nodes": int(sub_train_mask.sum().item()),
            "acc": float(perfm_attack['acc']),
            "fid": float(perfm_attack['fid']),
            "f1": float(perfm_attack['f1score']),
            "target_acc": float(target_performance['acc']),
            "target_f1": float(target_performance['f1score']),
        }
        comp_json = {
            "device": str(self.device),
            "attack_time": float(attack_total_time),
            "query_target_time": float(query_target_time),
            "train_surrogate_time": float(train_surrogate_time),
            # optional placeholders to align with AdvMEA if present there
            "inference_surrogate_time": None,
            "peak_memory": None,
        }
        return perf_json, comp_json

