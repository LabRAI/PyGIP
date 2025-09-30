import ast
import os
import random
from pathlib import Path
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Planetoid, CitationFull, WebKB
from torch_geometric.nn import GCNConv
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from pygip.datasets.datasets import Dataset as PyGIPDataset
from pygip.models.defense.base import BaseDefense
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_everything(seed)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        hidden = self.conv1(x, edge_index)
        x = F.relu(hidden)
        output = self.conv2(x, edge_index)
        return F.log_softmax(output, dim=1), output


def train_gcn(model, data, optimizer, criterion, epochs=200, verbose=True):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(data.x, data.edge_index)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"[GCN-Train] Epoch {epoch}, Loss: {loss.item()}")


class TargetGCN:
    def __init__(self, trained_model, data):
        self.model = trained_model
        self.data = data

    def predict(self, query_indices):
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(output[query_indices], dim=1).cpu().numpy()
        return probs

    def get_embedding(self):
        self.model.eval()
        with torch.no_grad():
            _, embeddings = self.model(self.data.x, self.data.edge_index)
        return embeddings


def get_node_embedding(model, data, node_idx):
    embeddings = model.get_embedding()
    return embeddings[node_idx]


def get_one_hop_neighbors(data, node_idx):
    edge_index = data.edge_index
    neighbors = edge_index[1][edge_index[0] == node_idx].tolist()
    return neighbors


def average_pooling_with_neighbors(model, data, node_idx):
    embeddings = model.get_embedding()
    neighbors = get_one_hop_neighbors(data, node_idx)
    neighbors.append(node_idx)
    neighbor_embeddings = embeddings[neighbors]
    pooled_embedding = torch.mean(neighbor_embeddings, dim=0)
    return pooled_embedding


def k_core_decomposition(graph):
    k_core_dict = nx.core_number(graph)
    return k_core_dict


def average_pooling_with_neighbors_batch(model, data, node_indices):
    embeddings = model.get_embedding()
    neighbors = [get_one_hop_neighbors(data, idx) for idx in node_indices]

    node_and_neighbors = [torch.tensor([idx] + list(neighbors[i])) for i, idx in enumerate(node_indices)]

    pooled_embeddings = torch.stack([
        embeddings[node_idx_list].mean(dim=0) for node_idx_list in node_and_neighbors
    ])
    return pooled_embeddings


def compute_embedding_batch(target_model, data, k_core_values_graph, max_k_core, node_indices, lamb=1.0):
    pooled_embeddings = average_pooling_with_neighbors_batch(target_model, data, node_indices)
    k_core_values = torch.tensor([k_core_values_graph[node_idx] for node_idx in node_indices], dtype=torch.float32).to(
        pooled_embeddings.device)

    max_k_core_tensor = torch.log(max_k_core)
    scaled_k_core = torch.log(k_core_values) / max_k_core_tensor
    scaling_function = 1 + lamb * (torch.sigmoid(scaled_k_core) - 0.5) * 2
    final_embeddings = pooled_embeddings * scaling_function.unsqueeze(-1)
    return final_embeddings


def simple_embedding_batch(target_model, data, node_indices):
    pooled_embeddings = average_pooling_with_neighbors_batch(target_model, data, node_indices)
    return pooled_embeddings


def precompute_all_node_embeddings(
        target_model,
        data,
        k_core_values_graph,
        max_k_core,
        lamb=1.0
):
    all_node_indices = list(range(data.num_nodes))
    all_embeddings = compute_embedding_batch(
        target_model,
        data,
        k_core_values_graph,
        max_k_core,
        all_node_indices,
        lamb=lamb
    )
    return all_embeddings


def precompute_simple_embeddings(target_model, data):
    all_node_indices = list(range(data.num_nodes))
    return simple_embedding_batch(target_model, data, all_node_indices)


def collate_fn_no_pad(batch):
    batch_seqs = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    return batch_seqs, torch.tensor(batch_labels, dtype=torch.long)


def preprocess_sequences(df):
    def convert_to_list(sequence):
        if isinstance(sequence, str):
            return ast.literal_eval(sequence)
        return sequence

    df["Sequence"] = df["Sequence"].apply(convert_to_list)
    return df


class SequencesDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.loc[idx, "Sequence"]
        lbl = self.df.loc[idx, "Label"]
        if isinstance(seq, str):
            raise TypeError(f"Sequence should be list[int], but received str: {seq}")
        return list(seq), int(lbl)


def split_and_adjust(dataset_sequences, seed):
    train_df, temp_df = train_test_split(dataset_sequences, test_size=0.3, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df


def build_loaders(
        csv_path="attack_CiteSeer.csv",
        batch_size=16,
        drop_last=True,
        seed=42,
):
    df = pd.read_csv(csv_path)
    df_unique = df.drop_duplicates(subset="Sequence")
    df = df_unique
    dataset_sequences = df[["Sequence", "Label"]].copy()
    dataset_sequences = preprocess_sequences(dataset_sequences)
    dataset_sequences["Label"] = dataset_sequences["Label"].astype(int)
    dataset_sequences = dataset_sequences[dataset_sequences['Sequence'].apply(len) > 1]

    train_df, val_df, test_df = split_and_adjust(dataset_sequences, seed)

    train_dataset = SequencesDataset(train_df)
    val_dataset = SequencesDataset(val_df)
    test_dataset = SequencesDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_no_pad,
        drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_no_pad,
        drop_last=drop_last
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_no_pad,
        drop_last=drop_last
    )

    return train_loader, val_loader, test_loader


def load_data_and_model(csv_path, batch_size, seed, data_path, lamb):
    try:
        script_dir = Path(__file__).resolve().parent
        parent_dir = script_dir.parent
    except NameError:
        parent_dir = Path.cwd().parent
        print(
            "If __file__ is not defined, "
            "the directory above the current working directory is used as the target directory."
        )

    os.chdir(parent_dir)

    train_loader, val_loader, test_loader = build_loaders(
        csv_path=csv_path,
        batch_size=batch_size,
        drop_last=True,
        seed=seed
    )

    # ======== Step 2: target_model, data =========
    if data_path == "CiteSeer":
        dataset = Planetoid(root="./data", name=data_path)
        data = dataset[0]
    elif data_path == "PubMed":
        dataset = Planetoid(root="./data", name="PubMed")
        data = dataset[0]
    elif data_path == "Cora":
        dataset = Planetoid(root="./data", name=data_path)
        data = dataset[0]
    elif data_path == "Cora_ML":
        dataset = CitationFull(root="./data", name="Cora_ML")
        data = dataset[0]
        num_nodes = data.num_nodes
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)
        num_test = num_nodes - num_train - num_val
        perm = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:num_train]] = True
        data.val_mask[perm[num_train:num_train + num_val]] = True
        data.test_mask[perm[num_train + num_val:]] = True
    elif data_path == "Cornell" or data_path == "Wisconsin":
        dataset = WebKB(root="./data", name=data_path)
        data = dataset[0]
        num_nodes = data.num_nodes
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)
        num_test = num_nodes - num_train - num_val

        perm = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:num_train]] = True
        data.val_mask[perm[num_train:num_train + num_val]] = True
        data.test_mask[perm[num_train + num_val:]] = True

    trained_gcn = GCN(dataset.num_features, 16, dataset.num_classes)
    target_model = TargetGCN(trained_model=trained_gcn, data=data)

    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    k_core_values_graph = k_core_decomposition(G)
    max_k_core = torch.tensor(max(k_core_values_graph.values()), dtype=torch.float32)

    all_embeddings = precompute_all_node_embeddings(
        target_model, data, k_core_values_graph, max_k_core, lamb=lamb
    )

    return train_loader, val_loader, test_loader, target_model, max_k_core, all_embeddings, dataset, data


class StateTransformMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StateTransformMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, prob_factor):
        x = prob_factor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FusionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FusionGRU, self).__init__()
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wg = nn.Linear(input_size * 2, input_size)
        self.bg = nn.Parameter(torch.zeros(input_size))

    def forward(self, h_it, h_it_m1, hidden_state):

        delta_it = h_it - h_it_m1

        concat_input = torch.cat((delta_it, h_it), dim=-1)
        g_t = torch.sigmoid(self.Wg(concat_input) + self.bg)

        x_t = g_t * delta_it + (1 - g_t) * h_it

        combined = torch.cat((x_t, hidden_state), dim=-1)
        z_t = torch.sigmoid(self.Wz(combined))
        r_t = torch.sigmoid(self.Wr(combined))

        r_h_prev = r_t * hidden_state
        combined_candidate = torch.cat((x_t, r_h_prev), dim=-1)
        h_tilde = torch.tanh(self.Wh(combined_candidate))

        h_next = (1 - z_t) * hidden_state + z_t * h_tilde
        return h_next

    def process_sequence(self, inputs, hidden_state=None):
        batch_size, seq_len, input_size = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        outputs = []
        h_it_m1 = torch.zeros(batch_size, input_size, device=inputs.device)

        for t in range(seq_len):
            h_it = inputs[:, t, :]
            hidden_state = self.forward(h_it, h_it_m1, hidden_state)
            outputs.append(hidden_state.unsqueeze(1))
            h_it_m1 = h_it

        return torch.cat(outputs, dim=1)


def test_model(agent, gru, mlp_transform, test_loader, target_model, data, all_embeddings, hidden_size, device):
    agent.eval()
    gru.eval()
    mlp_transform.eval()

    total_reward = 0.0
    action_dim = 2
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []

    with torch.no_grad():
        for batch_seqs, batch_labels in test_loader:
            batch_labels = batch_labels.to(device)

            batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
            padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)
            mask = (padded_seqs != 0).float().to(device)

            max_seq_len = padded_seqs.size(1)
            hidden_states = torch.zeros(len(batch_seqs), hidden_size, device=device)

            all_inputs = []
            for t in range(max_seq_len):
                node_indices = padded_seqs[:, t].tolist()
                cur_inputs = all_embeddings[node_indices]
                all_inputs.append(cur_inputs)

            all_inputs = torch.stack(all_inputs, dim=1).to(device)
            hidden_states = gru.process_sequence(all_inputs)
            masked_hidden_states = hidden_states * mask.unsqueeze(-1)

            prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)
            custom_states = (mlp_transform(prob_factors) * masked_hidden_states).detach()

            actions, probabilities, _, _ = agent.select_action(custom_states.view(-1, hidden_size))
            actions = actions.view(len(batch_seqs), max_seq_len)
            probabilities = probabilities.view(len(batch_seqs), max_seq_len)

            for i in range(len(batch_seqs)):
                last_valid_step = (mask[i].sum().long() - 1).item()
                predicted_action = actions[i, last_valid_step].item()
                predicted_prob = probabilities[i, last_valid_step].item()
                true_label = batch_labels[i].item()

                all_true_labels.append(true_label)
                all_predicted_labels.append(predicted_action)
                all_predicted_probs.append(predicted_prob)

                reward = custom_reward_function(predicted_action, true_label)
                total_reward += reward

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels, average='binary')
    recall = recall_score(all_true_labels, all_predicted_labels, average='binary')
    f1 = f1_score(all_true_labels, all_predicted_labels, average='binary')

    fpr, tpr, _ = roc_curve(all_true_labels, all_predicted_probs)
    auc_value = auc(fpr, tpr)

    return accuracy, precision, recall, f1, auc_value


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.entropies = []
        self.returns = []
        self.all_probs = {}
        self.masks = []

    def store(self, custom_states, action, log_prob, reward, done, entropy, probs=None, masks=None):
        for i in range(custom_states.size(0)):
            state_seq = custom_states[i]
            action_seq = action[i]
            log_prob_seq = log_prob[i]
            reward_seq = reward[i]
            done_seq = done[i]
            mask_seq = masks[i]

            valid_len = int(mask_seq.sum().item())

            state_seq = torch.cat([state_seq[:valid_len],
                                   torch.zeros(custom_states.size(1) - valid_len, custom_states.size(2),
                                               device=state_seq.device)])
            action_seq = torch.cat(
                [action_seq[:valid_len], torch.zeros(action.size(1) - valid_len, device=action_seq.device)])
            log_prob_seq = torch.cat(
                [log_prob_seq[:valid_len], torch.zeros(log_prob.size(1) - valid_len, device=log_prob_seq.device)])
            reward_seq = torch.cat(
                [reward_seq[:valid_len], torch.zeros(reward.size(1) - valid_len, device=reward_seq.device)])
            done_seq = torch.cat([done_seq[:valid_len], torch.zeros(done.size(1) - valid_len, device=done_seq.device)])
            mask_seq = torch.cat([mask_seq[:valid_len], torch.zeros(masks.size(1) - valid_len, device=mask_seq.device)])

            self.states.append(state_seq)
            self.actions.append(action_seq)
            self.log_probs.append(log_prob_seq)
            self.rewards.append(reward_seq)
            self.dones.append(done_seq)
            self.masks.append(mask_seq)

            consistent_shape = all(tensor.shape == self.states[0].shape for tensor in self.states)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.entropies = []
        self.returns = []
        self.masks = []


def compute_returns_and_advantages(memory, gamma=0.99, lam=0.95):
    rewards = torch.stack(memory.rewards, dim=0)
    dones = torch.stack(memory.dones, dim=0)
    masks = torch.stack(memory.masks, dim=0)
    batch_size, max_seq_len = rewards.size()

    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    running_return = torch.zeros(batch_size, device=rewards.device)
    running_advantage = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(max_seq_len)):
        mask_t = masks[:, t]
        reward_t = rewards[:, t]
        done_t = dones[:, t]

        running_return = reward_t + gamma * running_return * (1 - done_t)
        td_error = reward_t + gamma * (returns[:, t + 1] if t + 1 < max_seq_len else 0) * (1 - done_t) - reward_t

        running_return *= mask_t
        td_error *= mask_t

        returns[:, t] = running_return
        running_advantage = td_error + gamma * lam * running_advantage * (1 - done_t)
        running_advantage *= mask_t
        advantages[:, t] = running_advantage

    memory.returns = returns
    memory.advantages = advantages


def custom_reward_function(predicted, label, predicted_distribution=None):
    reward = 0.0
    if predicted_distribution is not None:
        if predicted_distribution > 0.90:
            reward += -8.0
    if predicted == 1 and label == 0:
        return -22.0
    if predicted == 0 and label == 1:
        return -18.0
    if predicted == 1 and label == 1:
        return 16.0
    if predicted == 0 and label == 0:
        return 16.0
    return reward


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_layer = nn.Linear(64, action_dim)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_layer(x)
        state_value = self.value_layer(x)
        return action_logits, state_value


class PPOAgent(nn.Module):
    def __init__(self, learning_rate, batch_size, K_epochs, state_dim, action_dim, gru, mlp, clip_epsilon, entropy_coef,
                 device):
        super(PPOAgent, self).__init__()
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(gru.parameters()) + list(mlp.parameters()),
            lr=learning_rate
        )
        self.policy_old = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.K_epochs = K_epochs
        self.device = device
        self.hidden_size = state_dim
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        device = next(self.policy.parameters()).device
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().to(device)
        else:
            state = torch.tensor(state, dtype=torch.float).to(device)

        with torch.no_grad():
            action_logits, _ = self.policy_old(state)
        probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return actions, log_probs, entropy, probs

    def update(self, memory):
        states = torch.stack(memory.states).view(self.batch_size, -1, self.hidden_size).to(self.device)
        actions = torch.cat(memory.actions, dim=0)
        actions = actions.view(self.batch_size, -1).to(self.device)
        log_probs_old = torch.cat(memory.log_probs, dim=0).view(self.batch_size, -1).to(self.device)
        returns = memory.returns.view(self.batch_size, -1).to(self.device)
        advantages = memory.advantages.view(self.batch_size, -1).to(self.device)

        for _ in range(self.K_epochs):
            action_logits, state_values = self.policy(states)
            probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(probs)

            log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1)
            entropy = dist.entropy().mean()

            log_probs = log_probs.view_as(advantages)
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values.squeeze(), returns) - \
                   self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


class ATOM(BaseDefense):
    supported_api_types = {"pyg"}
    supported_datasets = {
        "Cora", "CiteSeer", "PubMed",
        "Computers", "Photo",
        "CS", "Physics"
    }

    def __init__(self, dataset: PyGIPDataset, attack_node_fraction: float = 0):
        super().__init__(dataset, attack_node_fraction)
        self.dataset = dataset
        # Load dataset related information, following BackdoorWM pattern
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data

        # Get basic information
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_data = self.graph_data.to(device=self.device)

    def _load_data_and_model(self, batch_size=16, seed=0, lamb=0):
        """
        Load data and model following BackdoorWM format
        Use data from self.dataset directly instead of reloading
        """
        # Get dataset name
        dataset_name = self.dataset.__class__.__name__

        # Build CSV path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'csv_data', f'attack_{dataset_name}.csv')

        # Load sequence data
        train_loader, val_loader, test_loader = build_loaders(
            csv_path=csv_path,
            batch_size=batch_size,
            drop_last=True,
            seed=seed
        )

        # Use PyG format directly (since we declared supported_api_types = {"pyg"})
        features = self.graph_data.x
        labels = self.graph_data.y
        train_mask = self.graph_data.train_mask
        test_mask = self.graph_data.test_mask

        # Create and train GCN model
        trained_gcn = GCN(self.num_features, 16, self.num_classes).to(self.device)
        target_model = TargetGCN(trained_model=trained_gcn, data=self.graph_data)

        # Train model
        self._train_gcn_model(trained_gcn, self.graph_data, train_mask, labels)

        # Compute k-core decomposition of the graph
        G = to_networkx(self.graph_data, to_undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G))
        k_core_values_graph = k_core_decomposition(G)
        max_k_core = torch.tensor(max(k_core_values_graph.values()), dtype=torch.float32)

        # Precompute embeddings for all nodes
        all_embeddings = precompute_all_node_embeddings(
            target_model, self.graph_data, k_core_values_graph, max_k_core, lamb=lamb
        )

        return train_loader, val_loader, test_loader, target_model, max_k_core, all_embeddings

    def _convert_to_networkx(self):
        """
        Convert graph data to NetworkX format, compatible with different graph data structures
        """
        if hasattr(self.graph_data, 'edge_index'):
            # PyG format
            return to_networkx(self.graph_data, to_undirected=True)
        else:
            # DGL format - requires conversion
            # This may need adjustment based on specific DGL format
            edge_list = []
            if hasattr(self.graph_data, 'edges'):
                src, dst = self.graph_data.edges()
                edge_list = list(zip(src.cpu().numpy(), dst.cpu().numpy()))

            G = nx.Graph()
            G.add_nodes_from(range(self.num_nodes))
            G.add_edges_from(edge_list)
            return G

    def _create_default_masks(self):
        """
        Create default train/test masks for datasets without predefined masks
        """
        num_nodes = self.num_nodes
        num_train = int(num_nodes * 0.6)
        num_test = int(num_nodes * 0.2)

        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)

        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:num_train + num_test]] = True

        return train_mask, test_mask

    @staticmethod
    def _train_gcn_model(model, data, train_mask, labels):
        """
        Train GCN model for PyG format
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.NLLLoss()

        features = data.x
        edge_index = data.edge_index

        for epoch in range(200):
            optimizer.zero_grad()
            output, _ = model(features, edge_index)
            loss = criterion(output[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"  Epoch {epoch}: Loss: {loss.item():.4f}")

    @staticmethod
    def _evaluate_downstream_task(model, data, test_mask, labels):
        """
        Evaluate model performance on downstream task
        """
        model.eval()
        with torch.no_grad():
            output, _ = model(data.x, data.edge_index)
            preds = output[test_mask].argmax(dim=1).cpu()
            true_labels = labels[test_mask].cpu()

        return preds, true_labels

    def defend(self):
        """
        Execute ATOM defense, following BackdoorWM's defend method structure
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()

        print("====================ATOM Defense====================")

        # Configuration parameters
        config = {
            "seed": 37719,
            "K_epochs": 10,
            "batch_size": 16,
            "hidden_size": 196,
            "hidden_action_dim": 16,
            "clip_epsilon": 0.30,
            "entropy_coef": 0.05,
            "lr": 1e-3,
            "gamma": 0.99,
            "lam": 0.95,
            "num_epochs": 2,
            "lamb": 0
        }

        # Set random seed
        set_seed(config["seed"])

        # Load data and model
        defense_s = time()
        train_loader, val_loader, test_loader, target_model, max_k_core, all_embeddings = self._load_data_and_model(
            batch_size=config["batch_size"],
            seed=config["seed"],
            lamb=config["lamb"]
        )

        # Set device and parameters
        device = self.device
        action_dim = 2
        hidden_size = config["hidden_size"]

        # Initialize model components
        input_size = self.num_classes
        gru = FusionGRU(input_size=input_size, hidden_size=hidden_size).to(device)
        mlp_transform = StateTransformMLP(action_dim, config["hidden_action_dim"], hidden_size).to(device)
        agent = PPOAgent(
            learning_rate=config["lr"],
            batch_size=config["batch_size"],
            K_epochs=config["K_epochs"],
            state_dim=hidden_size,
            action_dim=action_dim,
            gru=gru,
            mlp=mlp_transform,
            clip_epsilon=config["clip_epsilon"],
            entropy_coef=config["entropy_coef"],
            device=device
        ).to(device)

        # Training process
        memory = Memory()
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        auc_value_list = []

        for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs", ncols=100):
            # Training logic
            episode_reward = 0.0
            for batch_idx, (batch_seqs, batch_labels) in enumerate(train_loader):
                batch_labels = batch_labels.to(device)

                # Convert sequences to tensors and pad them
                batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
                padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)
                mask = (padded_seqs != 0).float().to(device)
                max_seq_len = padded_seqs.size(1)

                # Extract embeddings for all time steps
                all_inputs = []
                for t in range(max_seq_len):
                    node_indices = padded_seqs[:, t].tolist()
                    cur_inputs = all_embeddings[node_indices]
                    all_inputs.append(cur_inputs)
                all_inputs = torch.stack(all_inputs, dim=1).to(device)

                # Process through GRU
                hidden_states = gru.process_sequence(all_inputs)
                masked_hidden_states = hidden_states * mask.unsqueeze(-1)

                # Prepare probability factors
                prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)
                if memory.all_probs:
                    prob_factors[:, :-1] = torch.stack([
                        torch.tensor(memory.all_probs.get(t, [1.0] * action_dim))
                        for t in range(max_seq_len - 1)
                    ], dim=1).to(device)

                # Transform states
                custom_states = (mlp_transform(prob_factors) * masked_hidden_states).detach()

                # Select actions using PPO agent
                actions, log_probs, entropies, probs = agent.select_action(
                    custom_states.view(-1, hidden_size)
                )

                # Reshape outputs
                actions = actions.view(len(batch_seqs), max_seq_len)
                log_probs = log_probs.view(len(batch_seqs), max_seq_len)
                entropies = entropies.view(len(batch_seqs), max_seq_len)
                probs = probs.view(len(batch_seqs), max_seq_len, action_dim)

                # Initialize rewards and done flags
                rewards = torch.zeros(len(batch_seqs), max_seq_len, device=device)
                dones = torch.zeros(len(batch_seqs), max_seq_len, device=device)

                # Compute rewards
                batch_predictions = actions.cpu().numpy()
                predicted_distribution = (batch_predictions == 1).mean()
                last_valid_steps = mask.sum(dim=1).long() - 1

                for i in range(len(batch_seqs)):
                    for t in range(last_valid_steps[i] + 1):
                        if mask[i, t] == 1:
                            r = custom_reward_function(
                                actions[i, t].item(),
                                batch_labels[i].item(),
                                predicted_distribution
                            )
                            rewards[i, t] = r
                            episode_reward += r
                    dones[i, last_valid_steps[i]] = 1.0

                # Store experience in memory
                memory.store(custom_states, actions, log_probs, rewards, dones,
                             entropy=entropies, masks=mask)

                # Compute returns and advantages
                compute_returns_and_advantages(memory, gamma=config["gamma"], lam=config["lam"])

                # Update agent
                agent.update(memory)
                memory.clear()

            # Testing and evaluation
            agent.eval()
            gru.eval()
            mlp_transform.eval()
            with torch.no_grad():
                accuracy, precision, recall, f1, auc_value = test_model(
                    agent, gru, mlp_transform, test_loader,
                    target_model, self.graph_data, all_embeddings, hidden_size, device
                )
                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                auc_value_list.append(auc_value)

        defense_e = time()
        inference_s = time()

        # Calculate final watermark results
        wm_accuracy = np.mean(accuracy_list)
        wm_precision = np.mean(precision_list)
        wm_recall = np.mean(recall_list)
        wm_f1 = np.mean(f1_list)
        wm_auc = np.mean(auc_value_list)

        # Evaluate downstream task performance
        if hasattr(self.graph_data, 'test_mask'):
            test_mask = self.graph_data.test_mask
            labels = self.graph_data.y if hasattr(self.graph_data, 'y') else self.graph_data.ndata.get('label')
        else:
            _, test_mask = self._create_default_masks()
            labels = self.graph_data.y if hasattr(self.graph_data, 'y') else self.graph_data.ndata.get('label')

        downstream_preds, downstream_true = self._evaluate_downstream_task(
            target_model.model, self.graph_data, test_mask, labels
        )

        # Convert watermark results to tensors for metric computation
        wm_predictions = torch.tensor([1 if acc > 0.5 else 0 for acc in accuracy_list[-10:]])
        wm_targets = torch.ones_like(wm_predictions)

        inference_e = time()

        # Compute metrics
        metric = DefenseMetric()
        metric.update(downstream_preds, downstream_true)
        metric.update_wm(wm_predictions, wm_targets)

        metric_comp.update(
            defense_time=(defense_e - defense_s),
            inference_defense_time=(inference_e - inference_s)
        )
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        res_comp = metric_comp.compute()

        # Print detailed results
        print(f"Downstream Task - Accuracy: {res['accuracy']:.4f}, F1: {res['f1']:.4f}")
        print(f"Watermark Detection - Accuracy: {wm_accuracy:.4f}, AUC: {wm_auc:.4f}")
        print(f"Defense Time: {res_comp['defense_time']:.4f}s")
        print(f"Inference Time: {res_comp['inference_defense_time']:.4f}s")

        return res, res_comp
