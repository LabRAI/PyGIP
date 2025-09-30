import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse
from tqdm import tqdm

from pygip.models.attack.base import BaseAttack
from pygip.models.nn import GCN
from pygip.utils.metrics import AttackMetric, AttackCompMetric


class AdvMEA(BaseAttack):
    supported_api_types = {"dgl"}

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.graph = dataset.graph_data.to(self.device)
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        self.train_mask = self.graph.ndata['train_mask']
        self.test_mask = self.graph.ndata['test_mask']

        # meta data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        if model_path is None:
            self._train_target_model()
        else:
            self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load a pre-trained model.
        """
        # Create the model
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)

        # Load the saved state dict
        self.net1.load_state_dict(torch.load(model_path, map_location=self.device))

        # Set to evaluation mode
        self.net1.eval()

    def _train_target_model(self):
        """
        Train the target model (GCN) on the original graph.
        """
        # Initialize GNN model
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)

        # Training loop
        for epoch in range(200):
            self.net1.train()

            # Forward pass
            logits = self.net1(self.graph, self.features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation (optional)
            if epoch % 20 == 0:
                self.net1.eval()
                with torch.no_grad():
                    logits_val = self.net1(self.graph, self.features)
                    logp_val = F.log_softmax(logits_val, dim=1)
                    pred = logp_val.argmax(dim=1)
                    acc_val = (pred[self.test_mask] == self.labels[self.test_mask]).float().mean()
                    # You could print validation accuracy here

        return self.net1

    # Define a local to_cpu method to avoid inheritance issues
    def _to_cpu(self, tensor):
        """
        Safely move tensor to CPU for NumPy operations
        """
        if tensor.is_cuda:
            return tensor.cpu()
        return tensor

    def attack(self):
        metric_comp = AttackCompMetric()
        metric_comp.start()
        g = self.graph.clone()
        # Move adjacency matrix to CPU for NumPy operations
        g_matrix = np.asmatrix(self._to_cpu(g.adjacency_matrix().to_dense()).numpy())
        edge_index = np.array(np.nonzero(g_matrix))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        attack_s1 = time.time()
        # Select a center node with certain size
        while True:
            node_index = torch.randint(0, self.num_nodes, (1,)).item()
            # print("node_index=",node_index)
            sub_node_index, sub_edge_index, _, _ = k_hop_subgraph(node_index, 2, edge_index, relabel_nodes=True,
                                                                  num_nodes=self.num_nodes)
            if 45 <= sub_node_index.size(0) <= 50:
                As = torch.zeros((sub_node_index.size(0), sub_node_index.size(0)))
                As[sub_edge_index[0], sub_edge_index[1]] = 1
                print("sub_node_index=", sub_node_index.size(0))
                # Ensure moved to CPU
                Xs = self._to_cpu(self.features[sub_node_index])
                break

        # Construct the prior distribution
        Fd = []
        Md = []
        for label in range(self.num_classes):
            # Ensure moved to CPU before converting to NumPy
            features_cpu = self._to_cpu(self.features)
            labels_cpu = self._to_cpu(self.labels)
            class_nodes = features_cpu[labels_cpu == label].numpy()

            feature_counts = class_nodes.sum(axis=0)
            feature_distribution = feature_counts / feature_counts.sum()
            Fd.append(feature_distribution)

            num_features_per_node = class_nodes.sum(axis=1)
            feature_count_distribution = np.bincount(num_features_per_node.astype(int), minlength=self.num_features)
            Md.append(feature_count_distribution / feature_count_distribution.sum())

        SA = [As]
        SX = [Xs]
        attack_e1 = time.time()

        query_s = time.time()
        # Query the target model
        self.net1.eval()
        with torch.no_grad():
            logits_query = self.net1(g, self.features)
            _, labels_query = torch.max(logits_query, dim=1)

        query_e = time.time()

        attack_s2 = time.time()
        src, dst = As.nonzero(as_tuple=True)
        initial_num_nodes = Xs.shape[0]
        initial_graph = dgl.graph((src, dst), num_nodes=initial_num_nodes).to(self.device)
        initial_graph.ndata['feat'] = Xs.to(self.device)

        self.net1.eval()
        with torch.no_grad():
            initial_query = self.net1(initial_graph, initial_graph.ndata['feat'])
            _, initial_label = torch.max(initial_query, dim=1)

        SL = self._to_cpu(initial_label).tolist()
        samples_per_class = 10
        n = samples_per_class

        for i in range(n):
            # For each class, generate and store a new sampled subgraph
            for c in range(self.num_classes):
                num_nodes = As.shape[0]
                Ac = torch.ones((num_nodes, num_nodes))
                Xc = torch.zeros(num_nodes, len(Fd[c]))
                for j in range(num_nodes):  # Use j to avoid conflict with outer loop variable i
                    m = np.random.choice(np.arange(len(Md[c])), p=Md[c])
                    features_idx = np.random.choice(len(Fd[c]), size=int(m), replace=False, p=Fd[c])
                    Xc[j, features_idx] = 1
                SA.append(Ac)
                SX.append(Xc)

                src, dst = Ac.nonzero(as_tuple=True)
                subgraph = dgl.graph((src, dst), num_nodes=num_nodes).to(self.device)
                subgraph.ndata['feat'] = Xc.to(self.device)

                self.net1.eval()
                with torch.no_grad():
                    api_query = self.net1(subgraph, subgraph.ndata['feat'])
                    _, label_query = torch.max(api_query, dim=1)

                SL.extend(self._to_cpu(label_query).tolist())

        AG_list = [dense_to_sparse(torch.tensor(a))[0] for a in SA]
        XG = torch.vstack([torch.tensor(x) for x in SX])

        SL = torch.tensor(SL, dtype=torch.long)

        # Filter valid labels and trim
        valid_mask = SL >= 0
        SL = SL[valid_mask]
        SL = SL[:XG.shape[0]]

        # Calculate nodes per subgraph
        num_nodes = XG.shape[0] // len(AG_list) if len(AG_list) > 0 else 0

        # Combine edge indices from all subgraphs, adjusting node indices to avoid overlap
        AG_combined = torch.cat([edge_index + i * num_nodes for i, edge_index in enumerate(AG_list)], dim=1)

        src, dst = AG_combined[0], AG_combined[1]
        num_total_nodes = XG.shape[0]
        sub_g = dgl.graph((src, dst), num_nodes=num_total_nodes).to(self.device)
        sub_g.ndata['feat'] = XG.to(self.device)

        attack_e2 = time.time()

        train_surrogate_s = time.time()
        # Create and train the extracted model
        net6 = GCN(XG.shape[1], self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(net6.parameters(), lr=0.01, weight_decay=5e-4)

        print("=========Model Extracting==========================")
        metric = AttackMetric()

        for epoch in tqdm(range(200)):
            net6.train()
            logits = net6(sub_g, sub_g.ndata['feat'])
            out = torch.log_softmax(logits, dim=1)
            loss = F.nll_loss(out, SL.to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Switch to evaluation mode
            t0 = time.time()
            net6.eval()
            with torch.no_grad():
                logits = net6(g, self.features)
                _, preds = torch.max(logits[self.test_mask], dim=1)
                metric.update(preds, self.labels[self.test_mask], labels_query[self.test_mask])
            metric_comp.update(inference_surrogate_time=(time.time() - t0))

        train_surrogate_e = time.time()

        print("========================Final results:=========================================")
        metric_comp.end()
        metric_comp.update(attack_time=(attack_e1 - attack_s1 + attack_e2 - attack_s2),
                           query_target_time=(query_e - query_s),
                           train_surrogate_time=(train_surrogate_e - train_surrogate_s))
        res = metric.compute()
        res_comp = metric_comp.compute()

        return res, res_comp
