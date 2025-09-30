import random
from time import time

import torch
import torch.nn.functional as F

from pygip.models.defense.base import BaseDefense
from pygip.models.nn import GCN
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric


class BackdoorWM(BaseDefense):
    supported_api_types = {"dgl"}

    def __init__(self, dataset, attack_node_fraction, model_path=None, trigger_rate=0.01, l=20, target_label=0):
        super().__init__(dataset, attack_node_fraction)
        # load data
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        self.graph_data = dataset.graph_data.to(device=self.device)
        self.model_path = model_path
        self.features = self.graph_data.ndata['feat']
        self.labels = self.graph_data.ndata['label']
        self.train_mask = self.graph_data.ndata['train_mask']
        self.test_mask = self.graph_data.ndata['test_mask']

        # load meta data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # wm params
        self.trigger_rate = trigger_rate
        self.l = l
        self.target_label = target_label

    def _load_model(self):
        """
        Load a pre-trained model.
        """
        assert self.model_path, "self.model_path should be defined"

        # Create the model
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)

        # Load the saved state dict
        self.net1.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # Set to evaluation mode
        self.net1.eval()

    def inject_backdoor_trigger(self, data, trigger_rate=None, trigger_feat_val=0.99, l=None, target_label=None):
        """Feature-based Trigger Injection"""
        if trigger_rate is None:
            trigger_rate = self.trigger_rate
        if l is None:
            l = self.l
        if target_label is None:
            target_label = self.target_label

        num_nodes = data.shape[0]
        num_feats = data.shape[1]
        num_trigger_nodes = int(trigger_rate * num_nodes)

        trigger_nodes = random.sample(range(num_nodes), num_trigger_nodes)
        for node in trigger_nodes:
            feature_indices = random.sample(range(num_feats), l)
            data[node][feature_indices] = trigger_feat_val
        return data, trigger_nodes

    def train_target_model(self, metric_comp: DefenseCompMetric):
        """
        Train the target model with backdoor injection.
        """
        # Initialize GNN model
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)

        # Inject backdoor trigger
        defense_s = time()
        poisoned_features = self.features.clone()
        poisoned_labels = self.labels.clone()

        poisoned_features_cpu = poisoned_features.cpu()
        poisoned_features_cpu, trigger_nodes = self.inject_backdoor_trigger(
            poisoned_features_cpu,
            trigger_rate=self.trigger_rate,
            l=self.l,
            target_label=self.target_label
        )
        poisoned_features = poisoned_features_cpu.to(self.device)

        # Modify labels for trigger nodes
        for node in trigger_nodes:
            poisoned_labels[node] = self.target_label

        self.trigger_nodes = trigger_nodes
        self.poisoned_features = poisoned_features
        self.poisoned_labels = poisoned_labels
        defense_e = time()

        # Training loop
        for epoch in range(200):
            self.net1.train()

            # Forward pass
            logits = self.net1(self.graph_data, poisoned_features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], poisoned_labels[self.train_mask])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation (optional)
            if epoch % 50 == 0:
                self.net1.eval()
                with torch.no_grad():
                    logits_val = self.net1(self.graph_data, poisoned_features)
                    logp_val = F.log_softmax(logits_val, dim=1)
                    pred = logp_val.argmax(dim=1)
                    acc_val = (pred[self.test_mask] == poisoned_labels[self.test_mask]).float().mean()
                    print(f"  Epoch {epoch}: training... Validation Accuracy: {acc_val.item():.4f}")

        metric_comp.update(defense_time=(defense_e - defense_s))

        return self.net1

    def verify_backdoor(self, model, trigger_nodes):
        """Verify backdoor attack success rate"""
        model.eval()
        with torch.no_grad():
            out = model(self.graph_data, self.poisoned_features)
            backdoor_preds = out.argmax(dim=1)[trigger_nodes]
            # correct = (pred[trigger_nodes] == target_label).sum().item()
        return backdoor_preds

    def evaluate_model(self, model, features):
        """Evaluate model performance"""
        model.eval()
        with torch.no_grad():
            out = model(self.graph_data, features)
            logits = out[self.test_mask]
            preds = logits.argmax(dim=1).cpu()

        return preds

    def defend(self):
        """
        Execute the backdoor watermark defense.
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================Backdoor Watermark====================")

        # If model wasn't trained yet, train it
        if not hasattr(self, 'net1'):
            self.train_target_model(metric_comp)

        # Evaluate the backdoored model
        preds = self.evaluate_model(self.net1, self.poisoned_features)
        inference_s = time()
        backdoor_preds = self.verify_backdoor(self.net1, self.trigger_nodes)
        inference_e = time()

        # metric
        metric = DefenseMetric()
        metric.update(preds, self.poisoned_labels[self.test_mask])
        target = torch.full_like(backdoor_preds, fill_value=self.target_label)
        metric.update_wm(backdoor_preds, target)
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        res_comp = metric_comp.compute()

        return res, res_comp
