import torch
import torch.nn.functional as F
from torch import optim
import dgl
import time
from models.defense.base import BaseDefense
from pygip.models.attack.my_custom_attack import MyCustomAttack
from pygip.models.nn import GCN

class MyCustomDefense(BaseDefense):
    supported_api_types = {"dgl"}
    supported_datasets = {"Cora"}

    def __init__(self, dataset, attack_node_fraction=0.3, hidden_dim=64, epochs=100, lr=0.01, weight_decay=5e-4, device=None):
        super().__init__(dataset, attack_node_fraction, device)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def defend(self):
        start_time = time.time()
        results = {}

        target_model = self._train_target_model()
        results["target_model_trained"] = True

        attack = MyCustomAttack(self.dataset, attack_node_fraction=self.attack_node_fraction, device=self.device)
        attack_result_target = attack.attack()
        results["attack_result_on_target"] = attack_result_target

        defense_model = self._train_defense_model_divergent()
        results["defense_model_trained"] = True

        attack2 = MyCustomAttack(self.dataset, attack_node_fraction=self.attack_node_fraction, device=self.device)
        attack_result_defense = attack2.attack()
        results["attack_result_on_defense"] = attack_result_defense

        tca = self._evaluate(target_model)
        eca = self._evaluate(defense_model)
        tba = attack_result_target.get("metrics", {}).get("TBA")
        eba = attack_result_defense.get("metrics", {}).get("EBA")
        fidelity = self._compute_fidelity(target_model, defense_model)

        results["metrics"] = {
            "TCA": tca,
            "ECA": eca,
            "TBA": tba,
            "EBA": eba,
            "Fidelity": fidelity
        }

        results["status"] = "defense completed"
        results["time_seconds"] = time.time() - start_time
        return results

    def _load_model(self):
        return GCN(self.num_features, self.num_classes).to(self.device)

    def _train_target_model(self):
        return self._train(self._load_model())

    def _train_defense_model_divergent(self):
        g = self.graph_data
        g_pruned = dgl.remove_self_loop(g)
        return self._train_noisy(self._load_model(), g_pruned, epochs=int(self.epochs * 0.7), lr=self.lr * 0.7)

    def _train(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        features = self.graph_data.ndata["feat"].to(self.device)
        labels = self.graph_data.ndata["label"].to(self.device)
        train_mask = self.graph_data.ndata["train_mask"].to(self.device)

        for _ in range(self.epochs):
            model.train()
            out = model(self.graph_data, features)
            loss = F.cross_entropy(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model

    def _train_noisy(self, model, graph, epochs, lr):
        graph = dgl.add_self_loop(graph)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        features = graph.ndata["feat"].to(self.device)
        labels = graph.ndata["label"].to(self.device)
        train_mask = graph.ndata["train_mask"].to(self.device)

        for _ in range(epochs):
            model.train()
            out = model(graph, features)
            loss = F.cross_entropy(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model

    def _evaluate(self, model):
        model.eval()
        features = self.graph_data.ndata["feat"].to(self.device)
        labels = self.graph_data.ndata["label"].to(self.device)
        test_mask = self.graph_data.ndata["test_mask"].to(self.device)
        with torch.no_grad():
            out = model(self.graph_data, features)
            pred = out.argmax(1)
            correct = (pred[test_mask] == labels[test_mask]).sum().item()
            acc = correct / test_mask.sum().item()
        return acc

    def _compute_fidelity(self, model1, model2):
        model1.eval()
        model2.eval()
        features = self.graph_data.ndata["feat"].to(self.device)
        with torch.no_grad():
            out1 = model1(self.graph_data, features).argmax(1)
            out2 = model2(self.graph_data, features).argmax(1)
            agreement = (out1 == out2).sum().item() / len(out1)
        return agreement
