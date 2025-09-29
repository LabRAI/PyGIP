import torch
import torch.nn.functional as F
from torch import optim
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
        results = {}
        target_model = self._train_target_model()
        results["target_model_trained"] = True
        attack = MyCustomAttack(self.dataset, attack_node_fraction=0.3, device=self.device)
        results["attack_result_on_target"] = attack.attack()
        defense_model = self._train_defense_model()
        results["defense_model_trained"] = True
        attack2 = MyCustomAttack(self.dataset, attack_node_fraction=0.3, device=self.device)
        results["attack_result_on_defense"] = attack2.attack()
        results["target_accuracy"] = self._evaluate(target_model)
        results["defense_accuracy"] = self._evaluate(defense_model)
        return results

    def _load_model(self):
        return GCN(self.num_features, self.num_classes).to(self.device)

    def _train_target_model(self):
        return self._train(self._load_model())

    def _train_defense_model(self):
        return self._train(self._load_model())

    def _train_surrogate_model(self):
        return self._train(self._load_model())

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