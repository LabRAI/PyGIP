
import copy, gc, torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from ..core.base import BaseAttack
from ..models.graph_classifiers import GCN, GraphSAGE

class GNNFingersAttack(BaseAttack):
    supported_api_types = {"pyg"}
    supported_datasets  = {"ENZYMES"}
    def __init__(self, dataset, attack_node_fraction: float=0.1,
                 model_path=None, device=None, victim_hidden=128, victim_out=128):
        super().__init__(dataset, attack_node_fraction, model_path, device)
        self.victim_hidden = victim_hidden
        self.victim_out = victim_out
        self.target_model = None
        self.positive_models = []
        self.negative_models = []

    def attack(self):
        self.target_model = self._load_model() if self.model_path else self._train_target_model()
        self._generate_suspects()
        return {
            'target_model': self.target_model,
            'positive_models': self.positive_models,
            'negative_models': self.negative_models,
        }

    def _train_target_model(self, epochs=300, lr=1e-3, batch_size=16, early_stop_patience=30):
        model = GCN(self.num_features, hidden=self.victim_hidden, out_dim=self.victim_out, dropout=0.2, num_classes=self.num_classes).to(self.device)
        opt = Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
        train_loader = DataLoader(self.dataset.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset.val_data, batch_size=batch_size, shuffle=False)

        best_acc = 0.0
        patience_counter = 0

        for ep in range(1, epochs+1):
            model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                logits, _ = model(batch)
                loss = F.cross_entropy(logits, batch.y)
                loss.backward()
                opt.step()
            scheduler.step()

            # Validation
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    logits, _ = model(batch)
                    pred = logits.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            val_acc = correct / max(1, total)
            if val_acc > best_acc:
                best_acc = val_acc; patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"[Target] Early stopping at epoch {ep}")
                break
            if ep % 30 == 0:
                print(f"[Target] epoch {ep:03d}: Val Acc={val_acc:.4f}")

        return model.eval()

    def _load_model(self):
        return torch.load(self.model_path, map_location=self.device)

    def _train_attack_model(self):
        pass

    def _fine_tune(self, model, epochs=10, lr=1e-3, batch_size=32, seed=7):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr)
        train_loader = DataLoader(self.dataset.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset.val_data, batch_size=batch_size, shuffle=False)

        best = float('inf'); patience = 5; pat = 0
        self.target_model.eval()
        for ep in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                with torch.no_grad():
                    soft_logits, _ = self.target_model(batch)
                    soft = torch.softmax(soft_logits, dim=1)
                logits, _ = model(batch)
                loss = F.kl_div(torch.log_softmax(logits, dim=1), soft, reduction='batchmean')
                loss.backward(); opt.step()

            # Val
            model.eval(); vloss = 0.0; n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    soft_logits, _ = self.target_model(batch)
                    soft = torch.softmax(soft_logits, dim=1)
                    logits, _ = model(batch)
                    loss = F.kl_div(torch.log_softmax(logits, dim=1), soft, reduction='batchmean')
                    vloss += loss.item(); n += 1
            vloss /= max(1, n)
            if vloss < best: best = vloss; pat = 0
            else:
                pat += 1
                if pat >= patience:
                    print(f"[Fine-tune] Early stopping at epoch {ep+1}")
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model.eval()

    def _train_independent(self, model, epochs=100, lr=1e-3, batch_size=32, seed=0):
        torch.manual_seed(seed)
        opt = Adam(model.parameters(), lr=lr)
        train_loader = DataLoader(self.dataset.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset.val_data, batch_size=batch_size, shuffle=False)

        best = 0.0; patience = 20; pat = 0
        for ep in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                logits, _ = model(batch)
                loss = F.cross_entropy(logits, batch.y)
                loss.backward(); opt.step()

            # Val
            model.eval(); correct = 0; total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    logits, _ = model(batch)
                    pred = logits.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            acc = correct / max(1, total)
            if acc > best: best = acc; pat = 0
            else:
                pat += 1
                if pat >= patience:
                    print(f"[Neg model] Early stopping at epoch {ep+1}")
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model.eval()

    def _generate_suspects(self, k_pos=20, k_neg=20):
        print(f"[Attack] Generating suspect models: +{k_pos}, -{k_neg}")
        # positives
        for i in range(k_pos):
            m = copy.deepcopy(self.target_model).to(self.device)
            self._fine_tune(m, epochs=10, lr=1e-3, batch_size=32, seed=7+i)
            self.positive_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # negatives
        for i in range(k_neg):
            if i % 2 == 0:
                m = GCN(self.num_features, hidden=self.victim_hidden, out_dim=self.victim_out, num_classes=self.num_classes).to(self.device)
            else:
                m = GraphSAGE(self.num_features, hidden=self.victim_hidden, out_dim=self.victim_out, num_classes=self.num_classes).to(self.device)
            self._train_independent(m, epochs=100, lr=1e-3, batch_size=32, seed=100+i)
            self.negative_models.append(m.eval())
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("[Attack] Suspects done.")
