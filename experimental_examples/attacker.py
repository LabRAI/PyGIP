# Attack class —  model training, fingerprint learning, evaluation, and defense hooks

from torch_geometric.nn import GCNConv, SAGEConv
import os, json, random
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from models import SmallGCN, SmallSAGE
from fingerprints import LearnableFingerprint, Univerifier

@dataclass
class FingerprintSpecLocal:
    num_graphs: int =64
    num_nodes: int = 32
    edge_density: float = 0.05
    proj_every: int = 25
    update_feat: bool = True
    update_adj: bool = True
    node_sample: int = 0


def make_deterministic(seed: int =123):
    # Set seeds for reproducibility.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False


class GNNFingersAttack(object):
    supported_api_types = {'pyg'}

    def __init__(self, dataset, attack_node_fraction: float = 0.3, model_path: Optional[str] =None,
                 fp_cfg: FingerprintSpecLocal = FingerprintSpecLocal(),joint_steps: int = 300, device: Optional[torch.device] = None):
        self.device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') )
        print(f"[GNNFingersAttack] using device: {self.device}")

        self.dataset =dataset
        self.graph_data = dataset.graph_data.to(self.device)

        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.attack_node_fraction = attack_node_fraction
        self.model_path = model_path

        self.fp_cfg = fp_cfg
        self.joint_steps = joint_steps

        self.feat_min= float(self.graph_data.x.min().item())
        self.feat_max = float(self.graph_data.x.max().item())

    def build_target(self, kind='gcn'):
        if kind == 'sage':
            return SmallSAGE(self.num_features,64, self.num_classes).to(self.device)
        return SmallGCN(self.num_features, 64,self.num_classes).to(self.device)

    def run_train_epoch(self, m, feat, ei, mask, labels, opt, edge_weight=None):
        m.train()
        opt.zero_grad()
        out =m(feat, ei, edge_weight=edge_weight)
        loss = F.cross_entropy(out[mask], labels[mask])
        loss.backward()
        opt.step()
        return loss.item()

    @torch.no_grad()
    def eval_split(self, m, feat, ei, labels, mask, edge_weight=None):
        m.eval()
        logits = m(feat, ei, edge_weight=edge_weight)
        predict = logits.argmax(dim=1)
        accuracy= (predict[mask] == labels[mask]).float().mean().item()
        return accuracy, logits

    # Target model utilities
    def _train_target_model(self, arch='gcn', epochs=200):
        make_deterministic(42)
        target_model = self.build_target(kind=arch)
        opt = torch.optim.Adam(target_model.parameters(), lr=0.01,weight_decay=5e-4)

        best_val, best_state = 0.0, None
        for ep in range(1, epochs+1):
            loss = self.run_train_epoch(target_model, self.graph_data.x, self.graph_data.edge_index,self.graph_data.train_mask, self.graph_data.y, opt)
            va, _ = self.eval_split(target_model, self.graph_data.x, self.graph_data.edge_index, self.graph_data.y,self.graph_data.val_mask)
            te, _ =self.eval_split(target_model, self.graph_data.x, self.graph_data.edge_index, self.graph_data.y, self.graph_data.test_mask)
            if va > best_val:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
            if ep in (1, 50, 100, 150, 200):
                print(f"[target] ep= {ep:3d} | loss = {loss:.4f} | val = {va:.4f} | test = {te:.4f}")

        if best_state:
            target_model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return target_model

    def _load_model(self, path):
        m = self.build_target(kind=os.environ.get('TARGET_ARCH', 'gcn'))
        m.load_state_dict(torch.load(path, map_location=self.device))
        m.to(self.device).eval()
        return m

    # Attack model training (fingerprints + univerifier)
    def _train_attack_model(self, target_model_path: Optional[str] =None, joint_steps: Optional[int] = None):
        if joint_steps is None:
            joint_steps = self.joint_steps

        suspects: List[Tuple[str, str, int]]= []

        if target_model_path:
            target_model = self._load_model(target_model_path)
        else:
            target_model = self._train_target_model()

        torch.save(target_model.state_dict(), './target_main.pt')
        suspects.append(('target', './target_main.pt', 1))

        def copy_model_like(m):
            new_m = self.build_target(kind='gcn' if isinstance(m, SmallGCN) else 'sage')
            new_m.load_state_dict(m.state_dict())
            return new_m

        def fine_tune_last_layer(m, steps=10, lr=0.01):
            for p in m.parameters():
                p.requires_grad = False
            last = None
            for module in m.modules():
                if isinstance(module, (GCNConv, SAGEConv)):
                    last = module
            for p in last.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=lr,weight_decay=5e-4)
            for _ in range(steps):
                self.run_train_epoch(m, self.graph_data.x, self.graph_data.edge_index, self.graph_data.train_mask, self.graph_data.y, opt)
            return m

        def partial_reinit_and_train(m,steps=10, lr=0.01):
            last =None
            for module in m.modules():
                if isinstance(module,(GCNConv, SAGEConv)):
                    last = module
            if hasattr(last,'reset_parameters'):
                last.reset_parameters()
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
            for _ in range(steps):
                self.run_train_epoch(m, self.graph_data.x, self.graph_data.edge_index, self.graph_data.train_mask, self.graph_data.y, opt)
            return m

        def fine_tune_all(m, steps=10, lr=0.005):
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
            for _ in range(steps):
                self.run_train_epoch(m, self.graph_data.x, self.graph_data.edge_index, self.graph_data.train_mask, self.graph_data.y, opt)
            return m

        make_deterministic(11)
        m1 = copy_model_like(target_model)
        torch.save(fine_tune_last_layer(m1).state_dict(),'./ft_last.pt')
        suspects.append(('ft_last', './ft_last.pt', 1))

        make_deterministic(12)
        m2 = copy_model_like(target_model)
        torch.save(partial_reinit_and_train(m2).state_dict(),'./reinit_last.pt')
        suspects.append(('reinit_last', './reinit_last.pt', 1))

        make_deterministic(13)
        m3 = copy_model_like(target_model)
        torch.save(fine_tune_all(m3).state_dict(), './ft_all.pt')
        suspects.append(('ft_all', './ft_all.pt', 1))

        for seed in (100, 101, 102):
            make_deterministic(seed)
            mn = self.build_target(kind=os.environ.get('NEG_ARCH', 'gcn'))
            opt = torch.optim.Adam(mn.parameters(), lr=0.01, weight_decay=5e-4)
            for _ in range(100):
                self.run_train_epoch(mn, self.graph_data.x, self.graph_data.edge_index, self.graph_data.train_mask, self.graph_data.y, opt)
            path = f'./neg_{seed}.pt'
            torch.save(mn.state_dict(), path)
            suspects.append((f'neg_{seed}', path, 0))

        print('[info] Built suspects:', [s[0] for s in suspects])

        model_entries = [(nm,self._load_model(pth), lbl) for (nm, pth, lbl) in suspects]

        fp_pool: List[LearnableFingerprint] = [
            LearnableFingerprint(self.fp_cfg.num_nodes, self.num_features, self.fp_cfg.edge_density, device=self.device).to(self.device)
            for _ in range(self.fp_cfg.num_graphs)
        ]

        dummy_sig = self.get_signature_from_model(model_entries[0][1], fp_pool, m_nodes=self.fp_cfg.node_sample)
        uv = Univerifier(in_dim=dummy_sig.numel()).to(self.device)

        fp_params =[]
        for fp in fp_pool:
            if self.fp_cfg.update_feat:
                fp_params.append(fp.feat)
            if self.fp_cfg.update_adj:
                fp_params.append(fp.adj_param)
        opt_fp = torch.optim.Adam(fp_params,lr=0.05)
        opt_uv = torch.optim.Adam(uv.parameters(), lr=1e-3,weight_decay=1e-4)

        print(f"[info] Joint steps: {joint_steps} | proj_every={self.fp_cfg.proj_every} | update_feat={self.fp_cfg.update_feat} | update_adj={self.fp_cfg.update_adj}")

        for step in range(1, joint_steps+1):
            uv.train()
            batch_inputs, batch_labels = [],[]
            for (nm, mdl, lbl) in model_entries:
                sig_pieces = []
                for fp in fp_pool:
                    out = fp.forward(mdl)
                    if self.fp_cfg.node_sample and 0 < self.fp_cfg.node_sample < fp.num_nodes:
                        idx = torch.randperm(fp.num_nodes,device=out.device)[:self.fp_cfg.node_sample]
                        probs = out[idx].softmax(dim =-1).mean(dim =0)
                    else:
                        probs =out.softmax(dim =-1).mean(dim=0)
                    sig_pieces.append(probs)
                sig_all = torch.cat(sig_pieces,dim = 0)
                batch_inputs.append(sig_all.unsqueeze(0))
                batch_labels.append(torch.tensor([lbl],device=self.device, dtype=torch.long))

            Xb= torch.cat(batch_inputs, dim=0)
            yb = torch.cat(batch_labels,dim=0)
            logits = uv(Xb.float())
            loss =F.cross_entropy(logits, yb)

            opt_uv.zero_grad()
            opt_fp.zero_grad()
            loss.backward()

            with torch.no_grad():
                for fp in fp_pool:
                    if self.fp_cfg.update_feat:
                        fp.feat.clamp_(self.feat_min, self.feat_max)

            opt_uv.step()
            opt_fp.step()

            if self.fp_cfg.update_adj and (step % self.fp_cfg.proj_every == 0 or step == joint_steps):
                for fp in fp_pool:
                    fp.harden_topk(self.fp_cfg.edge_density)

            if step % 25 == 0 or step == 1 or step == joint_steps:
                with torch.no_grad():
                    probs = logits.softmax(dim=1)[:,1]
                    avg_pos = probs[(yb==1)].mean().item() if (yb==1).any() else float('nan')
                    avg_neg = probs[(yb==0)].mean().item() if (yb==0).any() else float('nan')
                print(f"[joint] step={step:3d} | loss={loss.item():.4f} | avg_pos={avg_pos:.3f} | avg_neg={avg_neg:.3f}")

        torch.save(uv.state_dict(),'./univerifier.pt')
        torch.save({f'fp_{i}': (fp.feat.detach().cpu(), fp.adj_param.detach().cpu()) for i, fp in enumerate(fp_pool)}, './fingerprints.pt')

        metrics = self.evaluate_curves(uv, model_entries, fp_pool)
        with open('./verification_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Saved verification_metrics.json with labels and probs included.')

        return metrics

    # Public attack entrypoint
    def attack(self, *args, **kwargs):
        return self._train_attack_model(*args, **kwargs)


    # Defense interface + helpers
    def defense(self, method: str = 'default'):
        print(f"[defense] running defense with method={method}")

        # Train or load the victim (target) model
        target = self._train_target_model()

        # Optionally train a surrogate attack model (for adversarial defenses that need it)
        surrogate = self._train_surrogate_model()  # returns None if not used

        #Train defense model (depending on method)
        defense_model = self._train_defense_model(method = method, target_model = target, surrogate_model = surrogate)

        # Test defense: Ruyns the fingerprint-based verifier against the defended model
        metrics = None
        try:
            # reuse attack pipeline but evaluate on defended model as one of the suspects
            torch.save(defense_model.state_dict(), './defended_model.pt')
            suspects = [('defended', './defended_model.pt', 1)]

            # keep a copy of original suspects by reusing _train_attack_model's suspect creation
            # but here we will only run the verifier's evaluation stage using the defended model as positive
            # To be quick, load a subset of models from disk (target, negatives) and include defended
            # For simplicity, reuse existing saved files if present
            saved = []
            if os.path.exists('./target_main.pt'):
                saved.append(('target', './target_main.pt', 1))
            for seed in (100,101,102):
                p = f'./neg_{seed}.pt'
                if os.path.exists(p):
                    saved.append((f'neg_{seed}', p, 0))
            # append defended
            saved.append(('defended', './defended_model.pt', 1))

            model_entries = [(nm, self._load_model(pth), lbl) for (nm, pth, lbl) in saved]

            # build fingerprints (reuse fingerprint config)
            fp_pool: List[LearnableFingerprint] = [
                LearnableFingerprint(self.fp_cfg.num_nodes, self.num_features, self.fp_cfg.edge_density, device=self.device).to(self.device)
                for _ in range(self.fp_cfg.num_graphs)
            ]

            dummy_sig = self.get_signature_from_model(model_entries[0][1], fp_pool, m_nodes=self.fp_cfg.node_sample)
            uv = Univerifier(in_dim = dummy_sig.numel()).to(self.device)

            # quick joint training of univerifier only (fp fixed) to see verification metrics against defended model
            X = self.collect_signatures_all(model_entries, fp_pool, m_nodes=self.fp_cfg.node_sample).to(self.device).float()
            logits = uv(X)
            prob_pos = logits.softmax(dim=1)[:,1]
            # compute simple ROC-AUC for this quick eval
            from sklearn.metrics import roc_auc_score
            labels = [lbl for (_,_,lbl) in model_entries]
            auc_val = roc_auc_score(labels, prob_pos.detach().cpu().numpy())
            metrics = {'quick_ROC_AUC': float(auc_val)}
        except Exception as e:
            print('[defense] evaluation failed: ', e)

        return {
            'defense_model': defense_model,
            'surrogate': surrogate,
            'metrics': metrics
        }

    def _train_defense_model(self, method: str = 'default', target_model = None, surrogate_model = None):
        #Trains a defense model. This is a stub that demonstrates the expected interface.

        print(f"[_train_defense_model] training defense using method={method}")

        # simple baseline
        if target_model is None:
            target_model = self._train_target_model()

        def partial_reinit_and_train(m, steps=10, lr=0.01):
            last = None
            for module in m.modules():
                if isinstance(module, (GCNConv, SAGEConv)):
                    last = module
            if hasattr(last, 'reset_parameters'):
                last.reset_parameters()
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
            for _ in range( steps):
                self.run_train_epoch(m, self.graph_data.x, self.graph_data.edge_index, self.graph_data.train_mask, self.graph_data.y, opt)
            return m

        defended = partial_reinit_and_train(self.build_target(), steps=10)
        return defended

    def _train_surrogate_model(self):
        #Trains a surrogate (attack) model. Returns a model instance or None if not needed.

        print('[ _train_surrogate_model ] training surrogate model (simple retrain)')
        make_deterministic(21)
        mn= self.build_target()
        opt = torch.optim.Adam(mn.parameters(), lr=0.01, weight_decay=5e-4)
        for _ in range(50):
            self.run_train_epoch(mn, self.graph_data.x, self.graph_data.edge_index, self.graph_data.train_mask, self.graph_data.y, opt)
        return mn

    # Signature utilities
    @torch.no_grad()
    def get_signature_from_model(self, m, fps: List[LearnableFingerprint],m_nodes: int = 0):
        pieces = []
        for fp in fps:
            out = fp.forward(m)
            if m_nodes and 0 < m_nodes < fp.num_nodes:
                idx =torch.randperm(fp.num_nodes, device=out.device)[:m_nodes]
                probs = out[idx].softmax(dim=-1).mean(dim=0)
            else:
                probs =out.softmax(dim=-1).mean(dim=0)
            pieces.append(probs)
        return torch.cat(pieces, dim=0)

    @torch.no_grad()
    def collect_signatures_all(self, models, fps, m_nodes =0):
        bag = []
        for (_, mdl, _lbl) in models:
            bag.append(self.get_signature_from_model(mdl,fps, m_nodes = m_nodes).unsqueeze(0))
        X = torch.cat(bag, dim=0)
        return X

    def evaluate_curves(self, uv,models, fps, thresholds = None):
        uv.eval()
        if thresholds is None:
            thresholds = torch.linspace(0.0, 1.0, steps = 101)
        labels = torch.tensor([lbl for (_, _, lbl) in models], device = self.device)
        X = self.collect_signatures_all(models, fps, m_nodes = self.fp_cfg.node_sample).to(self.device).float()
        logits =uv(X)
        prob_pos = logits.softmax(dim = 1)[:,1]

        labels_list= labels.detach().cpu().numpy().tolist()
        probs_list= prob_pos.detach().cpu().numpy().tolist()

        pos_mask = labels == 1
        neg_mask = labels ==0

        rob_list, uniq_list, acc_list = [], [],[]
        for t in thresholds:
            pred_pos = (prob_pos >=t).long()
            tp = ((pred_pos == 1) & pos_mask).sum().item()
            tn =((pred_pos == 0) &neg_mask).sum().item()
            p_total = pos_mask.sum().item()
            n_total = neg_mask.sum().item()
            robustness =tp /max(1, p_total)
            uniqueness = tn / max(1, n_total)
            mean_acc = (tp + tn) / max(1, (p_total + n_total))
            rob_list.append(robustness)
            uniq_list.append(uniqueness)
            acc_list.append(mean_acc)

        import numpy as np
        inter = np.minimum(np.array(rob_list), np.array(uniq_list))
        aruc= float(np.trapezoid(inter, x=np.linspace(0, 1, len(inter))))
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(labels_list, probs_list)

        return {
            'thresholds': thresholds.tolist(),
            'robustness': rob_list,
            'uniqueness': uniq_list,
            'mean_accuracy': acc_list,
            'ARUC': aruc,
            'ROC_AUC': auc_val,
            'labels': labels_list,
            'probs': probs_list
        }
