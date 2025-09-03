
import random, torch, torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Union
from ..core.base import BaseDefense
from ..fingerprints.pairs import FingerprintSet, Verifier

class GNNFingersDefense(BaseDefense):
    supported_api_types = {"pyg"}
    supported_datasets  = {"AIDS"}
    def __init__(self, dataset, attack_node_fraction: float=0.1,
                 device: Optional[Union[str, torch.device]]=None,
                 P=64, fp_nodes=32, backbone_out=64, joint_iters=150,
                 verif_lr=5e-4, fp_lr=2e-3):
        super().__init__(dataset, attack_node_fraction, device)
        self.P = P
        self.fp_nodes = fp_nodes
        self.backbone_out = backbone_out
        self.joint_iters = joint_iters
        self.verif_lr = verif_lr
        self.fp_lr = fp_lr
        self.fp_set = FingerprintSet(P=self.P, n_nodes=self.fp_nodes, feat_dim=self.num_features, device=self.device)
        in_dim = self.fp_set.dim_out(self.backbone_out)
        self.verifier = Verifier(in_dim).to(self.device)

    def defend(self, attack_results=None):
        if attack_results is None:
            from ..attacks.gnnfingers_aids import GNNFingersAttack
            attack = GNNFingersAttack(self.dataset, self.attack_node_fraction, device=self.device,
                                      victim_hidden=64, victim_out=self.backbone_out)
            attack_results = attack.attack()
        tgt = attack_results['target_model']
        pos = attack_results['positive_models']
        neg = attack_results['negative_models']
        self._joint_train(tgt, pos, neg)
        return {'fingerprint_set': self.fp_set, 'verifier': self.verifier}

    def _joint_train(self, target_model, pos_models, neg_models):
        opt_ver = Adam(self.verifier.parameters(), lr=self.verif_lr)
        fp_params = [p for fp in self.fp_set.fps for p in fp.parameters()]
        opt_fp = Adam(fp_params, lr=self.fp_lr)

        for it in range(1, self.joint_iters+1):
            p_m = random.choice([target_model] + pos_models).to(self.device)
            n_m = random.choice(neg_models).to(self.device)
            for p in fp_params: p.requires_grad_(False)
            self.verifier.train()
            opt_ver.zero_grad()
            X_pos = self.fp_set.model_response(p_m, require_grad=False)
            X_neg = self.fp_set.model_response(n_m, require_grad=False)
            logits = self.verifier(torch.cat([X_pos, X_neg], dim=0))
            labels = torch.tensor([1, 0], device=self.device, dtype=torch.long)
            loss_ver = F.cross_entropy(logits, labels)
            loss_ver.backward(); opt_ver.step()

            for p in fp_params: p.requires_grad_(True)
            self.verifier.eval()
            opt_fp.zero_grad()
            X_pos = self.fp_set.model_response(p_m, require_grad=True)
            X_neg = self.fp_set.model_response(n_m, require_grad=True)
            logits = self.verifier(torch.cat([X_pos, X_neg], dim=0))
            labels = torch.tensor([1, 0], device=self.device, dtype=torch.long)
            loss_fp = F.cross_entropy(logits, labels)
            loss_fp.backward(); opt_fp.step()

            if it % 10 == 0:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    print(f"[Joint] iter {it:03d} | L_ver {loss_ver.item():.4f} | L_fp {loss_fp.item():.4f} | pos_p~{probs[0].item():.3f} | neg_p~{probs[1].item():.3f}")

    @torch.no_grad()
    def verify_ownership(self, suspect_model):
        suspect_model.eval()
        x = self.fp_set.model_response(suspect_model, require_grad=False)
        logits = self.verifier(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        return {'is_stolen': int(prob > 0.5), 'confidence': prob}
