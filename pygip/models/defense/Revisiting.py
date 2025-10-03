from typing import Any, Dict, Iterable, Tuple

import dgl
import torch
import torch.nn.functional as F
from dgl.dataloading import NeighborSampler, NodeCollator
from torch.utils.data import DataLoader
from tqdm import tqdm

from pygip.models.defense.base import BaseDefense
from pygip.models.nn import GraphSAGE
from pygip.utils.metrics import DefenseCompMetric


class Revisiting(BaseDefense):
    """
    A lightweight defense that 'revisits' node features via neighbor mixing.

    Idea (defense intuition)
    ------------------------
    We pick a subset of nodes (size ~ attack_node_fraction * |V|) and *smoothly*
    mix their features with their 1-hop / 2-hop neighborhoods using a mixing
    factor `alpha`. This keeps utility (accuracy) largely intact while making
    local feature structure less extractable for subgraph-based queries.

    API shape follows RandomWM:
      - lives under models/defense/
      - inherits BaseDefense
      - public entrypoint: .defend()

    Parameters
    ----------
    dataset : Any
        A dataset object providing a DGLGraph in `dataset.graph_data` and
        ndata fields: 'feat', 'label', 'train_mask', 'test_mask'.
    attack_node_fraction : float, default=0.2
        Fraction of nodes used as the 'focus set' for our revisiting transform.
    alpha : float, default=0.8
        Mixing coefficient in [0,1]. Higher -> stronger neighbor mixing.
    """

    supported_api_types = {"dgl"}

    def __init__(
            self,
            dataset: Any,
            attack_node_fraction: float = 0.2,
            alpha: float = 0.8,
    ) -> None:
        super().__init__(dataset, attack_node_fraction)

        # knobs
        self.alpha = float(alpha)

        # cache handles similar to RandomWM for consistency
        self.dataset = dataset
        self.graph: dgl.DGLGraph = dataset.graph_data

        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_focus_nodes = max(1, int(self.num_nodes * attack_node_fraction))

        self.features: torch.Tensor = self.graph.ndata["feat"]
        self.labels: torch.Tensor = self.graph.ndata["label"]
        self.train_mask: torch.Tensor = self.graph.ndata["train_mask"]
        self.test_mask: torch.Tensor = self.graph.ndata["test_mask"]

        if self.device != "cpu":
            self.graph = self.graph.to(self.device)
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
            self.train_mask = self.train_mask.to(self.device)
            self.test_mask = self.test_mask.to(self.device)

    # --------------------------------------------------------------------- #
    # Public entrypoint
    # --------------------------------------------------------------------- #
    def defend(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        1) Train a baseline GraphSAGE on the original graph (utility baseline)
        2) Apply revisiting feature-mixing on a subset of nodes
        3) Train a defended GraphSAGE on the transformed features
        4) Return accuracy metrics and basic metadata
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        # ---- Baseline (no transform) ------------------------------------- #
        baseline_acc = self._train_and_eval_graphsage(use_transformed_features=False)

        # ---- Build transformed features (revisiting) --------------------- #
        feat_defended, picked = self._build_revisiting_features()

        # ---- Train with defended features -------------------------------- #
        # Temporarily override graph features, then restore
        orig_feat = self.graph.ndata["feat"]
        try:
            self.graph.ndata["feat"] = feat_defended
            defense_acc = self._train_and_eval_graphsage(use_transformed_features=True)
        finally:
            self.graph.ndata["feat"] = orig_feat  # restore

        res = {
            "ok": True,
            "method": "Revisiting",
            "alpha": self.alpha,
            "focus_nodes": int(self.num_focus_nodes),
            "baseline_test_acc": float(baseline_acc),
            "defense_test_acc": float(defense_acc),
            "acc_delta": float(defense_acc - baseline_acc),
            # returning a small sample of picked nodes for debuggability
            "sample_picked_nodes": picked[:10].tolist() if isinstance(picked, torch.Tensor) else [],
        }

        return res, metric_comp.compute()

    # --------------------------------------------------------------------- #
    # Core: feature revisiting (neighbor mixing)
    # --------------------------------------------------------------------- #
    def _build_revisiting_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a new feature tensor where a subset of nodes (and optionally
        their neighbors) are mixed with neighbor features.

        Mixing rule (simple & stable):
          - For each picked node u:
              x[u] <- (1 - alpha) * x[u] + alpha * mean(x[N(u)])
          - For each 1-hop neighbor v in N(u) we apply a *lighter* mix
              x[v] <- (1 - 0.5*alpha) * x[v] + (0.5*alpha) * mean(x[N(v)])

        This keeps the transform localized and smooth.
        """
        g = self.graph
        x = self.features.clone()

        # pick focus nodes
        picked = torch.randperm(self.num_nodes, device=self.device)[: self.num_focus_nodes]

        # precompute neighbor lists (on CPU tensors if needed)
        # we'll use undirected neighborhood by combining predecessors/successors
        def neighbors(nodes: Iterable[int]) -> torch.Tensor:
            cols = []
            for n in nodes:
                # concatenate in- and out-neighbors to emulate undirected
                nb = torch.unique(
                    torch.cat([g.successors(int(n)), g.predecessors(int(n))], dim=0)
                )
                if nb.numel() > 0:
                    cols.append(nb)
            if not cols:
                return torch.empty(0, dtype=torch.long, device=self.device)
            return torch.unique(torch.cat(cols))

        # 1) mix picked nodes with mean of their neighbors
        for u in picked.tolist():
            nb = neighbors([u])
            if nb.numel() == 0:
                continue
            mean_nb = self.features[nb].mean(dim=0)
            x[u] = (1.0 - self.alpha) * self.features[u] + self.alpha * mean_nb

        # 2) lightly mix 1-hop neighbors as well (half strength)
        one_hop = neighbors(picked.tolist())
        for v in one_hop.tolist():
            nb = neighbors([v])
            if nb.numel() == 0:
                continue
            mean_nb = self.features[nb].mean(dim=0)
            x[v] = (1.0 - 0.5 * self.alpha) * self.features[v] + (0.5 * self.alpha) * mean_nb

        return x, picked

    # --------------------------------------------------------------------- #
    # Training/Eval (same style as RandomWM)
    # --------------------------------------------------------------------- #
    def _train_and_eval_graphsage(self, use_transformed_features: bool) -> float:
        """
        Train a GraphSAGE for a few epochs and return test accuracy.
        Uses NeighborSampler + NodeCollator (same pattern as RandomWM).
        """
        model = GraphSAGE(
            in_channels=self.num_features,
            hidden_channels=128,
            out_channels=self.num_classes,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        sampler = NeighborSampler([5, 5])

        train_nids = self.train_mask.nonzero(as_tuple=True)[0].to(self.device)
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(self.device)

        train_collator = NodeCollator(self.graph, train_nids, sampler)
        test_collator = NodeCollator(self.graph, test_nids, sampler)

        train_loader = DataLoader(
            train_collator.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=train_collator.collate,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_collator.dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_collator.collate,
            drop_last=False,
        )

        best_acc = 0.0
        for _ in tqdm(range(1, 51),
                      desc=("GraphSAGE (defended)" if use_transformed_features else "GraphSAGE (baseline)")):
            # ---- Train
            model.train()
            for _, _, blocks in train_loader:
                blocks = [b.to(self.device) for b in blocks]
                feats = blocks[0].srcdata["feat"]
                labels = blocks[-1].dstdata["label"]

                optimizer.zero_grad()
                logits = model(blocks, feats)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

            # ---- Eval
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, _, blocks in test_loader:
                    blocks = [b.to(self.device) for b in blocks]
                    feats = blocks[0].srcdata["feat"]
                    labels = blocks[-1].dstdata["label"]
                    logits = model(blocks, feats)
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.numel()

            acc = correct / max(1, total)
            best_acc = max(best_acc, acc)

        return best_acc
