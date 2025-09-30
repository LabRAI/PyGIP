import copy
import random
import time

import dgl
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, from_networkx

from .base import BaseDefense
from pygip.models.nn import GCN
from pygip.utils.metrics import DefenseCompMetric, DefenseMetric


class QueryBasedVerificationDefense(BaseDefense):
    supported_api_types = {"dgl"}
    supported_datasets = {}

    def __init__(self, dataset, defense_ratio=0.1, model_path=None):
        super().__init__(dataset, defense_ratio)
        self.model = None
        self.defense_ratio = defense_ratio
        self.graph_data = dataset.graph_data
        # compute related parameters
        self.k = max(1, int(dataset.num_nodes * defense_ratio))
        self.model_path = model_path

    def defend(self, fingerprint_mode='inductive', knowledge='full', attack_type='bitflip',
               k=5, num_trials=1, use_edge_perturbation=False, verbose=True, **kwargs):
        """
        Execute the query-based verification defense.
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================Query Based Verification Defense====================")

        # If model wasn't trained yet, train it
        if not hasattr(self, 'model_trained'):
            self.train_target_model(metric_comp)

        # Generate fingerprints and evaluate the defended model
        fingerprints = self._generate_fingerprints(
            self.model, mode=fingerprint_mode, knowledge=knowledge, k=k,
            perturb_fingerprints=use_edge_perturbation,
            perturb_budget=kwargs.get('perturb_budget', 5), **kwargs
        )

        preds = self.evaluate_model(self.model, self.dataset)
        inference_s = time.time()
        detection_results = self.verify_defense(self.model, fingerprints, attack_type, **kwargs)
        inference_e = time.time()

        # metric
        metric = DefenseMetric()
        labels = self.dataset.graph_data.ndata['label'][self.dataset.graph_data.ndata['test_mask']]
        metric.update(preds, labels)

        # Convert detection results to binary format
        detection_preds, detection_targets = self._convert_detection_to_binary(detection_results)
        metric.update_wm(detection_preds, detection_targets)
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        res_comp = metric_comp.compute()

        return res, res_comp

    def train_target_model(self, metric_comp: DefenseCompMetric):
        """Train the target model with defense mechanism."""
        defense_s = time.time()

        # Training and fingerprint generation (defense mechanism)
        self.model = self._train_target_model()
        self.model_trained = True

        defense_e = time.time()
        metric_comp.update(defense_time=(defense_e - defense_s))
        return self.model

    def evaluate_model(self, model, dataset):
        """Evaluate model performance on downstream task"""
        model.eval()
        features = self._get_features().to(self.device)
        labels = dataset.graph_data.ndata['label'].to(self.device)
        test_mask = dataset.graph_data.ndata['test_mask']

        with torch.no_grad():
            logits = model(dataset.graph_data.to(self.device), features)
            preds = logits.argmax(dim=1)[test_mask].cpu()
        return preds

    def verify_defense(self, model, fingerprints, attack_type, **kwargs):
        """Verify defense effectiveness by running attack and checking fingerprints"""
        # Run attack
        poisoned_model, attack_info = self._run_attack(
            model, attack_type=attack_type, **kwargs
        )

        # Evaluate fingerprints
        flipped_info = self._evaluate_fingerprints(poisoned_model, fingerprints)

        return {
            'flip_rate': flipped_info['flip_rate'],
            'flipped_fingerprints': flipped_info['flipped'],
            'total_fingerprints': len(fingerprints)
        }

    @staticmethod
    def _convert_detection_to_binary(detection_results):
        """Convert detection results to binary classification format"""
        total = detection_results['total_fingerprints']
        flipped = len(detection_results['flipped_fingerprints'])

        # Create binary predictions: 1 for attack detected, 0 for no attack
        detection_preds = torch.tensor([1 if flipped > 0 else 0])
        # In this case, we assume attack was actually performed, so target is 1
        detection_targets = torch.tensor([1])

        return detection_preds, detection_targets

    def _get_features(self):
        return self.graph_data.ndata['feat'] if hasattr(self.graph_data, 'ndata') else self.graph_data.x

    def _train_target_model(self, epochs=200):
        """
        Trains target GCN model according to protocol in
        Wu et al. (2023), Section 6.1 for graph node classification.

        Returns
        -------
        model : torch.nn.Module
            The trained GCN model.
        """
        model = GCN(
            feature_number=self.dataset.num_features,
            label_number=self.dataset.num_classes
        ).to(self.device)
        print(f"Training target model on device: {self.device} ...")

        optimizer = Adam(model.parameters(), lr=0.02)
        loss_fn = torch.nn.NLLLoss()

        features = self._get_features().to(self.device)
        labels = self.dataset.graph_data.ndata['label'].to(self.device)
        train_mask = self.dataset.graph_data.ndata['train_mask'].to(self.device)
        val_mask = getattr(self.dataset.graph_data.ndata, "val_mask", None)
        if val_mask is None:
            val_mask = self.dataset.graph_data.ndata['test_mask']
        val_mask = val_mask.to(self.device)

        for epoch in range(epochs):
            model.train()
            logits = model(self.graph_data.to(self.device), features)
            log_probs = F.log_softmax(logits, dim=1)
            loss = loss_fn(log_probs[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(self.graph_data.to(self.device), features)
                    val_log_probs = F.log_softmax(val_logits, dim=1)
                    val_pred = val_log_probs[val_mask].max(1)[1]
                    val_acc = (val_pred == labels[val_mask]).float().mean().item()
                    print(f"Epoch {epoch + 1}: Loss={loss.item():.4f} | Val Acc={val_acc:.4f}")

        return model

    def _load_model(self, model_path):
        model = GCN(
            in_feats=self.dataset.feature_number,
            hidden_feats=16,
            out_feats=self.dataset.label_number
        )
        model.load_state_dict(torch.load(model_path))
        return model

    def _generate_fingerprints(self, model, mode='transductive', knowledge='full', k=5, **kwargs):
        """
        Wrapper for fingerprint generation based on mode and knowledge level.
        Returns:
            List of fingerprints
        """
        if mode == 'transductive':
            generator = TransductiveFingerprintGenerator(
                model=model,
                dataset=self.dataset,
                candidate_fraction=kwargs.get('candidate_fraction', 1.0),
                random_seed=kwargs.get('random_seed', None),
                device=self.device,
                randomize=kwargs.get('randomize', True),
            )
            fingerprints = generator.generate_fingerprints(k=k, method=knowledge)

            unified_fingerprints = [(self.graph_data, node_id, label) for (node_id, label) in fingerprints]

        elif mode == 'inductive':
            generator = InductiveFingerprintGenerator(
                model=model,
                dataset=self.dataset,
                shadow_graph=self.dataset.graph_data,
                knowledge=knowledge,
                candidate_fraction=kwargs.get('candidate_fraction', 0.3),
                num_fingerprints=k,
                randomize=kwargs.get('randomize', True),
                random_seed=kwargs.get('random_seed', None),
                device=self.device,
                perturb_fingerprints=kwargs.get('perturb_fingerprints', False),
                perturb_budget=kwargs.get('perturb_budget', 5),
            )
            fingerprints = generator.generate_fingerprints(method=knowledge)
            if kwargs.get('perturb_fingerprints', False):
                for i, (graph, node_idx, label) in enumerate(fingerprints):
                    generator.shadow_graph = graph
                    generator.greedy_edge_perturbation(
                        node_idx=node_idx,
                        perturb_budget=kwargs.get('perturb_budget', 5),
                        knowledge=knowledge
                    )
                    fingerprints[i] = (generator.shadow_graph, node_idx, label)

            unified_fingerprints = fingerprints

        else:
            raise ValueError("Unknown fingerprinting mode. Use 'transductive' or 'inductive'.")

        return unified_fingerprints

    def _evaluate_fingerprints(self, model, fingerprints):
        """
        Checks if fingerprinted nodes have changed labels under the given model.

        Args:
            model: The model to evaluate.
            fingerprints: List of (graph, node_id, label) tuples.

        Returns:
            results: {
                'flipped': List[Tuple[node_id, old_label, new_label]],
                'flip_rate': float
            }
        """
        model.eval()
        flipped = []

        with torch.no_grad():
            for graph, node_id, expected_label in fingerprints:
                x = graph.ndata['feat'] if hasattr(graph, 'ndata') else graph.x
                logits = model(graph.to(self.device), x.to(self.device))
                pred = logits[node_id].argmax().item()
                if pred != expected_label:
                    flipped.append((node_id, expected_label, pred))

        return {
            'flipped': flipped,
            'flip_rate': len(flipped) / len(fingerprints) if fingerprints else 0.0
        }

    def _run_attack(self, model, attack_type='mettack', knowledge='full', **kwargs):
        """
        Run the specified attack on the model.
        Returns:
            poisoned_model: torch.nn.Module
            metadata: dict with info about the attack
        """
        if attack_type == 'bitflip':
            bit = kwargs.get('bit', 30)
            bfa_variant = kwargs.get('bfa_variant', 'BFA')
            attacker = BitFlipAttack(model, attack_type=bfa_variant, bit=bit)
            attack_info = attacker.apply()
            return model, attack_info

        elif attack_type == 'random':
            perturbed_graph = self._random_edge_addition_poisoning(
                node_fraction=kwargs.get('node_fraction', 0.1),
                edges_per_node=kwargs.get('edges_per_node', 5),
                random_seed=kwargs.get('random_seed', None),
            )
            poisoned_model = self._retrain_poisoned_model(
                poisoned_graph=perturbed_graph,
                epochs=kwargs.get('epochs', 200),
            )
            return poisoned_model, {'type': 'random_poison', 'graph': perturbed_graph}

        elif attack_type == 'mettack':
            num_edges = self.graph_data.num_edges()
            poison_frac = kwargs.get('poison_frac', 0.05)
            n_perturbations = int(poison_frac * num_edges)

            helper = MettackHelper(
                graph=self.graph_data,
                features=self._get_features(),
                labels=self.dataset.labels,
                train_mask=self.dataset.train_mask,
                val_mask=getattr(self.dataset, 'val_mask', None),
                test_mask=self.dataset.test_mask,
                n_perturbations=n_perturbations,
                device=self.device,
                max_perturbations=kwargs.get('max_perturbations', 50),
                surrogate_epochs=kwargs.get('surrogate_epochs', 30),
                candidate_sample_size=kwargs.get('candidate_sample_size', 20),
            )
            poisoned_graph, attack_metrics = helper.run()
            poisoned_model = self._retrain_poisoned_model(
                poisoned_graph=poisoned_graph,
                epochs=kwargs.get('epochs', 200),
            )
            return poisoned_model, {'type': 'mettack', 'metrics': attack_metrics, 'graph': poisoned_graph}

        else:
            raise ValueError(f"Unsupported attack_type: {attack_type}")

    def _random_edge_addition_poisoning(self, node_fraction=0.1, edges_per_node=5, random_seed=None):
        """
        Poison a fraction of nodes by adding random edges.

        Args:
            dataset: Dataset object (DGL-based)
            node_fraction: Fraction of nodes to poison (e.g., 0.1 = 10%)
            edges_per_node: Number of random edges to add per poisoned node
            random_seed: Optional seed

        Returns:
            poisoned_graph: DGLGraph
        """
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        poisoned_graph = copy.deepcopy(self.graph_data)
        num_nodes = poisoned_graph.num_nodes()
        num_poisoned_nodes = int(node_fraction * num_nodes)
        poisoned_nodes = random.sample(range(num_nodes), num_poisoned_nodes)

        new_edges = []

        for src in poisoned_nodes:
            for _ in range(edges_per_node):
                dst = random.randint(0, num_nodes - 1)
                if src != dst and \
                        not poisoned_graph.has_edges_between(src, dst) and \
                        not poisoned_graph.has_edges_between(dst, src):
                    new_edges.append((src, dst))
                    new_edges.append((dst, src))

        if new_edges:
            src, dst = zip(*new_edges)
            poisoned_graph.add_edges(src, dst)

        return poisoned_graph

    def _retrain_poisoned_model(self, poisoned_graph, epochs=200):
        """
        Retrain target GCN using the poisoned graph structure.

        Args:
            dataset: Original Dataset object (provides features, labels, masks)
            poisoned_graph: DGLGraph (with new random edges added)
            defense_class: The defense class to use for model training (e.g., QueryBasedVerificationDefense)
            device: 'cpu' or 'cuda'

        Returns:
            model: Trained GCN model
        """
        dataset_poisoned = copy.deepcopy(self.dataset)
        dataset_poisoned.graph_data = poisoned_graph

        defense = QueryBasedVerificationDefense(dataset=dataset_poisoned, defense_ratio=0.1)
        model = defense._train_target_model(epochs=epochs)
        return model

    def _evaluate_accuracy(self, model, dataset):
        """
        Evaluates test accuracy of the given model on the dataset.

        Args:
            model: Trained GCN model
            dataset: Dataset object (provides features, labels, test_mask, graph)

        Returns:
            accuracy: float (test accuracy, 0-1)
        """
        model.eval()
        features = self._get_features().to(self.device)
        labels = dataset.graph_data.ndata['label'].to(self.device)
        test_mask = dataset.graph_data.ndata['test_mask']

        with torch.no_grad():
            logits = model(dataset.graph_data.to(self.device), features)
            pred = logits.argmax(dim=1)
            correct = (pred[test_mask] == labels[test_mask]).float()
            accuracy = correct.sum().item() / test_mask.sum().item()
        return accuracy

    def run_full_pipeline(self, attack_type='random', mode='transductive', knowledge='full', k=5, trials=1, **kwargs):
        """
        Runs the full fingerprinting + attack + evaluation pipeline.

        Parameters:
            attack_type: 'random', 'bitflip', or 'mettack'
            mode: 'transductive' or 'inductive'
            knowledge: 'full' or 'limited'
            k: number of fingerprints
            trials: number of repeated trials
            kwargs: extra params for attack or fingerprinting

        Prints per-trial results and summary statistics.
        """
        flip_rates = []
        acc_drops = []

        for trial in range(trials):
            print(f"\n=== Trial {trial + 1}/{trials} ===")

            model_clean = self._train_target_model()
            acc_clean = self._evaluate_accuracy(model_clean, self.dataset)
            print(f"Clean model accuracy: {acc_clean:.4f}")

            fingerprints = self._generate_fingerprints(model_clean, mode=mode, knowledge=knowledge, k=k, **kwargs)

            model_poisoned, attack_meta = self._run_attack(model_clean, attack_type=attack_type, knowledge=knowledge,
                                                           **kwargs)
            acc_poisoned = self._evaluate_accuracy(model_poisoned, self.dataset)
            print(f"Poisoned model accuracy: {acc_poisoned:.4f}")

            eval_result = self._evaluate_fingerprints(model_poisoned, fingerprints)
            flip_rate = eval_result['flip_rate']
            print(f"Fingerprint flip rate: {flip_rate:.4f}")
            for (nid, old, new) in eval_result['flipped']:
                print(f"  Node {nid}: {old} → {new}")

            flip_rates.append(flip_rate)
            acc_drops.append(acc_clean - acc_poisoned)

        print("\n=== Summary ===")
        print(f"Avg Accuracy Drop: {np.mean(acc_drops):.4f}")
        print(f"Avg Fingerprint Flip Rate: {np.mean(flip_rates):.4f}")


class TransductiveFingerprintGenerator:
    def __init__(self, model, dataset, candidate_fraction=0.3, random_seed=None, device='cpu', randomize=True):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.graph_data = dataset.graph_data
        self.candidate_fraction = candidate_fraction
        self.random_seed = random_seed
        self.randomize = randomize

    def _get_features(self):
        """Backend-agnostic feature getter (DGL or PyG)."""
        return self.graph_data.ndata['feat'] if hasattr(self.graph_data, 'ndata') else self.graph_data.x

    def get_candidate_nodes(self):
        """Randomly sample a subset of nodes as candidates."""
        all_nodes = torch.arange(self.graph_data.num_nodes())
        num_candidates = max(1, int(len(all_nodes) * self.candidate_fraction))

        if self.randomize and self.candidate_fraction < 1.0:
            generator = torch.Generator(device=self.device)
            if self.random_seed is not None:
                generator.manual_seed(self.random_seed)
            idx = torch.randperm(len(all_nodes), generator=generator)[:num_candidates]
            return all_nodes[idx]
        return all_nodes

    def compute_fingerprint_scores_full(self, candidate_nodes):
        """Full-knowledge fingerprint scores (gradient-based)."""
        self.model.eval()
        scores = []
        x = self._get_features().to(self.device)
        logits = self.model(self.graph_data.to(self.device), x)

        for node in candidate_nodes:
            self.model.zero_grad()
            logit = logits[node]
            label = logit.argmax().item()
            loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([label], device=self.device))
            loss.backward(retain_graph=True)
            grad_norm = sum((p.grad ** 2).sum().item() for p in self.model.parameters() if p.grad is not None)
            scores.append(grad_norm)

        return torch.tensor(scores, device=self.device)

    def compute_fingerprint_scores_limited(self, candidate_nodes):
        """Limited-knowledge fingerprint scores (confidence margin)."""
        self.model.eval()
        x = self._get_features().to(self.device)
        with torch.no_grad():
            logits = self.model(self.graph_data.to(self.device), x)
            probs = F.softmax(logits, dim=1)
            labels = probs.argmax(dim=1)
            scores = 1.0 - probs[candidate_nodes, labels[candidate_nodes]]
        return scores

    def select_top_fingerprints(self, scores, candidate_nodes, k, method='full'):
        """Selects top-k fingerprint nodes after filtering out extreme score outliers."""
        q = 0.99 if method == 'full' else 1.0
        threshold = torch.quantile(scores, q)
        mask = scores <= threshold

        filtered_scores = scores[mask]
        filtered_candidates = candidate_nodes[mask]

        if filtered_scores.size(0) < k:
            k = filtered_scores.size(0)

        topk = torch.topk(filtered_scores, k)
        return filtered_candidates[topk.indices], topk.values

    def generate_fingerprints(self, k=5, method='full'):
        candidate_nodes = self.get_candidate_nodes().to(self.device)
        x = self._get_features().to(self.device)

        with torch.no_grad():
            logits = self.model(self.graph_data.to(self.device), x)
            labels = logits.argmax(dim=1)

        if method == 'full':
            scores = self.compute_fingerprint_scores_full(candidate_nodes)
        elif method == 'limited':
            scores = self.compute_fingerprint_scores_limited(candidate_nodes)
        else:
            raise ValueError("method must be 'full' or 'limited'")

        class_to_candidates = {}
        for i, node in enumerate(candidate_nodes):
            cls = int(labels[node])
            class_to_candidates.setdefault(cls, []).append((node.item(), scores[i].item()))

        rng = random.Random(self.random_seed)
        class_list = list(class_to_candidates.keys())
        rng.shuffle(class_list)

        fingerprints = []
        for cls in class_list:
            class_nodes = sorted(class_to_candidates[cls], key=lambda x: x[1], reverse=True)
            top_node = class_nodes[0][0]
            fingerprints.append((top_node, cls))
            if len(fingerprints) >= k:
                break

        if len(fingerprints) < k:
            fingerprint_nodes, _ = self.select_top_fingerprints(scores, candidate_nodes, k, method=method)
            fingerprints = [(int(n), int(labels[n])) for n in fingerprint_nodes]

        return fingerprints


class InductiveFingerprintGenerator:
    def __init__(self, model, dataset, shadow_graph=None, knowledge='limited',
                 candidate_fraction=0.3, num_fingerprints=5,
                 randomize=True, random_seed=None, device='cpu',
                 perturb_fingerprints=False, perturb_budget=5):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.shadow_graph = shadow_graph if shadow_graph is not None else dataset.graph_data
        self.knowledge = knowledge
        self.candidate_fraction = candidate_fraction
        self.num_fingerprints = num_fingerprints
        self.randomize = randomize
        self.random_seed = random_seed
        self.perturb_fingerprints = perturb_fingerprints
        self.perturb_budget = perturb_budget

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)

    def _get_features(self):
        return self.shadow_graph.ndata['feat'] if hasattr(self.shadow_graph, 'ndata') else self.shadow_graph.x

    def get_candidate_nodes(self):
        all_nodes = torch.arange(self.shadow_graph.num_nodes())
        num_candidates = max(1, int(len(all_nodes) * self.candidate_fraction))

        if self.randomize and self.candidate_fraction < 1.0:
            generator = torch.Generator(device=self.device)
            if self.random_seed is not None:
                generator.manual_seed(self.random_seed)
            idx = torch.randperm(len(all_nodes), generator=generator)[:num_candidates]
            candidates = all_nodes[idx]
        else:
            candidates = all_nodes

        return candidates

    def compute_fingerprint_score(self, node_idx, graph_override=None):
        """
        Computes the fingerprint score for a given node according to knowledge mode.
        If graph_override is provided, scoring is done on that graph instead of shadow_graph.
        """
        graph = graph_override if graph_override is not None else self.shadow_graph
        x = (graph.ndata['feat'] if hasattr(graph, 'ndata') else graph.x).to(self.device)
        self.model.eval()

        if self.knowledge == 'limited':
            with torch.no_grad():
                logits = self.model(graph.to(self.device), x)
                probs = torch.softmax(logits[node_idx], dim=0)
                pred_class = probs.argmax().item()
                return 1 - probs[pred_class].item()

        elif self.knowledge == 'full':
            x.requires_grad_(True)
            logits = self.model(graph.to(self.device), x)
            pred = logits[node_idx]
            label = pred.argmax().item()

            self.model.zero_grad()
            loss = torch.nn.functional.nll_loss(
                torch.log_softmax(pred.unsqueeze(0), dim=1),
                torch.tensor([label], device=self.device)
            )
            loss.backward(retain_graph=True)

            grad = x.grad[node_idx]
            grad_norm_sq = (grad ** 2).sum().item()
            x.requires_grad_(False)
            x.grad = None
            return grad_norm_sq
        else:
            raise ValueError("knowledge must be 'limited' or 'full'")

    def generate_fingerprint_nodes(self):
        """
        Step 3: Identifies and returns the top-k (num_fingerprints) nodes with the highest
        fingerprint scores from the candidate set. (Section 4.2.2)

        Returns:
            List[int]: Indices of selected fingerprint nodes.
        """
        candidates = self.get_candidate_nodes()
        scores = []
        for idx in candidates:
            score = self.compute_fingerprint_score(idx)
            scores.append((score, int(idx)))

        scores.sort(reverse=True)
        selected = [idx for (_, idx) in scores[:self.num_fingerprints]]
        return selected

    def save_fingerprint_tuples(self, node_indices):
        self.model.eval()
        x = self._get_features().to(self.device)
        with torch.no_grad():
            logits = self.model(self.shadow_graph.to(self.device), x)
            labels = logits.argmax(dim=1).cpu().numpy()
            return [(self.shadow_graph, int(idx), int(labels[idx])) for idx in node_indices]

    def generate_fingerprints(self, method='full'):
        """
        Generate inductive fingerprints for model watermarking.

        Parameters:
            method (str): 'full' for gradient-based or 'limited' for output-based

        Returns:
            List of fingerprints
        """
        if method == 'full':
            return self._generate_full()
        elif method == 'limited':
            return self._generate_limited()
        else:
            raise ValueError(f"Invalid fingerprinting method: '{method}'")

    def _generate_full(self):
        """
        Implements full knowledge fingerprint generation (gradient-based).
        Based on Section 4.2.1 and 5.2 of Wu et al. (2023).
        """
        self.knowledge = 'full'
        print("[Fingerprint] Generating FULL knowledge fingerprints...")
        fingerprint_nodes = self.generate_fingerprint_nodes()

        if self.perturb_fingerprints:
            print("[Fingerprint] Applying greedy feature perturbation (FULL)...")
            self.greedy_perturb_fingerprints(fingerprint_nodes)

        return self.save_fingerprint_tuples(fingerprint_nodes)

    def _generate_limited(self):
        """
        Implements limited knowledge fingerprint generation (output-based).
        Based on Section 4.2.2 and 5.2 of Wu et al. (2023).
        """
        self.knowledge = 'limited'
        print("[Fingerprint] Generating LIMITED knowledge fingerprints...")
        fingerprint_nodes = self.generate_fingerprint_nodes()

        if self.perturb_fingerprints:
            print("[Fingerprint] Applying greedy feature perturbation (LIMITED)...")
            self.greedy_perturb_fingerprints(fingerprint_nodes)

        return self.save_fingerprint_tuples(fingerprint_nodes)

    def greedy_perturb_fingerprints(self, node_indices):
        """
        Greedily perturbs each fingerprint node's features (not edges) to increase its
        fingerprint score, without changing the predicted label.

        - For each node, for each feature dimension:
            - Add or subtract a small epsilon.
            - Accept change if predicted label stays the same and fingerprint score increases.
            - Stop after perturb_budget attempts or no improvement.

        Returns:
            List[int]: Indices of perturbed fingerprint nodes (features in shadow_graph are updated in-place).
        """
        epsilon = 0.01
        features = self._get_features().clone().detach().to(self.device)
        self.shadow_graph = self.shadow_graph.to(self.device)

        for idx in node_indices:
            num_tries = 0
            improved = True
            while num_tries < self.perturb_budget and improved:
                improved = False
                current_score = self.compute_fingerprint_score(idx, graph_override=self.shadow_graph)

                self.model.eval()
                with torch.no_grad():
                    logits = self.model(self.shadow_graph, features)
                    pred_label = logits[idx].argmax().item()

                original_features = features[idx].clone()
                for dim in range(features.shape[1]):
                    for direction in [+1, -1]:
                        features[idx][dim] += direction * epsilon

                        self.model.eval()
                        with torch.no_grad():
                            logits_new = self.model(self.shadow_graph, features)
                            new_pred_label = logits_new[idx].argmax().item()
                        new_score = self.compute_fingerprint_score(idx, graph_override=self.shadow_graph)

                        if new_pred_label == pred_label and new_score > current_score:
                            current_score = new_score
                            improved = True
                            num_tries += 1
                        else:
                            features[idx][dim] = original_features[dim]

                        if num_tries >= self.perturb_budget:
                            break
                    if num_tries >= self.perturb_budget:
                        break

        if hasattr(self.shadow_graph, 'ndata'):
            self.shadow_graph.ndata['feat'] = features
        else:
            self.shadow_graph.x = features
        return node_indices

    def greedy_edge_perturbation(self, node_idx, perturb_budget=5, knowledge='full'):
        """
        Dispatch to greedy edge perturbation strategy based on verifier knowledge level.

        Args:
            node_idx (int): Fingerprint node index.
            perturb_budget (int): Number of edge perturbations allowed.
            knowledge (str): 'full' or 'limited'
        """
        if knowledge == 'full':
            self._greedy_edge_perturbation_f(node_idx, perturb_budget)
        elif knowledge == 'limited':
            self._greedy_edge_perturbation_l(node_idx, perturb_budget)
        else:
            raise ValueError("knowledge must be 'full' or 'limited'")

    def _greedy_edge_perturbation_f(self, node_idx, perturb_budget):
        """
        Full knowledge edge perturbation (Inductive-F).
        Increases fingerprint score using model gradients while preserving prediction.
        """

        g_nx = to_networkx(self.shadow_graph.to('cpu'), to_undirected=True)
        x = self._get_features().to(self.device)
        self.model.eval()

        with torch.no_grad():
            original_pred = self.model(self.shadow_graph.to(self.device), x)[node_idx].argmax().item()

        def score_fn(modified_graph):
            return self.compute_fingerprint_score(node_idx, graph_override=modified_graph)

        neighbors = list(g_nx.neighbors(node_idx))
        non_neighbors = list(set(range(self.shadow_graph.num_nodes())) - set(neighbors) - {node_idx})

        applied = 0
        while applied < perturb_budget:
            best_delta = 0
            best_graph = None
            best_action = None

            for nbr in non_neighbors:
                temp_g = copy.deepcopy(g_nx)
                temp_g.add_edge(node_idx, nbr)
                g_temp = from_networkx(temp_g).to(self.device)
                with torch.no_grad():
                    pred = self.model(g_temp, x)[node_idx].argmax().item()
                if pred != original_pred:
                    continue
                delta = score_fn(g_temp) - score_fn(self.shadow_graph)
                if delta > best_delta:
                    best_delta = delta
                    best_graph = g_temp
                    best_action = ('add', nbr)

            for nbr in neighbors:
                temp_g = copy.deepcopy(g_nx)
                if temp_g.has_edge(node_idx, nbr):
                    temp_g.remove_edge(node_idx, nbr)
                    g_temp = from_networkx(temp_g).to(self.device)
                    with torch.no_grad():
                        pred = self.model(g_temp, x)[node_idx].argmax().item()
                    if pred != original_pred:
                        continue
                    delta = score_fn(g_temp) - score_fn(self.shadow_graph)
                    if delta > best_delta:
                        best_delta = delta
                        best_graph = g_temp
                        best_action = ('remove', nbr)

            if best_graph is None:
                break
            self.shadow_graph = best_graph
            g_nx = to_networkx(best_graph.to('cpu'), to_undirected=True)

            if best_action[0] == 'add':
                non_neighbors.remove(best_action[1])
                neighbors.append(best_action[1])
            else:
                neighbors.remove(best_action[1])
                non_neighbors.append(best_action[1])

            applied += 1

    def _greedy_edge_perturbation_l(self, node_idx, perturb_budget):
        """
        Limited knowledge edge perturbation (Inductive-L).
        Uses confidence margin (1 - confidence) as proxy for fingerprint sensitivity.
        """

        g_nx = to_networkx(self.shadow_graph.to('cpu'), to_undirected=True)
        x = self._get_features().to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(self.shadow_graph.to(self.device), x)
            original_pred = logits[node_idx].argmax().item()
            original_conf = F.softmax(logits[node_idx], dim=0)[original_pred].item()
            original_score = 1 - original_conf

        def score_fn(modified_graph):
            with torch.no_grad():
                logits = self.model(modified_graph.to(self.device), x)
                pred = logits[node_idx].argmax().item()
                if pred != original_pred:
                    return -1
                conf = F.softmax(logits[node_idx], dim=0)[pred].item()
                return 1 - conf

        neighbors = list(g_nx.neighbors(node_idx))
        non_neighbors = list(set(range(self.shadow_graph.num_nodes())) - set(neighbors) - {node_idx})

        applied = 0
        while applied < perturb_budget:
            best_delta = 0
            best_graph = None
            best_action = None

            for nbr in non_neighbors:
                temp_g = copy.deepcopy(g_nx)
                temp_g.add_edge(node_idx, nbr)
                g_temp = from_networkx(temp_g).to(self.device)
                new_score = score_fn(g_temp)
                delta = new_score - original_score
                if new_score >= 0 and delta > best_delta:
                    best_delta = delta
                    best_graph = g_temp
                    best_action = ('add', nbr)

            for nbr in neighbors:
                temp_g = copy.deepcopy(g_nx)
                if temp_g.has_edge(node_idx, nbr):
                    temp_g.remove_edge(node_idx, nbr)
                    g_temp = from_networkx(temp_g).to(self.device)
                    new_score = score_fn(g_temp)
                    delta = new_score - original_score
                    if new_score >= 0 and delta > best_delta:
                        best_delta = delta
                        best_graph = g_temp
                        best_action = ('remove', nbr)

            if best_graph is None:
                break
            self.shadow_graph = best_graph
            g_nx = to_networkx(best_graph.to('cpu'), to_undirected=True)

            if best_action[0] == 'add':
                non_neighbors.remove(best_action[1])
                neighbors.append(best_action[1])
            else:
                neighbors.remove(best_action[1])
                non_neighbors.append(best_action[1])

            applied += 1


class BitFlipAttack:
    def __init__(self, model, attack_type='random', bit=0):
        self.model = model
        self.attack_type = attack_type
        self.bit = bit

    def _get_target_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad and p.numel() > 0]
        if self.attack_type in ['random', 'BFA']:
            return params
        elif self.attack_type == 'BFA-F':
            return [params[0]]
        elif self.attack_type == 'BFA-L':
            return [params[-1]]
        else:
            raise ValueError(f"Unknown attack_type {self.attack_type}")

    def _true_bit_flip(self, tensor, index=None, bit=0):
        a = tensor.detach().cpu().numpy().copy()
        flat = a.ravel()
        if index is None:
            index = np.random.randint(0, flat.size)
        old_val = flat[index]
        int_view = np.frombuffer(flat[index].tobytes(), dtype=np.uint32)[0]
        int_view ^= (1 << bit)
        new_val = np.frombuffer(np.uint32(int_view).tobytes(), dtype=np.float32)[0]
        flat[index] = new_val
        a = flat.reshape(a.shape)
        tensor.data = torch.from_numpy(a).to(tensor.device)
        return old_val, new_val, index

    def apply(self):
        params = self._get_target_params()
        with torch.no_grad():
            layer_idx = random.randrange(len(params))
            param = params[layer_idx]
            idx = random.randrange(param.numel())
            old_val, new_val, actual_idx = self._true_bit_flip(param, index=idx, bit=self.bit)
        return {
            'layer': layer_idx,
            'param_idx': actual_idx,
            'old_val': old_val,
            'new_val': new_val,
            'bit': self.bit,
            'attack_type': self.attack_type
        }


class MettackHelper:
    def __init__(self, graph, features, labels, train_mask, val_mask, test_mask,
                 n_perturbations=5, device='cpu', max_perturbations=50,
                 surrogate_epochs=30, candidate_sample_size=20):
        self.device = device
        self.graph = dgl.add_self_loop(graph).to(self.device)
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.surrogate_epochs = surrogate_epochs
        self.candidate_sample_size = candidate_sample_size
        if val_mask is not None:
            self.val_mask = val_mask.to(self.device)
        else:
            self.val_mask = self._create_val_mask_from_train(train_mask).to(self.device)

        self.test_mask = test_mask.to(self.device)

        self.n_perturbations = n_perturbations

        in_feats = features.shape[1]
        n_classes = int(labels.max().item()) + 1
        self.surrogate = GCN(in_feats, n_classes).to(self.device)

        torch.manual_seed(42)
        np.random.seed(42)

        self.modified_edges = set()

        original_graph_no_self_loop = dgl.remove_self_loop(graph)
        self.original_edges = set(zip(original_graph_no_self_loop.edges()[0].cpu().numpy(),
                                      original_graph_no_self_loop.edges()[1].cpu().numpy()))

        self.candidate_edges = self._get_candidate_edges()

    def _create_val_mask_from_train(self, train_mask):
        """
        Create a validation mask by taking a subset of training nodes.
        This is needed when the dataset doesn't provide a validation mask.
        """
        train_indices = torch.where(train_mask)[0]
        n_val = min(500, len(train_indices) // 4)

        perm = torch.randperm(len(train_indices))
        val_indices = train_indices[perm[:n_val]]

        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        val_mask[val_indices] = True

        self.train_mask = train_mask.clone()
        self.train_mask[val_indices] = False

        return val_mask

    def run(self):
        """
        Main entrypoint to run the Mettack algorithm.
        Returns:
            poisoned_graph (DGLGraph): The perturbed graph with edges changed.
            metrics (dict): Metrics for before/after attack, for evaluation.
        """
        print("Starting Mettack attack...")

        print("Training surrogate model...")
        self._train_surrogate()

        print("Applying structure attack...")
        poisoned_graph = self._apply_structure_attack()

        print("Evaluating attack results...")
        metrics = self._evaluate(poisoned_graph)

        return poisoned_graph, metrics

    def _train_surrogate(self):
        """
        Trains a surrogate GCN on the clean graph.
        (Matches Wu et al., Section 6.1)
        """
        optimizer = optim.Adam(self.surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        self.surrogate.train()

        for epoch in range(self.surrogate_epochs):
            optimizer.zero_grad()
            logits = self.surrogate(self.graph, self.features)
            loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                self.surrogate.eval()
                with torch.no_grad():
                    val_logits = self.surrogate(self.graph, self.features)
                    val_acc = self._compute_accuracy(val_logits[self.val_mask],
                                                     self.labels[self.val_mask])
                    print(f"Surrogate epoch {epoch}: Val Acc = {val_acc:.4f}")
                self.surrogate.train()

    def _apply_structure_attack(self):
        """
        Runs the Mettack structure perturbation loop (bi-level optimization).
        - At each step, modify the adjacency matrix (add/remove an edge).
        - Select the perturbation that maximizes surrogate model loss on the validation nodes.
        - Repeat up to n_perturbations times.
        Returns a new DGLGraph with edges modified.
        (See Appendix A.2 in Wu et al.)
        """
        current_graph = copy.deepcopy(self.graph)
        perturbed_edges = set()

        for step in range(self.n_perturbations):
            print(f"Perturbation step {step + 1}/{self.n_perturbations}")

            best_edge = None
            best_loss = -float('inf')
            best_action = None

            candidate_sample = np.random.choice(len(self.candidate_edges),
                                                min(self.candidate_sample_size, len(self.candidate_edges)),
                                                replace=False)

            for idx in tqdm(candidate_sample, desc="Evaluating candidates"):
                edge = self.candidate_edges[idx]

                if edge in perturbed_edges or (edge[1], edge[0]) in perturbed_edges:
                    continue

                for action in ['add', 'remove']:
                    if action == 'add' and edge in self.original_edges:
                        continue
                    if action == 'remove' and edge not in self.original_edges:
                        continue

                    temp_graph = self._apply_single_perturbation(current_graph, edge, action)

                    attack_loss = self._compute_attack_loss(temp_graph)

                    if attack_loss > best_loss:
                        best_loss = attack_loss
                        best_edge = edge
                        best_action = action

            if best_edge is not None:
                current_graph = self._apply_single_perturbation(current_graph, best_edge, best_action)
                perturbed_edges.add(best_edge)
                self.modified_edges.add((best_edge, best_action))
                print(f"Applied {best_action} edge {best_edge} with loss increase: {best_loss:.4f}")
            else:
                print("No beneficial perturbation found, stopping early.")
                break

        return current_graph

    def _get_candidate_edges(self):
        """
        Generate candidate edges for perturbation.
        Includes both existing edges (for removal) and non-existing edges (for addition).
        """
        n_nodes = self.graph.num_nodes()

        all_possible_edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                all_possible_edges.append((i, j))

        return all_possible_edges[:min(10000, len(all_possible_edges))]

    def _apply_single_perturbation(self, graph, edge, action):
        """
        Apply a single edge perturbation (add or remove) to the graph.
        """
        temp_graph = copy.deepcopy(graph)

        if action == 'add':
            temp_graph.add_edges([edge[0], edge[1]], [edge[1], edge[0]])
        elif action == 'remove':
            src, dst = temp_graph.edges()
            edge_ids = []

            for i, (s, d) in enumerate(zip(src.cpu().numpy(), dst.cpu().numpy())):
                if (s == edge[0] and d == edge[1]) or (s == edge[1] and d == edge[0]):
                    edge_ids.append(i)

            if edge_ids:
                temp_graph.remove_edges(edge_ids)

        temp_graph = dgl.add_self_loop(temp_graph)

        return temp_graph

    def _compute_attack_loss(self, perturbed_graph):
        """
        Compute the attack loss on a perturbed graph.
        This measures how much the surrogate model's performance degrades.
        Uses proper bi-level optimization as in the original Mettack paper.
        """

        temp_surrogate = copy.deepcopy(self.surrogate)
        temp_surrogate.train()

        optimizer = optim.Adam(temp_surrogate.parameters(), lr=0.01)

        for _ in range(5):
            optimizer.zero_grad()
            logits = temp_surrogate(perturbed_graph, self.features)
            loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            optimizer.step()

        temp_surrogate.eval()
        with torch.no_grad():
            val_logits = temp_surrogate(perturbed_graph, self.features)
            val_loss = F.cross_entropy(val_logits[self.val_mask], self.labels[self.val_mask])

        return val_loss.item()

    def _evaluate(self, poisoned_graph):
        """
        Evaluates GCN accuracy before/after poisoning, etc.
        """
        metrics = {}

        self.surrogate.eval()
        with torch.no_grad():
            clean_logits = self.surrogate(self.graph, self.features)
            clean_acc = self._compute_accuracy(clean_logits[self.test_mask],
                                               self.labels[self.test_mask])
            metrics['clean_test_acc'] = clean_acc

        poisoned_model = GCN(self.features.shape[1],
                             int(self.labels.max().item()) + 1).to(self.device)
        optimizer = optim.Adam(poisoned_model.parameters(), lr=0.01, weight_decay=5e-4)

        poisoned_model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            logits = poisoned_model(poisoned_graph, self.features)
            loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            optimizer.step()

        poisoned_model.eval()
        with torch.no_grad():
            poisoned_logits = poisoned_model(poisoned_graph, self.features)
            poisoned_acc = self._compute_accuracy(poisoned_logits[self.test_mask],
                                                  self.labels[self.test_mask])
            metrics['poisoned_test_acc'] = poisoned_acc

        metrics['accuracy_drop'] = clean_acc - poisoned_acc
        metrics['num_perturbations'] = len(self.modified_edges)

        return metrics

    def _compute_accuracy(self, logits, labels):
        """Helper function to compute accuracy."""
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        return correct / len(labels)
