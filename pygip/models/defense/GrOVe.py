
import random
from time import time

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neural_network import MLPClassifier

from pygip.models.defense.base import BaseDefense
from pygip.models.nn import GCN
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric


class GroveDefense(BaseDefense):
    supported_api_types = {"dgl", "pyg"}

    def __init__(self, dataset, attack_node_fraction, model_path=None,
                 hidden_dim=256, verification_threshold=0.5, num_surrogate_models=3):
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

        # grove params
        self.hidden_dim = hidden_dim
        self.verification_threshold = verification_threshold
        self.num_surrogate_models = num_surrogate_models

        # models
        self.target_model = None
        self.surrogate_models = []
        self.independent_models = []
        self.csim_classifier = None

    def _load_model(self):
        """
        Load a pre-trained target model.
        """
        if self.model_path:
            # Create the model
            self.target_model = GCN(self.num_features, self.num_classes).to(self.device)
            # Load the saved state dict
            self.target_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            # Train new target model
            self._train_target_model()

        # Set to evaluation mode
        self.target_model.eval()

    def _train_target_model(self):
        """
        Train the target model.
        """
        self.target_model = GCN(self.num_features, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.target_model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(200):
            self.target_model.train()

            # Forward pass
            logits = self.target_model(self.graph_data, self.features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.target_model.eval()

    def _train_surrogate_models(self):
        """
        Train surrogate models to simulate model stealing attacks.
        """
        print("Training surrogate models...")

        for i in range(self.num_surrogate_models):
            # Create surrogate model with different initialization
            torch.manual_seed(42 + i)
            surrogate_model = GCN(self.num_features, self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=0.01, weight_decay=5e-4)

            # Train with limited data (simulating stolen model scenario)
            for epoch in range(100):
                surrogate_model.train()

                logits = surrogate_model(self.graph_data, self.features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            surrogate_model.eval()
            self.surrogate_models.append(surrogate_model)

    def _train_independent_models(self):
        """
        Train independent models for comparison.
        """
        print("Training independent models...")

        for i in range(self.num_surrogate_models):
            # Create independent model with different random seed
            torch.manual_seed(100 + i)
            independent_model = GCN(self.num_features, self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(independent_model.parameters(), lr=0.01, weight_decay=5e-4)

            # Train independently
            for epoch in range(150):
                independent_model.train()

                logits = independent_model(self.graph_data, self.features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            independent_model.eval()
            self.independent_models.append(independent_model)

    def _get_model_embeddings(self, model):
        """
        Extract embeddings from a model.
        """
        model.eval()
        with torch.no_grad():
            # Get intermediate layer embeddings (before final classification)
            embeddings = model.get_embeddings(self.graph_data, self.features)
            return embeddings[self.test_mask]

    def _compute_distance_vectors(self, target_embeddings, suspect_embeddings):
        """
        Compute element-wise squared distance vectors between embeddings.
        """
        target_np = target_embeddings.detach().cpu().numpy()
        suspect_np = suspect_embeddings.detach().cpu().numpy()

        # Ensure same shape
        min_nodes = min(target_np.shape[0], suspect_np.shape[0])
        target_crop = target_np[:min_nodes]
        suspect_crop = suspect_np[:min_nodes]

        # Compute element-wise squared distance
        distance_vectors = (target_crop - suspect_crop) ** 2
        return distance_vectors

    def _train_csim_classifier(self):
        """
        Train the CSim classifier for ownership verification.
        """
        # Get target model embeddings
        target_embeddings = self._get_model_embeddings(self.target_model)

        # Prepare training data
        distance_vectors = []
        labels = []

        # Positive samples: (target, surrogate) pairs - label 1
        for surrogate_model in self.surrogate_models:
            surrogate_embeddings = self._get_model_embeddings(surrogate_model)
            dist_vectors = self._compute_distance_vectors(target_embeddings, surrogate_embeddings)
            distance_vectors.extend(dist_vectors)
            labels.extend([1] * len(dist_vectors))

        # Negative samples: (target, independent) pairs - label 0
        for independent_model in self.independent_models:
            independent_embeddings = self._get_model_embeddings(independent_model)
            dist_vectors = self._compute_distance_vectors(target_embeddings, independent_embeddings)
            distance_vectors.extend(dist_vectors)
            labels.extend([0] * len(dist_vectors))

        X_train = np.array(distance_vectors)
        y_train = np.array(labels)

        # Train MLP classifier
        self.csim_classifier = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation='relu',
            random_state=42,
            max_iter=1000
        )
        self.csim_classifier.fit(X_train, y_train)

    def verify_ownership(self, suspect_model):
        """
        Verify if suspect model is a surrogate (stolen) or independent.
        """
        target_embeddings = self._get_model_embeddings(self.target_model)
        suspect_embeddings = self._get_model_embeddings(suspect_model)

        # Compute distance vectors
        distance_vectors = self._compute_distance_vectors(target_embeddings, suspect_embeddings)

        # Predict using CSim classifier
        predictions = self.csim_classifier.predict(distance_vectors)
        probabilities = self.csim_classifier.predict_proba(distance_vectors)

        # Aggregate results
        surrogate_probability = np.mean(probabilities[:, 1])
        is_surrogate = surrogate_probability > self.verification_threshold

        return {
            'is_surrogate': is_surrogate,
            'surrogate_probability': surrogate_probability,
            'confidence': abs(surrogate_probability - 0.5) * 2
        }

    def defend(self):
        """
        Execute the Grove ownership verification defense.
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================Grove Ownership Verification====================")

        # Load or train target model
        if not hasattr(self, 'target_model') or self.target_model is None:
            self._load_model()

        # Train surrogate and independent models
        self._train_surrogate_models()
        self._train_independent_models()

        # Train CSim classifier
        defense_s = time()
        self._train_csim_classifier()
        defense_e = time()

        # Verify models
        inference_s = time()
        verification_results = []

        # Test on surrogate models (should be detected as stolen)
        for i, surrogate_model in enumerate(self.surrogate_models):
            result = self.verify_ownership(surrogate_model)
            result['model_type'] = 'surrogate'
            result['expected_surrogate'] = True
            verification_results.append(result)

        # Test on independent models (should be detected as independent)
        for i, independent_model in enumerate(self.independent_models):
            result = self.verify_ownership(independent_model)
            result['model_type'] = 'independent'
            result['expected_surrogate'] = False
            verification_results.append(result)

        inference_e = time()

        # Calculate metrics
        correct_predictions = sum(
            1 for result in verification_results
            if result['is_surrogate'] == result['expected_surrogate']
        )
        total_predictions = len(verification_results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        metric = DefenseMetric()

        # Pred and label for overall performance
        dummy_preds = torch.tensor([1 if r['is_surrogate'] else 0 for r in verification_results])
        dummy_labels = torch.tensor([1 if r['expected_surrogate'] else 0 for r in verification_results])
        metric.update(dummy_preds, dummy_labels)

        wm_preds = torch.tensor([r['surrogate_probability'] for r in verification_results])
        wm_labels = torch.tensor([1.0 if r['expected_surrogate'] else 0.0 for r in verification_results])
        metric.update_wm(wm_preds, wm_labels)

        # Update computation metrics with recorded times
        metric_comp.update(defense_time=(defense_e - defense_s))
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        metric_comp.end()

        print("====================Final Results====================")
        print(f"Verification Accuracy: {accuracy:.4f}")
        print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"Defense Time: {defense_e - defense_s:.4f}s")
        print(f"Inference Time: {inference_e - inference_s:.4f}s")

        res = metric.compute()
        res['verification_accuracy'] = accuracy
        res['correct_predictions'] = correct_predictions
        res['total_predictions'] = total_predictions
        res_comp = metric_comp.compute()

        return res, res_comp
