from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MetricBase(ABC):
    def __init__(self):
        self.preds = []
        self.labels = []

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update internal metric state."""
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Compute and return all metric results."""
        pass

    def reset(self) -> None:
        """Reset internal state."""
        self.preds = []
        self.labels = []

    @staticmethod
    def _cat_to_numpy(a: List) -> np.ndarray:
        if len(a) == 0:
            raise ValueError("Empty tensor list, nothing to compute.")
        return torch.cat(a).cpu().numpy()

    def compute_default_metrics(self, preds, labels) -> Dict[str, float]:
        preds = self._cat_to_numpy(preds)
        labels = self._cat_to_numpy(labels)
        return {
            'Acc': accuracy_score(labels, preds),
            'F1': f1_score(labels, preds, average='macro'),
            'Precision': precision_score(labels, preds, average='macro'),
            'Recall': recall_score(labels, preds, average='macro'),
        }

    def __repr__(self):
        results = self.compute()
        for name, value in results.items():
            print(f"{name}: {value:.4f}")


class AttackMetric(MetricBase):
    def __init__(self):
        super().__init__()
        self.query_label = []
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.query_label = []

    def update(self, preds, labels, query_label):
        self.preds.append(preds.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.query_label.append(query_label.detach().cpu())

    def compute_fidelity(self, preds_label, query_label) -> Dict[str, float]:
        preds_label = self._cat_to_numpy(preds_label)
        query_label = self._cat_to_numpy(query_label)
        return {
            'Fidelity': (preds_label == query_label).astype(float).mean().item()
        }

    def compute(self):
        defaults = self.compute_default_metrics(self.preds, self.labels)
        fidelity = self.compute_fidelity(self.preds, self.query_label)
        results = defaults | fidelity
        print(f"acc: {results['Acc']:.4f}, fidelity: {results['Fidelity']:.4f}")
        return results


class DefenseMetric(MetricBase):
    def __init__(self):
        super().__init__()
        self.wm_preds = []
        self.wm_label = []
        self.reset()

    def update(self, preds, labels):
        self.preds.append(preds.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def reset(self) -> None:
        super().reset()
        self.wm_preds = []
        self.wm_label = []

    def update_wm(self, wm_preds, wm_label):
        self.wm_preds.append(wm_preds.detach().cpu())
        self.wm_label.append(wm_label.detach().cpu())

    def compute_wm(self):
        wm_preds = self._cat_to_numpy(self.wm_preds)
        wm_label = self._cat_to_numpy(self.wm_label)
        return {"WM Acc": accuracy_score(wm_label, wm_preds)}

    def compute(self):
        defaults = self.compute_default_metrics(self.preds, self.labels)
        wm_acc = self.compute_wm()
        results = defaults | wm_acc
        print(f"acc: {results['Acc']:.4f}, wm acc: {results['WM Acc']:.4f}")
        return results


import torch
import numpy as np
import time


class AttackCompMetric:
    def __init__(self, gpu_count=None):
        self.train_target_time = []
        self.query_target_time = []
        self.train_surrogate_time = []
        self.inference_surrogate_time = []
        self.attack_time = []

        self.start_time = 0
        self.total_time = 0

        self.gpu_count = gpu_count or (1 if torch.cuda.is_available() else 0)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.total_time = time.time() - self.start_time

    def update(self, train_target_time=None, query_target_time=None, train_surrogate_time=None, attack_time=None,
               inference_surrogate_time=None):
        if train_target_time is not None:
            self.train_target_time.append(train_target_time)
        if query_target_time is not None:
            self.query_target_time.append(query_target_time)
        if train_surrogate_time is not None:
            self.train_surrogate_time.append(train_surrogate_time)
        if attack_time is not None:
            self.attack_time.append(attack_time)
        if inference_surrogate_time is not None:
            self.inference_surrogate_time.append(inference_surrogate_time)

    def compute(self):
        peak_mem = 0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        gpu_hours = (self.total_time / 3600.0) * self.gpu_count

        print(
            f"attack time: {np.mean(self.attack_time):.4f}, inference time: {np.mean(self.inference_surrogate_time):.4f}, gpu mem: {peak_mem:.4f}, gpu hours: {gpu_hours:.4f}")

        return {
            'train_target_time': np.mean(self.train_target_time),
            'query_target_time': np.mean(self.query_target_time),
            'train_surrogate_time': np.mean(self.train_surrogate_time),
            'attack_time': np.mean(self.attack_time),
            'inference_surrogate_time': np.mean(self.inference_surrogate_time),
            'total_time': self.total_time,
            'peak_gpu_mem(GB)': peak_mem,
            'gpu_hours': gpu_hours
        }


class DefenseCompMetric:
    def __init__(self, gpu_count=None):
        self.train_target_time = []
        self.train_defense_time = []
        self.inference_defense_time = []
        self.defense_time = []

        self.start_time = 0
        self.total_time = 0

        self.gpu_count = gpu_count or (1 if torch.cuda.is_available() else 0)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.total_time = time.time() - self.start_time

    def update(self, train_target_time=None, train_defense_time=None, inference_defense_time=None, defense_time=None):
        if train_target_time is not None:
            self.train_target_time.append(train_target_time)
        if train_defense_time is not None:
            self.train_defense_time.append(train_defense_time)
        if inference_defense_time is not None:
            self.inference_defense_time.append(inference_defense_time)
        if defense_time is not None:
            self.defense_time.append(defense_time)

    def compute(self):
        peak_mem = 0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        gpu_hours = (self.total_time / 3600.0) * self.gpu_count

        print(
            f"defense time: {np.mean(self.defense_time):.4f}, inference time: {np.mean(self.inference_defense_time):.4f}, gpu mem: {peak_mem:.4f}, gpu hours: {gpu_hours:.4f}")

        return {
            'train_target_time': np.mean(self.train_target_time),
            'train_defense_time': np.mean(self.train_defense_time),
            'inference_defense_time': np.mean(self.inference_defense_time),
            'defense_time': np.mean(self.defense_time),
            'total_time': self.total_time,
            'peak_gpu_mem(GB)': peak_mem,
            'gpu_hours': gpu_hours
        }


class GraphNeuralNetworkMetric:
    """
    Graph Neural Network Metric Class.

    This class evaluates two metrics, fidelity and accuracy, for a given
    GNN model on a specified graph and features.
    """

    def __init__(self, fidelity=0, accuracy=0, model=None,
                 graph=None, features=None, mask=None,
                 labels=None, query_labels=None):
        self.model = model if model is not None else None
        self.graph = graph if graph is not None else None
        self.features = features if features is not None else None
        self.mask = mask if mask is not None else None
        self.labels = labels if labels is not None else None
        self.query_labels = query_labels if query_labels is not None else None
        self.accuracy = accuracy
        self.fidelity = fidelity

    def evaluate_helper(self, model, graph, features, labels, mask):
        """Helper function to evaluate the model's performance."""
        if model is None or graph is None or features is None or labels is None or mask is None:
            return None
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    def evaluate(self):
        """Main function to update fidelity and accuracy scores."""
        self.accuracy = self.evaluate_helper(
            self.model, self.graph, self.features, self.labels, self.mask)
        self.fidelity = self.evaluate_helper(
            self.model, self.graph, self.features, self.query_labels, self.mask)

    def __str__(self):
        """Returns a string representation of the metrics."""
        return f"Fidelity: {self.fidelity:.4f}, Accuracy: {self.accuracy:.4f}"

    @staticmethod
    def calculate_surrogate_fidelity(target_model, surrogate_model, data, mask=None):
        """
        Calculate fidelity between target and surrogate model predictions.
        
        Args:
            target_model: Original model
            surrogate_model: Extracted surrogate model
            data: Input graph data
            mask: Optional mask for evaluation on specific nodes
            
        Returns:
            float: Fidelity score (percentage of matching predictions)
        """
        target_model.eval()
        surrogate_model.eval()

        with torch.no_grad():
            # Get predictions from both models
            target_logits = target_model(data)
            surrogate_logits = surrogate_model(data)

            # Apply mask if provided
            if mask is not None:
                target_logits = target_logits[mask]
                surrogate_logits = surrogate_logits[mask]

            # Get predicted classes
            target_preds = target_logits.argmax(dim=1)
            surrogate_preds = surrogate_logits.argmax(dim=1)

            # Calculate fidelity
            matches = (target_preds == surrogate_preds).sum().item()
            total = len(target_preds)

            return (matches / total) * 100

    @staticmethod
    def evaluate_surrogate_extraction(target_model, surrogate_model, data,
                                      train_mask=None, val_mask=None, test_mask=None):
        """
        Comprehensive evaluation of surrogate extraction attack.
        
        Args:
            target_model: Original model
            surrogate_model: Extracted surrogate model
            data: Input graph data
            train_mask: Mask for training nodes
            val_mask: Mask for validation nodes
            test_mask: Mask for test nodes
            
        Returns:
            dict: Dictionary containing fidelity scores for different data splits
        """
        results = {}

        # Overall fidelity
        results['overall_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
            target_model, surrogate_model, data
        )

        # Split-specific fidelity if masks are provided
        if train_mask is not None:
            results['train_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
                target_model, surrogate_model, data, train_mask
            )

        if val_mask is not None:
            results['val_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
                target_model, surrogate_model, data, val_mask
            )

        if test_mask is not None:
            results['test_fidelity'] = GraphNeuralNetworkMetric.calculate_surrogate_fidelity(
                target_model, surrogate_model, data, test_mask
            )

        return results
