# utils/evaluator.py

import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


def evaluate_metrics(preds, labels, save_path=None, average="binary"):
    """
    Evaluate classification metrics.

    Args:
        preds (np.ndarray or list): Predicted class labels (0/1 or multiclass).
        labels (np.ndarray or list): Ground truth labels.
        save_path (str, optional): If provided, save metrics to JSON.
        average (str): Averaging method for F1/precision/recall (binary, micro, macro, weighted).

    Returns:
        dict: Computed metrics.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    metrics = {}

    # ✅ Accuracy
    metrics["accuracy"] = accuracy_score(labels, preds)

    # ✅ F1 Score
    try:
        metrics["f1"] = f1_score(labels, preds, average=average)
    except Exception:
        metrics["f1"] = None

    # ✅ Precision & Recall
    try:
        metrics["precision"] = precision_score(labels, preds, average=average)
        metrics["recall"] = recall_score(labels, preds, average=average)
    except Exception:
        metrics["precision"] = None
        metrics["recall"] = None

    # ✅ ROC AUC (only valid if binary or multilabel probabilities available)
    try:
        metrics["auc"] = roc_auc_score(labels, preds)
    except Exception:
        metrics["auc"] = None

    # ✅ Save metrics
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics
