# utils/plotter.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

def plot_robustness_uniqueness(robustness_vals, uniqueness_vals, save_path=None):
    """
    Plots the Robustness-Uniqueness scatter plot.

    Args:
        robustness_vals (list or np.array): Robustness values for fingerprints.
        uniqueness_vals (list or np.array): Uniqueness values for fingerprints.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(robustness_vals, uniqueness_vals, alpha=0.7, c='blue', edgecolors='k')
    plt.xlabel("Robustness")
    plt.ylabel("Uniqueness")
    plt.title("Robustness vs Uniqueness")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_aruc_curve(robustness_curve, uniqueness_curve, save_path=None):
    """
    Plots the ARUC curve (Area under Robustness-Uniqueness Curve).

    Args:
        robustness_curve (list or np.array): Robustness values along curve.
        uniqueness_curve (list or np.array): Uniqueness values along curve.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    robustness_curve = np.array(robustness_curve)
    uniqueness_curve = np.array(uniqueness_curve)

    # Compute area under the curve
    aruc_score = auc(robustness_curve, uniqueness_curve)

    plt.figure(figsize=(6, 6))
    plt.plot(robustness_curve, uniqueness_curve, marker='o', linestyle='-', color='red')
    plt.fill_between(robustness_curve, uniqueness_curve, alpha=0.2, color='red')
    plt.xlabel("Robustness")
    plt.ylabel("Uniqueness")
    plt.title(f"ARUC Curve (Score = {aruc_score:.4f})")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return aruc_score

