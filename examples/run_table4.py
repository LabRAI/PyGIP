"""
Example Script: run_example_bgrove.py
--------------------------------------
Demonstrates how to reproduce one configuration of the BGrOVe
experiment (Table 4).
"""

import torch
from implementation.run_bgrove import run_bgrove_experiment
from pygip.datasets.pyg_datasets import Cora

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run_bgrove_experiment(Cora, condition="CondA ✓", setting="I", device=device)
    print("\n=== Single-run Result (Table 4 Example) ===")
    print("FPR, FNR, ACC =", res)

if __name__ == "__main__":
    main()
