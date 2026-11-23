"""
Example Script: run_example_bboxve.py
--------------------------------------
Demonstrates how to run the BBoxVe (Backdoor-based Ownership Verification)
experiment from Table 3 using PyGIP.
"""

import torch
from implementation.run_bboxve import run_experiment

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    res = run_experiment("Cora", "GCN", with_backdoor=True, device=device)
    print("\n=== Single-run Result (Table 3 Example) ===")
    print(res)

if __name__ == "__main__":
    main()
