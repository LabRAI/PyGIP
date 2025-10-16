"""
Example Script: run_example_table5.py
--------------------------------------
Demonstrates how to run the main Table 5 experiment (and Figure 3)
using the unified training pipeline.
"""

import torch
from implementation.run_table5_full import run_table5_full

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = run_table5_full(dataset_name="Cora", setting="I", device=device)
    print("\n=== Single-run Result (Table 5 Example) ===")
    print(df.head())

if __name__ == "__main__":
    main()
