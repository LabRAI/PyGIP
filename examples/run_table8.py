"""
Example Script: run_example_double_extraction.py
--------------------------------------
Demonstrates how to reproduce Table 8 (Double Extraction Robustness).
"""

from implementation.adversial_table8 import generate_table8

def main():
    print("Running Double Extraction analysis (Table 8) ...")
    generate_table8("results/table5_all_results.csv")

if __name__ == "__main__":
    main()
