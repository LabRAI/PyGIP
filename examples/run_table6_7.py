"""
Example Script: run_example_analyze_extended.py
--------------------------------------
Runs the analysis that produces Table 6 (fine-tuning robustness)
and Table 7 (false positives).
"""

from implementation.adversial import generate_tables

def main():
    print("Running analysis for Tables 6 & 7 ...")
    generate_tables("results/table5_all_results.csv")

if __name__ == "__main__":
    main()
