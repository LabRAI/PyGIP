# experiments/clean_results.py
import pandas as pd
import shutil

CSV_PATH = "experiments/results_summary.csv"
BACKUP_PATH = "experiments/results_summary_backup.csv"

# Create a backup before modifying
shutil.copy(CSV_PATH, BACKUP_PATH)
print(f"Backup saved to {BACKUP_PATH}")

df = pd.read_csv(CSV_PATH, keep_default_na=False)

def is_valid(val):
    try:
        float(val)
        return True
    except Exception:
        return False

# Keep only rows with valid numeric extraction_auc and pruning_test_auc
df_clean = df[df["extraction_auc"].apply(is_valid) & df["pruning_test_auc"].apply(is_valid)]

# Overwrite with cleaned dataframe
df_clean.to_csv(CSV_PATH, index=False)
print(f"Cleaned {CSV_PATH}, kept {len(df_clean)} rows (out of {len(df)})")
