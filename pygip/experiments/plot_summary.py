#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

fn = "experiments/results_summary.csv"
out_png = "experiments/auc_summary.png"

df = pd.read_csv(fn, keep_default_na=False)
# convert numeric columns
df['extraction_auc'] = pd.to_numeric(df['extraction_auc'], errors='coerce')
df['pruning_test_auc'] = pd.to_numeric(df['pruning_test_auc'], errors='coerce')
df['pruning_ratio'] = pd.to_numeric(df['pruning_ratio'], errors='coerce')

# keep only rows with both AUCs
plot_df = df.dropna(subset=['extraction_auc','pruning_test_auc']).copy()

if plot_df.empty:
    print("No complete AUC pairs to plot")
    plt.figure(figsize=(6,4))
    plt.text(0.5, 0.5, "No complete AUC pairs to plot", ha="center", va="center")
    plt.savefig(out_png)
    print("Wrote", out_png)
    raise SystemExit(0)

# unique datasets -> colors
datasets = sorted(plot_df['dataset'].unique())
colors = plt.cm.tab10(range(len(datasets)))
color_map = dict(zip(datasets, colors))

plt.figure(figsize=(7,6))
for ds in datasets:
    sub = plot_df[plot_df['dataset'] == ds]
    plt.scatter(sub['extraction_auc'], sub['pruning_test_auc'], label=ds, s=60, c=[color_map[ds]])
    # annotate with prune ratio (slightly offset)
    for _, r in sub.iterrows():
        plt.annotate(f"p={r['pruning_ratio']}", (r['extraction_auc']+0.001, r['pruning_test_auc']+0.001), fontsize=8)

plt.xlim(0.45, 1.0)
plt.ylim(0.45, 1.0)
plt.xlabel("Extraction surrogate test AUC")
plt.ylabel("Pruning test AUC")
plt.title("GENIE demo: extraction vs pruning (summary)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(out_png)
print("Wrote", out_png)
