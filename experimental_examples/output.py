import json
with open("verification_metrics.json") as f:
    metrics = json.load(f)
print(metrics["ROC_AUC"], metrics["ARUC"])
