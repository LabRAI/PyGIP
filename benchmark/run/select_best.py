# run/select_best.py
import argparse
import glob
import json
import os
from typing import Dict, Any, List

import pandas as pd


def read_jsonl(folder: str) -> List[Dict[str, Any]]:
    items = []
    for p in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def _score_attack(perf: Dict[str, Any]) -> float:
    fid = perf.get("fid", None) or perf.get("fidelity", None)
    if fid is not None:
        return float(fid)
    acc = perf.get("acc", None) or perf.get("accuracy", None) or perf.get("test_accuracy", None)
    return float(acc) if acc is not None else float("nan")


def _score_defense(perf: Dict[str, Any]) -> float:
    fid = perf.get("fid", None) or perf.get("fidelity", None) or perf.get("post_fidelity", None)
    acc = perf.get("def_acc", None) or perf.get("acc", None) or perf.get("defense_accuracy", None)
    if fid is None and acc is None:
        return float("nan")
    if fid is None:
        fid = 1.0
    if acc is None:
        acc = 1.0
    return (1.0 - float(fid)) * float(acc)


def select_best_attack(records: List[Dict[str, Any]]):
    rows = []
    for r in records:
        if r.get("track") != "attack" or r.get("error"):
            continue
        perf = r.get("perf", {})
        score = _score_attack(perf)
        row = {**r}
        row["score_attack"] = score
        rows.append(row)
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    keys = ["dataset", "attack", "budget_mult", "regime"]
    idx = df.groupby(keys)["score_attack"].idxmax()
    best_df = df.loc[idx].reset_index(drop=True)
    return df, best_df


def select_best_defense(records: List[Dict[str, Any]]):
    rows = []
    for r in records:
        if r.get("track") != "ownership" or r.get("error"):
            continue
        perf = r.get("perf", {})
        score = _score_defense(perf)
        row = {**r}
        row["score_defense"] = score
        rows.append(row)
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    keys = ["dataset", "defense"]
    idx = df.groupby(keys)["score_defense"].idxmax()
    best_df = df.loc[idx].reset_index(drop=True)
    return df, best_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1_dir", type=str, default="outputs/RQ1")
    ap.add_argument("--rq2_dir", type=str, default="outputs/RQ2_RQ3")
    ap.add_argument("--outdir", type=str, default="outputs/leaderboards")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rec1 = read_jsonl(args.rq1_dir)
    rec2 = read_jsonl(args.rq2_dir)

    if rec1:
        full1, best1 = select_best_attack(rec1)
        if full1 is not None:
            full1.to_csv(os.path.join(args.outdir, "RQ1_full.csv"), index=False)
            best1.to_csv(os.path.join(args.outdir, "RQ1_best.csv"), index=False)
            out = {}
            for _, row in best1.iterrows():
                ds = row["dataset"]; atk = row["attack"]
                key = f"{ds}::{atk}::budget{row['budget_mult']}::regime{row['regime']}"
                out[key] = {
                    "dataset": ds,
                    "attack": atk,
                    "budget_mult": row["budget_mult"],
                    "regime": row["regime"],
                    "config_index": int(row["config_index"]),
                    "config": row["config"],
                    "seed": int(row["seed"]),
                    "score_attack": float(row["score_attack"]),
                    "perf": row["perf"],
                    "comp": row["comp"],
                }
            with open(os.path.join(args.outdir, "RQ1_best_configs.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    if rec2:
        full2, best2 = select_best_defense(rec2)
        if full2 is not None:
            full2.to_csv(os.path.join(args.outdir, "RQ2_RQ3_full.csv"), index=False)
            best2.to_csv(os.path.join(args.outdir, "RQ2_RQ3_best.csv"), index=False)
            out = {}
            for _, row in best2.iterrows():
                ds = row["dataset"]; d = row["defense"]
                key = f"{ds}::{d}"
                out[key] = {
                    "dataset": ds,
                    "defense": d,
                    "config_index": int(row["config_index"]),
                    "config": row["config"],
                    "seed": int(row["seed"]),
                    "score_defense": float(row["score_defense"]),
                    "perf": row["perf"],
                    "comp": row["comp"],
                }
            with open(os.path.join(args.outdir, "RQ2_RQ3_best_configs.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    print(f"[DONE] Leaderboards written under {args.outdir}")


if __name__ == "__main__":
    main()
