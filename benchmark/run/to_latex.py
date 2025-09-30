# run/to_latex.py
import argparse
import json
import os
import pandas as pd


def to_percent(x, digits=1):
    try:
        return f"{float(x) * 100:.{digits}f}"
    except Exception:
        return "-"


def table_rq1(best_csv: str, out_tex: str):
    df = pd.read_csv(best_csv)
    keep = ["dataset", "attack", "budget_mult", "regime", "perf"]
    df = df[keep].copy()
    df["fid"] = df["perf"].apply(lambda s: json.loads(s.replace("'", '"')).get("fid", None)
                                           or json.loads(s.replace("'", '"')).get("fidelity", None))
    df["acc"] = df["perf"].apply(lambda s: json.loads(s.replace("'", '"')).get("acc", None)
                                           or json.loads(s.replace("'", '"')).get("accuracy", None)
                                           or json.loads(s.replace("'", '"')).get("test_accuracy", None))
    df["Fidelity(%)"] = df["fid"].apply(lambda v: to_percent(v, 1))
    df["Accuracy(%)"] = df["acc"].apply(lambda v: to_percent(v, 1))
    df = df.drop(columns=["perf", "fid", "acc"]).sort_values(["dataset", "budget_mult", "regime", "attack"])
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f"[LaTeX] RQ1 table -> {out_tex}")


def table_rq2(best_csv: str, out_tex: str):
    df = pd.read_csv(best_csv)
    keep = ["dataset", "defense", "perf"]
    df = df[keep].copy()
    df["fid"] = df["perf"].apply(lambda s: json.loads(s.replace("'", '"')).get("fid", None)
                                           or json.loads(s.replace("'", '"')).get("fidelity", None))
    df["def_acc"] = df["perf"].apply(lambda s: json.loads(s.replace("'", '"')).get("def_acc", None)
                                                or json.loads(s.replace("'", '"')).get("acc", None)
                                                or json.loads(s.replace("'", '"')).get("defense_accuracy", None))
    df["1-Fid(%)"] = df["fid"].apply(lambda v: to_percent(1.0 - float(v) if v is not None else None, 1))
    df["Utility(%)"] = df["def_acc"].apply(lambda v: to_percent(v, 1))
    df = df.drop(columns=["perf", "fid", "def_acc"]).sort_values(["dataset", "defense"])
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f"[LaTeX] RQ2/RQ3 table -> {out_tex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1_best", type=str, default="outputs/leaderboards/RQ1_best.csv")
    ap.add_argument("--rq2_best", type=str, default="outputs/leaderboards/RQ2_RQ3_best.csv")
    ap.add_argument("--outdir", type=str, default="outputs/leaderboards")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    table_rq1(args.rq1_best, os.path.join(args.outdir, "RQ1_best_table.tex"))
    table_rq2(args.rq2_best, os.path.join(args.outdir, "RQ2_RQ3_best_table.tex"))


if __name__ == "__main__":
    main()
