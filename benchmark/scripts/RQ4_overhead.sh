#!/usr/bin/env bash
set -euo pipefail

RQ1_DIR="outputs/RQ1"
RQ2_DIR="outputs/RQ2_RQ3"
OUTDIR="outputs/RQ4_summary"
LEADER_DIR="outputs/leaderboards"
mkdir -p "${OUTDIR}" "${LEADER_DIR}"

echo "[RQ4] Summarizing overhead ..."
python run/summarize_overhead.py \
  --rq1_dir "${RQ1_DIR}" \
  --rq2_dir "${RQ2_DIR}" \
  --outdir "${OUTDIR}"

python run/select_best.py --rq1_dir "${RQ1_DIR}" --rq2_dir "${RQ2_DIR}" --outdir "${LEADER_DIR}"
python run/to_latex.py --outdir "${LEADER_DIR}"

echo "[RQ4] Done. Summary CSVs -> ${OUTDIR}, Leaderboards & LaTeX -> ${LEADER_DIR}"
