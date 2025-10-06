chmod +x scripts/*.sh

# RQ1: attacks (includes MEA 0..5)
bash scripts/RQ1_attacks.sh 0

# RQ2: defenses (best operating points)
bash scripts/RQ2_defenses.sh 0

# RQ3: dense sweeps for trade-off curves
bash scripts/RQ3_tradeoff.sh 0

# RQ4: overhead summaries
bash scripts/RQ4_overhead.sh
