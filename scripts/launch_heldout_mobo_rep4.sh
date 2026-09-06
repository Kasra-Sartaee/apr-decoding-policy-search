#!/usr/bin/env bash
# Held-out evaluation of MOBO rep_4 on both benchmarks, strictly serial.
# Codeflaws first (primary, real RAPL), then the RunBugRun transfer.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=logs/mobo_rep4_heldout.log
mkdir -p logs
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== stray sweep before starting ==="
$PY -c "
import sys; sys.path.insert(0,'.')
from benchmarks.base import sweep_stray_workloads
s = sweep_stray_workloads(kill=True)
print(f'strays killed: {len(s)}')
for pid, cmd in s: print('  ', pid, cmd[:120])
" 2>&1 | tee -a "$LOG"

say "=== [1/2] Codeflaws — MOBO rep_4 (43 policies, ~13h expected) ==="
$PY scripts/evaluate_heldout.py \
    --benchmark codeflaws \
    --run-labels mobo_pop50_gen15_n50_v4 \
    --front-file results/mobo/rep_4/pareto_solutions.json \
    --out-tag heldout_moborep4_n100_seed0 \
    --n-heldout 100 --heldout-seed 0 \
    --workers 4 --energy-backend rapl_sysfs \
    --expect-heldout-sha256 365503a2457502e94a12ebe7c818b4cd30d335549a7682c7a14dac575b80b6f7 \
    >> "$LOG" 2>&1
rc1=$?
say "=== [1/2] Codeflaws exited rc=$rc1 ==="

if [ $rc1 -ne 0 ]; then
  say "!! Codeflaws failed; NOT starting the Python leg. Investigate before resuming."
  exit $rc1
fi

say "=== [2/2] RunBugRun transfer — MOBO rep_4 (43 policies, ~7h expected) ==="
$PY scripts/evaluate_heldout.py \
    --benchmark runbugrun_py \
    --run-labels py_n50_random \
    --front-file results/mobo/rep_4/pareto_solutions.json \
    --out-tag transfer_mobo_rep4_n100_seed0 \
    --n-heldout 100 --heldout-seed 0 \
    --workers 4 --energy-backend rapl_sysfs \
    --expect-heldout-sha256 e3d0637c2d1331fbf0d9116b05fe2192352d9897d0f9d0a46b2b55ca49bb8deb \
    >> "$LOG" 2>&1
rc2=$?
say "=== [2/2] RunBugRun exited rc=$rc2 ==="
say "=== BOTH LEGS DONE (codeflaws=$rc1 runbugrun=$rc2) ==="
