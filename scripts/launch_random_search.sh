#!/usr/bin/env bash
# Extend rs_pop50_gen15_n50_v4 from 1 repetition to 5, seeds 99..103.
#
# rep_0 (seed 99) exists and is skipped by the resume check, so only reps 1-4 execute.
# Those four match the four post-reboot GA repetitions: same machine epoch, same training
# problems (taken from the matched GA run's manifest), same 750-evaluation budget.
#
# NO BASELINE STAGE. evaluate_baseline.py derives its output directory from
# n_heldout/heldout_seed/n_reps, so re-running it with the production parameters resolves
# to the EXISTING pre-reboot baseline directory and resumes all ten repetitions from
# checkpoint -- measuring nothing while appearing to succeed. A fresh baseline needs an
# --out-tag flag on that script first. It is only required before held-out evaluation, not
# before these training runs.
#
# Pre-flight gate: a dead model server produces all-zero fitness and a checkpoint written
# without complaint. Probing /api/tags is insufficient; it returns 200 from a server that
# cannot generate.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="${OLLAMA_BIN_DIR:-$HOME/.local/bin}:$PATH"

echo "[rs-reps] pre-flight $(date)"
probe=$(timeout 120 curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5-coder:1.5b","prompt":"int main(){","stream":false,"options":{"num_predict":8}}' \
  | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print('OK' if d.get('done') and d.get('eval_count',0)>0 else 'BAD')
except Exception: print('BAD')" 2>/dev/null)
[ "$probe" = "OK" ] || { echo "[rs-reps] ABORT: Ollama did not complete a real generation." >&2; exit 1; }
[ -r /sys/class/powercap/intel-rapl:0/energy_uj ] || { echo "[rs-reps] ABORT: RAPL unreadable" >&2; exit 1; }
echo "[rs-reps] pre-flight OK"

echo "[rs-reps] start $(date)"
N_WORKERS=4 PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_random_search.py \
    --match-run pop50_gen15_n50_v4 \
    --n-samples 750 --reps 5 \
    --sample-seed 99 --seed-increment 1 \
    --min-feasible 1 \
    --energy-backend rapl_sysfs \
    --run-label rs_pop50_gen15_n50_v4
r=$?
echo "[rs-reps] exit $r at $(date)"
exit $r
