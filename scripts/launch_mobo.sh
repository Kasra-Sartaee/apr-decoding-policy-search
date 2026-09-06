#!/usr/bin/env bash
# Extend mobo_pop50_gen15_n50_v4 from 1 repetition to 5, seeds 123..127.
#
# rep_0 (seed 123) exists and is skipped by the resume check -- both its front and its
# history are present -- so only reps 1-4 execute, matching the four post-reboot GA and RS
# repetitions.
#
# Runs under the SCRATCH interpreter: botorch/torch are not in the project venv (installing
# them there exhausted a 10 GB quota and truncated itself). That venv also carries
# requirements.txt, so run_nsga2 and the benchmark adapters import normally.
#
# --seed-increment 1 varies the initial design AND the acquisition optimiser per repetition.
# Training problems come from the matched GA run's manifest and are identical throughout.
#
# Pre-flight gate: a dead model server yields all-zero fitness and a checkpoint written
# without complaint. Probing /api/tags is insufficient -- it returns 200 from a server that
# cannot generate.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="${OLLAMA_BIN_DIR:-$HOME/.local/bin}:$PATH"

echo "[mobo-reps] pre-flight $(date)"
probe=$(timeout 120 curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5-coder:1.5b","prompt":"int main(){","stream":false,"options":{"num_predict":8}}' \
  | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print('OK' if d.get('done') and d.get('eval_count',0)>0 else 'BAD')
except Exception: print('BAD')" 2>/dev/null)
[ "$probe" = "OK" ] || { echo "[mobo-reps] ABORT: Ollama did not complete a real generation." >&2; exit 1; }
[ -r /sys/class/powercap/intel-rapl:0/energy_uj ] || { echo "[mobo-reps] ABORT: RAPL unreadable" >&2; exit 1; }
echo "[mobo-reps] pre-flight OK"

echo "[mobo-reps] start $(date)"
N_WORKERS=4 PYTHONUNBUFFERED=1 "${MOBO_PYTHON:?set MOBO_PYTHON to the interpreter that has botorch installed}" scripts/run_mobo.py \
    --match-run pop50_gen15_n50_v4 \
    --n-samples 750 --n-init 50 --batch-size 5 --reps 5 \
    --sample-seed 123 --seed-increment 1 \
    --min-feasible 1 \
    --energy-backend rapl_sysfs \
    --run-label mobo_pop50_gen15_n50_v4
r=$?
echo "[mobo-reps] exit $r at $(date)"
exit $r
