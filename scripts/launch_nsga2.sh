#!/usr/bin/env bash
# Extend pop50_gen15_n50_v4 from 1 repetition to 5, for between-run variance.
#
# rep_0 (seed 42) exists and is SKIPPED by run_nsga2.py's resume check, which tests
# for pareto_solutions.json + intermediate_solutions.json (run_nsga2.py:595-601).
# Only reps 1-4 run, at seeds 43/44/45/46.
#
# --seed-increment 1 varies ONLY the search RNG: initial population and SBX/PM
# draws. The 50 training problems are chosen once from the BASE seed, outside the
# rep loop (run_nsga2.py:765), so all five reps see identical problems. Verified at
# launch: train_ids sha bab4ff5403a3b426 matches rep_0.
#
# PRE-FLIGHT GATE. A previous launch completed 11 generations in two minutes
# against a stopped Ollama server: every LLM call failed immediately, all 550
# candidates scored correctness 0.0 with 0/550 feasible, and the driver wrote a
# checkpoint over the result. No stage of the pipeline treats an unavailable model
# server as an error, so the gate below refuses to start until a generation
# actually succeeds. Polling /api/tags is insufficient; it returns 200 from a
# server that cannot generate.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="${OLLAMA_BIN_DIR:-$HOME/.local/bin}:$PATH"

echo "[ga-reps] pre-flight $(date)"
probe=$(timeout 120 curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5-coder:1.5b","prompt":"int main(){","stream":false,"options":{"num_predict":8}}' \
  | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print('OK' if d.get('done') and d.get('eval_count',0)>0 else 'BAD')
except Exception: print('BAD')" 2>/dev/null)
if [ "$probe" != "OK" ]; then
    echo "[ga-reps] ABORT: Ollama did not complete a real generation. Not starting." >&2
    exit 1
fi
echo "[ga-reps] pre-flight OK — Ollama generates"

[ -r /sys/class/powercap/intel-rapl:0/energy_uj ] || { echo "[ga-reps] ABORT: RAPL unreadable" >&2; exit 1; }
echo "[ga-reps] RAPL readable"

echo "[ga-reps] start $(date)"
N_WORKERS=4 PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_nsga2.py \
    --model qwen2.5-coder:1.5b \
    --pop 50 --gen 15 --reps 5 --train-size 50 \
    --seed 42 --seed-increment 1 \
    --min-feasible 1 \
    --energy-backend rapl_sysfs \
    --run-label pop50_gen15_n50_v4
rc=$?
echo "[ga-reps] exit $rc at $(date)"
exit $rc
