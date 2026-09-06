# Running the experiments on Linux

Step-by-step to clone, set up, and run the experiments on a Linux box
with real RAPL energy measurement. Applies to any Intel/AMD Linux host that exposes
`/sys/class/powercap/`.

---

## 1. Prerequisites (system-level, not pip)

```bash
# Compilers (the pipeline shells out to gcc / g++)
sudo apt-get update && sudo apt-get install -y build-essential

# Python 3.11+ and venv
python3 --version          # must be >= 3.11
sudo apt-get install -y python3-venv python3-pip

# Ollama (the LLM backend) — see https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh

# Pull the models used in the experiments. qwen2.5-coder:1.5b is the model every
# reported run used; 7b is the larger cross-model point.
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b
```

The `qwen2.5-coder:1.5b` tag is mutable upstream. Every reported run used the
build below; verify with `ollama list` and `ollama show` before reproducing, as
a retagged model would change the results without changing the command:

| field | value |
| --- | --- |
| Ollama id | `d7372fd82851` |
| model layer | `sha256:29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104` |
| size | 986,048,576 bytes |
| quantization | Q4_K_M |
| architecture | qwen2, 1.5B parameters, 32768 context |

Confirm Ollama is serving:

```bash
ollama list           # should show the pulled models
```

---

## 2. Verify real RAPL energy is available

This is the reason for running on Linux. Check the powercap interface:

```bash
cat /sys/class/powercap/intel-rapl:0/energy_uj
```

- **Returns an integer** → real RAPL energy is available; use
  `--energy-backend rapl_sysfs`.
- **No such file / permission denied** → RAPL is not exposed (bare
  metal needed; not a VM with the hypervisor intercepting MSRs). Fall
  back to `--energy-backend codecarbon` (TDP estimate) and label
  results accordingly. See the measurement-protocol report for the measurement protocol.

If the file exists but is root-only, energy reads may need elevated
permissions; check with `ls -l /sys/class/powercap/intel-rapl:0/energy_uj`.

---

## 3. Clone and set up

```bash
git clone <repo-url>
cd apr-decoding-policy-search

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # includes nvidia-ml-py for inference-energy telemetry
```

**Inference-energy telemetry (secondary metric)** is on by default: GPU (NVML) +
CPU (RAPL) counter deltas around each LLM call, recorded per-stage. It's
guarded/fail-open, so a missing GPU just disables the GPU term. Disable entirely
with `export AGENT_INFERENCE_ENERGY=0`. This is **separate** from F3 (the
program-execution energy objective) and never feeds the search.

The Codeflaws **sample manifest** (`benchmarks/codeflaws/data/sample_300_seed42.json`)
is committed, so the Linux run uses the identical 300 problems. The
266 MB Codeflaws tarball itself is **not** committed — it downloads
automatically on first run.

---

## 4. Smoke test (a few minutes)

Confirm the GA pipeline runs end-to-end before launching a long run. Use a tiny
search and the proxy energy backend (fast; no RAPL needed for the smoke):

```bash

python scripts/run_nsga2.py --model qwen2.5-coder:1.5b \
    --pop 4 --gen 1 --reps 1 --train-size 2 --min-feasible 1 \
    --energy-backend runtime_proxy --run-label smoke
```

Then confirm the output is sane (energy/complexity at true scale, not the ~1e10
of the old penalty bug) and delete the smoke folder:

```bash
D=outputs/ga/codeflaws/qwen2.5-coder_1.5b/smoke
cat $D/rep_0/pareto_solutions.json    # cl in single digits, energy in tens of mJ
rm -rf $D
```

---

## 5. Running the four-arm protocol

The protocol compares **four arms at a matched 750-evaluation budget** — GA, RS, MOBO, and the default
baseline as reference — replicated, then evaluated on a shared held-out set.


**Run the arms one at a time on an otherwise idle machine.** F3 is an energy
measurement; a competing job corrupts it silently, and under `runtime_proxy` F3
*is* wall-clock. Each arm costs roughly 48–60 h.

```bash

# GA — the arm the controls are matched against
N_WORKERS=4 python scripts/run_nsga2.py \
    --benchmark codeflaws --model qwen2.5-coder:1.5b \
    --pop 50 --gen 15 --train-size 50 --min-feasible 1 \
    --energy-backend rapl_sysfs --run-label <label>

# RS control — matched off the GA's manifest
N_WORKERS=4 python scripts/run_random_search.py \
    --benchmark codeflaws --match-run <label> \
    --n-samples 750 --sample-seed 99 --energy-backend rapl_sysfs

# MOBO control — needs the botorch interpreter, NOT the project venv
N_WORKERS=4 "$MOBO_PYTHON" scripts/run_mobo.py \
    --benchmark codeflaws --match-run <label> \
    --n-samples 750 --n-init 50 --batch-size 5 --sample-seed 123 \
    --energy-backend rapl_sysfs
```

The controls read the NSGA-II manifest and **fail closed** if the model, the
evaluation budget, the feasibility rule or the training ids do not match; a
differing worker count is reported as a warning rather than an error. That is
deliberate: a control that differs from the arm it controls for in anything but
the proposal mechanism is not a control. The energy backend and the producing
script are recorded in both manifests for comparison after the fact, but are not
asserted at startup.

**Replication.** `scripts/launch_nsga2.sh` extends an existing run to further
repetitions. It varies only the search RNG — the training problems are chosen
from the base seed outside the rep loop, so every rep sees identical problems.
It also gates on Ollama completing an actual generation before starting: a
previous attempt completed 11 generations in two minutes against a stopped model
server, every candidate scoring correctness 0.0, and no stage of the pipeline
treated that as an error. Checking `/api/tags` is not sufficient — it returns 200
from a server that cannot generate.

**Held-out evaluation, with the comparability guard:**

```bash
python scripts/evaluate_heldout.py \
    --benchmark codeflaws --run-labels <label> \
    --n-heldout 100 --heldout-seed 0 \
    --workers 4 --energy-backend rapl_sysfs \
    --expect-heldout-sha256 <digest from the first arm's manifest>
```

`--expect-heldout-sha256` aborts if the selected problem set differs from the
recorded one by a single id. Every arm must be evaluated under the same digest,
or the comparison is not a comparison. Add `--dry-run` first — the digest is
checked *before* the dry-run exits, so it validates cheaply.

**Cross-benchmark transfer** — evaluate one benchmark's front on the other's
problems with `--front-file` and a required `--out-tag`:

```bash
python scripts/evaluate_heldout.py \
    --benchmark runbugrun_py --front-file <path to C pareto_solutions.json> \
    --out-tag transfer_c_v4_n100_seed0 \
    --n-heldout 100 --heldout-seed 0 --energy-backend runtime_proxy \
    --expect-heldout-sha256 <the Python arm's digest>
```

**MOBO interpreter.** `botorch`/`torch` are not in the project venv. Build a
separate venv on the *same* Python 3.11 with `requirements.txt` on top, so the
drivers and adapters import identically. **Cap the threads** — the script
defaults to 4 via `MOBO_TORCH_THREADS`. Letting torch take all cores of a 6-core
host made one GP fit take 112 s and the acquisition exceed 900 s, against 1.5 s
and 31 s at 4 threads.

---

## 6. Optional: enable LLM-call CO2eq

To also record the carbon cost of running the pipeline (separate from
F3 binary energy):

```bash
export AGENT_CODECARBON=1
export AGENT_CO2_COUNTRY=GBR     # ISO code for grid mix
# then run as above
```

The telemetry schema this populates is not yet written up as a guide; read
`telemetry.py` and `inference_energy.py` for the field definitions.

---

## 7. Pulling results back

**Nothing under `outputs/` is tracked.** A search or evaluation writes its whole
tree there — fronts, manifests, per-candidate evidence, repaired sources and
verification records — and the directory is ignored in full. What this repository
carries instead is the curated set: the search fronts and protocols under
`results/`, the default-policy baselines under `baselines/`, and the executed
notebook. The held-out and baseline evaluation records the analysis reads are
published as a separate data release and are unpacked into `outputs/`.

Repaired programs execute with the working directory at the repository root and
leave scratch files there (`input.txt`, `random_strings.txt`,
`processed_input.txt` and similar). These are ignored, but check `git status`
after a run rather than staging blindly.

To copy results off the run host without going through git:

```bash
# from your local machine
scp -r user@host:~/apr-decoding-policy-search/outputs/ga ./ga-results
```

On a host where `outputs/` is a symlink to scratch storage, a pull that carries
tracked files under `outputs/` will replace the symlink with a real directory
and orphan the data. Check `ls -ld outputs` before pulling.

---

## Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `energy_backend=runtime_proxy` despite `--energy-backend rapl_sysfs` | RAPL not readable. The explicit flag should *fail loudly*; if it fell back, check the powercap file permissions. |
| Ollama connection refused | `ollama serve` not running, or `OLLAMA_HOST` mismatch. |
| gcc/g++ not found | `sudo apt-get install build-essential`. |
| Slow per-problem time | Expected — each problem is a full 4-stage LLM pipeline. Use a smaller `--train-size` for iteration. |
