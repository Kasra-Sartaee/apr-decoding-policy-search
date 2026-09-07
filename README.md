# Multi-Objective Search over the Decoding Policy of a Multi-Agent LLM Pipeline for Automated Program Repair

Code, search fronts and analysis for the dissertation of the same name. This
repository contains everything needed to inspect the method and re-run the
analysis; the held-out evaluation records are too large to track here and are
released separately (see **Data release** below).

## Layout

| path | contents |
| --- | --- |
| `agentic_workflow.py`, `agents.py` | the four-stage Analyzer→Repair→Verifier→Optimizer pipeline (§3.1) |
| `search_space.py` | the 11-dimensional decision vector and its decoder, imported by every driver (§3.2) |
| `contracts/` | the six versioned prompt contracts, in full (Appendix C) |
| `benchmarks/` | Codeflaws and RunBugRun adapters, with both pinned 300-problem sample manifests (§4.3) |
| `energy/` | RAPL and proxy energy backends (§3.4.3) |
| `scripts/` | the search drivers, evaluators, launchers and generators — see *Scripts and modules* below |
| `baselines/` | default-policy reference runs on both benchmarks, five repetitions each |
| `tests/` | unit and measurement-protocol regression tests |
| `results/` | the published search fronts, protocols and trajectories |
| `analysis/` | the analysis notebook, its generator and the table scripts |
| `docs/reports/` | the four reports the dissertation cites: the measurement protocol, budget matching, the replicated three-arm comparison and the search-cost accounting |

## Scripts and modules

Every tracked entry point and supporting module, and what it is for.

### Pipeline

| file | purpose |
| --- | --- |
| `agentic_workflow.py` | Orchestrates one repair attempt end to end: Analyzer → Repair → Verifier → Optimizer, with the two retry loops, and returns the three objectives for the retained source. |
| `agents.py` | The agent classes and their Ollama calls. Loads the prompt contract for each stage, applies the candidate policy's decoding parameters, and holds the readers-writer gate that keeps local inference out of an open RAPL window. |
| `search_space.py` | The 11-dimensional decision vector: bounds, types, and the decoder from a raw vector to a policy dictionary. Imported by all three drivers, so every arm provably searches the same space. |

### Search drivers

One repetition of one arm per invocation; all three write the same manifest and protocol format.

| file | purpose |
| --- | --- |
| `scripts/run_nsga2.py` | The NSGA-II arm (§3.3.1), and the driver that produced the published `results/nsga2/` fronts. Resumable per generation. |
| `scripts/run_random_search.py` | The uniform random-sampling control (§3.3.3). Verifies its budget, model and training split against the NSGA-II manifest before starting, so the comparison cannot silently drift. |
| `scripts/run_mobo.py` | The qLogNEHVI surrogate arm (§3.3.2). Needs the `mobo` extra, which pins botorch and a matching torch build. |
| `scripts/run_nsga2_runtime.py` | A secondary NSGA-II driver whose F3 defaults to the `runtime_proxy` backend rather than measured energy, so verifications can run concurrently. Useful on hosts without RAPL. Did not produce any reported result. |

### Evaluators

| file | purpose |
| --- | --- |
| `scripts/evaluate_heldout.py` | Re-evaluates every policy on a repetition's training Pareto front against a held-out problem set disjoint from the one searched. Resumable per policy; `--expect-heldout-sha256` pins the problem set by digest so an evaluation cannot run on the wrong problems. This produces the data release. |
| `scripts/evaluate_baseline.py` | The same measurement path applied to the default policy, giving the reference each arm is compared against. Measured identically to the fronts so the contrast is meaningful. |

### Launchers

Thin wrappers recording exactly how the reported runs were invoked.

| file | purpose |
| --- | --- |
| `scripts/launch_nsga2.sh` | Extends the NSGA-II arm from one repetition to five (seeds 42–46). Contains the pre-flight gate that refuses to start unless Ollama completes a real generation. |
| `scripts/launch_random_search.sh` | The same for random search, seeds 99–103. |
| `scripts/launch_mobo.sh` | The same for the surrogate arm, seeds 123–127. |
| `scripts/launch_heldout_mobo_rep3.sh`, `scripts/launch_heldout_mobo_rep4.sh` | Held-out evaluation of the last two surrogate repetitions across both benchmarks, strictly serial: Codeflaws under real RAPL first, then the RunBugRun transfer. |

### Benchmarks

| file | purpose |
| --- | --- |
| `benchmarks/base.py` | The `Benchmark`, `Problem` and `TestRunner` abstractions, and the two-phase measurement path (concurrent Phase A, serialised Phase B inside one RAPL window). |
| `benchmarks/registry.py` | Maps `--benchmark <name>` to an adapter, imported lazily. |
| `benchmarks/codeflaws/adapter.py` | The Codeflaws C benchmark. Downloads and extracts the corpus, follows each problem's official `test-valid.sh` case list, and batches executions so a RAPL window clears the counter's resolution floor. |
| `benchmarks/runbugrun_py/adapter.py` | The RunBugRun Python benchmark, with upstream release assets pinned by SHA-256 and groups its records by upstream task so the held-out split is independent. |

### Measurement

| file | purpose |
| --- | --- |
| `energy/backends.py` | The energy backends: Linux RAPL via sysfs, Apple Silicon, CodeCarbon, and the runtime proxy. |
| `energy/detect.py` | Resolves `--energy-backend`, either by name or by picking the best available. Fails loudly when an explicitly named backend is unavailable. |
| `inference_energy.py` | Secondary per-call inference-energy telemetry (CPU, and GPU via NVML). Descriptive only — it is never mixed into F3. |
| `telemetry.py` | Per-LLM-call records: token counts, wall time, retry counts. |
| `provenance.py` | Writes the commit, dirty flag, treatment and configuration into each run's `protocol.json`. See `PROVENANCE.md` for how to read the dirty flag. |

### Analysis

| file | purpose |
| --- | --- |
| `scripts/generators/build_three_arm_notebook.py` | Generates `analysis/three_arm_analysis.ipynb`. The notebook is generated, never hand-edited. |
| `scripts/generators/export_report.py` | Called from the notebook's last cell; writes `docs/reports/replicated_three_arm_comparison.md` and its backing JSON from the executed results, so no number in the report is transcribed by hand. |
| `analysis/pareto_table.py` | Emits the appendix table of selected Pareto policies, self-checking both problem-set digests and refusing to invent a record it cannot find. |
| `analysis/search_cost.py` | Produced `docs/reports/search_cost.md`. Its telemetry inputs were not retained, so it now exits rather than overwrite the preserved report. |
| `analysis/nsga2_vs_random_search_tables.py` | Auxiliary NSGA-II versus random-search quantities, predating the surrogate arm. Not a dissertation-table generator. |

### Tests

| file | purpose |
| --- | --- |
| `tests/test_measurement_protocol.py` | Regression tests for the v3.1 measurement path. |
| `tests/test_python_arm_parity.py` | Checks the Python arm is measured like the C arm, and that the held-out split is disjoint by upstream task, not merely by record id. |
| `tests/conftest.py` | Skips tests needing a benchmark corpus when it is absent, so the default run is green on a fresh clone. |

## Reproducing the reported numbers

Reported quantities fall into two classes, and the distinction matters for
anyone checking this work.

**Regenerated from released artefacts.** These run end to end given the data
release, and any of them can be recomputed from scratch:

| dissertation item | regenerated by | inputs required |
| --- | --- | --- |
| Tables 8-14, Figures 2-4 | `analysis/three_arm_analysis.ipynb` | `outputs/` from the data release, plus `results/` and `baselines/` in this repository |
| that notebook itself | `scripts/generators/build_three_arm_notebook.py` | none |
| the three-arm comparison report and its backing JSON | `scripts/generators/export_report.py`, called from the notebook's final cell | the executed notebook |
| Table 16 (selected policies) | `analysis/pareto_table.py` | `results/` only |

**Preserved derived results.** The search-cost and budget-matching figures were
computed from per-candidate verification telemetry that the searches wrote under
`outputs/` during the runs. That telemetry was not retained, so these figures are
the measurements taken while it was present and are *not* independently
recomputable from the release:

| dissertation item | reported in | status |
| --- | --- | --- |
| §5.1 cost figures | `docs/reports/search_cost.md` | preserved; `analysis/search_cost.py` ships as the exact code that produced them, but its inputs no longer exist |
| budget-matching accounting | `docs/reports/budget_matching.md` | mostly verifiable from `results/` (the pymoo evaluation counter and the stored per-candidate records); its third cross-check, over analyzer invocations, is telemetry-derived and preserved only |

`analysis/nsga2_vs_random_search_tables.py` is auxiliary rather than a
dissertation-table generator: it predates the surrogate arm and covers the
NSGA-II versus random-search pair only. Its output goes to the untracked
`analysis/out/` and still carries `[FILL]` placeholders. Every three-arm result
the dissertation reports comes from the notebook.

The notebook reads the held-out records from the data release. Download and
unpack it so that `outputs/` contains the evaluation
trees, then execute the notebook:

```bash
python -m venv .venv && .venv/bin/pip install -e '.[analysis]'
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    analysis/three_arm_analysis.ipynb
```

The committed copy of the notebook is already executed, so its outputs can be
read without re-running it.

## Re-running the searches

Each arm is a single driver over the shared search space. A search occupies one
host for 45–63 hours (§5.1) and requires Ollama serving `qwen2.5-coder:1.5b`
(model layer `sha256:29d8c98f…`, Q4_K_M — the tag is mutable upstream, so the
digest is recorded in `docs/guides/running_on_linux.md`),
and RAPL through `/sys/class/powercap/` for the energy objective. See
`docs/guides/running_on_linux.md`.

```bash
# NSGA-II            (§3.3.1)
N_WORKERS=4 python scripts/run_nsga2.py --benchmark codeflaws \
    --pop 50 --gen 15 --train-size 50 --energy-backend rapl_sysfs --run-label <label>
# random sampling    (§3.3.3) — verifies budget/model/split against the GA manifest
N_WORKERS=4 python scripts/run_random_search.py --benchmark codeflaws \
    --match-run <label> --n-samples 750 --energy-backend rapl_sysfs
# surrogate arm      (§3.3.2) — needs a separate interpreter with botorch
N_WORKERS=4 "$MOBO_PYTHON" scripts/run_mobo.py \
    --benchmark codeflaws --match-run <label> --n-samples 750 --n-init 50 --batch-size 5
# held-out evaluation, with the problem-set digest guard (Appendix B.4)
python scripts/evaluate_heldout.py --benchmark codeflaws --run-labels <label> \
    --n-heldout 100 --heldout-seed 0 --energy-backend rapl_sysfs \
    --expect-heldout-sha256 365503a2457502e94a12ebe7c818b4cd30d335549a7682c7a14dac575b80b6f7
```

## Tests

```bash
pip install -e '.[analysis,dev]'
python -m pytest tests -q
```

This passes on a fresh clone: 48 passed, 4 skipped. The skips are tests that
need a benchmark corpus. The Codeflaws (3.1 GB) and RunBugRun (263 MB) corpora
are not tracked here and are fetched on first use by each adapter's
`download()`; tests depending on them carry the `integration` marker and skip
when the corpus is absent.

To run those once the corpora are in place:

```bash
python -m pytest tests -q -m integration
```

## Data release

The held-out evaluation records the notebook reads are not tracked here. They
are released as an asset on the `v1.0` release of this repository:

> **<https://github.com/Kasra-Sartaee/apr-decoding-policy-search/releases/download/v1.0/apr-decoding-policy-search-heldout-records-v1.tar.gz>**
>
> 23 MB compressed, 762 MB unpacked, 1,450 files.
> SHA-256 `8831aac8082b8220a7131fb54d7ff76fb4f2c2023357b6953d94f8db65707611`

Unpack the archive at the repository root so that the following four trees
exist. Nothing else from the archive is required by the analysis:

```
outputs/ga/codeflaws/qwen2.5-coder_1.5b/
    pop50_gen15_n50_v4/rep_0/heldout_rep{0..4}_n100_seed0/
    rs_pop50_gen15_n50_v4/rep_0/heldout_rsrep{0..4}_n100_seed0/
    mobo_pop50_gen15_n50_v4/rep_0/heldout_moborep{0..4}_n100_seed0/
outputs/ga/runbugrun_py/qwen2.5-coder_1.5b/
    py_n50_random/rep_0/transfer_{c,rs,mobo}_rep{0..4}_n100_seed0/
```

Which tree backs which analysis:

| tree | backs |
| --- | --- |
| the three `codeflaws/…/heldout_*` sets | the primary-benchmark held-out results: Tables 8-13, Figures 2-4 |
| `runbugrun_py/…/transfer_*` | the cross-language transfer results, Table 14 |
| `baselines/` (tracked here, not in the archive) | the default-policy reference each arm is compared against |

The archive ships a `checksums.txt` of SHA-256 digests over the unpacked files;
verify with `sha256sum -c checksums.txt` from the repository root before running
the analysis. The notebook independently guards the problem sets it evaluates by
digest (`365503a2…` for Codeflaws, `e3d0637c…` for RunBugRun) and aborts if the
unpacked records do not match the published held-out split.

The per-candidate verification telemetry written during the searches is **not**
in the archive; it was not retained. See *Reproducing the reported numbers* above
for which figures that affects, and `PROVENANCE.md` for the full account.

## Attribution

This repository is not solely the dissertation author's work. The pipeline, its
agent graph and its prompt contracts are inherited, as is the idea of searching
the pipeline's decoding policy with a multi-objective evolutionary algorithm: a
fourteen-dimensional NSGA-II policy search was already in the repository before
this dissertation began. The work of this dissertation is the redesign into the
present eleven-dimensional shared formulation and its extraction into a common
module, the three search drivers and their budget-matching controls, the
two-phase energy measurement protocol, the benchmark adapters, the held-out and
transfer evaluators, and the whole of the analysis and reporting.

`ATTRIBUTION.md` records who contributed what, with the commits.
