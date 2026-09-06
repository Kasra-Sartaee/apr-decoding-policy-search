# Codeflaws Two-Phase Deferred Energy Measurement (v3)

**Date:** 24 July 2026 (revised 27 July 2026 — see §3.2)
**Context:** Fixing RAPL contamination and its performance cost under `N_WORKERS>1` + local Ollama
**Status:** Implemented (v3.1). §3 and §3.1 describe the superseded per-test measurement and are
retained as a record of the reasoning; **§3.2 supersedes them**. §3.6 records declared
limitations and §3.7 the fixes from the 28 July external audit.
**Scope:** All `N_WORKERS>1` Codeflaws GA runs from this commit onward. Also changes the
measurement protocol for `N_WORKERS=1` runs on Codeflaws (see §4) — **not** a parallelism-only fix.

---

## 1. Summary

Two related problems were found and fixed in the Codeflaws GA pipeline's F3 (energy) measurement:

1. **Correctness bug (fixed 2026-07-23):** under `N_WORKERS>1` with local Ollama, concurrent LLM
   inference and concurrent test executions could both run during another thread's "isolated" RAPL
   window, inflating F3. Confirmed via a 3.8x gap in mean F3 between a sequential and a
   `N_WORKERS=4` repetition of the same n50 search, with zero range overlap.
2. **Performance problem (fixed 2026-07-24):** the initial fix (a readers-writer lock,
   `_InferenceVerifyGate`) restored validity but not much speed — a 16-problem smoke test showed
   82% of wall-clock time was writer-active, because most verification calls are intermediate
   repair-retry / optimizer-retry attempts whose energy is *discarded* the moment a test fails, and
   partial/broken repairs frequently hang, each burning the full `run_timeout` under exclusive RAPL
   access.

The v3 protocol in this report fixes (2) by **never measuring energy for anything except the
single, final, already-known-correct artifact**, and does so via a true concurrency barrier rather
than a lock — which incidentally also closes a validity gap the lock-only design still had (gcc
compilation was never gated).

**Correction (27 July).** This originally read "F1 and F2 are unaffected by this report and remain
valid and comparable across v2 and v3." That is **no longer true**, and the claim is withdrawn:

- **F2** is now computed under a stricter two-execution rule. Phase A's diagnostic establishes which
  tests pass; Phase B re-executes exactly those, and any that fail to reproduce are demoted from the
  pass count (§3.2). Real downward revisions are observed in practice (e.g. 1.000 → 0.750). F2 under
  v3.1 is therefore a *stricter* measure than under v2 and the two are not directly comparable.
- **F1** changed too, via a separate fix: the optimised artifact is now accepted only when it
  preserves correctness, so F1 is reported for whichever artifact was accepted (§3.3).

F1 and F2 remain internally valid; they are simply not comparable with pre-v3.1 runs, including the
7B results in Reports 11, 12 and 14.

---

## 2. What Changed

### 2.1 v2 (previous): single-pass measurement

```
_CodeflawsTestRunner.__call__(source)
  compile source once
  for each held-out test:
    _energy_window():           <- RAPL opens/closes here, every call, every test
      run test, compare output
    if passed: record energy
  F3 = mean(energy over passing tests)
```

This ran for *every* verification call in the pipeline — the initial repaired-code check, every
repair retry, the optimized-code check, every optimize retry — even though only the **last**
verification's result is ever used by `evaluate_fitness()`. On the n50 training set (`rep_0` log,
750 candidate evaluations x 50 problems), suite executions broke down as:

| Suite type              | Count   | % of total |
|--------------------------|---------|------------|
| Initial verification     | 37,500  | 17.6%      |
| Repair-retry verification| 135,558 | 63.5%      |
| Optimizer verification   | 33,015  | 15.5%      |
| Optimizer-retry verification | 7,494 | 3.5%    |
| **Total**                | 213,567 | 100%       |

82.5% of all suite executions (repair-retry + optimizer-retry) measured energy that was *never
used* — it either fed into a retry decision and was then superseded, or belonged to a candidate
whose energy was ultimately read from a later verification call instead.

### 2.2 v3 (this report): two-phase deferred measurement

```
Phase A (per problem, concurrent across N_WORKERS):
  analyze -> repair -> diagnostic(repaired)         <- compile+test, NO RAPL, NO gate
    -> retry (diagnostic again, no RAPL) ...
  -> optimize -> diagnostic(optimized) -> retry ...
  Settles: final source path + which specific tests passed (passing_test_ids)

  [barrier: ThreadPoolExecutor .with block exits — every Phase A worker has joined]

Phase B (per problem, SERIAL, one problem at a time):
  measure_energy(final_source, passing_test_ids):
    recompile final_source once
    for each id in passing_test_ids:
      _energy_window():          <- RAPL opens/closes here, once per passing test, ONCE total
        rerun that one test, sanity-check it still passes
      if it reproduces: record energy
    F3 = mean(energy over the tests that reproduced)
```

Energy is measured exactly once per problem per candidate, over exactly the tests already known to
pass — never over a retry attempt, and never over a test known to fail or hang.

### 2.3 Why this needs a true barrier, not (only) a lock

The `_InferenceVerifyGate` readers-writer lock (still used by `agents.py`'s legacy path and
`benchmarks/base.py`'s `StdinStdoutRunner`/`CppAssertRunner`) protects the RAPL window against
concurrent *registered* operations (Ollama inference, other verify() calls). It does **not**
protect against `gcc` invocations, which were never wrapped in the gate, or any other unregistered
local CPU activity. Phase B's design avoids this by construction: it only begins after
`ThreadPoolExecutor.__exit__` has fully joined every Phase A worker thread, so nothing else in the
process — inference, compilation, or another problem's diagnostic testing — is running at all
during Phase B. `measure_energy()` still calls through `_energy_window()` (which acquires the gate
internally), but this is redundant overhead there, not a safety requirement: the barrier is what
makes the window valid.

**This does not extend to true machine-level isolation.** RAPL is a whole-package counter; other
processes, other users, cron jobs, or monitoring tools on the same host can still contribute energy
during Phase B. Final measurement runs should use an otherwise-idle, dedicated host, with no
interactive work (including code review, compilation, or ad hoc testing) performed on it while a
run is in progress.

---

## 3. Failure-mode handling

> **Superseded — read §3.2 first.** This section and §3.1 describe *per-test* energy windows, which
> were later found to sit below RAPL's resolution. They are kept because the reasoning trail matters
> (and because §3.1's premise turned out to be inverted), but the current behaviour is §3.2's.

- **Total measurement failure** (recompile fails after retry, RAPL fails, or zero of the
  `passing_test_ids` reproduce on rerun): `measure_energy()` returns `mean_energy_mj=None`.
  `run_pipeline_phase_b()` must treat this as `energy=inf` (excluded from the feasible aggregate,
  matching the existing compile-failure convention) — **never** as `0.0`, which would look like an
  unbeatable energy score to NSGA-II's minimisation. An earlier draft of this fix had this bug; it
  was caught before the corrected `rep_1` run reached generation 1 and fixed before any data was
  produced under it.
- **Non-positive samples**: an individual per-test RAPL reading of exactly `0.0` mJ (or negative) is
  rejected and treated as a reproduction failure for that specific test. A genuinely executing
  process cannot consume exactly zero package energy — idle background draw alone is nonzero over
  any real time window — so an exact `0.0` reading is a measurement artifact, not a legitimate
  fast-and-cheap execution.
- **Partial reproduction** (some but not all `passing_test_ids` reproduce): the non-reproducing
  tests are excluded individually and `mean_energy_mj` is the mean over whichever did reproduce.
  `matched_test_ids`/`mismatched_test_ids` are always recorded in the per-problem verification JSON
  (`phase_b_measurement` key) and a warning is printed whenever any exclusion happens, so this is
  auditable, never silent — see §3.1 for why an all-or-nothing policy was tried first and reverted.

### 3.1 Why partial reproduction is handled leniently, not as total invalidation

An earlier version of this fix invalidated the WHOLE measurement whenever ANY diagnostic-passing
test failed to reproduce, on the reasoning that the tests likeliest to fail to reproduce are the
ones that ran closest to `run_timeout` the first time — plausibly the highest-energy tests among
the passing set — so averaging over the survivors risked a systematic downward bias (dropping the
expensive tail skews the mean toward the cheap end of the distribution). This is a real,
mechanistically sound concern, and remains true as a description of the *direction* of the bias.

It was measured against the real `pop50_gen15_n50_v3` run (15 generations, pop 50, 50 training
problems, 37,500 total (candidate, problem) evaluations) and found to invalidate **83% of all Phase
B measurement attempts** (27,397 of 33,084) — only 17% (5,687) produced a usable energy value. The
typical mismatch, sampled from the actual verification JSONs, was small relative to the test suite
(e.g. 2-4 tests out of 40+), not wholesale failure — consistent with the hypothesised mechanism
(occasional timeout drift from residual thermal/frequency state after Phase A's preceding
sustained multi-threaded workload) rather than a deeper systemic problem, but the *rate* at which
it occurs, compounded across dozens of tests per problem (median 32 positive tests per Codeflaws
problem — see §1), makes "require 100% reproduction" prohibitively expensive in practice: even a
per-test reproduction-failure probability as low as ~5-10% compounds, across 20-40 tests, to a very
high probability that *some* test in the set fails to reproduce.

Weighed against that measured cost, the all-or-nothing policy was reverted (§3 above describes the
final, lenient behaviour). An estimator that is formally unbiased but usable only 17% of the time
serves NSGA-II worse than one with a small, bounded bias (proportional to the excluded fraction,
typically under 10% of a problem's tests) that is usable almost always — the search cannot
meaningfully distinguish candidates on F3 from a signal that is absent 83% of the time either way.
The exclusion is bounded and transparent (recorded per-problem, never silent), which was judged the
better trade for this dissertation's purposes. This reasoning was not derived in the abstract; it
required actually running the full search once under the strict policy to observe the real
invalidation rate (§6) before revising it.

**Consequence for existing data:** `pop50_gen15_n50_v3/rep_0` (the run that surfaced this finding)
was produced under the strict (now-reverted) policy — its F3 values reflect only the 17% of
(candidate, problem) pairs that achieved 100% reproduction, not the lenient policy described above.
It is not directly comparable to any later run produced under the code as it now stands, and should
either be re-run under the current lenient policy or clearly labelled as strict-policy data if kept.
`protocol.json` records `git_commit`, so which policy was in effect for a given rep is always
determinable from its commit hash, but the two are not interchangeable data.

---

## 3.2 Correction (v3.1): per-test windows were below RAPL's resolution

Everything in §3 and §3.1 above describes **per-test** energy windows. That design was
subsequently found to be measuring at the hardware's resolution limit, and has been replaced by a
single whole-batch window. The trigger was a supervisor question — Codeflaws programs are
deterministic C run on fixed inputs, so why would tests be "flaky" at all?

**They were not.** Every recorded mismatch in a full smoke run was `invalid_sample` (a RAPL
reading of ≤ 0) — **zero** `output_mismatch`, `nonzero_exit`, or `timeout`. No Codeflaws test ever
produced wrong output. The "non-reproduction" that §3.1's entire strict-vs-lenient debate was built
on was a measurement artifact, not program behaviour. Note this also means §3.1's premise was
backwards: the tests failing to yield a sample are the *fastest* ones, not the slowest, so
excluding them biases F3 **up**, not down.

**Measured on an idle host:**

| Quantity | Value |
| --- | --- |
| RAPL counter update period | **0.996 ms** |
| Median Codeflaws test execution | **0.596 ms** (incl. process spawn) |
| → executions per counter update | **~0.6** |

A per-test window therefore spans less than one counter tick and is quantised to roughly "0 or one
tick". Empirically that gives **4–17%** run-to-run spread; one window around the whole batch gives
**2–3%**.

**Two defects followed from this, both now fixed:**

1. **F2 was being decided by energy measurability.** `measure_energy()` checked the energy sample
   *before* comparing output and `continue`d, so a test that ran correctly but too fast to resolve
   was recorded as a reproduction failure — and, once Phase B began revising correctness, demoted
   F2. A correct program was losing correctness for being fast. Output comparison now always runs
   and is wholly independent of the energy path.
2. **F3 was quantisation-limited.** Replaced by one `_energy_window()` around the whole batch of
   passing tests, divided by executions performed. This is safe here though it was not in v1: v1
   wrapped the whole suite *including failing tests*, so a looping repair dumped
   `N_tests × run_timeout` seconds into one reading (§2.7 of Report 09). Phase B only reruns tests
   that already passed `diagnostic()`, and a timeout is recorded there as a failure, so it can never
   enter `passing_test_ids` — the v1 pathology cannot recur.

Small batches are repeated to clear `_MIN_WINDOW_EXECS = 30` executions. Without this a candidate
passing few tests would be measured at the resolution floor, making F3's *measurability* correlate
with F2 and systematically dropping low-correctness candidates out of the F1/F3 aggregates
(observed: a candidate with 1 passing test returned `energy=inf` despite reproducing correctly).

### Why no baseline subtraction

~60% of a reading is constant process-spawn overhead (a no-op binary measures 6.7 mJ/test against
a real solution's 11.0). Subtracting a measured no-op baseline was considered and **rejected on
evidence**: that constant already cancels when NSGA-II compares two candidates, so subtracting an
estimate of it recovers no signal while injecting the baseline's own variance. Over 15 interleaved
repeats, comparing an efficient repair against a deliberately less efficient one:

| | signal (Δmean) | pooled noise | **discrimination (Δmean / pooled sd)** |
| --- | --- | --- | --- |
| whole-batch, raw | 2.828 mJ/test | 0.299 | **9.45** |
| minus no-op baseline | 2.828 mJ/test (identical) | 0.510 | **5.55 (−41%)** |

The last column is an instrument-sensitivity ratio: how many pooled standard
deviations of measurement noise separate two configurations of the *measurement
procedure*. It is not a between-arm effect size, and the statistical policy of
dissertation 3.11 — Mann-Whitney U with Cliff's delta and Vargha-Delaney A12 —
governs comparisons between search arms, not instrument calibration.

If an interpretable absolute figure is wanted for the write-up, report the no-op baseline once as a
documented host constant rather than subtracting it per measurement.

**Sensitivity, corrected.** An earlier draft of this report warned that F3 could only distinguish
coarse differences. With whole-batch windows the pooled sd is ~0.30 mJ/test against a program
compute contribution of ~3.1 mJ, so differences of roughly **20% of a program's own energy** are
detectable at 2 sd. That is considerably better than the earlier caveat implied.

---

## 3.3 The optimise stage was silently damaging correctness

The optimiser contract states its output *"must produce identical outputs to the verified code for
all inputs"*. The pipeline never enforced that — it only checked that the optimised code
**compiled**. An optimisation that compiled while breaking tests was accepted, and its damaged
correctness became the candidate's F2.

Measured over the ~184k repair→optimise transitions already recorded on disk:

| | optimised failed to compile | correctness **worse** | unchanged | better |
| --- | --- | --- | --- | --- |
| 1.5B (n=168,501) | 2.0% | **21.5%** | 74.8% | 1.8% |
| 7B (n=15,268) | 4.4% | **15.5%** | 76.5% | 3.6% |

So there was a guard against the 2% failure mode and none against the 21.5% one. The damage is not
marginal: when correctness dropped, the **median loss was ~51 pp (1.5B) / ~32 pp (7B)**, only ~4%
of drops were a single lost test, and **21.4% of fully-correct 1.5B repairs were destroyed
outright** (12.6% at 7B). This is a large confound sitting inside the primary correctness objective,
and it affects both models — it is not a small-model artefact.

For balance, the stage does do useful work: it changes the code 100% of the time and reduces F1 in
~25% of cases (mean ΔF1 ≈ −1.0 on a base of ~9, median 0).

**Fix:** the optimised artifact is accepted only if its correctness is at least the repaired
artifact's; otherwise the repaired artifact is kept (ties go to the optimised version, which may
have lower complexity at equal correctness). This makes the stage monotonically non-harmful to F2
while retaining its F1 benefit where safe, and matches standard APR practice of validating a patch
before accepting it. Smoke test: 4 rejections in 20 problems, rescuing repairs at 95.8%, 97.2%,
81.1% and 65.2% that the optimiser had cut to 4.2%, 5.6%, 13.5% and 52.2%.

---

## 3.4 Prompt-example leakage inflates F1 (1.5B only)

`contracts/repairer_contract.md` contains an "### Example of valid output" section holding a
complete compilable C program (a header comment plus an unused `compute(int n)` function).
**The 1.5B model copies it into its output verbatim**; the 7B never does:

| model | generated files | containing the copied example | rate |
| --- | --- | --- | --- |
| qwen2.5-coder 1.5B | 1,197 | 224 | **~19%** |
| qwen2.5-coder 7B | 458 | **0** | **0%** |

Within the 1.5B the optimise stage is consistently worse than repair (15–41% vs 0–20%); the
optimiser contract carries the same example and also passes through whatever repair injected.

**Effect on F1:** `lizard` sums cyclomatic complexity over all functions, so the dead `compute()`
adds +2.15 on affected files — **37% of their F1** — and ~4.5% of total 1.5B F1 mass.

**Decision: the contracts are left unchanged.** The 7B follows the identical contract with zero
leakage across 458 files, so the contract is demonstrably followable and this is a model-capability
limitation, not a prompt defect. Rewriting the example to accommodate the weaker model would change
the stimulus in a cross-model comparison and delete the evidence of a genuine finding — a 1.5B model
cannot reliably distinguish "an example of the output *format*" from "content to reproduce". The F1
contamination is instead disclosed and quantified above; computing F1 over reachable functions only
would remove it from the measurement without touching the prompt, should that be preferred later.

---

## 3.5 Test-suite protocol: the official suite, and one quarantined case

### We now follow Codeflaws' own `test-valid.sh`

Every Codeflaws problem ships a `test-valid.sh` enumerating the positive test cases the benchmark
authors intend to be used. Earlier revisions of this adapter **ignored it**, globbing every
`heldout-input-pos*` file on disk instead. Those sets differ, and the official case list is not
contiguous — for `173-A` it runs physical indices `1–17, 19, 21, 22, 27, 29, 31, 32` (24 cases) out
of 38 files present.

Measured across all 3,904 shipped scripts:

| | glob-everything (old) | official suite (now) |
| --- | --- | --- |
| total positive executions | 148,809 | **137,108 (0.921×)** |
| problems where the two differ | — | 1,668 / 3,879 (43%) |
| worst individual case | — | `173-A`: 38 → 24 (1.58×) |

The adapter now parses `test-valid.sh` and restricts the suite to its positive cases, so **F2 is
measured over the same cases as published Codeflaws results**. Only positive cases are taken;
negative (`*-neg*`) tests are adversarial and excluded so F2 matches the published "fix" criterion.
Verified across all 3,904 scripts: none list negative cases in their case statement, and every
referenced positive file exists on disk. Parsed results are cached (0.5 ms warm for the whole
benchmark), so the per-candidate cost is negligible.

**Fallback:** 25 of 3,904 problems have no parseable positive case list. All 25 also have **zero**
physical positive test files, so globbing and the official list agree at zero — the fallback is
equivalent there, and is covered by a regression test.

**This is a protocol change.** F2 under it is not comparable with runs made before it, which used
the larger physical-file suite.

### One test case is quarantined

`173-A-bug-4169928-4169966`, test `16`, is excluded on top of the official filter — see §3.2 for the
diagnosis. The problem's I/O contract is three tokens and 37 of its 38 inputs supply three; this one
supplies two, so the third `scanf()` matches nothing and `s2` is read uninitialised. Confirmed by
valgrind (*"Conditional jump or move depends on uninitialised value(s)"*) and by direct
demonstration: the same binary on the same input returns `1 0` or `0 1` depending only on stack
offset (38 vs 4 of 42 invocations). Codeflaws' own reference solution has the same three `scanf()`
calls, so the expected output `1 0` was itself produced under undefined behaviour — **there is no
correct answer for this case**.

Critically, **adopting the official suite does not remove it**: it is `p16` in `173-A`'s own
`test-valid.sh`. The quarantine is therefore still required, and is applied after the official
filter (38 physical → 24 official → **23 executed**).

`_QUARANTINED_TESTS` is keyed by `(problem_id, test_id)` and pinned to the **SHA-256 of both the
input and expected-output files**. If the benchmark data ever changes the hashes stop matching, the
quarantine is *not* applied, and a warning is printed — so it can never silently suppress a case
other than the one examined. Each entry carries its evidence in the source. Covered by
`tests/test_measurement_protocol.py`, including a test asserting the quarantine is *refused* on
unreviewed data and one asserting the official suite still contains the case (so the quarantine
cannot become silently redundant).

---

## 3.6 Declared limitations (external audit, 28 July)

An external audit of the pipeline raised ten findings. Six were fixed in code (see the
commit `b0964b3` and §3.7). The three below are **not defects to be fixed** — they are
properties of the method that must be declared, and are recorded here so they can be cited
directly in the write-up.

### Passing the same tests does not prove semantic equivalence (patch overfitting)

The optimiser contract asks for identical behaviour on *every valid input*. No runtime guard
can establish that: the guard can only compare behaviour on the finite official suite. The
audit demonstrated this concretely — a repaired program using `sum25 > 2` (subtracting 3) and
an optimised version using `sum25 > 3` (subtracting 4) pass **exactly the same 22/30 official
tests**, yet diverge on the valid input `4 / 25 25 25 100`, where one prints `YES` and the
other `NO`.

**This is not treated as a pipeline defect.** It is *patch overfitting* — the standard and
well-documented limitation of all test-based program repair, and it applies to the repair
stage just as much as the optimise stage. The position taken here is:

> F1, F2 and F3 all describe **the exact artifact the pipeline emitted**, evaluated on the
> official Codeflaws suite. Behavioural equivalence beyond that suite is not verified and is
> not claimed.

The three objectives are internally consistent — complexity is computed from the accepted
source, energy is measured on that same source, and correctness is revised from that same
artifact's test outcomes — so the measurement is coherent even though semantic equivalence is
unprovable. What is withdrawn is any claim that the optimise stage *preserves semantics*; it
is simply a pipeline stage whose output is measured like any other.

### The build differs from Codeflaws' own Makefile

Codeflaws builds in two steps, and `-O2` appears **only at link time**:

```make
CFLAGS=-fno-optimize-sibling-calls -fno-strict-aliasing -fno-asm -std=c99   # no -O2
LDFLAGS=-lm -s -O2
	$(CC) $(CFLAGS) -c $(FILENAME).c -o $(FILENAME).o     # source compiled at -O0
	$(CC) $(FILENAME).o -o $(FILENAME) $(LDFLAGS)         # -O2 only here
```

Without LTO, link-time `-O2` does not optimise already-compiled code, so upstream effectively
evaluates an **O0** binary. This adapter compiles the source at `-O2`. Since the benchmark
demonstrably contains programs with undefined behaviour (§3.5), optimisation level can change
observable output: the audit found `215-A-bug-4160675-4160686` exits normally under the
upstream-style build but segfaults under `-O2` on three tests.

**Decision: keep `-O2` and declare the deviation.** `-O2` is the more defensible protocol for
an *energy* objective — measuring unoptimised binaries would characterise the compiler's
debug output rather than the program. The earlier claim that the adapter "matches the upstream
build exactly" is withdrawn. F2 is therefore not directly comparable with published Codeflaws
numbers on programs whose behaviour is optimisation-sensitive. The compiler version and full
build command should be recorded per run.

### Checkpoint resume is a warm restart, not a continuation

The checkpoint stores the population vectors but **not** pymoo's algorithm or RNG state, and
resuming constructs a fresh NSGA-II for the remaining generations. A resumed run therefore
does not reproduce the corresponding uninterrupted run: the saved parents are re-evaluated
(and, with a stochastic LLM, re-measured to different values), and one offspring-producing
step is lost. Call it a **warm restart**, not a continuation.

Mitigation for the canonical runs: the 5-repetition launcher executes all repetitions in a
single process from generation zero, so no resume occurs on the happy path. If a run is
interrupted, the resumed repetition should be reported as warm-restarted, or discarded and
re-run. §3.7's treatment gate additionally refuses to fold a *changed treatment* into a
resumed repetition.

---

## 3.7 Audit fixes applied

| # | Defect | Correction |
| - | ------ | ---------- |
| 2 | The "exact set" guard only checked for *lost* tests, so an optimisation that made previously-failing tests pass was accepted — out-of-contract, and it credits repair gains to the optimise stage (~2.5k such gains in the corpus). It was also vacuous for runners that do not populate `passing_test_ids`. | Exact set equality; **fails closed** when ids are absent or inconsistent with `tests_passed`. |
| 4 | `run_ga_staged.py` accepted any optimisation that merely **compiled**, so a 10/10 repair could be replaced by a 9/10 optimisation. | Reuses the shared `optimized_preserves_correctness()` predicate. |
| 5 | `evaluate_fitness()` mapped missing energy to `0.0`. F3 is minimised, so that is an unbeatable score — and being *finite* it entered the feasible mean (verified: it drags an 11.00 mean to 7.33). | Returns `inf`, which the existing `np.isfinite` feasibility filter excludes — precisely why that filter exists. |
| 6 | Output comparison was **stricter** than Codeflaws' own checker, which pipes through `sed '/^$/d'` and `'s/^[ \t]*//'` then diffs `--ignore-trailing-space`. Plain `.strip()` equality can fail a submission the benchmark accepts, biasing F2 downward. | Normalised identically, in both the diagnostic and the Phase B reproduction check. |
| 7 | Resume compatibility compared only commit and protocol label, so a changed contract, model, pop, gen, train-size or worker count under the same commit was silently folded into one repetition. | Compares the whole treatment and reports which keys differ. |
| 8 | `training_baseline_v2.py` imported the workflow **before** setting `AGENT_MODEL`; `agents.py` binds `DEFAULT_MODEL` at import, so agents silently ran on the **7B** fallback while the run was labelled 1.5B. | Bound before the import, as `run_nsga2.py` already did. |
| 10 | A missing contract file silently became an **empty** system prompt — an invisible change of treatment that still produces plausible numbers. Timeouts were mis-recorded: the manifest advertised `compile_timeout=40` while the runner's real default was 30. | Contracts fail closed; timeouts single-sourced from the adapter constants. |

Regression tests: 23 → 27. The audit also confirmed that every prior smoke test used
`--baseline`, which never enters `run_search()` — so pymoo, checkpointing and
`protocol.json` writing had gone unexercised across all of these changes. The GA path has
since been run directly and produced a correct `protocol.json` and checkpoint.

---

## 4. This is a new protocol, not a drop-in replacement

`run_pipeline()` (the single-shot wrapper used by `N_WORKERS=1` and by the `*_nreps.py` baseline
scripts) still exists and is interface-compatible, but it is **not** measurement-compatible with
v2: `run_pipeline_phase_a()` always prefers `_CodeflawsTestRunner.diagnostic()` when available,
regardless of `N_WORKERS`. This means even sequential (`N_WORKERS=1`) Codeflaws runs now measure
energy via the same two-execution (diagnose once, remeasure once) protocol described in §2.2,
rather than the old single-execution protocol — an intentional, deliberate change, but one that
means **v2 and v3 results are not a clean same-protocol comparison**, independent of `N_WORKERS`.

Concretely: **`rep_0` of the n50 search (pre-existing, v2 protocol, sequential) and any `rep_1`
produced under this code (v3 protocol) are not directly comparable as "two repetitions of the same
measurement methodology, differing only in seed."** They are two repetitions of the same *search*
(same policy space, same training subset, same model), but under different F3 measurement
protocols. A new baseline or held-out evaluation intended to compare against v3 data should be run
under v3, not v2.

The shared run-label-level `manifest.json` is insufficient to distinguish this, since it is
overwritten by whichever repetition ran most recently — a run-label whose repetitions span a
protocol change would otherwise have no durable record of which rep used which protocol. This is
fixed: `run_search()` now writes a `protocol.json` into each `rep_N/` directory it actually runs
(never retroactively into an already-completed, skipped rep), recording the protocol name, this
report's path, the git commit hash, seed, model, and pop/gen/train-size. `run_baseline()` records
the same protocol name and commit hash directly in `baseline_default_policy.json`. Neither is
written for repetitions that predate this code (e.g. `rep_0`), which correctly has no
`protocol.json` and should be assumed v2 in its absence.

---

## 5. Out of scope (considered, deliberately not done here)

The following were raised during review as further potential improvements but are methodology /
search-space changes, not measurement-timing fixes, and need separate explicit sign-off:

- Skip optimizing repairs that only partially pass (the optimizer contract assumes already-correct
  input; the implementation currently optimizes anything that compiles).
- Recalibrate `run_timeout` (currently 3s) down using the reference implementations' empirical
  runtime, rather than a fixed ceiling.
- Reconsider the `pi_v.retry`/`pi_o.retry` search space range if high retry counts rarely recover
  additional passing tests.

---

## 6. Validation performed

- Real-timestamp trace confirming every Phase B energy measurement starts at or after the last
  Phase A inference call ends (12/12, zero violations, 0.012s gap for the first).
- 20-problem `--baseline` smoke test: 290.1s even under an unlucky heavy-retry draw (11/20
  feasible), versus ~390-451s for the v2-protocol-with-fixed-locking design at similar retry burden.
- Full multi-generation GA run: `pop50_gen15_n50_v3/rep_0` (seed 42, `N_WORKERS=4`, pop 50, gen 15,
  50 training problems), launched 2026-07-24 22:05 BST, completed 2026-07-26 17:53 BST — 43.8 hours,
  zero crashes or retries, 27 feasible Pareto solutions, 0 infeasible dropped from the reported
  front. Run under the strict (since-reverted) partial-reproduction policy — see the "Consequence
  for existing data" note in §3.1; its F3 values reflect only the 17% success rate of that policy,
  not the current lenient one, and should be treated as a separate, lower-yield dataset rather than
  representative of the code as it now stands.
