# GA replication fronts — `pop50_gen15_n50_v4`, all five repetitions

Training Pareto fronts from the NSGA-II replication run, published so a second machine can
perform the held-out evaluation without re-running the 36 h searches.

| rep | seed | policies | protocol | commit |
| --- | --- | --- | --- | --- |
| rep_0 | 42 | 38 | `codeflaws_two_phase_v3.1` | `f50a3ef` |
| rep_1 | 43 | 45 | `codeflaws_two_phase_v3.1` | `e0302db` |
| rep_2 | 44 | 50 | `codeflaws_two_phase_v3.1` | `e0302db` |
| rep_3 | 45 | 46 | `codeflaws_two_phase_v3.1` | `c979e8b` |
| rep_4 | 46 | 43 | `codeflaws_two_phase_v3.1` | `e96602b` |

All five repetitions are complete: 226.6 h of search over 9.4 days, exit code 0, no model-server
failures.

## Between-run variance, and why every repetition must be evaluated

Size-matched training-front hypervolume on F1+F2, the two machine-independent objectives:

| rep | 0 | 1 | 2 | 3 | 4 | mean | sd | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HV | 0.8235 | 0.8558 | 0.8859 | **0.7453** | 0.8722 | 0.8365 | 0.0561 | 6.7% |

The range spans 17% of the mean, and rep_3 falls well below the rest.

**Against the completed random-search arm, NSGA-II leads by +0.0788 (1.41 random-search
standard deviations): Mann-Whitney U = 22, p = 0.0556; Cliff's delta = +0.76, A12 = 0.88.** An earlier version of this
note reported the two arms as indistinguishable (p = 0.30). That reading compared five NSGA-II
repetitions against a single random-search run and did not survive replication; see
`../random_search/README.md` for the corrected comparison and the sequence of intermediate
estimates. This quantity is in-sample and two-objective, and is not the held-out
three-objective comparison reported in `docs/reports/replicated_three_arm_comparison.md`.

**rep_3 is not a corrupted repetition and must not be discarded.** It is the one execution in
five that falls below the random-search mean, which is a property of the method rather than a
defect in that run. Excluding it would inflate the arm's apparent advantage, which is precisely
the selection bias this note exists to prevent. All five fronts are evaluated on held-out data.

The estimate stabilised only at n ≥ 4. At n=2 the between-arm gaps appeared unattributable, at
n=3 they appeared significant at approximately 1.7 sd, and at n=4–5 they are not significant. A
related artefact dissolved on the same schedule: the correlation between front size and
hypervolume was r = 1.000 over the first three repetitions and r = 0.273 (p = 0.73) over four.

All five repetitions share one training subset, selected once from the base seed outside the
repetition loop (`run_nsga2.py:765`), so `--seed-increment 1` varies only the search RNG — the
initial population and the SBX/PM draws. The training problems are identical across reps:

```
train_ids sha256 = bab4ff5403a3b42664578bae78a4b860cd62021a409afbacb28261256022083a
```

`manifest.json` is the run-level manifest containing those `train_ids`. It is required by
`evaluate_heldout.py:532`, which reads it to derive the held-out split.

---

## Running the held-out evaluation on another machine

### 1. Reconstruct the run directory

The evaluation script resolves paths under `outputs/ga/<benchmark>/<model>/<run_label>/`. Only
the manifest is needed there — the fronts are passed explicitly.

```bash
RUN=outputs/ga/codeflaws/qwen2.5-coder_1.5b/pop50_gen15_n50_v4
mkdir -p "$RUN"
cp results/nsga2/manifest.json "$RUN/manifest.json"
```

If `outputs/` is a symlink to scratch on that machine too, write through the symlink as above
rather than committing anything under `outputs/` — tracked files at that path replace the
symlink with a real directory on the next pull.

### 2. Evaluate each rep's front

One invocation per repetition, each with its own `--out-tag` so they do not overwrite one
another. `--front-file` supplies the policies; the held-out *problems* still derive from the
manifest's `train_ids`, which is what keeps the split identical across reps.

```bash
for i in 0 1 2 3 4; do
  .venv/bin/python scripts/evaluate_heldout.py \
      --benchmark codeflaws \
      --run-labels pop50_gen15_n50_v4 \
      --front-file "results/nsga2/rep_$i/pareto_solutions.json" \
      --out-tag "heldout_rep${i}_n100_seed0" \
      --n-heldout 100 --heldout-seed 0 \
      --workers 4 \
      --energy-backend rapl_sysfs
done
```

### 3. Verify the split matches

All five repetitions must be evaluated on the *same* 100 problems, or their results cannot be
pooled. Assert it rather than assume it:

```bash
--expect-heldout-sha256 365503a2457502e94a12ebe7c818b4cd30d335549a7682c7a14dac575b80b6f7
```

That digest is the held-out set used by the GA, RS and MOBO arms on the primary machine. If the
second machine's Codeflaws checkout resolves a different problem set, the flag aborts rather than
producing incomparable numbers.

---

## Energy is not portable across machines

**F1 and F2 transfer; F3 does not.** Cyclomatic complexity and test pass rate are properties of
the produced source. Energy is measured through RAPL at the `package-0` domain and is
proportional to elapsed execution time, so it reflects the host's performance.

This was observed directly on the primary machine: a reboot between rep_0 and rep_1 produced a
1.55× slowdown and a corresponding **1.68×** increase in measured energy, while F1 and F2
replicated to within 0.2% (ratios 1.0019 and 0.9998). A different machine will differ by at least
as much.

Consequences for the second machine:

- Its F3 values form their own measurement epoch and must not be pooled with the primary
  machine's, nor compared against the primary machine's baseline.
- **Run a default-policy baseline on that machine** before drawing any F3 conclusion. Ten
  repetitions over the same 100 held-out problems costs roughly 3 h and yields both the local
  noise floor and, against the primary machine's baseline, a median-based scale factor relating
  the two hosts.
- Report F3 medians rather than means. Per-rep energy on the primary machine has a 28.7%
  coefficient of variation under the mean against 2.3% under the median; the distribution is
  strongly right-skewed and a small number of pathological problems dominate the mean.

F1/F2 comparisons between the two machines are valid without correction.

---

## Cost

Roughly 0.25 h per policy at 100 problems, so 38 + 45 + 50 + 46 + 43 = 222 policies is about
**56 h** serially on hardware comparable to the primary machine. The evaluation is resumable per policy:
each completed policy writes a checkpoint, and re-invoking the same command skips it.

Do not run anything else on the machine during the evaluation. RAPL measures the whole CPU
package, so any concurrent workload contaminates F3 for every policy being measured.


## Search trajectories — `rep_N/intermediate_solutions.json`

Published so anytime-convergence statistics can be computed without the run outputs. One file
per repetition, 750 records each.

**Read the time axis carefully: this is not per-evaluation history.** The generation callback
records `algorithm.pop` — the population *surviving selection* at the end of each generation — so
each file holds 15 generations x 50 survivors. Offspring that were evaluated and then rejected by
selection are absent, and a survivor persisting across generations is recorded once per
generation it survives. In rep_0 the 750 records comprise only **190 distinct policies**.

Consequences:

- There is **no evaluation index**, and the order within a generation is not the order those
  candidates were evaluated in.
- Generation *g* corresponds to **50·g evaluations consumed**, since NSGA-II evaluates 50
  offspring per generation and never re-evaluates survivors. Anytime curves are therefore valid
  **at generation boundaries** — checkpoints at 50, 100, ..., 750 evaluations — and only there.
- The cumulative archive at generation *g* is the union of records from generations 1..g. That
  archive is a subset of what was actually evaluated, since rejected offspring never enter it.

Per record: `generation` (1..15), `policy`, `feasible`, and `fitness` with `cl`, `correctness`
and `energy`.

For a true per-evaluation axis use the random-search arm, whose
`rep_N/evaluations_summary.json` carries a genuine evaluation index. The two arms are not
directly comparable at finer than 50-evaluation granularity.
