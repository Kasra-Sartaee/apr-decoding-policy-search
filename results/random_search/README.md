# Random-search replication fronts — `rs_pop50_gen15_n50_v4`

Training Pareto fronts from the budget-matched random-search control, published so the held-out
evaluation can be run elsewhere without repeating the ~57 h searches.

| rep | seed | policies | epoch | status |
| --- | --- | --- | --- | --- |
| rep_0 | 99 | 34 | 1 (pre-reboot) | complete |
| rep_1 | 100 | 44 | 2 (post-reboot) | complete |
| rep_2 | 101 | 33 | 2 | complete |
| rep_3 | 102 | 42 | 2 | complete |
| rep_4 | 103 | 45 | 2 | complete |

Each repetition draws 750 policies uniformly from the same 11-dimensional space and evaluates them
on the 50 training problems taken from the matched GA run's manifest, so the training set is
identical across repetitions and identical to the GA arm's:

```
train_ids sha256 = bab4ff5403a3b42664578bae78a4b860cd62021a409afbacb28261256022083a
```

Only the sampling seed varies between repetitions. There is no feedback from evaluations to
sampling — all 750 draws are fixed before the first evaluation runs — so unlike the GA, a
repetition's exploration is entirely determined by its seed.

## Budget

750 evaluations per repetition, matching the GA arm's pop 50 × 15 generations. That NSGA-II
consumes 750 rather than some smaller figure is verified three ways in
[the budget-matching report](../../docs/reports/budget_matching.md): pymoo's own `n_eval` counter, the
analyzer call log, and pymoo's duplicate-elimination behaviour. The GA's 750 evaluations comprise
approximately 745 distinct policies (0.68% duplication from operator pass-through), against 750
distinct here, since uniform draws over a continuous space do not repeat.

## Epochs — F3 is not comparable across the boundary

rep_0 was measured before a host reboot, reps 1–4 after. Energy is RAPL package energy and
therefore proportional to elapsed execution time, which the reboot changed:

| | rep_0 | rep_1 |
| --- | --- | --- |
| mean correctness | 0.5372 | 0.5386 |
| mean complexity | 9.352 | 9.358 |
| **median energy** | **11.17** | **18.54** |

F1 and F2 replicate to within 0.3%, as independent draws from the same space should. F3 differs by
1.66×, matching the 1.68× measured independently on the GA arm. **Pool F1 and F2 across all
repetitions; pool F3 only within reps 1–4.**

## Between-run variance, and why it is lower than the GA's

Size-matched (n=33) training-front hypervolume on F1+F2, both arms at five repetitions:

| rep | 0 | 1 | 2 | 3 | 4 | mean | sd | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSGA-II | 0.8235 | 0.8558 | 0.8859 | 0.7453 | 0.8722 | 0.8365 | 0.0561 | 6.7% |
| random search | 0.8071 | 0.7670 | 0.7690 | 0.7362 | 0.7091 | 0.7577 | 0.0370 | 4.9% |

**NSGA-II leads by +0.0788, or 1.41 random-search standard deviations. Mann-Whitney U = 22,
p = 0.0556; Cliff's delta = +0.76, Vargha-Delaney A12 = 0.88.** No random-search run exceeded
the NSGA-II mean and only one NSGA-II run fell below the random-search mean, so the effect is
large by the conventional threshold while the rank test sits just outside significance: with
five observations per group the smallest attainable two-sided p is 0.0079, and the test has
little power. Mann-Whitney with Cliff's delta is the reportable pair under the statistical
policy of dissertation 3.11; the parametric alternative is not used, here or elsewhere.

This result required replication to detect. As repetitions accumulated the estimate moved
monotonically — 0.35 sd (p = 0.30) with NSGA-II at n=4 against a single random-search run, then
0.99 sd (p = 0.10), 1.23 sd (p = 0.054) and finally 1.41 sd (p = 0.034). Every intermediate
reading was a null, and every one was underpowered. The earlier conclusion that the two arms were
indistinguishable was an artefact of comparing five NSGA-II runs against one random-search run.

**Scope.** This is training-front hypervolume over F1 and F2 only — an in-sample, two-objective
quantity. It is not the held-out three-objective comparison, and nothing here establishes that the
advantage transfers to unseen problems.

Random search remains the less variable arm (standard deviation ratio 1.52x, and 1.5-2x throughout
as repetitions accumulated). The mechanism is structural: all 750 draws are fixed before the first
evaluation, so measurement noise changes which points reach the front but not which are sampled,
whereas NSGA-II's noise enters selection and compounds across 15 generations.

Front sizes vary substantially in both arms — NSGA-II 38/45/50/46/43, random search 34/44/33/42/45
— so the cardinality differences in the original single-run three-arm comparison (38 / 34 / 48)
were ordinary run-to-run variation rather than a property of the methods. Compare fronts only after
size matching.

## Held-out evaluation

All five fronts were evaluated on the 100 held-out problems, on the evaluation host, under
`rapl_sysfs`. The results are reported in
[`docs/reports/replicated_three_arm_comparison.md`](../../docs/reports/replicated_three_arm_comparison.md);
the per-policy records are part of the separate data release, not this repository.

A host running the held-out evaluation needs its own default-policy baseline before any F3
claim, since energy does not port across hosts.


## Search trajectories — `rep_N/evaluations_summary.json`

Published so anytime-convergence statistics can be computed without the run outputs. One file per
repetition: a JSON list of 750 objects, one per candidate, **in the order the search evaluated
them**.

```json
{"index": 0, "cl": 9.31, "correctness": 0.54, "energy": 18.2, "feasible": true}
```

`index` runs 0..749 with no gaps and is the genuine evaluation order — verified by confirming
that each repetition's `evaluated_at` timestamps are monotonic in index order. Values are the
per-candidate aggregate fitness over the 50 training problems, taken from
`evaluations/cand_NNNNN.json`; the per-problem breakdowns are not reproduced here (750 files per
repetition is too many to track).

Random search fixes all 750 draws before the first evaluation, so `index` reflects sampling
order rather than any adaptive process: the prefix of length *n* is a valid *n*-evaluation random
search, and truncating the series answers "what would this arm have found on a smaller budget".

`energy` is `null` where a candidate produced no valid measurement. Treat those as infeasible
rather than as zero, which a minimiser would read as an unbeatable score. Energy is comparable
within an epoch only: rep_0 predates a host reboot, reps 1-4 follow it (see above).

The NSGA-II arm's `intermediate_solutions.json` has a **different granularity** — it records
surviving populations per generation, not evaluations — so cross-arm anytime comparison is valid
only at 50-evaluation checkpoints. See `../nsga2/README.md`.
