# MOBO replication fronts — `mobo_pop50_gen15_n50_v4`

Training Pareto fronts from the multi-objective Bayesian optimisation arm, replicated to match
the five NSGA-II and five random-search repetitions.

| rep | seed | policies | epoch | status |
| --- | --- | --- | --- | --- |
| rep_0 | 123 | 48 | 1 (pre-reboot) | complete |
| rep_1 | 124 | 50 | 2 (post-reboot) | complete |
| rep_2 | 125 | 59 | 2 | complete |
| rep_3 | 126 | 45 | 2 | complete |
| rep_4 | 127 | 43 | 2 | complete |

Each repetition runs qLogNEHVI over per-objective `SingleTaskGP` surrogates (BoTorch): 50 uniform
random initial draws, then 140 acquisition steps of q=5, for 750 evaluations total. The seed
governs both the initial design and the acquisition optimiser's restarts. Training problems come
from the matched GA run's manifest and are identical across all repetitions and all three arms:

```
train_ids sha256 = bab4ff5403a3b42664578bae78a4b860cd62021a409afbacb28261256022083a
```

All five repetitions are complete, exit code 0, with no model-server failures over the arm.

All five fronts were evaluated on the 100 held-out problems of both benchmarks, on the
evaluation host, under `rapl_sysfs`. The results are reported in
[`docs/reports/replicated_three_arm_comparison.md`](../../docs/reports/replicated_three_arm_comparison.md);
the per-policy records are part of the separate data release, not this repository.

## Interpreter

This arm runs under a separate interpreter with `botorch` installed (`botorch 0.18.1`, `gpytorch 1.15.2`,
`torch 2.13.0+cpu`) rather than the project virtualenv, because torch and botorch cannot fit
alongside it under a 10 GB home quota. That environment also carries `requirements.txt`, so
`run_nsga2` and the benchmark adapters import normally, and the measurement path is shared with the
other arms — every candidate is scored by `run_nsga2._mean_fitness_over()`. The interpreter
difference therefore cannot affect how candidates are evaluated, only how they are proposed.

## Epochs

rep_0 was measured before a host reboot, reps 1–4 after. Energy is RAPL package energy and
proportional to elapsed execution time, which the reboot changed by a factor of about 1.68 on the
NSGA-II arm and 1.66 on random search, while complexity and correctness replicated to within 0.3%.
**Pool F1 and F2 across all repetitions; pool F3 only within reps 1–4.**

rep_4 spans a *second* host reboot (2 September 07:18, at candidate 394 of 750), which raised the
possibility that this repetition was internally inconsistent on F3 and would have to be excluded
from energy pooling. It was checked directly against its own per-candidate records:

| segment | n | median F3 |
| --- | --- | --- |
| before 2 Sep 07:18 | 394 | 19.33 mJ |
| after | 356 | 18.93 mJ |

The ratio is **0.979**, a 2.1% shift against the 68% shift produced by the earlier reboot. rep_4
is therefore internally consistent and reps 1–4 remain poolable on F3 as stated above.

The two reboots differing this much is itself worth recording: a reboot is not automatically an
epoch boundary. What matters is whether the machine comes back in a different performance state,
which has to be measured rather than assumed — in both directions. Treating every reboot as a
boundary would have discarded this repetition's energy data for no reason.

## Front sizes

48, 50, 59, 45 and 43, against NSGA-II's 38/45/50/46/43 and random search's 34/44/33/42/45.

rep_2's 59 briefly looked notable — the largest front any arm had produced, exceeding the whole
NSGA-II range. reps 3 and 4 close at 45 and 43, putting this arm's spread at 43–59 against
NSGA-II's 38–50 and random search's 33–45. Every arm varies by 12 or more policies between its
own repetitions and the three ranges overlap heavily, so front size does not distinguish the
methods.

This is a concrete instance of a pattern that has recurred throughout these replications: a
single repetition suggested a difference that the next repetition removed. It also confirms that
cardinality differences between single runs of different arms carry no information — the
38 / 34 / 48 spread in the original three-arm comparison sits comfortably inside what each arm
does to itself. Compare fronts only after size matching.

## Completed three-arm comparison

With all fifteen fronts available, training-front hypervolume was recomputed over **F1 and F2
only** — the two machine-independent objectives — normalised against the union of all fifteen
fronts and size-matched by subsampling every front to the smallest (n = 33, 400 draws per front).

| arm | mean | sd | CV | range |
| --- | --- | --- | --- | --- |
| NSGA-II | 0.6825 | 0.0477 | 7.0% | 0.6051–0.7264 |
| MOBO | 0.6761 | 0.0499 | 7.4% | 0.6001–0.7360 |
| random search | 0.6210 | 0.0308 | 5.0% | 0.5812–0.6618 |

| contrast | difference in mean hypervolume |
| --- | --- |
| NSGA-II vs random search | +0.0614 |
| MOBO vs random search | +0.0550 |
| NSGA-II vs MOBO | +0.0064 |

Differences only. Between-arm inference follows the statistical policy of dissertation 3.11 —
Mann-Whitney U with Holm-Bonferroni across objectives, effect sizes as Cliff's delta and
Vargha-Delaney A12 — and is reported in `analysis/three_arm_analysis.ipynb` and
`docs/reports/replicated_three_arm_comparison.md`, over the held-out evaluation rather than the
training fronts summarised here.

Kruskal-Wallis across the three arms gives H = 4.94, p = 0.085; the omnibus test does not clear
0.05 and should be reported alongside the pairwise result rather than the pairwise result alone.

Three conclusions. **The two directed methods are indistinguishable** — a difference of 0.0064
against a between-run standard deviation of roughly 0.05. The single-run comparison that put MOBO
ahead on hypervolume measured between-run noise. **Both beat random search by a similar margin in
the same direction**, but only NSGA-II's contrast clears significance, and it clears it narrowly;
MOBO's larger between-run variance is what costs it the threshold, not a smaller effect.
**Random search is the most stable arm** at CV 5.0%, which follows from its having no feedback
loop through which measurement noise can propagate into later proposals.

rep_3 is the weakest repetition for all three methods simultaneously — 0.6051, 0.6015, 0.6001,
the three lowest values in the table and within 0.005 of each other. Since the arms share their
training problems but not their seeds, this points at that repetition's problem interaction rather
than at any property of the search methods.

> These values are **not comparable to the hypervolumes quoted in
> [`../nsga2/README.md`](../nsga2/README.md)**, which normalise against a smaller union
> of fronts and size-match at n = 34. Hypervolume here is relative to the set of fronts entering
> the normalisation, so a figure is meaningless without naming the arms it was computed over.

---

## Search trajectories — `rep_N/intermediate_solutions.json`

Published so anytime-convergence statistics can be computed without the run outputs. One file per
repetition, 750 records each.

**This is a true per-evaluation log, in evaluation order.** One record per candidate submitted for
evaluation, no duplicates, no snapshots. Record *i* is the *i*-th evaluation the arm performed, so
the cumulative archive after *k* records is exactly what the search had seen after *k* evaluations,
and an anytime curve may be cut at any index.

**It is not the same kind of file as the GA arm's.** `nsga2/rep_N/intermediate_solutions.json`
records `algorithm.pop` — the population surviving selection at the end of each generation — so it
holds 15 generation snapshots of 50 survivors, comprising only ~184–204 distinct policies, omits
offspring that were evaluated and then rejected, and is valid only at 50-evaluation boundaries.
This arm's file has no such restrictions. It is directly comparable to the random-search arm's
`evaluations_summary.json`, which is also a genuine per-evaluation record.

| | records | distinct policies | valid at |
| --- | --- | --- | --- |
| MOBO `intermediate_solutions.json` | 750 | **750** | any index |
| RS `evaluations_summary.json` | 750 | 750 | any index |
| GA `intermediate_solutions.json` | 750 | ~190 | multiples of 50 only |

Per record: `policy`, `feasible`, and `fitness` with `cl`, `correctness` and `energy`.

### Structure of the 750

Records **0–49** are the **initial design**: uniform random draws over the search space, with no
model involved. Records **50–749** are **acquisition-driven**: 140 sequential qLogNEHVI steps of
q = 5 over `SingleTaskGP` surrogates refitted at each step.

**Evaluation index 50 is where the model takes over**, and any analysis separating exploration from
model-guided search should cut there. The initial design is identical in kind to a 50-evaluation
random search, which makes index 50 a natural within-arm control.

### Verification

All five repetitions, checked before publication:

| rep | records | distinct | feasible | non-finite objectives | front ⊆ history |
| --- | --- | --- | --- | --- | --- |
| rep_0 | 750 | 750 | 750 | 0 | 48/48 |
| rep_1 | 750 | 750 | 750 | 0 | 50/50 |
| rep_2 | 750 | 750 | 750 | 0 | 59/59 |
| rep_3 | 750 | 750 | 750 | 0 | 45/45 |
| rep_4 | 750 | 750 | 750 | 0 | 43/43 |

Distinct equals total in every repetition, confirming no candidate was evaluated twice. Every
published front policy appears in its own history. Every record carries finite `cl`, `correctness`
and `energy`, and every candidate was feasible.

### What the trajectories show

Anytime hypervolume of the cumulative archive, F1+F2 only, normalised against the union of all five
repetitions' evaluated points:

| evaluations | 50 | 150 | 250 | 450 | 650 | 750 |
| --- | --- | --- | --- | --- | --- | --- |
| mean HV | 0.6384 | 0.7316 | 0.7637 | 0.8040 | 0.8366 | 0.8439 |

The acquisition phase improves hypervolume by **+32.2%** over the initial design, and does so
monotonically in all five repetitions. 58.6% of the total gain is realised by evaluation 250 and
78.0% by evaluation 450, so the curve is still rising at the budget limit but well into diminishing
returns.

One observation that is easy to misread: **mean correctness falls** across the handover, from
0.5417 over the initial design to 0.5302 over the acquisition phase, and in rep_4 the best single
correctness value found in the first 50 random draws is never beaten in the following 700
evaluations. This is not a failure of the acquisition function. qLogNEHVI maximises hypervolume
over all three objectives jointly, not the mean of any one of them, so it spends evaluations
probing low-complexity and low-energy regions that necessarily score worse on correctness. The
hypervolume column above is the measure that reflects what it is optimising. Per-objective means
are the wrong summary for this arm and should not be used to characterise its progress.
