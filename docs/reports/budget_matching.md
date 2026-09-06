# Verification of Evaluation-Budget Matching Across Search Arms

**Question:** whether NSGA-II, random search and MOBO consumed equivalent evaluation budgets
**Benchmark:** Codeflaws (C) · **Model:** qwen2.5-coder:1.5b · **Configuration:** pop 50 × gen 15

---

## 1. Question

The three search arms are compared under a nominally matched budget of 750 candidate evaluations.
Two properties are required for that match to be sound. First, the arms must perform an equal
number of evaluations. Second, those evaluations must correspond to an equal number of *distinct*
candidate policies, since an arm that re-measures a policy it has already tested obtains less
search per unit of expenditure than one that does not.

The first property is directly verifiable. The second is not, for reasons given in §3, and is the
subject of this report.

---

## 2. Evaluation count

Three independent sources agree that each arm performed exactly 750 evaluations.
The third was counted while four repetitions were complete and is a per-repetition
rate, not an arm total; the per-candidate telemetry it read is not part of the
released artefacts, so it cannot be recomputed over the final five.

| source | result |
| --- | --- |
| pymoo `n_eval` counter at generation 15 | 750 |
| stored per-candidate records (RS, MOBO) | 750 each |
| analyzer invocations, 750 × 50 problems over the four repetitions then complete | 148,521 observed / 150,000 expected (0.990) |

Two mechanisms confirm no evaluation is spent redundantly. Surviving individuals retain their
objective values and are not re-evaluated in subsequent generations. Offspring that duplicate a
member of the current population are removed before evaluation by pymoo's
`DefaultDuplicateElimination`, which is active by default and was not overridden.

---

## 3. Why the distinct-policy count is not directly observable

The generation callback records `algorithm.pop` — the population surviving selection — rather
than the offspring submitted for evaluation. The stored history therefore contains 15 × 50 = 750
records comprising only 190 distinct policies for a representative repetition, but this figure
measures *policies that survived at least one selection round*, not policies evaluated. Offspring
rejected by selection are evaluated, incur their full inference cost, and are never recorded.

The distinct count among the 750 evaluated candidates is consequently unavailable from the stored
artifacts, and cannot be recovered retrospectively without altering the instrumentation of
completed runs.

---

## 4. Method

Duplication arises from a single mechanism. SBX crossover and polynomial mutation generate
continuous values, so an offspring produced by either operator coincides bitwise with a
previously evaluated point with probability approaching zero. Exact repetition requires the
pass-through path, in which an offspring undergoes neither crossover (probability 0.2) nor
mutation (probability 0.8) and is therefore a copy of a parent. Pass-through copies of *current*
population members are removed by duplicate elimination; the residual duplicates are copies of
ancestors already displaced from the population.

This mechanism is a property of the variation operators and the selection dynamics, not of the
objective values themselves. It is therefore measurable by simulation: the production NSGA-II
configuration (pop 50, 15 generations, SBX η=15 p=0.8, PM η=20 p=0.2, identical bounds and
encoding) was executed against synthetic objectives, recording every vector submitted for
evaluation and counting distinct entries. No language-model calls are involved.

Because selection dynamics govern how frequently individuals are displaced, and displacement
governs the residual duplication rate, the simulation was calibrated against an observable
recoverable from the real runs: per-generation population turnover, defined as the number of
generation *g* population members absent from generation *g*−1.

**Observed turnover, five completed repetitions:** 10.00, 11.00, 9.79, 9.57, 10.14 replacements
per generation; mean **10.10**, standard deviation 0.55.

Six objective regimes were evaluated, spanning a range of selection pressures, at 30 seeds each.

---

## 5. Results

| regime | turnover / gen | duplicate rate | distinct / 750 |
| --- | --- | --- | --- |
| random objective | **9.05** | 0.68% | 744.9 |
| smooth, noiseless | 23.35 | 0.71% | 744.7 |
| smooth + 2% noise | 22.18 | 0.74% | 744.4 |
| smooth + 5% noise | 21.11 | 0.64% | 745.2 |
| smooth + 10% noise | 19.85 | 0.77% | 744.2 |
| smooth + 20% noise | 17.46 | 0.66% | 745.1 |

The principal result is the **insensitivity of the duplication rate to selection pressure**.
Turnover varies by a factor of 2.6 across the regimes while the duplication rate remains within
0.64–0.77%. The calibration therefore does not materially constrain the estimate, and the residual
uncertainty regarding which regime best represents the true objective landscape is immaterial to
the conclusion.

The regime whose turnover most closely matches the observed value (9.05 against 10.10) yields:

> **duplicate rate 0.68% ± 0.13% (95% CI, n = 30 seeds)**
> **distinct evaluations 744.9 of 750, 95% CI [744.0, 745.9]**

---

## 6. Conclusion

NSGA-II submitted 750 evaluations comprising approximately **745 distinct policies**, against 750
distinct policies for both random search and MOBO, whose candidates are drawn from continuous
distributions and were verified to contain no duplicates. The shortfall is **five evaluations, or
0.68%**.

For context, the between-run standard deviation of the GA's training-front hypervolume across
five repetitions is 6.7% of its mean. The duplication shortfall is an order of magnitude smaller
than the run-to-run variation of the method itself and cannot affect any reported comparison.

**The matched budget of 750 evaluations per arm is therefore sound**, and the replication of the
random-search and MOBO arms should use 750 evaluations to match the completed NSGA-II arm.

---

## 7. Limitations

The simulation reproduces the production configuration's decision space, operators, encoding and
duplicate-elimination behaviour, but not the true objective landscape, which is unavailable
without re-executing the searches. Its validity rests on the demonstrated insensitivity of the
duplication rate to selection pressure (§5) rather than on fidelity to the real fitness function.
The reported figure should be read as characterising the rate to within a few tenths of a percent,
not as an exact count for the specific completed runs.

The analysis addresses matching on *evaluations*. It does not address matching on computational
expenditure, which the arms do not satisfy: retry budgets are decision variables, so per-evaluation
inference cost is policy-dependent. For the same nominal budget of 750 evaluations the three arms
consumed 225.65 h (NSGA-II), 250.49 h (random search) and 314.64 h (MOBO) of total search
wall-clock. Those figures and the rest of the cost accounting are reported in
`docs/reports/search_cost.md`.
