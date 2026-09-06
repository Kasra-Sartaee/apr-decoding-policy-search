# Attribution

This repository accompanies the dissertation *Multi-Objective Search over the
Decoding Policy of a Multi-Agent LLM Pipeline for Automated Program Repair*. It
is not solely the dissertation author's work, and this file records who
contributed what.

## Inherited work

**Prof. Enrique Alba** originated and supervised the research programme this
work sits within, including the design of the four-stage
Analyzer→Repair→Verifier→Optimizer pipeline. The contribution is one of design
and direction rather than of code: the version history contains no commits under
his name, and nothing here should be read as attributing implementation to him.

**Giordano d'Aloisio** (`giordano.daloisio@graduate.univaq.it`) implemented that
pipeline, in 34 commits between 2026-03-09 and 2026-06-24. `agentic_workflow.py`
and `agents.py` in this repository descend directly from his code; both retain
his tab-indentation style, which is why they differ from the rest of the tree.
The prompt contracts in `contracts/` are inherited from the same work.

He also wrote the first policy-search driver. Commit `b14df2b` (2026-03-13,
"drafted pymoo algorithm") already contains `optimize_policy_nsga2.py`: an
NSGA-II search over a fourteen-dimensional decoding-and-retry vector, with a
`decode_policy` function, per-stage `temperature`, `top_p`, `top_k` and retry
variables, and three objectives. **The idea of searching the decoding policy of
this pipeline with a multi-objective evolutionary algorithm therefore originates
with him, not with this dissertation.**

**Dave J. J. Williams** (`davejjwilliams@gmail.com`) established the repository
and its initial structure in March 2026.

## Work of this dissertation

**Kasra Sartaee** (`kasrasartaee@yahoo.com`), from 2026-04-22. The contribution
is not the idea of searching decoding policy, which is inherited as recorded
above, but the following:

- the redesign of the inherited fourteen-dimensional vector into the present
  eleven-dimensional shared formulation, and its extraction from a single
  driver into a common module (`search_space.py`) that every driver imports, so
  that all three strategies provably search an identical space;
- the three search drivers (`scripts/run_nsga2.py`, `scripts/run_random_search.py`,
  `scripts/run_mobo.py`) and their budget-matching controls;
- the two-phase energy measurement protocol (`benchmarks/base.py`,
  `energy/`), documented in `docs/reports/measurement_protocol.md`;
- the held-out and transfer evaluators (`scripts/evaluate_heldout.py`,
  `scripts/evaluate_baseline.py`);
- the benchmark adapters for Codeflaws and RunBugRun;
- the provenance and telemetry recording (`provenance.py`, `telemetry.py`,
  `inference_energy.py`);
- the whole of `analysis/` and `docs/reports/`.

## History

This repository was created as a curated snapshot for examination and carries a
single commit. The full development history — 142 commits from March to September
2026 — is in the originating repository; see `PROVENANCE.md` for how the two
relate.

## Third-party components

The benchmarks are external datasets, downloaded rather than redistributed here:

- **Codeflaws** — Tan et al., *Codeflaws: A Programming Competition Benchmark for
  Evaluating Automated Program Repair Tools*, ICSE-C 2017.
- **RunBugRun** — Prenner and Robbes, *RunBugRun: An executable dataset for
  automated program repair*, Empirical Software Engineering 31(4), 2026.

Only the pinned 300-problem sample manifests are committed
(`benchmarks/*/data/sample_300_seed42.json`); the corpora themselves are fetched
on first use.

Library dependencies are listed in `requirements.txt`. NSGA-II is provided by
pymoo (Blank and Deb, 2020) and the surrogate arm by BoTorch (Balandat et al.,
2020).
