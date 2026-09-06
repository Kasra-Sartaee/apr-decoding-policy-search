# Provenance of the published runs

This repository is a curated snapshot prepared for examination. It carries a
single commit, so the commit hashes recorded inside the published run artefacts
do not resolve against its history. They resolve against the originating
repository, and this file records the correspondence.

**The run artefacts are not rewritten.** Every `protocol.json` and
`manifest.json` under `results/` is exactly as the producing run wrote it. What
follows is the key needed to interpret them.

## Originating repository

    github.com/davejjwilliams/agent-lifecycle-optimisation

Development ran from 2026-03-05 to 2026-09-06 over 142 commits. See
`ATTRIBUTION.md` for who contributed what.

## Source commit per repetition

Each row is the commit the producing run recorded at the moment it started.

| arm | repetition | seed | source commit | measurement protocol |
| --- | --- | --- | --- | --- |
| NSGA-II | rep_0 | 42 | `f50a3ef04e47` | `codeflaws_two_phase_v3.1` |
| NSGA-II | rep_1 | 43 | `e0302db3eff8` | `codeflaws_two_phase_v3.1` |
| NSGA-II | rep_2 | 44 | `e0302db3eff8` | `codeflaws_two_phase_v3.1` |
| NSGA-II | rep_3 | 45 | `c979e8bad76b` | `codeflaws_two_phase_v3.1` |
| NSGA-II | rep_4 | 46 | `e96602ba32d7` | `codeflaws_two_phase_v3.1` |
| random search | rep_0 | 99 | `f50a3ef04e47` | `codeflaws_two_phase_v3.1` |
| random search | rep_1 | 100 | `b22230d1ff46` | `codeflaws_two_phase_v3.1` |
| random search | rep_2 | 101 | `b22230d1ff46` | `codeflaws_two_phase_v3.1` |
| random search | rep_3 | 102 | `3f1ebcedff9c` | `codeflaws_two_phase_v3.1` |
| random search | rep_4 | 103 | `aec589240d53` | `codeflaws_two_phase_v3.1` |
| MOBO | rep_0 | 123 | `ddc989d7c019` | `codeflaws_two_phase_v3.1` |
| MOBO | rep_1 | 124 | `44eea332bd56` | `codeflaws_two_phase_v3.1` |
| MOBO | rep_2 | 125 | `44eea332bd56` | `codeflaws_two_phase_v3.1` |
| MOBO | rep_3 | 126 | `54f26d1aab85` | `codeflaws_two_phase_v3.1` |
| MOBO | rep_4 | 127 | `18ad60380296` | `codeflaws_two_phase_v3.1` |

All eleven distinct commits resolve in the originating repository.

## `git_dirty` is `true` in every record

Every protocol reports a dirty tree. There is a known defect that would set the
flag on its own: the dirtiness check compared `git status` output against a list
of run-artefact path prefixes in order to ignore them, and those prefixes were
written relative to the repository root while `git status` was invoked from the
package subdirectory. The prefixes therefore never matched, and any untracked run
output — which a search produces continuously — set the flag.

That mechanism accounts for the flag being set, but it does not establish that
nothing else was modified: the flag is a single boolean and cannot distinguish
generated output from an edited source file. The recorded commit hash is
therefore the nearest recoverable base revision for each run, not a guarantee
that the producing working tree matched it. The exact tree is not recoverable.

The defect is fixed in this snapshot: flattening the package into the repository
root aligned the two, so `provenance.git_is_dirty()` now reports what it was
written to report. The historical records keep the value they were given.

## Filename correspondence

Paths inside the historical artefacts refer to the originating layout. The
mapping to this snapshot:

| in the run artefacts | here |
| --- | --- |
| `agentic-se-experiment/<path>` | `<path>` |
| `results/ga_reps_v4/` | `results/nsga2/` |
| `results/rs_reps_v4/` | `results/random_search/` |
| `results/mobo_reps_v4/` | `results/mobo/` |
| `contracts_new/` | `contracts/` |
| `scripts/run_ga.py` | `scripts/run_nsga2.py` |
| `scripts/run_random_search_v4.py` | `scripts/run_random_search.py` |
| `scripts/run_mobo_v4.py` | `scripts/run_mobo.py` |
| `scripts/heldout_eval_n15_runs.py` | `scripts/evaluate_heldout.py` |
| `scripts/baseline_heldout_twophase.py` | `scripts/evaluate_baseline.py` |
| `outputs/**/baseline_default_n100_seed0_reps5/` | `baselines/<benchmark>/` |

The run labels recorded *inside* the artefacts — `pop50_gen15_n50_v4`,
`rs_pop50_gen15_n50_v4`, `mobo_pop50_gen15_n50_v4`, `py_n50_random` — are
unchanged, because they are identifiers the artefacts refer to each other by. So
is `optimiser_contract_new.md`, whose filename is a key in the `contract_sha256`
record of every protocol.

## Verifying the prompt contracts

The contracts shipped here hash to the values the runs recorded:

```bash
python3 -c "
import json, hashlib
from pathlib import Path
r = json.load(open('results/nsga2/rep_0/protocol.json'))['contract_sha256']
for n, h in r.items():
    print(n, hashlib.sha256((Path('contracts') / n).read_bytes()).hexdigest() == h)"
```

## What is not recoverable

The per-candidate verification telemetry the searches wrote under `outputs/`
(analyzer invocation counts, per-stage token counts) no longer exists.

`docs/reports/search_cost.md` is computed entirely from that telemetry. It is
preserved as generated and cannot be recomputed from the released artefacts;
`analysis/search_cost.py` ships as the code that produced it and refuses to run
rather than overwrite the report with zero-record output.

`docs/reports/budget_matching.md` is affected only in part. Its first two
evidence sources — the pymoo evaluation counter and the stored per-candidate
records — are recoverable from `results/`. Its third, the analyzer-invocation
count, is telemetry-derived and is preserved only; the report states this at the
point of use.
