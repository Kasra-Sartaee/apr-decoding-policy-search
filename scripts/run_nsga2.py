#!/usr/bin/env python3
"""NSGA-II policy search over a Codeflaws training subset.

The research question (RQ2): does NSGA-II over agent decoding
hyperparameters find Pareto-superior ``(complexity, correctness,
energy)`` policies versus the default decoding policy?

Search space (11 dimensions):

    pi_a.temperature             Analyzer decoding temperature
    pi_a.top_p, pi_a.top_k      Analyzer decoding nucleus/top-k
    pi_r.temperature             Repair decoding temperature
    pi_r.top_p, pi_r.top_k      Repair decoding nucleus/top-k
    pi_v.retry                   Verifier retry budget
    pi_o.temperature             Optimizer decoding temperature
    pi_o.top_p, pi_o.top_k      Optimizer decoding nucleus/top-k
    pi_o.retry                   Optimizer retry budget

Temperature is searched (range 0–1) rather than fixed, following the
finding in docs/reports/05_ga_optimization_7b_corrected.md that pinning
temperature to 0 both regresses compilation and confounds the baseline
comparison. The Verifier (pi_v) has no temperature — it is not an LLM.

Objectives (all minimised by pymoo):

    F1 = mean cyclomatic complexity     (feasible problems only)
    F2 = -mean correctness              (all problems; higher → lower F2)
    F3 = mean energy mJ                 (feasible problems only, CPU RAPL)

Feasibility is handled via Deb 2002 constrained domination (out["G"]),
NOT as a penalty folded into the objectives. Compile-failures do not
contribute inf to F1/F3 — they are excluded from the feasible aggregate.
A candidate with zero feasible problems is ranked below all feasible
candidates regardless of its F values.

Examples
--------
Smoke test (tiny search, 2 problems)::

    python scripts/run_nsga2.py --model qwen2.5-coder:7b \\
        --pop 2 --gen 1 --reps 1 --train-size 2 --energy-backend runtime_proxy

Reduced configuration (pop 20, gen 5, reps 1) for a smoke run::

    python scripts/run_nsga2.py --model qwen2.5-coder:7b \\
        --pop 20 --gen 5 --reps 1 --train-size 15 --energy-backend rapl_sysfs

Default-policy baseline (no search; evaluate the default decoding once
on the same training subset, for comparison)::

    python scripts/run_nsga2.py --model qwen2.5-coder:7b \\
        --baseline --train-size 15 --energy-backend rapl_sysfs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

# Honour --model and --energy-backend BEFORE importing agentic_workflow so
# DEFAULT_MODEL resolves to the right Ollama tag at import time.
_early_model = None
for _i, _a in enumerate(sys.argv):
    if _a == "--model" and _i + 1 < len(sys.argv):
        _early_model = sys.argv[_i + 1]
    elif _a.startswith("--model="):
        _early_model = _a.split("=", 1)[1]
if _early_model:
    os.environ["AGENT_MODEL"] = _early_model

from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.core.callback import Callback  # noqa: E402
from pymoo.core.problem import ElementwiseProblem  # noqa: E402
from pymoo.core.sampling import Sampling  # noqa: E402
from pymoo.operators.crossover.sbx import SBX  # noqa: E402
from pymoo.operators.mutation.pm import PM  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

from agentic_workflow import (  # noqa: E402
    DEFAULT_MODEL,
    run_pipeline,
    run_pipeline_phase_a,
    run_pipeline_phase_b,
    set_energy_backend,
)
from benchmarks import get_benchmark  # noqa: E402
from benchmarks.codeflaws.adapter import (  # noqa: E402
    CODEFLAWS_COMPILE_TIMEOUT_S,
    CODEFLAWS_RUN_TIMEOUT_S,
)
from energy import resolve_backend  # noqa: E402
from provenance import (  # noqa: E402
    git_commit_hash as _git_commit_hash,
)
from provenance import (
    git_is_dirty as _git_is_dirty,
)
from provenance import (
    write_protocol_manifest,
)
from search_space import SEARCH_SPACE, build_policy_from_vector  # noqa: E402

# SEARCH_SPACE and build_policy_from_vector are imported from search_space.py —
# the single source of truth shared with the random-search ablation
# (run_random_search.py), so both searches optimise the identical 11-D space.

# NSGA-II operator settings. NOTE on pymoo semantics: PM's `prob` is the
# PER-INDIVIDUAL probability that an offspring is mutated at all; the per-gene
# rate is `prob_var`, which we leave at pymoo's default min(0.5, 1/n_var)=1/11.
# SBX's `prob` is the per-mating crossover probability.
SBX_ETA = 15
PM_ETA = 20
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2

# Number of problems evaluated concurrently within each NSGA-II candidate.
# N_WORKERS=1 (default) preserves the original sequential behaviour exactly
# (uses run_pipeline()'s single-shot Phase A + Phase B wrapper directly).
# Set N_WORKERS=4 for local Ollama with a matching OLLAMA_NUM_PARALLEL.
#
# For N_WORKERS>1, fitness evaluation runs as two phases (see
# agentic_workflow.run_pipeline_phase_a/_b and
# benchmarks/codeflaws/adapter.py's _CodeflawsTestRunner):
#   Phase A (concurrent, this pool): analyze/repair/retry/optimize/retry
#     for every problem, using diagnostic-only (no RAPL) verification.
#   Phase B (serial, after the pool above fully joins): recompile each
#     problem's FINAL source and rerun ONLY its already-known-passing
#     tests, once, under RAPL.
# This replaced an earlier design where every intermediate repair-retry
# and optimizer-retry verification ran under RAPL too, even though only
# the final attempt's energy is ever used — on the n50 training set,
# repair-retry + optimizer-retry suites were 82.5% of all suite
# executions (see memory/rapl_contamination_local_ollama.md), and a large
# fraction of those are hangs/timeouts on broken intermediate repairs,
# each burning the full run_timeout under exclusive RAPL access while
# blocking every other worker's inference. Phase B's true join-then-serial
# barrier also closes a gap the gate-only design had: compilation was
# never gated, so one thread's gcc could contaminate another's "isolated"
# RAPL window; during Phase B nothing else runs in the process at all.
_N_WORKERS = int(os.environ.get("N_WORKERS", "1"))

# Lock for serialised console output from parallel worker threads.
_print_lock = threading.Lock()


def _build_config(out_root: Path, base_dir: Path) -> dict[str, Any]:
    """Pipeline config dict. programs_dir/tests_dir/input_file are unused
    when a Problem is supplied, but run_pipeline reads the *_dir keys."""
    return {
        "base_dir": str(base_dir),
        "programs_dir": "programs",
        "tests_dir": "tests",
        "input_file": "unused-when-problem-supplied",
        "reports_dir": str(out_root / "reports"),
        "repaired_dir": str(out_root / "repaired"),
        "optimized_dir": str(out_root / "optimized"),
        "verification_json": str(out_root / "verification"),
        "model": DEFAULT_MODEL,
        "llm_timeout": 180,
        "compile_timeout": 40,
        "run_timeout": 3,
    }


def _select_training_subset(problems: list, train_size: int, seed: int) -> list:
    """Deterministically pick a training subset of the Codeflaws sample.

    The Codeflaws adapter already returns a seeded, stratified 300-problem
    sample in a stable order. We take a seeded random subset of that for
    training; the remaining problems form the held-out test set used in a
    later phase. Seeding here means the same training subset is used by
    the GA and by the baseline comparison.
    """
    rng = np.random.default_rng(seed)
    n = min(train_size, len(problems))
    idx = rng.choice(len(problems), size=n, replace=False)
    return [problems[i] for i in sorted(idx.tolist())]


# A problem is "feasible" for a candidate iff its repair compiled AND
# passed at least one test (correctness > 0). Complexity and energy are
# undefined for a non-executing program, so they are aggregated only over
# feasible problems; correctness is aggregated over ALL problems (a 0.0 on
# a failed problem is on the bounded [0,1] scale and legitimately lowers
# the mean without saturating). See docs/reports/05_ga_optimization_7b_corrected.md
# §6 and the constrained-domination rationale (Deb 2002).
#
# Objective placeholders for a candidate with ZERO feasible problems: these
# are never actually compared on objectives (the constraint marks such a
# candidate infeasible and Deb's constrained domination ranks it below every
# feasible candidate), but pymoo still needs finite numbers in out["F"].
# We use 0.0 — deliberately NOT a large penalty — so the objective scale is
# never polluted. Feasibility is carried entirely by the constraint, not F.
_NO_FEASIBLE_CL = 0.0
_NO_FEASIBLE_ENERGY = 0.0


def _eval_one_problem(
    p, config: dict, policy: dict | None
) -> tuple[float, float, float, Exception | None]:
    """Evaluate a single problem single-shot (Phase A + Phase B together).

    Used only by the N_WORKERS=1 sequential path, where there is no
    concurrency to protect against and no benefit to splitting the phases.
    Returns (cl, correctness, energy, exception_or_None). On infrastructure
    failure (LLM timeout, network error, etc.) the exception is returned
    rather than re-raised, so the caller can continue past one bad problem.
    """
    try:
        res = run_pipeline(config=config, policy=policy, problem=p, skip_optimize=False)
        return float(res.cl), float(res.correctness), float(res.energy_consumption), None
    except Exception as err:
        return float("inf"), 0.0, float("inf"), err


def _eval_one_problem_phase_a(p, config: dict, policy: dict | None):
    """Phase A only: analyze/repair/retry/optimize/retry, no RAPL. Safe to
    call from concurrent worker threads — see run_pipeline_phase_a()'s
    docstring. Returns (PendingResult_or_None, Exception_or_None)."""
    try:
        return run_pipeline_phase_a(config=config, policy=policy, problem=p, skip_optimize=False), None
    except Exception as err:
        return None, err


def _eval_one_problem_phase_b(pending) -> tuple[float, float, float, Exception | None]:
    """Phase B only: final energy measurement for one already-settled
    problem. Call this SERIALLY — see run_pipeline_phase_b()'s docstring
    for why (nothing else may be running concurrently during this call)."""
    try:
        res = run_pipeline_phase_b(pending)
        return float(res.cl), float(res.correctness), float(res.energy_consumption), None
    except Exception as err:
        return float("inf"), 0.0, float("inf"), err


def _mean_fitness_over(problems: list, config: dict, policy: dict | None) -> dict[str, Any]:
    """Run the pipeline over `problems` with `policy`; return aggregated fitness.

    When N_WORKERS > 1 (set via the N_WORKERS env var), problems are
    evaluated in two phases (see agentic_workflow.run_pipeline_phase_a/_b):
    Phase A runs every problem's full analyze/repair/retry/optimize/retry
    lifecycle concurrently across a ThreadPoolExecutor, using diagnostic-only
    (no RAPL) verification throughout — this is where LLM inference overlaps
    freely across workers. Only once every Phase A worker has fully joined
    does Phase B run: a plain serial loop that recompiles each problem's
    final source and reruns just its already-known-passing tests, once,
    under RAPL. Because Phase B starts only after the whole pool above has
    joined, nothing else in the process (inference, compilation, other
    problems' diagnostic testing) is running during it — so F3 is never
    contaminated by concurrent activity of any kind, registered or not.
    Results are aggregated in the original problem order regardless of
    thread completion order.

    N_WORKERS=1 (default) runs the original sequential loop with no threading
    overhead — identical behaviour to the pre-parallelisation code.

    Returns a dict with:
      mean_cl            mean cyclomatic complexity over FEASIBLE problems
      mean_correctness   mean correctness over ALL problems
      mean_energy        mean energy over FEASIBLE problems
      n_feasible         count of problems with correctness > 0
      n_total            number of problems evaluated
      per_problem        raw per-problem (cl, correctness, energy, feasible)

    No `1e12` penalty: complexity/energy are simply not averaged over
    problems where they are undefined. Infeasibility is expressed via
    `n_feasible` (consumed as an NSGA-II constraint by the caller), and via
    correctness (which drops toward 0 as problems fail).
    """
    # ── Collect raw results (parallel two-phase, or sequential) ───────────
    raw: dict[str, tuple] = {}
    if _N_WORKERS > 1:
        # Phase A: concurrent, RAPL-free repair/retry/optimize lifecycle
        # for every problem in this candidate's training subset.
        pending_by_problem: dict[str, tuple] = {}
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            futures = {pool.submit(_eval_one_problem_phase_a, p, config, policy): p for p in problems}
            for fut in as_completed(futures):
                p = futures[fut]
                pending_by_problem[p.problem_id] = fut.result()
        # Phase B: serial, RAPL-measured final energy. Runs only after the
        # `with` block above has fully exited — every Phase A worker has
        # joined, so nothing else in the process is doing inference,
        # compilation, or diagnostic testing during this loop.
        for p in problems:
            pending, err = pending_by_problem[p.problem_id]
            if err is not None:
                raw[p.problem_id] = (float("inf"), 0.0, float("inf"), err)
            else:
                raw[p.problem_id] = _eval_one_problem_phase_b(pending)
    else:
        # Sequential path — no threading overhead, original behaviour.
        for p in problems:
            raw[p.problem_id] = _eval_one_problem(p, config, policy)

    # ── Aggregate in original problem order ───────────────────────────────
    corrs: list[float] = []
    feasible_cls: list[float] = []
    feasible_energies: list[float] = []
    per_problem = []
    n_feasible = 0
    for p in problems:
        cl, corr, energy, err = raw[p.problem_id]
        if err is not None:
            with _print_lock:
                print(f"    [warn] {p.problem_id} raised: {err}")

        # bool() ensures Python bool, not numpy.bool_, which json.dumps
        # cannot serialise (np.isfinite returns numpy.bool_).
        feasible = bool(corr > 0.0 and np.isfinite(cl) and np.isfinite(energy))
        corrs.append(corr)
        if feasible:
            n_feasible += 1
            feasible_cls.append(cl)
            feasible_energies.append(energy)
        per_problem.append({
            "problem_id": p.problem_id,
            "cl": cl if np.isfinite(cl) else None,
            "correctness": corr,
            "energy": energy if np.isfinite(energy) else None,
            "feasible": feasible,
        })

    mean_cl = float(np.mean(feasible_cls)) if feasible_cls else _NO_FEASIBLE_CL
    mean_energy = float(np.mean(feasible_energies)) if feasible_energies else _NO_FEASIBLE_ENERGY
    mean_correctness = float(np.mean(corrs)) if corrs else 0.0

    return {
        "mean_cl": mean_cl,
        "mean_correctness": mean_correctness,
        "mean_energy": mean_energy,
        "n_feasible": n_feasible,
        "n_total": len(problems),
        "per_problem": per_problem,
    }


class AgentCodeflawsProblem(ElementwiseProblem):
    """One NSGA-II objective evaluation = aggregated fitness over the training subset.

    Feasibility is handled as a constraint (Deb 2002 constrained domination),
    NOT as a penalty folded into the objectives. A candidate is feasible iff
    it solves at least `min_feasible` of the training problems (correctness > 0
    each). pymoo applies the feasible-dominates-infeasible rule automatically
    once out["G"] is populated; complexity/energy never carry a penalty.
    """

    def __init__(self, train_problems: list, config: dict, min_feasible: int) -> None:
        self.train_problems = train_problems
        self.config = config
        self.min_feasible = min_feasible
        self._eval_count = 0
        lower = [d["low"] for d in SEARCH_SPACE]
        upper = [d["high"] for d in SEARCH_SPACE]
        # n_ieq_constr=1: a single inequality constraint, G <= 0 means feasible.
        super().__init__(
            n_var=len(SEARCH_SPACE), n_obj=3, n_ieq_constr=1, xl=lower, xu=upper
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        self._eval_count += 1
        policy = build_policy_from_vector(x)
        print(f"  [candidate {self._eval_count}] policy={json.dumps(policy)}")
        agg = _mean_fitness_over(self.train_problems, self.config, policy)
        print(
            f"    -> n_feasible={agg['n_feasible']}/{agg['n_total']} "
            f"mean_cl={agg['mean_cl']:.2f} "
            f"mean_corr={agg['mean_correctness']:.3f} "
            f"mean_energy={agg['mean_energy']:.0f}"
        )
        # Objectives — F1 complexity (min), F2 -correctness (min == max
        # correctness), F3 energy (min). Computed over feasible problems
        # only (complexity/energy) and all problems (correctness).
        out["F"] = [agg["mean_cl"], -agg["mean_correctness"], agg["mean_energy"]]
        # Constraint — feasible iff at least `min_feasible` problems solved.
        # G = min_feasible - n_feasible  ->  G <= 0 when feasible; the value
        # when infeasible is the shortfall (so "closer to feasible" candidates
        # are ranked above "more broken" ones, per Deb's constrained domination).
        out["G"] = [float(self.min_feasible - agg["n_feasible"])]


class _WarmStartSampling(Sampling):
    """Returns a saved population array as the initial generation for a warm restart."""

    def __init__(self, X: np.ndarray) -> None:
        super().__init__()
        self._X = X

    def _do(self, problem, n_samples, **kwargs):
        return self._X


class _GenCheckpoint(Callback):
    """Atomically saves population + full history after every generation.

    On restart, run_search loads this checkpoint and resumes from the saved
    generation, so a kill costs at most one generation rather than the whole
    run.
    """

    def __init__(
        self, checkpoint_path: Path, history_acc: list, gen_offset: int = 0, total_gen: int = 0
    ) -> None:
        super().__init__()
        self._path = checkpoint_path
        self._history = history_acc
        self._offset = gen_offset
        self._total = total_gen

    def notify(self, algorithm) -> None:
        gen = algorithm.n_gen + self._offset
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")
        for j in range(len(X)):
            feasible = bool(G is None or (np.atleast_1d(G[j]) <= 0).all())
            self._history.append({
                "generation": gen,
                "policy": build_policy_from_vector(X[j]),
                "feasible": feasible,
                "fitness": {
                    "cl": float(F[j][0]),
                    "correctness": float(-F[j][1]),
                    "energy": float(F[j][2]),
                },
            })
        ckpt = {
            "generation": gen,
            "pop_X": X.tolist(),
            "pop_F": F.tolist() if F is not None else None,
            "pop_G": G.tolist() if G is not None else None,
            "history": self._history,
        }
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(ckpt), encoding="utf-8")
            tmp.rename(self._path)
            print(f"[run_nsga2] checkpoint saved at generation {gen}/{self._total} -> {self._path}")
        except Exception as exc:
            print(f"[run_nsga2] WARNING: checkpoint save failed: {exc}")


def run_baseline(train_problems, config, out_root: Path, reps: int = 1) -> None:
    """Evaluate the DEFAULT decoding policy on the training subset, `reps` times.

    This is the comparison point for 'did NSGA-II actually help?'. Default
    policy means policy=None, which makes run_pipeline use the agents'
    default Ollama options.

    Repeating a FIXED policy measures the noise floor on the same problems the
    search selects against. Nothing varies between repetitions except LLM
    sampling and measurement, so the spread across them bounds how much of
    NSGA-II's selection pressure could be responding to real differences
    between candidates rather than to noise. Each repetition is written
    separately and resumed independently, so an interrupted run costs at most
    the repetition in flight.

    reps=1 writes baseline_default_policy.json at out_root, preserving the
    original single-measurement layout and filename. reps>1 additionally writes
    baseline_default_train_reps<N>/rep_<i>.json plus a summary.
    """
    multi = reps > 1
    reps_dir = out_root / f"baseline_default_train_reps{reps}"
    if multi:
        reps_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(reps):
        path = (reps_dir / f"rep_{i:02d}.json") if multi else (out_root / "baseline_default_policy.json")
        if path.is_file():
            print(f"[run_nsga2] baseline rep {i + 1}/{reps} already done; skipping")
            results.append(json.loads(path.read_text(encoding="utf-8")))
            continue

        print(f"[run_nsga2] evaluating DEFAULT-policy baseline on training subset "
              f"(rep {i + 1}/{reps})")
        t0 = time.time()
        agg = _mean_fitness_over(train_problems, config, policy=None)
        out = {
            "kind": "default_policy_baseline",
            "measurement_protocol": "codeflaws_two_phase_v3.1",
            "protocol_doc": "docs/reports/measurement_protocol.md",
            "git_commit": _git_commit_hash(),
            "git_dirty": _git_is_dirty(),
            "rep": i,
            "n_reps": reps,
            "n_train": len(train_problems),
            "n_feasible": agg["n_feasible"],
            "mean_cl": agg["mean_cl"],
            "mean_correctness": agg["mean_correctness"],
            "mean_energy": agg["mean_energy"],
            "wall_s": time.time() - t0,
            "note": (
                "mean_cl/mean_energy are over feasible problems only "
                "(correctness>0); mean_correctness over all problems."
            ),
            "per_problem": agg["per_problem"],
        }
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        results.append(out)
        print(
            f"[run_nsga2] baseline rep {i + 1}/{reps}: "
            f"n_feasible={agg['n_feasible']}/{agg['n_total']} "
            f"mean_cl={agg['mean_cl']:.2f} "
            f"mean_corr={agg['mean_correctness']:.3f} "
            f"mean_energy={agg['mean_energy']:.2f} -> {path}"
        )

    if not multi:
        return

    def _spread(key):
        v = np.array([r[key] for r in results if r.get(key) is not None], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return {"mean": float(v[0]) if len(v) else None, "sd": None, "cv_pct": None,
                    "min": float(v[0]) if len(v) else None, "max": float(v[0]) if len(v) else None}
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "cv_pct": float(100 * v.std(ddof=1) / v.mean()) if v.mean() else None,
                "min": float(v.min()), "max": float(v.max())}

    summary = {
        "kind": "default_policy_baseline_reps",
        "n_reps": reps,
        "n_train": len(train_problems),
        "measurement_protocol": "codeflaws_two_phase_v3.1",
        "git_commit": _git_commit_hash(),
        "note": (
            "Repeated measurement of ONE fixed policy on the training subset. "
            "The spread is the noise floor NSGA-II's selection operates against; "
            "it is not a search result."
        ),
        "mean_cl": _spread("mean_cl"),
        "mean_correctness": _spread("mean_correctness"),
        "mean_energy": _spread("mean_energy"),
        "n_feasible": _spread("n_feasible"),
    }
    spath = reps_dir / "summary.json"
    spath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[run_nsga2] baseline noise floor over {reps} repetitions on "
          f"{len(train_problems)} training problems:")
    for k in ("mean_cl", "mean_correctness", "mean_energy", "n_feasible"):
        s = summary[k]
        cv = f"{s['cv_pct']:.1f}%" if s["cv_pct"] is not None else "n/a"
        print(f"    {k:18s} {s['mean']:9.4f} ± {s['sd'] or 0:.4f}   CV {cv:>6s}   "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"[run_nsga2] summary -> {spath}")


def _write_rep_outputs(
    rep_idx: int,
    args,
    X: np.ndarray,
    F: np.ndarray | None,
    G: np.ndarray | None,
    history_acc: list,
    pareto_path: Path,
    history_path: Path,
    checkpoint_path: Path,
) -> None:
    """Write pareto_solutions.json and intermediate_solutions.json, then delete checkpoint."""
    X = np.atleast_2d(X) if X is not None and len(X) else np.empty((0, len(SEARCH_SPACE)))
    F = np.atleast_2d(F) if F is not None and len(F) else np.empty((0, 3))
    if G is not None and len(G):
        G = np.atleast_2d(G)
        feasible_mask = np.atleast_1d((G <= 0).all(axis=1))
    else:
        feasible_mask = np.ones(len(X), dtype=bool)

    pareto = [
        {
            "policy": build_policy_from_vector(X[k]),
            "fitness": {
                "cl": float(F[k][0]),
                "correctness": float(-F[k][1]),
                "energy": float(F[k][2]),
            },
        }
        for k in range(len(X)) if feasible_mask[k]
    ]
    n_dropped = int((~feasible_mask).sum())
    pareto_path.write_text(json.dumps(pareto, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history_acc, indent=2), encoding="utf-8")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("[run_nsga2] checkpoint removed (run complete)")

    if not pareto:
        print(
            f"[run_nsga2] rep {rep_idx + 1}: 0 FEASIBLE Pareto solutions "
            f"(all candidates solved < {args.min_feasible} problems). "
            f"Empty front written -> {pareto_path}"
        )
    else:
        print(
            f"[run_nsga2] rep {rep_idx + 1}: {len(pareto)} feasible Pareto solutions "
            f"({n_dropped} infeasible dropped from reported front) -> {pareto_path}"
        )


def _write_protocol_manifest(rep_dir: Path, args, seed: int) -> None:
    """Per-repetition provenance: which F3 measurement protocol and source
    commit produced this rep's data. The run-label-level manifest.json is
    shared across all reps and gets overwritten by whichever repetition ran
    most recently, so it cannot distinguish (e.g.) a v2-protocol rep_0 from
    a v3-protocol rep_1 under the same run-label — this file can. See
    docs/reports/measurement_protocol.md.

    Delegates to provenance.write_protocol_manifest so this driver and
    run_nsga2_runtime.py record provenance identically.
    """
    write_protocol_manifest(
        rep_dir,
        measurement_protocol="codeflaws_two_phase_v3.1",
        protocol_doc="docs/reports/measurement_protocol.md",
        language="C99",
        n_workers=_N_WORKERS,
        # The values actually in force, not the config dict's nominal ones:
        # the Codeflaws runner carries its own compile default and the LLM
        # timeout comes from an env var, so the nominal numbers could differ
        # from what really ran.
        timeouts_effective={
            "llm_request_s": float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "300")),
            "compile_s": CODEFLAWS_COMPILE_TIMEOUT_S,
            "run_s": CODEFLAWS_RUN_TIMEOUT_S,
        },
        seed=seed,
        seed_increment=args.seed_increment,
        model=args.model or DEFAULT_MODEL,
        pop=args.pop,
        gen=args.gen,
        train_size=args.train_size,
    )
def run_search(train_problems, config, args, out_root: Path) -> None:
    """Run NSGA-II for the requested repetitions; persist Pareto + history.

    Checkpointing: after every generation, the current population and
    accumulated history are atomically written to rep_N/checkpoint.json.
    On restart, this checkpoint is loaded and the search resumes from the
    saved generation — so an OOM kill costs at most one generation (~2-3 min),
    not the entire run.
    """
    for rep_idx in range(args.reps):
        rep_dir = out_root / f"rep_{rep_idx}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        pareto_path = rep_dir / "pareto_solutions.json"
        history_path = rep_dir / "intermediate_solutions.json"
        checkpoint_path = rep_dir / "checkpoint.json"

        if not args.overwrite and pareto_path.exists() and history_path.exists():
            print(f"[run_nsga2] rep {rep_idx + 1}/{args.reps} already done; skipping (use --overwrite)")
            continue

        # Written only for reps actually run under THIS code — never
        # retroactively for an already-completed (and skipped, above) rep
        # from an older protocol version.
        _write_protocol_manifest(rep_dir, args, seed=args.seed + rep_idx * args.seed_increment)

        # Load checkpoint if present.
        start_gen = 0
        history_acc: list = []
        sampling = None

        restart_log_path = rep_dir / "restart_log.json"

        if checkpoint_path.exists() and not args.overwrite:
            try:
                ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                start_gen = int(ckpt["generation"])
                history_acc = list(ckpt.get("history", []))
                sampling = _WarmStartSampling(np.array(ckpt["pop_X"]))
                print(
                    f"[run_nsga2] rep {rep_idx + 1}: checkpoint found at generation "
                    f"{start_gen}/{args.gen} — resuming"
                )
                # Append a permanent restart record so outputs self-document any
                # interruptions for reproducibility / dissertation transparency.
                prior = json.loads(restart_log_path.read_text(encoding="utf-8")) if restart_log_path.exists() else []
                prior.append({
                    "resumed_at": datetime.now(UTC).isoformat(),
                    "resumed_from_generation": start_gen,
                    "total_generations": args.gen,
                    "note": "RNG state reset for crossover/mutation; all fitness values are fresh measurements",
                })
                restart_log_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")
            except Exception as exc:
                print(f"[run_nsga2] checkpoint load failed ({exc}); starting fresh")
                start_gen = 0
                history_acc = []
                sampling = None

        remaining_gens = args.gen - start_gen

        if remaining_gens <= 0:
            # All generations already complete (checkpoint full) but outputs
            # were not yet written — reconstruct them from the checkpoint.
            print(
                f"[run_nsga2] rep {rep_idx + 1}: checkpoint already at gen {start_gen}; "
                "writing outputs from checkpoint"
            )
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            _write_rep_outputs(
                rep_idx, args,
                np.array(ckpt["pop_X"]),
                np.array(ckpt["pop_F"]) if ckpt.get("pop_F") else None,
                np.array(ckpt["pop_G"]) if ckpt.get("pop_G") else None,
                list(ckpt.get("history", [])),
                pareto_path, history_path, checkpoint_path,
            )
            continue

        print(
            f"[run_nsga2] === repetition {rep_idx + 1}/{args.reps} "
            f"(seed={args.seed + rep_idx * args.seed_increment}) — gen {start_gen + 1}..{args.gen} ==="
        )
        problem = AgentCodeflawsProblem(train_problems, config, min_feasible=args.min_feasible)
        problem._eval_count = start_gen * args.pop  # keep candidate numbering continuous

        callback = _GenCheckpoint(
            checkpoint_path, history_acc, gen_offset=start_gen, total_gen=args.gen
        )

        alg_kw: dict = dict(
            pop_size=args.pop,
            crossover=SBX(eta=SBX_ETA, prob=CROSSOVER_PROB),
            mutation=PM(eta=PM_ETA, prob=MUTATION_PROB),
        )
        if sampling is not None:
            alg_kw["sampling"] = sampling

        result = minimize(
            problem,
            NSGA2(**alg_kw),
            termination=("n_gen", remaining_gens),
            seed=args.seed + rep_idx * args.seed_increment,
            save_history=False,  # history accumulated in callback instead
            verbose=True,
            callback=callback,
        )

        _write_rep_outputs(
            rep_idx, args,
            np.atleast_2d(result.X) if result.X is not None else np.empty((0, len(SEARCH_SPACE))),
            np.atleast_2d(result.F) if result.F is not None else np.empty((0, 3)),
            result.G,
            history_acc,
            pareto_path, history_path, checkpoint_path,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=None, help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--pop", type=int, default=20, help="NSGA-II population size")
    parser.add_argument("--gen", type=int, default=5, help="NSGA-II generations")
    parser.add_argument("--reps", type=int, default=1, help="Independent NSGA-II repetitions")
    parser.add_argument("--train-size", type=int, default=15, help="# Codeflaws problems in the training subset")
    parser.add_argument(
        "--min-feasible",
        type=int,
        default=1,
        help="Feasibility constraint: a candidate is feasible iff it solves "
             "(correctness>0) at least this many of the training problems. "
             "Default 1 — keeps the feasible front populated. Raise to demand "
             "a candidate solve more of the subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed; rep i uses seed + i*seed_increment; also seeds subset selection")
    parser.add_argument(
        "--seed-increment", type=int, default=1,
        help="Per-repetition seed step (default 1: rep i uses seed+i, so each "
             "repetition explores a different GA search trajectory in addition "
             "to LLM-inherent stochasticity). Set to 0 to hold the GA's own RNG "
             "seed FIXED across all repetitions, isolating LLM sampling "
             "stochasticity as the only source of run-to-run variance — useful "
             "when comparing repetition-to-repetition variance directly against "
             "an N-rep baseline's variance, which is driven purely by LLM "
             "sampling with no search-trajectory randomness of its own.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run repetitions even if outputs exist")
    parser.add_argument("--baseline", action="store_true", help="Evaluate the default policy on the subset instead of searching")
    parser.add_argument("--no-download", action="store_true", help="Assume Codeflaws data is already present")
    parser.add_argument(
        "--energy-backend",
        default="auto",
        choices=["auto", "apple_silicon", "rapl_sysfs", "codecarbon", "runtime_proxy"],
        help="Energy backend",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional subfolder name under outputs/ga/codeflaws/<model>/. Use this to "
             "isolate a run from the canonical results — e.g. --run-label "
             "demo_windows_proxy keeps a reduced-scale Windows demo separate from the "
             "real Linux+RAPL lab results that occupy the bare <model> folder.",
    )
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL
    if args.model:
        os.environ["AGENT_MODEL"] = args.model

    # Resolve + register the energy backend before any pipeline call.
    energy_backend = resolve_backend(args.energy_backend)
    set_energy_backend(energy_backend)

    benchmark = get_benchmark("codeflaws")
    print(f"[run_nsga2] benchmark=codeflaws model={model} energy_backend={energy_backend.name}")

    if not args.no_download and not benchmark.is_downloaded():
        print(f"[run_nsga2] downloading codeflaws -> {benchmark.root}")
        benchmark.download()

    all_problems = list(benchmark.problems())
    train_problems = _select_training_subset(all_problems, args.train_size, args.seed)
    print(f"[run_nsga2] training subset: {len(train_problems)}/{len(all_problems)} problems")
    print(f"[run_nsga2] train ids: {[p.problem_id for p in train_problems]}")

    safe_model = model.replace("/", "_").replace(":", "_")
    out_root = REPO / "outputs" / "ga" / "codeflaws" / safe_model
    if args.run_label:
        # Isolate this run in its own subfolder so it never collides with the
        # canonical results that live directly under <model>/.
        out_root = out_root / args.run_label
    out_root.mkdir(parents=True, exist_ok=True)

    # A run is "canonical-grade" only if it uses a real-measurement energy
    # backend. runtime_proxy / codecarbon-tdp are estimates, not measurements.
    is_real_energy = energy_backend.name in ("apple_silicon", "rapl_sysfs")

    # Manifest for reproducibility.
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "baseline" if args.baseline else "nsga2",
        "model": model,
        "run_label": args.run_label,
        "energy_backend_requested": args.energy_backend,
        "energy_backend_resolved": energy_backend.name,
        "energy_is_real_measurement": is_real_energy,
        "pop": args.pop,
        "gen": args.gen,
        "reps": args.reps,
        "train_size": len(train_problems),
        "seed": args.seed,
        "train_ids": [p.problem_id for p in train_problems],
        "search_space": [d["path"] for d in SEARCH_SPACE],
        # Constraint formulation (Deb 2002 constrained domination via pymoo G).
        "test_suite": (
            "official: positive cases enumerated by each problem's test-valid.sh "
            "(NOT every heldout-input-pos* file on disk, which is ~1.09x larger); "
            "quarantined unsound cases excluded, see benchmarks/codeflaws/adapter.py "
            "_QUARANTINED_TESTS"
        ),
        "feasibility_criterion": "per problem: correctness > 0 (compiles AND passes >=1 test)",
        "min_feasible_threshold": args.min_feasible,
        "objective_aggregation": (
            "F1 (complexity) and F3 (energy) averaged over FEASIBLE problems only; "
            "F2 (correctness) averaged over ALL problems. No compile-failure penalty "
            "is folded into F1/F3 — feasibility is handled as an NSGA-II constraint."
        ),
        "temperature_handling": "searched per agent (pi_a/pi_r/pi_o, range 0-1), not fixed",
        "reported_front": "filtered to feasible (G<=0) solutions only",
        # NSGA-II operators. mutation_prob is pymoo's PER-INDIVIDUAL probability;
        # the per-gene rate is the pymoo default 1/n_var.
        "crossover_prob": CROSSOVER_PROB,
        "mutation_prob_per_individual": MUTATION_PROB,
        "mutation_prob_per_variable": round(1.0 / len(SEARCH_SPACE), 4),
        "sbx_eta": SBX_ETA,
        "pm_eta": PM_ETA,
        # Parallelisation settings — recorded for reproducibility.
        "n_workers": _N_WORKERS,
        "llm_backend": "ollama",
        # F3 energy methodology (see docs/reports/09 §2.7).
        "energy_measurement": (
            "whole-batch RAPL window (v3.1): Phase B reruns only the already-passing "
            "tests of the accepted artifact inside ONE energy window and divides by "
            "executions performed; small batches are repeated to clear the counter "
            "resolution floor. F3 = MEAN energy per test over PASSING tests only "
            "(timed-out/failing/non-reproducing executions excluded); run_timeout=3s per test execution"
        ),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # If this run does not use a real-measurement energy backend, drop a loud
    # marker so the folder self-documents as a demo / non-canonical run and
    # nobody mistakes its F3 numbers for measured energy.
    if not is_real_energy:
        marker = (
            "# NOT A CANONICAL RESULT\n\n"
            f"Energy backend: **{energy_backend.name}** — this is an ESTIMATE/PROXY,\n"
            "not a hardware energy measurement.\n\n"
            f"- model: {model}\n"
            f"- run_label: {args.run_label}\n"
            f"- config: pop={args.pop} gen={args.gen} reps={args.reps} "
            f"train_size={len(train_problems)}\n\n"
            "F1 (cyclomatic complexity) and F2 (correctness) here are real; F3\n"
            "(energy) is a proxy and is NOT comparable to the Linux+RAPL lab\n"
            "results. Use this run to demonstrate the GA pipeline works end-to-\n"
            "end, not for energy claims. Canonical results live under the bare\n"
            "`outputs/ga/codeflaws/<model>/` folder (real RAPL energy).\n"
        )
        (out_root / "_DEMO_NOT_REAL_ENERGY.md").write_text(marker, encoding="utf-8")

    t0 = time.time()
    if args.baseline:
        run_baseline(train_problems, _build_config(out_root, REPO), out_root, reps=args.reps)
    else:
        run_search(train_problems, _build_config(out_root, REPO), args, out_root)
    elapsed = time.time() - t0

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["elapsed_seconds"] = elapsed
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[run_nsga2] done in {elapsed:.1f}s -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
