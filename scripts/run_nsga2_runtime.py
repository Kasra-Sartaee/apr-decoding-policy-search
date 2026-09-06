#!/usr/bin/env python3
"""NSGA-II over LLM decoding policy — local Ollama inference, parallel verify.

F3 is supplied by whichever backend `--energy-backend` resolves to. It defaults
to `runtime_proxy`, under which F3 is wall-clock runtime per passing test rather
than hardware energy; `rapl_sysfs` and `apple_silicon` give real measured energy.
The distinction matters for concurrency: runtime is attributed per process by the
OS, so proxy verifications may run concurrently without cross-contamination,
whereas a RAPL reading is package-wide and requires the serialised Phase B window
described in `docs/reports/measurement_protocol.md`. The manifest records both
the requested and the resolved backend, and `energy_is_real_measurement`.

PARALLELISM MODEL
─────────────────
  • N_LOCAL_WORKERS candidate threads run in parallel (default 4 — matches
    Ollama's auto-detected parallel slot count on the RTX 2080 Ti with 1.5B Q4).
  • Each thread evaluates its candidate policy over all N_train problems
    sequentially (pipeline: analyze → repair → verify → [retry] → optimize →
    verify_opt).
  • Across threads, all stages — including verify — run concurrently. With 4
    threads × 15 problems this means up to 4 simultaneous gcc compile and
    test-suite runs, which fits within 6 CPU cores.

F3 PROXY NOTE
─────────────
Under `runtime_proxy`, energy_mj is wall_time_s × TDP_W × 1000, TDP_W defaulting
to 15 W and overridable through `ENERGY_PROXY_TDP_W`. The value is proportional
to runtime and meaningful for relative comparison within one run, but is not
comparable with the RAPL measurements the reported searches used; the manifest
records energy_is_real_measurement=False in that case.

OUTPUT FORMAT
─────────────
manifest.json, rep_N/pareto_solutions.json, rep_N/intermediate_solutions.json,
rep_N/checkpoint.json.

USAGE
─────
  N_LOCAL_WORKERS=4 \\
  python scripts/run_nsga2_runtime.py \\
      --model qwen2.5-coder:1.5b \\
      --pop 50 --gen 15 --train-size 15 \\
      --seed 42 \\
      --run-label rt_rep1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

UTC = UTC

# Resolve --model before importing agents so DEFAULT_MODEL is set correctly:
# agents.py binds DEFAULT_MODEL at import time from AGENT_MODEL.
_early_model = None
for _i, _a in enumerate(sys.argv):
    if _a == "--model" and _i + 1 < len(sys.argv):
        _early_model = sys.argv[_i + 1]
    elif _a.startswith("--model="):
        _early_model = _a.split("=", 1)[1]

# Only enforce when this file is the program being run. evaluate_heldout.py
# imports this module as a library and sets AGENT_MODEL itself, so a module-level
# exit here would break it — sys.argv then belongs to the importing script.
_IS_ENTRYPOINT = Path(sys.argv[0]).name == "run_nsga2_runtime.py"

if _early_model:
    os.environ["AGENT_MODEL"] = _early_model
elif _IS_ENTRYPOINT and not any(_a in ("-h", "--help") for _a in sys.argv):
    # FAIL CLOSED. agents.py defaults AGENT_MODEL to qwen2.5-coder:7b, while
    # evaluate_heldout.py defaults it to 1.5b — the two halves of one
    # experiment disagreed, and a run launched without --model silently took
    # the 7B treatment while being compared against 1.5B arms. The model IS
    # the treatment; refuse to infer it. (Same principle as load_contract's
    # fail-closed behaviour: an implicit treatment is an invisible one.)
    sys.exit(
        "run_nsga2_runtime.py: --model is required.\n"
        "  The model is part of the experimental treatment and is never inferred.\n"
        "  Completed arms used: qwen2.5-coder:1.5b\n"
        "  e.g. --model qwen2.5-coder:1.5b"
    )

from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.core.callback import Callback  # noqa: E402
from pymoo.core.problem import ElementwiseProblem  # noqa: E402
from pymoo.core.sampling import Sampling  # noqa: E402
from pymoo.operators.crossover.sbx import SBX  # noqa: E402
from pymoo.operators.mutation.pm import PM  # noqa: E402
from pymoo.operators.sampling.rnd import FloatRandomSampling  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

from agentic_workflow import (  # noqa: E402
    DEFAULT_MODEL,
    _report_has_no_issues,
    evaluate_fitness,
    is_no_safe_transformation,
    optimized_preserves_correctness,
    set_energy_backend,
)
from agents import (  # noqa: E402
    AnalyzerAgent,
    OptimizerAgent,
    RepairAgent,
)
from benchmarks import get_benchmark  # noqa: E402
from energy import resolve_backend  # noqa: E402
from provenance import write_protocol_manifest  # noqa: E402
from search_space import SEARCH_SPACE, build_policy_from_vector  # noqa: E402

# ── NSGA-II operator settings (identical to run_nsga2.py) ──────────────────────
SBX_ETA = 15
PM_ETA  = 20
CROSSOVER_PROB = 0.8
MUTATION_PROB  = 0.2

# ── Parallelism ───────────────────────────────────────────────────────────────
# 4 workers matches the Ollama parallel-slot capacity measured on this machine
# (RTX 2080 Ti, 10.6 GB free VRAM, 1.5B Q4 model) — see bench_workers.py.
_N_LOCAL_WORKERS = int(os.environ.get("N_LOCAL_WORKERS", "4"))

_print_lock = threading.Lock()


# ── Fitness placeholders ──────────────────────────────────────────────────────
_NO_FEASIBLE_CL     = 0.0
_NO_FEASIBLE_RUNTIME = 0.0


# ── Per-problem pipeline state ────────────────────────────────────────────────

@dataclass
class _ProblemState:
    # Immutable inputs
    problem: Any
    source: str
    language: str
    no_main: bool
    relative: str
    path_report: Path
    path_repaired: Path
    path_optimized: Path
    run_test: Any          # callable: (Path) -> VerificationResult
    retries_left: int
    opt_retries_left: int

    # Mutable pipeline state
    report: str = ""
    repaired_code: str = ""
    verify_result: Any = None
    opt_code: str | None = None
    cl: float = float("inf")
    correctness: float = 0.0
    energy: float = float("inf")   # mJ, from the resolved energy backend
    error: Exception | None = None

    # ── Phase A → Phase B handoff (two-phase measurement, the measurement-protocol report) ──────────
    # Phase A settles WHICH artifact was accepted and WHICH tests it passes,
    # measuring no energy at all. Phase B then reruns exactly those tests once,
    # serially, inside a single energy window.
    #
    # `phase_b_runner` is None when the benchmark's TestRunner has no
    # .diagnostic() method — the legacy single-shot path, where Phase A already
    # produced real energy and Phase B is a no-op passthrough.
    phase_b_runner: Any = None
    final_source_path: Path | None = None
    final_verify: Any = None
    passing_test_ids: list = field(default_factory=list)

    # Audit trail, persisted per candidate evaluation. The artifact files
    # themselves are overwritten by the next candidate, so this is the only
    # durable record of what was measured.
    phase_b_mismatched: list = field(default_factory=list)
    phase_b_reasons: dict = field(default_factory=dict)
    retries_used: int = 0
    opt_retries_used: int = 0

    @property
    def failed(self) -> bool:
        return self.error is not None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _finalise_from_repaired(st: _ProblemState) -> None:
    if st.failed or st.verify_result is None or not st.verify_result.compiled:
        return
    try:
        st.cl, st.correctness, st.energy = evaluate_fitness(
            st.path_repaired, st.verify_result
        )
        # Record the accepted artifact so Phase B measures the same source that
        # F1 was computed from — the three objectives must describe one artifact.
        st.final_source_path = st.path_repaired
        st.final_verify = st.verify_result
        st.passing_test_ids = list(getattr(st.verify_result, "passing_test_ids", []))
    except Exception as exc:
        st.error = exc


def _finalise_from_optimized(st: _ProblemState, verify_opt: Any) -> None:
    try:
        st.cl, st.correctness, st.energy = evaluate_fitness(
            st.path_optimized, verify_opt
        )
        st.final_source_path = st.path_optimized
        st.final_verify = verify_opt
        st.passing_test_ids = list(getattr(verify_opt, "passing_test_ids", []))
    except Exception as exc:
        st.error = exc
        _finalise_from_repaired(st)


def _build_config(out_root: Path, base_dir: Path) -> dict[str, Any]:
    return {
        "base_dir":          str(base_dir),
        "programs_dir":      "programs",
        "tests_dir":         "tests",
        "input_file":        "unused-when-problem-supplied",
        "reports_dir":       str(out_root / "reports"),
        "repaired_dir":      str(out_root / "repaired"),
        "optimized_dir":     str(out_root / "optimized"),
        "verification_json": str(out_root / "verification"),
        "model":             DEFAULT_MODEL,
        "llm_timeout":       180,
        "compile_timeout":   40,
        "run_timeout":       3,
    }


def _select_training_subset(problems: list, train_size: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    n   = min(train_size, len(problems))
    idx = rng.choice(len(problems), size=n, replace=False)
    return [problems[i] for i in sorted(idx.tolist())]


def _make_state(problem: Any, config: dict, policy: dict | None) -> _ProblemState:
    base = Path(config["base_dir"]).resolve()
    rel  = problem.relative_label
    ext  = getattr(problem, "source_extension", ".c")

    retry_budget     = int(policy["pi_v"].get("retry", 0)) if policy else 4
    opt_retry_budget = int(policy["pi_o"].get("retry", 0)) if policy else 4

    # Two-phase resolution, mirroring agentic_workflow.run_pipeline_phase_a:
    # if the runner exposes .diagnostic(), Phase A uses it (no energy measured
    # at all) and Phase B measures once at the end. Otherwise fall back to the
    # runner's __call__, which measures inline, and Phase B becomes a no-op.
    test_runner = problem.test_runner
    _diagnostic = getattr(test_runner, "diagnostic", None)

    return _ProblemState(
        problem=problem,
        source=Path(problem.buggy_source_path).read_text(encoding="utf-8"),
        language=getattr(problem, "language_hint", "C99"),
        no_main=getattr(problem, "harness_concatenated", False),
        relative=rel,
        path_report=   (base / config["reports_dir"]).resolve()  / f"{rel}.json",
        path_repaired= (base / config["repaired_dir"]).resolve() / f"{rel}{ext}",
        path_optimized=(base / config["optimized_dir"]).resolve()/ f"{rel}{ext}",
        run_test=_diagnostic if _diagnostic is not None else test_runner,
        phase_b_runner=test_runner if _diagnostic is not None else None,
        retries_left=retry_budget,
        opt_retries_left=opt_retry_budget,
    )


# ── Per-problem pipeline ──────────────────────────────────────────────────────

def _run_problem_pipeline(
    st: _ProblemState,
    analyzer: AnalyzerAgent,
    repairer: RepairAgent,
    optimizer: OptimizerAgent,
) -> None:
    """Full analyze→repair→verify→optimize pipeline for one problem.

    No verify_lock here — runtime_proxy measures wall time per subprocess,
    which is per-process and requires no shared hardware counter.  All worker
    threads can compile and run tests concurrently with zero interference.
    """
    try:
        # Analyze
        st.report = analyzer.analyze(st.source, st.relative, language=st.language)
        _write_text(st.path_report, st.report)

        # Repair
        if _report_has_no_issues(st.report):
            st.repaired_code = st.source
        else:
            st.repaired_code = repairer.repair(
                st.source, st.report, st.relative,
                language=st.language, no_main=st.no_main,
            )
        _write_text(st.path_repaired, st.repaired_code)

        # Verify (no lock — safe for concurrent execution)
        st.verify_result = st.run_test(st.path_repaired)

        # Retry loop
        report_had_issues = not _report_has_no_issues(st.report)
        while (
            report_had_issues
            and st.verify_result is not None
            and not st.verify_result.is_success
            and st.retries_left > 0
        ):
            st.repaired_code = repairer.repair(
                st.source, st.report, st.relative,
                verification_report=st.verify_result,
                language=st.language, no_main=st.no_main,
            )
            _write_text(st.path_repaired, st.repaired_code)
            st.verify_result = st.run_test(st.path_repaired)
            st.retries_left -= 1
            st.retries_used += 1

        if st.verify_result is None or not st.verify_result.compiled:
            _finalise_from_repaired(st)
            return

        # Optimize
        opt = optimizer.optimize(
            st.repaired_code, st.report, st.verify_result,
            st.relative, language=st.language, no_main=st.no_main,
        )

        # ONE parser for every optimiser response, shared with run_nsga2.py. An
        # exact string compare let a Markdown-wrapped `**NO_SAFE_TRANSFORMATION**`
        # through, which was then written to disk as if it were source.
        if is_no_safe_transformation(opt):
            _finalise_from_repaired(st)
            return

        st.opt_code = opt
        _write_text(st.path_optimized, opt)

        verify_opt = st.run_test(st.path_optimized)

        opt_retries = st.opt_retries_left
        while not verify_opt.compiled and opt_retries > 0:
            opt = optimizer.optimize(
                st.repaired_code, st.report, verify_opt,
                st.relative, language=st.language, no_main=st.no_main,
            )
            _write_text(st.path_optimized, opt)
            verify_opt = st.run_test(st.path_optimized)
            st.opt_code = opt
            opt_retries -= 1
            st.opt_retries_used += 1

        # Accept the optimised artifact only if it preserves the EXACT set of
        # passing tests. Checking `compiled` alone (the previous behaviour) let
        # correctness-damaging optimisations through: measured over ~184k
        # repair->optimise transitions, 21.5% made correctness WORSE, median
        # loss ~51pp, and 21.4% of fully-correct repairs were destroyed
        # outright. See the measurement-protocol report section 3.3.
        accepted, why = optimized_preserves_correctness(st.verify_result, verify_opt)
        if accepted:
            _finalise_from_optimized(st, verify_opt)
        else:
            if verify_opt.compiled:
                with _print_lock:
                    print(f"    [opt-reject] {st.problem.problem_id}: {why}")
            _finalise_from_repaired(st)

    except Exception as exc:
        st.error = exc


_BENCHMARK_LANGUAGE = {"codeflaws": "C99", "runbugrun_py": "Python3"}


def _write_protocol_manifest(rep_dir: Path, args, seed: int) -> None:
    """Per-repetition provenance, shared with run_nsga2.py via provenance.py.

    `measurement_protocol` is benchmark-qualified because the two arms run the
    same two-phase protocol over different languages and toolchains; the label
    must make a Python rep distinguishable from a C one at a glance.
    """
    language = _BENCHMARK_LANGUAGE.get(args.benchmark, "C99")

    # Timeouts ACTUALLY in force, taken from the adapter that owns them rather
    # than this driver's nominal config dict.
    if args.benchmark == "runbugrun_py":
        from benchmarks.runbugrun_py.adapter import RUNBUGRUN_RUN_TIMEOUT_S
        timeouts = {
            "llm_request_s": float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "300")),
            "compile_s": None,  # interpreted; no compile step
            "run_s": RUNBUGRUN_RUN_TIMEOUT_S,
        }
    else:
        from benchmarks.codeflaws.adapter import (
            CODEFLAWS_COMPILE_TIMEOUT_S,
            CODEFLAWS_RUN_TIMEOUT_S,
        )
        timeouts = {
            "llm_request_s": float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "300")),
            "compile_s": CODEFLAWS_COMPILE_TIMEOUT_S,
            "run_s": CODEFLAWS_RUN_TIMEOUT_S,
        }

    write_protocol_manifest(
        rep_dir,
        measurement_protocol=f"{args.benchmark}_two_phase_v3.1",
        protocol_doc="docs/reports/measurement_protocol.md",
        language=language,
        n_workers=_N_LOCAL_WORKERS,
        timeouts_effective=timeouts,
        seed=seed,
        seed_increment=args.seed_increment,
        model=args.model or DEFAULT_MODEL,
        pop=args.pop,
        gen=args.gen,
        train_size=args.train_size,
        extra={
            "benchmark": args.benchmark,
            "energy_backend": args.energy_backend,
            "init_sampling": args.sampling,
            "test_suite": (
                "full upstream suite, uncapped"
                if args.benchmark == "runbugrun_py"
                else "official test-valid.sh positive cases, minus _QUARANTINED_TESTS"
            ),
        },
    )


def _run_problem_phase_b(st: _ProblemState) -> None:
    """Measure F3 for one already-settled problem, and revise F2.

    MUST be called serially, after every Phase A worker has joined. Nothing
    else in the process may be running: the energy window is only valid if no
    other thread is doing inference or executing tests at the same time.

    A no-op when the runner is not diagnostic-capable (Phase A already produced
    real energy) or when Phase A never settled on an artifact (compile failure
    persisted, so the worst-case fitness already stands).
    """
    if st.failed or st.phase_b_runner is None or st.final_source_path is None:
        return
    try:
        m = st.phase_b_runner.measure_energy(
            st.final_source_path, st.passing_test_ids
        )
    except Exception as exc:
        st.error = exc
        return

    # F2 revision: tests that passed in Phase A but did not reproduce here are
    # demoted. This makes correctness a two-execution property, matching the C
    # arm. Denominator stays the full suite.
    st.phase_b_mismatched = list(m.mismatched_ids)
    st.phase_b_reasons = dict(m.mismatch_reasons)

    total = getattr(st.final_verify, "tests_total", 0) or 0
    revised_passed = max(0, len(st.passing_test_ids) - len(m.mismatched_ids))
    st.correctness = (revised_passed / total) if total > 0 else 0.0

    if m.mean_energy_mj is None:
        # inf, never 0.0 — F3 is minimised, so 0.0 would be an unbeatable
        # score. inf is excluded by the np.isfinite feasibility filter below.
        with _print_lock:
            print(
                f"    [phase-b] {st.problem.problem_id}: energy measurement "
                f"invalid ({len(m.matched_ids)}/{len(st.passing_test_ids)} tests "
                f"reproduced) — energy=inf, excluded from the feasible aggregate."
            )
        st.energy = float("inf")
    else:
        st.energy = m.mean_energy_mj


def _mean_fitness_runtime(
    problems: list,
    config: dict,
    policy: dict | None,
) -> dict[str, Any]:
    """Evaluate `policy` over `problems` using parallel local Ollama threads.

    Problems are distributed across _N_LOCAL_WORKERS threads.  Each thread
    handles one problem at a time; all threads run concurrently including
    their verification steps (no verify_lock).
    """
    if policy is None:
        analyzer  = AnalyzerAgent()
        repairer  = RepairAgent()
        optimizer = OptimizerAgent()
    else:
        analyzer  = AnalyzerAgent(config=policy["pi_a"])
        repairer  = RepairAgent(config=policy["pi_r"])
        optimizer = OptimizerAgent(config=policy["pi_o"])

    states = [_make_state(p, config, policy) for p in problems]

    def _dispatch(st: _ProblemState) -> None:
        _run_problem_pipeline(st, analyzer, repairer, optimizer)

    # ── Phase A: concurrent analyze/repair/retry/optimize, NO energy ──────
    if _N_LOCAL_WORKERS <= 1:
        for st in states:
            _dispatch(st)
    else:
        with ThreadPoolExecutor(
            max_workers=min(_N_LOCAL_WORKERS, len(states)),
        ) as pool:
            futures = {pool.submit(_dispatch, st): st for st in states}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    futures[fut].error = exc

    # ── Phase B: serial, measured ────────────────────────────────────────
    # The `with` block above has fully exited, so ThreadPoolExecutor.shutdown
    # has joined every Phase A worker. THAT join is the barrier — nothing else
    # in this process is doing inference or running tests during the loop
    # below, which is what makes each energy window valid. A no-op for runners
    # without .diagnostic(), which already measured inline in Phase A.
    for st in states:
        _run_problem_phase_b(st)

    # ── Aggregate in original problem order ────────────────────────────────
    corrs: list[float] = []
    feasible_cls: list[float] = []
    feasible_runtimes: list[float] = []
    per_problem: list[dict] = []
    n_feasible = 0

    for st in states:
        if st.failed:
            with _print_lock:
                print(f"    [warn] {st.problem.problem_id} raised: {st.error}")
            cl, corr, runtime = float("inf"), 0.0, float("inf")
        else:
            cl, corr, runtime = st.cl, st.correctness, st.energy

        feasible = bool(corr > 0.0 and np.isfinite(cl) and np.isfinite(runtime))
        corrs.append(corr)
        if feasible:
            n_feasible += 1
            feasible_cls.append(cl)
            feasible_runtimes.append(runtime)
        # Everything needed to audit this evaluation after the fact. The
        # repaired/optimized artifact files are overwritten by the next
        # candidate, so if the evidence is not captured here it is gone: which
        # artifact was accepted, which tests it passed in Phase A, and what
        # failed to reproduce in Phase B all become unrecoverable.
        accepted = None
        source_sha = None
        if st.final_source_path is not None and st.final_source_path.is_file():
            accepted = (
                "optimized" if st.final_source_path == st.path_optimized else "repaired"
            )
            try:
                source_sha = hashlib.sha256(
                    st.final_source_path.read_bytes()
                ).hexdigest()
            except OSError:
                source_sha = None

        per_problem.append({
            "problem_id":         st.problem.problem_id,
            "group_id":           (st.problem.metadata or {}).get(
                                      "problem_id", st.problem.problem_id),
            "cl":                 cl      if np.isfinite(cl)      else None,
            "correctness":        corr,
            "runtime_mj":         runtime if np.isfinite(runtime) else None,
            "feasible":           feasible,
            "accepted_artifact":  accepted,
            "accepted_sha256":    source_sha,
            "phase_a_passing_ids": list(st.passing_test_ids),
            "phase_b_mismatched":  list(st.phase_b_mismatched),
            "phase_b_reasons":     dict(st.phase_b_reasons),
            "repair_retries_used": st.retries_used,
            "opt_retries_used":    st.opt_retries_used,
            "error":               repr(st.error) if st.error else None,
        })

    return {
        "mean_cl":          float(np.mean(feasible_cls))       if feasible_cls       else _NO_FEASIBLE_CL,
        "mean_correctness": float(np.mean(corrs))              if corrs              else 0.0,
        "mean_energy":      float(np.mean(feasible_runtimes))  if feasible_runtimes  else _NO_FEASIBLE_RUNTIME,
        "n_feasible":       n_feasible,
        "n_total":          len(problems),
        "per_problem":      per_problem,
    }


# ── NSGA-II problem wrapper ───────────────────────────────────────────────────

class AgentCodeflawsProblemRuntime(ElementwiseProblem):
    def __init__(self, train_problems: list, config: dict, min_feasible: int,
                 evaluations_dir: Path | None = None) -> None:
        self.train_problems = train_problems
        self.config         = config
        self.min_feasible   = min_feasible
        self._eval_count    = 0
        # Per-candidate audit trail. Without it the only surviving record of
        # an evaluation is its aggregate fitness: the repaired/optimized
        # artifacts are overwritten by the next candidate, so a Phase B
        # mismatch or artifact-selection decision becomes unauditable.
        self.evaluations_dir = evaluations_dir
        lower = [d["low"]  for d in SEARCH_SPACE]
        upper = [d["high"] for d in SEARCH_SPACE]
        super().__init__(
            n_var=len(SEARCH_SPACE), n_obj=3, n_ieq_constr=1,
            xl=lower, xu=upper,
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        self._eval_count += 1
        policy = build_policy_from_vector(x)
        with _print_lock:
            print(f"  [candidate {self._eval_count}] policy={json.dumps(policy)}")
        agg = _mean_fitness_runtime(self.train_problems, self.config, policy)
        with _print_lock:
            print(
                f"    -> n_feasible={agg['n_feasible']}/{agg['n_total']} "
                f"mean_cl={agg['mean_cl']:.2f} "
                f"mean_corr={agg['mean_correctness']:.3f} "
                f"mean_runtime_mj={agg['mean_energy']:.0f}"
            )
        if self.evaluations_dir is not None:
            try:
                self.evaluations_dir.mkdir(parents=True, exist_ok=True)
                rec = {
                    "candidate":   self._eval_count,
                    "evaluated_at": datetime.now(UTC).isoformat(),
                    "policy":      policy,
                    "F":           [agg["mean_cl"], -agg["mean_correctness"],
                                    agg["mean_energy"]],
                    "n_feasible":  agg["n_feasible"],
                    "n_total":     agg["n_total"],
                    "per_problem": agg["per_problem"],
                }
                path = self.evaluations_dir / f"cand_{self._eval_count:05d}.json"
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(rec, indent=2, default=str),
                               encoding="utf-8")
                tmp.replace(path)
            except Exception as exc:   # never let auditing kill a run
                with _print_lock:
                    print(f"    [warn] could not persist evaluation record: {exc}")

        out["F"] = [agg["mean_cl"], -agg["mean_correctness"], agg["mean_energy"]]
        out["G"] = [float(self.min_feasible - agg["n_feasible"])]


# ── Checkpointing / output ────────────────────────────────────────────────────

class _WarmStartSampling(Sampling):
    def __init__(self, X: np.ndarray) -> None:
        super().__init__()
        self._X = X

    def _do(self, problem, n_samples, **kwargs):
        return self._X


class _GenCheckpoint(Callback):
    def __init__(self, checkpoint_path: Path, history_acc: list,
                 gen_offset: int = 0, total_gen: int = 0) -> None:
        super().__init__()
        self._path    = checkpoint_path
        self._history = history_acc
        self._offset  = gen_offset
        self._total   = total_gen

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
                "policy":     build_policy_from_vector(X[j]),
                "feasible":   feasible,
                "fitness": {
                    "cl":          float(F[j][0]),
                    "correctness": float(-F[j][1]),
                    "energy":      float(F[j][2]),  # mJ
                },
            })
        ckpt = {
            "generation": gen,
            "pop_X":      X.tolist(),
            "pop_F":      F.tolist() if F is not None else None,
            "pop_G":      G.tolist() if G is not None else None,
            "history":    self._history,
        }
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(ckpt), encoding="utf-8")
            tmp.rename(self._path)
            print(f"[run_nsga2_runtime] checkpoint saved gen {gen}/{self._total} -> {self._path}")
        except Exception as exc:
            print(f"[run_nsga2_runtime] WARNING: checkpoint save failed: {exc}")


def _write_rep_outputs(
    rep_idx: int, args: argparse.Namespace,
    X: np.ndarray, F: np.ndarray, G: Any,
    history_acc: list,
    pareto_path: Path, history_path: Path, checkpoint_path: Path,
) -> None:
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
                "cl":          float(F[k][0]),
                "correctness": float(-F[k][1]),
                "energy":      float(F[k][2]),
            },
        }
        for k in range(len(X)) if feasible_mask[k]
    ]
    n_dropped = int((~feasible_mask).sum())
    pareto_path.write_text(json.dumps(pareto,      indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history_acc, indent=2), encoding="utf-8")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("[run_nsga2_runtime] checkpoint removed (run complete)")

    if not pareto:
        print(
            f"[run_nsga2_runtime] rep {rep_idx + 1}: 0 FEASIBLE Pareto solutions "
            f"(all candidates solved < {args.min_feasible} problems). "
            f"Empty front written -> {pareto_path}"
        )
    else:
        print(
            f"[run_nsga2_runtime] rep {rep_idx + 1}: {len(pareto)} feasible Pareto "
            f"solutions ({n_dropped} infeasible dropped) -> {pareto_path}"
        )


# ── Baseline + search ─────────────────────────────────────────────────────────

def run_baseline(train_problems: list, config: dict, out_root: Path) -> None:
    print("[run_nsga2_runtime] evaluating DEFAULT-policy baseline on training subset")
    agg = _mean_fitness_runtime(train_problems, config, policy=None)
    out = {
        "kind":             "default_policy_baseline",
        "n_train":          len(train_problems),
        "n_feasible":       agg["n_feasible"],
        "mean_cl":          agg["mean_cl"],
        "mean_correctness": agg["mean_correctness"],
        "mean_energy":      agg["mean_energy"],   # mJ
        "mean_energy_mj":   agg["mean_energy"],
        "per_problem":      agg["per_problem"],
    }
    path = out_root / "training_baseline" / "policy_default.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"aggregate": out}, indent=2), encoding="utf-8")
    print(
        f"[run_nsga2_runtime] baseline: n_feasible={agg['n_feasible']}/{agg['n_total']} "
        f"mean_cl={agg['mean_cl']:.2f} "
        f"mean_corr={agg['mean_correctness']:.3f} "
        f"mean_runtime_mj={agg['mean_energy']:.0f} -> {path}"
    )


def run_search(train_problems: list, config: dict, args: argparse.Namespace,
               out_root: Path) -> None:
    for rep_idx in range(args.reps):
        rep_dir         = out_root / f"rep_{rep_idx}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        pareto_path     = rep_dir / "pareto_solutions.json"
        history_path    = rep_dir / "intermediate_solutions.json"
        checkpoint_path = rep_dir / "checkpoint.json"

        if not args.overwrite and pareto_path.exists() and history_path.exists():
            print(f"[run_nsga2_runtime] rep {rep_idx + 1}/{args.reps} already done; skipping")
            continue

        # Per-repetition provenance. Written BEFORE the checkpoint is loaded and
        # never overwritten, so a repetition whose generations spanned a code
        # change stays visible rather than being retroactively relabelled.
        _write_protocol_manifest(rep_dir, args, seed=args.seed + rep_idx * args.seed_increment)

        start_gen   = 0
        history_acc: list = []
        sampling    = None
        restart_log_path = rep_dir / "restart_log.json"

        if checkpoint_path.exists() and not args.overwrite:
            try:
                ckpt        = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                start_gen   = int(ckpt["generation"])
                history_acc = list(ckpt.get("history", []))
                sampling    = _WarmStartSampling(np.array(ckpt["pop_X"]))
                print(
                    f"[run_nsga2_runtime] rep {rep_idx + 1}: checkpoint at gen "
                    f"{start_gen}/{args.gen} — resuming"
                )
                prior = (
                    json.loads(restart_log_path.read_text(encoding="utf-8"))
                    if restart_log_path.exists() else []
                )
                prior.append({
                    "resumed_at":              datetime.now(UTC).isoformat(),
                    "resumed_from_generation": start_gen,
                    "total_generations":       args.gen,
                })
                restart_log_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")
            except Exception as exc:
                print(f"[run_nsga2_runtime] checkpoint load failed ({exc}); starting fresh")
                start_gen = 0; history_acc = []; sampling = None

        # If starting fresh (no checkpoint resume), draw the initial population.
        if sampling is None and start_gen == 0:
            sampling = FloatRandomSampling()
            print("[run_nsga2_runtime] init sampling: random")

        remaining_gens = args.gen - start_gen
        if remaining_gens <= 0:
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
            f"[run_nsga2_runtime] === repetition {rep_idx + 1}/{args.reps} "
            f"(seed={args.seed + rep_idx}) — gen {start_gen + 1}..{args.gen} ==="
        )
        problem = AgentCodeflawsProblemRuntime(
            train_problems, config, args.min_feasible,
            evaluations_dir=rep_dir / "evaluations",
        )
        problem._eval_count = start_gen * args.pop

        callback = _GenCheckpoint(
            checkpoint_path, history_acc,
            gen_offset=start_gen, total_gen=args.gen,
        )

        alg_kw: dict = dict(
            pop_size=args.pop,
            crossover=SBX(eta=SBX_ETA, prob=CROSSOVER_PROB),
            mutation=PM(eta=PM_ETA,  prob=MUTATION_PROB),
        )
        if sampling is not None:
            alg_kw["sampling"] = sampling

        result = minimize(
            problem,
            NSGA2(**alg_kw),
            termination=("n_gen", remaining_gens),
            seed=args.seed + rep_idx * args.seed_increment,
            save_history=False,
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Ollama model tag. REQUIRED — the model is part of "
                             "the experimental treatment and is never inferred "
                             "(e.g. qwen2.5-coder:1.5b).")
    parser.add_argument("--pop",           type=int, default=20)
    parser.add_argument("--gen",           type=int, default=5)
    parser.add_argument("--reps",          type=int, default=1)
    parser.add_argument("--train-size",    type=int, default=15)
    parser.add_argument("--min-feasible",  type=int, default=1)
    parser.add_argument("--seed",          type=int, default=42,
                        help="Base RNG seed; rep i uses seed + i*seed_increment. "
                             "Also seeds the training-subset selection, which uses "
                             "--seed alone so every repetition trains on the SAME "
                             "problems.")
    parser.add_argument(
        "--seed-increment", type=int, default=1,
        help="Per-repetition seed step (default 1: rep i uses seed+i, so each "
             "repetition explores a different GA search trajectory in addition "
             "to LLM-inherent stochasticity). Set to 0 to hold the GA's own RNG "
             "seed FIXED across all repetitions, isolating LLM sampling "
             "stochasticity as the only source of run-to-run variance.",
    )
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--baseline",      action="store_true")
    parser.add_argument("--no-download",   action="store_true")
    parser.add_argument(
        "--benchmark", default="codeflaws",
        choices=["codeflaws", "runbugrun_py"],
        help="Benchmark to optimise on (default: codeflaws).",
    )
    parser.add_argument(
        "--energy-backend", default="runtime_proxy",
        choices=["auto", "apple_silicon", "rapl_sysfs", "codecarbon", "runtime_proxy"],
        help="F3 backend (default: runtime_proxy — wall-clock per test, parallelism-safe)",
    )
    parser.add_argument("--run-label", default=None,
                        help="Sub-directory label under the model output dir")
    parser.add_argument(
        "--sampling", default="random", choices=["random"],
        help="Initial population sampling strategy. Uniform random is the only "
             "strategy used by the reported runs and the only one retained.",
    )
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL
    if args.model:
        os.environ["AGENT_MODEL"] = args.model

    energy_backend = resolve_backend(args.energy_backend)
    set_energy_backend(energy_backend)

    benchmark = get_benchmark(args.benchmark)

    is_real_energy = energy_backend.name in ("apple_silicon", "rapl_sysfs")
    print(
        f"[run_nsga2_runtime] benchmark={args.benchmark} model={model} "
        f"energy_backend={energy_backend.name} "
        f"n_local_workers={_N_LOCAL_WORKERS} "
        f"real_energy={is_real_energy}"
    )

    if not args.no_download and not benchmark.is_downloaded():
        print(f"[run_nsga2_runtime] downloading {args.benchmark} -> {benchmark.root}")
        benchmark.download()

    all_problems   = list(benchmark.problems())
    train_problems = _select_training_subset(all_problems, args.train_size, args.seed)
    print(f"[run_nsga2_runtime] training subset: {len(train_problems)}/{len(all_problems)} problems")
    print(f"[run_nsga2_runtime] train ids: {[p.problem_id for p in train_problems]}")

    safe_model = model.replace("/", "_").replace(":", "_")
    out_root   = REPO / "outputs" / "ga" / args.benchmark / safe_model
    if args.run_label:
        out_root = out_root / args.run_label
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at":                    datetime.now(UTC).isoformat(),
        "script":                        "run_nsga2_runtime.py",
        "pipeline":                      "per-problem local-ollama-parallel, concurrent verify",
        "mode":                          "baseline" if args.baseline else "nsga2",
        "benchmark":                     args.benchmark,
        "model":                         model,
        "run_label":                     args.run_label,
        "n_local_workers":               _N_LOCAL_WORKERS,
        "energy_backend_requested":      args.energy_backend,
        "energy_backend_resolved":       energy_backend.name,
        "energy_is_real_measurement":    is_real_energy,
        # Derived from the backend that actually resolved, not assumed: this
        # driver accepts rapl_sysfs, and a hardcoded proxy label would record
        # false provenance for a real measurement.
        "f3_note":                       (
            "F3 = measured CPU-package energy per passing test via "
            f"{energy_backend.name}."
            if is_real_energy else
            "F3 = runtime_proxy (wall-clock per passing test × TDP_W × 1000 mJ). "
            "Proportional to runtime; not comparable to RAPL-measured energy runs."
        ),
        "verify_lock":                   False,
        "verify_parallelism":            "concurrent across candidate threads (no lock)",
        "pop":                           args.pop,
        "gen":                           args.gen,
        "reps":                          args.reps,
        "train_size":                    len(train_problems),
        "seed":                          args.seed,
        "train_ids":                     [p.problem_id for p in train_problems],
        "search_space":                  [d["path"] for d in SEARCH_SPACE],
        "feasibility_criterion":         "per problem: correctness > 0",
        "min_feasible_threshold":        args.min_feasible,
        "objective_aggregation": (
            f"F1 (complexity) and F3 ({'measured energy' if is_real_energy else 'runtime proxy'}) "
            "averaged over feasible problems only; "
            "F2 (correctness) averaged over all problems."
        ),
        "crossover_prob":                CROSSOVER_PROB,
        "mutation_prob_per_individual":  MUTATION_PROB,
        "mutation_prob_per_variable":    round(1.0 / len(SEARCH_SPACE), 4),
        "sbx_eta":                       SBX_ETA,
        "pm_eta":                        PM_ETA,
        "init_sampling":                 args.sampling,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not is_real_energy:
        marker = (
            "# NOT A CANONICAL ENERGY RESULT\n\n"
            f"Energy backend: **{energy_backend.name}** — runtime proxy, not hardware measurement.\n\n"
            f"- model: {model}\n- run_label: {args.run_label}\n"
            f"- config: pop={args.pop} gen={args.gen} train_size={len(train_problems)}\n\n"
            "F1 and F2 are real; F3 = wall-clock runtime × TDP constant (proportional to time, "
            "not directly comparable to RAPL runs).\n"
        )
        (out_root / "_DEMO_NOT_REAL_ENERGY.md").write_text(marker, encoding="utf-8")

    t0     = time.time()
    config = _build_config(out_root, REPO)

    if args.baseline:
        run_baseline(train_problems, config, out_root)
    else:
        run_search(train_problems, config, args, out_root)

    elapsed = time.time() - t0
    manifest["finished_at"]     = datetime.now(UTC).isoformat()
    manifest["elapsed_seconds"] = elapsed
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[run_nsga2_runtime] done in {elapsed:.1f}s -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
