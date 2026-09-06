#!/usr/bin/env python3
"""Resumable held-out evaluation of a training Pareto front.

Every policy on the front produced by one search repetition is re-evaluated on a
held-out problem set disjoint from the one the search optimised against. This is
what separates a policy that generalises from one that fits the training split,
and it is run identically for all three arms so their fronts are comparable.

The held-out set is drawn from the benchmark with a fixed seed and is the same
for every arm and repetition. ``--expect-heldout-sha256`` pins its digest and
aborts on any mismatch, so an evaluation cannot silently run on a different set
of problems from the one it is compared against.

The expensive unit of work is one policy over N held-out problems. A completed
policy is written atomically to ``policies/policy_NNN.json`` immediately, so an
interruption loses at most the policy currently in flight. Re-running the same
command resumes from those files; ``--no-resume`` forces re-evaluation.

Examples:

    # Held-out evaluation of one repetition's front, as published.
    .venv/bin/python scripts/evaluate_heldout.py \
        --benchmark codeflaws \
        --run-labels mobo_pop50_gen15_n50_v4 \
        --front-file results/mobo/rep_3/pareto_solutions.json \
        --out-tag heldout_moborep3_n100_seed0 \
        --n-heldout 100 --heldout-seed 0 \
        --workers 4 --energy-backend rapl_sysfs \
        --expect-heldout-sha256 365503a2...

    # Inspect the plan without calling Ollama.
    .venv/bin/python scripts/evaluate_heldout.py --dry-run

``scripts/launch_heldout_mobo_rep3.sh`` shows a complete two-benchmark
invocation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("AGENT_MODEL", "qwen2.5-coder:1.5b")

MODEL = "qwen2.5-coder:1.5b"

# Initialisation variants under comparison. These are the *variant* names and
# never change; the GA run *directory* each maps to is prefix + variant, which
# does change per benchmark (see --run-prefix).
VARIANTS = ["random", "lhs", "llm_warmstart"]
DEFAULT_BENCHMARK = "codeflaws"
DEFAULT_RUN_PREFIX = "rt_n15_"  # Codeflaws labels: rt_n15_random, rt_n15_lhs, …

# Module-level defaults so the helpers below (load_pareto, _output_dir) keep
# working as-is. main() rebinds both once --benchmark / --run-prefix are known.
GA_BASE = REPO / "outputs" / "ga" / DEFAULT_BENCHMARK / MODEL.replace(":", "_")
RUNS = {v: f"{DEFAULT_RUN_PREFIX}{v}" for v in VARIANTS}


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a sibling temporary file, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _policy_key(policy: dict) -> str:
    return json.dumps(policy, sort_keys=True, separators=(",", ":"))


def _ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# Set by main() from --front-file. Overrides where policies come from, WITHOUT
# touching where the held-out problems come from — that stays derived from the
# named run's train_ids. This is what makes cross-language transfer evaluation
# a controlled comparison: identical problems, identical protocol, different
# policies. A decoding policy is only temperatures, top_p/top_k and retry
# budgets, so it carries across languages unchanged.
FRONT_FILE: Path | None = None


def load_pareto(run_label: str) -> list[dict]:
    """Load the complete feasible Pareto front to evaluate."""
    path = FRONT_FILE or (GA_BASE / run_label / "rep_0" / "pareto_solutions.json")
    pareto = json.loads(path.read_text(encoding="utf-8"))
    if not pareto:
        raise RuntimeError(f"empty Pareto front: {path}")
    return pareto


def group_key(problem) -> str:
    """The unit that must not straddle the train/held-out boundary.

    A benchmark can ship several bug records derived from the SAME upstream
    problem — different users' buggy submissions to one contest task, sharing
    that task's entire test suite. Excluding only the training *record* ids
    then lets a sibling record into the held-out set, so the policy is
    evaluated on a problem it was tuned on and held-out uncertainty is
    understated.

    Measured on the 300-record samples:
      runbugrun_py : 152 unique upstream problems (1.97 records each)
      codeflaws    : 261 unique contest problems  (1.15 records each)

    Grouping applies only where the adapter publishes an upstream id in
    `metadata["problem_id"]` — currently RunBugRun alone. Everything else,
    Codeflaws included, falls back to the record id and therefore selects
    EXACTLY what it did before this function existed.

    That asymmetry is deliberate, not an oversight. Codeflaws' residual is 3
    records out of 285 (~1%) against RunBugRun's 30 (~10.5%), and its held-out
    evaluation is already complete; grouping it too would change its selection
    and invalidate finished results to remove a 1% effect. The residual is
    declared in docs/reports/16 threats-to-validity instead. Deriving a
    Codeflaws group from the `<contest>-<letter>-bug-...` directory name is a
    two-line change here if that trade is ever revisited.
    """
    meta = getattr(problem, "metadata", None) or {}
    upstream = meta.get("problem_id")
    return str(upstream) if upstream else problem.problem_id


def select_heldout(all_problems: list, train_ids: set[str], n: int, seed: int) -> list:
    """Held-out sample, excluding every record that shares a GROUP with training.

    Excluding by record id alone leaked siblings across the split — see
    group_key(). The RNG draw is unchanged, so with a benchmark whose groups
    are all singletons this returns exactly what the record-level version did.
    """
    train_groups = {
        group_key(p) for p in all_problems if p.problem_id in train_ids
    }
    candidates = [
        p for p in all_problems
        if p.problem_id not in train_ids and group_key(p) not in train_groups
    ]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    return [candidates[i] for i in sorted(idx.tolist())]


# Set by main() from --out-tag. Without it a transfer evaluation would write to
# the same directory as the run's own held-out evaluation and destroy it.
OUT_TAG: str | None = None


def _output_dir(
    run_label: str,
    n_heldout: int,
    heldout_seed: int,
) -> Path:
    tag = OUT_TAG or f"heldout_n{n_heldout}_seed{heldout_seed}_all_pareto"
    return GA_BASE / run_label / "rep_0" / tag


def _validate_or_write_manifest(
    out_dir: Path,
    *,
    run_label: str,
    variant: str,
    heldout_ids: list[str],
    train_ids: set[str],
    train_group_ids: set[str],
    heldout_group_ids: set[str],
    pareto_size: int,
    args: argparse.Namespace,
) -> dict:
    manifest_path = out_dir / "manifest.json"
    expected = {
        "schema_version": 3,
        "script": "evaluate_heldout.py",
        "benchmark": args.benchmark,
        "model": MODEL,
        "run_label": run_label,
        "variant": variant,
        "population_source": "complete_training_pareto_front",
        # Where the POLICIES came from. Differs from run_label only for a
        # transfer evaluation, where the front was trained elsewhere (e.g. on
        # another benchmark) and is being evaluated on this run's held-out set.
        "front_source": str(FRONT_FILE.relative_to(REPO)) if FRONT_FILE else "self",
        "is_transfer_evaluation": FRONT_FILE is not None,
        "n_pareto_policies": pareto_size,
        "n_heldout": len(heldout_ids),
        "heldout_seed": args.heldout_seed,
        "heldout_ids": heldout_ids,
        "heldout_ids_sha256": _ids_digest(heldout_ids),
        "train_ids": sorted(train_ids),
        # Group ids make the train/held-out independence auditable after the
        # fact: these two sets must be disjoint, and for a benchmark whose
        # records map many-to-one onto upstream problems that is NOT implied by
        # the record ids being disjoint.
        "split_unit": "metadata.problem_id where published, else record id",
        "train_group_ids": sorted(train_group_ids),
        "heldout_group_ids": sorted(heldout_group_ids),
        "n_train_groups": len(train_group_ids),
        "n_heldout_groups": len(heldout_group_ids),
        "n_problem_workers": args.workers,
        "energy_backend": args.energy_backend,
        "energy_is_real_measurement": args.energy_backend != "runtime_proxy",
        "f3_note": (
            "wall-clock runtime per passing test × ENERGY_PROXY_TDP_W; "
            "explicitly activated in this process; not comparable to RAPL"
            if args.energy_backend == "runtime_proxy" else
            f"real measured energy via {args.energy_backend}, v3.1 two-phase: "
            "Phase A concurrent with no measurement, thread-pool join as "
            "barrier, Phase B serial inside the energy window. One policy "
            "evaluated at a time. Same scale as the GA training run's F3."
        ),
        "pipeline": "full analyze-repair-verify-retry-optimize-verify pipeline",
        "resume_enabled": not args.no_resume,
    }

    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        consistency_keys = (
            "schema_version",
            "model",
            "run_label",
            "population_source",
            "n_pareto_policies",
            "heldout_ids_sha256",
            "n_problem_workers",
            "energy_backend",
        )
        mismatched = [k for k in consistency_keys if existing.get(k) != expected.get(k)]
        if mismatched:
            raise RuntimeError(
                f"{manifest_path} does not match this run ({', '.join(mismatched)}). "
                "Use a matching command or --overwrite."
            )
        return existing

    expected["started_at"] = datetime.now(UTC).isoformat()
    _atomic_write_json(manifest_path, expected)
    return expected


def _load_checkpoint(
    path: Path,
    *,
    expected_idx: int,
    expected_policy: dict,
) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("policy_idx") != expected_idx:
        return None
    if _policy_key(payload.get("policy", {})) != _policy_key(expected_policy):
        return None
    evaluation = payload.get("evaluation")
    return evaluation if isinstance(evaluation, dict) else None


def eval_policy(
    policy: dict,
    problems: list,
    *,
    workers: int,
    scratch_root: Path,
) -> dict:
    """Evaluate one policy with exactly ``workers`` total problem pipelines."""
    import run_nsga2_runtime as ga

    ga._N_LOCAL_WORKERS = workers
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="policy_", dir=scratch_root) as tmp:
        config = ga._build_config(Path(tmp), REPO)
        config["model"] = MODEL
        t0 = time.perf_counter()
        result = ga._mean_fitness_runtime(problems, config, policy)
        result["wall_s"] = time.perf_counter() - t0
        return result


def _progress_payload(
    completed: list[dict],
    *,
    total: int,
    started: float,
    current_policy_idx: int | None,
) -> dict:
    elapsed = time.perf_counter() - started
    measured = [float(r.get("wall_s", 0.0)) for r in completed if r.get("wall_s")]
    mean_policy_s = float(np.mean(measured)) if measured else None
    remaining = total - len(completed)
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "completed": len(completed),
        "total": total,
        "current_policy_idx": current_policy_idx,
        "elapsed_s_this_process": elapsed,
        "mean_policy_wall_s": mean_policy_s,
        "eta_s": mean_policy_s * remaining if mean_policy_s is not None else None,
    }


def _summary(run_label: str, variant: str, results: list[dict], args) -> dict:
    valid_cl = [r["mean_cl"] for r in results if r.get("mean_cl", 0) > 0]
    valid_f3 = [r["mean_energy"] for r in results if r.get("mean_energy", 0) > 0]
    return {
        "run_label": run_label,
        "variant": variant,
        "n_policies_evaluated": len(results),
        "n_heldout": args.n_heldout,
        "heldout_seed": args.heldout_seed,
        "mean_correctness": float(np.mean([r["mean_correctness"] for r in results])),
        "best_correctness": float(max(r["mean_correctness"] for r in results)),
        "mean_cl": float(np.mean(valid_cl)) if valid_cl else None,
        "best_cl": float(min(valid_cl)) if valid_cl else None,
        "mean_runtime_proxy_mj": float(np.mean(valid_f3)) if valid_f3 else None,
        "total_policy_wall_s": float(sum(r.get("wall_s", 0.0) for r in results)),
    }


def _normalised_hypervolumes(result_sets: dict[str, list[dict]],
                             energy_backend: str = "runtime_proxy") -> tuple[dict, dict]:
    """Compute comparable HVs using one union-derived normalisation/reference."""
    from pymoo.indicators.hv import HV

    matrices: dict[str, np.ndarray] = {}
    for variant, results in result_sets.items():
        rows = [
            [r["mean_cl"], -r["mean_correctness"], r["mean_energy"]]
            for r in results
            if (
                r.get("n_feasible", 0) > 0
                and r.get("mean_cl", 0) > 0
                and r.get("mean_energy", 0) > 0
                and np.isfinite(r["mean_cl"])
                and np.isfinite(r["mean_correctness"])
                and np.isfinite(r["mean_energy"])
            )
        ]
        matrices[variant] = np.asarray(rows, dtype=float).reshape((-1, 3))

    nonempty = [m for m in matrices.values() if len(m)]
    if not nonempty:
        return {variant: 0.0 for variant in matrices}, {}

    union = np.vstack(nonempty)
    ideal = union.min(axis=0)
    nadir = union.max(axis=0)
    span = nadir - ideal
    span[span == 0] = 1.0
    reference = np.full(3, 1.1)
    indicator = HV(ref_point=reference)

    values = {}
    for variant, matrix in matrices.items():
        normalised = (matrix - ideal) / span if len(matrix) else matrix
        values[variant] = float(indicator(normalised)) if len(normalised) else 0.0
    methodology = {
        "kind": "normalised_hypervolume",
        # Third objective is named for the backend that produced it: the
        # historical "runtime_proxy" label is wrong once --energy-backend
        # rapl_sysfs is used, and a stale label on a real-energy result is
        # exactly the kind of thing that gets mis-cited later.
        "objectives": ["mean_cl", "-mean_correctness",
                       f"mean_energy_mj[{energy_backend}]"],
        "energy_backend": energy_backend,
        "energy_is_real_measurement": energy_backend != "runtime_proxy",
        "normalisation_ideal_union": ideal.tolist(),
        "normalisation_nadir_union": nadir.tolist(),
        "reference_point_normalised": reference.tolist(),
        "note": "One common normalisation and reference point across all variants.",
    }
    return values, methodology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-heldout", type=int, default=100)
    parser.add_argument("--heldout-seed", type=int, default=0)
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        choices=["codeflaws", "runbugrun_py"],
        help=f"Benchmark the GA runs were trained on (default: {DEFAULT_BENCHMARK}). "
             "Must match the --benchmark used by run_nsga2_runtime.py.",
    )
    parser.add_argument(
        "--run-prefix",
        default=DEFAULT_RUN_PREFIX,
        help="Prefix of the GA run directory names; the directory evaluated for "
             "each variant is <prefix><variant> (default: "
             f"{DEFAULT_RUN_PREFIX!r} -> {DEFAULT_RUN_PREFIX}random, "
             f"{DEFAULT_RUN_PREFIX}lhs, {DEFAULT_RUN_PREFIX}llm_warmstart). "
             "Set this to match the --run-label prefix used by the GA run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("N_LOCAL_WORKERS", "4")),
        help="Total concurrent problem pipelines (default: N_LOCAL_WORKERS or 4).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help="Initialisation variants to evaluate.",
    )
    parser.add_argument(
        "--run-labels",
        nargs="+",
        default=None,
        help="Evaluate these GA run directories verbatim, instead of deriving "
             "names as <run-prefix><variant>. Use for a run whose label is not "
             "one of the three initialisation variants (e.g. a single "
             "pop50_gen15_n50_v4 run). Each label is used as its own key in the "
             "outputs, and --run-prefix/--variants are ignored.",
    )
    parser.add_argument(
        "--front-file", default=None,
        help="Evaluate the Pareto front in this file instead of the named run's "
             "own pareto_solutions.json. The held-out PROBLEMS still come from "
             "the named run (its train_ids drive the exclusions), so this gives "
             "a controlled cross-language transfer test: identical problems and "
             "protocol, policies from elsewhere. Requires --out-tag.",
    )
    parser.add_argument(
        "--out-tag", default=None,
        help="Output subdirectory name under <run>/rep_0/. Defaults to "
             "heldout_n<N>_seed<S>_all_pareto. Required with --front-file, "
             "which would otherwise overwrite the run's own evaluation.",
    )
    parser.add_argument(
        "--expect-heldout-sha256", default=None,
        help="Abort unless the selected held-out set has this digest. Use it to "
             "PROVE a transfer evaluation ran on exactly the problems another "
             "evaluation used; without it a silent split difference would make "
             "the two sets of numbers incomparable while still looking fine.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing per-policy checkpoints and evaluate again.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacing outputs for the same run configuration.",
    )
    parser.add_argument(
        "--energy-backend",
        default="runtime_proxy",
        help="Energy backend for held-out F3 (default: runtime_proxy, which is "
             "wall_time x TDP and NOT on the same scale as a real-mJ training "
             "run). `rapl_sysfs` is valid here despite the 4 concurrent "
             "workers: _mean_fitness_runtime evaluates one policy at a time, "
             "runs Phase A concurrently with NO energy measurement, joins the "
             "pool as a barrier, then runs Phase B serially inside the energy "
             "window — the same v3.1 two-phase protocol as the training run. "
             "The default is left as the proxy so previously published "
             "held-out evaluations reproduce unchanged.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.n_heldout < 1:
        parser.error("--n-heldout must be at least 1")
    return args


def main() -> int:
    args = parse_args()

    # Resolve which GA runs we are evaluating. Both are module-level globals so
    # the helpers above (load_pareto, _output_dir) pick them up without threading
    # the benchmark through every signature.
    global GA_BASE, RUNS, FRONT_FILE, OUT_TAG
    GA_BASE = REPO / "outputs" / "ga" / args.benchmark / MODEL.replace(":", "_")

    if args.front_file:
        if not args.out_tag:
            raise SystemExit(
                "--front-file requires --out-tag: without it this would write over "
                "the run's own held-out evaluation."
            )
        FRONT_FILE = Path(args.front_file)
        if not FRONT_FILE.is_absolute():
            FRONT_FILE = (REPO / FRONT_FILE).resolve()
        if not FRONT_FILE.is_file():
            raise SystemExit(f"--front-file not found: {FRONT_FILE}")
    OUT_TAG = args.out_tag
    if args.run_labels:
        # Explicit labels: each run directory is its own key, so a run whose
        # label is not one of the three initialisation variants can be
        # evaluated without inventing a variant name for it.
        RUNS = {label: label for label in args.run_labels}
        args.variants = list(args.run_labels)
    else:
        RUNS = {v: f"{args.run_prefix}{v}" for v in VARIANTS}

    if not GA_BASE.is_dir():
        raise SystemExit(
            f"No GA output directory for benchmark {args.benchmark!r}: {GA_BASE}\n"
            f"Check --benchmark, or run run_nsga2_runtime.py --benchmark {args.benchmark} first."
        )

    # The backend is always registered EXPLICITLY, never auto-selected, so the
    # scale of F3 is a recorded decision rather than a property of whichever
    # host this happens to run on.
    #
    # `runtime_proxy` (the default) is wall_time x TDP. `rapl_sysfs` is also
    # valid here, despite --workers > 1: _mean_fitness_runtime evaluates ONE
    # policy at a time, runs Phase A concurrently with NO energy measurement,
    # joins the thread pool as a barrier, and only then runs Phase B serially
    # inside the energy window. That is the same v3.1 two-phase protocol the
    # training run uses, so nothing is inferring while a window is open.
    import run_nsga2_runtime as ga

    from benchmarks import get_benchmark
    from energy import resolve_backend

    runtime_backend = resolve_backend(args.energy_backend)
    if runtime_backend.name != args.energy_backend:
        raise SystemExit(
            f"Requested energy backend {args.energy_backend!r} resolved to "
            f"{runtime_backend.name!r}. Refusing to run: F3 would silently be "
            f"on a different scale than requested."
        )
    ga.set_energy_backend(runtime_backend)
    ga._N_LOCAL_WORKERS = args.workers
    print(f"[heldout] energy backend: {runtime_backend.name} "
          f"(real measurement: {args.energy_backend != 'runtime_proxy'})")

    # A stray candidate workload from an earlier run contaminates F3 for every
    # policy measured afterwards: RAPL reports the whole CPU package, so one
    # escaped infinite loop is charged to whatever is being measured. This has
    # happened -- a Codeflaws binary survived a restart and held a core for
    # 2 days 4 hours across two complete held-out evaluations, inflating every
    # reading by a measured 1.42x. Clear them before the first measurement
    # rather than discovering it in the results.
    from benchmarks.base import sweep_stray_workloads
    strays = sweep_stray_workloads()
    if strays:
        print(f"[heldout] WARNING: {len(strays)} stray candidate workload(s) "
              f"from an earlier run are consuming CPU. Killing them before "
              f"measuring, or F3 would be inflated for every policy:",
              file=sys.stderr)
        for pid, cmd in strays:
            print(f"[heldout]   pid {pid}  {cmd[:100]}", file=sys.stderr)
        sweep_stray_workloads(kill=True)
        print("[heldout] strays cleared.", file=sys.stderr)

    missing = [
        str((GA_BASE / RUNS[v] / "manifest.json").relative_to(REPO))
        for v in args.variants
        if not (GA_BASE / RUNS[v] / "manifest.json").is_file()
    ]
    if missing:
        raise SystemExit(
            "No GA run found for:\n  " + "\n  ".join(missing) + "\n"
            f"Check --benchmark ({args.benchmark}) and --run-prefix "
            f"({args.run_prefix!r}), or run run_nsga2_runtime.py first."
        )
    manifests = {
        variant: json.loads(
            (GA_BASE / RUNS[variant] / "manifest.json").read_text(encoding="utf-8")
        )
        for variant in args.variants
    }
    canonical_train_ids = set(next(iter(manifests.values()))["train_ids"])
    for variant, manifest in manifests.items():
        if set(manifest["train_ids"]) != canonical_train_ids:
            raise RuntimeError(f"{variant} uses a different training subset")
        if manifest.get("model") != MODEL:
            raise RuntimeError(f"{variant} uses model {manifest.get('model')}, expected {MODEL}")

    benchmark = get_benchmark(args.benchmark)
    all_problems = list(benchmark.problems())
    heldout = select_heldout(
        all_problems, canonical_train_ids, args.n_heldout, args.heldout_seed
    )
    heldout_ids = [p.problem_id for p in heldout]

    # Group ids for the manifest, and a hard assertion that the split is
    # actually independent. Disjoint record ids do NOT imply disjoint groups
    # when a benchmark maps several records onto one upstream problem.
    train_group_ids = {
        group_key(p) for p in all_problems if p.problem_id in canonical_train_ids
    }
    heldout_group_ids = {group_key(p) for p in heldout}
    _digest = _ids_digest(heldout_ids)
    if args.expect_heldout_sha256 and _digest != args.expect_heldout_sha256:
        raise SystemExit(
            "held-out set does not match the expected digest, so results would "
            "NOT be comparable with the evaluation that produced it.\n"
            f"  expected {args.expect_heldout_sha256}\n"
            f"  got      {_digest}\n"
            "Check --benchmark, --run-labels, --n-heldout and --heldout-seed."
        )

    _bleed = train_group_ids & heldout_group_ids
    if _bleed:
        raise RuntimeError(
            f"train/held-out split is not independent: {len(_bleed)} shared "
            f"group(s), e.g. {sorted(_bleed)[:5]}. select_heldout() should have "
            f"excluded these."
        )

    plans: dict[str, tuple[str, list[dict], Path]] = {}
    total_policies = 0
    for variant in args.variants:
        run_label = RUNS[variant]
        population = load_pareto(run_label)
        out_dir = _output_dir(run_label, args.n_heldout, args.heldout_seed)
        plans[variant] = (run_label, population, out_dir)
        total_policies += len(population)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Resumable held-out evaluation of a training Pareto front   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(
        f"  heldout={len(heldout)} seed={args.heldout_seed} "
        f"workers={args.workers} backend={runtime_backend.name}"
    )
    print(f"  complete Pareto policies={total_policies}")
    for variant, (run_label, population, out_dir) in plans.items():
        print(
            f"  {variant:<14} policies={len(population):>3} "
            f"-> {out_dir.relative_to(REPO)}"
        )
    if args.dry_run:
        return 0

    scratch_root = GA_BASE / "_heldout_scratch" / "resumable"
    shared_policy_cache: dict[str, dict] = {}
    result_sets: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}

    for variant, (run_label, population, out_dir) in plans.items():
        manifest = _validate_or_write_manifest(
            out_dir,
            run_label=run_label,
            variant=variant,
            heldout_ids=heldout_ids,
            train_ids=canonical_train_ids,
            train_group_ids=train_group_ids,
            heldout_group_ids=heldout_group_ids,
            pareto_size=len(population),
            args=args,
        )
        policy_dir = out_dir / "policies"
        process_started = time.perf_counter()
        completed: list[dict] = []

        print(
            f"\n{'═' * 68}\n"
            f"  Run: {run_label} (complete Pareto front, "
            f"{len(population)} policies)\n"
            f"{'═' * 68}"
        )
        for policy_idx, solution in enumerate(population):
            position = policy_idx + 1
            policy = solution["policy"]
            key = _policy_key(policy)
            checkpoint_path = policy_dir / f"policy_{policy_idx:03d}.json"

            evaluation = None
            if not args.no_resume and not args.overwrite and checkpoint_path.exists():
                evaluation = _load_checkpoint(
                    checkpoint_path,
                    expected_idx=policy_idx,
                    expected_policy=policy,
                )
                if evaluation is not None:
                    print(
                        f"  [{variant}] {position}/{len(population)} "
                        f"policy {policy_idx:03d} "
                        "— resumed"
                    )

            if evaluation is None and key in shared_policy_cache:
                evaluation = copy.deepcopy(shared_policy_cache[key])
                evaluation["wall_s"] = 0.0
                evaluation["reused_exact_policy"] = True
                print(
                    f"  [{variant}] {position}/{len(population)} "
                    f"policy {policy_idx:03d} "
                    "— exact duplicate reused"
                )

            if evaluation is None:
                progress = _progress_payload(
                    completed,
                    total=len(population),
                    started=process_started,
                    current_policy_idx=policy_idx,
                )
                _atomic_write_json(out_dir / "progress.json", progress)
                print(
                    f"  [{variant}] {position}/{len(population)} "
                    f"policy {policy_idx:03d} "
                    "— evaluating",
                    flush=True,
                )
                evaluation = eval_policy(
                    policy,
                    heldout,
                    workers=args.workers,
                    scratch_root=scratch_root,
                )
                print(
                    f"    feasible={evaluation['n_feasible']}/{len(heldout)} "
                    f"corr={evaluation['mean_correctness']:.3f} "
                    f"cl={evaluation['mean_cl']:.2f} "
                    f"runtimeF3={evaluation['mean_energy']:.3f} "
                    f"wall={evaluation['wall_s']:.0f}s",
                    flush=True,
                )

            evaluation["policy_idx"] = policy_idx
            evaluation["policy"] = policy
            evaluation["train_fitness"] = solution["fitness"]
            shared_policy_cache[key] = copy.deepcopy(evaluation)
            completed.append(evaluation)

            _atomic_write_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "policy_idx": policy_idx,
                    "policy": policy,
                    "evaluation": evaluation,
                },
            )
            _atomic_write_json(
                out_dir / "progress.json",
                _progress_payload(
                    completed,
                    total=len(population),
                    started=process_started,
                    current_policy_idx=None,
                ),
            )

        _atomic_write_json(out_dir / "results.json", completed)
        summary = _summary(run_label, variant, completed, args)
        summaries[variant] = summary
        result_sets[variant] = completed

        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["n_policies_completed"] = len(completed)
        manifest["elapsed_seconds_this_process"] = time.perf_counter() - process_started
        _atomic_write_json(out_dir / "manifest.json", manifest)

    hypervolumes, hv_method = _normalised_hypervolumes(result_sets, args.energy_backend)
    for variant, summary in summaries.items():
        summary["normalised_hypervolume"] = hypervolumes[variant]
        summary["hypervolume_methodology"] = hv_method
        out_dir = plans[variant][2]
        _atomic_write_json(out_dir / "summary.json", summary)

    hv_ranking = sorted(hypervolumes, key=hypervolumes.get, reverse=True)
    pairwise_hv_deltas = {
        f"{left}_minus_{right}": hypervolumes[left] - hypervolumes[right]
        for left in hypervolumes
        for right in hypervolumes
        if left != right
    }
    comparison = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "n_heldout": args.n_heldout,
        "heldout_seed": args.heldout_seed,
        "population_source": "complete_training_pareto_front",
        "hypervolume_methodology": hv_method,
        "normalised_hypervolume_ranking_best_first": hv_ranking,
        "pairwise_normalised_hypervolume_deltas": pairwise_hv_deltas,
        "delta_interpretation": "For A_minus_B, a positive value means A has higher HV.",
        "statistical_scope": (
            "Descriptive held-out comparison for one completed GA replicate per "
            "initialisation method; GA-seed-level inference requires repeated runs."
        ),
        "variants": summaries,
    }
    # The filename MUST identify which runs were compared. It used to be fixed
    # per (n_heldout, seed), so evaluating a second arm silently overwrote the
    # first arm's comparison — and, worse, the surviving file's union ideal/
    # nadir came from only the arm evaluated last, making its normalised
    # hypervolumes non-comparable with the numbers the earlier arm printed.
    # Both failures are silent. Encoding the compared set in the name means a
    # second arm writes a new file, and a genuine multi-arm comparison is
    # produced by passing every arm in ONE invocation (which is cheap: policies
    # already evaluated resume from their per-policy checkpoints).
    arms = "_vs_".join(sorted(summaries))
    if len(arms) > 120:  # keep the path sane for many-arm comparisons
        arms = f"{len(summaries)}arms_{_ids_digest(sorted(summaries))[:12]}"
    comparison_path = GA_BASE / (
        f"heldout_comparison_n{args.n_heldout}_"
        f"seed{args.heldout_seed}_{arms}.json"
    )
    comparison["compared_run_labels"] = sorted(summaries)
    comparison["comparison_is_valid_only_within_this_file"] = (
        "Normalised hypervolumes are computed against the ideal/nadir of the "
        "UNION of the arms listed in compared_run_labels. They are NOT "
        "comparable with normalised hypervolumes from any other file, which "
        "used a different union. To compare arms, evaluate them in one "
        "invocation."
    )
    _atomic_write_json(comparison_path, comparison)

    print(f"\n{'═' * 76}")
    print("  HELD-OUT COMPARISON")
    print(f"{'═' * 76}")
    print(f"  {'Variant':<16} {'N':>4} {'HV(norm)':>10} {'BestCorr':>10} {'MeanCorr':>10}")
    for variant, summary in summaries.items():
        print(
            f"  {variant:<16} {summary['n_policies_evaluated']:>4} "
            f"{summary['normalised_hypervolume']:>10.4f} "
            f"{summary['best_correctness']:>10.4f} "
            f"{summary['mean_correctness']:>10.4f}"
        )
    if "llm_warmstart" in hypervolumes:
        for baseline in ("random", "lhs"):
            if baseline in hypervolumes:
                delta = pairwise_hv_deltas[f"llm_warmstart_minus_{baseline}"]
                print(f"  ΔHV LLM - {baseline:<6} = {delta:+.4f}")
    print(f"\n  Saved -> {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
