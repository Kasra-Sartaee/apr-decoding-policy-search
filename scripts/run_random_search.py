#!/usr/bin/env python3
"""Random-search control for the NSGA-II run, at matched budget and protocol.

The GA's result only means something if evolutionary search beats drawing the
same number of policies at random. That control is only valid if the two differ
in one thing, how policies are proposed, and are identical in everything else:
training problems, evaluation budget, fitness function, worker count, energy
backend, feasibility rule, and front extraction.

This script therefore does not reimplement evaluation. It imports run_nsga2.py's own
`_mean_fitness_over()`, the same function NSGA-II calls for every candidate.
That function implements the v3.1 two-phase protocol:

    Phase A  analyze/repair/retry/optimize across N_WORKERS threads, no energy
    barrier  ThreadPoolExecutor shutdown joins every worker
    Phase B  serial re-execution of passing tests inside one RAPL window

which is why `rapl_sysfs` is valid at N_WORKERS=4: nothing infers while a
window is open.

OUTPUT
------
Writes `pareto_solutions.json` in exactly the GA's schema, so the held-out
evaluator runs against it unmodified:

    scripts/evaluate_heldout.py --run-labels <run-label> \
        --n-heldout 100 --heldout-seed 0 --energy-backend rapl_sysfs

It also persists `evaluations/cand_NNNNN.json` for EVERY candidate, including
per-problem F1/F2/F3. run_nsga2.py does not do this on the C path, which is why
the GA's training F3 cannot be re-aggregated as a median after the fact. This
control does not inherit that gap.

Resumable: every evaluated candidate is checkpointed, so an interruption loses
at most the candidate in flight.

Usage:
  N_WORKERS=4 python scripts/run_random_search.py \
      --match-run pop50_gen15_n50_v4 --n-samples 750 --sample-seed 99
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _nondominated(F: np.ndarray) -> list[int]:
    """Indices of the non-dominated set, minimising every column.

    Plain O(n^2) pairwise test rather than pymoo's fast sort: n is 750, this
    runs once, and the explicitness matters more than the microseconds.
    """
    n = len(F)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-run", default="pop50_gen15_n50_v4",
                    help="GA run to match: its manifest supplies train_ids and model.")
    ap.add_argument("--benchmark", default="codeflaws")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--n-samples", type=int, default=750,
                    help="Evaluation budget. Must equal the GA's pop*gen to be a "
                         "fair control (50*15=750).")
    ap.add_argument("--sample-seed", type=int, default=99,
                    help="Base seed for policy sampling. Deliberately NOT the GA's 42: "
                         "these are different draws, not a replay.")
    ap.add_argument("--reps", type=int, default=1,
                    help="Number of repetitions, written to rep_0..rep_N-1. A repetition "
                         "whose pareto_solutions.json AND intermediate_solutions.json both "
                         "exist is skipped, so an existing single-rep run can be extended "
                         "without re-executing it. Mirrors run_nsga2.py --reps.")
    ap.add_argument("--seed-increment", type=int, default=1,
                    help="Per-repetition seed step: rep i draws from sample_seed + "
                         "i*seed_increment. Each repetition therefore samples a different "
                         "750 policies from the same space. Set to 0 to hold the draw fixed "
                         "across repetitions, isolating LLM sampling noise as the only "
                         "source of run-to-run variance (the training problems are fixed "
                         "regardless: they come from the matched GA run's manifest).")
    ap.add_argument("--min-feasible", type=int, default=1,
                    help="Feasibility threshold, must match the GA's.")
    ap.add_argument("--energy-backend", default="rapl_sysfs")
    ap.add_argument("--run-label", default=None,
                    help="Output label (default: rs_<match-run>).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Bind the model before importing run_nsga2: DEFAULT_MODEL is read at import.
    os.environ["AGENT_MODEL"] = args.model

    import run_nsga2 as ga

    from agentic_workflow import set_energy_backend
    from benchmarks import get_benchmark
    from energy import resolve_backend
    from search_space import SEARCH_SPACE, build_policy_from_vector

    model_dir = args.model.replace(":", "_")
    ga_base = REPO / "outputs" / "ga" / args.benchmark / model_dir
    src_manifest = ga_base / args.match_run / "manifest.json"
    if not src_manifest.is_file():
        raise SystemExit(f"No GA manifest to match at {src_manifest}")
    src = json.loads(src_manifest.read_text(encoding="utf-8"))
    if src.get("model") != args.model:
        raise SystemExit(f"{args.match_run} used {src.get('model')}, not {args.model}")

    budget = src.get("pop", 0) * src.get("gen", 0)
    if budget and args.n_samples != budget:
        raise SystemExit(
            f"Budget mismatch: {args.match_run} evaluated pop*gen = {budget} "
            f"candidates, but --n-samples is {args.n_samples}. A random-search "
            f"control at a different budget is not a fair comparison. Pass "
            f"--n-samples {budget} or override deliberately."
        )
    if args.min_feasible != src.get("min_feasible_threshold", args.min_feasible):
        raise SystemExit(
            f"--min-feasible {args.min_feasible} != GA's "
            f"{src.get('min_feasible_threshold')}; feasibility rule must match."
        )

    train_ids = set(src["train_ids"])
    benchmark = get_benchmark(args.benchmark)
    all_problems = list(benchmark.problems())
    train_problems = [p for p in all_problems if p.problem_id in train_ids]
    if len(train_problems) != len(train_ids):
        raise SystemExit(
            f"resolved {len(train_problems)} of {len(train_ids)} training problems"
        )

    backend = resolve_backend(args.energy_backend)
    if backend.name != args.energy_backend:
        raise SystemExit(
            f"Requested {args.energy_backend!r} but resolved {backend.name!r}; "
            "refusing to run with F3 on an unintended scale."
        )
    set_energy_backend(backend)

    run_label = args.run_label or f"rs_{args.match_run}"
    out_root = ga_base / run_label

    n_workers = int(os.environ.get("N_WORKERS", "1"))
    print(f"[rs] matching {args.match_run}: {len(train_problems)} train problems, "
          f"budget {args.n_samples}")
    print(f"[rs] model={args.model} workers={n_workers} energy={backend.name}")
    print(f"[rs] {args.reps} repetition(s), seeds "
          f"{[args.sample_seed + i * args.seed_increment for i in range(args.reps)]}")
    print(f"[rs] out -> {out_root}")
    if n_workers != src.get("n_workers"):
        print(f"[rs] WARNING: N_WORKERS={n_workers} but the GA used "
              f"{src.get('n_workers')}; execution mode differs.", file=sys.stderr)

    lower = np.array([d["low"] for d in SEARCH_SPACE], dtype=float)
    upper = np.array([d["high"] for d in SEARCH_SPACE], dtype=float)

    if args.dry_run:
        for i in range(args.reps):
            seed_i = args.sample_seed + i * args.seed_increment
            rep_dir = out_root / f"rep_{i}"
            done = ((rep_dir / "pareto_solutions.json").is_file()
                    and (rep_dir / "intermediate_solutions.json").is_file())
            state = ("ALREADY DONE — would skip" if done else
                     f"would evaluate {args.n_samples} policies over "
                     f"{len(train_problems)} problems")
            print(f"[rs] dry run: rep_{i} seed={seed_i} — {state}")
        X0 = np.random.default_rng(args.sample_seed).uniform(
            lower, upper, size=(args.n_samples, len(SEARCH_SPACE)))
        print(f"[rs] example policy: "
              f"{json.dumps(build_policy_from_vector(X0[0]), default=str)[:160]}…")
        print("[rs] no Ollama calls made.")
        return 0

    _atomic_write_json(out_root / "manifest.json", {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "random_search",
        "script": "run_random_search.py",
        "model": args.model,
        "run_label": run_label,
        "matched_ga_run": args.match_run,
        "energy_backend_requested": args.energy_backend,
        "energy_backend_resolved": backend.name,
        "energy_is_real_measurement": args.energy_backend != "runtime_proxy",
        "n_samples": args.n_samples,
        "sample_seed": args.sample_seed,
        "seed_increment": args.seed_increment,
        "reps": args.reps,
        "rep_seeds": [args.sample_seed + i * args.seed_increment for i in range(args.reps)],
        "train_size": len(train_problems),
        "train_ids": sorted(train_ids),
        "search_space": [d["path"] for d in SEARCH_SPACE],
        "min_feasible_threshold": args.min_feasible,
        "n_workers": n_workers,
        "feasibility_criterion": src.get("feasibility_criterion"),
        "objective_aggregation": src.get("objective_aggregation"),
        "test_suite": src.get("test_suite"),
        "reported_front": "filtered to feasible (G<=0) solutions only",
        "measurement": (
            "v3.1 two-phase via run_nsga2._mean_fitness_over — the SAME function "
            "NSGA-II calls per candidate. Phase A concurrent with no "
            "measurement, thread-pool join as barrier, Phase B serial inside "
            "the RAPL window."
        ),
    })
    # Call the shared writer directly rather than run_nsga2's private wrapper,
    # which expects run_nsga2's own argparse namespace (pop/gen/seed_increment).
    from benchmarks.codeflaws.adapter import (
        CODEFLAWS_COMPILE_TIMEOUT_S,
        CODEFLAWS_RUN_TIMEOUT_S,
    )
    # Same source run_nsga2.py reads: the effective timeout is the env var, which
    # may differ from the nominal default.
    llm_timeout_s = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "300"))

    # run_nsga2's own builder: the pipeline reads reports_dir/repaired_dir/
    # optimized_dir/verification_json and several timeout keys, so a
    # hand-rolled dict silently fails every problem with a KeyError.
    # run_nsga2.py passes out_root (not rep_dir) here, so match that — the working
    # artifact directories are shared across repetitions exactly as they are in
    # run_nsga2.py, since they hold per-problem intermediates rather than results.
    config = ga._build_config(out_root, REPO)
    config["model"] = args.model

    for rep_idx in range(args.reps):
        seed_i = args.sample_seed + rep_idx * args.seed_increment
        rep_dir = out_root / f"rep_{rep_idx}"
        evals_dir = rep_dir / "evaluations"
        pareto_path = rep_dir / "pareto_solutions.json"
        history_path = rep_dir / "intermediate_solutions.json"

        # Same resume rule as run_nsga2.py:599 — a repetition is complete only when
        # BOTH its front and its history exist. This lets an earlier single-rep
        # run be extended without re-executing the 36 h search it already did.
        if pareto_path.is_file() and history_path.is_file():
            print(f"[rs] rep {rep_idx + 1}/{args.reps} already done; skipping")
            continue

        print(f"\n[rs] === repetition {rep_idx + 1}/{args.reps} (seed={seed_i}) ===")
        rep_dir.mkdir(parents=True, exist_ok=True)
        _run_one_rep(
            rep_dir, evals_dir, pareto_path, history_path,
            seed_i, lower, upper, args, train_problems, config, ga,
            n_workers, llm_timeout_s,
            CODEFLAWS_COMPILE_TIMEOUT_S, CODEFLAWS_RUN_TIMEOUT_S,
        )

    print(f"\n[rs] all {args.reps} repetition(s) complete -> {out_root}")
    print(f"\nNext: evaluate the fronts on the held-out set with the same protocol:\n"
          f"  .venv/bin/python scripts/evaluate_heldout.py "
          f"--run-labels {run_label} --n-heldout 100 --heldout-seed 0 "
          f"--workers {n_workers} --energy-backend {args.energy_backend}")
    return 0


def _run_one_rep(rep_dir, evals_dir, pareto_path, history_path, seed_i,
                 lower, upper, args, train_problems, config, ga,
                 n_workers, llm_timeout_s, compile_timeout, run_timeout):
    """Evaluate one repetition's 750 draws and write its front and history.

    Extracted verbatim from the original single-repetition body so the
    measurement path is unchanged: the same _mean_fitness_over call, the same
    per-candidate checkpointing, the same feasibility filter and the same
    non-dominance rule.
    """
    # main() imports these locally (AGENT_MODEL must be bound before run_nsga2 is
    # imported), so they are not in this function's scope. Re-import here rather
    # than threading five more parameters through the signature.
    from provenance import write_protocol_manifest
    from search_space import SEARCH_SPACE, build_policy_from_vector

    write_protocol_manifest(
        rep_dir,
        measurement_protocol="codeflaws_two_phase_v3.1",
        protocol_doc="docs/reports/measurement_protocol.md",
        language="C99",
        n_workers=n_workers,
        timeouts_effective={
            "llm_request_s": llm_timeout_s,
            "compile_s": compile_timeout,
            "run_s": run_timeout,
        },
        seed=seed_i,
        seed_increment=args.seed_increment,
        model=args.model,
        pop=args.n_samples,   # random search: one "generation" of n_samples
        gen=1,
        train_size=len(train_problems),
        extra={"mode": "random_search", "matched_ga_run": args.match_run},
    )

    X = np.random.default_rng(seed_i).uniform(
        lower, upper, size=(args.n_samples, len(SEARCH_SPACE)))
    records: list[dict] = []
    t0 = time.perf_counter()

    for i, x in enumerate(X):
        cand_path = evals_dir / f"cand_{i:05d}.json"
        if cand_path.is_file():
            records.append(json.loads(cand_path.read_text(encoding="utf-8")))
            continue

        policy = build_policy_from_vector(x)
        agg = ga._mean_fitness_over(train_problems, config, policy)
        feasible = agg["n_feasible"] >= args.min_feasible
        rec = {
            "index": i,
            "policy": policy,
            "fitness": {
                "cl": agg["mean_cl"],
                "correctness": agg["mean_correctness"],
                "energy": agg["mean_energy"],
            },
            "n_feasible": agg["n_feasible"],
            "n_total": agg["n_total"],
            "feasible": bool(feasible),
            "per_problem": agg.get("per_problem", []),
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(cand_path, rec)
        records.append(rec)

        el = time.perf_counter() - t0
        done = i + 1
        eta = el / done * (len(X) - done) / 3600
        print(f"  [{done}/{len(X)}] cl={agg['mean_cl']:.2f} "
              f"corr={agg['mean_correctness']:.3f} energy={agg['mean_energy']:.2f} "
              f"feas={agg['n_feasible']}/{agg['n_total']} — ETA {eta:.1f} h",
              flush=True)

    feas = [r for r in records if r["feasible"]
            and np.isfinite(r["fitness"]["energy"]) and np.isfinite(r["fitness"]["cl"])]
    print(f"\n[rs] {len(feas)}/{len(records)} feasible with finite objectives")
    if not feas:
        raise SystemExit("no feasible candidate; cannot form a front")

    F = np.array([[r["fitness"]["cl"], -r["fitness"]["correctness"], r["fitness"]["energy"]]
                  for r in feas], dtype=float)
    front_idx = _nondominated(F)
    pareto = [{"policy": feas[i]["policy"], "fitness": feas[i]["fitness"]} for i in front_idx]
    _atomic_write_json(pareto_path, pareto)
    _atomic_write_json(history_path,
                       [{"policy": r["policy"], "fitness": r["fitness"],
                         "feasible": r["feasible"]} for r in records])

    print(f"[rs] Pareto front: {len(pareto)} policies -> {pareto_path}")
    print(f"[rs] rep wall {(time.perf_counter() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    raise SystemExit(main())
