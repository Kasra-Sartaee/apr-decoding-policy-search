#!/usr/bin/env python3
"""Multi-objective Bayesian optimisation arm, at matched budget and protocol.

NSGA-II spends its budget on population-based search; random search spends it
on uniform draws. Neither uses what previous evaluations revealed about the
space. With ~3 minutes per evaluation, a 750-evaluation budget and 11 continuous
dimensions -- roughly 68 samples per dimension -- this is the regime a
surrogate-based method exists for, and the GA's observed plateau after
generation 10 suggests the budget is the binding constraint rather than the
operators.

This is the same experiment with a different proposal mechanism, so it is
directly comparable to the GA and RS arms. It imports run_nsga2.py's own
`_mean_fitness_over()` -- the function NSGA-II calls per candidate -- so the
v3.1 two-phase protocol comes along unchanged:

    Phase A  analyze/repair/retry/optimize across N_WORKERS threads, no energy
    barrier  ThreadPoolExecutor shutdown joins every worker
    Phase B  serial re-execution of passing tests inside one RAPL window

Training problems, budget, fitness function, worker count, energy backend,
feasibility rule and front extraction are identical or asserted equal at
startup. Only how the next policy is chosen differs.

METHOD
------
qLogNEHVI -- log-transformed *noisy* expected hypervolume improvement (Daulton
et al., 2021) -- over an independent SingleTaskGP per objective, via BoTorch.

The noisy variant is a substantive choice, not a stylistic one. F3 has a
measured coefficient of variation of 28.7% across ten identical repetitions of
the same policy on the same problems (see the v4 analysis notebook). qEHVI
assumes noiseless observations and would treat one lucky draw as ground truth;
qNEHVI integrates over the posterior at the observed points instead. NSGA-II has
no noise model at all, which is worth stating as a limitation of the GA arm
rather than as a virtue of this one.

Objectives are converted to maximisation, as BoTorch expects:

    y = [ -mean_cl , +mean_correctness , -mean_energy ]

Candidates whose objectives are not all finite carry no usable signal and are
excluded from the GP training data, but are still recorded on disk.

INTERPRETER
-----------
Needs botorch/torch, which are NOT in the project venv -- installing them there
consumed a third of a 10 GB quota and truncated itself. They live in a separate
scratch venv instead:

    N_WORKERS=4 $MOBO_PYTHON scripts/run_mobo.py

That venv also carries the project's requirements.txt, so run_nsga2 and the
benchmark adapters import normally.

BATCHING
--------
`--batch-size` trades sample efficiency against acquisition overhead. q=1 is
most sample-efficient but needs ~700 acquisition optimisations; at large n each
costs tens of seconds, which is hours of pure overhead on a ~36 h run. The
default q=5 keeps that near an hour. The batch is a *proposal* batch, not
parallel execution: candidates are still evaluated one at a time, each
internally using N_WORKERS threads across problems.

OUTPUT
------
`pareto_solutions.json` in exactly the GA's schema, so the held-out evaluator
runs against it unmodified:

    scripts/evaluate_heldout.py --run-labels <label> \
        --n-heldout 100 --heldout-seed 0 --energy-backend rapl_sysfs

Also persists `evaluations/cand_NNNNN.json` per candidate with per-problem
F1/F2/F3, which run_nsga2.py does not do on the C path. Resumable: an interruption
loses at most the candidate in flight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
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

    Same plain O(n^2) rule the RS arm uses, so the three arms' fronts are
    extracted identically.
    """
    return [i for i in range(len(F))
            if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                       for j in range(len(F)) if j != i)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-run", default="pop50_gen15_n50_v4",
                    help="GA run to match: supplies train_ids, model and budget.")
    ap.add_argument("--benchmark", default="codeflaws")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--n-samples", type=int, default=750,
                    help="Total evaluation budget; must equal the GA's pop*gen.")
    ap.add_argument("--n-init", type=int, default=50,
                    help="Random initial design before the surrogate takes over. "
                         "Default 50 mirrors the GA's first generation.")
    ap.add_argument("--batch-size", type=int, default=5,
                    help="Candidates proposed per acquisition step.")
    ap.add_argument("--sample-seed", type=int, default=123,
                    help="Seed for the initial design and acquisition restarts. "
                         "Deliberately not the GA's 42 nor RS's 99.")
    ap.add_argument("--reps", type=int, default=1,
                    help="Number of repetitions, written to rep_0..rep_N-1. A repetition whose "
                         "pareto_solutions.json AND intermediate_solutions.json both exist is "
                         "skipped, so an existing single-rep run can be extended without "
                         "re-executing it. Mirrors run_nsga2.py and run_random_search.py.")
    ap.add_argument("--seed-increment", type=int, default=1,
                    help="Per-repetition seed step: rep i uses sample_seed + i*seed_increment "
                         "for BOTH the initial design and the acquisition optimiser. The training "
                         "problems come from the matched GA run's manifest and are identical "
                         "across repetitions regardless.")
    ap.add_argument("--min-feasible", type=int, default=1)
    ap.add_argument("--energy-backend", default="rapl_sysfs")
    ap.add_argument("--run-label", default=None)
    ap.add_argument("--mc-samples", type=int, default=128,
                    help="Monte-Carlo samples for the qLogNEHVI estimator.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.environ["AGENT_MODEL"] = args.model

    import run_nsga2 as ga

    from agentic_workflow import set_energy_backend
    from benchmarks import get_benchmark
    from energy import resolve_backend
    from provenance import write_protocol_manifest
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
            f"Budget mismatch: {args.match_run} evaluated pop*gen = {budget}, but "
            f"--n-samples is {args.n_samples}. A control at a different budget is "
            f"not a fair comparison. Pass --n-samples {budget} or override "
            f"deliberately.")
    if args.min_feasible != src.get("min_feasible_threshold", args.min_feasible):
        raise SystemExit("--min-feasible must match the GA's feasibility rule.")
    if args.n_init >= args.n_samples:
        raise SystemExit("--n-init must be smaller than --n-samples.")

    train_ids = set(src["train_ids"])
    benchmark = get_benchmark(args.benchmark)
    all_problems = list(benchmark.problems())
    train_problems = [p for p in all_problems if p.problem_id in train_ids]
    if len(train_problems) != len(train_ids):
        raise SystemExit(
            f"resolved {len(train_problems)} of {len(train_ids)} training problems")

    backend = resolve_backend(args.energy_backend)
    if backend.name != args.energy_backend:
        raise SystemExit(
            f"Requested {args.energy_backend!r} but resolved {backend.name!r}; "
            "refusing to run with F3 on an unintended scale.")
    set_energy_backend(backend)

    run_label = args.run_label or f"mobo_{args.match_run}"
    out_root = ga_base / run_label
    n_workers = int(os.environ.get("N_WORKERS", "1"))

    lower = np.array([d["low"] for d in SEARCH_SPACE], dtype=float)
    upper = np.array([d["high"] for d in SEARCH_SPACE], dtype=float)
    n_steps = -(-(args.n_samples - args.n_init) // args.batch_size)

    print(f"[mobo] matching {args.match_run}: {len(train_problems)} train problems, "
          f"budget {args.n_samples}")
    print(f"[mobo] {args.n_init} random init + {n_steps} acquisition steps of "
          f"q={args.batch_size}")
    print(f"[mobo] model={args.model} workers={n_workers} energy={backend.name} "
          f"seeds={[args.sample_seed + i * args.seed_increment for i in range(args.reps)]}")
    print(f"[mobo] out -> {out_root}")
    if n_workers != src.get("n_workers"):
        print(f"[mobo] WARNING: N_WORKERS={n_workers} but the GA used "
              f"{src.get('n_workers')}; execution mode differs.", file=sys.stderr)

    try:
        import botorch as _bt
        import torch
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from gpytorch.mlls import SumMarginalLogLikelihood
    except ImportError as exc:
        raise SystemExit(
            f"MISSING DEPENDENCY: {exc}\n"
            "botorch/torch are not in the project venv (installing them there "
            "exhausted a 10 GB quota). Run this with the scratch interpreter:\n"
            "  N_WORKERS=4 $MOBO_PYTHON "
            "scripts/run_mobo.py")

    print(f"[mobo] botorch {_bt.__version__}, torch {torch.__version__}")
    if args.dry_run:
        print("[mobo] dry run: plan validated, no Ollama calls made.")
        return 0

    tkw = {"dtype": torch.double, "device": "cpu"}
    bounds = torch.tensor(np.vstack([lower, upper]), **tkw)
    unit = torch.stack([torch.zeros(len(SEARCH_SPACE), **tkw),
                        torch.ones(len(SEARCH_SPACE), **tkw)])

    _atomic_write_json(out_root / "manifest.json", {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "mobo_qlognehvi",
        "script": "run_mobo.py",
        "model": args.model,
        "run_label": run_label,
        "matched_ga_run": args.match_run,
        "energy_backend_requested": args.energy_backend,
        "energy_backend_resolved": backend.name,
        "energy_is_real_measurement": args.energy_backend != "runtime_proxy",
        "n_samples": args.n_samples,
        "n_init": args.n_init,
        "batch_size": args.batch_size,
        "n_acquisition_steps": n_steps,
        "sample_seed": args.sample_seed,
        "reps": args.reps,
        "seed_increment": args.seed_increment,
        "rep_seeds": [args.sample_seed + i * args.seed_increment for i in range(args.reps)],
        "acquisition": "qLogNoisyExpectedHypervolumeImprovement",
        "surrogate": "ModelListGP of SingleTaskGP, Standardize outcome transform",
        "mc_samples": args.mc_samples,
        "botorch_version": _bt.__version__,
        "torch_version": torch.__version__,
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
            "NSGA-II calls per candidate."),
    })

    config = ga._build_config(out_root, REPO)
    config["model"] = args.model

    for rep_idx in range(args.reps):
        seed_i = args.sample_seed + rep_idx * args.seed_increment
        rep_dir = out_root / f"rep_{rep_idx}"
        evals_dir = rep_dir / "evaluations"
        pareto_path = rep_dir / "pareto_solutions.json"
        history_path = rep_dir / "intermediate_solutions.json"

        # Same resume rule as run_nsga2.py:599 and run_random_search.py: a repetition
        # is complete only when BOTH its front and its history exist, so an earlier
        # single-rep run is reused rather than re-executed.
        if pareto_path.is_file() and history_path.is_file():
            print(f"[mobo] rep {rep_idx + 1}/{args.reps} already done; skipping")
            continue

        print(f"\n[mobo] === repetition {rep_idx + 1}/{args.reps} (seed={seed_i}) ===")
        torch.manual_seed(seed_i)
        rep_dir.mkdir(parents=True, exist_ok=True)
        write_protocol_manifest(
            rep_dir,
            measurement_protocol="codeflaws_two_phase_v3.1",
            protocol_doc="docs/reports/measurement_protocol.md",
            language="C99", n_workers=n_workers,
            timeouts_effective={
                "llm_request_s": float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "300")),
                "compile_s": 30, "run_s": 3,
            },
            seed=seed_i, seed_increment=args.seed_increment, model=args.model,
            pop=args.n_samples, gen=1, train_size=len(train_problems),
            extra={"mode": "mobo_qlognehvi", "matched_ga_run": args.match_run},
        )

        records: list[dict] = []
        resumed = sum(1 for i in range(args.n_samples)
                      if (evals_dir / f"cand_{i:05d}.json").is_file())
        if resumed:
            print(f"[mobo] resuming: {resumed} candidates already on disk")
        t0 = time.perf_counter()
        evaluated_here = 0

        def evaluate(x: np.ndarray, idx: int) -> dict:
            nonlocal evaluated_here
            cand_path = evals_dir / f"cand_{idx:05d}.json"
            if cand_path.is_file():
                return json.loads(cand_path.read_text(encoding="utf-8"))
            policy = build_policy_from_vector(x)
            agg = ga._mean_fitness_over(train_problems, config, policy)
            rec = {
                "index": idx, "x": [float(v) for v in x], "policy": policy,
                "fitness": {"cl": agg["mean_cl"], "correctness": agg["mean_correctness"],
                            "energy": agg["mean_energy"]},
                "n_feasible": agg["n_feasible"], "n_total": agg["n_total"],
                "feasible": bool(agg["n_feasible"] >= args.min_feasible),
                "per_problem": agg.get("per_problem", []),
                "evaluated_at": datetime.now(UTC).isoformat(),
            }
            _atomic_write_json(cand_path, rec)
            evaluated_here += 1
            return rec

        def log(idx: int, rec: dict, phase: str) -> None:
            done = idx + 1
            if evaluated_here:
                eta = (time.perf_counter() - t0) / evaluated_here * (args.n_samples - done) / 3600
                eta_s = f"{eta:.1f} h"
            else:
                eta_s = "resuming"
            f = rec["fitness"]
            print(f"  [{done}/{args.n_samples}] {phase} cl={f['cl']:.2f} "
                  f"corr={f['correctness']:.3f} energy={f['energy']:.2f} "
                  f"feas={rec['n_feasible']}/{rec['n_total']} — ETA {eta_s}", flush=True)

        # ── initial design ────────────────────────────────────────────────────
        rng = np.random.default_rng(seed_i)
        X_init = rng.uniform(lower, upper, size=(args.n_init, len(SEARCH_SPACE)))
        for i in range(args.n_init):
            rec = evaluate(X_init[i], i)
            records.append(rec)
            log(i, rec, "init")

        # ── surrogate loop ────────────────────────────────────────────────────
        def gp_data():
            """Training tensors from candidates with finite values on every objective."""
            xs, ys = [], []
            for r in records:
                f = r["fitness"]
                if not all(np.isfinite(f[k]) for k in ("cl", "correctness", "energy")):
                    continue
                xs.append(r["x"])
                ys.append([-f["cl"], f["correctness"], -f["energy"]])   # maximise
            return (torch.tensor(np.asarray(xs), **tkw),
                    torch.tensor(np.asarray(ys), **tkw))

        idx = args.n_init
        step = 0
        while idx < args.n_samples:
            q = min(args.batch_size, args.n_samples - idx)
            Xr, Y = gp_data()
            if len(Xr) < 2:
                raise SystemExit("too few finite observations to fit a surrogate")

            Xn = normalize(Xr, bounds)
            model = ModelListGP(*[
                SingleTaskGP(Xn, Y[:, k:k + 1], outcome_transform=Standardize(m=1))
                for k in range(Y.shape[-1])])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))

            # Reference point derived from the data, not fixed a priori: the three
            # objectives have unrelated scales (CL ~9, correctness ~0.5, energy ~11),
            # so a hardcoded point would silently weight them.
            worst = Y.min(dim=0).values
            span = (Y.max(dim=0).values - worst).clamp(min=1e-9)
            ref_point = (worst - 0.1 * span).tolist()

            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=ref_point, X_baseline=Xn,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([args.mc_samples])),
                prune_baseline=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cand_n, _ = optimize_acqf(
                    acq_function=acqf, bounds=unit, q=q,
                    num_restarts=10, raw_samples=256,
                    options={"batch_limit": 5, "maxiter": 200},
                    sequential=True,
                )
            cand = unnormalize(cand_n.detach(), bounds).numpy()
            step += 1

            for j in range(q):
                rec = evaluate(cand[j], idx)
                records.append(rec)
                log(idx, rec, f"acq{step}/{n_steps} n={len(Xr)}")
                idx += 1

        # ── front extraction (identical rule to the GA and RS arms) ───────────
        feas = [r for r in records if r["feasible"]
                and np.isfinite(r["fitness"]["energy"]) and np.isfinite(r["fitness"]["cl"])]
        print(f"\n[mobo] {len(feas)}/{len(records)} feasible with finite objectives")
        if not feas:
            raise SystemExit("no feasible candidate; cannot form a front")
        F = np.array([[r["fitness"]["cl"], -r["fitness"]["correctness"], r["fitness"]["energy"]]
                      for r in feas], dtype=float)
        pareto = [{"policy": feas[i]["policy"], "fitness": feas[i]["fitness"]}
                  for i in _nondominated(F)]
        _atomic_write_json(rep_dir / "pareto_solutions.json", pareto)
        _atomic_write_json(rep_dir / "intermediate_solutions.json",
                           [{"policy": r["policy"], "fitness": r["fitness"],
                             "feasible": r["feasible"]} for r in records])

        print(f"[mobo] Pareto front: {len(pareto)} policies -> "
              f"{rep_dir / 'pareto_solutions.json'}")
        print(f"[mobo] total wall {(time.perf_counter() - t0) / 3600:.2f} h")

    print(f"\n[mobo] all {args.reps} repetition(s) complete -> {out_root}")
    print(f"\nNext:\n  .venv/bin/python scripts/evaluate_heldout.py "
          f"--run-labels {run_label} --n-heldout 100 --heldout-seed 0 "
          f"--workers {n_workers} --energy-backend {args.energy_backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
