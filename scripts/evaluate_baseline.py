#!/usr/bin/env python3
"""Default-policy baseline on the held-out set, measured EXACTLY like the front.

A baseline is only comparable with the front if both are measured the same
way. Evaluating one problem at a time through `run_pipeline()` is a valid RAPL
setup, but it is not the setup the Pareto front is evaluated under, and any gap
between the two would then mix the policy effect with the execution mode.

This script instead calls `run_nsga2_runtime._mean_fitness_runtime()` with
`policy=None` — literally the same function, with the same worker count and the
same energy backend, that `evaluate_heldout.py` calls for every Pareto
policy. The only difference between baseline and front is the decoding policy,
which is the whole point.

That function implements the v3.1 two-phase protocol:
  Phase A  analyze/repair/retry/optimize across N worker threads, NO energy
  barrier  ThreadPoolExecutor shutdown joins every worker
  Phase B  serial re-execution of passing tests inside the energy window

so `rapl_sysfs` is valid at --workers 4: nothing is inferring while a window is
open. This is also why it is ~3-4x faster than the serial script.

The held-out problems are selected by the SAME code path as the front
(`select_heldout` over the GA manifest's train_ids), and the resulting id digest
is asserted equal to the front's `heldout_ids_sha256`. If the two ever diverge
the run aborts rather than producing an incomparable baseline.

Each repetition is checkpointed atomically to `reps/rep_NN.json`, so an
interruption loses at most the repetition in flight and re-running resumes.

Usage:
  python scripts/evaluate_baseline.py --run-label pop50_gen15_n50_v4 \
      --n-reps 10 --n-heldout 100 --heldout-seed 0 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

MODEL = "qwen2.5-coder:1.5b"
os.environ.setdefault("AGENT_MODEL", MODEL)


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-label", default="pop50_gen15_n50_v4",
                    help="GA run whose train_ids define the held-out complement.")
    ap.add_argument("--benchmark", default="codeflaws")
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--n-heldout", type=int, default=100)
    ap.add_argument("--heldout-seed", type=int, default=0)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("N_LOCAL_WORKERS", "4")))
    ap.add_argument("--energy-backend", default="rapl_sysfs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ga_base = REPO / "outputs" / "ga" / args.benchmark / MODEL.replace(":", "_")
    run_dir = ga_base / args.run_label
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"No GA manifest at {manifest_path}")

    import run_nsga2_runtime as ga
    from evaluate_heldout import _ids_digest, group_key, select_heldout

    from benchmarks import get_benchmark
    from energy import resolve_backend

    backend = resolve_backend(args.energy_backend)
    if backend.name != args.energy_backend:
        raise SystemExit(
            f"Requested {args.energy_backend!r} but resolved {backend.name!r}; "
            "refusing to run with F3 on an unintended scale."
        )
    ga.set_energy_backend(backend)
    ga._N_LOCAL_WORKERS = args.workers

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != MODEL:
        raise SystemExit(f"GA run used model {manifest.get('model')}, expected {MODEL}")
    train_ids = set(manifest["train_ids"])

    benchmark = get_benchmark(args.benchmark)
    all_problems = list(benchmark.problems())
    heldout = select_heldout(all_problems, train_ids, args.n_heldout, args.heldout_seed)
    heldout_ids = [p.problem_id for p in heldout]
    digest = _ids_digest(heldout_ids)

    # Independence, asserted rather than assumed.
    leak = set(heldout_ids) & train_ids
    if leak:
        raise SystemExit(f"training leakage into held-out: {sorted(leak)[:5]}")
    train_groups = {group_key(p) for p in all_problems if p.problem_id in train_ids}
    bleed = train_groups & {group_key(p) for p in heldout}
    if bleed:
        raise SystemExit(f"train/held-out share {len(bleed)} group(s)")

    # The comparability guarantee: identical problem set to the Pareto front.
    # Glob rather than deriving the exact path: a baseline run with different
    # --n-heldout/--heldout-seed would look for a front tag that does not exist
    # and silently skip the check, which is exactly the case worth catching.
    #
    # Match ANY evaluation manifest under rep_0, not just the default
    # `heldout_n*_all_pareto` tag. evaluate_heldout.py writes whatever
    # --out-tag it is given, and every replication and transfer evaluation uses
    # a custom one (`heldout_rep3_n100_seed0`, `transfer_c_rep3_n100_seed0`).
    # The narrower glob matched none of them, so on a run whose only
    # evaluations were replications the check silently skipped -- producing
    # exactly the unverified baseline this block exists to prevent. Selecting on
    # the presence of `heldout_ids_sha256` keeps non-evaluation manifests out.
    front_manifests = sorted(
        p for p in (run_dir / "rep_0").glob("*/manifest.json")
        if "heldout_ids_sha256" in json.loads(p.read_text(encoding="utf-8"))
    )
    if front_manifests:
        matching = [
            p for p in front_manifests
            if json.loads(p.read_text(encoding="utf-8")).get("heldout_ids_sha256") == digest
        ]
        if not matching:
            found = {
                p.parent.name: json.loads(p.read_text(encoding="utf-8")).get("heldout_ids_sha256", "")[:16]
                for p in front_manifests
            }
            raise SystemExit(
                "held-out set does NOT match any evaluated Pareto front:\n"
                f"  baseline {digest[:16]}…\n"
                + "".join(f"  front    {k}: {v}…\n" for k, v in found.items())
                + "Refusing to produce an incomparable baseline. Check "
                  "--n-heldout / --heldout-seed against the front's."
            )
        # Prefer a matching front that also shares this backend: several
        # evaluations of the same problem set can coexist on different backends,
        # and warning about an unrelated one would be noise.
        same_backend = [
            p for p in matching
            if json.loads(p.read_text(encoding="utf-8")).get("energy_backend")
            == args.energy_backend
        ]
        front_manifest = (same_backend or matching)[0]
        fm = json.loads(front_manifest.read_text(encoding="utf-8"))
        if fm.get("energy_backend") != args.energy_backend:
            print(f"[baseline] WARNING: front used energy_backend="
                  f"{fm.get('energy_backend')!r}, this run uses "
                  f"{args.energy_backend!r} — F3 will NOT be comparable.",
                  file=sys.stderr)
        else:
            print(f"[baseline] verified: same 100 problems and same energy "
                  f"backend as the Pareto front (sha {digest[:16]}…)")
    else:
        print("[baseline] NOTE: front manifest not found yet; problem-set "
              "equality will not be cross-checked.")

    out_dir = run_dir / "rep_0" / \
        f"baseline_default_n{args.n_heldout}_seed{args.heldout_seed}_reps{args.n_reps}"
    reps_dir = out_dir / "reps"

    print(f"[baseline] model={MODEL} reps={args.n_reps} n_heldout={len(heldout_ids)} "
          f"workers={args.workers} energy={backend.name}")
    print("[baseline] policy=None (default decoding policy)")
    print(f"[baseline] out -> {out_dir}")

    if args.dry_run:
        print("[baseline] dry run: plan validated, no Ollama calls made.")
        return 0

    _atomic_write_json(out_dir / "manifest.json", {
        "schema_version": 1,
        "script": "evaluate_baseline.py",
        "written_at": datetime.now(UTC).isoformat(),
        "benchmark": args.benchmark,
        "model": MODEL,
        "run_label": args.run_label,
        "policy": "default (policy=None)",
        "n_reps": args.n_reps,
        "n_heldout": len(heldout_ids),
        "heldout_seed": args.heldout_seed,
        "heldout_ids": heldout_ids,
        "heldout_ids_sha256": digest,
        "train_ids": sorted(train_ids),
        "n_problem_workers": args.workers,
        "energy_backend": backend.name,
        "energy_is_real_measurement": args.energy_backend != "runtime_proxy",
        "measurement": (
            "v3.1 two-phase via run_nsga2_runtime._mean_fitness_runtime — the SAME "
            "function, worker count and backend used for every Pareto policy in "
            "evaluate_heldout.py. Phase A concurrent with no measurement, "
            "thread-pool join as barrier, Phase B serial inside the energy window."
        ),
    })

    scratch_root = REPO / "outputs" / "_baseline_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    completed: list[dict] = []
    for rep in range(args.n_reps):
        rep_path = reps_dir / f"rep_{rep:02d}.json"
        if rep_path.is_file():
            completed.append(json.loads(rep_path.read_text(encoding="utf-8")))
            print(f"  [rep {rep + 1}/{args.n_reps}] resumed from checkpoint")
            continue

        print(f"  [rep {rep + 1}/{args.n_reps}] evaluating…", flush=True)
        with tempfile.TemporaryDirectory(prefix="baseline_", dir=scratch_root) as tmp:
            config = ga._build_config(Path(tmp), REPO)
            config["model"] = MODEL
            t0 = time.perf_counter()
            result = ga._mean_fitness_runtime(heldout, config, None)
            result["wall_s"] = time.perf_counter() - t0
        result["rep"] = rep
        _atomic_write_json(rep_path, result)
        completed.append(result)
        print(f"    feasible={result.get('n_feasible')}/{result.get('n_total')} "
              f"cl={result.get('mean_cl'):.3f} corr={result.get('mean_correctness'):.3f} "
              f"energy={result.get('mean_energy'):.3f} wall={result['wall_s']:.0f}s",
              flush=True)

    def agg(key: str) -> dict:
        vals = [c[key] for c in completed if c.get(key) is not None]
        return {
            "mean": st.mean(vals) if vals else None,
            "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n": len(vals),
        }

    summary = {
        "n_reps": len(completed),
        "mean_cl": agg("mean_cl"),
        "mean_correctness": agg("mean_correctness"),
        "mean_energy": agg("mean_energy"),
        "n_feasible": agg("n_feasible"),
        "heldout_ids_sha256": digest,
        "energy_backend": backend.name,
    }
    _atomic_write_json(out_dir / "summary.json", summary)

    print("\n" + "=" * 66)
    print(f"DEFAULT-POLICY BASELINE — {len(completed)} reps x {len(heldout_ids)} held-out problems")
    print("=" * 66)
    for k in ("mean_cl", "mean_correctness", "mean_energy", "n_feasible"):
        a = summary[k]
        print(f"  {k:>17}: {a['mean']:9.4f} ± {a['std']:.4f}   "
              f"[{a['min']:.3f}, {a['max']:.3f}]  n={a['n']}")
    print("=" * 66)
    print(f"summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
