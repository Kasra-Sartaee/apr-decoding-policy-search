#!/usr/bin/env python3
"""Auxiliary NSGA-II versus random-search tables.

This is **not** the generator behind the dissertation's reported results. The
three-arm results, and every table and figure the dissertation cites, come from
`analysis/three_arm_analysis.ipynb`. This script predates the surrogate arm and
covers the NSGA-II/random-search pair only; it is kept because it computes a
handful of pairwise quantities the notebook does not, and because it records
which of them could not be derived from the artefacts at all.

Its output is written to `analysis/out/`, which is untracked: the LaTeX fragment
it emits still carries `[FILL]` placeholders for quantities that were never
recoverable, so it is working material rather than a submitted artefact.

One invocation, from committed artefacts under outputs/ and results/:

    .venv/bin/python analysis/nsga2_vs_random_search_tables.py

Writes
    analysis/out/results.json           every quantity, keyed by LaTeX label,
                                        each with value / source / n
    analysis/out/results_fragment.tex   table bodies only, column order matching
                                        the chapter, [FILL] retained where null
    analysis/out/GAPS.md                what could not be computed, why, and
                                        which artefact would supply it

POLICY (dissertation 3.11), enforced here and not overridable:
  * Mann-Whitney U for between-arm comparisons.
  * Holm-Bonferroni across the three objectives within each arm pair.
  * Effect sizes: Cliff's delta and Vargha-Delaney A12 ONLY. Cohen's d is
    never computed.
  * F3 is program-execution energy (RAPL). Inference energy is descriptive
    telemetry and is never mixed into F3.
  * Energy is compared within one benchmark and within one measurement epoch.

A quantity that cannot be derived from an artefact is emitted as null with an
entry in GAPS.md. Nothing is estimated, extrapolated or simulated.

This script reports the NSGA-II and random-search arms only. The surrogate arm
is complete and is analysed in full by analysis/three_arm_analysis.ipynb, which
is the source of every reported three-arm quantity; this script was written for
the pairwise comparison and its per-pair machinery is specific to it.
"""
from __future__ import annotations

import json
import math
import platform
import statistics as st
import subprocess
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = "qwen2.5-coder_1.5b"
REPS = range(5)
MOBO_SCOPE_NOTE = (
    "Surrogate arm not reported by this script. It is complete at five "
    "repetitions and is analysed by analysis/three_arm_analysis.ipynb; this "
    "script covers the NSGA-II versus random-search pair only."
)
ARMS = ["GA", "RS"]


BENCH = {
    "primary_C": {
        "benchmark": "codeflaws",
        "root": REPO / "outputs/ga/codeflaws" / MODEL_DIR,
        "GA": ("pop50_gen15_n50_v4/rep_0", "heldout_rep{i}_n100_seed0"),
        "RS": ("rs_pop50_gen15_n50_v4/rep_0", "heldout_rsrep{i}_n100_seed0"),
        "MOBO": ("mobo_pop50_gen15_n50_v4/rep_0", "heldout_moborep{i}_n100_seed0"),
        "baseline_dir": REPO / "baselines" / "codeflaws",
        "train_manifest": {"GA": REPO / "results/nsga2/manifest.json",
                           "RS": REPO / "results/random_search/manifest.json",
                           "MOBO": REPO / "results/mobo/manifest.json"},
        "front": {"GA": REPO / "results/nsga2/rep_{i}/pareto_solutions.json",
                  "RS": REPO / "results/random_search/rep_{i}/pareto_solutions.json",
                  "MOBO": REPO / "results/mobo/rep_{i}/pareto_solutions.json"},
    },
    "secondary_Py": {
        "benchmark": "runbugrun_py",
        "root": REPO / "outputs/ga/runbugrun_py" / MODEL_DIR,
        "GA": ("py_n50_random/rep_0", "transfer_c_rep{i}_n100_seed0"),
        "RS": ("py_n50_random/rep_0", "transfer_rs_rep{i}_n100_seed0"),
        "MOBO": ("py_n50_random/rep_0", "transfer_mobo_rep{i}_n100_seed0"),
        "baseline_dir": REPO / "baselines" / "runbugrun_py",
        "train_manifest": {},                   # no search on the secondary benchmark
        "front": {"GA": REPO / "results/nsga2/rep_{i}/pareto_solutions.json",
                  "RS": REPO / "results/random_search/rep_{i}/pareto_solutions.json",
                  "MOBO": REPO / "results/mobo/rep_{i}/pareto_solutions.json"},
    },
}

OBJ = [("F1", "mean_cl", "lo"), ("F2", "mean_correctness", "hi"),
       ("F3", "mean_energy", "lo")]

GAPS: list[dict] = []
RESULTS: dict = {}


def gap(label: str, quantity: str, why: str, needs: str) -> None:
    GAPS.append({"label": label, "quantity": quantity, "why": why, "supplied_by": needs})


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO.parent))
    except ValueError:
        return str(p)


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def results_of(p):
    d = jload(p)
    return d["results"] if isinstance(d, dict) and "results" in d else d


def finite(x, *keys) -> bool:
    return all(isinstance(x.get(k), (int, float)) and x[k] == x[k]
               and math.isfinite(x[k]) for k in keys)


def rec(label, key, value, source, n=None, note=None):
    RESULTS.setdefault(label, {})[key] = {
        "value": value,
        "source": source if isinstance(source, list) else [source],
        "n": n,
        **({"note": note} if note else {}),
    }


# ── effect sizes (Cliff's delta / A12 only; Cohen's d deliberately absent) ──
def cliffs_delta(a, b) -> float:
    a, b = list(a), list(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def a12(a, b) -> float:
    """Vargha-Delaney A12 = P(X>Y) + 0.5 P(X=Y). Equals (delta+1)/2."""
    return (cliffs_delta(a, b) + 1.0) / 2.0


def delta_magnitude(d: float) -> str:
    ad = abs(d)
    return ("negligible" if ad < 0.147 else "small" if ad < 0.33
            else "medium" if ad < 0.474 else "large")


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def load_policies(bk: str, arm: str):
    cfg = BENCH[bk]
    d, tag = cfg[arm]
    out = {}
    for i in REPS:
        p = cfg["root"] / d / tag.format(i=i) / "results.json"
        if not p.is_file():
            continue
        out[i] = [x for x in results_of(p)
                  if finite(x, "mean_cl", "mean_correctness", "mean_energy")]
    return out


def load_baseline(bk: str):
    bd = BENCH[bk]["baseline_dir"]
    return {
        "dir": bd,
        "summary": jload(bd / "summary.json"),
        "manifest": jload(bd / "manifest.json"),
        "reps": [jload(p) for p in sorted((bd / "reps").glob("*.json"))],
    }


def per_problem(x):
    q = x["per_problem"]
    return q if isinstance(q, list) else list(q.values())


# ══ tab:res-config ════════════════════════════════════════════════════════
def block_config():
    LBL = "tab:res-config"
    for bk, cfg in BENCH.items():
        col = "primary" if bk == "primary_C" else "secondary"
        bench = cfg["benchmark"]
        sample = REPO / f"benchmarks/{bench}/data/sample_300_seed42.json"
        if sample.is_file():
            sj = jload(sample)
            # the sample file is a manifest, not a bare list: read the recorded
            # size and cross-check it against the problem list it carries.
            n_sample = sj.get("n_actual") or sj.get("sample_size")
            n_listed = len(sj.get("problem_dirs", []))
            if n_sample != n_listed:
                gap(LBL, f"{col}.benchmark_sample",
                    f"sample manifest reports {n_sample} but lists {n_listed} problems",
                    rel(sample))
            rec(LBL, f"{col}.benchmark_sample", f"{bench}, {n_sample} problems "
                f"(stratified sample, seed {sj.get('seed')})", rel(sample),
                n=n_sample, note=f"problem_dirs lists {n_listed}")
        else:
            rec(LBL, f"{col}.benchmark_sample", None, rel(sample))
            gap(LBL, f"{col}.benchmark_sample", "sample file not found", str(sample))

        # held-out evaluation manifest: digest, n, seed, model, workers
        d, tag = cfg["GA"]
        hm_p = cfg["root"] / d / tag.format(i=0) / "manifest.json"
        hm = jload(hm_p)
        rec(LBL, f"{col}.model", hm.get("model"), rel(hm_p))
        rec(LBL, f"{col}.eval_problems", hm.get("n_heldout"), rel(hm_p),
            n=hm.get("n_heldout"))
        rec(LBL, f"{col}.problem_set_digest", hm.get("heldout_ids_sha256"), rel(hm_p))
        rec(LBL, f"{col}.eval_seed", hm.get("heldout_seed"), rel(hm_p))
        rec(LBL, f"{col}.energy_backend", hm.get("energy_backend"), rel(hm_p))

        base = load_baseline(bk)
        rec(LBL, f"{col}.default_baseline_reps", base["summary"].get("n_reps"),
            rel(base["dir"] / "summary.json"), n=base["summary"].get("n_reps"))

        if cfg["train_manifest"]:
            tm_p = cfg["train_manifest"]["GA"]
            tm = jload(tm_p)
            rec(LBL, f"{col}.training_problems", tm.get("train_size"), rel(tm_p),
                n=tm.get("train_size"))
            rec(LBL, f"{col}.training_ids_sha256",
                __import__("hashlib").sha256(
                    "\n".join(sorted(tm["train_ids"])).encode()).hexdigest(),
                rel(tm_p), n=len(tm["train_ids"]))
            budget = (tm.get("pop") or 0) * (tm.get("gen") or 0) or tm.get("n_samples")
            rec(LBL, f"{col}.budget_per_arm", budget, rel(tm_p))
            rec(LBL, f"{col}.replicates_per_arm",
                {a: len(load_policies(bk, a)) for a in ARMS}, rel(tm_p))
            seeds = {}
            for a in ARMS:
                p = cfg["train_manifest"].get(a)
                if p and Path(p).is_file():
                    m = jload(p)
                    seeds[a] = m.get("seed", m.get("sample_seed"))
            seeds["train_subset"] = 42
            seeds["heldout"] = hm.get("heldout_seed")
            seeds["MOBO"] = None
            rec(LBL, f"{col}.seeds", seeds,
                [rel(p) for p in cfg["train_manifest"].values()])
            rec(LBL, f"{col}.reps_per_candidate", 1, rel(tm_p),
                note="one evaluation per candidate; no within-candidate repetition")
        else:
            for k in ("training_problems", "training_ids_sha256", "budget_per_arm",
                      "replicates_per_arm", "reps_per_candidate"):
                rec(LBL, f"{col}.{k}", None, None,
                    note="no search performed on the secondary benchmark")

        # wall-clock
        if cfg["train_manifest"].get("GA"):
            tm = jload(cfg["train_manifest"]["GA"])
            es = tm.get("elapsed_seconds")
            if es:
                rec(LBL, f"{col}.wallclock_GA_search_h", round(es / 3600, 1),
                    rel(cfg["train_manifest"]["GA"]), n=5,
                    note="total over 5 repetitions, search host")
            rs_m = cfg["train_manifest"].get("RS")
            if rs_m and "elapsed_seconds" not in jload(rs_m):
                rec(LBL, f"{col}.wallclock_RS_search_h", None, rel(rs_m))
                gap(LBL, "wallclock_RS_search_h",
                    "RS training manifest records no elapsed_seconds; the RS README "
                    "states '~57 h' in prose, which is not a machine-readable artefact",
                    "re-publish results/random_search/manifest.json with elapsed_seconds")
        # held-out wall-clock, this host, from the evaluator's own records
        for a in ARMS:
            d, tag = cfg[a]
            tot = 0.0
            ok = True
            for i in REPS:
                p = cfg["root"] / d / tag.format(i=i) / "results.json"
                if not p.is_file():
                    ok = False; break
                tot += sum(x.get("wall_s", 0.0) for x in results_of(p))
            rec(LBL, f"{col}.wallclock_{a}_heldout_h",
                round(tot / 3600, 1) if ok else None,
                rel(cfg["root"] / d), n=5 if ok else None,
                note="sum of per-policy wall_s on the evaluation host")
    # MOBO placeholders
    for f in ("budget_per_arm", "replicates_per_arm", "seeds", "wallclock"):
        rec(LBL, f"MOBO.{f}", None, None, note=MOBO_SCOPE_NOTE)
    gap(LBL, "MOBO.*", MOBO_SCOPE_NOTE,
        "read the three-arm notebook for these quantities")


def block_hardware():
    LBL = "sec:res-setup"
    cpu = sockets = cores = ram = None
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):   cpu = line.split(":", 1)[1].strip()
            if line.startswith("Socket(s):"):    sockets = int(line.split(":", 1)[1])
            if line.startswith("Core(s) per socket:"): cores = int(line.split(":", 1)[1])
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = round(int(line.split()[1]) / 1048576, 1)
    except Exception:
        pass
    rec(LBL, "evaluation_host.cpu", cpu, "lscpu (evaluation host)")
    rec(LBL, "evaluation_host.sockets", sockets, "lscpu (evaluation host)")
    rec(LBL, "evaluation_host.cores_per_socket", cores, "lscpu (evaluation host)")
    rec(LBL, "evaluation_host.ram_gb", ram, "/proc/meminfo (evaluation host)")
    rec(LBL, "n_hosts", 2, "run provenance",
        note="search executed on one host, held-out evaluation on another; only "
             "policy vectors were transferred, and every F3 reported here was "
             "measured on the evaluation host")
    rec(LBL, "search_host.cpu", None, None)
    gap(LBL, "search_host.{cpu,sockets,ram}",
        "no manifest or protocol records the search host's hardware",
        "add a hardware block to provenance.write_protocol_manifest and re-publish, "
        "or supply the search host's lscpu output")

    # spawn-overhead share: documented constant, cited not recomputed
    ad = REPO / "benchmarks/codeflaws/adapter.py"
    txt = ad.read_text(errors="replace")
    if "process-spawn overhead" in txt:
        rec(LBL, "spawn_overhead_share.primary", 0.60,
            f"{rel(ad)} (docstring of measure_energy), citing "
            "docs/reports/measurement_protocol.md",
            note="constant process-spawn fraction of a Phase B reading; measured "
                 "in the measurement-protocol report, not recomputed here")
    else:
        rec(LBL, "spawn_overhead_share.primary", None, rel(ad))
        gap(LBL, "spawn_overhead_share.primary", "documented constant not found", "")
    rec(LBL, "spawn_overhead_share.secondary", None, None)
    gap(LBL, "spawn_overhead_share.secondary",
        "the baseline-subtraction analysis was performed for Codeflaws only; no "
        "equivalent measurement exists for the Python runner",
        "repeat the report-15 baseline-subtraction experiment on runbugrun_py")


# ══ RQ1: training fronts ══════════════════════════════════════════════════
def block_rq1_training():
    LBL = "sec:res-rq1"
    for a in ARMS:
        sizes, per_obj = [], {o: [] for o, _, _ in OBJ}
        srcs = []
        for i in REPS:
            p = Path(str(BENCH["primary_C"]["front"][a]).format(i=i))
            if not p.is_file():
                continue
            F = jload(p); srcs.append(rel(p))
            sizes.append(len(F))
            for o, _, _ in OBJ:
                k = {"F1": "cl", "F2": "correctness", "F3": "energy"}[o]
                per_obj[o].extend(f["fitness"][k] for f in F)
        rec(LBL, f"training_front.{a}.sizes", sizes, srcs, n=len(sizes))
        for o in per_obj:
            v = [x for x in per_obj[o] if isinstance(x, (int, float)) and math.isfinite(x)]
            rec(LBL, f"training_front.{a}.{o}",
                {"min": round(min(v), 4), "median": round(st.median(v), 4),
                 "max": round(max(v), 4)}, srcs, n=len(v))
    rec(LBL, "training_front.MOBO", None, None, note=MOBO_SCOPE_NOTE)
    rec(LBL, "training_front.note",
        "Training F3 was measured on the search host and is not comparable with "
        "held-out F3 measured on the evaluation host; it is reported as a "
        "descriptive range only.", None)


# ══ tab:res-rq1: GA vs RS held-out ════════════════════════════════════════
UNIT = ("policy", "Each observation is one front policy, scored as its mean over "
                  "the 100 held-out problems. Policies from one repetition share a "
                  "search trajectory and are not independent; the repetition-level "
                  "test is reported alongside as unit_note.")


def block_rq1_heldout():
    LBL = "tab:res-rq1"
    rec(LBL, "unit_of_analysis", UNIT[0], None, note=UNIT[1])
    for bk in BENCH:
        col = "primary" if bk == "primary_C" else "secondary"
        pol = {a: load_policies(bk, a) for a in ARMS}
        srcs = [rel(BENCH[bk]["root"] / BENCH[bk][a][0]) for a in ARMS]
        raw = []
        for o, key, better in OBJ:
            x = [v[key] for i in REPS for v in pol["GA"].get(i, [])]
            y = [v[key] for i in REPS for v in pol["RS"].get(i, [])]
            U, p = mannwhitneyu(x, y, alternative="two-sided")
            raw.append((o, U, p, x, y))
        adj = holm([r[2] for r in raw])
        for (o, U, p, x, y), pa in zip(raw, adj):
            d = cliffs_delta(x, y)
            better = dict((k, b) for k, _, b in OBJ)[o]
            if abs(d) < 1e-12:
                direction = "no difference"
            else:
                ga_higher = d > 0
                if better == "hi":
                    direction = "GA higher" if ga_higher else "RS higher"
                else:
                    direction = "GA higher (worse)" if ga_higher else "RS higher (worse)"
            rec(LBL, f"{col}.GA_vs_RS.{o}", {
                "U": float(U), "p_nominal": float(p), "p_holm": float(pa),
                "A12": round(a12(x, y), 4), "cliffs_delta": round(d, 4),
                "delta_magnitude": delta_magnitude(d),
                "direction": direction,
                "n_GA": len(x), "n_RS": len(y),
            }, srcs, n=len(x) + len(y))
        # repetition-level companion, same objectives, Holm within the family
        raw_r = []
        for o, key, better in OBJ:
            gr = [st.mean([v[key] for v in pol["GA"][i]]) for i in REPS]
            rr = [st.mean([v[key] for v in pol["RS"][i]]) for i in REPS]
            U, p = mannwhitneyu(gr, rr, alternative="two-sided")
            raw_r.append((o, U, p, gr, rr))
        adj_r = holm([r[2] for r in raw_r])
        for (o, U, p, gr, rr), pa in zip(raw_r, adj_r):
            rec(LBL, f"{col}.GA_vs_RS.{o}.per_repetition", {
                "U": float(U), "p_nominal": float(p), "p_holm": float(pa),
                "A12": round(a12(gr, rr), 4), "cliffs_delta": round(cliffs_delta(gr, rr), 4),
                "n_GA": len(gr), "n_RS": len(rr),
            }, srcs, n=len(gr) + len(rr),
                note="observation = one search repetition; the unit that generalises "
                     "to the choice of algorithm")
        for x, y in (("GA", "MOBO"), ("MOBO", "RS")):
            rec(LBL, f"{col}.{x}_vs_{y}", None, None, note=MOBO_SCOPE_NOTE)


# ══ each arm vs the default baseline ══════════════════════════════════════
def block_vs_baseline():
    LBL = "sec:res-rq1"
    for bk in BENCH:
        col = "primary" if bk == "primary_C" else "secondary"
        base = load_baseline(bk)
        bsrc = rel(base["dir"] / "summary.json")
        for o, key, better in OBJ:
            s = base["summary"][key]
            rec(LBL, f"{col}.baseline.{o}",
                {"mean": round(s["mean"], 6), "sd": round(s["std"], 6),
                 "min": round(s["min"], 6), "max": round(s["max"], 6)},
                bsrc, n=s["n"])
        for a in ARMS:
            pol = load_policies(bk, a)
            src = rel(BENCH[bk]["root"] / BENCH[bk][a][0])
            raw = []
            for o, key, better in OBJ:
                x = [v[key] for i in REPS for v in pol.get(i, [])]
                y = [r[key] for r in base["reps"]]
                U, p = mannwhitneyu(x, y, alternative="two-sided")
                raw.append((o, U, p, x, y))
            adj = holm([r[2] for r in raw])
            for (o, U, p, x, y), pa in zip(raw, adj):
                d = cliffs_delta(x, y)
                rec(LBL, f"{col}.{a}_vs_baseline.{o}", {
                    "U": float(U), "p_nominal": float(p), "p_holm": float(pa),
                    "A12": round(a12(x, y), 4), "cliffs_delta": round(d, 4),
                    "delta_magnitude": delta_magnitude(d),
                    "arm_mean": round(st.mean(x), 6),
                    "baseline_mean": round(st.mean(y), 6),
                    "n_arm": len(x), "n_baseline": len(y),
                }, [src, bsrc], n=len(x) + len(y))
        rec(LBL, f"{col}.MOBO_vs_baseline", None, None, note=MOBO_SCOPE_NOTE)


# ══ tab:res-rq1-hv: set-quality indicators ════════════════════════════════
def _nd(F):
    return np.array([i for i in range(len(F))
                     if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                                for j in range(len(F)) if j != i)], int)


def _hv(F, ref, seed=20260829, n=200_000):
    rng = np.random.default_rng(seed)
    P = F[_nd(F)] if len(F) else F
    if not len(P):
        return 0.0
    S = rng.random((n, F.shape[1])) * ref
    dom = np.zeros(len(S), bool)
    for p in P:
        dom |= np.all(S >= p, axis=1)
    return float(dom.mean() * np.prod(ref))


def _igdplus(F, R):
    return float(np.mean([min(np.sqrt(np.sum(np.maximum(f - r, 0) ** 2)) for f in F)
                          for r in R]))


def block_front_quality():
    LBL = "tab:res-rq1-hv"
    MARGINS = [1.02, 1.05, 1.10, 1.20, 1.35, 1.50]
    MAIN, DRAWS = 1.10, 40
    for bk in BENCH:
        col = "primary" if bk == "primary_C" else "secondary"
        pol = {a: load_policies(bk, a) for a in ARMS}
        srcs = [rel(BENCH[bk]["root"] / BENCH[bk][a][0]) for a in ARMS]
        M = {(a, i): np.array([[v["mean_cl"], -v["mean_correctness"], v["mean_energy"]]
                               for v in pol[a][i]], float) for a in ARMS for i in REPS}
        allF = np.vstack(list(M.values()))
        ideal, nadir = allF.min(0), allF.max(0)
        span = np.where(nadir - ideal == 0, 1.0, nadir - ideal)
        N = {k: (v - ideal) / span for k, v in M.items()}
        union = np.vstack(list(N.values()))
        REF_SET = union[_nd(union)]
        rec(LBL, f"{col}.reference_point",
            {"rule": "union ideal/nadir over both arms, all repetitions; objectives "
                     "mapped to minimisation and normalised to the unit cube; "
                     "reference point = margin x (1,1,1)",
             "margin_main": MAIN, "ideal": [round(v, 6) for v in ideal],
             "nadir": [round(v, 6) for v in nadir]}, srcs)
        rec(LBL, f"{col}.reference_set",
            {"rule": "non-dominated union of every front on this benchmark",
             "size": int(len(REF_SET))}, srcs, n=int(len(REF_SET)))
        K = min(len(N[(a, i)]) for a in ARMS for i in REPS)
        rec(LBL, f"{col}.cardinality_control",
            {"subsample_size": int(K), "subsamples_per_repetition": DRAWS}, srcs)
        rng = np.random.default_rng(20260829)
        for a in ARMS:
            sizes = [len(N[(a, i)]) for i in REPS]
            hv_raw = [_hv(N[(a, i)], np.full(3, MAIN)) for i in REPS]
            ig_raw = [_igdplus(N[(a, i)], REF_SET) for i in REPS]
            hv_ctl, ig_ctl = [], []
            for i in REPS:
                S = N[(a, i)]
                idxs = [rng.choice(len(S), K, replace=False) for _ in range(DRAWS)]
                hv_ctl.append(st.mean([_hv(S[ix], np.full(3, MAIN)) for ix in idxs]))
                ig_ctl.append(st.mean([_igdplus(S[ix], REF_SET) for ix in idxs]))
            rec(LBL, f"{col}.{a}", {
                "front_size_mean": round(st.mean(sizes), 2),
                "front_sizes": sizes,
                "HV_raw_mean": round(st.mean(hv_raw), 4),
                "HV_raw_sd": round(st.stdev(hv_raw), 4),
                "HV_controlled_mean": round(st.mean(hv_ctl), 4),
                "HV_controlled_sd": round(st.stdev(hv_ctl), 4),
                "IGDplus_raw_mean": round(st.mean(ig_raw), 4),
                "IGDplus_raw_sd": round(st.stdev(ig_raw), 4),
                "IGDplus_controlled_mean": round(st.mean(ig_ctl), 4),
                "IGDplus_controlled_sd": round(st.stdev(ig_ctl), 4),
            }, srcs, n=len(REPS))
        # reference-margin sensitivity: is the GA/RS ordering stable?
        orderings = {}
        for m in MARGINS:
            means = {a: st.mean([_hv(N[(a, i)], np.full(3, m)) for i in REPS]) for a in ARMS}
            orderings[m] = "GA" if means["GA"] > means["RS"] else "RS"
        rec(LBL, f"{col}.reference_sensitivity", {
            "margins": MARGINS, "leader_by_margin": orderings,
            "ordering_stable": len(set(orderings.values())) == 1,
        }, srcs, n=len(MARGINS))
        rec(LBL, f"{col}.MOBO", None, None, note=MOBO_SCOPE_NOTE)


# ══ tab:res-rq2: policy transfer ══════════════════════════════════════════
def block_rq2_transfer():
    LBL = "tab:res-rq2"
    for bk in BENCH:
        col = "primary" if bk == "primary_C" else "secondary"
        base = load_baseline(bk)
        bsrc = rel(base["dir"] / "summary.json")
        for a in ARMS:
            pol = load_policies(bk, a)
            src = rel(BENCH[bk]["root"] / BENCH[bk][a][0])
            allp = [v for i in REPS for v in pol[i]]
            entry = {"front_size_total": len(allp),
                     "front_sizes": [len(pol[i]) for i in REPS]}
            for o, key, better in OBJ:
                bm = base["summary"][key]["mean"]
                vals = [v[key] for v in allp]
                above = sum(1 for v in vals if (v > bm if better == "hi" else v < bm))
                deltas = [(v - bm) if better == "hi" else (bm - v) for v in vals]
                entry[o] = {"n_above_baseline": above,
                            "share_above": round(above / len(vals), 4),
                            "median_delta": round(st.median(deltas), 6),
                            "baseline_mean": round(bm, 6),
                            "direction": "higher is better" if better == "hi"
                                         else "lower is better"}
            rec(LBL, f"{col}.{a}", entry, [src, bsrc], n=len(allp))
        rec(LBL, f"{col}.MOBO", None, None, note=MOBO_SCOPE_NOTE)
    rec(LBL, "above_baseline_definition",
        "A policy counts as above baseline on an objective if its held-out mean "
        "beats the baseline's mean over its 5 repetitions, in that objective's "
        "direction. Reported per objective; there is no single-objective "
        "aggregate.", None)


# ══ sec:res-rq2: rank correlation, geometry, difficulty, F3 ═══════════════
def _spearman_ci(rho, n, alpha=0.05):
    if n < 4 or abs(rho) >= 1.0:
        return None
    z = math.atanh(rho); se = 1.0 / math.sqrt(n - 3)
    from scipy.stats import norm
    c = norm.ppf(1 - alpha / 2)
    return [round(math.tanh(z - c * se), 4), round(math.tanh(z + c * se), 4)]


def block_rq2_generalisation():
    LBL = "sec:res-rq2"
    TK = {"F1": "cl", "F2": "correctness", "F3": "energy"}
    for bk in BENCH:
        col = "primary" if bk == "primary_C" else "secondary"
        for a in ARMS:
            pol = load_policies(bk, a)
            src = rel(BENCH[bk]["root"] / BENCH[bk][a][0])
            recs = [v for i in REPS for v in pol[i] if v.get("train_fitness")]
            for o, key, _ in OBJ:
                t = [v["train_fitness"][TK[o]] for v in recs]
                h = [v[key] for v in recs]
                keep = [(x, y) for x, y in zip(t, h)
                        if all(isinstance(q, (int, float)) and math.isfinite(q) for q in (x, y))]
                if len(keep) < 4:
                    rec(LBL, f"{col}.rank_corr.{a}.{o}", None, src); continue
                rho, p = spearmanr([k[0] for k in keep], [k[1] for k in keep])
                rec(LBL, f"{col}.rank_corr.{a}.{o}", {
                    "spearman_rho": round(float(rho), 4), "p": float(p),
                    "ci95_fisher": _spearman_ci(float(rho), len(keep)),
                }, src, n=len(keep),
                    note="training fitness measured on the search host; held-out on "
                         "the evaluation host. For F3 the two are on different "
                         "instruments, so that row is a rank comparison only.")
        rec(LBL, f"{col}.rank_corr.MOBO", None, None, note=MOBO_SCOPE_NOTE)

        # trade-off geometry: inter-objective correlation, training vs held-out
        for a in ARMS:
            pol = load_policies(bk, a)
            recs = [v for i in REPS for v in pol[i] if v.get("train_fitness")]
            geom = {}
            for (o1, k1, _), (o2, k2, _) in combinations(OBJ, 2):
                tr = spearmanr([v["train_fitness"][TK[o1]] for v in recs],
                               [v["train_fitness"][TK[o2]] for v in recs])[0]
                ho = spearmanr([v[k1] for v in recs], [v[k2] for v in recs])[0]
                geom[f"{o1}_{o2}"] = {"training_rho": round(float(tr), 4),
                                      "heldout_rho": round(float(ho), 4),
                                      "sign_preserved": bool((tr > 0) == (ho > 0))}
            rec(LBL, f"{col}.tradeoff_geometry.{a}", geom,
                rel(BENCH[bk]["root"] / BENCH[bk][a][0]), n=len(recs),
                note="measure: Spearman correlation between each pair of objectives, "
                     "computed over front policies, on training and on held-out. "
                     "'sign_preserved' is whether the direction of the trade-off is "
                     "the same in both.")

        # intrinsic difficulty
        def prob_vec(arm):
            acc = {}
            for i in REPS:
                for v in load_policies(bk, arm)[i]:
                    for q in per_problem(v):
                        c = q.get("correctness")
                        if isinstance(c, (int, float)) and c == c:
                            acc.setdefault(q["problem_id"], []).append(c)
            return {k: st.mean(v) for k, v in acc.items()}
        pv = {a: prob_vec(a) for a in ARMS}
        common = sorted(set(pv["GA"]) & set(pv["RS"]))
        rho_arms, p_arms = spearmanr([pv["GA"][k] for k in common],
                                     [pv["RS"][k] for k in common])
        rec(LBL, f"{col}.difficulty.across_arms",
            {"spearman_rho": round(float(rho_arms), 4), "p": float(p_arms),
             "ci95_fisher": _spearman_ci(float(rho_arms), len(common))},
            [rel(BENCH[bk]["root"] / BENCH[bk][a][0]) for a in ARMS], n=len(common),
            note="per-problem mean correctness, GA policies vs RS policies")
        for a in ARMS:
            pol = load_policies(bk, a)
            mats = []
            for i in REPS:
                for v in pol[i]:
                    d = {q["problem_id"]: q.get("correctness") for q in per_problem(v)}
                    mats.append(d)
            keys = sorted(set().union(*[set(m) for m in mats]))
            arr = np.array([[m.get(k, np.nan) for k in keys] for m in mats], float)
            good = ~np.isnan(arr).any(0)
            A = arr[:, good]
            rhos = []
            rng = np.random.default_rng(20260829)
            pairs = rng.choice(len(A), size=(min(500, len(A) * (len(A) - 1) // 2), 2))
            for u, w in pairs:
                if u == w:
                    continue
                r = spearmanr(A[u], A[w])[0]
                if r == r:
                    rhos.append(float(r))
            rec(LBL, f"{col}.difficulty.within_{a}",
                {"median_pairwise_spearman": round(st.median(rhos), 4),
                 "n_policy_pairs": len(rhos)},
                rel(BENCH[bk]["root"] / BENCH[bk][a][0]), n=int(good.sum()),
                note="pairwise Spearman between policies' per-problem correctness "
                     "vectors, sampled")

        # F3 specifically
        for a in ARMS:
            pol = load_policies(bk, a)
            src = rel(BENCH[bk]["root"] / BENCH[bk][a][0])
            recs = [v for i in REPS for v in pol[i] if v.get("train_fitness")]
            t = [v["train_fitness"]["energy"] for v in recs]
            h = [v["mean_energy"] for v in recs]
            keep = [(x, y) for x, y in zip(t, h)
                    if all(isinstance(q, (int, float)) and math.isfinite(q) for q in (x, y))]
            rho, p = spearmanr([k[0] for k in keep], [k[1] for k in keep])
            allv = np.array([v["mean_energy"] for i in REPS for v in pol[i]], float)
            trimmed = float(np.mean(np.sort(allv)[int(.05 * len(allv)):
                                                  len(allv) - int(.05 * len(allv))]))
            rec(LBL, f"{col}.F3.{a}", {
                "train_vs_heldout_spearman": round(float(rho), 4), "p": float(p),
                "ci95_fisher": _spearman_ci(float(rho), len(keep)),
                "heldout_mean": round(float(allv.mean()), 4),
                "heldout_median": round(float(np.median(allv)), 4),
                "heldout_trimmed_mean_10pct": round(trimmed, 4),
                "mean_over_median": round(float(allv.mean() / np.median(allv)), 4),
            }, src, n=len(keep),
                note="robust statistic = 10% symmetric trimmed mean, reported "
                     "alongside the median and the mean. Training F3 was measured "
                     "on the search host, held-out F3 on the evaluation host; the "
                     "correlation is therefore rank-only and no ratio between them "
                     "is reported.")
        ga_m = st.mean([v["mean_energy"] for i in REPS for v in load_policies(bk, "GA")[i]])
        rs_m = st.mean([v["mean_energy"] for i in REPS for v in load_policies(bk, "RS")[i]])
        ga_d = st.median([v["mean_energy"] for i in REPS for v in load_policies(bk, "GA")[i]])
        rs_d = st.median([v["mean_energy"] for i in REPS for v in load_policies(bk, "RS")[i]])
        rec(LBL, f"{col}.F3.estimator_agreement", {
            "lower_by_mean": "GA" if ga_m < rs_m else "RS",
            "lower_by_median": "GA" if ga_d < rs_d else "RS",
            "agree": ("GA" if ga_m < rs_m else "RS") == ("GA" if ga_d < rs_d else "RS"),
        }, None, note="whether mean and median order the two arms the same way")


# ══ tab:res-rq2-transfer: cross-benchmark ═════════════════════════════════
def block_cross_benchmark():
    LBL = "tab:res-rq2-transfer"
    bk = "secondary_Py"
    base = load_baseline(bk)
    bsrc = rel(base["dir"] / "summary.json")
    for a in ARMS:
        pol = load_policies(bk, a)
        if not pol:
            rec(LBL, a, None, None)
            gap(LBL, a, "no secondary-benchmark evaluation for this arm", "")
            continue
        src = rel(BENCH[bk]["root"] / BENCH[bk][a][0])
        allp = [v for i in REPS for v in pol[i]]
        entry = {"front_size_total": len(allp)}
        for o, key, better in OBJ:
            vals = [v[key] for v in allp]
            bm = base["summary"][key]["mean"]
            above = sum(1 for v in vals if (v > bm if better == "hi" else v < bm))
            entry[o] = {"mean": round(st.mean(vals), 6),
                        "median": round(st.median(vals), 6),
                        "baseline_mean": round(bm, 6),
                        "baseline_sd": round(base["summary"][key]["std"], 6),
                        "n_above_baseline": above}
        rec(LBL, a, entry, [src, bsrc], n=len(allp))
    for o, key, _ in OBJ:
        s = base["summary"][key]
        rec(LBL, f"secondary_baseline.{o}",
            {"mean": round(s["mean"], 6), "sd": round(s["std"], 6)}, bsrc, n=s["n"])
    rec(LBL, "MOBO", None, None, note=MOBO_SCOPE_NOTE)
    rec(LBL, "scope",
        "Every row is measured on the secondary benchmark's evaluation problems "
        "with rapl_sysfs on the evaluation host, so the table is internally "
        "comparable. It is not a comparison between benchmarks: the F1 convention "
        "differs by language and F3 is not comparable across a compiled and an "
        "interpreted runtime. No front was searched on the secondary benchmark, "
        "so there is no upper bound against which a shortfall could be measured.",
        None)


# ══ tab:res-compute: cost of the search ═══════════════════════════════════
def block_cost():
    LBL = "tab:res-compute"
    bk = "primary_C"
    for a in ARMS:
        pol = load_policies(bk, a)
        src = rel(BENCH[bk]["root"] / BENCH[bk][a][0])
        rr, orr = [], []
        for i in REPS:
            for v in pol[i]:
                for q in per_problem(v):
                    if isinstance(q.get("repair_retries_used"), (int, float)):
                        rr.append(q["repair_retries_used"])
                    if isinstance(q.get("opt_retries_used"), (int, float)):
                        orr.append(q["opt_retries_used"])
        rec(LBL, f"{a}.retry_budget", {
            "repair_retries_mean": round(st.mean(rr), 4) if rr else None,
            "optimiser_retries_mean": round(st.mean(orr), 4) if orr else None,
            "combined_mean": round(st.mean(rr) + st.mean(orr), 4) if rr and orr else None,
            "unit": "per problem evaluation (policy x problem), held-out",
        }, src, n=len(rr))
        tot = sum(v.get("wall_s", 0.0) for i in REPS for v in pol[i])
        rec(LBL, f"{a}.wallclock_heldout_h", round(tot / 3600, 2), src, n=len(list(REPS)),
            note="evaluation host; the search wall-clock is a separate row")
        rec(LBL, f"{a}.generation_calls", None, None)
        rec(LBL, f"{a}.inference_energy_kJ", None, None)
    tm = jload(REPO / "results/nsga2/manifest.json")
    rec(LBL, "GA.wallclock_search_h", round(tm["elapsed_seconds"] / 3600, 1),
        rel(REPO / "results/nsga2/manifest.json"), n=5,
        note="search host, total over 5 repetitions")
    rec(LBL, "RS.wallclock_search_h", None, None)
    rec(LBL, "MOBO", None, None, note=MOBO_SCOPE_NOTE)

    gap(LBL, "{GA,RS}.generation_calls",
        "no held-out or training artefact records a count of LLM generation calls; "
        "the per-problem records carry retry counts but not call totals",
        "re-run with telemetry enabled, or publish the per-call telemetry rows the "
        "pipeline can emit (telemetry.py) alongside each run")
    gap(LBL, "{GA,RS}.inference_energy_kJ",
        "inference energy (NVML/RAPL around each chat() call) is not present in any "
        "committed artefact for these runs; F3 is program-execution energy and is "
        "never substituted for it",
        "re-run with AGENT_INFERENCE_ENERGY=1 and publish the telemetry, or supply "
        "the search host's telemetry records for the existing runs")
    gap("sec:res-telemetry", "energy_ledger",
        "the ledger requires total inference energy consumed by each arm, which is "
        "unavailable (above). The F3 saving per execution is computable, but a "
        "break-even count cannot be formed without the numerator",
        "inference-energy telemetry for the GA and RS search runs")

    # the half of the ledger that IS computable
    base = load_baseline(bk)
    for a in ARMS:
        pol = load_policies(bk, a)
        vals = [v["mean_energy"] for i in REPS for v in pol[i]]
        bm = base["summary"]["mean_energy"]["mean"]
        rec("sec:res-telemetry", f"{a}.F3_saving_per_execution_mJ", {
            "arm_mean": round(st.mean(vals), 4), "baseline_mean": round(bm, 4),
            "saving_mean_mJ": round(bm - st.mean(vals), 4),
            "arm_median": round(st.median(vals), 4),
            "baseline_median": None,
            "saving_median_mJ": None,
        }, [rel(BENCH[bk]["root"] / BENCH[bk][a][0]),
            rel(base["dir"] / "summary.json")], n=len(vals),
            note="mean per-test-execution F3 of the arm's front policies against the "
                 "repeated default baseline, primary benchmark. The break-even "
                 "count is not computable: see GAPS.")


# ══ measurement epochs ════════════════════════════════════════════════════
def block_epochs():
    LBL = "sec:res-telemetry"
    readmes = {
        "GA": REPO / "results/nsga2/README.md",
        "RS": REPO / "results/random_search/README.md",
    }
    found = {}
    for a, p in readmes.items():
        if p.is_file() and "reboot" in p.read_text(errors="replace").lower():
            found[a] = rel(p)
    rec(LBL, "epochs.search_host", {
        "boundary": "a host restart between repetition 0 and repetition 1 on the "
                    "search host, reported in the arm READMEs",
        "consequence": "training F3 is not pooled across that boundary",
        "documented_in": found,
    }, list(found.values()) or None,
        note="prose in the published READMEs; not recorded as a machine-readable "
             "field in any manifest")
    rec(LBL, "epochs.evaluation_host", {
        "boundary": "none",
        "evidence": "every held-out and baseline measurement reported here was taken "
                    "on one host with rapl_sysfs, on an idle machine, with a stray-"
                    "workload sweep before and after each stage",
        "verified": "the search host's epoch boundary does not propagate: measured "
                    "ratio of repetition-0 to later-repetition held-out median F3",
    }, None)
    for bk in BENCH:
        col = "primary" if bk == "primary_C" else "secondary"
        for a in ARMS:
            pol = load_policies(bk, a)
            med = {i: st.median([v["mean_energy"] for v in pol[i]]) for i in REPS}
            later = st.mean([med[i] for i in (1, 2, 3, 4)])
            rec(LBL, f"epochs.{col}.{a}.rep0_vs_later_F3_ratio",
                round(later / med[0], 4),
                rel(BENCH[bk]["root"] / BENCH[bk][a][0]), n=5,
                note="1.00 indicates the search host's epoch boundary did not reach "
                     "the locally measured F3")
    rec(LBL, "epochs.comparisons_within_one_epoch",
        "All energy comparisons in this section are drawn between measurements taken "
        "on the evaluation host in a single epoch. Training F3 is never compared with "
        "held-out F3, and no ratio between the two hosts is reported.", None)


# ══ emitters ══════════════════════════════════════════════════════════════
FILL = "[FILL]"


def fmt(v, nd=4):
    if v is None:
        return FILL
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, (list, dict)):
        return str(v)
    return str(v)


def get(label, key, *path, nd=4):
    node = RESULTS.get(label, {}).get(key)
    if node is None or node.get("value") is None:
        return FILL
    v = node["value"]
    for p in path:
        if not isinstance(v, dict) or p not in v or v[p] is None:
            return FILL
        v = v[p]
    return fmt(v, nd)


def emit_tex() -> str:
    L = []
    add = L.append
    add("% Generated by analysis/nsga2_vs_random_search_tables.py -- do not edit by hand.")
    add(f"% {datetime.now().isoformat(timespec='seconds')}")
    add("% Table bodies only. [FILL] marks a quantity that could not be computed;")
    add("% see analysis/out/GAPS.md for why and what would supply it.")
    add("")

    # ---- tab:res-config -------------------------------------------------
    add("% ---- tab:res-config ----")
    C = "tab:res-config"
    rows = [
        ("Benchmark and sample", get(C, "primary.benchmark_sample"), get(C, "secondary.benchmark_sample")),
        ("Model", get(C, "primary.model"), get(C, "secondary.model")),
        ("Training problems", get(C, "primary.training_problems"), "--- (no search)"),
        ("Evaluation problems", get(C, "primary.eval_problems"), get(C, "secondary.eval_problems")),
        ("Problem-set digest", f"\\texttt{{{get(C,'primary.problem_set_digest')[:16]}\\ldots}}",
                               f"\\texttt{{{get(C,'secondary.problem_set_digest')[:16]}\\ldots}}"),
        ("Evaluation budget per arm", get(C, "primary.budget_per_arm"), "---"),
        ("Repetitions per candidate", get(C, "primary.reps_per_candidate"), "---"),
        ("Replicates per arm", get(C, "primary.replicates_per_arm"), get(C, "secondary.replicates_per_arm")),
        ("Default-baseline reps", get(C, "primary.default_baseline_reps"), get(C, "secondary.default_baseline_reps")),
        ("Energy backend", "\\texttt{rapl\\_sysfs}", "\\texttt{rapl\\_sysfs}"),
        ("Seeds (train/eval/GA/RS/MOBO)", get(C, "primary.seeds"), get(C, "secondary.seeds")),
        ("Wall-clock (search, GA)", f"{get(C,'primary.wallclock_GA_search_h',nd=1)} h", "---"),
        ("Wall-clock (search, RS)", get(C, "primary.wallclock_RS_search_h"), "---"),
        ("Wall-clock (held-out, GA)", f"{get(C,'primary.wallclock_GA_heldout_h',nd=1)} h",
                                      f"{get(C,'secondary.wallclock_GA_heldout_h',nd=1)} h"),
        ("Wall-clock (held-out, RS)", f"{get(C,'primary.wallclock_RS_heldout_h',nd=1)} h",
                                      f"{get(C,'secondary.wallclock_RS_heldout_h',nd=1)} h"),
    ]
    for a, b, c in rows:
        add(f"{a} & {b} & {c} \\\\")
    add("")

    # ---- tab:res-rq1 ----------------------------------------------------
    add("% ---- tab:res-rq1 (GA vs RS; MOBO rows retain [FILL]) ----")
    add(f"% unit of analysis: {RESULTS['tab:res-rq1']['unit_of_analysis']['value']}")
    for pair in ("GA_vs_RS",):
        for o, _, _ in OBJ:
            k = f"primary.{pair}.{o}"
            node = RESULTS["tab:res-rq1"].get(k, {}).get("value")
            lab = "GA vs.\\ RS" if o == "F1" else ""
            if node:
                add(f"{lab} & {o} & {node['U']:.0f}, {node['p_nominal']:.3g} & "
                    f"{node['p_holm']:.3g} & {node['A12']:.3f} & {node['direction']} \\\\")
            else:
                add(f"{lab} & {o} & {FILL} & {FILL} & {FILL} & {FILL} \\\\")
    for pair, lab in (("GA vs.\\ MOBO", "GA_vs_MOBO"), ("MOBO vs.\\ RS", "MOBO_vs_RS")):
        for j, (o, _, _) in enumerate(OBJ):
            add(f"{pair if j==0 else ''} & {o} & {FILL} & {FILL} & {FILL} & {FILL} \\\\")
    add("")

    # ---- tab:res-rq1-hv -------------------------------------------------
    add("% ---- tab:res-rq1-hv ----")
    for arm, lab in (("GA", "NSGA-II"), ("MOBO", "MOBO"), ("RS", "Random sampling")):
        n = RESULTS["tab:res-rq1-hv"].get(f"primary.{arm}", {}).get("value")
        if n:
            add(f"{lab} & {n['front_size_mean']:.1f} & {n['HV_raw_mean']:.4f} & "
                f"{n['HV_controlled_mean']:.4f} & {n['IGDplus_raw_mean']:.4f} & "
                f"{n['IGDplus_controlled_mean']:.4f} \\\\")
        else:
            add(f"{lab} & {FILL} & {FILL} & {FILL} & {FILL} & {FILL} \\\\")
    add("")

    # ---- tab:res-rq2 ----------------------------------------------------
    add("% ---- tab:res-rq2 (counts and median delta are per objective; F2 shown) ----")
    for arm, lab in (("GA", "NSGA-II"), ("MOBO", "MOBO"), ("RS", "Random sampling")):
        n = RESULTS["tab:res-rq2"].get(f"primary.{arm}", {}).get("value")
        if n:
            add(f"{lab} & {n['front_size_total']} & {n['F2']['n_above_baseline']} & "
                f"{n['F2']['median_delta']:+.4f} \\\\")
        else:
            add(f"{lab} & {FILL} & {FILL} & {FILL} \\\\")
    add("")

    # ---- tab:res-rq2-transfer -------------------------------------------
    add("% ---- tab:res-rq2-transfer ----")
    for arm, lab in (("GA", "NSGA-II"), ("MOBO", "MOBO"), ("RS", "Random sampling")):
        n = RESULTS["tab:res-rq2-transfer"].get(arm, {}).get("value")
        if n:
            add(f"{lab} & {n['front_size_total']} & {n['F1']['mean']:.3f} & "
                f"{n['F2']['mean']:.4f} & {n['F3']['mean']:.2f} & "
                f"{n['F2']['n_above_baseline']} \\\\")
        else:
            add(f"{lab} & {FILL} & {FILL} & {FILL} & {FILL} & {FILL} \\\\")
    b1 = RESULTS["tab:res-rq2-transfer"].get("secondary_baseline.F1", {}).get("value")
    b2 = RESULTS["tab:res-rq2-transfer"].get("secondary_baseline.F2", {}).get("value")
    b3 = RESULTS["tab:res-rq2-transfer"].get("secondary_baseline.F3", {}).get("value")
    add(f"Secondary default baseline & --- & {b1['mean']:.3f} & {b2['mean']:.4f} & "
        f"{b3['mean']:.2f} & --- \\\\" if b1 else f"Secondary default baseline & --- & {FILL} & {FILL} & {FILL} & --- \\\\")
    add("")

    # ---- tab:res-compute ------------------------------------------------
    add("% ---- tab:res-compute ----")
    for arm, lab in (("GA", "NSGA-II"), ("MOBO", "MOBO"), ("RS", "Random sampling")):
        rb = RESULTS["tab:res-compute"].get(f"{arm}.retry_budget", {}).get("value")
        wc = RESULTS["tab:res-compute"].get(f"{arm}.wallclock_search_h", {}).get("value")
        add(f"{lab} & {rb['combined_mean']:.2f} & {fmt(wc,1) if wc else FILL} & "
            f"{FILL} & {FILL} \\\\" if rb else
            f"{lab} & {FILL} & {FILL} & {FILL} & {FILL} \\\\")
    return "\n".join(L) + "\n"


def emit_gaps() -> str:
    L = ["# Gaps", "",
         "Quantities cited in `5_Results_and_Analysis.tex` that could not be computed",
         "from committed artefacts. Nothing below was estimated or substituted.", "",
         f"Generated {datetime.now().isoformat(timespec='seconds')} by "
         "`analysis/nsga2_vs_random_search_tables.py`.", ""]
    by_label: dict[str, list] = {}
    for g in GAPS:
        by_label.setdefault(g["label"], []).append(g)
    for label, items in by_label.items():
        L.append(f"## `{label}`")
        L.append("")
        for g in items:
            L.append(f"**{g['quantity']}**")
            L.append("")
            L.append(f"- *Why:* {g['why']}")
            if g["supplied_by"]:
                L.append(f"- *Supplied by:* {g['supplied_by']}")
            L.append("")
    L += ["## Scope", "",
          f"- **Surrogate arm.** {MOBO_SCOPE_NOTE}", "",
          "- **Cohen's d** is not computed anywhere in this script, per the",
          "  statistical policy. Effect sizes are Cliff's delta and Vargha-Delaney",
          "  A12 only.", ""]
    return "\n".join(L)


def main() -> int:
    block_config()
    block_hardware()
    block_rq1_training()
    block_rq1_heldout()
    block_vs_baseline()
    block_front_quality()
    block_rq2_transfer()
    block_rq2_generalisation()
    block_cross_benchmark()
    block_cost()
    block_epochs()

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "analysis/nsga2_vs_random_search_tables.py",
        "python": platform.python_version(),
        "arms_included": ARMS,
        "arms_excluded": {"MOBO": MOBO_SCOPE_NOTE},
        "statistical_policy": {
            "between_arm_test": "Mann-Whitney U (two-sided)",
            "correction": "Holm-Bonferroni across the three objectives within each arm pair",
            "effect_sizes": ["Cliff's delta", "Vargha-Delaney A12"],
            "cohens_d": "not computed anywhere, by policy",
        },
        "n_gaps": len(GAPS),
    }
    (OUT / "results.json").write_text(
        json.dumps({"_meta": meta, **RESULTS}, indent=2, default=str), encoding="utf-8")
    (OUT / "results_fragment.tex").write_text(emit_tex(), encoding="utf-8")
    (OUT / "GAPS.md").write_text(emit_gaps(), encoding="utf-8")

    n_vals = sum(1 for lab in RESULTS for k in RESULTS[lab])
    n_null = sum(1 for lab in RESULTS for k in RESULTS[lab]
                 if RESULTS[lab][k]["value"] is None)
    print(f"  wrote {OUT/'results.json'}          {n_vals} quantities, {n_null} null")
    print(f"  wrote {OUT/'results_fragment.tex'}")
    print(f"  wrote {OUT/'GAPS.md'}               {len(GAPS)} gaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
