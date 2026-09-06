#!/usr/bin/env python3
"""Regenerate the search-cost table for dissertation section 5.4.

Reads only committed/on-disk run artefacts under outputs/ and writes
docs/reports/search_cost.md. One invocation regenerates every number.

    .venv/bin/python analysis/search_cost.py

The count is not a naive sum over the artefacts. Each (candidate, problem)
evaluation writes three files into
    <run>/verification/codeflaws/<problem>/verification_summary_<ns>_<seq>.json
  1. after the repair stage      -> per_stage {analyze, repair}
  2. after the optimise stage    -> per_stage {analyze, optimize, repair}
  3. a zero-call record for the Phase B measured re-execution (no LLM work)

Record 2 is cumulative over record 1 -- it repeats analyze+repair and adds
optimize. Summing every file therefore double-counts calls, tokens and energy
(measured: 57,039 calls vs a true 30,303 for one problem directory).

The counters reset to zero at each Phase B record, so evaluations are recovered
by segmenting a problem directory's files in timestamp order at every drop in
the cumulative call count, then taking the maximum within each segment.

The filename's first number is a nanosecond epoch timestamp; it is the only
per-record time reference and is what attributes a record to a repetition.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "ga" / "codeflaws" / "qwen2.5-coder_1.5b"
DEST = REPO / "docs" / "reports" / "search_cost.md"

ARMS = {                       # label -> run directory
    "GA (NSGA-II)":   "pop50_gen15_n50_v4",
    "RS (uniform)":   "rs_pop50_gen15_n50_v4",
    "MOBO (qLogNEHVI)": "mobo_pop50_gen15_n50_v4",
}
NREPS, NEVALS, NPROBLEMS = 5, 750, 50
CAP = 2048          # configured max generation length
FNAME = re.compile(r"verification_summary_(\d+)_(\d+)\.json$")


# ---------------------------------------------------------------- telemetry --
def scan_problem_dir(d: str) -> dict:
    """Aggregate one problem directory into per-evaluation totals.

    Returns sums over evaluations, plus the (ns, calls) of each evaluation so
    the caller can attribute it to a repetition window.
    """
    files = []
    for name in os.listdir(d):
        m = FNAME.search(name)
        if m:
            files.append((int(m.group(1)), os.path.join(d, name)))
    files.sort()

    evals, cur, prev = [], None, -1
    for ns, path in files:
        try:
            j = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        t = j.get("telemetry") or {}
        rc = j.get("retry_counts") or {}
        calls = t.get("total_llm_calls", 0) or 0
        rec = {
            "ns": ns, "calls": calls,
            "prompt": t.get("total_prompt_tokens", 0) or 0,
            "completion": t.get("total_completion_tokens", 0) or 0,
            "cpu_mj": t.get("total_cpu_energy_mj"),
            "gpu_mj": t.get("total_gpu_energy_mj"),
            "llm_s": t.get("total_wall_time_s_llm", 0.0) or 0.0,
            "retry_repair": rc.get("repair", 0) or 0,
            "retry_optimize": rc.get("optimize", 0) or 0,
            # exact single-call completion lengths (a stage with n_calls == 1)
            "single": [s.get("completion_tokens", 0) or 0
                       for s in (t.get("per_stage") or {}).values()
                       if s.get("n_calls") == 1],
        }
        if calls < prev and cur is not None:      # counters reset -> new evaluation
            evals.append(cur); cur = None
        if cur is None or calls >= cur["calls"]:  # keep the cumulative maximum
            cur = rec
        prev = calls
    if cur is not None:
        evals.append(cur)
    evals = [e for e in evals if e["calls"] > 0]  # drop pure Phase-B segments

    agg = {
        "n_evals": len(evals),
        "calls": sum(e["calls"] for e in evals),
        "prompt": sum(e["prompt"] for e in evals),
        "completion": sum(e["completion"] for e in evals),
        "llm_s": sum(e["llm_s"] for e in evals),
        "retry_repair": sum(e["retry_repair"] for e in evals),
        "retry_optimize": sum(e["retry_optimize"] for e in evals),
        "cpu_mj": sum(e["cpu_mj"] for e in evals if e["cpu_mj"] is not None),
        "cpu_n": sum(1 for e in evals if e["cpu_mj"] is not None),
        "gpu_mj": sum(e["gpu_mj"] for e in evals if e["gpu_mj"] is not None),
        "gpu_n": sum(1 for e in evals if e["gpu_mj"] is not None),
        # single-call observations come from the final cumulative record of each
        # evaluation only; earlier records repeat the same per_stage entries.
        "single_max": max([s for e in evals for s in e["single"]] or [0]),
        "single_n": sum(len(e["single"]) for e in evals),
        "single_at_cap": sum(1 for e in evals for s in e["single"] if s >= CAP),
        "stamps": [(e["ns"], e["calls"], e["prompt"], e["completion"],
                    e["cpu_mj"], e["gpu_mj"], e["retry_repair"], e["retry_optimize"])
                   for e in evals],
    }
    return agg


def scan_arm(run_dir: Path) -> dict:
    vdir = run_dir / "verification" / "codeflaws"
    if not vdir.is_dir():
        return {"missing": True}
    dirs = [str(p) for p in sorted(vdir.iterdir()) if p.is_dir()]
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        parts = list(ex.map(scan_problem_dir, dirs, chunksize=1))
    total: defaultdict = defaultdict(float)
    stamps = []
    total["single_max"] = 0
    for p in parts:
        for k, v in p.items():
            if k == "stamps":
                stamps.extend(v)
            elif k == "single_max":
                total["single_max"] = max(total["single_max"], v)
            else:
                total[k] += v
    total["stamps"] = stamps
    total["n_problem_dirs"] = len(dirs)
    return dict(total)


# ------------------------------------------------------------- wall - clock --
def rep_windows(run_dir: Path) -> dict[int, tuple[datetime, datetime, int, float]]:
    """Per-repetition (first, last, n, largest_gap_s) from evaluations/*.json.

    Only arms that write per-candidate records have this; the GA does not.
    """
    out = {}
    for i in range(NREPS):
        ed = run_dir / f"rep_{i}" / "evaluations"
        if not ed.is_dir():
            continue
        ts = []
        for f in ed.glob("*.json"):
            try:
                v = json.load(open(f)).get("evaluated_at")
            except (json.JSONDecodeError, OSError):
                continue
            if v:
                ts.append(datetime.fromisoformat(v))
        if ts:
            ts.sort()
            gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
            out[i] = (ts[0], ts[-1], len(ts), max(gaps) if gaps else 0.0)
    return out


def manifest_timing(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if not p.is_file():
        return {}
    d = json.load(open(p))
    return {k: d.get(k) for k in
            ("elapsed_seconds", "started_at", "finished_at", "reps", "n_workers")}


def resume_events(run_dir: Path) -> list[tuple[int, str, str]]:
    ev = []
    for i in range(NREPS):
        p = run_dir / f"rep_{i}" / "protocol.json"
        if not p.is_file():
            continue
        for r in (json.load(open(p)).get("resumed_under") or []):
            ev.append((i, r.get("written_at", "?"), r.get("git_commit", "?")[:8]))
    return ev


def host_spec() -> dict:
    spec = {}
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True,
                                   timeout=20).stdout.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("Model name", "Socket(s)", "Core(s) per socket",
                             "CPU(s)", "Thread(s) per core", "CPU max MHz"):
                spec[k.strip()] = v.strip()
    except Exception:
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                spec["RAM"] = f"{int(line.split()[1]) / 1048576:.1f} GiB"
                break
    except OSError:
        pass
    try:
        spec["current boot"] = subprocess.run(["uptime", "-s"], capture_output=True,
                                              text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return spec


# -------------------------------------------------------------------- main --
def hours(s):  return s / 3600.0
def fmt(v, n=2): return "NOT RECOVERABLE" if v is None else f"{v:,.{n}f}"


def main() -> int:
    if not OUT.is_dir():
        print(f"run root not found: {OUT}", file=sys.stderr)
        return 1
    arms = {}
    for label, run in ARMS.items():
        rd = OUT / run
        print(f"[scan] {label} ...", file=sys.stderr, flush=True)
        arms[label] = {
            "run": run, "dir": rd,
            "tel": scan_arm(rd),
            "win": rep_windows(rd),
            "man": manifest_timing(rd),
            "res": resume_events(rd),
        }

    # Fail closed. The per-candidate verification records live under outputs/,
    # which is the separate data release rather than part of this repository.
    # Without them every count is zero, and writing that over a valid report
    # replaces real numbers with plausible-looking nulls.
    empty = [lab for lab, a in arms.items() if not a["tel"].get("problem_dirs")]
    if empty:
        print(
            "no evaluation records found for: " + ", ".join(empty) + "\n"
            f"expected them under {OUT}\n"
            "unpack the data release before regenerating; refusing to overwrite "
            f"{DEST.name} with zero-record output.",
            file=sys.stderr,
        )
        return 1

    write_report(arms)
    print_summary(arms)
    return 0


def print_summary(arms):
    print("\n" + "=" * 78)
    print("SEARCH COST SUMMARY".center(78))
    print("=" * 78)
    hdr = f"{'arm':<18}{'reps':>5}{'wall h':>10}{'retry/eval':>12}{'gen calls':>12}{'CPU kJ':>12}"
    print(hdr); print("-" * 78)
    for label, a in arms.items():
        t, w = a["tel"], a["win"]
        wall = (sum((b - x).total_seconds() for x, b, *_ in w.values()) if w
                else a["man"].get("elapsed_seconds"))
        n = int(t.get("n_evals", 0))
        retry = (t["retry_repair"] + t["retry_optimize"]) / n if n else None
        print(f"{label:<18}{NREPS:>5}{fmt(hours(wall) if wall else None):>10}"
              f"{fmt(retry, 3):>12}{int(t.get('calls', 0)):>12,}"
              f"{t.get('cpu_mj', 0) / 1e6:>12,.1f}")
    print("-" * 78)
    print(f"report written to {DEST.relative_to(REPO)}")


def write_report(arms):
    L = []
    A = L.append
    now = datetime.now(UTC).isoformat(timespec="seconds")
    A("# Search Cost — NSGA-II, Random Search and MOBO\n")
    A(f"Generated by `analysis/search_cost.py` at {now}.")
    A("Every figure below was recomputed from run artefacts by that script at the "
      "time of generation; none is entered by hand. Quantities not recoverable "
      "from the artefacts are marked `NOT RECOVERABLE` with the instrumentation "
      "that would be required.")
    A("> **Status: preserved derived result.** The per-candidate verification "
      "telemetry this report is computed from was written under `outputs/` during "
      "the searches and was not retained. The figures below are the measurements "
      "taken while it was present; they cannot be recomputed from the released "
      "artefacts. See `PROVENANCE.md`.")

    # ---- 1 provenance
    A("## 1. Provenance\n")
    A("| arm | run directory | problem dirs read | evaluation records recovered |")
    A("| --- | --- | --- | --- |")
    for label, a in arms.items():
        t = a["tel"]
        A(f"| {label} | `outputs/ga/codeflaws/qwen2.5-coder_1.5b/{a['run']}/` | "
          f"{int(t.get('n_problem_dirs', 0))} | {int(t.get('n_evals', 0)):,} of "
          f"{NREPS * NEVALS * NPROBLEMS:,} expected |")
    A("")
    A("Telemetry is read from "
      "`<run>/verification/codeflaws/<problem>/verification_summary_<ns>_<seq>.json`. "
      "Each evaluation writes three such records and the second is cumulative over "
      "the first, so per-evaluation totals are taken as the maximum within a segment "
      "rather than a sum over files (see the module docstring). All figures below are "
      "**sums over recovered evaluations** unless stated as a mean or a single reading.\n")
    A("The 50 problem directories are the training set; held-out evaluations write "
      "elsewhere and are excluded by construction.\n")
    dr = []
    for a in arms.values():
        st = [s[0] for s in a["tel"].get("stamps", [])]
        if st:
            dr.append((min(st), max(st)))
    if dr:
        lo = datetime.fromtimestamp(min(x for x, _ in dr) / 1e9)
        hi = datetime.fromtimestamp(max(y for _, y in dr) / 1e9)
        A(f"**Date range of all records read:** {lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} "
          "(local), from the nanosecond timestamp in each filename.\n")

    # ---- 2 main table
    A("## 2. Cost per strategy\n")
    A("| arm | reps | total search wall-clock (h) | mean wall-clock / rep (h) | "
      "mean combined retry budget | generation calls | inference energy CPU (kJ) | "
      "inference energy GPU (kJ) |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for label, a in arms.items():
        t, w, m = a["tel"], a["win"], a["man"]
        n = int(t.get("n_evals", 0))
        if w:
            spans = [(b - x).total_seconds() for x, b, *_ in w.values()]
            tot, mean = sum(spans), sum(spans) / len(spans)
            tot_s, mean_s = f"{hours(tot):,.2f}", f"{hours(mean):,.2f}"
        elif m.get("elapsed_seconds"):
            tot_s = f"{hours(m['elapsed_seconds']):,.2f}"
            mean_s = "NOT RECOVERABLE"
        else:
            tot_s = mean_s = "NOT RECOVERABLE"
        retry = (t["retry_repair"] + t["retry_optimize"]) / n if n else None
        gpu_kj = (f"{t['gpu_mj'] / 1e6:,.1f} (partial, {int(t['gpu_n']):,}/{n:,} records)"
                  if t.get("gpu_n") else "NOT RECOVERABLE")
        A(f"| {label} | {NREPS} | {tot_s} | {mean_s} | {fmt(retry, 3)} | "
          f"{int(t.get('calls', 0)):,} | {t.get('cpu_mj', 0) / 1e6:,.1f} | {gpu_kj} |")
    A("")
    A("Wall-clock provenance differs by arm and the two columns are not interchangeable; "
      "see §3 and §5.\n")

    # ---- 3 per-repetition wall-clock
    A("## 3. Wall-clock per repetition\n")
    for label, a in arms.items():
        w, m = a["win"], a["man"]
        A(f"**{label}**\n")
        if not w:
            A(f"Per-repetition wall-clock is `NOT RECOVERABLE`. This arm writes no "
              f"per-candidate evaluation record, so no timestamp exists at repetition "
              f"granularity. The run-level manifest gives a single reading of "
              f"{fmt(hours(m['elapsed_seconds']) if m.get('elapsed_seconds') else None)} h "
              f"covering all {NREPS} repetitions "
              f"(`results/nsga2/manifest.json`, `elapsed_seconds`), started "
              f"{m.get('started_at', '?')}, finished {m.get('finished_at', '?')}.\n")
            A("Recovering the per-repetition split would require the search driver to "
              "record a per-candidate timestamp, as the other two arms already do.\n")
            continue
        A("| rep | first evaluation | last evaluation | span (h) | evaluations | "
          "largest single gap (h) |")
        A("| --- | --- | --- | --- | --- | --- |")
        for i in sorted(w):
            s, e, n, g = w[i]
            A(f"| rep_{i} | {s:%Y-%m-%d %H:%M} | {e:%Y-%m-%d %H:%M} | "
              f"{hours((e - s).total_seconds()):,.2f} | {n} | {hours(g):,.2f} |")
        spans = [(b - x).total_seconds() for x, b, *_ in w.values()]
        mean = sum(spans) / len(spans)
        sd = (sum((v - mean) ** 2 for v in spans) / (len(spans) - 1)) ** 0.5 if len(spans) > 1 else 0.0
        A(f"\nMean {hours(mean):,.2f} h, sd {hours(sd):,.2f} h, "
          f"total {hours(sum(spans)):,.2f} h. Source: `rep_N/evaluations/*.json`, field "
          "`evaluated_at`; span is last minus first, so it includes any idle time within "
          "a repetition. The largest-gap column exposes that: a large value means the "
          "span overstates active compute.\n")

    # ---- 4 retries
    A("## 4. Retry budget consumed\n")
    A("Per **(candidate, problem) evaluation** — not per candidate. A candidate is "
      f"scored over {NPROBLEMS} problems, so the per-candidate figure is {NPROBLEMS}x these.\n")
    A("| arm | n (evaluations) | mean repair retries | mean optimize retries | "
      "mean combined | repair share |")
    A("| --- | --- | --- | --- | --- | --- |")
    for label, a in arms.items():
        t = a["tel"]; n = int(t.get("n_evals", 0))
        if not n:
            A(f"| {label} | 0 | NOT RECOVERABLE | NOT RECOVERABLE | NOT RECOVERABLE | — |")
            continue
        r, o = t["retry_repair"] / n, t["retry_optimize"] / n
        share = f"{100 * r / (r + o):.1f}%" if (r + o) else "—"
        A(f"| {label} | {n:,} | {r:.3f} | {o:.3f} | {r + o:.3f} | {share} |")
    A("\nSource: `retry_counts.repair` and `retry_counts.optimize` in each evaluation's "
      "final cumulative verification record. Sums over evaluations, divided by n.\n")

    # ---- 5 calls, tokens
    A("## 5. Generation calls and tokens\n")
    A("| arm | generation calls | mean calls / evaluation | prompt tokens | "
      "completion tokens | total tokens | LLM wall-time (h) |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for label, a in arms.items():
        t = a["tel"]; n = int(t.get("n_evals", 0))
        A(f"| {label} | {int(t.get('calls', 0)):,} | "
          f"{t.get('calls', 0) / n if n else 0:.3f} | {int(t.get('prompt', 0)):,} | "
          f"{int(t.get('completion', 0)):,} | "
          f"{int(t.get('prompt', 0) + t.get('completion', 0)):,} | "
          f"{hours(t.get('llm_s', 0)):,.2f} |")
    A("\n`mean calls / evaluation` is per (candidate, problem). LLM wall-time is the sum of "
      "`total_wall_time_s_llm` over evaluations; because Phase A runs multiple workers in "
      "parallel it exceeds the elapsed wall-clock of §3 and is a compute measure, not an "
      "elapsed one.\n")
    A("**Maximum single-call completion length.** Only stages with `n_calls == 1` give an "
      "exact single-call length; multi-call stages record only a per-stage sum.\n")
    A(f"| arm | max exact single-call completion tokens | exact single-call observations | "
      f"at the {CAP}-token cap | share at cap |")
    A("| --- | --- | --- | --- | --- |")
    for label, a in arms.items():
        t_ = a["tel"]
        n_s, at = int(t_.get("single_n", 0)), int(t_.get("single_at_cap", 0))
        A(f"| {label} | {int(t_.get('single_max', 0)):,} | {n_s:,} | {at:,} | "
          f"{100 * at / n_s if n_s else 0:.2f}% |")
    A(f"\nThe observed maximum equals the configured cap of {CAP} tokens in every arm, and "
      f"the counts above are calls that reached it exactly. The cap was therefore reached "
      f"during these runs; it was not left unattained.\n")
    A("The true global maximum over every individual call is `NOT RECOVERABLE`, and so is "
      "the total number of calls that hit the cap: telemetry aggregates per stage, so a "
      "stage that made several calls reports only their sum and any capped call inside it "
      "is invisible. The figures above are therefore a **lower bound** on cap-reaching "
      "calls, counted only over stages that made exactly one call. Per-call token logging "
      "would be required for an exact count.\n")

    # ---- 6 energy
    A("## 6. Inference energy\n")
    A("| arm | CPU (RAPL) kJ | records with CPU | GPU (NVML) kJ | records with GPU | GPU coverage |")
    A("| --- | --- | --- | --- | --- | --- |")
    for label, a in arms.items():
        t = a["tel"]; n = int(t.get("n_evals", 0))
        cov = f"{100 * t.get('gpu_n', 0) / n:.1f}%" if n else "—"
        gpu = f"{t.get('gpu_mj', 0) / 1e6:,.1f}" if t.get("gpu_n") else "NOT RECOVERABLE"
        A(f"| {label} | {t.get('cpu_mj', 0) / 1e6:,.1f} | {int(t.get('cpu_n', 0)):,} | "
          f"{gpu} | {int(t.get('gpu_n', 0)):,} | {cov} |")
    A("\nCPU is the RAPL package counter delta accumulated around generation calls; GPU is "
      "the NVML delta. **The GPU figure is a partial sum over only those evaluations that "
      "carry a value and is not a total for the arm.** It must not be scaled up to full "
      "coverage, and CPU and GPU columns cover different evaluation subsets, so they must "
      "not be added.\n")
    A("This is inference energy and is distinct from objective F3, which is program-execution "
      "energy measured in Phase B. Neither is substituted for the other anywhere here.\n")

    # ---- 7 host
    A("## 7. Host and epochs\n")
    spec = host_spec()
    A("| property | value |")
    A("| --- | --- |")
    for k, v in spec.items():
        A(f"| {k} | {v} |")
    A("\nRead from `lscpu`, `/proc/meminfo` and `uptime -s` on the search host at "
      "generation time.\n")
    A("**Resume events recorded in `rep_N/protocol.json` (`resumed_under`):**\n")
    any_ev = False
    for label, a in arms.items():
        for i, when, commit in a["res"]:
            A(f"- {label} rep_{i}: resumed {when}, commit `{commit}`")
            any_ev = True
    if not any_ev:
        A("- none recorded")
    A("")
    A("**Host restart times are `NOT RECOVERABLE` from committed artefacts.** No artefact "
      "records a reboot; `uptime -s` gives only the current boot. Restarts are visible "
      "only indirectly, as resume events above and as shifts in measured F3, and the "
      "repetition indices they fall between are documented in the arm READMEs under "
      "`results/`. Recording a boot id per evaluation would make epoch boundaries explicit.\n")

    # ---- 8 gaps
    A("## 8. GAPS\n")
    A("| quantity | status | why | what would be needed |")
    A("| --- | --- | --- | --- |")
    A("| GA per-repetition wall-clock | NOT RECOVERABLE | the GA writes no per-candidate "
      "record; only a run-level `elapsed_seconds` covering all 5 repetitions exists | a "
      "per-candidate timestamp in the GA driver, as RS and MOBO already write |")
    A("| RS / MOBO `elapsed_seconds` | ABSENT from manifest | the field is not written by "
      "those drivers; their manifests also lack `finished_at` | wall-clock in §2/§3 for "
      "these arms is derived from evaluation timestamps instead, which includes idle time |")
    A("| GPU inference energy, full coverage | PARTIAL | NVML returned a value for only "
      "some evaluations; the rest record `null` | NVML available for the whole run; the "
      "partial sums in §6 are not totals |")
    A("| Max completion tokens for every individual call | NOT RECOVERABLE | telemetry "
      "aggregates per stage, so multi-call stages give only a sum | per-call token logging |")
    A("| Host restart times | NOT RECOVERABLE | no artefact records reboots | a boot id "
      "recorded per evaluation |")
    A("| Exact evaluation count | PARTIAL | recovered counts fall slightly below "
      f"{NREPS * NEVALS * NPROBLEMS:,}; evaluations that failed before writing a "
      "cumulative record leave none to recover | a per-evaluation record written "
      "unconditionally, including on failure |")
    A("")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
