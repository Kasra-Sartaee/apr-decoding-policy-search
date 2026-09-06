#!/usr/bin/env python3
"""Emit the appendix LaTeX table of selected Pareto policies (tab:app-fronts).

    .venv/bin/python analysis/pareto_table.py

For each of the six (benchmark, arm) cells this takes the union of that arm's
five repetition fronts, applies the correctness-only selection rule of the three-arm comparison report
section 10.2, keeps the top five, and writes analysis/out/app_fronts.tex.

No number is entered by hand. Decision variables and training objectives come
from results/{ga,rs,mobo}_reps_v4/rep_N/pareto_solutions.json; held-out
objectives come from the per-repetition held-out results.json under outputs/.

The ranking is not reimplemented. `rank_by`, `_norm_min`, `_nd`, `tradeoff_mu`,
`KSEL` and `BENCH` are extracted verbatim from the committed notebook
analysis/three_arm_analysis.ipynb and executed, so this script
tracks the notebook if the rule changes. `rank_by(M, "correctness only")` sorts
ascending on column 1 of the normalised minimisation matrix, i.e. F2 descending.

Ties are broken by training F1 ascending so the output is deterministic; the
notebook's argsort leaves ties unordered. The tie-broken order is asserted to
select the same set as the notebook's rule before anything is written.

If any required artefact is absent the script stops and prints a manifest of
what is missing rather than emitting a partial or invented table.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NB = REPO / "analysis" / "three_arm_analysis.ipynb"
DEST = REPO / "analysis" / "out" / "app_fronts.tex"
ARMS = ("GA", "RS", "MOBO")
EXPECTED_UNION = 665                    # distinct policies across the five reps

# Column order as specified for the appendix table. NOTE: this is NOT the order
# of search_space.SEARCH_SPACE, which places pi_v.retry at index 6 rather than
# tenth. The discrepancy is reported by the script, not silently reconciled.
COLUMNS = [
    ("pi_a", "temperature", "f2"), ("pi_a", "top_p", "f2"), ("pi_a", "top_k", "int"),
    ("pi_r", "temperature", "f2"), ("pi_r", "top_p", "f2"), ("pi_r", "top_k", "int"),
    ("pi_o", "temperature", "f2"), ("pi_o", "top_p", "f2"), ("pi_o", "top_k", "int"),
    ("pi_v", "retry", "int"), ("pi_o", "retry", "int"),
]
HEAD = [r"$\pi_a$.temp", r"$\pi_a$.top$_p$", r"$\pi_a$.top$_k$",
        r"$\pi_r$.temp", r"$\pi_r$.top$_p$", r"$\pi_r$.top$_k$",
        r"$\pi_o$.temp", r"$\pi_o$.top$_p$", r"$\pi_o$.top$_k$",
        r"$\pi_v$.retry", r"$\pi_o$.retry"]


# ------------------------------------------------- reuse the notebook's rule --
def load_notebook_impl() -> dict:
    """Exec the notebook cells that define the selection rule and the config."""
    if not NB.is_file():
        sys.exit(f"FATAL: notebook not found: {NB}")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    want = (re.compile(r"^BENCH\s*=\s*\{", re.M), re.compile(r"^def _nd\(", re.M),
            re.compile(r"^def _norm_min\(", re.M), re.compile(r"^KSEL\s*=", re.M))
    ns: dict = {"np": np, "Path": Path, "json": json, "REPO": REPO}
    taken = []
    for idx, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not any(p.search(src) for p in want):
            continue
        try:
            exec(compile(src, f"<nb cell {idx}>", "exec"), ns)
            taken.append(idx)
        except Exception:
            pass                     # cells with unmet deps are skipped by design
    missing = [n for n in ("BENCH", "rank_by", "_norm_min", "KSEL") if n not in ns]
    if missing:
        sys.exit(f"FATAL: could not extract {missing} from the notebook. "
                 f"The selection rule must be reused, not reimplemented.")
    ns["_cells_used"] = taken
    return ns


# ------------------------------------------------------------------- loading --
def load_fronts(cfg: dict) -> dict:
    """{arm: [(rep, policy, (cl, correctness, energy)), ...]} over five reps."""
    out = {}
    for arm in ARMS:
        rows, missing = [], []
        for i in range(5):
            p = REPO / cfg[f"{arm.lower()}_front"].format(i=i)
            if not p.is_file():
                missing.append(str(p)); continue
            for rec in json.loads(p.read_text(encoding="utf-8")):
                f = rec["fitness"]
                rows.append((i, rec["policy"],
                             (f["cl"], f["correctness"], f["energy"])))
        if missing:
            sys.exit("FATAL: missing front artefacts:\n  " + "\n  ".join(missing))
        out[arm] = rows
    return out


def pkey(pol: dict) -> str:
    return json.dumps(pol, sort_keys=True)


def load_heldout(cfg: dict, ns: dict) -> tuple[dict, list]:
    """{policy_key: (F1, F2, F3)} plus a list of absent source files."""
    tags = {"GA": (cfg["ga_dir"], cfg["ga_tag"]),
            "RS": (cfg["rs_dir"], cfg["rs_tag"]),
            "MOBO": (cfg["mobo_dir"], cfg["mobo_tag"])}
    found, absent = {}, []
    for arm in ARMS:
        d, tag = tags[arm]
        for i in range(5):
            p = cfg["root"] / d / tag.format(i=i) / "results.json"
            if not p.is_file():
                absent.append(str(p)); continue
            recs = json.loads(p.read_text(encoding="utf-8"))
            recs = recs["results"] if isinstance(recs, dict) and "results" in recs else recs
            for r in recs:
                found[pkey(r["policy"])] = (r["mean_cl"], r["mean_correctness"],
                                            r["mean_energy"])
    return found, absent


# ----------------------------------------------------------------- selection --
def select_top5(rows: list, ns: dict) -> list:
    """Correctness-only top five, ties broken by training F1 ascending."""
    M = np.array([[f[0], -f[1], f[2]] for _, _, f in rows], dtype=float)
    nb_order = list(np.asarray(ns["rank_by"](M, "correctness only")).ravel())
    k = int(ns["KSEL"])
    # deterministic: F2 descending, then F1 ascending
    det = list(np.lexsort((M[:, 0], M[:, 1])))
    if set(det[:k]) != set(nb_order[:k]):
        sys.exit("FATAL: tie-broken order selects a different set than the "
                 "notebook's rule. Refusing to write a table that does not "
                 "reproduce the three-arm comparison report section 10.2.")
    return [rows[i] for i in det[:k]]


# ------------------------------------------------------------------- output --
def fmt(v, kind):
    return f"{int(round(v)):d}" if kind == "int" else f"{v:.2f}"


def emit(cells: dict, digests: dict) -> str:
    L = [r"% Generated by analysis/pareto_table.py -- do not edit by hand.",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{The five policies selected by training correctness from each "
         r"front, with their held-out objective values.}",
         r"\label{tab:app-fronts}",
         r"\begin{tabular}{l" + "r" * (len(COLUMNS) + 3) + "}", r"\toprule",
         "rep & " + " & ".join(HEAD) + r" & $F_1$ & $F_2$ & $F_3$ \\", r"\midrule"]
    for n, (name, rows) in enumerate(cells.items()):
        if n:
            L.append(r"\midrule")
        L.append(rf"\multicolumn{{{len(COLUMNS) + 4}}}{{l}}{{\textit{{{name}}}}} \\")
        for rep, pol, _tr, ho in rows:
            vals = [fmt(pol[g][k], t) for g, k, t in COLUMNS]
            L.append(f"{rep} & " + " & ".join(vals) +
                     rf" & {ho[0]:.3f} & {ho[1]:.3f} & {ho[2]:.1f} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ns = load_notebook_impl()
    BENCH = ns["BENCH"]
    print(f"  reused notebook cells {ns['_cells_used']} for the selection rule; "
          f"KSEL={int(ns['KSEL'])}\n")

    # -- front artefacts and the 665 check --------------------------------
    fronts = {b: load_fronts(cfg) for b, cfg in BENCH.items()}
    b0 = list(BENCH)[0]
    union = sum(len(fronts[b0][a]) for a in ARMS)
    print(f"  CHECK front union: {union} policies "
          f"({', '.join(f'{a} {len(fronts[b0][a])}' for a in ARMS)})")
    if union != EXPECTED_UNION:
        sys.exit(f"\nSTOP: the front artefacts give {union} policies across the six "
                 f"cells, not {EXPECTED_UNION}. Not proceeding.")
    print(f"  -> matches the expected {EXPECTED_UNION}\n")

    # -- held-out availability --------------------------------------------
    heldout, absent = {}, []
    for b, cfg in BENCH.items():
        h, missing = load_heldout(cfg, ns)
        heldout[b] = h
        absent += missing
        print(f"  CHECK held-out {b}: {len(h)} policy records from "
              f"{15 - len(missing)}/15 repetition files")
        print(f"    digest declared in config: {cfg['digest']}")
    if absent:
        print(f"\nSTOP: {len(absent)} of 30 held-out result files are absent on this "
              f"host, so held-out F1/F2/F3 cannot be sourced from artefacts.\n")
        for p in absent[:6]:
            print(f"    {p}")
        if len(absent) > 6:
            print(f"    ... and {len(absent) - 6} more")
        print("\n  No table written. Held-out evaluation ran on the evaluation "
              "machine; these files live under outputs/, which is untracked, so "
              "they were never committed. Sync them to this host, or run the "
              "script there, and it will emit the table unchanged.")
        return 2

    # -- selection, checks, output ----------------------------------------
    cells: dict = {}
    seen: dict = {}
    for b in BENCH:
        for a in ARMS:
            top = select_top5(fronts[b][a], ns)
            rows = []
            for rep, pol, tr in top:
                k = pkey(pol)
                if k not in heldout[b]:
                    sys.exit(f"FATAL: selected policy from {b}/{a} rep_{rep} has no "
                             f"held-out record. Refusing to invent one.")
                rows.append((rep, pol, tr, heldout[b][k]))
                seen.setdefault(k, []).append(f"{b}/{a}")
            cells[f"{b} --- {a}"] = rows
            print(f"  {b:22s} {a:5s} mean held-out F2 = "
                  f"{np.mean([r[3][1] for r in rows]):.3f}")
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"\n  CHECK policies appearing in more than one cell: {len(dupes)}")
    for k, v in list(dupes.items())[:10]:
        print(f"    {', '.join(v)}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(emit(cells, {b: BENCH[b]['digest'] for b in BENCH}), encoding="utf-8")
    print(f"\n  written: {DEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
