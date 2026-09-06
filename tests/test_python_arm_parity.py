"""Parity tests: the Python arm must be measured like the C arm.

The cross-language comparison ("does the optimal decoding policy transfer from
C to Python?") is only meaningful if both arms measure F1/F2/F3 under the same
protocol. These pin the invariants that make that true, each of which was
violated at some point while the Python benchmark was being added.

Fabricated sources and tests only — never the LLM, never a real GA run.

    .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import requires_corpus

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentic_workflow import (  # noqa: E402
    compute_complexity,
    optimized_preserves_correctness,
    set_energy_backend,
)
from benchmarks.base import (  # noqa: E402
    EnergyMeasurement,
    PythonStdinStdoutRunner,
    normalise_output,
)
from benchmarks.codeflaws.adapter import _official_normalise  # noqa: E402
from energy import resolve_backend  # noqa: E402

set_energy_backend(resolve_backend("runtime_proxy"))


# ---------------------------------------------------------------- fixtures
def _make_problem(tmp_path: Path, n_tests: int = 5, body: str | None = None) -> Path:
    """A trivial problem: double the input integer."""
    for i in range(1, n_tests + 1):
        (tmp_path / f"input_{i}.txt").write_text(f"{i}\n")
        (tmp_path / f"expected_{i}.txt").write_text(f"{i * 2}\n")
    src = tmp_path / "prog.py"
    src.write_text(body if body is not None else "print(int(input()) * 2)\n")
    return src


def _runner(tmp_path: Path, **kw) -> PythonStdinStdoutRunner:
    return PythonStdinStdoutRunner(tmp_path, "input_*.txt", "expected_*.txt", **kw)


# ------------------------------------------------- two-phase interface shape
def test_runner_implements_both_phase_methods():
    """Implementing diagnostic() without measure_energy() is worse than
    implementing neither: run_pipeline_phase_a duck-types on `diagnostic`
    alone, but run_pipeline_phase_b calls `measure_energy` unconditionally.
    The AttributeError is then swallowed into (inf, 0.0, inf), silently
    marking every problem infeasible."""
    for name in ("diagnostic", "measure_energy"):
        assert callable(getattr(PythonStdinStdoutRunner, name, None)), name


def test_diagnostic_measures_no_energy(tmp_path):
    """Phase A runs concurrently across worker threads. If it opened an energy
    window, every worker would contaminate every other worker's reading."""
    src = _make_problem(tmp_path)
    diag = _runner(tmp_path).diagnostic(src)
    assert diag.energy_mj is None
    assert diag.energy_backend == ""


def test_passing_test_ids_count_matches_tests_passed(tmp_path):
    """optimized_preserves_correctness() fails CLOSED when these disagree, so
    a runner that omits ids makes the optimise stage rejected on every problem
    — silently turning 4 of the 11 GA search dimensions (pi_o) inert."""
    src = _make_problem(tmp_path, n_tests=5)
    diag = _runner(tmp_path).diagnostic(src)
    assert diag.tests_passed == 5
    assert len(diag.passing_test_ids) == diag.tests_passed


# ------------------------------------------------------------ energy window
def test_invalid_measurement_is_none_never_zero(tmp_path):
    """F3 is MINIMISED, so 0.0 is an unbeatable score. An unmeasured problem
    must be excluded by the isfinite feasibility filter, not crowned."""
    src = _make_problem(tmp_path)
    m = _runner(tmp_path).measure_energy(src, [])
    assert m.mean_energy_mj is None
    assert m.mean_energy_mj != 0.0


def test_energy_window_is_whole_batch_not_per_test(tmp_path, monkeypatch):
    """One window around the whole batch, not one per test. Per-test windows
    sit at the counter's resolution floor (the measurement-protocol report section 3.2)."""
    import benchmarks.base as base

    src = _make_problem(tmp_path, n_tests=4)
    opens = {"n": 0}
    real = base._energy_window

    def counting_window():
        opens["n"] += 1
        return real()

    monkeypatch.setattr(base, "_energy_window", counting_window)
    diag = _runner(tmp_path).diagnostic(src)
    _runner(tmp_path).measure_energy(src, diag.passing_test_ids)
    assert opens["n"] == 1, f"expected 1 whole-batch window, got {opens['n']}"


def test_small_batches_are_repeated_to_clear_the_floor(tmp_path):
    """A candidate passing few tests must not be measured at the resolution
    floor, or F3's measurability would correlate with F2 and systematically
    drop low-correctness candidates out of the aggregates."""
    src = _make_problem(tmp_path, n_tests=1)
    r = _runner(tmp_path, min_window_execs=6)
    m = r.measure_energy(src, ["1"])
    assert m.mean_energy_mj is not None and m.mean_energy_mj > 0


# ---------------------------------------------- correctness vs measurability
def test_correctness_is_independent_of_energy_measurability(tmp_path, monkeypatch):
    """A correct program must not lose correctness for being fast. An earlier
    C version checked the energy sample first and `continue`d, demoting F2 for
    tests that ran correctly but too quickly to resolve."""
    from contextlib import contextmanager

    import benchmarks.base as base

    @contextmanager
    def dead_window():
        holder = {"energy_mj": None, "backend": "none", "subtype": ""}
        yield holder

    src = _make_problem(tmp_path, n_tests=3)
    diag = _runner(tmp_path).diagnostic(src)
    monkeypatch.setattr(base, "_energy_window", dead_window)
    m = _runner(tmp_path).measure_energy(src, diag.passing_test_ids)

    assert m.mean_energy_mj is None, "energy genuinely unmeasurable here"
    assert m.mismatched_ids == [], "but correctness must be untouched by that"
    assert sorted(m.matched_ids) == sorted(diag.passing_test_ids)


def test_reproduction_failure_is_recorded_with_a_reason(tmp_path):
    """Phase B revises F2 downward from these; the reason must be a
    correctness fact, not an energy artifact."""
    src = _make_problem(tmp_path, n_tests=3)
    diag = _runner(tmp_path).diagnostic(src)
    broken = tmp_path / "broken.py"
    broken.write_text("n=int(input())\nprint(n*2 if n!=2 else 999)\n")
    m = _runner(tmp_path).measure_energy(broken, diag.passing_test_ids)
    assert "2" in m.mismatched_ids
    assert m.mismatch_reasons["2"] == "output_mismatch"
    assert m.mean_energy_mj is not None, "a mismatch must not invalidate energy"


# --------------------------------------------------------------- normalisation
@pytest.mark.parametrize("raw,expected", [
    ("1 0\n", ["1 0"]),
    ("1 0", ["1 0"]),
    ("  1 0\n", ["1 0"]),          # leading whitespace stripped
    ("1 0   \n", ["1 0"]),         # trailing whitespace ignored
    ("\n1 0\n\n", ["1 0"]),        # blank lines dropped
    ("a\n\nb\n", ["a", "b"]),
])
def test_normalisation_matches_the_c_arm(raw, expected):
    """Both arms must apply the SAME rule. Plain .strip() equality is stricter
    than the benchmark's own checker and biases F2 downward on whichever arm
    uses it."""
    assert normalise_output(raw) == expected
    assert _official_normalise(raw) == expected


# ------------------------------------------------------------- optimiser gate
def test_optimiser_gate_accepts_identical_and_rejects_changes(tmp_path):
    """Exact-set equality in both directions: a lost test is damage, and a
    GAINED test is out of contract (it credits a repair gain to optimise)."""
    src = _make_problem(tmp_path, n_tests=3)
    worse = tmp_path / "worse.py"
    worse.write_text("n=int(input())\nprint(n*2 if n!=2 else 999)\n")

    r = _runner(tmp_path)
    good, bad = r.diagnostic(src), r.diagnostic(worse)

    assert optimized_preserves_correctness(good, good)[0] is True
    assert optimized_preserves_correctness(good, bad)[0] is False
    assert optimized_preserves_correctness(bad, good)[0] is False


# ----------------------------------------------------------------- F1 metric
def test_complexity_is_layout_independent(tmp_path):
    """lizard is function-granularity and returns 0 for a flat script — 84% of
    RunBugRun. These two programs are the same program; F1 must agree."""
    flat = tmp_path / "flat.py"
    flat.write_text(
        "n=int(input())\ntotal=0\nfor i in range(n):\n"
        "    if i%2==0:\n        total+=i\n    elif i%3==0:\n        total-=i\n"
        "print(total)\n"
    )
    wrapped = tmp_path / "wrapped.py"
    wrapped.write_text(
        "def solve(n):\n    total=0\n    for i in range(n):\n"
        "        if i%2==0:\n            total+=i\n        elif i%3==0:\n"
        "            total-=i\n    return total\n"
        "n=int(input())\nprint(solve(n))\n"
    )
    assert compute_complexity(flat) == compute_complexity(wrapped)
    assert compute_complexity(flat) > 0


def test_unmeasurable_complexity_is_the_zero_sentinel(tmp_path):
    """0 means "not measured", not "trivially simple" — the minimum real
    whole-program complexity is 1."""
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n    pass\n")
    assert compute_complexity(broken) == 0


# ------------------------------------------------------------- shared struct
def test_energy_measurement_struct_is_shared():
    """Phase B consumes this by attribute from whichever benchmark produced
    it; a per-adapter copy would be free to drift out of the contract."""
    from benchmarks.codeflaws.adapter import EnergyMeasurement as CodeflawsEM
    assert CodeflawsEM is EnergyMeasurement
    for f in ("mean_energy_mj", "backend", "backend_subtype",
              "matched_ids", "mismatched_ids", "mismatch_reasons"):
        assert f in EnergyMeasurement.__dataclass_fields__, f


# -------------------------------------------------------------- provenance
def test_contract_hashes_are_language_scoped():
    """Adding a Python contract must not change a C run's recorded treatment:
    contract_sha256 is a treatment key in the resume comparison, so globbing
    every *.md made an untouched Codeflaws resume report a spurious change."""
    from provenance import contract_hashes
    c, p = contract_hashes("C99"), contract_hashes("Python3")
    assert len(c) == 3 and len(p) == 3
    assert not (set(c) & set(p)), "the two arms must not share contract files"


# =====================================================================
# Pre-flight audit regressions (28 July). Each pins a defect that would
# have silently corrupted the first Python GA run.
# =====================================================================

def test_syntax_error_reports_not_compiled(tmp_path):
    """Python has no build step, so an earlier version returned compiled=True
    unconditionally — which disabled pi_o.retry, one of the 11 GA dimensions:
    the optimiser retry loop is `while not verify_opt.compiled`."""
    src = _make_problem(tmp_path, n_tests=2)
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n    pass\n")

    r = _runner(tmp_path)
    good, bad = r.diagnostic(src), r.diagnostic(broken)
    assert good.compiled is True
    assert bad.compiled is False, "optimiser retry loop can never fire otherwise"
    assert "SyntaxError" in bad.compile_stderr
    assert bad.tests_total == 0 and bad.tests_passed == 0
    # and the optimiser gate must reject it on the compile check
    assert optimized_preserves_correctness(good, bad) == (
        False, "optimised code did not compile"
    )


def test_phase_b_aborts_and_invalidates_rather_than_inflating(tmp_path):
    """A program that passes Phase A then stalls must not charge its timeout
    duration to F3. Under runtime_proxy F3 IS wall clock, so reporting a
    finite-but-huge value would enter the aggregate and distort the objective;
    it must be invalid (None -> inf -> excluded by the isfinite filter)."""
    n = 8
    for i in range(1, n + 1):
        (tmp_path / f"input_{i}.txt").write_text("1\n")
        (tmp_path / f"expected_{i}.txt").write_text("ok\n")
    src = tmp_path / "p.py"
    src.write_text(
        "import os, time\n"
        "if os.path.exists(os.path.join(os.path.dirname(__file__), 'STALL')):\n"
        "    time.sleep(30)\n"
        "print('ok')\n"
    )
    r = _runner(tmp_path, run_timeout=1, max_consecutive_timeouts=3)
    diag = r.diagnostic(src)
    assert diag.tests_passed == n

    (tmp_path / "STALL").write_text("x")      # stall only in Phase B
    m = r.measure_energy(src, diag.passing_test_ids)

    assert m.mean_energy_mj is None, "aborted batch must be INVALID, not inflated"
    assert len(m.mismatched_ids) == n, "every unreproduced id must be recorded"
    assert "aborted" in set(m.mismatch_reasons.values())


@pytest.mark.integration
def test_heldout_split_is_group_disjoint():
    """Disjoint record ids do NOT imply an independent split when several
    records derive from one upstream problem sharing a test suite."""
    requires_corpus('runbugrun_py')
    sys.path.insert(0, str(REPO / "scripts"))
    import numpy as np
    from evaluate_heldout import group_key, select_heldout

    from benchmarks import get_benchmark

    probs = list(get_benchmark("runbugrun_py").problems())
    rng = np.random.default_rng(42)
    idx = rng.choice(len(probs), size=15, replace=False)
    train = [probs[i] for i in sorted(idx.tolist())]
    train_ids = {p.problem_id for p in train}

    heldout = select_heldout(probs, train_ids, 100, 0)
    train_groups = {group_key(p) for p in train}
    heldout_groups = {group_key(p) for p in heldout}

    assert not (train_groups & heldout_groups), "upstream problem leaked across split"
    assert not ({p.problem_id for p in heldout} & train_ids)


@pytest.mark.integration
def test_codeflaws_split_selection_is_unchanged():
    """The group fix is Python-only by decision: Codeflaws' held-out evaluation
    is already complete, and grouping it would change its selection to remove a
    ~1% effect. Its selection must therefore be byte-identical."""
    requires_corpus('codeflaws')
    sys.path.insert(0, str(REPO / "scripts"))
    import numpy as np
    from evaluate_heldout import select_heldout

    from benchmarks import get_benchmark

    probs = list(get_benchmark("codeflaws").problems())
    rng = np.random.default_rng(42)
    idx = rng.choice(len(probs), size=15, replace=False)
    train_ids = {probs[i].problem_id for i in sorted(idx.tolist())}

    new = [p.problem_id for p in select_heldout(probs, train_ids, 50, 0)]

    # the pre-fix, record-level selection
    cands = [p for p in probs if p.problem_id not in train_ids]
    r2 = np.random.default_rng(0)
    oidx = r2.choice(len(cands), size=50, replace=False)
    old = [cands[i].problem_id for i in sorted(oidx.tolist())]

    assert new == old, "Codeflaws held-out selection must not change"


def test_no_hardcoded_credentials_in_scripts():
    """A live API key sat in four scripts, entering every clone and diff."""
    import re
    pat = re.compile(r"""["'](rc_[A-Za-z0-9]{32,})["']""")
    offenders = []
    for path in (REPO / "scripts").glob("*.py"):
        m = pat.search(path.read_text(encoding="utf-8", errors="replace"))
        if m:
            offenders.append(f"{path.name}:{m.group(1)[:10]}...")
    assert not offenders, f"hardcoded credential(s): {offenders}"
