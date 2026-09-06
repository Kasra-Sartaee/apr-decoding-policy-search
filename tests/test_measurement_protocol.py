"""Regression tests for the v3.1 measurement path.

These pin the invariants that were each violated by a real bug found on
2026-07-27. They use fabricated sources/tests, never the LLM, and never a
real GA run, so they are fast and deterministic. Run with:

    .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import telemetry  # noqa: E402
from benchmarks.codeflaws.adapter import _CodeflawsTestRunner  # noqa: E402


# ---------------------------------------------------------------- fixtures
def _make_problem(tmp_path: Path, n_tests: int = 40, body: str = "") -> Path:
    """A trivial problem: echo the input integer. n_tests well-formed cases."""
    for i in range(1, n_tests + 1):
        (tmp_path / f"heldout-input-pos{i}").write_text(f"{i}\n")
        (tmp_path / f"heldout-output-pos{i}").write_text(f"{i}\n")
    src = tmp_path / "prog.c"
    src.write_text(
        "#include <stdio.h>\n"
        f"{body}\n"
        "int main(void){int x; if(scanf(\"%d\",&x)!=1) return 1; printf(\"%d\\n\",x); return 0;}\n"
    )
    return src


@pytest.fixture
def runner(tmp_path):
    return _CodeflawsTestRunner(
        problem_dir=tmp_path,
        compile_command_template=[],
        compile_timeout=30,
        run_timeout=3,
    )


# ------------------------------------------------------- energy vs correctness
def test_correctness_is_not_decided_by_energy_measurability(tmp_path, runner):
    """A test that runs correctly must never be recorded as a reproduction
    failure because its energy could not be measured.

    Regression: measure_energy() used to check the energy sample BEFORE
    comparing output and `continue`d, so a correct-but-too-fast test was
    marked mismatched -- and, once Phase B revised F2, demoted correctness.
    """
    src = _make_problem(tmp_path, n_tests=40)
    diag = runner.diagnostic(src)
    assert diag.compiled and diag.tests_passed == 40

    m = runner.measure_energy(src, diag.passing_test_ids)
    # Every test genuinely reproduces, so nothing may be flagged, regardless
    # of what the energy backend managed to record.
    assert m.mismatched_ids == [], f"correct tests flagged as failures: {m.mismatch_reasons}"
    assert len(m.matched_ids) == 40


def test_mismatch_reasons_are_correctness_reasons_only(tmp_path, runner):
    """mismatched_ids must only ever carry output-based causes now that the
    energy window is per batch, never energy-sample artifacts."""
    src = _make_problem(tmp_path, n_tests=10)
    diag = runner.diagnostic(src)
    m = runner.measure_energy(src, diag.passing_test_ids)
    allowed = {"timeout", "nonzero_exit", "output_mismatch", "missing_test_file"}
    assert set(m.mismatch_reasons.values()) <= allowed


def test_genuine_output_mismatch_is_still_detected(tmp_path, runner):
    """The mechanism must still catch a real non-reproducing test."""
    src = _make_problem(tmp_path, n_tests=5)
    diag = runner.diagnostic(src)
    assert diag.tests_passed == 5
    # Corrupt one expected file so that test can no longer reproduce.
    (tmp_path / "heldout-output-pos3").write_text("999\n")
    m = runner.measure_energy(src, diag.passing_test_ids)
    assert "3" in m.mismatched_ids
    assert m.mismatch_reasons["3"] == "output_mismatch"


# ------------------------------------------------------------ resolution floor
def test_small_batches_are_repeated_above_the_resolution_floor(tmp_path, runner):
    """A candidate passing very few tests must still yield a valid reading.

    Regression: with one passing test the batch window was ~0.6 ms against a
    ~1 ms RAPL tick, so energy came back invalid and the problem was dropped
    from the F1/F3 aggregates -- making F3's measurability correlate with F2.
    """
    from benchmarks.codeflaws.adapter import _MIN_WINDOW_EXECS

    src = _make_problem(tmp_path, n_tests=1)
    diag = runner.diagnostic(src)
    assert diag.tests_passed == 1
    m = runner.measure_energy(src, diag.passing_test_ids)
    assert m.mismatched_ids == []
    # Either a real backend produced a value, or none is registered in this
    # environment; what must NOT happen is a valid-but-zero reading.
    assert m.mean_energy_mj is None or m.mean_energy_mj > 0
    assert _MIN_WINDOW_EXECS >= 30


def test_no_passing_tests_is_invalid_not_zero(tmp_path, runner):
    """An unmeasurable result must never become 0.0 -- to a minimiser that
    reads as an unbeatable score."""
    src = _make_problem(tmp_path, n_tests=3)
    m = runner.measure_energy(src, [])
    assert m.mean_energy_mj is None


# ------------------------------------------------------------------ telemetry
def test_telemetry_is_isolated_between_threads():
    """Phase A workers call telemetry.reset() concurrently; one worker must
    not wipe or pollute another's per-problem accumulator."""
    telemetry.reset()
    telemetry.set_stage("main")
    telemetry.record_llm_call(model="m", wall_time_s=1.0, prompt_tokens=1,
                              completion_tokens=1, co2_kg=None, co2_backend="none")

    def worker():
        telemetry.reset()
        telemetry.record_llm_call(model="w", wall_time_s=9.0, prompt_tokens=99,
                                  completion_tokens=99, co2_kg=None, co2_backend="none")

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    s = telemetry.summary()
    assert s["total_llm_calls"] == 1, "another thread's reset/record leaked in"
    assert s["total_prompt_tokens"] == 1


# ------------------------------------------------------------------ provenance
def test_protocol_manifest_is_never_clobbered_on_resume(tmp_path, monkeypatch):
    """Resuming after a code change must not rewrite the provenance of the
    generations already computed."""
    sys.path.insert(0, str(REPO / "scripts"))
    import run_nsga2

    import provenance

    class Args:
        model = "m"; pop = 2; gen = 2; train_size = 2; seed_increment = 1

    # Patch on `provenance`, not on `run_nsga2`: the manifest writer is shared by
    # both GA drivers and resolves the commit hash there. run_nsga2 only re-exports
    # the helper under its old private name for its own run_manifest.
    monkeypatch.setattr(provenance, "git_commit_hash", lambda: "aaaaaaaa")
    run_nsga2._write_protocol_manifest(tmp_path, Args(), seed=42)
    first = (tmp_path / "protocol.json").read_text()

    monkeypatch.setattr(provenance, "git_commit_hash", lambda: "bbbbbbbb")
    run_nsga2._write_protocol_manifest(tmp_path, Args(), seed=42)

    import json
    after = json.loads((tmp_path / "protocol.json").read_text())
    assert after["git_commit"] == "aaaaaaaa", "original provenance was overwritten"
    assert after["resumed_under"][-1]["git_commit"] == "bbbbbbbb", "resume not recorded"
    assert "aaaaaaaa" in first


# ----------------------------------------------------------------- quarantine
def test_quarantine_applies_only_when_file_hashes_match(tmp_path, capsys):
    """A quarantine entry must be pinned to the exact reviewed data.

    If the benchmark files ever change, the quarantine must NOT silently
    suppress a test that is no longer the one that was examined.
    """
    from benchmarks.codeflaws.adapter import _QUARANTINED_TESTS, _quarantined_ids

    (pid, tid), rec = next(iter(_QUARANTINED_TESTS.items()))
    d = tmp_path / pid
    d.mkdir()

    # (a) content that does NOT match the recorded hashes -> not quarantined
    (d / f"heldout-input-pos{tid}").write_text("something else entirely\n")
    (d / f"heldout-output-pos{tid}").write_text("different\n")
    assert _quarantined_ids(d, pid) == set(), "quarantine applied to unreviewed data"
    assert "NOT applied" in capsys.readouterr().out

    # (b) the real files -> quarantined
    real = REPO / "benchmarks/codeflaws/data/codeflaws" / pid
    if (real / f"heldout-input-pos{tid}").is_file():
        assert _quarantined_ids(real, pid) == {tid}


def test_quarantined_test_is_excluded_from_the_suite():
    """The quarantined case must leave tests_total and passing ids entirely."""
    from benchmarks.codeflaws.adapter import _QUARANTINED_TESTS, _CodeflawsTestRunner

    (pid, tid), _ = next(iter(_QUARANTINED_TESTS.items()))
    d = REPO / "benchmarks/codeflaws/data/codeflaws" / pid
    if not d.is_dir():
        pytest.skip("codeflaws data not present")
    from benchmarks.codeflaws.adapter import _official_test_ids

    ref = sorted(p for p in d.glob("*.c") if "bug" not in p.name)
    if not ref:
        pytest.skip("no reference solution")
    official = _official_test_ids(d)
    assert official is not None and tid in official, (
        "this quarantine must remain necessary: the official suite still runs it"
    )
    r = _CodeflawsTestRunner(problem_dir=d, compile_command_template=[],
                             compile_timeout=30, run_timeout=3)
    res = r.diagnostic(ref[0])
    # official cases, minus the one quarantined case
    assert res.tests_total == len(official) - 1
    assert tid not in res.passing_test_ids


def test_adapter_follows_the_official_test_valid_suite():
    """F2 must be measured over Codeflaws' own case list, not every physical
    heldout-input-pos* file (which is up to 1.58x larger)."""
    from benchmarks.codeflaws.adapter import _CodeflawsTestRunner, _official_test_ids

    d = REPO / "benchmarks/codeflaws/data/codeflaws/140-A-bug-10658248-10658284"
    if not d.is_dir():
        pytest.skip("codeflaws data not present")
    official = _official_test_ids(d)
    physical = len(list(d.glob("heldout-input-pos*")))
    assert official is not None
    ref = sorted(p for p in d.glob("*.c") if "bug" not in p.name)
    r = _CodeflawsTestRunner(problem_dir=d, compile_command_template=[],
                             compile_timeout=30, run_timeout=3)
    res = r.diagnostic(ref[0])
    assert res.tests_total == len(official), (
        f"ran {res.tests_total} tests; official suite is {len(official)} "
        f"({physical} physical files present)"
    )


def test_unparseable_script_falls_back_to_globbing(tmp_path):
    """A problem without a usable test-valid.sh must still run its tests."""
    from benchmarks.codeflaws.adapter import _official_test_ids
    assert _official_test_ids(tmp_path) is None


# ------------------------------------------------------- optimiser sentinel
@pytest.mark.parametrize("text", [
    "NO_SAFE_TRANSFORMATION",
    "  NO_SAFE_TRANSFORMATION\n",
    "**NO_SAFE_TRANSFORMATION**",            # observed in real output
    "`NO_SAFE_TRANSFORMATION`",
    "```\nNO_SAFE_TRANSFORMATION\n```",
    "> **`NO_SAFE_TRANSFORMATION`**",
])
def test_sentinel_is_recognised_through_markdown_wrappers(text):
    """A decline must be recognised however the model wraps it, on the initial
    call AND on any retry. Five 22-byte 'C files' containing only the sentinel
    were produced before this was centralised."""
    from agentic_workflow import is_no_safe_transformation
    assert is_no_safe_transformation(text)


@pytest.mark.parametrize("text", [
    "#include <stdio.h>\nint main(void){return 0;}",
    "/* NO_SAFE_TRANSFORMATION */\nint main(void){return 0;}",
    "",
])
def test_real_source_is_never_mistaken_for_a_decline(text):
    from agentic_workflow import is_no_safe_transformation
    assert not is_no_safe_transformation(text)


# --------------------------------------------------- optimiser accept guard
def _vr(compiled, total, passing_ids):
    from agents import VerificationResult
    return VerificationResult(
        compiled=compiled, compile_stdout="", compile_stderr="",
        tests_total=total, tests_passed=len(passing_ids), failed_tests=[],
        passing_test_ids=list(passing_ids),
    )


def test_guard_rejects_same_count_but_different_passing_set():
    """The whole point of comparing sets rather than fractions.

    Real smoke output had forbidden edits accepted (a changed loop bound
    `i < N` -> `i < N - 3`, a changed branch condition, an uncommented qsort)
    because the passing COUNT happened to tie.
    """
    from agentic_workflow import optimized_preserves_correctness
    ok, why = optimized_preserves_correctness(_vr(True, 10, "12345"),
                                              _vr(True, 10, "12346"))
    assert not ok and "no longer pass" in why


def test_guard_rejects_a_non_compiling_artifact_even_on_a_zero_zero_tie():
    """A compiling repair must never be replaced by a non-compiling optimise
    just because both score 0.0."""
    from agentic_workflow import optimized_preserves_correctness
    ok, why = optimized_preserves_correctness(_vr(True, 10, ""), _vr(False, 0, ""))
    assert not ok and "did not compile" in why


def test_guard_accepts_only_when_every_passing_test_survives():
    from agentic_workflow import optimized_preserves_correctness
    assert optimized_preserves_correctness(_vr(True, 10, "123"), _vr(True, 10, "123"))[0]
    # a SUPERSET must be rejected: the optimiser contract forbids fixing bugs,
    # so gaining a previously-failing test is out-of-contract behaviour and
    # would credit a repair gain to the optimise stage (biasing F2 upward).
    ok, why = optimized_preserves_correctness(_vr(True, 10, "123"), _vr(True, 10, "1234"))
    assert not ok and "now pass" in why
    # losing one is not
    assert not optimized_preserves_correctness(_vr(True, 10, "1234"), _vr(True, 10, "123"))[0]
    # a different suite size is not comparable
    assert not optimized_preserves_correctness(_vr(True, 10, "123"), _vr(True, 9, "123"))[0]


def test_guard_fails_closed_when_a_runner_omits_per_test_ids():
    """Without per-test ids the set comparison is vacuous, so the guard must
    refuse rather than rubber-stamp. Non-Codeflaws runners do not populate
    passing_test_ids."""
    from agentic_workflow import optimized_preserves_correctness
    from agents import VerificationResult
    def novr(passed):
        return VerificationResult(compiled=True, compile_stdout="", compile_stderr="",
                                  tests_total=10, tests_passed=passed, failed_tests=[],
                                  passing_test_ids=[])
    ok, why = optimized_preserves_correctness(novr(9), novr(1))
    assert not ok and "per-test ids" in why


def test_missing_energy_is_infinite_not_zero():
    """F3 is minimised, so 0.0 for an unmeasured problem is an unbeatable
    score that also drags the feasible mean down. inf is excluded by the
    feasibility filter instead."""
    import math
    import pathlib
    import tempfile

    from agentic_workflow import evaluate_fitness
    from agents import VerificationResult
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "p.c"
        src.write_text("int main(void){return 0;}\n")
        v = VerificationResult(compiled=True, compile_stdout="", compile_stderr="",
                               tests_total=10, tests_passed=5, failed_tests=[],
                               energy_mj=None, passing_test_ids=[str(i) for i in range(5)])
        _cl, corr, energy = evaluate_fitness(src, v)
        assert math.isinf(energy), f"unmeasured energy became {energy}"
        assert corr == 0.5


def test_official_output_normalisation_matches_codeflaws_checker():
    """Codeflaws pipes output through sed '/^$/d' and 's/^[ \t]*//' then
    diffs with --ignore-trailing-space. Plain .strip() equality is stricter
    and can fail a submission the benchmark would accept."""
    from benchmarks.codeflaws.adapter import _official_normalise as n
    assert n("  1\n\n  2\n") == n("1\n2\n")
    assert n("1  \n2\n") == n("1\n2")
    assert n("1\n2\n") != n("1\n3\n")


def test_contracts_fail_closed_when_missing():
    """A missing contract silently became an EMPTY system prompt -- an
    invisible change of experimental treatment that still produces numbers."""
    from agents import load_contract
    with pytest.raises(FileNotFoundError):
        load_contract(Path("/nonexistent/contract.md"))


def test_every_runner_populates_passing_test_ids(tmp_path):
    """Fail-closed only works if runners actually report per-test ids.

    optimized_preserves_correctness() refuses to accept an optimisation when
    ids are missing. Only the Codeflaws adapter used to populate them, so on
    any OTHER benchmark the optimise stage would have been a silent no-op --
    rejecting even a byte-identical optimisation, leaving pi_o.{temperature,
    top_p,top_k,retry} (4 of the 11 GA dimensions) with no effect on any
    objective. Nothing would have crashed or warned.
    """
    from agentic_workflow import optimized_preserves_correctness
    from benchmarks.base import FunctionHarnessRunner, StdinStdoutRunner

    for i in (1, 2, 3):
        (tmp_path / f"input{i}.txt").write_text(f"{i}\n")
        (tmp_path / f"expected{i}.txt").write_text(f"{i}\n")
    src = tmp_path / "p.c"
    src.write_text('#include <stdio.h>\n'
                   'int main(void){int x;if(scanf("%d",&x)!=1)return 1;'
                   'printf("%d\\n",x);return 0;}\n')

    r = StdinStdoutRunner(test_dir=tmp_path)(src)
    assert r.tests_passed == 3
    assert len(r.passing_test_ids) == r.tests_passed, "StdinStdoutRunner omits ids"
    ok, why = optimized_preserves_correctness(r, r)
    assert ok, f"an identical optimisation was rejected: {why}"

    drv = 'int main(void){printf("%d\\n", f(1)); printf("%d\\n", f(2)); return 0;}\n'
    s2 = tmp_path / "q.c"
    s2.write_text("#include <stdio.h>\nint f(int x){return x;}\n")
    r2 = FunctionHarnessRunner(driver_source=drv, expected_outputs=["1", "2"])(s2)
    assert len(r2.passing_test_ids) == r2.tests_passed, "FunctionHarnessRunner omits ids"
