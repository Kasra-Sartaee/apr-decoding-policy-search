"""Benchmark abstraction.

Each benchmark yields `Problem` objects. A `Problem` carries the path to a
buggy source file plus a `TestRunner` callable that knows how to verify
candidate fixes against that benchmark's native test format.

This abstraction lets the pipeline drive heterogeneous benchmarks
(stdin/stdout-tested, function-harnessed, file-based, compiled or interpreted)
without the verifier needing to know which benchmark a program came from.
"""
from __future__ import annotations

import atexit
import dataclasses
import glob
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agents import _INFERENCE_VERIFY_GATE, VerificationResult, read_text
from energy import get_active_backend

# ── containment for candidate workloads ───────────────────────────────────
# Candidate programs are LLM-generated and routinely contain infinite loops.
# Plain subprocess.run(timeout=...) is not enough to contain them, for two
# independent reasons:
#
#   1. On timeout it kills only the DIRECT child. A test that forks leaves
#      the grandchild running.
#   2. If this process is killed, its cleanup never runs at all. The child is
#      reparented to init and runs forever.
#
# Both were observed. A Codeflaws candidate escaped during a wrapper restart
# and held one core at 99.9% for 2 days 4 hours, spanning two complete
# held-out evaluations; because RAPL measures the whole CPU package, it
# inflated every F3 reading by a measured 1.42x. Two Python candidates did the
# same for 12 and 7 days.
#
# The containment below is layered:
#   * every workload runs as its own process-group leader, so a timeout kill
#     reaches the whole tree rather than one pid;
#   * live groups are tracked, and SIGTERM/SIGINT/normal-exit all sweep them;
#   * sweep_stray_workloads() finds anything that escaped anyway, so a
#     pre-flight can refuse to start on a contaminated machine.
#
# process_group=0 (Python 3.11+) does the setsid in C, before exec. It is used
# in preference to preexec_fn because Phase A is threaded and preexec_fn runs
# Python between fork and exec, where it can deadlock on a lock held by
# another thread.
_LIVE_GROUPS: set[int] = set()
_LIVE_GROUPS_LOCK = threading.Lock()


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _sweep_live_groups() -> None:
    with _LIVE_GROUPS_LOCK:
        groups = list(_LIVE_GROUPS)
        _LIVE_GROUPS.clear()
    for pgid in groups:
        _kill_group(pgid)


def _install_containment_handlers() -> None:
    atexit.register(_sweep_live_groups)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            previous = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def handler(signum, frame, _prev=previous):
            _sweep_live_groups()
            if callable(_prev):
                _prev(signum, frame)
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Not the main thread, or the platform disallows it. The timeout
            # path and atexit still apply.
            pass


if threading.current_thread() is threading.main_thread():
    _install_containment_handlers()


def run_contained(
    cmd: list[str],
    *,
    timeout: float | None,
    input: str | None = None,          # noqa: A002 - mirrors subprocess.run
    cwd: str | Path | None = None,
    text: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """`subprocess.run` for candidate workloads that must not outlive us.

    Same signature subset, same return value, and the same
    `subprocess.TimeoutExpired` on timeout, so call sites are unchanged. The
    difference is that the timeout kills the whole process group and that an
    interrupted parent takes its workloads with it.
    """
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    stdin = subprocess.PIPE if input is not None else None

    proc = subprocess.Popen(                       # noqa: S603
        cmd, stdin=stdin, stdout=stdout, stderr=stderr,
        text=text, cwd=str(cwd) if cwd is not None else None,
        process_group=0,                           # own group: killpg reaches forks
    )
    pgid = proc.pid                                # group leader == child pid
    with _LIVE_GROUPS_LOCK:
        _LIVE_GROUPS.add(pgid)
    try:
        try:
            out, err = proc.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(pgid)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = "", ""
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    finally:
        if proc.poll() is None:
            _kill_group(pgid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        with _LIVE_GROUPS_LOCK:
            _LIVE_GROUPS.discard(pgid)


def sweep_stray_workloads(*, kill: bool = False) -> list[tuple[int, str]]:
    """Find candidate workloads that escaped a previous run.

    Returns `(pid, command)` for every live process that looks like a test
    workload this harness spawns: a compiled Codeflaws binary under a
    `codeflaws_runner_*` temp dir, or an interpreter running a candidate
    source from `outputs/`. Used by the pre-flight so a run cannot start on a
    machine whose CPU package is already contaminated.
    """
    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid=,args="], text=True, capture_output=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    me = os.getpid()
    strays: list[tuple[int, str]] = []
    for line in listing.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == me:
            continue
        looks_like_workload = (
            "codeflaws_runner_" in args
            or ("/outputs/ga/" in args and "/repaired/" in args)
            or ("/outputs/ga/" in args and "/optimized/" in args)
        )
        if looks_like_workload and "ps -eo" not in args:
            strays.append((pid, args))

    if kill:
        for pid, _ in strays:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    return strays


@contextmanager
def _energy_window():
    """Measure energy of the wrapped block using the run's active backend.

    Yields a mutable dict that, after the block exits, holds:
      {"energy_mj": float, "backend": str, "subtype": str}

    This is how F3 is measured: the TestRunner wraps *its own test-run
    loop*. For StdinStdoutRunner below, that loop is the
    same execution that determines tests_passed/tests_total, so energy and
    correctness come from one pass. Codeflaws' _CodeflawsTestRunner is a
    deliberate exception (see its class docstring and
    docs/reports/measurement_protocol.md): correctness
    comes from a separate, earlier diagnostic() pass with no energy
    measurement at all, and this function is only called again later, by
    measure_energy(), to rerun just the tests already known to pass. The
    runtime_proxy backend amplifies sub-ms programs internally; real
    backends (rapl_sysfs, apple_silicon, codecarbon) measure one pass.

    Acquires _INFERENCE_VERIFY_GATE as the sole writer for the duration of
    this call — see agents.py. For the Codeflaws v3.1 path the gate is
    uncontested overhead rather than a safety mechanism: Phase B runs only
    after the whole Phase A worker pool has joined, so nothing else in the
    process is doing inference, compilation or testing while the window is
    open. The gate still matters for the legacy/other-benchmark runners
    below, which measure inline while other workers may be active.
    """
    holder: dict[str, Any] = {"energy_mj": None, "backend": "", "subtype": ""}
    backend = get_active_backend()
    with _INFERENCE_VERIFY_GATE.write_lock():
        try:
            with backend.measure() as window:
                yield holder
            holder["energy_mj"] = window.energy_mj
            holder["backend"] = window.backend_name
            holder["subtype"] = window.backend_subtype
        except Exception:
            # Energy measurement must never break verification. Leave the
            # holder's energy_mj as None on any failure.
            holder["backend"] = "none"
    return


@dataclasses.dataclass
class EnergyMeasurement:
    """Result of a two-phase runner's measure_energy().

    Defined here rather than in any one adapter because
    `agentic_workflow.run_pipeline_phase_b()` consumes it structurally, by
    attribute, from whichever benchmark produced it. A per-adapter copy would
    be free to drift out of that contract silently.

    `mean_energy_mj` is the whole-batch energy window divided by the number
    of tests executed in it (F3's unit is unchanged: mean energy per test).
    It is None iff the measurement is INVALID -- compile failed after
    retries, nothing could be executed, or the batch window gave no usable
    reading. Callers MUST treat None as "no valid measurement" and never
    substitute 0.0: an unmeasured result is not a perfect, zero-energy one,
    and 0.0 would read as an unbeatable score to a minimiser.

    `mismatched_ids` lists tests that were expected to pass (they passed in
    diagnostic()) but did NOT reproduce on this rerun, with `mismatch_reasons`
    giving each one's cause: "timeout", "nonzero_exit", "output_mismatch",
    or "missing_test_file". These are CORRECTNESS facts and are what Phase B
    uses to revise F2 downward -- they are determined purely by comparing
    output, never by whether energy was measurable. (An earlier version
    conflated the two, marking a correct-but-unmeasurable test as a
    reproduction failure; with a whole-batch window the unmeasurable case
    no longer arises at all.) `mismatched_ids` can be non-empty while
    `mean_energy_mj` is still valid -- the window covered every executed
    test regardless of outcome.
    """
    mean_energy_mj: float | None
    backend: str
    backend_subtype: str
    matched_ids: list[str]
    mismatched_ids: list[str]
    mismatch_reasons: dict[str, str] = dataclasses.field(default_factory=dict)


def normalise_output(text: str) -> list[str]:
    """Canonical stdout normalisation, shared by every stdin/stdout runner.

    Drops empty lines, strips leading spaces/tabs, ignores trailing space.
    This is exactly what Codeflaws' own `test-valid.sh` does
    (`sed -e '/^$/d' -e 's/^[ \t]*//'` then `diff --ignore-trailing-space`),
    and `codeflaws/adapter.py::_official_normalise` is the C arm's copy of it.

    It matters that BOTH arms use the same rule. Plain `.strip()` equality is
    STRICTER -- it fails a submission the benchmark itself would accept -- so
    an arm using `.strip()` measures F2 downward-biased relative to an arm
    using this, and the two arms' correctness numbers stop being comparable.
    """
    out: list[str] = []
    for ln in text.split("\n"):
        if ln == "":
            continue
        out.append(ln.lstrip(" \t").rstrip())
    while out and out[-1] == "":
        out.pop()
    return out


class TestRunner(Protocol):
    """A callable that compiles + tests a candidate C source.

    Signature: (source_path: Path) -> VerificationResult.

    Implementations may compile, run, and compare outputs however the source
    benchmark requires. The returned VerificationResult is the same shape
    the existing pipeline already consumes.
    """

    def __call__(self, source_path: Path) -> VerificationResult: ...


@dataclass
class Problem:
    """A single benchmark problem to feed through the pipeline."""

    benchmark: str                    # e.g. "codeflaws"
    problem_id: str                   # stable, unique within benchmark
    buggy_source_path: Path           # the source file the analyzer/repair sees
    test_runner: TestRunner           # closure that verifies a candidate
    metadata: dict[str, Any] = field(default_factory=dict)
    source_extension: str = ".c"      # extension for repaired/optimized output files
    language_hint: str = "C99"        # fed into agent prompts as context
    harness_concatenated: bool = False
    # True when the test runner concatenates the candidate source with a
    # verifier-side harness that provides `main()`. In that case the
    # repair/optimize prompts must tell the model not to emit its own
    # `main()`. False for the full-program benchmarks shipped here
    # (Codeflaws, RunBugRun) whose source owns `main()` and
    # would break if it were stripped.

    @property
    def relative_label(self) -> str:
        """Stable label used as the pipeline's `relative_path` argument.

        Format: `<benchmark>/<problem_id>` so downstream output paths
        (reports/, repaired/, optimized/, verification/) collide-free.
        """
        return f"{self.benchmark}/{self.problem_id}"


class Benchmark(ABC):
    """Base class for benchmark adapters."""

    name: str = ""        # e.g. "codeflaws"
    version: str = ""     # benchmark version / commit / dataset revision

    def __init__(self, root: Path) -> None:
        """`root` is the per-benchmark cache directory under
        `benchmarks/<name>/data/`. Adapters store
        downloaded/extracted artifacts there so re-runs are idempotent.
        """
        self.root = root

    @abstractmethod
    def download(self) -> None:
        """Fetch the benchmark to `self.root`. Must be idempotent — if data is
        already present and valid, return without re-downloading."""

    @abstractmethod
    def problems(self) -> Iterator[Problem]:
        """Yield `Problem` instances. Order should be deterministic."""

    def is_downloaded(self) -> bool:
        """Override if a cheap check exists. Default: directory non-empty."""
        return self.root.is_dir() and any(self.root.iterdir())


class StdinStdoutRunner:
    """Default TestRunner: compile with gcc -O2, feed stdin from input*.txt,
    compare stdout to expected*.txt. Matches the existing synthetic pipeline.
    """

    def __init__(
        self,
        test_dir: Path,
        compile_timeout: int = 30,
        run_timeout: int = 10,
        extra_gcc_flags: list[str] | None = None,
    ) -> None:
        self.test_dir = test_dir
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.extra_gcc_flags = extra_gcc_flags or []

    def __call__(self, source_path: Path) -> VerificationResult:
        with tempfile.TemporaryDirectory(
            prefix="bench_runner_", ignore_cleanup_errors=True
        ) as tmpdir:
            exe = Path(tmpdir) / "prog.out"
            cmd = [
                "gcc",
                str(source_path),
                "-O2",
                "-Wall",
                "-Wextra",
                *self.extra_gcc_flags,
                "-o",
                str(exe),
            ]
            try:
                compile_proc = run_contained(
                    cmd,
                    timeout=self.compile_timeout,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    compiled=False,
                    compile_stdout="",
                    compile_stderr=f"gcc timed out after {self.compile_timeout}s",
                    tests_total=0,
                    tests_passed=0,
                    failed_tests=[],
                )

            if compile_proc.returncode != 0:
                return VerificationResult(
                    compiled=False,
                    compile_stdout=compile_proc.stdout,
                    compile_stderr=compile_proc.stderr,
                    tests_total=0,
                    tests_passed=0,
                    failed_tests=[],
                )

            inputs = sorted(glob.glob(str(self.test_dir / "input*.txt")))
            failed_tests: list[str] = []
            passed = 0
            # Per-test ids are REQUIRED: optimized_preserves_correctness()
            # fails closed without them, which would silently reject every
            # optimisation and make pi_o.* inert for this benchmark.
            passing_ids: list[str] = []

            # Measure F3 over the real test-run workload (all inputs once).
            with _energy_window() as energy:
                for input_path in inputs:
                    input_file = Path(input_path)
                    expected_file = input_file.with_name(
                        input_file.name.replace("input", "expected", 1)
                    )
                    if not expected_file.exists():
                        failed_tests.append(
                            f"{input_file.name}: missing {expected_file.name}"
                        )
                        continue

                    try:
                        run_proc = run_contained(
                            [str(exe)],
                            input=read_text(input_file),
                            timeout=self.run_timeout,
                        )
                    except subprocess.TimeoutExpired:
                        failed_tests.append(
                            f"{input_file.name}: timeout after {self.run_timeout}s"
                        )
                        continue

                    expected = read_text(expected_file).strip()
                    actual = run_proc.stdout.strip()

                    if run_proc.returncode != 0:
                        failed_tests.append(
                            f"{input_file.name}: rc={run_proc.returncode} "
                            f"stderr={run_proc.stderr.strip()}"
                        )
                    elif actual != expected:
                        failed_tests.append(f"{input_file.name}: output mismatch")
                    else:
                        passed += 1
                        passing_ids.append(input_file.name)

            return VerificationResult(
                compiled=True,
                compile_stdout=compile_proc.stdout,
                compile_stderr=compile_proc.stderr,
                tests_total=len(inputs),
                tests_passed=passed,
                failed_tests=failed_tests,
                passing_test_ids=passing_ids,
                energy_mj=energy["energy_mj"],
                energy_backend=energy["backend"],
                energy_backend_subtype=energy["subtype"],
            )


# Minimum program executions inside a single Phase B energy window, for the
# Python runner. Deliberately NOT the Codeflaws value (30): that is calibrated
# to ~0.6 ms C executions against a ~1 ms RAPL tick, where a single execution
# sits below the counter's resolution. A CPython process start alone is
# ~20-30 ms — already ~20-30 ticks — so one Python execution is comfortably
# resolvable and forcing 30 of them costs ~7x runtime for no precision gain.
#
# Measured on this host with scripts/calibrate_py_energy_window.py against
# rapl_sysfs (coefficient of variation over 40 trials, two replicates):
#
#   execs |  CV rep1 |  CV rep2 |  wall
#       1 |    5.24% |    5.48% |   0.4s
#       3 |    3.23% |    2.41% |   1.1s
#      10 |    3.23% |    4.87% |   3.7s
#      30 |        — |    3.62% |   7.0s
#
# A single execution is reliably the worst; past ~3 there is no monotone
# improvement (30 measured worse than 3), the residual differences being
# within the CV estimator's own noise. 3 sits in the same 2-3% band Codeflaws
# achieves with its whole-batch window, so F3 precision is comparable across
# the two arms — which is what the cross-language comparison requires.
PY_MIN_WINDOW_EXECS = 3


class PythonStdinStdoutRunner:
    """TestRunner for interpreted Python programs: no compilation, just run.

    Feeds each `input*.txt` as stdin, compares stdout to the paired
    `expected*.txt` under `normalise_output`.

    Two-phase interface, mirroring `_CodeflawsTestRunner` so both arms are
    measured under the identical protocol (see
    docs/reports/measurement_protocol.md):
      - diagnostic(source_path): run every test, NO energy measurement at
        all — never touches _energy_window or _INFERENCE_VERIFY_GATE. Safe
        to call freely from concurrent Phase A worker threads. Records
        passing_test_ids so Phase B can rerun exactly those.
      - measure_energy(source_path, passing_test_ids): rerun ONLY the
        already-known-passing tests inside ONE whole-batch energy window.
    __call__() composes both, for single-shot callers with no phase split.

    BOTH methods must exist. `run_pipeline_phase_a` duck-types on
    `diagnostic` alone, but `run_pipeline_phase_b` calls `measure_energy`
    unconditionally — a runner with only the first raises AttributeError,
    which the driver swallows into (inf, 0.0, inf), silently marking every
    problem infeasible.

    The glob patterns default to `input*.txt` / `expected*.txt` to match
    the naming convention used by `StdinStdoutRunner` and the synthetic
    benchmark. Adapters that use different names (e.g. `input_1.txt` and
    `output_1.txt`) can override `input_glob` / `expected_glob`.
    """

    def __init__(
        self,
        problem_dir: Path,
        input_glob: str = "input*.txt",
        expected_glob: str = "expected*.txt",
        run_timeout: int = 5,
        max_consecutive_timeouts: int = 3,
        min_window_execs: int = PY_MIN_WINDOW_EXECS,
    ) -> None:
        self.problem_dir = problem_dir
        self.input_glob = input_glob
        self.expected_glob = expected_glob
        self.run_timeout = run_timeout
        # Suites here run the full upstream test set (~100 cases). A repair that
        # loops would otherwise cost n_tests * run_timeout seconds — ~500s on one
        # problem. After this many consecutive timeouts the candidate is declared
        # non-viable and the remaining tests are scored as failures without being
        # run. Consecutive rather than total, so a program that is merely slow on
        # a few large inputs still gets credit for the cases it passes.
        #
        # This guard belongs to diagnostic() only. measure_energy() reruns just
        # the tests that already passed there, so a timeout cannot occur by
        # construction — which is exactly why a whole-batch window is safe here
        # when it was not in v1 (see the measurement-protocol report §3.2).
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.min_window_execs = min_window_execs

    def _cases(self) -> list[tuple[Path, Path]]:
        """Input/expected file pairs for this problem, keyed by case index."""
        by_key_in = {self._case_key(p): p for p in self.problem_dir.glob(self.input_glob)}
        by_key_exp = {self._case_key(p): p for p in self.problem_dir.glob(self.expected_glob)}
        return [
            (by_key_in[k], by_key_exp[k])
            for k in sorted(by_key_in.keys() & by_key_exp.keys(), key=lambda s: (len(s), s))
        ]

    @staticmethod
    def _case_key(path: Path) -> str:
        """Trailing digits of the filename stem — the test-case index.

        `input_12.txt` and `expected_12.txt` both key on "12", so cases are
        paired by identity rather than by position. Zipping two sorted lists
        would also work here, but only because both sides happen to share the
        same numeric ordering; keying is explicit and cannot silently mispair
        if the glob patterns or naming ever change.
        """
        m = re.search(r"(\d+)$", path.stem)
        return m.group(1) if m else ""

    def diagnostic(self, source_path: Path) -> VerificationResult:
        """Run every test, comparing output. NO energy measurement.

        Never opens an _energy_window, so this is safe to call concurrently
        from Phase A worker threads without contaminating anyone's reading.

        `compiled` reports whether the source is syntactically valid, via a
        compile() preflight. Python has no build step, so an earlier version
        returned compiled=True unconditionally — which silently disabled
        `pi_o.retry`, one of the 11 searched GA dimensions: the optimiser
        retry loop is `while not verify_opt.compiled`, so it could never fire
        on this arm. Mapping a SyntaxError onto compiled=False restores the C
        arm's semantics for every consumer of that flag (the retry loops,
        _finalise_from_repaired's compile guard, and the compile check inside
        optimized_preserves_correctness).
        """
        try:
            compile(
                source_path.read_text(encoding="utf-8", errors="replace"),
                str(source_path),
                "exec",
            )
        except SyntaxError as exc:
            return VerificationResult(
                compiled=False,
                compile_stdout="",
                compile_stderr=f"SyntaxError: {exc}",
                tests_total=0,
                tests_passed=0,
                failed_tests=[],
            )
        except (OSError, ValueError) as exc:
            # Unreadable file, or source containing null bytes.
            return VerificationResult(
                compiled=False,
                compile_stdout="",
                compile_stderr=f"{type(exc).__name__}: {exc}",
                tests_total=0,
                tests_passed=0,
                failed_tests=[],
            )

        pairs = self._cases()
        passed = 0
        failed_tests: list[str] = []
        passing_ids: list[str] = []
        consecutive_timeouts = 0

        for pair_idx, (inp, exp) in enumerate(pairs):
            if consecutive_timeouts >= self.max_consecutive_timeouts:
                remaining = len(pairs) - pair_idx
                failed_tests.append(
                    f"aborted after {consecutive_timeouts} consecutive timeouts; "
                    f"{remaining} remaining test(s) scored as failures"
                )
                break

            stdin_text = inp.read_text(encoding="utf-8", errors="replace")
            expected_out = exp.read_text(encoding="utf-8", errors="replace")

            try:
                run_proc = run_contained(
                    [sys.executable, str(source_path)],
                    input=stdin_text,
                    timeout=self.run_timeout,
                )
            except subprocess.TimeoutExpired:
                consecutive_timeouts += 1
                failed_tests.append(f"{inp.name}: timeout after {self.run_timeout}s")
                continue
            consecutive_timeouts = 0

            if run_proc.returncode != 0:
                failed_tests.append(
                    f"{inp.name}: rc={run_proc.returncode} "
                    f"stderr={run_proc.stderr.strip()[:200]}"
                )
                continue
            if normalise_output(run_proc.stdout) != normalise_output(expected_out):
                failed_tests.append(f"{inp.name}: output mismatch")
                continue

            passed += 1
            passing_ids.append(self._case_key(inp))

        return VerificationResult(
            compiled=True,
            compile_stdout="",
            compile_stderr="",
            tests_total=len(pairs),
            tests_passed=passed,
            failed_tests=failed_tests,
            # Required by optimized_preserves_correctness(), which fails CLOSED
            # when a runner omits these — without them the optimise stage would
            # be rejected on every problem and pi_o would become inert.
            passing_test_ids=passing_ids,
        )

    def measure_energy(
        self, source_path: Path, passing_test_ids: list[str]
    ) -> EnergyMeasurement:
        """Rerun ONLY already-passing tests inside ONE whole-batch window.

        Two concerns, deliberately decoupled (the measurement-protocol report §3.2):

        1. CORRECTNESS (feeds F2). Every rerun test's output is compared,
           always, regardless of whether energy was measurable. An earlier
           version of the C runner checked the energy sample first and
           `continue`d, so a correct-but-unmeasurable test was recorded as a
           reproduction failure — a correct program losing correctness for
           being fast.
        2. ENERGY (F3). One window spans the whole batch; F3 is the total
           divided by executions performed.

        A whole-batch window is safe here though it was not in v1: v1 wrapped
        the whole suite *including failing tests*, so a looping repair dumped
        N_tests x run_timeout into one reading. Only tests that already passed
        diagnostic() are rerun, and a timeout is recorded as a failure there,
        so it can never reach passing_test_ids.
        """
        result = EnergyMeasurement(
            mean_energy_mj=None, backend="", backend_subtype="",
            matched_ids=[], mismatched_ids=list(passing_test_ids),
        )
        if not passing_test_ids:
            return result

        wanted = set(passing_test_ids)
        # Read every input BEFORE opening the window so Python file I/O is
        # not charged to the program's energy.
        cases: list[tuple[str, str, str]] = []
        for inp, exp in self._cases():
            idx = self._case_key(inp)
            if idx not in wanted:
                continue
            cases.append((
                idx,
                inp.read_text(encoding="utf-8", errors="replace"),
                exp.read_text(encoding="utf-8", errors="replace"),
            ))
        if not cases:
            result.backend = "none"
            return result

        # Repeat the batch enough times to clear min_window_execs, so a
        # candidate passing few tests is not measured at the resolution floor
        # — otherwise F3's measurability would correlate with F2 and
        # systematically drop low-correctness candidates from the aggregates.
        reps = max(1, -(-self.min_window_execs // len(cases)))  # ceil div
        outcomes: list[tuple[str, object]] = []
        n_execs = 0
        consecutive_timeouts = 0
        aborted = False
        with _energy_window() as energy:
            for r in range(reps):
                if aborted:
                    break
                for idx, data, _expected in cases:
                    # These tests all passed diagnostic(), so a timeout here is
                    # a program that is stateful, nondeterministic, or sitting
                    # right on the threshold. Without this guard a suite of ~100
                    # cases costs ~100 x run_timeout for ONE problem — and under
                    # runtime_proxy (F3 = wall clock) that stall is charged
                    # straight to F3, inflating the objective instead of
                    # excluding the candidate.
                    if consecutive_timeouts >= self.max_consecutive_timeouts:
                        aborted = True
                        break
                    try:
                        proc = run_contained(
                            [sys.executable, str(source_path)],
                            input=data, timeout=self.run_timeout,
                        )
                        consecutive_timeouts = 0
                    except subprocess.TimeoutExpired:
                        proc = None
                        consecutive_timeouts += 1
                    n_execs += 1
                    if r == 0:  # correctness from the first pass only
                        outcomes.append((idx, proc))

        # --- correctness evidence, entirely outside the energy path ---
        matched_ids: list[str] = []
        mismatched_ids: list[str] = []
        mismatch_reasons: dict[str, str] = {}
        expected_by_idx = {idx: exp for idx, _d, exp in cases}
        for idx, proc in outcomes:
            if proc is None:
                mismatched_ids.append(idx)
                mismatch_reasons[idx] = "timeout"
            elif proc.returncode != 0:
                mismatched_ids.append(idx)
                mismatch_reasons[idx] = "nonzero_exit"
            elif normalise_output(proc.stdout) != normalise_output(expected_by_idx[idx]):
                mismatched_ids.append(idx)
                mismatch_reasons[idx] = "output_mismatch"
            else:
                matched_ids.append(idx)
        # ids that never ran: either a missing test file, or skipped because the
        # batch aborted. Both are reproduction failures for F2's purposes.
        for idx in passing_test_ids:
            if idx not in matched_ids and idx not in mismatched_ids:
                mismatched_ids.append(idx)
                mismatch_reasons[idx] = "aborted" if aborted else "missing_test_file"

        result.backend = energy["backend"]
        result.backend_subtype = energy["subtype"]
        result.matched_ids = matched_ids
        result.mismatched_ids = mismatched_ids
        result.mismatch_reasons = mismatch_reasons

        total = energy["energy_mj"]
        if aborted:
            # The window is dominated by timeout stalls, so the reading measures
            # how long the candidate hung, not how efficiently it runs. Report
            # it as INVALID (-> inf -> excluded by the isfinite feasibility
            # filter) rather than as a very large but finite energy, which would
            # enter the aggregate and distort F3.
            result.mean_energy_mj = None
        elif total is None or not math.isfinite(total) or total <= 0:
            # A batch window spans many counter ticks, so a non-positive total
            # is a genuine measurement failure, not a resolution artifact.
            # None, never 0.0 — 0.0 is an unbeatable score to a minimiser.
            result.mean_energy_mj = None
        else:
            # Divide by executions ACTUALLY attempted, not the nominal
            # len(cases) * reps, so an aborted batch cannot understate the mean.
            result.mean_energy_mj = total / n_execs if n_execs else None
        return result

    def __call__(self, source_path: Path) -> VerificationResult:
        """Single-shot path: diagnostic + immediate energy measurement, for
        callers that do not use the two-phase split."""
        diag = self.diagnostic(source_path)
        if not diag.compiled or diag.tests_passed == 0:
            return diag
        m = self.measure_energy(source_path, diag.passing_test_ids)
        return dataclasses.replace(
            diag, energy_mj=m.mean_energy_mj, energy_backend=m.backend,
            energy_backend_subtype=m.backend_subtype,
        )


class FunctionHarnessRunner:
    """TestRunner for benchmarks that ship a function under test plus a
    test driver template. The runner concatenates `<source> + <driver>`
    into a single .c file, compiles, and runs.

    Applies where each problem is a single function and the driver calls it
    with a list of fixed inputs. No benchmark shipped here uses it; it is
    retained because the measurement-protocol tests exercise it as a second
    runner shape when checking passing-test-id bookkeeping.
    """

    def __init__(
        self,
        driver_source: str,
        compile_timeout: int = 30,
        run_timeout: int = 10,
        extra_gcc_flags: list[str] | None = None,
        expected_outputs: list[str] | None = None,
    ) -> None:
        self.driver_source = driver_source
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.extra_gcc_flags = extra_gcc_flags or []
        # If supplied, the runner expects driver to print N newline-separated
        # outputs; each line is checked against expected_outputs[i].
        self.expected_outputs = expected_outputs or []

    def __call__(self, source_path: Path) -> VerificationResult:
        with tempfile.TemporaryDirectory(
            prefix="bench_funcharness_", ignore_cleanup_errors=True
        ) as tmpdir:
            tmp_path = Path(tmpdir)
            combined = tmp_path / "combined.c"
            user_src = read_text(source_path)
            combined.write_text(user_src + "\n\n" + self.driver_source, encoding="utf-8")

            exe = tmp_path / "prog.out"
            cmd = [
                "gcc",
                str(combined),
                "-O2",
                "-Wall",
                "-Wextra",
                *self.extra_gcc_flags,
                "-o",
                str(exe),
            ]
            try:
                compile_proc = run_contained(
                    cmd, timeout=self.compile_timeout
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    compiled=False,
                    compile_stdout="",
                    compile_stderr=f"gcc timed out after {self.compile_timeout}s",
                    tests_total=0,
                    tests_passed=0,
                    failed_tests=[],
                )
            if compile_proc.returncode != 0:
                return VerificationResult(
                    compiled=False,
                    compile_stdout=compile_proc.stdout,
                    compile_stderr=compile_proc.stderr,
                    tests_total=0,
                    tests_passed=0,
                    failed_tests=[],
                )

            try:
                run_proc = run_contained(
                    [str(exe)],
                    timeout=self.run_timeout,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    compiled=True,
                    compile_stdout=compile_proc.stdout,
                    compile_stderr=compile_proc.stderr,
                    tests_total=len(self.expected_outputs),
                    tests_passed=0,
                    failed_tests=[f"runtime timeout after {self.run_timeout}s"],
                )

            if run_proc.returncode != 0:
                return VerificationResult(
                    compiled=True,
                    compile_stdout=compile_proc.stdout,
                    compile_stderr=compile_proc.stderr,
                    tests_total=len(self.expected_outputs),
                    tests_passed=0,
                    failed_tests=[
                        f"runtime rc={run_proc.returncode} stderr={run_proc.stderr.strip()}"
                    ],
                )

            actual_lines = run_proc.stdout.strip().splitlines()
            failed: list[str] = []
            passed = 0
            passing_ids: list[str] = []
            total = len(self.expected_outputs)
            for i, expected in enumerate(self.expected_outputs):
                actual = actual_lines[i].strip() if i < len(actual_lines) else "<missing>"
                if actual == expected.strip():
                    passed += 1
                    passing_ids.append(str(i))
                else:
                    failed.append(f"case {i}: expected {expected!r} got {actual!r}")

            return VerificationResult(
                compiled=True,
                compile_stdout=compile_proc.stdout,
                compile_stderr=compile_proc.stderr,
                tests_total=total,
                tests_passed=passed,
                failed_tests=failed,
                passing_test_ids=passing_ids,
            )
