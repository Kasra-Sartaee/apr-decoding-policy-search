"""Codeflaws adapter — real C bug-repair benchmark from competitive programming.

Source:   Tan, Yi, Yuan, Roychoudhury (ASE 2017), "Codeflaws: A Programming
          Competition Benchmark for Evaluating Automated Program Repair Tools".
URL:      http://www.comp.nus.edu.sg/~release/codeflaws/
License:  CC-BY-SA 4.0 per the dataset README.

Layout in the upstream tarball (after extraction):
  codeflaws/
    71-A-bug-18078-18079/                          # one problem
      71-A-bug-18078-18079.c                       # BUGGY source
      71-A-bug-18078-18079.cpp                     # (some entries are .cpp)
      71-A.c                                       # reference (correct) source
      compile.sh                                   # gcc/g++ command
      input1, output1, input2, output2, ...        # test cases
      heldout-tests/                               # additional held-out tests
      ...
    codeflaws-defect-class.txt                     # defect taxonomy file
    codeflaws-defect-class-table-deeprepair.txt    # secondary classification

We use `codeflaws-defect-class.txt` to stratify-sample 300 problems balanced
across defect classes (seeded RNG -> deterministic).
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading as _threading
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from agents import VerificationResult

from ..base import (
    Benchmark,
    EnergyMeasurement,
    Problem,
    _energy_window,
    normalise_output,
    run_contained,
)


def _assert_members_stay_within(tf: tarfile.TarFile, dest: Path) -> None:
    """Reject any member that would write outside `dest`.

    Equivalent to the checks tarfile's "data" filter performs, for interpreters
    that predate it (before 3.11.4).
    """
    root = dest.resolve()
    for m in tf.getmembers():
        target = (root / m.name).resolve()
        if not target.is_relative_to(root):
            raise RuntimeError(f"archive member escapes the destination: {m.name!r}")
        if m.issym() or m.islnk():
            link = (target.parent / m.linkname).resolve()
            if not link.is_relative_to(root):
                raise RuntimeError(f"archive link escapes the destination: {m.name!r}")
        if m.isdev():
            raise RuntimeError(f"archive contains a device node: {m.name!r}")


CODEFLAWS_URL = "https://www.comp.nus.edu.sg/~release/codeflaws/codeflaws.tar.gz"
# The upstream host has historically served this over plain HTTP only. The
# fallback exists so the download still works there, but it is announced rather
# than silent, and an HTTP-fetched archive is unauthenticated unless
# CODEFLAWS_TARBALL_SHA256 is set. Extraction is filtered either way.
CODEFLAWS_URL_INSECURE = "http://www.comp.nus.edu.sg/~release/codeflaws/codeflaws.tar.gz"
# Set CODEFLAWS_TARBALL_SHA256 to pin the archive. Left unset upstream publishes
# no digest; supply one to make the download verifiable.
CODEFLAWS_TARBALL_SHA256 = os.environ.get("CODEFLAWS_TARBALL_SHA256", "")
SAMPLE_SIZE_DEFAULT = 300
SAMPLE_SEED = 42  # stable; documented in README

# Minimum program executions inside a single Phase B energy window. The
# RAPL counter updates only every ~1 ms while one Codeflaws test runs
# ~0.6 ms, so a window must span many executions to sit clear of the
# resolution floor. ~30 executions gives ~2-3% run-to-run spread
# (measured), versus 4-17% for single-execution windows.
_MIN_WINDOW_EXECS = 30

# The timeouts ACTUALLY in force for Codeflaws. problems() constructs the
# runner without passing these, so these defaults are what every run really
# uses -- run_nsga2.py's config dict advertised compile_timeout=40, which was
# never the active value. Exported so the manifest records the truth.
CODEFLAWS_COMPILE_TIMEOUT_S = 30
CODEFLAWS_RUN_TIMEOUT_S = 3


# NOTE: EnergyMeasurement moved to benchmarks/base.py so the C and Python arms
# return the identical structure to run_pipeline_phase_b(), which consumes it by
# attribute. It is imported above, so `from ...codeflaws.adapter import
# EnergyMeasurement` still resolves. Pure move — no field or behaviour change.


# ── Official test suite (test-valid.sh) ───────────────────────────────────────
# Each Codeflaws problem ships a `test-valid.sh` enumerating the positive test
# cases the benchmark authors intend to be used. Its case list is NOT the same
# as the set of `heldout-input-pos*` files on disk, and NOT contiguous: for
# 173-A it runs physical indices 1-17, 19, 21, 22, 27, 29, 31, 32 (24 cases)
# out of 38 files present.
#
# Earlier revisions of this adapter globbed every physical file, so F2 was
# computed over an "extended physical-file suite" ~1.09x the official one
# (148,795 vs 137,094 executions benchmark-wide; 43% of problems affected,
# up to 1.58x on individual problems). We now follow the official list so F2
# is measured over the same cases as published Codeflaws results.
#
# Only positive cases are taken: negative (`*-neg*`) tests are adversarial and
# excluded so F2 matches the published Codeflaws "fix" criterion. Verified
# across all 3,904 shipped scripts: none list negative cases in their case
# statement, and no referenced positive file is missing from disk.
_OFFICIAL_CASE_RE = re.compile(r'run_test\s+"\$INPUT_NAME"(\d+)')
_official_cache: dict[str, frozenset[str] | None] = {}
_official_lock = _threading.Lock()


def _official_test_ids(problem_dir: Path) -> frozenset[str] | None:
    """Positive test ids from the problem's test-valid.sh, or None if it is
    absent/unparseable (25 of 3,904 problems, all of which also have zero
    physical positive tests, so the fallback is equivalent there)."""
    key = str(problem_dir)
    with _official_lock:
        if key in _official_cache:
            return _official_cache[key]
    sh = problem_dir / "test-valid.sh"
    ids: frozenset[str] | None = None
    if sh.is_file():
        try:
            found = _OFFICIAL_CASE_RE.findall(sh.read_text(errors="replace"))
            if found:
                ids = frozenset(found)
        except OSError:
            ids = None
    with _official_lock:
        _official_cache[key] = ids
    return ids


# ── Quarantined test cases ────────────────────────────────────────────────────
# Individual Codeflaws test cases that are UNSOUND -- the input is malformed
# relative to the problem's own I/O contract, so every conforming program has
# undefined behaviour on it and there is no well-defined correct answer. Such a
# test passes or fails depending on memory layout, which makes it a source of
# spurious non-reproduction rather than a real correctness signal.
#
# Keyed by (problem_id, test_id) AND pinned to the SHA-256 of the input and
# expected-output files. If the benchmark data ever changes under us the hash
# stops matching, the quarantine is NOT applied, and a warning is emitted --
# so this can never silently suppress a test that is no longer the one we
# examined. Each entry must cite the evidence for its exclusion.
_QUARANTINED_TESTS: dict[tuple[str, str], dict[str, str]] = {
    ("173-A-bug-4169928-4169966", "16"): {
        "input_sha256":  "dbc7fc44256034937ae7b10385f40b2320269cbbbadebe0a69d45449e6350973",
        "output_sha256": "8fad34bbb0c1ed095fbf1b50cb0e48785a030d5af68dc1a4957cbb583c3c1e5a",
        "reason": (
            "Malformed input: the problem's I/O contract is three tokens "
            "(n, s1, s2) and 37 of its 38 held-out inputs supply three, but "
            "this one supplies two. The third scanf() therefore matches "
            "nothing and s2 is read uninitialised. Confirmed with valgrind "
            "('Conditional jump or move depends on uninitialised value(s)') "
            "and by direct demonstration: the same binary on the same input "
            "returns '1 0' or '0 1' depending only on stack offset (38 vs 4 "
            "of 42 invocations). Codeflaws' own reference solution "
            "(173-A-4169966.c) has the same three scanf() calls, so the "
            "expected output '1 0' was itself produced by a program with "
            "undefined behaviour -- there is no correct answer for this "
            "case. Note it IS included in the official test-valid.sh (p16), "
            "so following the official suite would not exclude it. "
            "See docs/reports/15 3.5."
        ),
    },
}


def _quarantined_ids(problem_dir: Path, problem_id: str) -> set[str]:
    """Test ids to skip for this problem, after verifying their file hashes."""
    skip: set[str] = set()
    for (pid, tid), rec in _QUARANTINED_TESTS.items():
        if pid != problem_id:
            continue
        fin = problem_dir / f"heldout-input-pos{tid}"
        fexp = problem_dir / f"heldout-output-pos{tid}"
        if not fin.is_file() or not fexp.is_file():
            continue
        got_in = hashlib.sha256(fin.read_bytes()).hexdigest()
        got_out = hashlib.sha256(fexp.read_bytes()).hexdigest()
        if got_in == rec["input_sha256"] and got_out == rec["output_sha256"]:
            skip.add(tid)
        else:
            print(
                f"[codeflaws] WARNING: quarantine for {pid} test {tid} NOT applied — "
                f"file hashes differ from the reviewed data (input {got_in[:12]} vs "
                f"{rec['input_sha256'][:12]}). The test will run normally; re-review it."
            )
    return skip


def _official_normalise(text: str) -> list[str]:
    """Normalise program output the way Codeflaws' own test-valid.sh does.

    The upstream script pipes output through `sed -e '/^$/d' -e 's/^[ \t]*//'`
    (drop empty lines, strip leading spaces/tabs) and then compares with
    `diff --brief --ignore-trailing-space`. Comparing raw `.strip()` equality,
    as this adapter previously did, is STRICTER than the benchmark's own
    checker and can therefore fail a submission Codeflaws would accept --
    biasing F2 downward.

    Applied symmetrically to both sides. Upstream only normalises the
    candidate's output, so this is marginally more lenient in the rare case
    where the EXPECTED file itself contains a blank line; it is never
    stricter.

    Delegates to benchmarks.base.normalise_output so the C and Python arms
    cannot drift apart: an edit here that was not mirrored there would make F2
    stricter on one arm than the other while both still looked correct.
    """
    return normalise_output(text)


class _CodeflawsTestRunner:
    """Compiles the candidate buggy source with the upstream compile command,
    runs each input* file as stdin, compares stdout to output*.

    Two-phase interface (used by scripts/run_nsga2.py's N_WORKERS>1 path):
      - diagnostic(source_path): compile + run every held-out test, NO
        energy measurement at all — never touches _INFERENCE_VERIFY_GATE
        or RAPL. Safe to call freely from concurrent worker threads.
        Records passing_test_ids so a later pass can rerun exactly those.
      - measure_energy(source_path, passing_test_ids): recompile once and
        rerun ONLY the given (already-known-passing) test ids, measuring
        F3 per test via _energy_window(), which does acquire
        _INFERENCE_VERIFY_GATE internally (same as any other caller of
        _energy_window()). In practice this is uncontested and just
        overhead: real safety comes from the CALLER's true
        join-then-serial-loop barrier (see run_nsga2.py's Phase B) — by the
        time this runs, every Phase A worker has already joined, so there
        is no concurrent inference for the gate to ever actually block.
    __call__() composes both in sequence, for single-shot callers that
    don't do the two-phase split (legacy sequential mode).

    Rationale: under the previous single-phase design, EVERY intermediate
    repair-retry and optimizer-retry verification ran its tests under
    RAPL, even though only the FINAL attempt's energy is ever used — and
    partial/broken repairs frequently hang, each timeout burning the full
    run_timeout under exclusive RAPL access while blocking every other
    worker's inference. On the n50 training set (rep_0 log), repair-retry
    + optimizer-retry suites were 82.5% of all suite executions (135,558 +
    7,494 of 213,567), and every one of those energy readings was
    discarded. Deferring energy measurement to one final pass over only
    the tests already known to pass removes nearly all of that wasted,
    serialised RAPL time — see memory/rapl_contamination_local_ollama.md.
    """

    def __init__(
        self,
        problem_dir: Path,
        compile_command_template: list[str],
        compile_timeout: int = CODEFLAWS_COMPILE_TIMEOUT_S,
        run_timeout: int = CODEFLAWS_RUN_TIMEOUT_S,
        is_cpp: bool = False,
    ) -> None:
        self.problem_dir = problem_dir
        self.compile_command_template = compile_command_template
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.is_cpp = is_cpp

    def _compile(self, source_path: Path, exe: Path):
        """Shared compile step; not gated (compilation itself is never
        RAPL-measured, only the test executions that follow it are).
        Returns (compile_proc_or_None, compiler_name); compile_proc is
        None on a compile timeout."""
        compiler = "g++" if self.is_cpp else "gcc"
        std_flag = "-std=c++17" if self.is_cpp else "-std=c99"
        # Codeflaws Makefile uses these flags for C; keep them so we
        # match the upstream build exactly.
        extra_c = ["-fno-optimize-sibling-calls", "-fno-strict-aliasing", "-fno-asm"]
        flags = [std_flag, "-O2", "-Wall", "-Wextra"] + ([] if self.is_cpp else extra_c)
        cmd = [compiler, str(source_path), *flags, "-lm", "-o", str(exe)]
        try:
            return run_contained(cmd, timeout=self.compile_timeout), compiler
        except subprocess.TimeoutExpired:
            return None, compiler

    def diagnostic(self, source_path: Path) -> VerificationResult:
        """Compile + run every held-out positive test, comparing output.
        No energy measurement. Codeflaws ships positive tests as
        `heldout-input-posN` paired with `heldout-output-posN`; negative
        tests (`*-neg*`) are adversarial and excluded so F2 matches the
        published Codeflaws "fix" criterion."""
        with tempfile.TemporaryDirectory(
            prefix="codeflaws_runner_", ignore_cleanup_errors=True
        ) as tmpdir:
            exe = Path(tmpdir) / ("prog.exe" if os.name == "nt" else "prog.out")
            compile_proc, compiler = self._compile(source_path, exe)
            if compile_proc is None:
                return VerificationResult(
                    compiled=False, compile_stdout="",
                    compile_stderr=f"{compiler} timed out after {self.compile_timeout}s",
                    tests_total=0, tests_passed=0, failed_tests=[],
                )
            if compile_proc.returncode != 0:
                return VerificationResult(
                    compiled=False,
                    compile_stdout=compile_proc.stdout,
                    compile_stderr=compile_proc.stderr,
                    tests_total=0, tests_passed=0, failed_tests=[],
                )

            inputs = sorted(
                p for p in self.problem_dir.iterdir()
                if p.is_file() and p.name.startswith("heldout-input-pos")
            )
            # Restrict to the official test-valid.sh case list (see above).
            _official = _official_test_ids(self.problem_dir)
            if _official is not None:
                inputs = [
                    p for p in inputs
                    if p.name.replace("heldout-input-pos", "", 1) in _official
                ]
            # Then drop any quarantined case (see _QUARANTINED_TESTS). Note the
            # official list does NOT exclude these -- 173-A test 16 is p16 in
            # its own test-valid.sh -- so this filter is still required.
            _skip = _quarantined_ids(self.problem_dir, self.problem_dir.name)
            if _skip:
                inputs = [
                    p for p in inputs
                    if p.name.replace("heldout-input-pos", "", 1) not in _skip
                ]
            failed_tests: list[str] = []
            passed = 0
            passing_ids: list[str] = []
            for input_file in inputs:
                idx = input_file.name.replace("heldout-input-pos", "", 1)
                expected_file = self.problem_dir / f"heldout-output-pos{idx}"
                if not expected_file.is_file():
                    failed_tests.append(
                        f"{input_file.name}: missing heldout-output-pos{idx}"
                    )
                    continue
                try:
                    run_proc = run_contained(
                        [str(exe)],
                        input=input_file.read_text(encoding="utf-8", errors="replace"),
                        timeout=self.run_timeout,
                    )
                except subprocess.TimeoutExpired:
                    failed_tests.append(f"{input_file.name}: timeout after {self.run_timeout}s")
                    continue

                expected = expected_file.read_text(encoding="utf-8", errors="replace")
                actual = run_proc.stdout
                if run_proc.returncode != 0:
                    failed_tests.append(
                        f"{input_file.name}: rc={run_proc.returncode} "
                        f"stderr={run_proc.stderr.strip()[:200]}"
                    )
                elif _official_normalise(actual) != _official_normalise(expected):
                    failed_tests.append(f"{input_file.name}: output mismatch")
                else:
                    passed += 1
                    passing_ids.append(idx)

            return VerificationResult(
                compiled=True,
                compile_stdout=compile_proc.stdout,
                compile_stderr=compile_proc.stderr,
                tests_total=len(inputs),
                tests_passed=passed,
                failed_tests=failed_tests,
                passing_test_ids=passing_ids,
            )

    def measure_energy(
        self, source_path: Path, passing_test_ids: list[str], compile_retries: int = 2
    ) -> EnergyMeasurement:
        """Recompile source_path, rerun ONLY passing_test_ids, and measure
        their energy in ONE window around the whole batch.

        Two separate concerns, deliberately decoupled:

        1. CORRECTNESS evidence (feeds F2). Every rerun test's output is
           compared, always. A test that fails to reproduce (timeout,
           wrong output, crash) is recorded in `mismatched_ids` with a
           reason. This is completely independent of whether energy was
           measurable -- an earlier version checked the energy sample
           FIRST and `continue`d, so a correct-but-unmeasurable test was
           silently recorded as a reproduction failure and (once Phase B
           began revising F2) demoted correctness. That was a bug: energy
           measurability says nothing about whether a program is correct.

        2. ENERGY (F3). ONE `_energy_window()` spans the entire batch, and
           F3 is that total divided by the number of tests executed.
           Per-test windows were abandoned because a single Codeflaws test
           runs ~0.6 ms while the RAPL counter only updates every ~1 ms, so
           an individual window sat at the resolution floor: readings were
           quantised to roughly "0 or one tick", giving 4-17% run-to-run
           spread and a class of spurious exactly-0.0 samples. Measuring
           the whole batch at once puts the window ~N ticks wide instead of
           ~0.6, cutting run-to-run spread to 2-3% and eliminating the
           zero-sample class entirely.

        Why a batch window is safe here, when it was not in v1: v1 wrapped
        the WHOLE suite including failing tests, so one looping repair
        dumped N_tests * run_timeout seconds into a single reading (up to
        ~24M mJ; docs/reports/09 §2.7). Phase B only ever reruns tests that
        ALREADY PASSED in diagnostic(), and a timed-out test is recorded as
        a failure there, so it can never appear in `passing_test_ids`. The
        pathology that killed v1 cannot occur.

        No baseline subtraction: ~60% of a reading is constant
        process-spawn overhead, but that constant cancels when NSGA-II
        compares two candidates, so subtracting an estimate of it recovers
        no signal while injecting the baseline's own variance. Measured:
        identical signal (2.828 mJ/test either way), pooled noise 0.299 ->
        0.510, discrimination (signal / pooled sd) 9.45 -> 5.55. See
        docs/reports/measurement_protocol.md.

        `mean_energy_mj` is None (INVALID -- callers must treat as
        infeasible, never as F3=0.0, which would read as an unbeatable
        score to a minimiser) when recompilation fails, when no test could
        be executed, or when the batch window yields no usable reading.
        """
        result = EnergyMeasurement(
            mean_energy_mj=None, backend="", backend_subtype="",
            matched_ids=[], mismatched_ids=list(passing_test_ids),
        )
        if not passing_test_ids:
            return result
        with tempfile.TemporaryDirectory(
            prefix="codeflaws_energy_", ignore_cleanup_errors=True
        ) as tmpdir:
            exe = Path(tmpdir) / ("prog.exe" if os.name == "nt" else "prog.out")
            compile_proc = None
            for _attempt in range(compile_retries + 1):
                compile_proc, compiler = self._compile(source_path, exe)
                if compile_proc is not None and compile_proc.returncode == 0:
                    break
            else:
                # Shouldn't normally happen (diagnostic() already compiled
                # this exact source successfully) -- but genuinely possible
                # (disk/resource exhaustion, transient toolchain hiccup) and
                # must surface as an invalid measurement, not F3=0.
                result.backend = "none"
                return result

            # Read every input BEFORE opening the window so Python file I/O
            # is not charged to the program's energy.
            cases = []
            for idx in passing_test_ids:
                fin = self.problem_dir / f"heldout-input-pos{idx}"
                fexp = self.problem_dir / f"heldout-output-pos{idx}"
                if not fin.is_file() or not fexp.is_file():
                    continue
                cases.append((
                    idx,
                    fin.read_text(encoding="utf-8", errors="replace"),
                    fexp.read_text(encoding="utf-8", errors="replace"),
                ))
            if not cases:
                result.backend = "none"
                return result

            # --- ONE energy window around the whole batch ---
            # A batch of N tests spans ~N RAPL ticks. When a candidate
            # passes only a handful of tests the batch would be back at the
            # ~1 ms resolution floor, so the batch is repeated enough times
            # to clear _MIN_WINDOW_EXECS executions. Without this, precision
            # would depend on how many tests a candidate passes -- i.e. F3's
            # measurability would correlate with F2, systematically dropping
            # low-correctness candidates out of the F1/F3 aggregates.
            # Repeats cost ~0.6 ms each and only occur for small batches.
            reps = max(1, -(-_MIN_WINDOW_EXECS // len(cases)))  # ceil div
            outcomes: list[tuple[str, object]] = []
            with _energy_window() as energy:
                for r in range(reps):
                    for idx, data, _expected in cases:
                        try:
                            proc = run_contained(
                                [str(exe)], input=data, timeout=self.run_timeout,
                            )
                        except subprocess.TimeoutExpired:
                            proc = None
                        if r == 0:      # correctness from the first pass only
                            outcomes.append((idx, proc))
            n_execs = len(cases) * reps

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
                elif _official_normalise(proc.stdout) != _official_normalise(expected_by_idx[idx]):
                    mismatched_ids.append(idx)
                    mismatch_reasons[idx] = "output_mismatch"
                else:
                    matched_ids.append(idx)
            # ids dropped for a missing input/expected file never ran
            for idx in passing_test_ids:
                if idx not in matched_ids and idx not in mismatched_ids:
                    mismatched_ids.append(idx)
                    mismatch_reasons[idx] = "missing_test_file"

            result.backend = energy["backend"]
            result.backend_subtype = energy["subtype"]
            result.matched_ids = matched_ids
            result.mismatched_ids = mismatched_ids
            result.mismatch_reasons = mismatch_reasons

            total = energy["energy_mj"]
            if total is None or not math.isfinite(total) or total <= 0:
                # A batch window spans ~N RAPL ticks, so unlike the old
                # per-test windows a non-positive total is a genuine
                # measurement failure, not a resolution artifact.
                result.mean_energy_mj = None
            else:
                # Mean per test EXECUTED in the window (covering every
                # execution, including repeats and any test that turned out
                # not to reproduce), preserving F3's "mean energy per test"
                # unit regardless of how many repeats were needed.
                result.mean_energy_mj = total / n_execs
            return result

    def __call__(self, source_path: Path) -> VerificationResult:
        """Single-shot path: diagnostic + immediate energy measurement in
        one call, for callers that don't use the two-phase split (legacy
        sequential mode, or any driver invoking this TestRunner directly
        without a Phase A/B batching barrier)."""
        diag = self.diagnostic(source_path)
        if not diag.compiled or diag.tests_passed == 0:
            return diag
        m = self.measure_energy(source_path, diag.passing_test_ids)
        return dataclasses.replace(
            diag, energy_mj=m.mean_energy_mj, energy_backend=m.backend,
            energy_backend_subtype=m.backend_subtype,
        )


class CodeflawsBenchmark(Benchmark):
    name = "codeflaws"
    version = "comp.nus.edu.sg/codeflaws-tarball"

    def __init__(self, root: Path, sample_size: int = SAMPLE_SIZE_DEFAULT, seed: int = SAMPLE_SEED) -> None:
        super().__init__(root)
        self.sample_size = sample_size
        self.seed = seed
        self.extracted_root = self.root / "codeflaws"
        self.sample_path = self.root / f"sample_{sample_size}_seed{seed}.json"

    def is_downloaded(self) -> bool:
        return self.extracted_root.is_dir() and (self.root / "_DOWNLOAD_OK").is_file()

    def download(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tarball = self.root / "codeflaws.tar.gz"

        if not tarball.is_file():
            try:
                print(f"[codeflaws] downloading {CODEFLAWS_URL} -> {tarball}")
                with urllib.request.urlopen(CODEFLAWS_URL) as resp, tarball.open("wb") as f:
                    shutil.copyfileobj(resp, f)
            except urllib.error.URLError as exc:
                print(f"[codeflaws] HTTPS failed ({exc}); retrying over plain HTTP")
                with urllib.request.urlopen(CODEFLAWS_URL_INSECURE) as resp, tarball.open("wb") as f:
                    shutil.copyfileobj(resp, f)

        if CODEFLAWS_TARBALL_SHA256:
            got = hashlib.sha256(tarball.read_bytes()).hexdigest()
            if got != CODEFLAWS_TARBALL_SHA256:
                tarball.unlink()
                raise RuntimeError(
                    f"codeflaws.tar.gz digest {got} does not match the expected "
                    f"{CODEFLAWS_TARBALL_SHA256}; the download was discarded."
                )
        else:
            print("[codeflaws] archive digest not pinned "
                  "(set CODEFLAWS_TARBALL_SHA256 to verify the download)")

        if not self.extracted_root.is_dir():
            print(f"[codeflaws] extracting {tarball}")
            with tarfile.open(tarball, "r:gz") as tf:
                # `filter="data"` rejects absolute paths, parent traversal and
                # links pointing outside the destination. It arrived in 3.11.4,
                # so where it is unavailable the same conditions are checked
                # directly rather than falling back to unfiltered extraction,
                # which would let a tampered archive write anywhere.
                if "filter" in inspect.signature(tf.extractall).parameters:
                    tf.extractall(self.root, filter="data")
                else:
                    _assert_members_stay_within(tf, self.root)
                    tf.extractall(self.root)

        # Sentinel.
        (self.root / "_DOWNLOAD_OK").write_text(
            json.dumps({"version": self.version}), encoding="utf-8"
        )

    def _read_defect_classes(self) -> dict[str, str]:
        """problem_dir_name -> defect class label, parsed from
        codeflaws-defect-detail-info.txt. Lines are tab-separated:
            <problem_dir>\t<defect_class>\t<failure_kind>\t<construct_tags>
        We use column 1 (defect_class) as the stratification key — the
        coarse-grained taxonomy used by the Codeflaws ASE'17 paper.
        """
        path = self.extracted_root / "codeflaws-defect-detail-info.txt"
        out: dict[str, str] = {}
        if not path.is_file():
            return out
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = raw.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1].strip() or "_unclassified"
        return out

    def _enumerate_problem_dirs(self) -> list[Path]:
        """Every directory that looks like a problem dir.

        Codeflaws problem dirs contain a `Makefile` (the upstream build
        driver) and at least one `heldout-input-pos*` test case. Dir
        name pattern: `<contest>-<problem>-bug-<buggy_rev>-<fixed_rev>`.
        """
        out: list[Path] = []
        for entry in sorted(self.extracted_root.iterdir()):
            if not entry.is_dir():
                continue
            if "-bug-" not in entry.name:
                continue
            if not (entry / "Makefile").is_file():
                continue
            if not any(p.name.startswith("heldout-input-pos") for p in entry.iterdir() if p.is_file()):
                continue
            out.append(entry)
        return out

    def _stratified_sample(self) -> list[Path]:
        """Returns a deterministic stratified sample across defect classes."""
        if self.sample_path.is_file():
            saved = json.loads(self.sample_path.read_text(encoding="utf-8"))
            return [self.extracted_root / d for d in saved["problem_dirs"]]

        defect_classes = self._read_defect_classes()
        all_dirs = self._enumerate_problem_dirs()

        # Group by defect class (problems missing a class go to "_unclassified").
        groups: dict[str, list[Path]] = defaultdict(list)
        for d in all_dirs:
            label = defect_classes.get(d.name, "_unclassified")
            groups[label].append(d)

        # Allocate quota per class proportional to class size, total = sample_size.
        rng = random.Random(self.seed)
        total_pool = sum(len(v) for v in groups.values())
        if total_pool == 0:
            raise RuntimeError(
                f"Codeflaws: no problem directories found under {self.extracted_root}"
            )

        # Fractional quotas, rounded with largest-remainder so totals match.
        raw_quotas = {
            k: (len(v) / total_pool) * self.sample_size for k, v in groups.items()
        }
        floors = {k: int(q) for k, q in raw_quotas.items()}
        remainder = self.sample_size - sum(floors.values())
        # Distribute remainder by largest fractional part.
        fractions = sorted(
            ((raw_quotas[k] - floors[k], k) for k in raw_quotas),
            reverse=True,
        )
        for _, k in fractions[:remainder]:
            floors[k] += 1

        sampled: list[Path] = []
        for k, dirs in groups.items():
            quota = min(floors[k], len(dirs))
            sampled.extend(rng.sample(dirs, quota))

        sampled.sort()  # deterministic order

        # Persist the sample so re-runs are stable.
        self.sample_path.write_text(
            json.dumps({
                "sample_size": self.sample_size,
                "seed": self.seed,
                "n_actual": len(sampled),
                "by_class": {k: floors[k] for k in floors},
                "problem_dirs": [d.name for d in sampled],
            }, indent=2),
            encoding="utf-8",
        )
        return sampled

    @staticmethod
    def _resolve_buggy_source(problem_dir: Path) -> tuple[Path | None, bool]:
        """Codeflaws dirs are named `<contest>-<problem>-bug-<buggy_rev>-<fixed_rev>`.

        The buggy source file is `<contest>-<problem>-<buggy_rev>.c` (or .cpp).
        We parse the buggy revision id from the dir name and look it up.
        """
        parts = problem_dir.name.split("-")
        # Expect at least: contest, problem, "bug", buggy_rev, fixed_rev
        if len(parts) < 5 or "bug" not in parts:
            return None, False
        try:
            bug_idx = parts.index("bug")
            contest = parts[0]
            problem_letter = parts[1]
            buggy_rev = parts[bug_idx + 1]
        except (ValueError, IndexError):
            return None, False

        basename = f"{contest}-{problem_letter}-{buggy_rev}"
        c_path = problem_dir / f"{basename}.c"
        cpp_path = problem_dir / f"{basename}.cpp"
        if c_path.is_file():
            return c_path, False
        if cpp_path.is_file():
            return cpp_path, True
        # Fallback: the buggy file is whichever .c is older (lower revision)
        # — find both source files and pick by name comparison.
        sources = sorted(
            f for f in problem_dir.iterdir()
            if f.is_file() and f.suffix in (".c", ".cpp") and not f.name.endswith(".revlog")
        )
        # Filter out non-revision files (Makefile.genprog etc.)
        sources = [s for s in sources if any(c.isdigit() for c in s.stem.split("-")[-1])]
        if sources:
            buggy = sources[0]
            return buggy, buggy.suffix == ".cpp"
        return None, False

    def problems(self) -> Iterator[Problem]:
        if not self.is_downloaded():
            raise RuntimeError(
                f"Codeflaws not downloaded; call download() first (root={self.root})"
            )

        defect_classes = self._read_defect_classes()
        for problem_dir in self._stratified_sample():
            buggy_path, is_cpp = self._resolve_buggy_source(problem_dir)
            if buggy_path is None:
                continue

            yield Problem(
                benchmark=self.name,
                problem_id=problem_dir.name,
                buggy_source_path=buggy_path,
                test_runner=_CodeflawsTestRunner(
                    problem_dir=problem_dir,
                    compile_command_template=[],  # populated lazily inside runner
                    is_cpp=is_cpp,
                ),
                metadata={
                    "defect_class": defect_classes.get(problem_dir.name, "_unclassified"),
                    "source_codeforces": problem_dir.name.split("-")[0],
                    "is_cpp": is_cpp,
                },
                source_extension=".cpp" if is_cpp else ".c",
                language_hint="C++17" if is_cpp else "C99",
            )
