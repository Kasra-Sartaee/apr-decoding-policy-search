"""RunBugRun Python adapter — Python APR benchmark from competitive programming.

Source:   Böhme, van Tonder, Soremekun, Zeller (EMSE 2026),
          "RunBugRun — An Executable Dataset for Automated Program Repair".
DOI:      10.1007/s10664-025-10790-3
Data:     https://github.com/giganticode/run_bug_run_data (releases v0.0.1)
License:  CC-BY-4.0 per upstream README.

Dataset layout (GitHub releases, v0.0.1):
  python_train0.jsonl.gz  — 50 K bug records (buggy/fixed code, no tests)
  python_train1.jsonl.gz  — 50 K records
  python_train2.jsonl.gz  — 33 K records
  tests_all.jsonl.gz      — 321 K test cases (problem_id → input/output)

Each Python record has fields: id, buggy_submission_id, fixed_submission_id,
problem_id, user_id, buggy_code, fixed_code, labels, change_count, line_hunks.

Tests are stored separately in tests_all.jsonl.gz and joined via problem_id.
Median tests per problem is 101 and the FULL suite is used — see the note on
test subsampling below.

On first call to download(), the adapter:
  1. Streams tests_all.jsonl.gz once to count tests per problem (counts only).
  2. Fetches python_train0.jsonl.gz (50 K records; sufficient for 300-sample).
  3. Draws a stratified sample of SAMPLE_SIZE_DEFAULT records (by bug-label
     category), streams the test corpus a second time to load the full suites
     for just those problems, and writes per-problem directories:

    data/problems/<record_id>/
        buggy.py          — buggy Python source
        correct.py        — fixed Python source (reference; not used by pipeline)
        metadata.json     — {problem_id, labels, change_count, ...}
        input_1.txt  …    — stdin for test N  (1-indexed, capped at 10)
        expected_1.txt …  — expected stdout for test N

  4. Writes a _DOWNLOAD_OK sentinel.

Subsequent calls to download() return immediately (sentinel present).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..base import Benchmark, Problem, PythonStdinStdoutRunner

_BASE_URL = (
    "https://github.com/giganticode/run_bug_run_data/releases/download/v0.0.1"
)
TESTS_URL = f"{_BASE_URL}/tests_all.jsonl.gz"
# SHA-256 of the release assets actually used for the reported runs. The tag is
# immutable, so a mismatch means the asset was re-cut or the transfer was
# corrupted; either way the run would not be comparable to the published one.
ASSET_SHA256 = {
    "tests_all.jsonl.gz": "2fab824c5dd5244e1471d38655faea668fc6cd6a669cb4ce124d1fee7ce70d96",
    "python_train0.jsonl.gz": "6da3c23fe1fbc156a07a2c7fdbc2aed0c010ddaa1d087030936887e828e92659",
}
PYTHON_SPLITS = ["python_train0"]  # 50 K records — more than enough for 300-sample

# Tests are NOT subsampled. An earlier revision capped each problem at 10
# randomly-chosen tests to keep evaluation fast; that silently corrupted F2.
# Only ~3.7% of a problem's tests expose its bug on average (3.9 of ~104), so a
# 10-test sample missed the bug entirely for 21% of problems — they scored a
# perfect F2 while still being broken — and inflated F2 for the rest. Verified
# by re-running all 42 such problems against their complete suites: 42/42 fail.
# Test execution is a rounding error next to the LLM calls that dominate a GA
# evaluation, so the cap bought nothing and cost measurement validity.

# The timeout ACTUALLY in force for RunBugRun. problems() constructs the runner
# with this value, so it is what every run really uses. Exported so the protocol
# manifest records the truth rather than a driver's nominal config value — the
# C arm's manifest previously advertised a compile_timeout that was never active.
# There is no compile timeout: Python is interpreted.
RUNBUGRUN_RUN_TIMEOUT_S = 5

SAMPLE_SIZE_DEFAULT = 300
SAMPLE_SEED = 42  # stable; documented in README


class RunBugRunPyBenchmark(Benchmark):
    name = "runbugrun_py"
    version = "github-giganticode-run_bug_run_data-v0.0.1"

    def __init__(
        self,
        root: Path,
        sample_size: int = SAMPLE_SIZE_DEFAULT,
        seed: int = SAMPLE_SEED,
    ) -> None:
        super().__init__(root)
        self.sample_size = sample_size
        self.seed = seed
        self.problems_root = self.root / "problems"
        self.sample_path = self.root / f"sample_{sample_size}_seed{seed}.json"

    # ── download ────────────────────────────────────────────────────────────

    def is_downloaded(self) -> bool:
        return (self.root / "_DOWNLOAD_OK").is_file()

    def download(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.problems_root.mkdir(parents=True, exist_ok=True)

        if not self.sample_path.is_file():
            self._build_sample_dirs()

        (self.root / "_DOWNLOAD_OK").write_text(
            json.dumps({"version": self.version}), encoding="utf-8"
        )

    def _fetch_gz(self, url: str, dest: Path) -> Path:
        """Download `url` to `dest` once, verified against ASSET_SHA256.

        Downloads land on a temporary path and are renamed only after the digest
        matches, so an interrupted transfer leaves no file that the `is_file()`
        check on the next call would mistake for a complete one.
        """
        if dest.is_file():
            if self._digest_ok(dest):
                return dest
            print(f"[runbugrun_py] {dest.name} failed its digest check; re-downloading")
            dest.unlink()

        print(f"[runbugrun_py] downloading {url} -> {dest}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if not self._digest_ok(tmp):
            got = hashlib.sha256(tmp.read_bytes()).hexdigest()
            tmp.unlink()
            raise RuntimeError(
                f"{dest.name} digest {got} does not match the expected "
                f"{ASSET_SHA256[dest.name]}; the download was discarded."
            )
        tmp.replace(dest)
        return dest

    @staticmethod
    def _digest_ok(path: Path) -> bool:
        expected = ASSET_SHA256.get(path.name.removesuffix(".part"))
        if expected is None:
            return True  # asset not pinned; nothing to check against
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest() == expected

    def _scan_test_counts(self, archive: Path) -> dict[str, int]:
        """Pass 1 over tests_all.jsonl.gz: problem_id -> number of test cases.

        Only counts are held in memory. The full corpus is ~321 K test cases
        whose payloads would be hundreds of MB; we defer loading the actual
        inputs/outputs to pass 2, which reads only the sampled problems.
        """
        print("[runbugrun_py] scanning test corpus…")
        counts: dict[str, int] = defaultdict(int)
        with gzip.open(archive, "rt", encoding="utf-8") as fh:
            for line in fh:
                counts[json.loads(line)["problem_id"]] += 1
        print(f"[runbugrun_py] {len(counts)} problems have tests")
        return counts

    def _load_tests_for(self, archive: Path, wanted: set[str]) -> dict[str, list[dict]]:
        """Pass 2: load every test case for the sampled problems only.

        Test order is the upstream file order — deterministic, and no longer
        subsampled, so F2 is the true pass rate over the complete suite.
        """
        print(f"[runbugrun_py] loading tests for {len(wanted)} sampled problems…")
        out: dict[str, list[dict]] = defaultdict(list)
        with gzip.open(archive, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                pid = rec["problem_id"]
                if pid in wanted:
                    out[pid].append({"input": rec["input"], "output": rec["output"]})
        return out

    def _build_sample_dirs(self) -> None:
        """Fetch train JSONL(s), draw stratified sample, write problem dirs."""
        tests_archive = self._fetch_gz(TESTS_URL, self.root / "tests_all.jsonl.gz")
        test_counts = self._scan_test_counts(tests_archive)

        # Collect all candidate records (only those with ≥1 test).
        groups: dict[str, list[dict]] = defaultdict(list)
        for split in PYTHON_SPLITS:
            archive = self._fetch_gz(
                f"{_BASE_URL}/{split}.jsonl.gz", self.root / f"{split}.jsonl.gz"
            )
            print(f"[runbugrun_py] scanning {split}…")
            with gzip.open(archive, "rt", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line.strip())
                    if not test_counts.get(rec["problem_id"]):
                        continue
                    # Stratification key = first label category (e.g. "literal",
                    # "expression", "call", "io") for bug-type diversity.
                    raw_labels = rec.get("labels", [])
                    if isinstance(raw_labels, str):
                        try:
                            raw_labels = json.loads(raw_labels)
                        except Exception:
                            raw_labels = [raw_labels]
                    cat = (
                        raw_labels[0].split(".")[0] if raw_labels else "_unlabelled"
                    )
                    groups[cat].append(rec)

        total_pool = sum(len(v) for v in groups.values())
        if total_pool == 0:
            raise RuntimeError(
                "RunBugRun: no usable records found. "
                "Check that tests_all.jsonl.gz was downloaded correctly."
            )

        # Proportional stratified sample with largest-remainder rounding.
        rng = random.Random(self.seed)
        raw_quotas = {
            k: (len(v) / total_pool) * self.sample_size for k, v in groups.items()
        }
        floors = {k: int(q) for k, q in raw_quotas.items()}
        remainder = self.sample_size - sum(floors.values())
        fractions = sorted(
            ((raw_quotas[k] - floors[k], k) for k in raw_quotas), reverse=True
        )
        for _, k in fractions[:remainder]:
            floors[k] += 1

        sampled: list[dict] = []
        for cat, recs in groups.items():
            quota = min(floors[cat], len(recs))
            sampled.extend(rng.sample(recs, quota))
        sampled.sort(key=lambda r: r["id"])

        by_cat = {k: floors[k] for k in floors}
        print(f"[runbugrun_py] sample: {len(sampled)} records ({by_cat})")

        # Pass 2 — load the full test suites for just the sampled problems.
        tests_by_pid = self._load_tests_for(
            tests_archive, {r["problem_id"] for r in sampled}
        )

        # Write per-problem directories.
        print("[runbugrun_py] writing problem directories…")
        problem_dirs = []
        n_tests_written = 0
        for rec in sampled:
            record_id = str(rec["id"])
            tests = tests_by_pid[rec["problem_id"]]
            n_tests_written += len(tests)
            pdir = self.problems_root / record_id
            pdir.mkdir(exist_ok=True)

            (pdir / "buggy.py").write_text(rec["buggy_code"], encoding="utf-8")
            (pdir / "correct.py").write_text(rec["fixed_code"], encoding="utf-8")

            raw_labels = rec.get("labels", [])
            if isinstance(raw_labels, str):
                try:
                    raw_labels = json.loads(raw_labels)
                except Exception:
                    raw_labels = [raw_labels]

            meta = {
                "problem_id": rec["problem_id"],
                "labels": raw_labels,
                "change_count": rec.get("change_count"),
                "line_hunks": rec.get("line_hunks"),
            }
            (pdir / "metadata.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )

            for i, t in enumerate(tests, start=1):
                (pdir / f"input_{i}.txt").write_text(t["input"], encoding="utf-8")
                (pdir / f"expected_{i}.txt").write_text(t["output"], encoding="utf-8")

            problem_dirs.append(record_id)

        self.sample_path.write_text(
            json.dumps(
                {
                    "sample_size": self.sample_size,
                    "seed": self.seed,
                    "n_actual": len(sampled),
                    "by_category": by_cat,
                    "tests_per_problem": "uncapped (full upstream suite)",
                    "n_tests_total": n_tests_written,
                    "problem_dirs": problem_dirs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[runbugrun_py] {len(sampled)} problems ready under {self.problems_root} "
            f"({n_tests_written} tests total, "
            f"{n_tests_written / max(len(sampled), 1):.0f} per problem)"
        )

    # ── sampling ────────────────────────────────────────────────────────────

    def _stratified_sample(self) -> list[Path]:
        saved = json.loads(self.sample_path.read_text(encoding="utf-8"))
        return [
            self.problems_root / d
            for d in saved["problem_dirs"]
            if (self.problems_root / d).is_dir()
        ]

    # ── problems() ──────────────────────────────────────────────────────────

    def problems(self) -> Iterator[Problem]:
        if not self.is_downloaded():
            raise RuntimeError(
                f"RunBugRun not downloaded; call download() first (root={self.root})"
            )

        for pdir in self._stratified_sample():
            meta_path = pdir / "metadata.json"
            meta: dict = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            yield Problem(
                benchmark=self.name,
                problem_id=pdir.name,
                buggy_source_path=pdir / "buggy.py",
                test_runner=PythonStdinStdoutRunner(
                    problem_dir=pdir,
                    input_glob="input_*.txt",
                    expected_glob="expected_*.txt",
                    run_timeout=RUNBUGRUN_RUN_TIMEOUT_S,
                ),
                metadata=meta,
                source_extension=".py",
                language_hint="Python3",
                harness_concatenated=False,
            )
