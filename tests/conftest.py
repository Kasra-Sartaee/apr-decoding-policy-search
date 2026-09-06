"""Shared fixtures and corpus gating for the test suite.

Two benchmark corpora (Codeflaws, 3.1 GB; RunBugRun, 263 MB) are fetched on
first use rather than tracked, so tests that need them cannot run on a fresh
clone. Such tests carry the `integration` marker and skip when their corpus is
absent, which keeps the default `pytest tests` run green while leaving the full
suite available once the corpora are in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def requires_corpus(name: str) -> None:
    """Skip the calling test unless benchmark `name` is downloaded."""
    from benchmarks import get_benchmark

    try:
        bench = get_benchmark(name)
    except Exception as exc:  # registry or import failure is a real error
        raise RuntimeError(f"cannot construct benchmark {name!r}: {exc}") from exc
    if not bench.is_downloaded():
        pytest.skip(
            f"{name} corpus not downloaded; run `pytest -m integration` after "
            f"fetching it (see README, Tests)"
        )
