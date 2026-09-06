"""Benchmark adapters for the agent-lifecycle pipeline.

Each benchmark lives in its own subpackage and exposes a `Benchmark` subclass
that yields `Problem` instances. The pipeline driver
iterates over those problems and feeds each into
`run_pipeline()` via the `problem` keyword argument.

Adding a new benchmark:
  1. Create a subpackage `benchmarks/<name>/`.
  2. Implement a `Benchmark` subclass with `name`, `version`, `download()`, and
     `problems()`.
  3. Register it in `benchmarks.registry.BENCHMARKS`.
"""
from .base import (
    Benchmark,
    FunctionHarnessRunner,
    Problem,
    PythonStdinStdoutRunner,
    StdinStdoutRunner,
    TestRunner,
)
from .registry import BENCHMARKS, get_benchmark

__all__ = [
    "Benchmark",
    "Problem",
    "TestRunner",
    "StdinStdoutRunner",
    "PythonStdinStdoutRunner",
    "FunctionHarnessRunner",
    "BENCHMARKS",
    "get_benchmark",
]
