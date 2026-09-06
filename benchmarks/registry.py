"""Benchmark registry — maps `--benchmark <name>` to adapter classes.

Adapters self-register here so the driver can resolve names without each
adapter being importable globally.
"""
from __future__ import annotations

from pathlib import Path

from .base import Benchmark

_HERE = Path(__file__).resolve().parent

# Lazy registry — adapter modules are imported on first lookup so the
# top-level package import stays cheap.
_REGISTRY: dict[str, str] = {
    "codeflaws": "benchmarks.codeflaws.adapter:CodeflawsBenchmark",
    "runbugrun_py": "benchmarks.runbugrun_py.adapter:RunBugRunPyBenchmark",
}

BENCHMARKS = list(_REGISTRY.keys())


def get_benchmark(name: str) -> Benchmark:
    """Resolve a benchmark name to an instantiated adapter."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    spec = _REGISTRY[name]
    module_path, class_name = spec.split(":")
    import importlib

    module = importlib.import_module(module_path)
    cls: type[Benchmark] = getattr(module, class_name)
    data_dir = _HERE / name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return cls(root=data_dir)
