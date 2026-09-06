"""Cross-platform energy-measurement backends.

The pipeline measures F3 (energy consumption) via a pluggable backend.
Each backend implements `EnergyBackend` and reports per-window energy in
millijoules. Backends differ in fidelity:

- `apple_silicon`  — real, Apple M-series via zeus_apple_silicon (powermetrics)
- `rapl_sysfs`     — real, Linux + Intel/AMD RAPL via /sys/class/powercap/
- `codecarbon`     — real where possible, TDP-estimate fallback otherwise
- `runtime_proxy`  — wall_time_s × constant TDP, last-resort

See docs/guides/running_on_linux.md for the measurement setup.
"""
from .backends import (
    AppleSiliconBackend,
    CodeCarbonBackend,
    EnergyBackend,
    EnergyResult,
    RaplSysfsBackend,
    RuntimeProxyBackend,
)
from .detect import (
    BACKEND_PRIORITY,
    get_active_backend,
    resolve_backend,
    set_active_backend,
)

__all__ = [
    "EnergyBackend",
    "EnergyResult",
    "AppleSiliconBackend",
    "RaplSysfsBackend",
    "CodeCarbonBackend",
    "RuntimeProxyBackend",
    "BACKEND_PRIORITY",
    "resolve_backend",
    "set_active_backend",
    "get_active_backend",
]
