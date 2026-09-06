"""Energy-backend selection: explicit-by-name or auto-pick by availability."""
from __future__ import annotations

from .backends import (
    AppleSiliconBackend,
    CodeCarbonBackend,
    EnergyBackend,
    RaplSysfsBackend,
    RuntimeProxyBackend,
    all_backends,
)

# Priority order from highest fidelity to lowest. Used by `auto` resolution.
# codecarbon outranks runtime_proxy because codecarbon's per-CPU-model TDP
# table is more accurate than our single repo-wide 15 W constant.
BACKEND_PRIORITY: list[str] = [
    "apple_silicon",
    "rapl_sysfs",
    "codecarbon",
    "runtime_proxy",
]


def _instantiate(name: str) -> EnergyBackend:
    mapping = {
        "apple_silicon": AppleSiliconBackend,
        "rapl_sysfs": RaplSysfsBackend,
        "codecarbon": CodeCarbonBackend,
        "runtime_proxy": RuntimeProxyBackend,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown energy backend {name!r}. "
            f"Choose from: {sorted(mapping.keys())} or 'auto'."
        )
    return mapping[name]()


def resolve_backend(choice: str = "auto") -> EnergyBackend:
    """Resolve a `--energy-backend` choice to a concrete backend.

    Choices:
      - "auto"           pick the highest-priority available backend
      - explicit name    return that backend, raise if unavailable

    Raises:
      ValueError if `choice` is an unknown name.
      RuntimeError if explicitly-requested backend is not available.
    """
    if choice == "auto":
        for be in all_backends():
            if be.is_available():
                return be
        # RuntimeProxy is always available, so this is unreachable.
        return RuntimeProxyBackend()

    be = _instantiate(choice)
    if not be.is_available():
        raise RuntimeError(
            f"Requested energy backend {choice!r} is not available on this host. "
            f"Install its dependency or use --energy-backend auto (or runtime_proxy)."
        )
    return be


# ---------------------------------------------------------------------------
# Active-backend singleton.
#
# The driver resolves a backend once at startup and registers it here so any
# component that needs to measure energy (the pipeline's fitness step, the
# benchmark TestRunners) can read the same instance without circular imports.
# Lazily auto-resolves on first read if the driver never set one (keeps
# legacy entry points working).
# ---------------------------------------------------------------------------
_ACTIVE_BACKEND: EnergyBackend | None = None


def set_active_backend(backend: EnergyBackend) -> None:
    """Register the backend the whole run should use for energy measurement."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def get_active_backend() -> EnergyBackend:
    """Return the registered backend, auto-resolving 'auto' on first use."""
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = resolve_backend("auto")
    return _ACTIVE_BACKEND
