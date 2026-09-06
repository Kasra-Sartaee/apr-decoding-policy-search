"""Low-overhead per-call inference-energy measurement (secondary telemetry).

Records the energy consumed by a single LLM generation as hardware-counter
*deltas* taken immediately around the call:

  * GPU: NVML cumulative energy counter (``nvmlDeviceGetTotalEnergyConsumption``,
    millijoules) — the dominant term, since the model runs on the GPU.
  * CPU package: Intel RAPL (``/sys/class/powercap/intel-rapl:0/energy_uj``) —
    the orchestration/sampling term. Linux + Intel only.

Each term is two counter reads (~microseconds); there is no power-sampling
thread, so overhead is negligible and the measurement cannot perturb the work it
measures. This is a SECONDARY, descriptive metric ("cost of generation"); it is
**not** the F3 optimisation objective, which is the repaired program's EXECUTION
energy. The two are measured in different windows (this one wraps the Ollama
call; F3 wraps the compiled program's test loop) and never interact.

Design notes:
  * Lazy, guarded, tri-state cached. If NVML or RAPL is unavailable on the host
    the corresponding term is ``None`` and measurement never raises into the
    pipeline.
  * Cross-platform by construction: on the Linux lab box both GPU and CPU terms
    populate; on a non-Intel/Windows box the CPU term is ``None`` (RAPL absent)
    while the GPU term still works if an NVIDIA GPU is present.
  * GPU reading is *device-level* (not per-process) — correct on a GPU
    dedicated to inference; note this when reporting on a shared GPU.
  * No idle subtraction: the value is energy *during the call window*. With
    ``keep_alive`` holding the model resident, between-call idle is outside the
    window. The first call after a cold model load includes load energy.

Unlike codecarbon (opt-in, heavier per-call tracker), this is a pure counter
delta and therefore enabled BY DEFAULT. Disable with ``AGENT_INFERENCE_ENERGY=0``.

Config:
  AGENT_INFERENCE_ENERGY=0   disable (default on)
  AGENT_GPU_INDEX=<n>        NVML device index (default 0)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


def _enabled() -> bool:
    return os.environ.get("AGENT_INFERENCE_ENERGY", "1") != "0"


# ---------------------------------------------------------------------------
# GPU — NVML cumulative energy counter
# ---------------------------------------------------------------------------
# Tri-state: None = not probed, False = unavailable, otherwise the device handle.
_NVML_STATE: Any = None
_NVML = None  # the imported pynvml module


def _gpu_handle():
    global _NVML_STATE, _NVML
    if _NVML_STATE is not None:
        return _NVML_STATE or None
    if not _enabled():
        _NVML_STATE = False
        return None
    try:
        import pynvml  # provided by the `nvidia-ml-py` package

        pynvml.nvmlInit()
        idx = int(os.environ.get("AGENT_GPU_INDEX", "0"))
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        # Probe the energy counter once; raises if the device/driver lacks it.
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        _NVML = pynvml
        _NVML_STATE = handle
        return handle
    except Exception:
        _NVML_STATE = False
        return None


def _read_gpu_mj() -> float | None:
    handle = _gpu_handle()
    if handle is None:
        return None
    try:
        return float(_NVML.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CPU package — Intel RAPL (reuses the validated reader from the energy backend)
# ---------------------------------------------------------------------------
# Tri-state: None = not probed, False = unavailable, otherwise a RaplSysfsBackend.
_RAPL: Any = None


def _rapl_backend():
    global _RAPL
    if _RAPL is not None:
        return _RAPL or None
    if not _enabled():
        _RAPL = False
        return None
    try:
        from energy.backends import RaplSysfsBackend

        backend = RaplSysfsBackend()
        if backend.is_available():
            _RAPL = backend
            return backend
    except Exception:
        pass
    _RAPL = False
    return None


def _read_cpu_uj() -> int | None:
    backend = _rapl_backend()
    if backend is None:
        return None
    try:
        return backend._read_counter()
    except Exception:
        return None


def _cpu_delta_mj(start_uj: int, end_uj: int) -> float:
    backend = _rapl_backend()
    delta_uj = end_uj - start_uj
    if delta_uj < 0 and backend is not None:
        # RAPL counter wrapped — add the range, same handling as the F3 backend.
        delta_uj += backend._read_max()
    return delta_uj / 1000.0  # microjoules -> millijoules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@contextmanager
def measure_inference_energy():
    """Context manager yielding a dict that, after the block, holds::

        {"gpu_mj": float | None, "cpu_mj": float | None}

    Energy consumed during the wrapped block, as counter deltas. Never raises —
    energy instrumentation must not break a generation.
    """
    holder: dict[str, float | None] = {"gpu_mj": None, "cpu_mj": None}
    gpu_start = _read_gpu_mj()
    cpu_start = _read_cpu_uj()
    try:
        yield holder
    finally:
        try:
            gpu_end = _read_gpu_mj()
            if gpu_start is not None and gpu_end is not None:
                holder["gpu_mj"] = max(0.0, gpu_end - gpu_start)
            cpu_end = _read_cpu_uj()
            if cpu_start is not None and cpu_end is not None:
                holder["cpu_mj"] = max(0.0, _cpu_delta_mj(cpu_start, cpu_end))
        except Exception:
            # Leave whatever was captured; never propagate.
            pass
