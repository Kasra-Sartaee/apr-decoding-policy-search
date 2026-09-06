"""Energy-measurement backends.

Each backend is a context-manager-style wrapper exposing:

    backend.is_available() -> bool          # cheap check at startup
    with backend.measure() as window:
        ...do work...
    window.energy_mj                        # millijoules consumed
    window.backend_name                     # stable label written to JSON

The backend abstraction lets `evaluate_fitness` call one interface
regardless of platform. The actual energy measurement quality is captured
explicitly in the `backend_name` label so downstream analysis can never
silently mix RAPL-real with TDP-proxy data.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------------

@dataclass
class EnergyResult:
    """One measurement window's outcome.

    Attributes:
        energy_mj: total energy in millijoules over the window. May be 0
            if the window was empty, the workload finished instantly, or
            the backend couldn't produce a reading (e.g. compile failed).
        wall_time_s: wall-clock duration of the window in seconds.
        backend_name: stable label, written to verification JSONs and
            results CSV so downstream analysis can distinguish measurement
            modes. One of:
              "apple_silicon", "rapl_sysfs", "codecarbon", "runtime_proxy",
              "none" (window failed to start)
        backend_subtype: backend-specific extra info ("rapl_pkg",
            "codecarbon_tdp_fallback", etc.). Optional; informational.
    """
    energy_mj: float
    wall_time_s: float
    backend_name: str
    backend_subtype: str = ""


# ----------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------

class EnergyBackend(ABC):
    """Interface every energy backend implements."""

    name: str = ""  # stable label, overridden per backend

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check: can this backend produce a reading on this host?
        Called once at pipeline startup. Should not raise."""

    @abstractmethod
    @contextmanager
    def measure(self) -> Iterator[_Window]:
        """Context manager. Inside the `with`, do the work. After exit,
        the window object has `energy_mj` populated.

        Example:
            with backend.measure() as w:
                subprocess.run([exe], ...)
            print(w.energy_mj)
        """

    def describe(self) -> str:
        """One-line human-readable description for the manifest."""
        return self.name


class _Window:
    """Mutable result holder yielded by `measure()`."""

    __slots__ = ("energy_mj", "wall_time_s", "backend_name", "backend_subtype")

    def __init__(self, backend_name: str, backend_subtype: str = "") -> None:
        self.energy_mj: float = 0.0
        self.wall_time_s: float = 0.0
        self.backend_name = backend_name
        self.backend_subtype = backend_subtype


# ----------------------------------------------------------------------------
# 1. Apple Silicon (zeus_apple_silicon)
# ----------------------------------------------------------------------------

class AppleSiliconBackend(EnergyBackend):
    """Real package energy on Apple M-series via zeus_apple_silicon.

    Requires:
      - macOS on Apple Silicon
      - `zeus_apple_silicon` installed (uses Apple's powermetrics under the hood)
      - sudo on first run for the powermetrics helper
    """
    name = "apple_silicon"

    def __init__(self) -> None:
        self._monitor: Any = None  # zeus_apple_silicon.AppleEnergyMonitor, imported lazily

    def is_available(self) -> bool:
        try:
            from zeus_apple_silicon import AppleEnergyMonitor  # noqa: F401
            return True
        except Exception:
            return False

    @contextmanager
    def measure(self) -> Iterator[_Window]:
        from zeus_apple_silicon import AppleEnergyMonitor
        if self._monitor is None:
            self._monitor = AppleEnergyMonitor()
        w = _Window(self.name, "powermetrics")
        t0 = time.perf_counter()
        self._monitor.begin_window("agent_fitness")
        try:
            yield w
        finally:
            result = self._monitor.end_window("agent_fitness")
            w.wall_time_s = time.perf_counter() - t0
            # zeus_apple_silicon exposes cpu_total_mj.
            w.energy_mj = float(getattr(result, "cpu_total_mj", 0.0) or 0.0)


# ----------------------------------------------------------------------------
# 2. Linux RAPL via /sys/class/powercap/
# ----------------------------------------------------------------------------

class RaplSysfsBackend(EnergyBackend):
    """Real package energy via Linux's powercap sysfs interface.

    Reads /sys/class/powercap/intel-rapl:0/energy_uj at window start and
    end. The counter is in microjoules and wraps at
    `max_energy_range_uj`; we handle the wrap explicitly. Works on:

      - Linux + Intel CPU with RAPL exposed (most server + laptop hardware)
      - Linux + AMD CPU with amd-energy module
      - NOT WSL2 (Hyper-V intercepts the model-specific registers RAPL reads)
    """
    name = "rapl_sysfs"

    ENERGY_FILE = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    RANGE_FILE = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")

    def __init__(self) -> None:
        self._max_uj: int | None = None

    def is_available(self) -> bool:
        try:
            if not self.ENERGY_FILE.is_file():
                return False
            # Must be readable as the current user. RAPL is often
            # root-only by default on modern distros (CVE-2020-8694).
            with self.ENERGY_FILE.open("r") as fh:
                int(fh.read().strip())
            return True
        except (OSError, ValueError, PermissionError):
            return False

    def _read_counter(self) -> int:
        with self.ENERGY_FILE.open("r") as fh:
            return int(fh.read().strip())

    def _read_max(self) -> int:
        if self._max_uj is None:
            try:
                with self.RANGE_FILE.open("r") as fh:
                    self._max_uj = int(fh.read().strip())
            except (OSError, ValueError):
                # Fallback: assume 32-bit wrap (common on consumer CPUs).
                self._max_uj = 2 ** 32
        return self._max_uj

    @contextmanager
    def measure(self) -> Iterator[_Window]:
        w = _Window(self.name, "intel-rapl:0 (package)")
        t0 = time.perf_counter()
        start_uj = self._read_counter()
        try:
            yield w
        finally:
            end_uj = self._read_counter()
            w.wall_time_s = time.perf_counter() - t0
            delta_uj = end_uj - start_uj
            if delta_uj < 0:
                # Counter wrapped.
                delta_uj += self._read_max()
            # microjoules -> millijoules
            w.energy_mj = delta_uj / 1000.0


# ----------------------------------------------------------------------------
# 3. CodeCarbon
# ----------------------------------------------------------------------------

class CodeCarbonBackend(EnergyBackend):
    """codecarbon-based measurement.

    codecarbon picks its own per-platform path:
      - Intel RAPL (Linux) — real
      - Apple powermetrics — real (requires sudo on first use)
      - NVIDIA GPU via NVML — real
      - Windows / unknown CPU — TDP-from-lookup-table proxy

    This means `backend_subtype` is captured carefully so downstream
    analysis can separate real-measurement rows from TDP-fallback rows.
    """
    name = "codecarbon"

    def __init__(self) -> None:
        self._tracker_cls = None

    def is_available(self) -> bool:
        try:
            from codecarbon import EmissionsTracker  # noqa: F401
            return True
        except Exception:
            return False

    @contextmanager
    def measure(self) -> Iterator[_Window]:
        from codecarbon import EmissionsTracker
        # Sub-process tracker — short-running, no file I/O, no logging.
        tracker = EmissionsTracker(
            measure_power_secs=1,
            tracking_mode="process",
            log_level="error",
            save_to_file=False,
            allow_multiple_runs=True,
        )
        # codecarbon tells us at startup whether it's using real RAPL or
        # the TDP fallback — capture that for the subtype label.
        subtype = "codecarbon_tdp_fallback"
        try:
            # codecarbon's internal `cpu` attribute records which mode it
            # picked. Accessed defensively because internals vary by version.
            cpu_obj = getattr(tracker, "_cpu_power", None) or getattr(tracker, "_hardware", None)
            if cpu_obj is not None:
                mode = type(cpu_obj).__name__
                if "Rapl" in mode or "rapl" in mode:
                    subtype = "codecarbon_rapl"
                elif "Apple" in mode or "Powermetrics" in mode:
                    subtype = "codecarbon_powermetrics"
        except Exception:
            pass

        w = _Window(self.name, subtype)
        t0 = time.perf_counter()
        tracker.start()
        try:
            yield w
        finally:
            try:
                emissions = tracker.stop()  # noqa: F841 (kg CO2; we use energy not emissions)
            except Exception:
                pass
            w.wall_time_s = time.perf_counter() - t0
            # codecarbon's final data is in tracker.final_emissions_data.
            try:
                data = tracker.final_emissions_data
                # data.cpu_energy is in kWh; convert kWh -> mJ.
                # 1 kWh = 3,600,000 J = 3,600,000,000 mJ.
                cpu_kwh = float(getattr(data, "cpu_energy", 0.0) or 0.0)
                w.energy_mj = cpu_kwh * 3_600_000_000.0
            except Exception:
                # If codecarbon couldn't produce a reading, fall back to 0.
                w.energy_mj = 0.0


# ----------------------------------------------------------------------------
# 4. Runtime proxy (last resort)
# ----------------------------------------------------------------------------

class RuntimeProxyBackend(EnergyBackend):
    """Wall-time × fixed TDP. NOT a measurement.

    This is the fallback when nothing real is available. It's explicitly
    labeled `runtime_proxy` so it cannot be confused with measured energy.

    The reported number is:
        energy_mj = wall_time_s × TDP_W × 1000

    where TDP_W is a single repo-level constant (default 15 W). For
    cross-platform comparability of the *proxy itself*, the constant must
    not vary between runs of the same dissertation.
    """
    name = "runtime_proxy"

    DEFAULT_TDP_W = 15.0

    def __init__(self, tdp_w: float | None = None) -> None:
        self.tdp_w = float(tdp_w if tdp_w is not None else os.environ.get("ENERGY_PROXY_TDP_W", self.DEFAULT_TDP_W))

    def is_available(self) -> bool:
        return True  # always available; that's the point

    @contextmanager
    def measure(self) -> Iterator[_Window]:
        w = _Window(self.name, f"tdp_{self.tdp_w}W")
        t0 = time.perf_counter()
        try:
            yield w
        finally:
            w.wall_time_s = time.perf_counter() - t0
            w.energy_mj = w.wall_time_s * self.tdp_w * 1000.0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def all_backends() -> list[EnergyBackend]:
    """Instantiate one of each backend, in priority order (highest first)."""
    return [
        AppleSiliconBackend(),
        RaplSysfsBackend(),
        CodeCarbonBackend(),
        RuntimeProxyBackend(),
    ]
