"""Per-LLM-call telemetry: tokens, wall-time, optional CO2eq.

Module-level accumulator. The pipeline calls `reset()` at the start of
each problem, `set_stage("analyze"|"repair"|"verify"|"optimize")` at
each stage boundary, and `summary()` at the end to flatten the
collected calls into per-stage aggregates suitable for the per-problem
CSV row and the verification JSON.

Keeping this separate from `agents.py` leaves `AnalyzerAgent`,
`RepairAgent` and `OptimizerAgent` agnostic to how telemetry is collected: they
call `record_llm_call()` after each Ollama response, and the pipeline driver
owns reset and aggregation.

CO2eq capture: opt-in via `AGENT_CODECARBON=1` environment variable.
When set, every recorded LLM call is wrapped in a codecarbon
OfflineEmissionsTracker that reports kg_co2eq for that single call.
On Linux+RAPL this is a real measurement; on Windows or macOS-Intel
it's a TDP-fallback proxy (codecarbon falls back automatically). The
backend in use is recorded per-call so downstream analysis can filter
proxy rows from measured rows.

The codecarbon import is *lazy and guarded* — if the library isn't
installed, the env var is silently ignored. This is intentional so
Windows users can run the pipeline without installing codecarbon.
"""
from __future__ import annotations

import os
import threading as _threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# codecarbon lazy import
# ---------------------------------------------------------------------------
_CODECARBON_AVAILABLE: bool | None = None  # tri-state: None = not probed yet
_CODECARBON_TRACKER_CLS = None
_CODECARBON_BACKEND_LABEL: str = ""


def _probe_codecarbon() -> bool:
    """Lazily import codecarbon and remember the result.

    Returns True iff the library imports successfully *and* the env
    flag `AGENT_CODECARBON=1` is set. We honor both conditions so
    machines that have codecarbon installed don't pay the per-call
    overhead unless the user explicitly opts in.
    """
    global _CODECARBON_AVAILABLE, _CODECARBON_TRACKER_CLS, _CODECARBON_BACKEND_LABEL
    if _CODECARBON_AVAILABLE is not None:
        return _CODECARBON_AVAILABLE
    if os.environ.get("AGENT_CODECARBON", "0") != "1":
        _CODECARBON_AVAILABLE = False
        return False
    try:
        from codecarbon import OfflineEmissionsTracker  # type: ignore
        _CODECARBON_TRACKER_CLS = OfflineEmissionsTracker
        _CODECARBON_AVAILABLE = True
        # Best-effort backend identification — the same probing
        # codecarbon does internally. Defaults to "tdp_fallback" on
        # Windows; "rapl_sysfs" on Linux with /sys/class/powercap.
        if os.name == "posix":
            if os.path.isdir("/sys/class/powercap/intel-rapl"):
                _CODECARBON_BACKEND_LABEL = "codecarbon_rapl_sysfs"
            else:
                _CODECARBON_BACKEND_LABEL = "codecarbon_tdp_fallback"
        else:
            _CODECARBON_BACKEND_LABEL = "codecarbon_tdp_fallback"
    except Exception:
        _CODECARBON_AVAILABLE = False
    return _CODECARBON_AVAILABLE


@contextmanager
def _measure_co2():
    """Wrap a block; on exit, yield kg_co2eq via .value.

    No-op (yields None) when codecarbon is disabled. Always silently
    swallows codecarbon's own errors — instrumentation must never
    break the pipeline.
    """
    holder = {"value": None}
    if not _probe_codecarbon():
        yield holder
        return
    try:
        # Quiet, country-set-via-env for offline use; log_level=error
        # suppresses the chatty INFO output codecarbon emits per call.
        tracker = _CODECARBON_TRACKER_CLS(
            country_iso_code=os.environ.get("AGENT_CO2_COUNTRY", "USA"),
            log_level="error",
            save_to_file=False,
            tracking_mode="process",
        )
        tracker.start()
        try:
            yield holder
        finally:
            try:
                emissions = tracker.stop()
                if emissions is not None:
                    holder["value"] = float(emissions)
            except Exception:
                pass
    except Exception:
        # Fail open — the LLM call must succeed regardless.
        yield holder


# ---------------------------------------------------------------------------
# Per-call records
# ---------------------------------------------------------------------------
@dataclass
class LLMCall:
    """One LLM invocation's telemetry."""
    stage: str               # "analyze" / "repair" / "verify" / "optimize" / "unknown"
    model: str
    wall_time_s: float
    prompt_tokens: int       # `prompt_eval_count` from Ollama, 0 if unknown
    completion_tokens: int   # `eval_count` from Ollama, 0 if unknown
    co2_kg: float | None     # None if codecarbon disabled
    co2_backend: str         # "" if disabled, otherwise see _CODECARBON_BACKEND_LABEL
    # Inference energy of this call (counter deltas; see inference_energy.py).
    # Secondary descriptive metric, NOT the F3 objective. None if unavailable.
    gpu_energy_mj: float | None = None   # NVML GPU energy during the call
    cpu_energy_mj: float | None = None   # RAPL CPU-package energy during the call


@dataclass
class TelemetryAccumulator:
    """Per-problem accumulator. Reset between problems."""
    calls: list[LLMCall] = field(default_factory=list)
    current_stage: str = "unknown"


# Per-THREAD accumulator. Under N_WORKERS>1 the Phase A worker pool runs
# run_pipeline_phase_a() concurrently, and each call begins with
# telemetry.reset(); with a single shared module-level accumulator one
# worker's reset wiped another's in-flight calls, and record_llm_call()
# from different problems interleaved into the same list. GA fitness never
# reads telemetry so objectives were unaffected, but the per-problem
# token/wall-time/CO2 figures written into the verification JSONs were not
# attributable to the problem they were filed under. Thread-local storage
# gives each worker its own accumulator; the sequential drivers
# (single-worker drivers, N_WORKERS=1) are unaffected since they only ever use
# the main thread.
_TL = _threading.local()


def _acc() -> TelemetryAccumulator:
    a = getattr(_TL, "acc", None)
    if a is None:
        a = TelemetryAccumulator()
        _TL.acc = a
    return a


def reset() -> None:
    """Clear THIS THREAD's accumulator. Driver calls this at problem start."""
    _TL.acc = TelemetryAccumulator()


def set_stage(stage: str) -> None:
    """Label the next record_llm_call() invocations with this stage."""
    _acc().current_stage = stage


def record_llm_call(
    model: str,
    wall_time_s: float,
    prompt_tokens: int,
    completion_tokens: int,
    co2_kg: float | None,
    co2_backend: str,
    gpu_energy_mj: float | None = None,
    cpu_energy_mj: float | None = None,
) -> None:
    """Append one LLMCall to the active accumulator. Called by agents.py."""
    _acc().calls.append(
        LLMCall(
            stage=_acc().current_stage,
            model=model,
            wall_time_s=float(wall_time_s),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            co2_kg=(None if co2_kg is None else float(co2_kg)),
            co2_backend=co2_backend,
            gpu_energy_mj=(None if gpu_energy_mj is None else float(gpu_energy_mj)),
            cpu_energy_mj=(None if cpu_energy_mj is None else float(cpu_energy_mj)),
        )
    )


def co2_capture():
    """Context manager: codecarbon wrapper around a block. Use as

        with telemetry.co2_capture() as co2:
            ... do work ...
        co2_value = co2["value"]  # kg_co2eq or None
    """
    return _measure_co2()


def co2_backend_label() -> str:
    """Empty string if codecarbon is disabled, else descriptive label."""
    _probe_codecarbon()
    return _CODECARBON_BACKEND_LABEL


# ---------------------------------------------------------------------------
# Aggregation for the per-problem CSV row + verification JSON
# ---------------------------------------------------------------------------
def summary() -> dict[str, Any]:
    """Aggregate the accumulated calls into a flat dict for CSV + JSON.

    Returns keys:
      total_llm_calls
      total_wall_time_s_llm
      total_prompt_tokens
      total_completion_tokens
      total_tokens
      total_co2_kg              (None if codecarbon disabled)
      co2_backend               ("" if disabled)
      total_gpu_energy_mj       inference GPU energy (None if NVML unavailable)
      total_cpu_energy_mj       inference CPU-package energy (None if no RAPL)
      total_inference_energy_mj gpu+cpu (None if neither available)
      per_stage                 {stage: {n_calls, wall_time_s, prompt_tokens,
                                          completion_tokens, co2_kg,
                                          gpu_energy_mj, cpu_energy_mj}}

    The pipeline flattens `per_stage` into prefixed CSV columns
    (e.g. `llm_analyze_wall_s`, `llm_repair_tokens`) at the write step.
    """
    calls = _acc().calls
    out: dict[str, Any] = {
        "total_llm_calls": len(calls),
        "total_wall_time_s_llm": sum(c.wall_time_s for c in calls),
        "total_prompt_tokens": sum(c.prompt_tokens for c in calls),
        "total_completion_tokens": sum(c.completion_tokens for c in calls),
        "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in calls),
        "co2_backend": co2_backend_label(),
    }
    any_co2 = any(c.co2_kg is not None for c in calls)
    out["total_co2_kg"] = (
        sum(c.co2_kg or 0.0 for c in calls if c.co2_kg is not None)
        if any_co2 else None
    )

    # Inference energy (secondary metric). None when the host exposes no
    # counter for that domain; otherwise summed over all calls.
    any_gpu = any(c.gpu_energy_mj is not None for c in calls)
    any_cpu = any(c.cpu_energy_mj is not None for c in calls)
    out["total_gpu_energy_mj"] = (
        sum(c.gpu_energy_mj or 0.0 for c in calls if c.gpu_energy_mj is not None)
        if any_gpu else None
    )
    out["total_cpu_energy_mj"] = (
        sum(c.cpu_energy_mj or 0.0 for c in calls if c.cpu_energy_mj is not None)
        if any_cpu else None
    )
    out["total_inference_energy_mj"] = (
        (out["total_gpu_energy_mj"] or 0.0) + (out["total_cpu_energy_mj"] or 0.0)
        if (any_gpu or any_cpu) else None
    )

    per_stage: dict[str, dict[str, Any]] = {}
    for c in calls:
        s = per_stage.setdefault(c.stage, {
            "n_calls": 0,
            "wall_time_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "co2_kg": None,
            "gpu_energy_mj": None,
            "cpu_energy_mj": None,
        })
        s["n_calls"] += 1
        s["wall_time_s"] += c.wall_time_s
        s["prompt_tokens"] += c.prompt_tokens
        s["completion_tokens"] += c.completion_tokens
        if c.co2_kg is not None:
            s["co2_kg"] = (s["co2_kg"] or 0.0) + c.co2_kg
        if c.gpu_energy_mj is not None:
            s["gpu_energy_mj"] = (s["gpu_energy_mj"] or 0.0) + c.gpu_energy_mj
        if c.cpu_energy_mj is not None:
            s["cpu_energy_mj"] = (s["cpu_energy_mj"] or 0.0) + c.cpu_energy_mj
    out["per_stage"] = per_stage
    return out


def flat_csv_columns(summ: dict[str, Any]) -> dict[str, Any]:
    """Flatten summary() output into CSV-row-friendly columns.

    Adds per-stage prefixed keys: llm_<stage>_n_calls, llm_<stage>_wall_s,
    llm_<stage>_tokens, llm_<stage>_co2_kg. Stages with no calls are
    omitted (CSV will have NaN for them — pandas handles missing
    columns cleanly across rows).
    """
    out: dict[str, Any] = {
        "llm_total_calls": summ["total_llm_calls"],
        "llm_total_wall_s": round(summ["total_wall_time_s_llm"], 3),
        "llm_total_prompt_tokens": summ["total_prompt_tokens"],
        "llm_total_completion_tokens": summ["total_completion_tokens"],
        "llm_total_tokens": summ["total_tokens"],
        "llm_total_co2_kg": summ["total_co2_kg"],
        "llm_co2_backend": summ["co2_backend"],
        "llm_total_gpu_energy_mj": summ.get("total_gpu_energy_mj"),
        "llm_total_cpu_energy_mj": summ.get("total_cpu_energy_mj"),
        "llm_total_inference_energy_mj": summ.get("total_inference_energy_mj"),
    }
    for stage, s in summ["per_stage"].items():
        out[f"llm_{stage}_n_calls"] = s["n_calls"]
        out[f"llm_{stage}_wall_s"] = round(s["wall_time_s"], 3)
        out[f"llm_{stage}_tokens"] = s["prompt_tokens"] + s["completion_tokens"]
        out[f"llm_{stage}_co2_kg"] = s["co2_kg"]
        out[f"llm_{stage}_gpu_energy_mj"] = s.get("gpu_energy_mj")
        out[f"llm_{stage}_cpu_energy_mj"] = s.get("cpu_energy_mj")
    return out
