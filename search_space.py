"""Shared decoding-policy search space and vector->policy decoder.

Single source of truth imported by BOTH scripts/run_nsga2.py (NSGA-II) and
scripts/run_random_search.py (the LHS random-search ablation). Keeping the 11-D
space and the decoder in one module guarantees the two searches operate over the
*identical* space — a prerequisite for the random-search ablation to be a valid
like-for-like comparison. If each script defined its own copy they could silently
drift, invalidating the comparison with no error.

Temperature is SEARCHED (one dim per LLM agent: analyzer / repair / optimizer,
range 0-1) rather than pinned to 0; report 04 (findings F4/F5) showed temp-0
regressed compilation and confounded the baseline. The Verifier (pi_v) has no
temperature — it is not an LLM, only a retry budget.
"""
from __future__ import annotations

from typing import Any

SEARCH_SPACE: list[dict[str, Any]] = [
    {"path": ("pi_a", "temperature"), "low": 0.0, "high": 1.0, "dtype": "float"},
    {"path": ("pi_a", "top_p"), "low": 0.5, "high": 1.0, "dtype": "float"},
    {"path": ("pi_a", "top_k"), "low": 10, "high": 100, "dtype": "int"},
    {"path": ("pi_r", "temperature"), "low": 0.0, "high": 1.0, "dtype": "float"},
    {"path": ("pi_r", "top_p"), "low": 0.5, "high": 1.0, "dtype": "float"},
    {"path": ("pi_r", "top_k"), "low": 10, "high": 100, "dtype": "int"},
    {"path": ("pi_v", "retry"), "low": 1, "high": 8, "dtype": "int"},
    {"path": ("pi_o", "temperature"), "low": 0.0, "high": 1.0, "dtype": "float"},
    {"path": ("pi_o", "top_p"), "low": 0.5, "high": 1.0, "dtype": "float"},
    {"path": ("pi_o", "top_k"), "low": 10, "high": 100, "dtype": "int"},
    {"path": ("pi_o", "retry"), "low": 1, "high": 8, "dtype": "int"},
]


def build_policy_from_vector(vector) -> dict[str, Any]:
    """Convert a raw parameter vector (a pymoo chromosome or an LHS row) into a
    nested policy dict.

    Integer genes are rounded (the operators / samplers are continuous).
    ``float()`` coerces numpy scalars to Python floats so the policy serialises
    cleanly to JSON and to the Ollama options payload.
    """
    policy: dict[str, Any] = {}
    for i, descriptor in enumerate(SEARCH_SPACE):
        value = float(vector[i])
        if descriptor["dtype"] == "int":
            value = int(round(value))
        level = policy
        for key in descriptor["path"][:-1]:
            level = level.setdefault(key, {})
        level[descriptor["path"][-1]] = value
    # pi_v (Verifier) carries only a retry budget — ensure the block exists
    # even though it has no decoding params.
    policy.setdefault("pi_v", {})
    return policy
