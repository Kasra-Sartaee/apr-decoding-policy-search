"""Run provenance: what code and what treatment produced a given result.

Shared by every GA driver so the recorded provenance cannot drift between
them. `scripts/run_nsga2.py` (C, two-phase RAPL) and `scripts/run_nsga2_runtime.py`
(cross-benchmark, two-phase) both write `protocol.json` through here.

The point of this file is that a results directory should answer "which code,
which prompts, which settings produced this?" without relying on anyone
remembering. `run_manifest.json` is overwritten by whichever process wrote
last, so it cannot distinguish a rep whose generations spanned a code change;
`protocol.json` is append-only and can.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent


def git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# Paths a run WRITES. Untracked entries here are the experiment's own output,
# not un-committed code, so they say nothing about whether `git_commit`
# identifies the treatment.
_RUN_ARTIFACT_PREFIXES = (
    "outputs/",
    "logs/",
)


def git_is_dirty() -> bool:
    """True if uncommitted CODE could have produced this run's data.

    Counts any tracked modification, and any untracked file that is not run
    output. A plain `git status --porcelain` check does not work here: a run
    creates its own untracked output directory as it goes, so the flag was
    True for every run including ones launched from a perfectly clean tree —
    which made it useless for the thing it exists to detect.

    An untracked *source* file still counts: a new module that never got
    committed is exactly the case where `git_commit` fails to identify the
    treatment.

    Failure to run git at all is treated as dirty/unknown, to fail safe.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(REPO),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return True
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            status, _, path = line.partition(" ")
            path = path.strip().strip('"')
            if line.startswith("??"):
                path = line[2:].strip().strip('"')
                if path.startswith(_RUN_ARTIFACT_PREFIXES):
                    continue          # the run's own output
            return True
        return False
    except Exception:
        return True


def contract_hashes(language: str = "C99") -> dict[str, str]:
    """SHA-256 of the prompt contracts ACTUALLY IN FORCE for this run.

    The contracts are part of the experimental treatment, not just
    configuration: a 1.5B model was observed copying a worked example out of
    the repair contract into its output ~19% of the time (0% at 7B), which
    inflated F1 by dead code. Recording the hashes makes "which prompt
    produced this data" answerable after the fact, and makes a contract edit
    visible as a treatment change rather than an invisible one.

    Scoped BY LANGUAGE, not globbed. An earlier version hashed every `*.md` in
    `contracts/`, so merely adding a Python contract changed the recorded
    treatment of every C run — and since `contract_sha256` is a treatment key
    in the resume comparison, resuming an untouched Codeflaws repetition would
    report a spurious "treatment changed" and append a bogus `resumed_under`
    entry. A contract that a run never loads is not part of that run's
    treatment.
    """
    from agents import (
        ANALYZER_CONTRACT_PATH,
        OPTIMISER_CONTRACT_PATH,
        REPAIR_CONTRACT_PATH,
        _select_contract,
    )

    out: dict[str, str] = {}
    for base in (ANALYZER_CONTRACT_PATH, REPAIR_CONTRACT_PATH, OPTIMISER_CONTRACT_PATH):
        path = _select_contract(language, base)
        try:
            out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            out[path.name] = "unreadable"
    return out


# Treatment keys compared on resume. A difference in any of these means the
# repetition spans a change of experiment, not merely a restart.
TREATMENT_KEYS = (
    "git_commit", "measurement_protocol", "contract_sha256",
    "model", "pop", "gen", "train_size", "n_workers", "seed",
)


def write_protocol_manifest(
    rep_dir: Path,
    *,
    measurement_protocol: str,
    protocol_doc: str,
    language: str,
    n_workers: int,
    timeouts_effective: dict,
    seed: int,
    seed_increment: int,
    model: str,
    pop: int,
    gen: int,
    train_size: int,
    extra: dict | None = None,
) -> None:
    """Write (or append to) `rep_dir/protocol.json`.

    NEVER overwrites an existing file. This runs before the checkpoint is
    loaded, so overwriting would replace the provenance of generations already
    computed — making a repetition whose generations spanned a code change look
    as though all of it ran under the latest commit. The original record is
    preserved and each resume appended, so a mixed-protocol repetition stays
    visible.
    """
    path = rep_dir / "protocol.json"
    manifest = {
        "measurement_protocol": measurement_protocol,
        "protocol_doc": protocol_doc,
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "contract_sha256": contract_hashes(language),
        "contract_language": language,
        "written_at": datetime.now(UTC).isoformat(),
        "n_workers": n_workers,
        "timeouts_effective": timeouts_effective,
        "seed": seed,
        "seed_increment": seed_increment,
        "model": model,
        "pop": pop,
        "gen": gen,
        "train_size": train_size,
    }
    if extra:
        manifest.update(extra)

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, dict):
            # Compare the whole treatment, not just the commit. A changed
            # contract, model, population, generation count, training size or
            # worker count under the SAME commit is a different experiment and
            # must not be silently folded into one repetition.
            if any(existing.get(k) != manifest.get(k) for k in TREATMENT_KEYS):
                existing.setdefault("resumed_under", []).append(manifest)
                diff = [k for k in TREATMENT_KEYS if existing.get(k) != manifest.get(k)]
                print(
                    f"[provenance] WARNING: treatment changed since this repetition "
                    f"started (differs in: {diff}). {path} was written under commit "
                    f"{str(existing.get('git_commit'))[:9]} / "
                    f"{existing.get('measurement_protocol')}, but this process is "
                    f"{manifest['git_commit'][:9]} / {manifest['measurement_protocol']}. "
                    f"This repetition spans a code change — recorded in 'resumed_under'."
                )
            else:
                existing.setdefault("resumed_under", []).append(
                    {"written_at": manifest["written_at"], "note": "resumed, same commit"}
                )
            path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            return
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
