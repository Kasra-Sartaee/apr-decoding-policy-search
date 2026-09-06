import glob
import json
import os as _os
import subprocess
import tempfile
import textwrap
import threading as _threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import ollama as _ollama
from ollama import ChatResponse

import telemetry
from inference_energy import measure_inference_energy

# Hard timeout for every Ollama HTTP call. If a generation stalls (context
# overflow, GPU swap, driver freeze) this breaks the hung call so the
# per-problem try/except in _mean_fitness_over can score it 0 and continue,
# rather than blocking the entire run indefinitely. Overridable via env var.
_LLM_HTTP_TIMEOUT = float(_os.environ.get("OLLAMA_REQUEST_TIMEOUT", "300"))
_ollama_client = _ollama.Client(timeout=_LLM_HTTP_TIMEOUT)

# ── Inference / verification gate ─────────────────────────────────────────────
# Linux RAPL sysfs counters accumulate energy for the ENTIRE CPU package (all
# cores, uncore, DRAM) — there is no way to scope a reading to one process.
#
# A plain "only one verify() at a time" lock (the previous _RAPL_LOCK) is
# NOT sufficient with a local Ollama backend: while one thread holds the lock
# and runs its test suite inside a RAPL window, OTHER threads are still free
# to run their analyze/repair/optimize LLM calls — and a local Ollama call
# drives real local CPU/GPU load. That inference load bleeds into the "isolated"
# verify() window and inflates/corrupts its F3 reading. Confirmed empirically:
# a second n50 GA repetition run with N_WORKERS=4 against local Ollama showed
# a 3.8x higher mean F3 than the N_WORKERS=1 baseline with zero range overlap.
#
# Fix: a readers-writer gate between local inference and verification.
#   - Local Ollama chat() calls are READERS — any number may run concurrently.
#   - The RAPL-measured block in VerifierAgent.verify() is the sole WRITER —
#     before opening its measurement window it blocks new inference from
#     starting and waits for all inference currently in flight to finish, so
#     the whole-CPU-package counter sees nothing but the verified program's
#     own execution. It then reopens the gate so inference resumes.
# This reproduces, per-thread-pool, exactly the "infer in batches, pause for
# verification, then resume" behaviour requested, without needing to
# restructure the per-problem repair/optimize retry loop (which genuinely
# needs verify() feedback interleaved between repair attempts).
#
# For N_WORKERS=1 the gate is never contested and costs nanoseconds.
from contextlib import contextmanager as _contextmanager


class _InferenceVerifyGate:
	"""Readers-writer barrier: local LLM inference (readers) vs. the
	RAPL-measured verification window (sole writer). Strict FIFO ticket
	ordering — a request (read or write) only waits behind requests that
	arrived strictly before it, regardless of kind.

	Two simpler designs were tried and rejected empirically:
	  - Preemptive writer-priority (a pending writer immediately blocks ALL
	    new reads) created a convoy: once verify() calls hold the writer
	    role for much longer (a whole test suite, see _energy_window()'s
	    docstring), multiple threads' verify() calls cluster in time (four
	    workers reach similar pipeline stages together), and the "block new
	    reads" condition stayed latched until the entire queue of writers
	    drained — measured slower than plain sequential (~390s vs ~230s on
	    a 20-problem smoke test).
	  - No priority at all (writers just wait for a natural gap) removed the
	    convoy but let busy, overlapping reader traffic starve a waiting
	    writer for a long time (observed up to 70s drain-wait on one write).
	FIFO ordering avoids both: a writer can't be perpetually cut in line by
	readers that arrive after it (bounding its wait), but readers that
	arrived before that writer are completely unaffected by it (no
	convoy) — new readers only ever wait behind requests strictly older
	than themselves.
	"""

	def __init__(self) -> None:
		self._cv = _threading.Condition()
		self._active_readers = 0
		self._writer_active = False
		self._next_ticket = 0
		self._pending: list[tuple[int, str]] = []  # (ticket, "r"|"w"), arrival order

	def _wait_turn(self, kind: str) -> int:
		with self._cv:
			ticket = self._next_ticket
			self._next_ticket += 1
			self._pending.append((ticket, kind))
			while True:
				ahead = [k for (t, k) in self._pending if t < ticket]
				if kind == "r":
					ready = not self._writer_active and "w" not in ahead
				else:
					ready = not self._writer_active and self._active_readers == 0 and not ahead
				if ready:
					break
				self._cv.wait()
			self._pending.remove((ticket, kind))
			if kind == "r":
				self._active_readers += 1
			else:
				self._writer_active = True
			return ticket

	def acquire_read(self) -> None:
		self._wait_turn("r")

	def release_read(self) -> None:
		with self._cv:
			self._active_readers -= 1
			if self._active_readers == 0:
				self._cv.notify_all()

	def acquire_write(self) -> None:
		self._wait_turn("w")

	def release_write(self) -> None:
		with self._cv:
			self._writer_active = False
			self._cv.notify_all()

	@_contextmanager
	def read_lock(self):
		self.acquire_read()
		try:
			yield
		finally:
			self.release_read()

	@_contextmanager
	def write_lock(self):
		self.acquire_write()
		try:
			yield
		finally:
			self.release_write()


_INFERENCE_VERIFY_GATE = _InferenceVerifyGate()

from pydantic import BaseModel

# DEFAULT_MODEL can be overridden per-process by AGENT_MODEL env var so the
# benchmark driver's --model flag actually changes which Ollama tag the
# agents call. Falls back to the project default when unset.
DEFAULT_MODEL = _os.environ.get("AGENT_MODEL", "qwen2.5-coder:7b")

# Keep the model resident between calls so the per-candidate compile/energy
# phase doesn't trigger Ollama's default 5-min idle eviction + cold reload.
# Accepts an Ollama duration ("30m"), seconds (int), or "-1" to pin until exit.
AGENT_KEEP_ALIVE = _os.environ.get("AGENT_KEEP_ALIVE", "30m")

# GLOBAL_CONTRACT_PATH = Path(__file__).resolve().parent / "contracts" / "global_contract.md"
ANALYZER_CONTRACT_PATH = Path(__file__).resolve().parent / "contracts" / "analyzer_contract.md"
REPAIR_CONTRACT_PATH = Path(__file__).resolve().parent / "contracts" / "repairer_contract.md"
OPTIMISER_CONTRACT_PATH = Path(__file__).resolve().parent / "contracts" / "optimiser_contract_new.md"


def load_contract(path: Path) -> str:
	"""Load a prompt contract, failing CLOSED if it is missing.

	Previously a missing file silently produced an EMPTY system prompt, so a
	bad mount or a renamed file would yield unconditioned agents that still
	emitted plausible C and still produced numbers -- an invisible change of
	experimental treatment. The contracts ARE the treatment; refuse to run
	without them.
	"""
	if not path.is_file():
		raise FileNotFoundError(
			f"Prompt contract missing: {path}. The contracts define the agents' "
			f"treatment; refusing to run with an empty prompt."
		)
	text = path.read_text(encoding="utf-8").strip()
	if not text:
		raise ValueError(f"Prompt contract is empty: {path}")
	return text


def _is_python(language: str) -> bool:
	return language.lower().startswith("python")


def _structure_note(language: str, no_main: bool, *, stage: str) -> str:
	"""Output-shape instruction injected into the repair/optimize user message.

	This is separate from the role contract and was previously hardcoded for C.
	It matters for Python: those programs have no `main()` and are not compiled,
	so the C wording ("compiled on its own", "including `main()`", "do not drop
	or rename `main()`") would push the model to wrap top-level statements in a
	main() that never gets called — silently breaking a working script.

	The C strings below are byte-identical to the originals so the existing
	Codeflaws runs stay reproducible.
	"""
	if no_main:
		# Harness-concatenated benchmark: the verifier appends its own main(),
		# so the model must not emit one.
		if stage == "repair":
			return (
				"IMPORTANT: Your output is concatenated with a verifier-side test\n"
				"harness that provides `main()`. You MUST NOT include a `main()`\n"
				"function or test code. Output only the function(s) referenced by\n"
				"the analyzer/report, plus any required #include directives, with\n"
				"the same signature as the input."
			)
		return (
			"IMPORTANT: Your output is concatenated with a verifier-side test\n"
			"harness that provides `main()`. Do NOT include `main()` or test\n"
			"code. Preserve the function signature(s) exactly."
		)

	if _is_python(language):
		# Full-program Python script: top-level statements ARE the program.
		if stage == "repair":
			return (
				"IMPORTANT: This is a complete standalone Python program executed\n"
				"as a script. Return the COMPLETE source file, preserving the\n"
				"top-level statements exactly as they are structured. Do NOT wrap\n"
				"the code in a `main()` function or add an `if __name__ ==\n"
				"'__main__':` guard. Do not change the I/O behaviour (reading\n"
				"stdin, writing stdout) or any function signatures."
			)
		return (
			"IMPORTANT: This is a complete standalone Python program executed as\n"
			"a script. Return the COMPLETE source file. Do NOT wrap the code in a\n"
			"`main()` function or add an `if __name__ == '__main__':` guard.\n"
			"Preserve I/O behaviour and all function signatures exactly."
		)

	# Full-program compiled benchmark (Codeflaws): the source owns
	# main() and is compiled standalone; keep the complete program.
	if stage == "repair":
		return (
			"IMPORTANT: This is a complete standalone program compiled on its\n"
			"own. Return the COMPLETE source file including `main()` and all\n"
			"existing functions. Do not drop or rename `main()`; do not change\n"
			"the I/O behaviour or function signatures."
		)
	return (
		"IMPORTANT: This is a complete standalone program. Return the\n"
		"COMPLETE source file including `main()` and all existing functions.\n"
		"Preserve I/O behaviour and all function signatures exactly."
	)


def _select_contract(language: str, base_path: Path) -> Path:
	"""Return a language-specific contract file if one exists, else the base.

	Naming convention: <stem>_python.md for Python3, etc.
	If no language variant is found the base contract is returned unchanged,
	so the C99 pipeline is completely unaffected by this helper.
	"""
	_LANG_SLUG = {"Python3": "python", "Python": "python"}
	slug = _LANG_SLUG.get(language, "")
	if slug:
		candidate = base_path.with_name(base_path.stem + f"_{slug}" + base_path.suffix)
		if candidate.is_file():
			return candidate
	return base_path

def read_text(path: Path) -> str:
	return path.read_text(encoding="utf-8")

def strip_code_fences(text: str) -> str:
	stripped = text.strip()
	if stripped.startswith("```"):
		lines = stripped.splitlines()
		if len(lines) >= 2 and lines[-1].strip() == "```":
			first = lines[0].strip()
			if first.startswith("```"):
				return "\n".join(lines[1:-1])
	return text

class Agent:
	def __init__(self, model: str = DEFAULT_MODEL, config: dict = None) -> None:
		self.model = model
		self.config = config
		# self.global_contract = load_contract(GLOBAL_CONTRACT_PATH)


	def chat(self, role_prompt: str, user_prompt: str, format: str = None) -> str:
		# Per-call telemetry: wall-time around the Ollama call, token
		# counts pulled from the response, optional codecarbon CO2eq
		# capture if AGENT_CODECARBON=1. The whole block is wrapped in a
		# try/finally so an exception during the Ollama call still
		# records the partial timing (useful when the model crashes
		# mid-generation or context-overflows).
		_t0 = time.perf_counter()
		_prompt_tokens = 0
		_completion_tokens = 0
		_co2_kg: float | None = None
		_co2_backend = telemetry.co2_backend_label()
		# Inference energy (secondary, descriptive): GPU+CPU counter deltas around
		# the Ollama call. Distinct from F3 (the repaired program's execution
		# energy). Enabled by default; disable with AGENT_INFERENCE_ENERGY=0.
		_ie: dict = {"gpu_mj": None, "cpu_mj": None}

		try:
			with telemetry.co2_capture() as _co2_holder, measure_inference_energy() as _ie:
				# Cap generation length by default; a policy can override by
				# setting num_predict explicitly in its config dict.
				_opts: dict = dict(self.config) if self.config else {}
				_opts.setdefault("num_predict", 2048)

				_messages = [
					{"role": "system", "content": role_prompt},
					{"role": "user",   "content": user_prompt},
				]

				# ── Ollama (default) ──────────────────────────────────────
				# Local inference: acquire the gate as a reader so a
				# pending/active RAPL verification window (the writer) can
				# drain and block this call rather than run concurrently
				# with it. See _InferenceVerifyGate above.
				with _INFERENCE_VERIFY_GATE.read_lock():
					response: ChatResponse = _ollama_client.chat(
						model=self.model,
						messages=_messages,
						stream=False,
						think=False,
						options=_opts,
						format=format if format else None,
						keep_alive=AGENT_KEEP_ALIVE,
					)
				# Extract token counts. Ollama returns prompt_eval_count and
				# eval_count on the response object (or in the dict form).
				if isinstance(response, dict):
					_prompt_tokens = int(response.get("prompt_eval_count") or 0)
					_completion_tokens = int(response.get("eval_count") or 0)
					message = response.get("message", {})
					response_text = message.get("content", "")
				else:
					_prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
					_completion_tokens = int(getattr(response, "eval_count", 0) or 0)
					message = getattr(response, "message", None)
					response_text = getattr(message, "content", "") if message is not None else ""
					if not response_text and hasattr(response, "model_dump"):
						response_text = response.model_dump().get("message", {}).get("content", "")

			_co2_kg = _co2_holder.get("value")

			if not response_text:
				raise RuntimeError("LLM response was empty (backend=ollama)")
			return response_text
		finally:
			_wall = time.perf_counter() - _t0
			telemetry.record_llm_call(
				model=self.model,
				wall_time_s=_wall,
				prompt_tokens=_prompt_tokens,
				completion_tokens=_completion_tokens,
				co2_kg=_co2_kg,
				co2_backend=_co2_backend,
				gpu_energy_mj=_ie.get("gpu_mj"),
				cpu_energy_mj=_ie.get("cpu_mj"),
			)


@dataclass
class VerificationResult:
	compiled: bool
	compile_stdout: str
	compile_stderr: str
	tests_total: int
	tests_passed: int
	failed_tests: list[str]
	# Energy consumed running the actual test workload, in millijoules.
	# Populated by TestRunners that measure energy over their run loop;
	# None when not measured (the runner didn't wrap the run, or compile
	# failed). This is the source-of-truth for F3 — measured over exactly
	# the workload that produced tests_passed/tests_total.
	energy_mj: float | None = None
	energy_backend: str = ""
	energy_backend_subtype: str = ""
	# Identifiers of the specific tests that passed (TestRunner-defined
	# format, e.g. Codeflaws' "heldout-input-posN" suffix N). Populated by
	# diagnostic-capable TestRunners so a later, separate energy-measurement
	# pass can rerun exactly these tests instead of the whole suite — see
	# _CodeflawsTestRunner.diagnostic()/measure_energy() in
	# benchmarks/codeflaws/adapter.py. Empty when not applicable.
	passing_test_ids: list[str] = field(default_factory=list)

	@property
	def is_success(self) -> bool:
		return self.compiled and self.tests_passed == self.tests_total

class SeverityType(Enum):
	LOW = "LOW"
	MEDIUM = "MEDIUM"
	HIGH = "HIGH"
	CRITICAL = "CRITICAL"


class AnalysisOutput(BaseModel):
	issue_id: str
	line_start: int
	line_end: int
	type: str
	severity: SeverityType
	short_desc: str

class AnalysisList(BaseModel):
	issues: list[AnalysisOutput]

class AnalyzerAgent:
	def __init__(self, config: dict = None) -> None:
		self.llm = Agent(config=config)

	def analyze(self, source_code: str, relative_path: str, language: str = "C99") -> str:
		role = load_contract(_select_contract(language, ANALYZER_CONTRACT_PATH))
		user = textwrap.dedent(
			f"""
			<LANGUAGE>{language}</LANGUAGE>
			<CODE>
			{source_code}
			</CODE>
			"""
		).strip()
		raw = self.llm.chat(role, user)
		return strip_code_fences(raw).strip()
		# return self.llm.chat(role, user, format=AnalysisList.model_json_schema())


class RepairAgent:
	def __init__(self, config: dict = None) -> None:
		self.llm = Agent(config=config)


	def repair(self, source_code: str, report: str, relative_path: str, verification_report: VerificationResult = None, language: str = "C99", no_main: bool = False) -> str:
		system = load_contract(_select_contract(language, REPAIR_CONTRACT_PATH))
		structure_note = _structure_note(language, no_main, stage="repair")
		user = textwrap.dedent(
			f"""
			File: {relative_path}
			<LANGUAGE>{language}</LANGUAGE>

			{structure_note}

			<REPORT>
			{report}
			</REPORT>

			<CODE>
			{source_code}
			</CODE>
			"""
		).strip()
		if verification_report is not None:
			verification_json = json.dumps({
				"compiled": verification_report.compiled,
				"compile_stderr": verification_report.compile_stderr,
				"tests_total": verification_report.tests_total,
				"tests_passed": verification_report.tests_passed,
				"failed_tests": verification_report.failed_tests,
			}, indent=2)
			user += f"\n\n<VERIFICATION_REPORT>\n{verification_json}\n</VERIFICATION_REPORT>"
		return strip_code_fences(self.llm.chat(system, user)).strip() + "\n"


class VerifierAgent:
	def __init__(self, compile_timeout: int = 30, run_timeout: int = 10) -> None:
		self.compile_timeout = compile_timeout
		self.run_timeout = run_timeout

	def verify(self, source_path: Path, test_dir: Path) -> VerificationResult:
		with tempfile.TemporaryDirectory(prefix="agent_verify_", ignore_cleanup_errors=True) as tmpdir:
			exe = Path(tmpdir) / "prog.out"

			compile_proc = subprocess.run(
				["gcc", str(source_path), "-O2", "-Wall", "-Wextra", "-o", str(exe)],
				text=True,
				capture_output=True,
				timeout=self.compile_timeout,
			)

			if compile_proc.returncode != 0:
				return VerificationResult(
					compiled=False,
					compile_stdout=compile_proc.stdout,
					compile_stderr=compile_proc.stderr,
					tests_total=0,
					tests_passed=0,
					failed_tests=[],
				)

			inputs = sorted(glob.glob(str(test_dir / "input*.txt")))
			failed_tests: list[str] = []
			passed = 0
			passing_ids: list[str] = []

			# Measure F3 over the real test-run workload, same as the
			# benchmark TestRunners do. Lazy import to avoid a hard
			# energy-package dependency in agents.py at module load.
			from energy import get_active_backend
			_e_mj = None
			_e_backend = ""
			_e_subtype = ""
			# Acquire the gate as the (sole) writer before opening the
			# measurement window. RAPL counts energy for the full CPU
			# package, so this must exclude both other verify() calls AND
			# any local Ollama inference running in other threads — see
			# _InferenceVerifyGate above. Held only for the duration of the
			# test suite (~1-5 s); readers resume as soon as it releases.
			with _INFERENCE_VERIFY_GATE.write_lock():
				try:
					_backend = get_active_backend()
					with _backend.measure() as _window:
						for input_path in inputs:
							input_file = Path(input_path)
							expected_file = input_file.with_name(input_file.name.replace("input", "expected", 1))
							if not expected_file.exists():
								failed_tests.append(f"{input_file.name}: missing {expected_file.name}")
								continue

							try:
								run_proc = subprocess.run(
									[str(exe)],
									input=read_text(input_file),
									text=True,
									capture_output=True,
									timeout=self.run_timeout,
								)
							except subprocess.TimeoutExpired:
								failed_tests.append(f"{input_file.name}: timeout after {self.run_timeout}s")
								continue

							expected = read_text(expected_file).strip()
							actual = run_proc.stdout.strip()

							if run_proc.returncode != 0:
								failed_tests.append(
									f"{input_file.name}: runtime return code {run_proc.returncode}, stderr={run_proc.stderr.strip()}"
								)
							elif actual != expected:
								failed_tests.append(f"{input_file.name}: output mismatch")
							else:
								passed += 1
								passing_ids.append(input_file.name)
					_e_mj = _window.energy_mj
					_e_backend = _window.backend_name
					_e_subtype = _window.backend_subtype
				except Exception:
					_e_backend = _e_backend or "none"

			return VerificationResult(
				compiled=True,
				compile_stdout=compile_proc.stdout,
				compile_stderr=compile_proc.stderr,
				tests_total=len(inputs),
				tests_passed=passed,
				passing_test_ids=passing_ids,
				energy_mj=_e_mj,
				energy_backend=_e_backend,
				energy_backend_subtype=_e_subtype,
				failed_tests=failed_tests,
			)


class OptimizerAgent:
	def __init__(self, config: dict = None) -> None:
		self.llm = Agent(config=config)

	def optimize(self, repaired_code: str, report: str, verification: VerificationResult, relative_path: str, language: str = "C99", no_main: bool = False) -> str:
		system = load_contract(_select_contract(language, OPTIMISER_CONTRACT_PATH))
		verification_summary = {
			"compiled": verification.compiled,
			"tests_passed": verification.tests_passed,
			"tests_total": verification.tests_total,
			"failed_tests": verification.failed_tests,
		}
		structure_note = _structure_note(language, no_main, stage="optimize")
		user = textwrap.dedent(
			f"""
			<LANGUAGE>{language}</LANGUAGE>

			{structure_note}

			<REPORT>
			{report}
			</REPORT>

			<VERIFICATION>
			{json.dumps(verification_summary, indent=2)}
			</VERIFICATION>

			<CODE>
			{repaired_code}
			</CODE>
			"""
		).strip()
		return strip_code_fences(self.llm.chat(system, user)).strip()