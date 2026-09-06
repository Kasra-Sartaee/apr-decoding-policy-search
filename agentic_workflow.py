#!/usr/bin/env python3
import ast
import json
import time
from dataclasses import dataclass
from pathlib import Path

import lizard

import telemetry
from agents import (
	DEFAULT_MODEL,
	AnalyzerAgent,
	OptimizerAgent,
	RepairAgent,
	VerificationResult,
	VerifierAgent,
	read_text,
)
from energy import EnergyBackend, set_active_backend

# Energy-measurement state. F3 is now measured by the TestRunner over the
# real test-run workload and read off the VerificationResult; these
# module-level labels are set by evaluate_fitness() from that result and
# written into every verification JSON (e.g. "apple_silicon", "rapl_sysfs",
# "codecarbon", "runtime_proxy", or "none").
_ENERGY_BACKEND: EnergyBackend | None = None
_LAST_ENERGY_BACKEND: str = "none"
_LAST_ENERGY_SUBTYPE: str = ""


def set_energy_backend(backend: EnergyBackend) -> None:
	"""Driver calls this once at startup, after CLI parsing.

	Registers the backend both locally and with the energy package so the
	benchmark TestRunners (which measure F3 over the real test workload)
	use the same explicitly-resolved backend.
	"""
	global _ENERGY_BACKEND
	_ENERGY_BACKEND = backend
	set_active_backend(backend)


@dataclass
class ResultObject:
  def __init__(self, cl, correctness, energy_consumption):
    self.cl = cl
    self.correctness = correctness
    self.energy_consumption = energy_consumption


def write_text(path: Path, content: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")


# ── F1: whole-program cyclomatic complexity ───────────────────────────────────
# Convention: M = 1 + (total decision points over the whole translation unit).
#
# The pipeline previously used lizard's "sum of per-function CCN". lizard is a
# *function-granularity* analyzer: it enumerates function definitions and
# reports CCN per definition, so code outside any function body is invisible to
# it. For C that is harmless (all executable C code lives inside functions), but
# 84% of RunBugRun Python submissions are flat scripts with no `def` at all, for
# which lizard silently returns 0 rather than erroring. That is a tool-scope
# artifact, not a property of the code — these two programs are equivalent:
#
#   flat.py : lizard n_funcs=0 sum_CCN=0 | whole-program CC=4
#   func.py : lizard n_funcs=1 sum_CCN=4 | whole-program CC=4
#
# The two conventions differ by (n_funcs - 1) plus any module-level decision
# points, so applying one to Python and the other to C would reintroduce
# cross-language incomparability. Both arms therefore use whole-program CC.
#
# Impact on the C arm is negligible and rank-preserving: 80% of Codeflaws
# programs define exactly one function, giving Spearman rho = 0.995 /
# Pearson r = 0.996 against the old convention and a mean shift of 0.33.
# The convention and its consequences are stated in the methodology.
_CC_DECISION_NODES = (
  ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
  ast.Assert, ast.IfExp, ast.comprehension,
)


class _McCabeVisitor(ast.NodeVisitor):
  """Counts decision points across an entire Python module (radon convention)."""

  def __init__(self) -> None:
    self.decision_points = 0

  def visit(self, node):
    if isinstance(node, _CC_DECISION_NODES):
      self.decision_points += 1
    elif isinstance(node, ast.BoolOp):
      # `a and b and c` contributes two branch points, not one.
      self.decision_points += len(node.values) - 1
    elif isinstance(node, ast.Match):  # py3.10+ structural pattern matching
      self.decision_points += len(node.cases)
    self.generic_visit(node)


def compute_complexity(source_path: Path) -> int:
  """Whole-program cyclomatic complexity of `source_path`.

  Dispatches on file extension, which the benchmark adapter sets via
  `Problem.source_extension`:

    .py            -> AST walk (sees module-level code lizard cannot)
    everything else -> lizard, folded to whole-program as
                       sum(CCN) - n_funcs + 1

  Returns 0 when complexity cannot be measured (syntax error, empty file,
  no parseable functions). 0 is a "not measured" sentinel, not a real CC —
  the minimum for any real program is 1. Such candidates fail verification
  anyway and are excluded from the feasible aggregate that F1 averages over.
  """
  if source_path.suffix == ".py":
    try:
      tree = ast.parse(source_path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError):
      return 0
    visitor = _McCabeVisitor()
    visitor.visit(tree)
    return visitor.decision_points + 1

  try:
    funcs = lizard.analyze_file(str(source_path)).function_list
  except Exception:
    return 0
  if not funcs:
    return 0
  # sum(dp_i + 1) - n_funcs + 1 == sum(dp_i) + 1, the whole-unit McCabe.
  return sum(f.cyclomatic_complexity for f in funcs) - len(funcs) + 1


def evaluate_fitness(optimised_code: Path, verification_report: VerificationResult) -> tuple[float, float, float]:
  """Compute the (F1, F2, F3) fitness tuple for a verified candidate.

    F1 (cyclomatic complexity): whole-program McCabe — see compute_complexity().
    F2 (correctness): tests_passed / tests_total from the verification.
    F3 (energy): read directly from the verification result. The TestRunner
        measured energy over the *real test-run workload* — exactly the
        execution that produced F2 — so F3 is the energy of running the
        benchmark's tests, not a standalone empty-stdin run. If the runner
        didn't measure energy (energy_mj is None), F3 is `inf` — never 0.0;
        see the comment at the return below.
  """
  global _LAST_ENERGY_BACKEND, _LAST_ENERGY_SUBTYPE
  cl = compute_complexity(optimised_code)
  correctness = verification_report.tests_passed / verification_report.tests_total if verification_report.tests_total > 0 else 0.0

  # F3 comes from the verification result — measured over the test workload.
  energy_mj = getattr(verification_report, "energy_mj", None)
  if energy_mj is None:
    _LAST_ENERGY_BACKEND = getattr(verification_report, "energy_backend", "") or "none"
    _LAST_ENERGY_SUBTYPE = getattr(verification_report, "energy_backend_subtype", "")
    # inf, NOT 0.0. F3 is MINIMISED, so 0.0 is an unbeatable score -- an
    # unmeasured problem would outrank every real measurement. inf is excluded
    # from the aggregate by the feasibility filter (np.isfinite) in
    # _mean_fitness_over, which is exactly why that filter exists.
    return cl, correctness, float("inf")
  _LAST_ENERGY_BACKEND = getattr(verification_report, "energy_backend", "") or "none"
  _LAST_ENERGY_SUBTYPE = getattr(verification_report, "energy_backend_subtype", "")
  return cl, correctness, energy_mj


def build_paths_for_file(
	c_file: Path,
	programs_dir: Path,
	reports_dir: Path,
	repaired_dir: Path,
	verification_dir: Path,
	optimized_dir: Path,
) -> dict[str, Path]:
	relative = c_file.relative_to(programs_dir)
	stem = relative.stem
	return {
		"relative": relative,
		"report": reports_dir / relative.parent / f"{stem}.json",
		"verification": verification_dir / relative.parent,
		"repaired": repaired_dir / relative,
		"optimized": optimized_dir / relative,
	}

_REPO_ROOT = Path(__file__).resolve().parent


def _report_has_no_issues(report: str) -> bool:
	"""True iff the analyzer report parses as JSON with an empty `issues` list.

	Small models sometimes confabulate fixes when the analyzer correctly
	reports no issues. Short-circuiting here avoids a guaranteed-broken
	repair cycle on already-correct programs.
	"""
	try:
		data = json.loads(report)
	except (json.JSONDecodeError, TypeError):
		return False
	if not isinstance(data, dict):
		return False
	issues = data.get("issues")
	return isinstance(issues, list) and len(issues) == 0


def _portable_path(p: Path) -> str:
	"""Return a POSIX-style path relative to the repo root when possible.

	Keeps verification JSONs portable across machines — no more embedded
	/Users/<someone>/... or C:\\Users\\<someone>\\... absolute paths.
	"""
	try:
		rel = p.resolve().relative_to(_REPO_ROOT)
		return rel.as_posix()
	except (ValueError, OSError):
		return p.as_posix()


# Module-level snapshots that the search drivers read
# after each run_pipeline() call to flatten into the per-problem CSV row.
# Same pattern as _LAST_ENERGY_BACKEND above. Reset every problem; safe so
# long as run_pipeline is called serially (which the driver does).
_LAST_RUN_TELEMETRY: dict = {}
_LAST_RUN_REPAIR_RETRIES: int = 0
_LAST_RUN_OPTIMIZE_RETRIES: int = 0


_VERIFICATION_WRITE_COUNTER = 0


def write_results_to_json(paths: dict[str, Path],
						  verify_result: VerificationResult,
						  verify_optimized: VerificationResult = None,
						  cl: float = None,
						  correctness: float = None,
						  result: float = None,
						  repair_retries: int = 0,
						  optimize_retries: int = 0,
						  phase_b_measurement=None) -> Path:
	"""Write the per-problem verification summary JSON. Returns the path written.

	Returning the path lets callers print the exact filename instead of
	recomputing a (possibly different) timestamp. The filename uses a
	nanosecond timestamp plus a monotonic counter so multiple writes for
	the same problem in the same second cannot collide.
	"""
	global _LAST_RUN_TELEMETRY, _LAST_RUN_REPAIR_RETRIES, _LAST_RUN_OPTIMIZE_RETRIES
	global _VERIFICATION_WRITE_COUNTER
	tel_summary = telemetry.summary()
	_LAST_RUN_TELEMETRY = tel_summary
	_LAST_RUN_REPAIR_RETRIES = repair_retries
	_LAST_RUN_OPTIMIZE_RETRIES = optimize_retries

	results = {
			"report_path": _portable_path(paths["report"]),
			"repaired_path": _portable_path(paths["repaired"]),
			"optimized_path": _portable_path(paths["optimized"]),
			"repaired_verification": {
				"compiled": verify_result.compiled,
				"tests_passed": verify_result.tests_passed,
				"tests_total": verify_result.tests_total,
				"failed_tests": verify_result.failed_tests,
				"compile_stderr": verify_result.compile_stderr,
			},
			"retry_counts": {
				"repair": repair_retries,
				"optimize": optimize_retries,
			},
			"telemetry": tel_summary,
	}
	if verify_optimized is not None:
		results["optimized_verification"] = {
				"compiled": verify_optimized.compiled,
				"tests_passed": verify_optimized.tests_passed,
				"tests_total": verify_optimized.tests_total,
				"failed_tests": verify_optimized.failed_tests,
				"compile_stderr": verify_optimized.compile_stderr,
			}
	if cl is not None and correctness is not None and result is not None:
		results["fitness_functions"] = {
				"cyclomatic_complexity": cl,
				"correctness": correctness,
				"energy_consumption_pkg": result,
				"energy_backend": _LAST_ENERGY_BACKEND,
				"energy_backend_subtype": _LAST_ENERGY_SUBTYPE,
			}
	if phase_b_measurement is not None:
		# Records exactly which of the diagnostic-passing tests reproduced
		# under the Phase B energy rerun and which didn't, so a partial
		# reproduction (or a total failure, surfaced as result=inf above)
		# is auditable rather than silently absorbed into a smaller/zeroed
		# average — see EnergyMeasurement in benchmarks/codeflaws/adapter.py.
		results["phase_b_measurement"] = {
				"expected_test_count": len(phase_b_measurement.matched_ids) + len(phase_b_measurement.mismatched_ids),
				"matched_test_ids": phase_b_measurement.matched_ids,
				"mismatched_test_ids": phase_b_measurement.mismatched_ids,
				# Why each mismatched id didn't reproduce: "timeout" is
				# plausibly a measurement-architecture artifact (residual
				# thermal/frequency state from Phase A's preceding workload);
				# "output_mismatch"/"nonzero_exit" is direct evidence the
				# program isn't reliably correct for that input. See
				# docs/reports/measurement_protocol.md.
				"mismatch_reasons": phase_b_measurement.mismatch_reasons,
			}
	_VERIFICATION_WRITE_COUNTER += 1
	verification_json = (
		paths["verification"]
		/ f"verification_summary_{time.time_ns()}_{_VERIFICATION_WRITE_COUNTER}.json"
	)
	write_text(verification_json, json.dumps(results, indent=2))
	return verification_json



@dataclass
class PendingResult:
	"""Settled state after Phase A (analyze/repair/retry/optimize/retry).
	`energy` is always provisional (deferred to Phase B). `correctness` is
	ALSO provisional whenever Phase B still has work to do (see
	`final_source_path` below) — it reflects a single diagnostic execution,
	and Phase B may revise it downward once it re-executes those same
	tests and finds some don't reproduce. `cl` is the only field that's
	genuinely final already (static analysis of the source, independent of
	test execution).

	`final_source_path` is None in two distinct cases, both meaning
	"no Phase B work needed, `energy` AND `correctness` below are already
	final":
	  - a compile failure persisted through all retries (worst-case fitness)
	  - the TestRunner in use isn't diagnostic-capable (no `.diagnostic()`
	    method — legacy path, or a benchmark other than Codeflaws), so
	    Phase A already measured real energy (and correctness needs no
	    revision, since there's no separate reproduction check) in one
	    shot via `__call__`.
	When it IS set, `test_runner.measure_energy(final_source_path,
	passing_test_ids)` must be called (serially, see run_pipeline_phase_b)
	to get the real F3 value AND the revised F2 value.
	"""
	cl: float
	correctness: float
	energy: float
	final_source_path: Path | None
	passing_test_ids: list
	test_runner: object | None
	paths: dict
	verify_result: VerificationResult
	verify_optimized: VerificationResult | None
	repair_retries: int
	optimize_retries: int


_NO_SAFE_SENTINEL = "NO_SAFE_TRANSFORMATION"


def is_no_safe_transformation(text: str) -> bool:
    """True iff the Optimizer declined to transform the program.

    ONE parser, used by every optimiser response -- the initial call and every
    retry. Previously only the initial call was checked, so a decline arriving
    on a retry was written straight to disk as if it were C, producing 22-byte
    "source" files containing the sentinel; and the check was an exact string
    comparison, so a Markdown-wrapped `**NO_SAFE_TRANSFORMATION**` slipped
    through even on the initial path. Both were observed in real smoke output
    (5 such artifacts across two runs).

    Tolerates the wrappers small models add despite being told not to: code
    fences, bold/italic emphasis, backticks, blockquote markers. The ENTIRE
    stripped response must reduce to the sentinel, so a real program that
    merely mentions the word is never mistaken for a decline.
    """
    if not text:
        return False
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            t = "\n".join(lines[1:-1]).strip()
    for _ in range(3):                      # peel nested wrappers, e.g. > **`X`**
        t = t.strip().lstrip(">").strip()
        t = t.strip("`").strip()
        t = t.strip("*").strip()
        t = t.strip("_").strip()
    return t.upper() == _NO_SAFE_SENTINEL


def optimized_preserves_correctness(
    repaired: VerificationResult, optimized: VerificationResult
) -> tuple[bool, str]:
    """Whether the optimised artifact may be accepted in place of the repaired one.

    Compares the EXACT SET of passing tests, not the correctness fraction. Two
    programs can pass the same NUMBER of different tests -- the fraction hides
    that, and real smoke output showed forbidden semantic edits (a changed loop
    bound `i < N` -> `i < N - 3`, a changed branch condition `sum25>2` ->
    `sum25>3`, an uncommented qsort call) being accepted because the counts
    happened to tie.

    Also requires the optimised artifact to COMPILE and to have been run against
    the same number of tests. Without the compile requirement a non-compiling
    artifact (correctness 0.0) tied with a compiling repair that also scored 0.0
    and was accepted, replacing a valid program with a broken one.
    """
    if not optimized.compiled:
        return False, "optimised code did not compile"
    if optimized.tests_total != repaired.tests_total:
        return False, (
            f"test count changed ({repaired.tests_total} -> {optimized.tests_total})"
        )
    # Fail CLOSED when a runner does not populate passing_test_ids: without
    # them the set comparison below is vacuous and would rubber-stamp anything.
    for tag, v in (("repaired", repaired), ("optimised", optimized)):
        if len(v.passing_test_ids) != v.tests_passed:
            return False, (
                f"{tag} verification did not report per-test ids "
                f"({len(v.passing_test_ids)} ids vs tests_passed={v.tests_passed}); "
                f"cannot establish equivalence"
            )
    rep_ids, opt_ids = set(repaired.passing_test_ids), set(optimized.passing_test_ids)
    lost, gained = rep_ids - opt_ids, opt_ids - rep_ids
    if lost:
        return False, (
            f"{len(lost)}/{len(rep_ids)} previously-passing test(s) no longer "
            f"pass: {sorted(lost)[:8]}"
        )
    if gained:
        # An optimisation that makes previously-FAILING tests pass has changed
        # behaviour -- the contract forbids fixing bugs, that is the Repair
        # stage's job. Accepting it would credit repair gains to the optimise
        # stage and bias F2 upward under a treatment that forbids it. An
        # earlier revision allowed this ("nothing regressed"), which was wrong.
        return False, (
            f"{len(gained)} previously-failing test(s) now pass "
            f"({sorted(gained)[:8]}) — the optimiser must not change behaviour"
        )
    return True, ""



def run_pipeline_phase_a(config: dict, policy: dict = None, problem=None, skip_optimize: bool = False) -> PendingResult:
	"""Phase A of the pipeline: analyze -> repair -> verify -> retry ->
	optimize -> verify -> retry. When the active TestRunner supports it
	(Codeflaws), verification here calls `.diagnostic()` — compile + run
	tests, comparing output, but NEVER touching RAPL — so this entire
	function is safe to run concurrently across problems in a worker pool.

	Energy is deferred to run_pipeline_phase_b(), which must be called
	afterwards (serially, one problem at a time, only once every Phase A
	worker in the current batch has fully joined) to get the final
	(cl, correctness, energy) fitness. This two-phase split exists because,
	under the old single-phase design, EVERY intermediate repair-retry and
	optimizer-retry verification ran under RAPL even though only the FINAL
	attempt's energy is ever used — and partial/broken repairs frequently
	hang, each timeout burning the full run_timeout under exclusive RAPL
	access while blocking every other worker's inference. See
	memory/rapl_contamination_local_ollama.md and
	benchmarks/codeflaws/adapter.py's _CodeflawsTestRunner docstring.

	For TestRunners without a `.diagnostic()` method (legacy non-Problem
	path, or benchmarks other than Codeflaws), this falls back to the
	original single-shot behaviour: `_run_test` calls the TestRunner
	directly, which already returns real energy immediately — Phase B is
	then a no-op passthrough for those problems.
	"""
	base_dir = Path(config["base_dir"]).resolve()
	reports_dir = (base_dir / config["reports_dir"]).resolve()
	repaired_dir = (base_dir / config["repaired_dir"]).resolve()
	optimized_dir = (base_dir / config["optimized_dir"]).resolve()
	verification_dir = (base_dir / config["verification_json"]).resolve()

	# Decide source-of-truth: Problem (new) vs config['input_file'] (legacy).
	if problem is not None:
		c_file = Path(problem.buggy_source_path).resolve()
		if not c_file.is_file():
			raise FileNotFoundError(
				f"problem.buggy_source_path not found: {c_file} "
				f"(benchmark={problem.benchmark} id={problem.problem_id})"
			)
		# Output paths derive from the problem's stable label, not from
		# programs_dir-relative paths.
		relative_label = problem.relative_label  # e.g. "codeflaws/173-A-bug-…"
		ext = getattr(problem, "source_extension", ".c")
		paths = {
			"relative": Path(relative_label),
			"report": reports_dir / f"{relative_label}.json",
			"verification": verification_dir / relative_label,
			"repaired": repaired_dir / f"{relative_label}{ext}",
			"optimized": optimized_dir / f"{relative_label}{ext}",
		}
		test_runner = problem.test_runner
		_diagnostic = getattr(test_runner, "diagnostic", None)
		_run_test = _diagnostic if _diagnostic is not None else test_runner
		_phase_b_runner = test_runner if _diagnostic is not None else None
	else:
		programs_dir = (base_dir / config["programs_dir"]).resolve()
		tests_dir = (base_dir / config["tests_dir"]).resolve()
		c_file = (programs_dir / config["input_file"]).resolve()
		if not programs_dir.is_dir():
			raise FileNotFoundError(
				f"programs_dir does not exist: {programs_dir}. "
				f"See the benchmark adapter for the expected layout."
			)
		if not c_file.is_file():
			raise FileNotFoundError(
				f"input_file not found: {c_file}. "
				f"Check config['input_file'] (current: {config['input_file']})."
			)
		paths = build_paths_for_file(c_file, programs_dir, reports_dir, repaired_dir, verification_dir, optimized_dir)
		algo_name = paths["relative"].parts[0] if paths["relative"].parts else ""
		algo_test_dir = tests_dir / algo_name
		# Use the legacy verifier wrapped as a callable matching TestRunner.
		_legacy_verifier = VerifierAgent()
		def _run_test(path: Path):
			return _legacy_verifier.verify(path, algo_test_dir)
		_phase_b_runner = None  # legacy path has no diagnostic/measure_energy split

	if policy is None:
		analyzer = AnalyzerAgent()
		repairer = RepairAgent()
		optimizer = OptimizerAgent()
	else:
		analyzer = AnalyzerAgent(config=policy['pi_a'])
		repairer = RepairAgent(config=policy['pi_r'])
		optimizer = OptimizerAgent(config=policy['pi_o'])

	relative = str(paths["relative"])
	language = getattr(problem, "language_hint", "C99") if problem is not None else "C99"
	# Harness-concatenated benchmarks need the model to omit main();
	# full-program benchmarks (Codeflaws, RunBugRun) need it kept.
	no_main = getattr(problem, "harness_concatenated", False) if problem is not None else False

	# Per-problem telemetry: tokens / wall-time / optional CO2eq per LLM call,
	# tagged with the current pipeline stage. Reset at problem start so each
	# row in the per-problem CSV reflects only that problem.
	telemetry.reset()
	repair_retry_count = 0
	optimize_retry_count = 0

	def _pending(cl, correctness, energy, final_source_path, verify_result, verify_optimized) -> PendingResult:
		ids_source = verify_optimized if verify_optimized is not None else verify_result
		return PendingResult(
			cl=cl, correctness=correctness, energy=energy,
			final_source_path=final_source_path,
			passing_test_ids=ids_source.passing_test_ids,
			# _phase_b_runner is None whenever the active TestRunner isn't
			# diagnostic-capable; run_pipeline_phase_b() treats either that
			# or a None final_source_path as "already final, skip Phase B".
			test_runner=_phase_b_runner,
			paths=paths, verify_result=verify_result, verify_optimized=verify_optimized,
			repair_retries=repair_retry_count, optimize_retries=optimize_retry_count,
		)

	# Step 1: Analyze
	telemetry.set_stage("analyze")
	print(f"[ANALYZE] {relative}")
	source = read_text(c_file)
	report = analyzer.analyze(source, relative, language=language)
	write_text(paths["report"], report)

	# Step 2: Repair
	telemetry.set_stage("repair")
	if _report_has_no_issues(report):
		print(f"[REPAIR] {relative} — analyzer reported no issues, skipping repair")
		repaired_code = source
	else:
		print(f"[REPAIR] {relative}")
		repaired_code = repairer.repair(source, report, relative, language=language, no_main=no_main)
	write_text(paths["repaired"], repaired_code)

	# Step 3: Verify repaired code
	telemetry.set_stage("verify")
	print(f"[VERIFY] {relative}")
	verify_result = _run_test(paths["repaired"])

	# Retry the repair when it didn't fully succeed — either it failed to
	# compile, OR it compiled but didn't pass every test. The repairer
	# already receives `compile_stderr` + `failed_tests` via
	# verification_report, so test-failure feedback is actionable. We only
	# retry when the analyzer actually reported issues; if it reported none
	# (and we passed the source through unchanged) there is nothing for the
	# repair LLM to fix, so retrying would just invite confabulation.
	report_had_issues = not _report_has_no_issues(report)
	if report_had_issues and not verify_result.is_success:
		attempts = policy['pi_v'].get("retry", 0) if policy is not None else 4
		reason = "did not compile" if not verify_result.compiled else "failed some tests"
		print(f"Initial repair {reason} for {relative}. Retrying with verification feedback.")
		while attempts > 0 and not verify_result.is_success:
			repair_retry_count += 1
			telemetry.set_stage("repair")
			print(f"[REPAIR RETRY] {relative}, attempts left: {attempts}")
			repaired_code = repairer.repair(source, report, relative, verification_report=verify_result, language=language, no_main=no_main)
			write_text(paths["repaired"], repaired_code)
			telemetry.set_stage("verify")
			print(f"[VERIFY RETRY] {relative}, attempts left: {attempts}")
			verify_result = _run_test(paths["repaired"])
			attempts -= 1

	# Only a true compile failure produces worst-case (inf) fitness and
	# skips optimization. A repair that compiles but fails some/all tests
	# still has a meaningful CL + partial correctness + energy, so it
	# proceeds to fitness evaluation and (if enabled) optimization.
	if not verify_result.compiled:
		print(f"Repaired code for {relative} does not compile after retries. Skipping optimization.")
		vjson = write_results_to_json(
			paths, verify_result, cl=float('inf'), correctness=0, result=float('inf'),
			repair_retries=repair_retry_count, optimize_retries=optimize_retry_count,
		)
		print(f"\nFinished. Verification summary written to: {vjson}")
		return _pending(float('inf'), 0, float('inf'), None, verify_result, None)

	cl, correctness, result = evaluate_fitness(paths["repaired"], verify_result)
	# Energy is only final here when there's no Phase B (non-diagnostic
	# TestRunner already measured it); otherwise omit it from this interim
	# JSON entirely so a reader can't mistake the 0.0 placeholder for real
	# F3 — the definitive write happens in Phase B once energy is known.
	write_results_to_json(
		paths, verify_result, cl=cl, correctness=correctness,
		result=(result if _phase_b_runner is None else None),
		repair_retries=repair_retry_count, optimize_retries=optimize_retry_count,
	)

	# Step 4: Optimize (skippable for baseline characterization runs)
	if skip_optimize:
		print(f"[OPTIMIZE] {relative} — skipped (skip_optimize=True). Fitness from repaired code.")
		print(f"Repaired fitness for {relative} - CL: {cl}, Correctness: {correctness:.2%}")
		return _pending(cl, correctness, result, paths["repaired"], verify_result, None)

	telemetry.set_stage("optimize")
	print(f"[OPTIMIZE] {relative}")
	optimized_code = optimizer.optimize(repaired_code, report, verify_result, relative, language=language, no_main=no_main)

	if is_no_safe_transformation(optimized_code):
		cl, correctness, result = evaluate_fitness(paths["repaired"], verify_result)
		print(f"No safe transformation available for {relative}. Evaluating repaired code - Cyclomatic Complexity: {cl}, Correctness: {correctness:.2%}")
		vjson = write_results_to_json(
			paths, verify_result, cl=cl, correctness=correctness,
			result=(result if _phase_b_runner is None else None),
			repair_retries=repair_retry_count, optimize_retries=optimize_retry_count,
		)
		print(f"\nFinished. Verification summary written to: {vjson}")
		return _pending(cl, correctness, result, paths["repaired"], verify_result, None)

	write_text(paths["optimized"], optimized_code)
	telemetry.set_stage("verify")
	print(f"[EVALUATE OPTIMIZED CODE] {relative}")
	verify_optimized = _run_test(paths["optimized"])
	if verify_optimized.compiled:
		cl, correctness, result = evaluate_fitness(paths["optimized"], verify_optimized)
		print(f"Optimized code for {relative} - Cyclomatic Complexity: {cl}, Correctness: {correctness:.2%}")
	else:
		print(f"Optimized code for {relative} did not compile. Retrying optimization with verification feedback.")
		attempts = policy['pi_o'].get("retry", 0) if policy is not None else 4
		_declined_on_retry = False
		while attempts > 0 and not verify_optimized.compiled:
			optimize_retry_count += 1
			telemetry.set_stage("optimize")
			print(f"[OPTIMIZE RETRY] {relative}, attempts left: {attempts}")
			optimized_code = optimizer.optimize(repaired_code, report, verify_optimized, relative, language=language, no_main=no_main)
			# A decline may arrive on ANY attempt, not just the first. Detect it
			# BEFORE writing, or the sentinel lands on disk as a 22-byte "C
			# file" that then fails to compile (observed 5x in real output).
			if is_no_safe_transformation(optimized_code):
				print(f"[OPTIMIZE RETRY] {relative}: optimiser declined — keeping the repaired artifact.")
				_declined_on_retry = True
				break
			write_text(paths["optimized"], optimized_code)
			telemetry.set_stage("verify")
			print(f"[VERIFY OPTIMIZED RETRY] {relative}, attempts left: {attempts}")
			verify_optimized = _run_test(paths["optimized"])
			attempts -= 1
		if _declined_on_retry:
			cl, correctness, result = evaluate_fitness(paths["repaired"], verify_result)
			vjson = write_results_to_json(
				paths, verify_result, cl=cl, correctness=correctness,
				result=(result if _phase_b_runner is None else None),
				repair_retries=repair_retry_count, optimize_retries=optimize_retry_count,
			)
			print(f"\nFinished. Verification summary written to: {vjson}")
			return _pending(cl, correctness, result, paths["repaired"], verify_result, None)
		if verify_optimized.compiled:
			cl, correctness, result = evaluate_fitness(paths["optimized"], verify_optimized)
			print(f"Optimized code for {relative} after retry - Cyclomatic Complexity: {cl}, Correctness: {correctness:.2%}")
		else:
			print(f"Optimized code for {relative} did not compile after retries. Skipping fitness evaluation.")
			cl, correctness, result = float('inf'), 0, float('inf')
	# ── Accept the optimized code ONLY if it preserved correctness ──────────
	# The optimiser contract states its output "must produce identical outputs
	# to the verified code for all inputs". Until now the pipeline only ever
	# checked that the optimized code COMPILED, so an optimisation that
	# compiled but silently broke tests was accepted and its damaged
	# correctness became the candidate's F2. Measured over ~184k repair->
	# optimise transitions that happened in 21.5% of 1.5B cases (15.5% at 7B),
	# and when it happened the median loss was ~51 percentage points (~32 pp
	# at 7B) -- roughly one in five fully-correct repairs was destroyed. That
	# is a large confound sitting inside the primary correctness objective.
	#
	# Now: if the optimized artifact does not at least match the repaired
	# artifact's correctness, fall back to the repaired one. This makes the
	# optimize stage monotonically non-harmful to F2 while keeping its F1
	# benefit wherever it is safe, and aligns the implementation with the
	# contract (and with standard APR practice, where a patch is validated
	# before being accepted). Ties go to the optimized version, since it may
	# have lower complexity at equal correctness.
	#
	# NOTE: this is a deliberate protocol change -- F1/F2 under it are NOT
	# comparable with runs made before it. See docs/reports/15.
	_ok, _why = optimized_preserves_correctness(verify_result, verify_optimized)
	_rejected_optimize = not _ok
	if _rejected_optimize:
		print(
			f"[OPTIMIZE REJECTED] {relative}: {_why} — keeping the repaired "
			f"artifact (the contract requires identical outputs)."
		)
		cl, correctness, result = evaluate_fitness(paths["repaired"], verify_result)
		final_verify, final_path = verify_result, paths["repaired"]
	else:
		final_verify, final_path = verify_optimized, paths["optimized"]

	# Pass verify_optimized so the JSON records the optimized test report
	# alongside the optimized fitness (#6 fix).
	vjson = write_results_to_json(
		paths, verify_result, verify_optimized=verify_optimized,
		cl=cl, correctness=correctness,
		result=(result if (_phase_b_runner is None or not final_verify.compiled) else None),
		repair_retries=repair_retry_count, optimize_retries=optimize_retry_count,
	)
	print(f"\nFinished. Verification summary written to: {vjson}")
	if not final_verify.compiled:
		return _pending(float('inf'), 0, float('inf'), None, verify_result, verify_optimized)
	# Phase B must re-measure the artifact we actually accepted, and take its
	# passing-test ids from that same artifact's verification.
	return _pending(cl, correctness, result, final_path, verify_result,
	                None if _rejected_optimize else verify_optimized)


def run_pipeline_phase_b(pending: PendingResult) -> ResultObject:
	"""Phase B: final energy measurement AND correctness revision. Call this
	SERIALLY, one problem at a time, only after every Phase A worker in the
	current batch has fully joined (the ThreadPoolExecutor's `with` block
	has exited). During this call nothing else in the process should be
	doing local inference, compilation, or testing — that's what makes the
	RAPL reading valid; see _CodeflawsTestRunner.measure_energy()'s
	docstring.

	Correctness (F2), not just energy (F3), is finalised here: Phase A's
	`correctness` is provisional (from a single diagnostic execution).
	Phase B re-executes exactly the tests that diagnostic run counted as
	passing, and any that fail to reproduce demote from the pass count —
	F2 and F3 must be computed over the same evidence, not two different
	subsets of it, regardless of why a given test didn't reproduce. This
	can only revise correctness downward (Phase B never reruns a
	diagnostic failure, so it can't discover a new pass).

	When `pending.final_source_path` is None, both energy and correctness
	were already finalised in Phase A (compile failed, or a non-diagnostic
	TestRunner already measured everything in one execution) — this is
	then a pure passthrough, no RAPL and no revision involved.
	"""
	if pending.final_source_path is None or pending.test_runner is None:
		return ResultObject(pending.cl, pending.correctness, pending.energy)

	measurement = pending.test_runner.measure_energy(
		pending.final_source_path, pending.passing_test_ids
	)

	global _LAST_ENERGY_BACKEND, _LAST_ENERGY_SUBTYPE
	_LAST_ENERGY_BACKEND = measurement.backend or "none"
	_LAST_ENERGY_SUBTYPE = measurement.backend_subtype

	# Correctness must be measured at the same stage as energy, over the
	# same evidence -- Phase B is re-executing tests that Phase A already
	# counted as passes, and it's strictly more information about those
	# same tests, not a different question. A test that doesn't reproduce
	# here demotes to a failure for F2 too, not just an exclusion from F3 --
	# regardless of *why* it didn't reproduce (timeout included): F2 and F3
	# being computed over different sets of tests for the same candidate is
	# not a defensible measurement, whatever the cause of the mismatch.
	# This can only revise correctness DOWNWARD (Phase B only reruns tests
	# that diagnostically passed, never promotes a diagnostic failure).
	final_verify = pending.verify_optimized if pending.verify_optimized is not None else pending.verify_result
	revised_tests_passed = len(pending.passing_test_ids) - len(measurement.mismatched_ids)
	correctness = (
		revised_tests_passed / final_verify.tests_total if final_verify.tests_total > 0 else 0.0
	)

	if measurement.mismatched_ids:
		# Visible, not silent: some diagnostic-passing tests genuinely
		# failed to reproduce on the Phase B rerun (wrong output, crash,
		# timeout, or a missing test file). These are correctness facts
		# -- determined purely by output comparison, never by whether
		# energy was measurable -- so they demote F2's pass count. F3 is
		# unaffected by which tests reproduced: the single batch window
		# spans every executed test regardless of outcome.
		reason_counts: dict[str, int] = {}
		for _id, _reason in measurement.mismatch_reasons.items():
			reason_counts[_reason] = reason_counts.get(_reason, 0) + 1
		print(
			f"[PHASE B] WARNING: {len(measurement.mismatched_ids)}/"
			f"{len(pending.passing_test_ids)} diagnostic-passing test(s) for "
			f"{pending.paths['relative']} did not reproduce on the Phase B "
			f"rerun: {measurement.mismatched_ids} (reasons: {reason_counts}). "
			f"Correctness revised {pending.correctness:.3f} -> {correctness:.3f}."
		)

	if measurement.mean_energy_mj is None:
		# Nothing usable came out of this call at all (recompile failed
		# after retries, RAPL failed, or zero of passing_test_ids
		# reproduced -- note that last case also means correctness above
		# was just revised down to reflect zero reproducible passes).
		# energy must NOT become 0.0: that would look like a perfect,
		# unbeatable energy score to NSGA-II's minimisation. inf marks this
		# problem infeasible for CL/energy purposes (matching the existing
		# compile-failure convention).
		# Do not assert "0/N reproduced" -- this branch also fires when the
		# batch window itself yielded no usable reading, in which case tests
		# may well have reproduced fine. Report what is actually known.
		print(
			f"[PHASE B] WARNING: energy measurement invalid for "
			f"{pending.paths['relative']} "
			f"({len(measurement.matched_ids)}/{len(pending.passing_test_ids)} tests "
			f"reproduced; recompile, RAPL, or the batch window failed) — marking "
			f"energy=inf, excluded from the feasible aggregate."
		)
		energy = float('inf')
	else:
		energy = measurement.mean_energy_mj

	# Now that energy (and revised correctness) are known, write the
	# definitive verification JSON (Phase A's interim write for this
	# problem omitted `result` entirely, and used the pre-revision
	# correctness since Phase B hadn't run yet).
	vjson = write_results_to_json(
		pending.paths, pending.verify_result, verify_optimized=pending.verify_optimized,
		cl=pending.cl, correctness=correctness, result=energy,
		repair_retries=pending.repair_retries, optimize_retries=pending.optimize_retries,
		phase_b_measurement=measurement,
	)
	print(f"[PHASE B] final energy for {pending.paths['relative']}: {energy} -> {vjson}")
	return ResultObject(pending.cl, correctness, energy)


def run_pipeline(config: dict, policy: dict = None, problem=None, skip_optimize: bool = False) -> ResultObject:
	"""Backward-compatible single-shot entry point: runs Phase A then
	immediately Phase B. Use this when two-phase batching isn't needed —
	N_WORKERS=1 sequential mode, or any caller invoking the pipeline
	directly without a batch-level Phase A/B split (e.g. the __main__
	block below, or the *_nreps.py baseline scripts). For concurrent
	batches, call run_pipeline_phase_a() across a worker pool, join, then
	run_pipeline_phase_b() serially — see scripts/run_nsga2.py's
	_mean_fitness_over().
	"""
	pending = run_pipeline_phase_a(config, policy=policy, problem=problem, skip_optimize=skip_optimize)
	return run_pipeline_phase_b(pending)


def parse_args() -> dict:
	return {
		"base_dir": ".",
		"programs_dir": "programs",
		"tests_dir": "tests",
		"input_file": "aes/aes_original.c",
		"reports_dir": "outputs/reports",
		"repaired_dir": "outputs/repaired",
		"optimized_dir": "outputs/optimized",
		"model": DEFAULT_MODEL,
		"llm_timeout": 180,
		"compile_timeout": 40,
		"run_timeout": 10,
	}


if __name__ == "__main__":
	config = {
		"base_dir": ".",
		"programs_dir": "programs",
		"tests_dir": "tests",
		"input_file": "astar/astar_bug_all.c",
		"reports_dir": "outputs/reports",
		"repaired_dir": "outputs/repaired",
		"optimized_dir": "outputs/optimized",
		"verification_json": "outputs/verification",
		"model": DEFAULT_MODEL,
		"llm_timeout": 180,
		"compile_timeout": 40,
		"run_timeout": 10,
	}

	for i in range(3):
		result = run_pipeline(config)
		with open(f"final_result_{i}.json", "w") as f:
			json.dump({
				"fitness_functions": {
					"cyclomatic_complexity": result.cl,
					"correctness": result.correctness,
					"energy_consumption": result.energy_consumption,
				}
			}, f, indent=2)
