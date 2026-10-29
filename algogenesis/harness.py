"""Scoring harness: runs any optimizer on the benchmarks under a fixed budget.

Every algorithm, hand-written or LLM-generated, must follow one contract:

    def optimize(func, dim, bounds, budget) -> (best_x, best_f, history)

    func    callable taking a numpy array of shape (dim,) and returning a float
    dim     int, number of dimensions
    bounds  (lower, upper), two numpy arrays of shape (dim,)
    budget  int, maximum number of calls to func

See INTERFACE.md for the full description.

The harness does not trust what the optimizer reports. It wraps ``func`` so it
can count evaluations, stop the run at the budget and record the best-so-far
value after every single call. All reported numbers come from that record, so
every algorithm is measured the same way regardless of how it was written.
"""
from __future__ import annotations

import inspect
import json
import math
import multiprocessing as mp
import queue
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .benchmarks import Problem, make_problem

Optimizer = Callable[..., tuple]

# Errors are put on a log scale between 1e-8 and 1e2 for AOCC, as in LLaMEA.
LOG_ERR_LOW, LOG_ERR_HIGH = -8.0, 2.0
SUCCESS_ERROR = 1e-8
DEFAULT_BUDGET_PER_DIM = 1000

# Problem instances (i.e. optimum shifts) are split so that nothing the LLM loop
# trains on is ever used to report final results.
TRAIN_INSTANCES = (1, 2, 3)
TEST_INSTANCES = tuple(range(4, 19))
assert not set(TRAIN_INSTANCES) & set(TEST_INSTANCES), "train and test instances must be disjoint"


class BudgetExhausted(BaseException):
    """Raised when an optimizer calls func after its budget is spent.

    Subclasses BaseException (like KeyboardInterrupt) so a generated
    algorithm's blanket ``except Exception`` cannot swallow it.
    """


class CountingFunction:
    """Wraps a Problem: counts calls, enforces the budget, records best-so-far.

    Never hand this object to an optimizer: its attributes (the problem, the
    call counter) would let it read the optimum or reset the budget. ``score``
    gives optimizers a plain function that calls it instead.
    """

    def __init__(self, problem: Problem, budget: int):
        self.problem = problem
        self.budget = int(budget)
        self.evals = 0
        self.best_f = math.inf
        self.best_x: np.ndarray | None = None
        self.trace = np.empty(self.budget)
        self.out_of_bounds = 0
        self.func_time = 0.0
        self._lower, self._upper = problem.bounds

    def __call__(self, x) -> float:
        if self.evals >= self.budget:
            raise BudgetExhausted(f"budget of {self.budget} evaluations exhausted")
        x = np.asarray(x, dtype=float)
        if x.shape != (self.problem.dim,):
            raise ValueError(f"func expects an array of shape ({self.problem.dim},), got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("func was called with a non-finite point (NaN or inf)")
        if np.any(x < self._lower) or np.any(x > self._upper):
            self.out_of_bounds += 1

        t0 = time.perf_counter()
        f = self.problem(x)
        self.func_time += time.perf_counter() - t0

        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        self.trace[self.evals] = self.best_f
        self.evals += 1
        return f


@dataclass
class RunResult:
    """Outcome of one optimizer run on one problem instance."""

    function: str
    dim: int
    budget: int
    seed: int
    instance: int
    ok: bool                    # False if the optimizer raised an error
    error: str | None           # traceback if it did
    best_error: float           # best f found minus the known optimum (lower is better)
    best_x: np.ndarray | None
    reported_best_f: float | None  # what the optimizer itself returned, for sanity checks
    evals: int
    runtime_s: float            # wall time of the whole run
    func_time_s: float          # part of runtime spent inside the benchmark function
    out_of_bounds: int          # evaluations outside the box (allowed, but reported)
    aocc: float                 # area over the convergence curve in [0, 1], higher is better
    history: np.ndarray = field(repr=False)  # best-so-far error after each evaluation

    @property
    def overhead_s(self) -> float:
        """Time the optimizer spent on its own bookkeeping, i.e. its compute cost."""
        return self.runtime_s - self.func_time_s


def aocc(error_history: np.ndarray, budget: int) -> float:
    """Normalised area over the convergence curve (the fitness used by LLaMEA).

    Each evaluation contributes 1 when the best-so-far error is <= 1e-8 and 0
    when it is >= 1e2, linear in log10(error) between. Unused budget is padded
    with the final value, so finishing early is neither rewarded nor punished.
    """
    if len(error_history) == 0:
        return 0.0
    padded = np.full(budget, error_history[-1], dtype=float)
    padded[: len(error_history)] = error_history
    logs = np.log10(np.clip(padded, 10**LOG_ERR_LOW, 10**LOG_ERR_HIGH))
    return float(np.mean((LOG_ERR_HIGH - logs) / (LOG_ERR_HIGH - LOG_ERR_LOW)))


def score(
    optimizer: Optimizer,
    function: str | Problem = "sphere",
    dim: int = 10,
    budget: int | None = None,
    seed: int = 0,
    instance: int | None = None,
) -> RunResult:
    """Run ``optimizer`` once and measure it.

    ``instance`` defaults to ``seed + 1``, so each seed also gets a differently
    shifted optimum. Pass ``instance=0`` for the canonical unshifted function.
    """
    if isinstance(function, Problem):
        problem = function
    else:
        problem = make_problem(function, dim, seed + 1 if instance is None else instance)
    dim = problem.dim
    budget = int(budget or DEFAULT_BUDGET_PER_DIM * dim)

    random.seed(seed)
    np.random.seed(seed)
    counted = CountingFunction(problem, budget)

    def func(x):
        return counted(x)

    ok, err, reported = True, None, None

    t0 = time.perf_counter()
    try:
        out = optimizer(func, dim, problem.bounds, budget)
        if not (isinstance(out, (tuple, list)) and len(out) == 3):
            raise TypeError(f"optimize must return (best_x, best_f, history), got {type(out).__name__}")
        reported = float(out[1])
    except BudgetExhausted:
        pass  # ran to the limit without returning; the recorded best still counts
    except Exception:
        ok, err = False, traceback.format_exc()
    runtime = time.perf_counter() - t0

    # The recorder is out of the optimizer's reach, but check it anyway: a run
    # that somehow overran the budget or rewrote its history doesn't count.
    trace = counted.trace[: counted.evals]
    if ok and (counted.evals > budget or np.any(np.diff(trace) > 0)):
        ok, err = False, "policy violation: the evaluation record was tampered with"

    history = trace - problem.f_opt
    best_error = float(history[-1]) if counted.evals else math.inf
    return RunResult(
        function=problem.name,
        dim=dim,
        budget=budget,
        seed=seed,
        instance=problem.instance,
        ok=ok,
        error=err,
        best_error=best_error,
        best_x=counted.best_x,
        reported_best_f=reported,
        evals=counted.evals,
        runtime_s=runtime,
        func_time_s=counted.func_time,
        out_of_bounds=counted.out_of_bounds,
        aocc=aocc(history, budget) if ok else 0.0,
        history=history,
    )


@dataclass
class SuiteResult:
    """All runs of one optimizer across functions, dimensions and seeds."""

    name: str
    runs: list[RunResult]
    error: str | None = None  # set when the candidate could not run at all (bad code, timeout)

    @property
    def fitness(self) -> float:
        """Single number for the LLM loop: mean AOCC over all runs (failed runs count as 0)."""
        if self.error or not self.runs:
            return 0.0
        return float(np.mean([r.aocc for r in self.runs]))

    def summary(self) -> list[dict]:
        """One row of aggregate statistics per (function, dim)."""
        groups: dict[tuple[str, int], list[RunResult]] = {}
        for r in self.runs:
            groups.setdefault((r.function, r.dim), []).append(r)
        rows = []
        for (fn, dim), runs in groups.items():
            errs = np.array([r.best_error for r in runs])
            rows.append({
                "algorithm": self.name,
                "function": fn,
                "dim": dim,
                "runs": len(runs),
                "failures": sum(not r.ok for r in runs),
                "mean_error": float(np.mean(errs)),
                "std_error": float(np.std(errs)),
                "median_error": float(np.median(errs)),
                "best_error": float(np.min(errs)),
                "success_rate": float(np.mean(errs <= SUCCESS_ERROR)),
                "mean_aocc": float(np.mean([r.aocc for r in runs])),
                "mean_runtime_s": float(np.mean([r.runtime_s for r in runs])),
                "mean_overhead_s": float(np.mean([r.overhead_s for r in runs])),
            })
        return rows

    def to_dict(self, history_points: int = 200) -> dict:
        """JSON-friendly dict. Histories are subsampled to ``history_points`` evaluations."""
        runs = []
        for r in self.runs:
            idx = np.unique(np.linspace(0, max(r.evals - 1, 0), history_points).astype(int)) if r.evals else []
            runs.append({
                "function": r.function, "dim": r.dim, "budget": r.budget, "seed": r.seed,
                "instance": r.instance, "ok": r.ok, "error": r.error, "best_error": r.best_error,
                "reported_best_f": r.reported_best_f, "evals": r.evals, "runtime_s": r.runtime_s,
                "func_time_s": r.func_time_s, "out_of_bounds": r.out_of_bounds, "aocc": r.aocc,
                "history_evals": [int(i) + 1 for i in idx],
                "history_error": [float(r.history[i]) for i in idx],
            })
        return {"name": self.name, "fitness": self.fitness, "error": self.error,
                "summary": self.summary(), "runs": runs}

    def save_json(self, path: str | Path, history_points: int = 200) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(history_points), indent=2))


def run_suite(
    optimizer: Optimizer,
    name: str = "optimizer",
    functions: Sequence[str] = ("sphere", "rastrigin"),
    dims: Sequence[int] = (10,),
    n_runs: int = 10,
    budget_per_dim: int = DEFAULT_BUDGET_PER_DIM,
    first_seed: int = 0,
    instances: Sequence[int] = TEST_INSTANCES,
) -> SuiteResult:
    """Score ``optimizer`` on every function x dim, ``n_runs`` runs each, in this process.

    Run k uses seed ``first_seed + k`` and cycles through ``instances``. The
    default is the test instances; the LLM loop's training fitness passes
    TRAIN_INSTANCES (see fitness.py).
    """
    if not instances:
        raise ValueError("instances must not be empty")
    runs = [
        score(optimizer, fn, dim, budget_per_dim * dim, first_seed + k, instance=instances[k % len(instances)])
        for fn in functions
        for dim in dims
        for k in range(n_runs)
    ]
    return SuiteResult(name, runs)


# --- Running LLM-generated code -------------------------------------------------

def load_optimizer(code: str) -> Optimizer:
    """Execute candidate source code and return its ``optimize`` function."""
    namespace: dict = {"__name__": "candidate"}
    exec(compile(code, "<candidate>", "exec"), namespace)
    fn = namespace.get("optimize")
    if not callable(fn):
        raise ValueError("the code must define a function optimize(func, dim, bounds, budget)")
    try:
        inspect.signature(fn).bind(None, None, None, None)
    except TypeError:
        raise ValueError(
            f"optimize must accept (func, dim, bounds, budget); its signature is {inspect.signature(fn)}"
        ) from None
    return fn


def _suite_worker(code: str, name: str, kwargs: dict, out: mp.Queue) -> None:
    try:
        out.put(("ok", run_suite(load_optimizer(code), name=name, **kwargs)))
    except BaseException:
        out.put(("error", traceback.format_exc()))


def evaluate_code(code: str, name: str = "candidate", timeout: float = 300.0, **suite_kwargs) -> SuiteResult:
    """Score candidate source code in a separate process with a time limit.

    This is the entry point for the LLM loop. Errors, infinite loops and hard
    crashes in the generated code come back as a SuiteResult with ``error`` set
    and fitness 0, rather than taking down the loop. Note that a subprocess is
    not a security sandbox: only run code from the LLM, never from strangers.

    On Windows, call this from inside ``if __name__ == "__main__":``.
    """
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    proc = ctx.Process(target=_suite_worker, args=(code, name, suite_kwargs, out), daemon=True)
    proc.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                status, payload = out.get(timeout=0.5)
                break
            except queue.Empty:
                if not proc.is_alive():
                    return SuiteResult(name, [], error=f"evaluation process crashed (exit code {proc.exitcode})")
                if time.monotonic() > deadline:
                    proc.kill()
                    return SuiteResult(name, [], error=f"evaluation timed out after {timeout:.0f} s")
    finally:
        if proc.is_alive():
            proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()
    if status == "error":
        return SuiteResult(name, [], error=payload)
    return payload
