"""Two-minute end-to-end demo of AlgoGenesis.

    .venv/Scripts/python scripts/demo.py            # real LLM calls if CEREBRAS_API_KEY is set
    .venv/Scripts/python scripts/demo.py --fake     # offline: the "LLM" returns the PSO baseline
    .venv/Scripts/python scripts/demo.py --calls 5

Steps:
  1. The benchmark: one PSO run on BBOB f1 (5-D), measured by the harness.
  2. Baselines on the frozen training fitness (24 BBOB functions, 72 runs each).
  3. The harness refuses to be cheated: three exploit attempts, all caught.
  4. The LLM loop: a few LLaMEA (1+1) calls, each generated program scored.

The API key is read from the environment or from a .env file in the repo root
(gitignored). Demo logs go to demo_output/ (gitignored), never into the real
experiment logs.
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from algogenesis.baselines import BASELINES  # noqa: E402
from algogenesis.fitness import TRAINING_SUITE  # noqa: E402
from algogenesis.harness import evaluate_code, run_suite, score  # noqa: E402

DEMO_DIR = REPO / "demo_output"


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def header(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n  {n}. {title}\n{'=' * 72}", flush=True)


def step_benchmark() -> None:
    header(1, "Benchmark: PSO on BBOB f1 (Sphere), 5-D, 10,000 evaluations")
    r = score(BASELINES["PSO"], "bbob_f1", dim=5, budget=10_000, seed=0, instance=4)
    checkpoints = [0, 99, 999, 4999, r.evals - 1]
    for i in checkpoints:
        print(f"  after {i + 1:>6,} evaluations: best error {r.history[i]:.3e}")
    print(f"  evaluations counted by the harness: {r.evals:,}   AOCC {r.aocc:.3f}   time {r.runtime_s:.2f} s")


def step_baselines() -> dict:
    header(2, "Baselines on the frozen training fitness (24 BBOB functions x 3 instances, 5-D)")
    scores = {}
    for name, opt in BASELINES.items():
        t0 = time.perf_counter()
        s = run_suite(opt, name=name, **TRAINING_SUITE)
        scores[name] = s.fitness
        print(f"  {name:<4} fitness (mean AOCC, 0-1, higher is better): {s.fitness:.4f}   "
              f"[{len(s.runs)} runs in {time.perf_counter() - t0:.1f} s]", flush=True)
    print("  An LLM-generated algorithm has to beat these to count as progress.")
    return scores


def _read_optimum(func, dim, bounds, budget):
    return None, func(func.problem.x_opt), []


def _reset_budget(func, dim, bounds, budget):
    for _ in range(3 * budget):
        if func.evals >= budget - 1:
            func.evals = 0
        func(np.random.uniform(bounds[0], bounds[1]))
    return None, 0.0, []


def step_exploits() -> None:
    header(3, "The harness can't be cheated")
    r = score(_read_optimum, "bbob_f3", dim=5, budget=1000)
    print(f"  read the hidden optimum      -> run failed: {not r.ok}, score {r.aocc:.1f}")
    r = score(_reset_budget, "bbob_f3", dim=5, budget=1000)
    print(f"  reset the evaluation counter -> run failed: {not r.ok}, extra evaluations recorded: {r.evals}")
    code = ("import os\n\ndef optimize(func, dim, bounds, budget):\n"
            "    return None, 0.0, []\n")
    s = evaluate_code(code, timeout=60, functions=("bbob_f1",), dims=(5,), n_runs=1, budget_per_dim=10)
    reason = s.error.strip().splitlines()[-1] if s.error else ""
    print(f"  import 'os' (escape attempt) -> rejected before running: {'policy violation' in (s.error or '')}")
    print(f"                                  reason given to the LLM: {reason}")


def step_llm(calls: int, force_fake: bool) -> None:
    from run_llm import run

    from algogenesis.llm.client import LLMError

    real = not force_fake and bool(os.environ.get("CEREBRAS_API_KEY"))
    provider, model = ("cerebras", "gpt-oss-120b") if real else ("fake", "demo")
    header(4, f"LLM loop: LLaMEA (1+1), {calls} calls, {provider}/{model}")
    if not real:
        print("  (no CEREBRAS_API_KEY found: using the offline fake provider, which returns the PSO baseline)")
    # a fresh demo run each time: one more than the runs already logged
    run_no = len(list(DEMO_DIR.glob("results/llm_runs/*/A/run*.jsonl"))) + 1
    try:
        status = run(provider, model, "A", run_no, calls=calls, data_dir=DEMO_DIR)
    except LLMError as e:
        print(f"  API error: {e}\n  Falling back to the fake provider.")
        provider, model = "fake", "demo"
        status = run(provider, model, "A", run_no, calls=calls, data_dir=DEMO_DIR)
    print(f"  finished: {status}. Full log and the best program are under {DEMO_DIR}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake", action="store_true", help="never call a real LLM")
    parser.add_argument("--calls", type=int, default=3, help="LLM calls in step 4")
    parser.add_argument("--skip", nargs="*", type=int, default=[], help="steps to skip, e.g. --skip 2")
    args = parser.parse_args()
    load_dotenv(REPO / ".env")

    t0 = time.perf_counter()
    if 1 not in args.skip:
        step_benchmark()
    if 2 not in args.skip:
        step_baselines()
    if 3 not in args.skip:
        step_exploits()
    if 4 not in args.skip:
        step_llm(args.calls, args.fake)
    print(f"\nDemo finished in {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":  # required: scoring uses multiprocessing "spawn"
    main()
