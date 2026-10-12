"""Run the baselines through the harness and save the numbers.

    python scripts/run_baselines.py                      # dim 10, 15 runs each
    python scripts/run_baselines.py --dims 10 30 --runs 25

Outputs go to results/: baselines.json (every run, with subsampled convergence
histories) and baselines.md (summary table).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algogenesis.baselines import BASELINES  # noqa: E402
from algogenesis.benchmarks import BENCHMARKS  # noqa: E402
from algogenesis.harness import run_suite  # noqa: E402


def markdown_table(rows):
    header = ("| Algorithm | Function | Dim | Mean error | Std | Median | Best | Success | AOCC "
              "| Runtime (s) | Overhead (s) |")
    lines = [header, "|" + "---|" * 11]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {r['function']} | {r['dim']} | {r['mean_error']:.3e} | {r['std_error']:.2e} "
            f"| {r['median_error']:.3e} | {r['best_error']:.3e} | {r['success_rate']:.0%} | {r['mean_aocc']:.4f} "
            f"| {r['mean_runtime_s']:.3f} | {r['mean_overhead_s']:.3f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--functions", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--dims", nargs="+", type=int, default=[10])
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--budget-per-dim", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    suites, rows = [], []
    for name, optimizer in BASELINES.items():
        print(f"Running {name} ...", flush=True)
        suite = run_suite(optimizer, name=name, functions=args.functions, dims=args.dims,
                          n_runs=args.runs, budget_per_dim=args.budget_per_dim)
        suites.append(suite)
        rows.extend(suite.summary())

    settings = {"functions": args.functions, "dims": args.dims, "runs": args.runs,
                "budget_per_dim": args.budget_per_dim}
    (args.out / "baselines.json").write_text(json.dumps(
        {"settings": settings, "algorithms": [s.to_dict() for s in suites]}, indent=2))
    table = markdown_table(rows)
    (args.out / "baselines.md").write_text(
        f"# Baseline results\n\nSettings: {settings}. Error = best f − f* at the end of the budget "
        f"(shifted instances); success = error ≤ 1e-8.\n\n{table}\n", encoding="utf-8")

    print()
    print(table)
    print(f"\nSaved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
