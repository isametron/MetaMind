# AlgoGenesis

LLM-guided evolutionary discovery of metaheuristic optimization algorithms (reproducing and extending LLaMEA).

## Layout

- `algogenesis/benchmarks.py`: Sphere and Rastrigin, plus shifted problem instances
- `algogenesis/harness.py`: scoring pipeline (`score`, `run_suite`, `evaluate_code`)
- `algogenesis/baselines/`: PSO and real-coded GA baselines
- `algogenesis/report.py`: results tables (Markdown + LaTeX), significance tests, convergence curves
- `scripts/compare.py`: full comparison of baselines and LLM-found algorithms, writes `results/comparison/`
- `scripts/run_baselines.py`: quick baseline-only numbers (Phase 1)
- `INTERFACE.md`: the contract every algorithm must follow

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/compare.py --runs 30
```

To add LLM-found algorithms, save each as a `.py` file defining `optimize(func, dim, bounds, budget)` and pass them in:

```bash
.venv/Scripts/python scripts/compare.py --candidates found/*.py --runs 30
```
