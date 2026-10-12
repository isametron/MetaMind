# AlgoGenesis

LLM-guided evolutionary discovery of metaheuristic optimization algorithms (reproducing and extending LLaMEA).

## Layout

- `algogenesis/benchmarks.py`: Sphere and Rastrigin, plus shifted problem instances
- `algogenesis/harness.py`: scoring pipeline (`score`, `run_suite`, `evaluate_code`)
- `algogenesis/baselines/pso.py`: standard global-best PSO baseline
- `scripts/run_baselines.py`: runs the baselines and writes `results/`
- `INTERFACE.md`: the contract every algorithm must follow

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_baselines.py --dims 10 30
```
