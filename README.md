# AlgoGenesis

LLM-guided evolutionary discovery of metaheuristic optimization algorithms.

We reproduce and extend [LLaMEA](https://arxiv.org/abs/2405.20132) (van Stein & Bäck, IEEE TEVC 2025). A large language model acts as the evolutionary operator: it writes optimizer code, the code is scored on benchmark functions, and the score is fed back so the LLM can improve on its best design. LLaMEA used paid GPT-4; this project uses only free open-weight models (gpt-oss, Qwen) at zero cost. We also compare the loop against a sampling control with no feedback, audit the evaluator for exploits, and check whether the "new" algorithms are known ones renamed.

Course project for Bio-Inspired Algorithms (BMSCE) by Siddhant, Pratyush and Om. The full plan, prior-art audit and risk register are in [`reports/`](reports/).

## How evaluation works

Every algorithm, hand-written or LLM-generated, implements one function:

```python
def optimize(func, dim, bounds, budget):
    ...
    return best_x, best_f, history
```

The harness measures it the same way whatever its source. See [INTERFACE.md](INTERFACE.md) for the full contract.

- **Harness-measured.** The harness counts every call and records the best value itself. What the algorithm reports is only used as a cross-check.
- **Hard budget.** Calling `func` past `budget` stops the run.
- **Exploit-resistant.** `func` is a plain function, so the optimum and the call counter are out of reach. Optima are shifted away from the centre of the box. Generated code is scanned before it runs (`algogenesis/code_audit.py`): escape routes are rejected, and suspicious habits are flagged for the exploit audit. Canary tests in `tests/test_exploits.py` check that known cheats fail.
- **Train/test split.** The LLM loop only sees instances 1–3 (`TRAIN_INSTANCES`). Final results use the disjoint instances 4–18 (`TEST_INSTANCES`).
- **Sandboxed.** Generated code runs in a separate process with a time limit, so crashes and infinite loops come back as errors. A subprocess is not a security sandbox, though.
- **Fitness = AOCC.** This is the area over the convergence curve, as in LLaMEA: log10 of the error, clipped to [1e-8, 1e2]. It runs from 0 to 1, higher is better, and a failed run scores 0.

## Training fitness (frozen, 31 Oct 2026)

`algogenesis.fitness.training_fitness(code)` is the only call the LLM loop makes. It scores a candidate on:

| Setting | Value |
|---|---|
| Functions | all 24 BBOB functions (via COCO's `cocoex`) |
| Dimension | 5 |
| Instances | 1, 2, 3, one seed each (72 runs per candidate) |
| Budget | 2000 × dim = 10,000 evaluations per run, as in LLaMEA |
| Time limit | 300 s per candidate (PSO takes about 5 s) |

## Layout

| Path | What it is |
|---|---|
| `algogenesis/harness.py` | Scoring: `score`, `run_suite`, `evaluate_code`, AOCC, train/test instances |
| `algogenesis/fitness.py` | `training_fitness(code)`: the frozen training suite for the LLM loop |
| `algogenesis/bbob.py` | The 24 BBOB functions adapted to the harness (box [-5, 5]) |
| `algogenesis/benchmarks.py` | Classic shifted Sphere and Rastrigin; `make_problem` for every suite |
| `algogenesis/baselines/` | PSO and real-coded GA (DE, CMA-ES and BIPOP-CMA-ES coming next) |
| `algogenesis/report.py` | Results tables (Markdown and LaTeX), significance tests, convergence plots |
| `scripts/compare.py` | Full comparison of baselines and LLM-found algorithms, written to `results/comparison/` |
| `scripts/run_baselines.py` | Quick baseline-only numbers (Phase 1) |
| `tests/` | Harness, BBOB, report and exploit-canary tests |
| `reports/`, `research_notes/` | Master technical report and the research behind it |

## Setup

Python 3.12 or newer (`coco-experiment` needs it).

Windows:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
```

Mac or Linux: the same commands with `.venv/bin/python`.

## Usage

Score one piece of generated code. On Windows this must run under `if __name__ == "__main__":`.

```python
from algogenesis.fitness import training_fitness

result = training_fitness(code_string)
print(result.fitness, result.error)
```

Run the comparison: the baselines plus any LLM-found algorithms, each a `.py` file defining `optimize`. It runs on test instances by default.

```bash
.venv/Scripts/python scripts/compare.py --candidates found/*.py --runs 30
```

## Status

- [x] Harness, classic benchmarks, PSO and GA baselines, comparison pipeline
- [x] Exploit hardening and train/test split
- [x] BBOB suite; training fitness frozen
- [ ] LLM client, (1+1) loop and no-feedback control, run on GitHub Actions (Pratyush)
- [ ] Prompts, pilot measurement, run monitoring and analysis (Om)
- [ ] DE, CMA-ES and BIPOP-CMA-ES baselines; statistics upgrade; BBOB test phase (Siddhant)
