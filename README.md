# MetaMind

LLM-guided evolutionary discovery of metaheuristic optimization algorithms.

We reproduce and extend [LLaMEA](https://arxiv.org/abs/2405.20132) (van Stein & Bäck, IEEE TEVC 2025). A large language model acts as the evolutionary operator: it writes optimizer code, the code is scored on benchmark functions, and the score is fed back so the LLM can improve on its best design. LLaMEA used paid GPT-4; this project uses only free open-weight models (gpt-oss, Qwen) at zero cost. We also compare the loop against a sampling control with no feedback, audit the evaluator for exploits, and check whether the "new" algorithms are known ones renamed.

Course project for Bio-Inspired Algorithms (BMSCE) by Siddhant, Pratyush and Om.

**Explainer page:** [`site/index.html`](site/index.html) presents the project and its results for a general reader, with interactive charts, every function's convergence curves, the 5/10/20-D comparison, and each LLM run call by call with the generated code. It is a single self-contained file: download it and open it in any browser, with no server or install needed. See [Explainer page](#explainer-page) below.

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
| `algogenesis/baselines/` | PSO, real-coded GA, DE, CMA-ES, BIPOP-CMA-ES and random search |
| `algogenesis/report.py` | Results tables (Markdown and LaTeX), significance tests, convergence plots |
| `scripts/run_llm.py` | One LLM run: the (1+1) loop (arm A) or the no-feedback control (arm B); resumable |
| `scripts/run_queue.py` | A queue of runs for one model, used by the GitHub Actions workflow in `.github/workflows/` |
| `scripts/compare.py` | Full comparison of baselines and LLM-found algorithms, written to `results/comparison/` by default |
| `scripts/run_baselines.py` | Quick baseline-only numbers (Phase 1) |
| `scripts/demo.py` | Two-minute end-to-end demo; `--fake` runs offline |
| `scripts/make_paper_assets.py` | Every statistic, table and figure for the paper, from the saved test-phase results |
| `scripts/make_site.py` | Builds the explainer page `site/index.html` from `site/template.html` and the saved results |
| `scripts/analyze_runs.py` | Run analysis of the LLM logs: search progress, failures, cost and program families ([data/run_analysis/summary.md](data/run_analysis/summary.md)) |
| `tests/` | Harness, BBOB, report and exploit-canary tests |
| `data/` | Logs and best programs from the LLM runs (see [data/README.md](data/README.md)) |

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
.venv/Scripts/python scripts/compare.py --candidates data/found/*.py --runs 30
```

The paper's test phase runs everything on all 24 BBOB functions, 15 held-out instances, in 5-D and then in 10-D and 20-D. Qwen A3 is left out of the 10-D and 20-D runs because it exceeded the one-hour limit in 5-D. At 20-D the run takes about two hours, and the machine must not sleep: the per-candidate time limit keeps counting while it does.

```bash
.venv/Scripts/python scripts/compare.py --suite bbob --candidates data/found/*.py --out results/test_phase
```

```bash
C=$(ls data/found/*.py | grep -v qwen-3.8-27b_A_run3)
.venv/Scripts/python scripts/compare.py --suite bbob --dims 10 --candidates $C --out results/test_phase_10d
.venv/Scripts/python scripts/compare.py --suite bbob --dims 20 --candidates $C --out results/test_phase_20d
```

Then build the paper's numbers and the explainer page from those results. `results/` and `paper/` are not tracked (`comparison.json` is about 75 MB per dimension), so both scripts need the runs above on your machine.

```bash
.venv/Scripts/python scripts/make_paper_assets.py
.venv/Scripts/python -X utf8 scripts/make_site.py
```

## Results

Mean AOCC on the held-out instances (higher is better). The programs were searched for in 5-D only. Qwen A1 and gpt-oss A2 are the programs with the highest training fitness for each model.

| Algorithm | 5-D | 10-D | 20-D |
|---|---|---|---|
| BIPOP-CMA-ES | 0.588 | 0.509 | 0.446 |
| Qwen A1 (CMA-ES with restarts) | 0.554 | 0.443 | 0.241 |
| gpt-oss A2 (jDE + Nelder-Mead + (1+1)-ES) | 0.531 | 0.289 | 0.120 |
| CMA-ES | 0.517 | 0.465 | 0.408 |
| DE | 0.371 | 0.152 | 0.064 |
| PSO | 0.327 | 0.238 | 0.187 |
| GA | 0.249 | 0.193 | 0.159 |
| Random search | 0.091 | 0.043 | 0.025 |

- In 5-D the free models reach CMA-ES level but not BIPOP-CMA-ES. That advantage does not transfer: in 10-D and 20-D no generated program beats CMA-ES.
- With five runs per arm, the feedback loop's best programs are not significantly better than the control's (p = 0.31 in 5-D). Feedback does raise the average valid program (exploratory, p = 0.008).
- gpt-oss A2 crashes on 9–11% of test runs (depending on dimension) with a NumPy error, after no failures in training, and Qwen A3 hangs on test instances.

## Explainer page

[`site/index.html`](site/index.html) explains the project and shows all of the results above for a general reader. It covers:

- the summary verdict and the overall scores;
- the 5/10/20-D comparison;
- per-function convergence curves;
- each LLM run call by call, with the code of every improvement;
- the statistical tests and the integrity audit.

To view it, open the file in a browser (on GitHub, download it first, since GitHub shows HTML as source). It works offline; only the web fonts need the internet, and system fonts are used without them.

The page is generated, so don't edit `index.html` by hand. Edit `site/template.html`, then rebuild with `scripts/make_site.py` (see [Usage](#usage)); every number on the page comes from the saved results.

## Status

- [x] Harness, classic benchmarks, PSO and GA baselines, comparison pipeline
- [x] Exploit hardening and train/test split
- [x] BBOB suite; training fitness frozen
- [x] LLM client, (1+1) loop and no-feedback control, run on GitHub Actions (Pratyush)
- [x] Prompts and run monitoring (Om)
- [x] Run analysis (Om): [data/run_analysis/summary.md](data/run_analysis/summary.md)
- [x] DE, CMA-ES, BIPOP-CMA-ES and random-search baselines (Siddhant)
- [x] Statistics upgrade; BBOB test phase in 5-D, 10-D and 20-D (Siddhant)
- [x] Paper assets script and explainer page

The LLM runs are finished. The Cerebras free tier began returning HTTP 402 on 4 November 2026, so no further runs are planned.
