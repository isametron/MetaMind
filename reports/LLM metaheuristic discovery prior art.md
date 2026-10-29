# Zero-Cost LLaMEA: Auditing Open-Weight Algorithm Discovery

## Master technical report for MetaMind / AlgoGenesis: prior-art audit, novelty claim, enhancements, evaluation protocol, risk register and 8-day execution plan (written 2026-10-28, target 2026-11-05)

**Bottom line.** A paper that says "we discovered a new metaheuristic with an LLM" will not hold up. Prior art already covers that, and the critics of metaphor-based algorithms are ready for it. MetaMind can defend a narrower claim, and it is open as of October 2026: **a zero-dollar, audited replication of LLaMEA using current open-weight *reasoning* models (gpt-oss-120b, gpt-oss-20b, Qwen3.8-27B) under free-tier token caps**. It would answer four questions that earlier work answered only for closed or 2024-era models, or not at all:

1. Does the evolutionary loop beat feedback-free sampling at equal calls and tokens?
2. What does each model cost in tokens and wall-clock time?
3. What do the models try to exploit in the evaluator?
4. Are the "discovered" algorithms just DE, CMA-ES or PSO under new names?

The plan has to fit what is actually available. **Llama is no longer on Groq's free tier: it was shut down for free and developer users on 2026-08-16.** The binding constraint is **200K tokens per day per model on Groq**, which is about 50 LLaMEA calls. That makes **Cerebras's free trial (1M tokens/day for gpt-oss-120b and Qwen3.8-27B) the main engine**, with Groq for gpt-oss-20b and as a reserve. That gives roughly **1,650 LLM calls in four days**, about a third of LLaMEA's 4,500.

The harness is ahead of most student work: shifted optima, a best-so-far trace recorded by the harness, and the budget-stop exception. But an audit today found three concrete problems:

- **The optimum can be read.** A candidate can read `func.problem.x_opt`, which gave AOCC = 1.000 after one evaluation.
- **The budget can be reset.** A candidate can reset `func.evals`, which overruns the budget without detection and inflates AOCC.
- **Training and test instances overlap.** Instance numbers are `seed + 1`, so the final 30-run evaluation would reuse the instances seen during evolution.

Fixing these, moving to real BBOB through `cocoex` (it installs on the laptop's Python 3.14; `ioh` does not), and adding CMA-ES/DE baselines with corrected statistics are the day-0 to day-1 critical path. **If the fitness function and prompts are not frozen by noon on Thursday 2026-10-29, the 2026-11-05 date is at risk.** The other two projects also need to be ready by 2026-11-05, which is the largest risk to the schedule. MetaMind is the natural fit for ICCI 2027 (deadline 2026-11-13, computational intelligence), with the other two projects going to the 2027-01-04 venues.

> **How to read the tags.** Citations keep the research notes' verification status.
> - **[V]** checked this session against the arXiv abstract or HTML, the GitHub README or the official page.
> - **[S]** search snippet or secondary summary only; check the full text before citing.
> - **[B]** background knowledge, not re-fetched.
> - **[SR]** venue self-reported on an author's page.
> - **[L]** local install or timing check on this laptop, from the notes.
> - **[R]** checked by me in the MetaMind repo today, including a probe run.
> - **[EST]** my own arithmetic or estimate, not a measurement.
> - **[MEASURE]** a number the team must fill in from its own runs. No result in this report is a MetaMind LLM result; the loop does not exist yet.

---

## 1. Prior art covers LLM-designed optimisers well but leaves a zero-cost audited replication open

### 1.1 What LLaMEA established, and where MetaMind currently stands

LLaMEA (van Stein & Bäck, IEEE TEVC 29(2):331–345, 2025) is a single-parent, single-child evolution strategy over code. The LLM writes a complete Python optimiser class. Each candidate is scored by mean normalised AOCC (area over the convergence curve) on the 24 noiseless BBOB functions at 5-D with 10,000 evaluations, and the loop then either refines the best-so-far program ((1+1) elitist) or refines the latest one ((1,1)) ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V]. The feedback prompt contains:

- the task prompt;
- a list of previous algorithm names and scores;
- the selected parent's code with its mean and standard deviation of AOCC;
- the single instruction "Either refine or redesign to improve the algorithm".

Programs that crash get fitness 0 and their error text is fed back ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V].

The models were gpt-3.5-turbo-0125, gpt-4-turbo-2024-04-09 and gpt-4o-2024-05-13, at temperature 0.8, with 100 iterations per run. There were about 5 runs per configuration: the equation uses k = 5, but the HTML text is garbled, so check the TEVC PDF. In total there were **4,500 LLM calls, of which 18.7% (843) produced execution errors** ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V].

The headline algorithm, ERADS_QuantumFluxUltraRefined, scored **AOCC 0.733 on 5-D BBOB, against 0.703 for default CMA-ES and 0.628 for DE**. A tuned modular CMA-ES ("CMA-best", 0.742) still beat it. The paper body says ERADS "fails to generalize well to higher dimensions", even though the abstract calls the 10-D and 20-D results competitive ([arXiv HTML](https://arxiv.org/html/2405.20132); [arXiv abstract](https://arxiv.org/abs/2405.20132)) [V]. The authors also say plainly that ERADS closely resembles JADE, and that most top algorithms are DE or CMA/DE hybrids ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V]. LLaMEA reports no formal significance tests and no token or money cost ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V].

MetaMind today is a harness without the loop [R]:

- `algogenesis/benchmarks.py` registers only shifted Sphere and Rastrigin. Ackley and Griewank are marked "planned".
- `harness.py` provides the budget-enforcing `CountingFunction`, LLaMEA-style AOCC (log10 error clipped to [1e-8, 1e2], padded with the final value) and subprocess evaluation with a timeout.
- `report.py` produces Mann-Whitney win/tie/loss tables on final error.
- The interface is a function, `optimize(func, dim, bounds, budget)`, not LLaMEA's `__init__(budget, dim)` / `__call__(f)` class. The task prompt therefore has to be adapted, and the paper must say so.

The PSO and GA baselines (30 runs, 1000·d evaluations) show the kind of numbers reviewers will look at [R]. PSO solves Sphere at 10-D (error 5.2e-13, AOCC 0.724) but succeeds in only 87% of 30-D runs. GA leaves 7.4e-4 error on 10-D Sphere (AOCC 0.421), which points to a weak GA configuration. On 30-D Rastrigin, AOCC is **0.008 for PSO and 0.052 for GA**, because final errors of 86 and 12 sit near the 1e2 clip. This matters for the loop. A mean-AOCC fitness that mixes unimodal and multimodal functions is dominated by the unimodal ones, so the LLM is pushed to be fast on Sphere-like problems (see risk R9).

### 1.2 The Leiden follow-ups have closed most "obvious" extensions

The Leiden group has extended LLaMEA almost every quarter. Each item below closes off a novelty claim a student team might reach for:

| Follow-up | What it established | Claim it closes off |
|---|---|---|
| LLaMEA-HPO (ACM TELO 2025) | The LLM writes the structure and SMAC tunes the exposed hyperparameters, giving "superior or comparable performance" with far fewer LLM queries ([arXiv](https://arxiv.org/abs/2410.16309)) [V] | HPO in the loop |
| Mutation control, Yin et al. (EvoApplications 2025) | Tells the LLM what fraction of the code to change, using a heavy-tailed rate. **GPT-3.5 ignored the instruction and GPT-4o followed it** ([arXiv](https://arxiv.org/abs/2412.03250)) [V] | Mutation-rate control |
| BLADE (GECCO 2025 workshop) | Standardises MA-BBOB and SBOX-COST benchmarks with train/test instance splits. Runs LLaMEA with **local Codestral-22B and Qwen2.5-Coder-14B against Gemini Flash**: Gemini-2.0-flash was significantly better in **only 2 of 10 setups** ([arXiv HTML](https://arxiv.org/html/2504.20183v1)) [V] | "Open models are competitive" as a discovery |
| Behaviour-space analysis (Springer 2026, o4-mini) | Six mutation-prompt variants. The best was **(1+1) elitist with a "simplify" prompt plus random perturbation**, and better algorithms exploited more intensively ([arXiv PDF](https://arxiv.org/pdf/2507.03605); [Springer](https://link.springer.com/chapter/10.1007/978-3-032-15635-8_23)) [V/S]. A third-party review notes the ranking rests on 5 runs ([Pith review](https://pith.science/paper/2507.03605)) [S] | Prompt-mix comparisons |
| Code Evolution Graphs (GECCO 2025) | Code complexity grows with repeated prompting and can hurt performance; different LLMs have distinct coding styles ([arXiv](https://arxiv.org/abs/2503.16668)) [V] | Code-bloat analysis |
| LLaMEA-SAGE (preprint, January 2026) | Turns explainable-AI analysis of AST features into mutation instructions and reports gains on MA-BBOB ([arXiv](https://arxiv.org/abs/2601.21511)) [V] | Code-feature-guided search |
| Huang et al. (preprint, March 2026; v2 July 2026) | Strong algorithm examples in the prompt substantially improve LLM-driven design. A search snippet says the paper compares Gemini, GPT and **Qwen**; the Qwen version was not confirmed ([arXiv](https://arxiv.org/abs/2603.02792)) [V/S] | Seeding with good examples; possibly an open-model comparison |

The Huang et al. paper is the single most important item to check before claiming anything about open models (verification item 1).

Outside Leiden, the closest continuous-domain work is **Šenkeřík et al. (EvoApplications 2025)**. They found that GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet and **Llama 3.1** all generated metaheuristics that beat random search and CMA-ES on an engineering problem ([AGH record](https://badap.agh.edu.pl/publikacja/159982)) [V].

The main multi-LLM study is **Zhang et al. (PPSN 2024)**. It compared nine LLMs, including CodeLlama-7B and DeepSeek-Coder-6.7B, inside FunSearch, EoH, ReEvo and a (1+1)-EPS baseline on combinatorial problems ([arXiv HTML](https://arxiv.org/html/2407.10873v1)) [V/S]. Its main findings:

- Larger and code-specialised models were not reliably better.
- A standalone LLM without search was ineffective at every size.
- **(1+1)-EPS was the most sensitive to the choice of LLM.**
- Cost per run was about **$10 (GPT-3.5), $100 (GPT-4) and $200 (Claude 3 Opus)**.

The rest of the field (FunSearch, EoH, ReEvo, MCTS-AHD, HSEvo, AlphaEvolve, ShinkaEvolve, and the LLM4AD platform) mostly evolves single heuristic functions for combinatorial or mathematical problems, not full continuous optimisers ([LLM4AD](https://github.com/Optima-CityU/LLM4AD); [MCTS-AHD](https://arxiv.org/abs/2501.08603); [ShinkaEvolve](https://arxiv.org/pdf/2509.19349)) [V].

Two 2026 preprints already compare methods **at equal token budgets**: "Budget-Efficient Automatic Algorithm Design via Code Graph" ([arXiv 2605.10598](https://arxiv.org/html/2605.10598)) [S] and A2DEPT ([arXiv 2604.24043](https://arxiv.org/pdf/2604.24043)) [S]. Token-normalised comparison on its own is therefore no longer new.

### 1.3 Novelty matrix (October 2026 audit)

"n/r" means not retrieved by the research, not "absent". Treat those cells as unknown.

| Work | Domain / benchmark | LLM operator(s) | Current open-weight models? | Equal-budget no-search control | Token / cost ledger | Evaluator-exploit audit | Rediscovery / component audit | Held-out test | Status |
|---|---|---|---|---|---|---|---|---|---|
| **LLaMEA** ([arXiv](https://arxiv.org/html/2405.20132)) | Continuous, BBOB-24, 5-D | GPT-3.5/4/4o | No | Partial: 100-prompt LLM random search | Calls only (4,500) | No (errors → 0) | Qualitative ("ERADS ≈ JADE") | Instances + 10/20-D | TEVC 2025 [V] |
| **LLaMEA-HPO** ([arXiv](https://arxiv.org/abs/2410.16309)) | BBOB, bin packing, TSP | n/r | n/r | n/r | Fewer queries claimed | No | No | n/r | TELO 2025 [V] |
| **Mutation control** ([arXiv](https://arxiv.org/abs/2412.03250)) | LLaMEA setting | GPT-3.5, GPT-4o | No | No | No | No | No | n/r | EvoApps 2025 [V] |
| **BLADE** ([arXiv](https://arxiv.org/html/2504.20183v1)) | MA-BBOB, SBOX-COST, photonics | Gemini 1.5/2.0 Flash, Codestral-22B, Qwen2.5-Coder-14B | 2024-era, local | Not run (framework can host random search) | Logged, not reported | "Information exploitation" mentioned [S] | CEG tooling | 20 train / 50 test MA-BBOB | GECCO'25 workshop [V] |
| **Behaviour space** ([arXiv](https://arxiv.org/abs/2507.03605)) | BBOB, 10 functions, 5-D | o4-mini | No | No | No | No | Behaviour/STN, not vs canonical | 5 train / 10 validation instances [S] | Springer 2026 [V/S] |
| **Code Evolution Graphs** ([arXiv](https://arxiv.org/abs/2503.16668)) | Analysis of LLaMEA/HPO/EoH runs | Several (n/r) | n/r | — | — | — | AST features within lineages | — | GECCO 2025 [V] |
| **LLaMEA-SAGE** ([arXiv](https://arxiv.org/abs/2601.21511)) | MA-BBOB | n/r | n/r | No | Tokens per run [S] | No | XAI features used to steer, not to explain the final algorithm | MA-BBOB | Preprint 2026 [V] |
| **Huang et al., strong priors** ([arXiv](https://arxiv.org/abs/2603.02792)) | PBO, BBOB | Gemini, GPT, Qwen (version n/r) [S] | **Possibly; check** | n/r | n/r | n/r | Token attribution | n/r | Preprint 2026 [V/S] |
| **LLaMEA-BO** ([arXiv](https://arxiv.org/pdf/2505.21034)) | BBOB (Bayesian optimisation) | Single LLM | n/r | — | — | Anecdote: origin-sampling "cheat", instances excluded [S] | — | 10–40-D, Bayesmark (self-reported) | GECCO 2026 [SR] |
| **Šenkeřík et al.** ([AGH](https://badap.agh.edu.pl/publikacja/159982)) | Engineering, constrained | GPT-4o, 4o-mini, Claude 3.5 Sonnet, Llama 3.1 | Llama 3.1 | Random search and CMA-ES baselines | n/r | n/r | n/r | n/r | EvoApps 2025 [V] |
| **Zhang et al.** ([arXiv](https://arxiv.org/html/2407.10873v1)) | Combinatorial / math | 9 LLMs incl. CodeLlama, DeepSeek-Coder | 2024-era | **Yes** (standalone LLM) | $ per run | No | No | n/r | PPSN 2024 [V/S] |
| **LLM4AD** ([arXiv](https://arxiv.org/abs/2412.17287)) | Mostly combinatorial | 8 LLMs incl. Llama-3.1-8B [S] | 2024-era | Random sampling [S] | n/r | No | No | n/r | Preprint [V/S] |
| **EoH / ReEvo / MCTS-AHD** ([LLM4AD](https://github.com/Optima-CityU/LLM4AD)) | Combinatorial | GPT-3.5-class and others | Mostly no | ReEvo: one setting [S] | EoH about 2,000 queries [S] | No | No | — | ICML'24 / NeurIPS'24 / ICML'25 [V] |
| **FunSearch / AlphaEvolve / ShinkaEvolve** ([Nature](https://www.nature.com/articles/s41586-023-06924-6); [blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); [arXiv](https://arxiv.org/pdf/2509.19349)) | Math, systems | Codey / Gemini / ensemble | No | FunSearch: hand-crafted-mutation ablation ([Davis](https://cs.nyu.edu/davise/papers/FunSearch.pdf)) | Shinka ≈150 samples | No | No | — | Nature 2024 [B] / preprints [V] |
| **BehaveSim** ([arXiv](https://arxiv.org/abs/2603.02787)) | AAD tasks inside FunSearch/EoH | n/r | n/r | — | — | — | Behavioural similarity used for diversity and clustering | — | ICLR 2026 [V] |
| **Code Graph / A2DEPT** ([2605.10598](https://arxiv.org/html/2605.10598); [2604.24043](https://arxiv.org/pdf/2604.24043)) | AAD | Several model scales | n/r | — | **Equal-token comparison** | — | A2DEPT ablates its own framework | — | Preprints [S] |
| **MetaMind (planned)** | BBOB-24 via `cocoex`. Trained at 5-D; tested on held-out instances at 5/10/20-D and on shifted-and-rotated Ackley/Griewank outside the suite | gpt-oss-120b, gpt-oss-20b, Qwen3.8-27B (free tier) | **Yes, the 2025–26 reasoning generation** | **Yes, at equal calls and equal tokens** | **Full per-call ledger** | **Systematic, with a hardened harness** | **Component table + knock-out + nearest-canonical baseline** | Yes | Target: ICCI 2027 |

### 1.4 The defensible novelty claim

**Do not claim:**

- **A new algorithm.** LLaMEA's own winner was "very similar" to JADE.
- **The first open-model comparison.** BLADE, Zhang et al. and Šenkeřík et al. all did one.
- **The first token-budget comparison.** Code Graph and A2DEPT did.
- **The first awareness of evaluator cheating.** LLaMEA-BO and the Sakana CUDA case below came first.

**Claim the combination.** To our knowledge, as of October 2026 (the searches were not exhaustive), this is the first LLaMEA-style study of continuous black-box optimiser design that:

1. runs entirely at zero monetary cost under free-tier token caps;
2. uses current open-weight reasoning models, with a same-family size pair (gpt-oss-120b vs gpt-oss-20b);
3. tests the loop against an equal-call and equal-token feedback-free sampling control;
4. reports a per-call token and wall-clock ledger;
5. audits evaluator exploits in a hardened harness, and quantifies how often each model attempts them;
6. decomposes the best discovered algorithms into standard components, maps each to its nearest canonical algorithm, and runs that algorithm as a baseline.

The notes found no LLaMEA or BLADE result that uses gpt-oss or Qwen 3.x ([BLADE](https://arxiv.org/html/2504.20183)) [V], but that search was not exhaustive. The Huang et al. Qwen comparison must be checked.

**Recommended paper claim (title-level thesis):**

> **"Is the Loop Worth It? Zero-Cost LLaMEA with Open-Weight Reasoning Models, an Equal-Budget Sampling Control, and Exploit and Rediscovery Audits."**
>
> *Thesis sentence:* Under a zero-dollar budget of at most about 1M tokens per model per day, open-weight reasoning LLMs driving a LLaMEA (1+1) loop [MEASURE: do / do not] outperform feedback-free sampling at equal calls and tokens on held-out BBOB instances and dimensions. Their best designs [MEASURE: beat PSO/GA but not BIPOP-CMA-ES], and component analysis shows them to be [MEASURE: k of n] recombinations of DE, CMA-ES and PSO mechanisms.

The thesis is written so that **either outcome can be published**. "The loop does not beat sampling for small models" is as useful a finding as the reverse, and pre-registering the hypotheses (Section 4.1) is what makes a negative result credible.

---

## 2. The core method keeps LLaMEA's loop and changes only what the free tier forces

### 2.1 The AlgoGenesis-(1+1) loop, specified so Pratyush can build it today

The representation is the team's existing contract: a module that defines `optimize(func, dim, bounds, budget)` and returns `(best_x, best_f, history)`. The harness ignores the returned values except for sanity checks [R].

**Task prompt.** Adapt LLaMEA's task prompt to this signature: design a novel metaheuristic for box-constrained continuous minimisation, give the name on one line, then one fenced Python block. Copy the wording verbatim from the `llamea` source on PyPI (MIT licence, v1.4.2 released 2026-10-23) rather than from memory, and cite it ([PyPI llamea](https://pypi.org/project/llamea/1.4.2/)) [L]. The notes did not capture the verbatim prompt text. Add four interface rules from INTERFACE.md that LLaMEA did not need [R]:

- call `func` with one vector of shape `(dim,)` at a time;
- use `bounds` rather than literal box constants;
- use only `numpy`, `math` and `random`, with `np.random` global seeding (no internal reseeding);
- never touch attributes of `func`.

**Feedback prompt.** Use LLaMEA's structure:

- the task prompt;
- the names and scores of previous candidates, truncated to the last 10 to save tokens (a declared deviation);
- the elitist parent's code with its mean and standard deviation of AOCC;
- the last child's traceback, truncated to 20 lines, if the child failed;
- the instruction "Either refine or redesign to improve the algorithm."

**Selection, sampling and fitness.**

- **Selection.** Elitist (1+1): the child replaces the parent if its fitness is at least as high.
- **Sampling.** Temperature 0.8, as in LLaMEA, and `reasoning_effort="low"` for every reasoning model, set explicitly. Groq's documents disagree on Qwen3.8-27B's default ([Groq API reference](https://console.groq.com/docs/api-reference); [Groq Qwen page](https://console.groq.com/docs/model/qwen/qwen3.8-27b)) [V].
- **Fitness (frozen at gate G1).** Mean AOCC over BBOB f1–f24 at 5-D, instances 1–3, one seed per instance, with a budget of 2000·d = 10,000 evaluations. This matches LLaMEA's budget, and one candidate costs 72 runs and 720,000 evaluations [EST]. Each candidate gets a 300-second wall-clock limit. A crash, timeout or policy violation scores 0 and its message is fed back.

**Logging.** Write one JSONL record per call with:

- run ID, arm, model ID string, provider, `system_fingerprint`, timestamp, temperature, reasoning effort and seed parameter;
- prompt, completion and reasoning tokens, latency and 429 retries;
- the raw response and the hash of the extracted code;
- error class (syntax, interface, runtime, timeout, policy);
- fitness, per-function AOCC, lines of code and AST node count.

Archive every generated program. On Groq, determinism is "not guaranteed" even with a seed ([Groq API reference](https://console.groq.com/docs/api-reference)) [V]. Reproducibility therefore means re-evaluating the archived code, not re-sampling the LLM.

### 2.2 Three arms turn a replication into an experiment

**Arm A** is the LLaMEA (1+1) loop above, with T = 50 calls per run. That matches BLADE's 50 candidates per run in its LLM comparison ([BLADE](https://arxiv.org/html/2504.20183)) [V], and is half of LLaMEA's 100. State this as a forced deviation.

**Arm B** is the feedback-free control. It draws 80 independent samples from the task prompt alone, at the same temperature, and keeps the one with the best training fitness. The comparison with Arm A is made at **call 50 (equal calls)** and at **the call where Arm B's cumulative tokens match Arm A's (equal tokens)**. Arm B's prompts are shorter, so both views come from one run [EST]. This directly tests the Davis critique that the LLM is "a subroutine in a genetic programming algorithm" ([Davis](https://cs.nyu.edu/davise/papers/FunSearch.pdf)) [V], and the Zhang et al. finding that LLM generation alone is insufficient ([arXiv](https://arxiv.org/html/2407.10873v1)) [S].

**Arm C** is a history-only ablation, run for gpt-oss-120b only. The prompt keeps the names and scores of past candidates but drops the parent's code and the traceback. It asks whether the code in context matters, or whether an OPRO-style score history is enough. LLaMEA already found that richer per-group feedback "never showed an increase in performance" ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V], so this ablation removes information rather than adding it.

### 2.3 Harness changes that must land before any production run

The probe run today [R] wrapped three adversarial optimisers in the harness's `score()`:

| Exploit | Result | Status |
|---|---|---|
| `func(func.problem.x_opt)` (read the optimum) | **AOCC 1.000, error 0, 1 evaluation**, reported as a successful run | Confirmed hole |
| Reset `func.evals = 0` whenever the budget nears its end | Made 30,000 real calls on a 10,000 budget, yet the harness recorded `evals = 3` and padded AOCC from the overwritten trace | Confirmed hole: silent overrun and AOCC inflation |
| Bare `except:` around `func` | Swallowed `BudgetExhausted` (a `BaseException`), but the budget held at 10,000 evaluations | Wastes wall-clock time but cannot overrun |

The INTERFACE.md statement that the budget exception "cannot be caught with `except Exception`" is correct but incomplete: a bare `except:` or `except BaseException:` still catches it.

The fixes are cheap:

1. **Read-only objective.** Pass candidates a plain closure, not the `CountingFunction` object. Keep the counter state in the closure, and after the run check that the number of calls is at most the budget and that the trace is monotone.
2. **Static AST scan before execution.** It rejects, with feedback, any code that:
   - accesses an attribute of `func`;
   - uses dunder attributes such as `__closure__` or `__globals__`;
   - calls `getattr`, `setattr`, `vars`, `globals`, `eval`, `exec`, `compile` or `__import__`;
   - imports anything outside `numpy`, `math` and `random`;
   - does file I/O.

   It flags, without rejecting, bare `except` clauses, literal box constants such as ±5 and ±5.12, and internal `np.random.seed` calls. Internal reseeding makes "30 runs" into pseudo-replicates.
3. **Disjoint instances.** `score()` sets `instance = seed + 1`, `run_suite` starts at `first_seed = 0`, and `scripts/compare.py` exposes no seed offset [R]. A 30-run final evaluation would therefore reuse every training instance. Pass explicit, disjoint instance lists, and assert that they are disjoint in code.
4. **Canary tests.** Add the three probe optimisers above to `tests/`, and require each to be caught or scored as failed.

Python cannot be sandboxed in-process (INTERFACE.md already says the subprocess "is not a security sandbox" [R]). For the test-phase evaluation of the top algorithms only, the objective can also be evaluated in the parent process over a pipe. The overhead is unmeasured and estimated at tens of microseconds per call [EST].

---

## 3. Twelve enhancements ranked by impact against effort

The ranking weighs reviewer impact against person-days and against the token budget. Items 1–7 make up the paper. Items 8–9 are cheap add-ons. Items 10–12 are stretch goals and are the first things cut.

| # | Enhancement | Impact | Effort | Owner | Why it earns its place |
|---|---|---|---|---|---|
| **E1** | Harness hardening, exploit taxonomy, and per-model exploit-attempt counts | High | Low (0.5 day) | Siddhant | Closes the confirmed holes (§2.3) and turns them into a contribution. Precedents: LLaMEA-BO's origin-sampling anecdote ([arXiv](https://arxiv.org/pdf/2505.21034)) [S]; Sakana's CUDA Engineer, whose median speed-up fell from **1.13× to 0.82×** and successful tasks from **63 to 22** once hacks were removed ([TechCrunch](https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/embed/)) [V]; and a reward-hacking benchmark where simple hardening cut exploits by **87.7%** relative ([arXiv](https://arxiv.org/abs/2605.02964)) [S] |
| **E2** | Real BBOB-24 via `cocoex` with an explicit train/test split | High | Low–medium (0.5–1 day) | Siddhant | Removes the "why not BBOB?" objection and allows direct comparison with LLaMEA and BLADE. `coco-experiment` 2.8.3 ships cp314 Windows wheels and evaluates in about **2.6 µs**; `ioh` does not install on 3.14 ([PyPI](https://pypi.org/project/coco-experiment/); [PyPI ioh](https://pypi.org/project/ioh/)) [L] |
| **E3** | Multi-model, zero-cost comparison with a token and wall-clock ledger | High | Low–medium (runtime-bound) | Pratyush | The core empirical gap. gpt-oss-120b vs gpt-oss-20b gives a clean size pair within one family ([Groq rate limits](https://console.groq.com/docs/rate-limits)) [V] |
| **E4** | Equal-budget feedback-free control (Arm B) | High | Low (same loop with an empty history) | Pratyush | The control reviewers ask for first; aimed directly at the FunSearch/LLaMEA "is the LLM just a mutation operator" critique |
| **E5** | Strong baselines: DE/rand/1/bin, CMA-ES default, BIPOP-CMA-ES, random search, all behind the same budget wrapper | High | Low (0.5 day) | Siddhant | LLaMEA's own result flipped once CMA-ES was tuned (0.742 vs 0.733) ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V]. Without restarts, pycma stopped by itself after **10.7k of 20k** evaluations ([pycma docs](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html)) [L] |
| **E6** | Statistics upgrade: AOCC-based tests, Holm correction, Vargha-Delaney A12, Wilcoxon over functions, Friedman with critical-difference diagram, ECDF | Medium–high | Low (0.5 day) | Siddhant | `report.py` currently tests final error only, with no multiplicity correction [R]. Derrac et al. is the reference standard ([PDF](https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/sicidm/2011-Derrac-SWEVO.pdf)) [V] |
| **E7** | Rediscovery audit and component knock-out of the top 2–3 algorithms | High | Medium (1.5 days, evaluation only) | Om + Pratyush | Answers Sörensen and Camacho-Villalón with their own method ([Dorigo lab PDF](https://iridia.ulb.ac.be/~mdorigo/Published_papers/All_Dorigo_papers/CamDorStu2022itor.pdf)) [V]. The notes found no paper that systematically measures LLM-generated metaheuristics against canonical algorithms |
| **E8** | Transfer tests: dimension (5 → 10 → 20-D) and out-of-suite (shifted and rotated Ackley and Griewank on their native boxes) | Medium | Low | Pratyush | Reuses evolved code with no LLM calls. Native boxes (±32.768 and ±600) expose code that hard-codes ±5, as ERADS did ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V] |
| **E9** | History-only ablation (Arm C, gpt-oss-120b) | Medium | Low (0.45M tokens) | Pratyush | Isolates the value of seeing the parent's code |
| E10 | Local Qwen2.5-Coder-7B/14B via Ollama | Medium | Medium (GPU throughput unknown) | Pratyush | Links to BLADE's qwen2.5-coder:14b ([BLADE](https://arxiv.org/html/2504.20183)) [V]. Mixes model with provider and quantisation |
| E11 | Mixed small/large LLM operator, or BLADE-style refine/new/simplify prompt mix | Medium–high | Low code, high tokens | Pratyush | Suggested in CEG and LLaMEA future work ([CEG](https://arxiv.org/abs/2503.16668)) [V]. Competes for the same scarce tokens |
| E12 | Post-hoc Optuna tuning of the baselines and the best LLM algorithm at equal budget | Medium | Medium | Siddhant | Defuses the "the LLM loop is just tuning" critique. Future work unless E1–E9 finish early |

The notes deprioritise MAP-Elites or quality-diversity in code space (heavy, with close prior art in AlphaEvolve and LLaMEA's niching options), a classical GP head-to-head, and a full search-trajectory-network replication (already done for o4-mini) ([LLaMEA README](https://github.com/XAI-liacs/LLaMEA); [arXiv](https://arxiv.org/abs/2507.03605v1)) [V/S]. None belongs in an 8-day plan.

---

## 4. The evaluation protocol separates training, selection and testing

### 4.1 Pre-registered hypotheses

Om writes these down before production runs begin (by gate G1) and Siddhant commits them. That timestamp is what makes a null result publishable.

- **H1 (loop value):** for each model, the held-out test AOCC of the best-of-run algorithm is higher in Arm A than in Arm B, at equal calls and at equal tokens.
- **H2 (size within a family):** gpt-oss-120b beats gpt-oss-20b on best-of-run test AOCC and on valid-code rate, and uses fewer tokens per accepted improvement.
- **H3 (strong baselines):** the best LLM algorithm beats PSO and GA, is comparable to default CMA-ES at 5-D, and loses to BIPOP-CMA-ES at 20-D. This is the prediction implied by LLaMEA's own generalisation result, not a claim.
- **H4 (rediscovery):** most top algorithms reduce, component by component, to DE, CMA-ES or PSO mechanisms.
- **H5 (exploits):** exploit attempts and policy violations occur at a non-zero, model-dependent rate, and every one is caught by the hardened harness.

### 4.2 Benchmark, split and budget

**Benchmark.** Use BBOB f1–f24 from `cocoex` on [−5, 5]^d. `cocoex.Suite("bbob", "", "dimensions:10,20 instance_indices:1-15")` built 720 problems on this laptop ([COCO](https://numbbo.github.io/coco/)) [L].

COCO's standard dimensions are 2, 3, 5, 10, 20 and 40, so **move the BBOB experiments from 10/30-D to 5/10/20-D**. Keep 30-D only on the team's own classical suite for continuity with the course. That 30-D is not a COCO dimension is an inference from the BBOB convention; check it on coco-platform.org [S].

The cocoex API details were not checked: how to read f_opt (needed to compute error f − f_opt), and whether arbitrary instance numbers such as 4–18 can be requested. Settle both on day 0 (verification item 3).

**Train/test split.**

| Split | Functions | Dimension | Instances | Runs | Budget | Purpose |
|---|---|---|---|---|---|---|
| Training (inside the loop) | f1–f24 | 5 | 1–3 | 1 seed each | 10,000 (2000·d) | Candidate fitness, as in LLaMEA ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V] |
| Selection | — | — | — | — | — | Best-of-run is chosen **on training fitness only**, never on test data |
| Test: held-out instances | f1–f24 | 5 | 4–18 (15 instances) | × 2 seeds = 30 | 2000·d | Instance generalisation |
| Test: held-out dimensions | f1–f24 | 10, 20 | 4–18 | 30 | 2000·d (20k, 40k) | Dimension generalisation (LLaMEA's known weakness) |
| Test: out-of-suite | Shifted + rotated Ackley, Griewank (+ Rastrigin, Sphere) | 10, 30 | Seeds ≥ 1000 | 30 | 2000·d | Function transfer and box-hard-coding check |

Avoid BLADE's default overlap, where instance 5 is in both the training and test sets ([iohblade source](https://pypi.org/project/iohblade/1.1.1/)) [L].

**Baselines.** Run each with the same budget, bounds and initialisation box, behind the counting wrapper:

- random search;
- PSO (Clerc-Kennedy constants, as now);
- GA (as now, flagged as weakly configured);
- DE/rand/1/bin with NP = 10·d, F = 0.5, CR = 0.9, written by hand so the budget is exact;
- pycma CMA-ES with σ0 = 2.5, in two versions: default without restarts (LLaMEA's baseline) and BIPOP with `restarts=9, bipop=True` (the strong baseline reviewers expect) ([pycma docs](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html)) [V];
- the Arm B best-of-run and the initial (generation-0) algorithm, as internal ablations.

Avoid SciPy's `differential_evolution` defaults. Its population is 15·N and polishing spends evaluations beyond `maxiter` ([SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)) [V].

### 4.3 Metrics and statistics

**Metrics.**

- **Primary:** AOCC per run. The harness formula already matches the `llamea`/`iohblade` code: log10 of the error clipped to [1e-8, 1e2], best-so-far, padded with the final value, failures scored 0 ([PyPI llamea](https://pypi.org/project/llamea/1.4.2/)) [L].
- **Secondary:**
  - final log10 error at budget B;
  - success rate at 1e-8;
  - runtime ECDFs over **51 log-spaced targets from 1e2 to 1e-8**, aggregated per BBOB group and per dimension, never across dimensions ([COCO performance assessment](https://arxiv.org/html/1605.03560v1)) [V];
  - median/IQR convergence curves for f1, f3, f10, f15 and f21.

LLaMEA states that AOCC "is equivalent to the area under the so-called ECDF curve" ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V]. That holds on a linear budget axis. COCO plots ECDFs on a log axis, which weights early progress more [EST]. Say so in the paper.

**Statistics.**

- **Within one function and dimension:** two-sided Mann-Whitney U on 30 AOCC values (primary) and on final error (secondary, keeping the existing rule that errors ≤ 1e-8 tie). Apply Holm correction across each table's family. With 24 functions × 3 dimensions × 6 baselines, that is **432 tests** [EST]. Report A12 next to every p-value; 0.56, 0.64 and 0.71 are the conventional small, medium and large thresholds ([Vargha & Delaney DOI](https://doi.org/10.3102/10769986025002101)) [B].
- **Across functions:** Wilcoxon signed-rank on per-function median AOCC (24 pairs per dimension), plus Friedman with Holm or Nemenyi post-hoc tests and a critical-difference diagram ([Derrac et al.](https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/sicidm/2011-Derrac-SWEVO.pdf); [Demšar](https://www.jmlr.org/papers/v7/demsar06a.html)) [V/B].
- **Across LLM runs (H1, H2):** the sample is 5 best-of-run algorithms per arm. **With 5 against 5, the smallest possible two-sided Mann-Whitney p-value is 2/252 ≈ 0.008** [EST]. Only complete separation can survive correction. So pre-register **A12 with a bootstrap CI as the primary run-level statistic**, plot all five points, and present p-values as secondary. This is the honest answer to the third-party criticism that the behaviour-space ranking rests on 5 runs ([Pith review](https://pith.science/paper/2507.03605)) [S].
- **Anytime search curves:** best-so-far training fitness against call index and against cumulative tokens, with median and min/max over runs.
- **Optional:** one Bayesian signed-rank plot (`baycomp`, ROPE 0.01 AOCC) for the main comparison with BIPOP-CMA-ES ([Benavoli et al.](https://www.jmlr.org/papers/v18/16-305.html)) [V].

### 4.4 Audits that run on the archive with no LLM calls

**Exploit audit.** Run the static scanner over every archived program and tabulate, per model:

- attempts to access `func` attributes;
- forbidden imports or calls;
- bare `except` clauses;
- literal box constants;
- internal reseeding;
- mismatches between the self-reported and harness-measured best value;
- timeouts.

Then run two dynamic checks on the top algorithms. First, a **centre-bias test**: compare performance on unshifted instances with performance on instances whose optimum is near the boundary. Kůdela found centre-bias operators in **47 of 90** examined methods ([arXiv](https://arxiv.org/abs/2301.01984v1)) [V]. Second, the out-of-suite box test.

**Rediscovery audit.** For the top 2–3 algorithms:

1. Strip the LLM's name.
2. Fill in a component table covering initialisation, population structure, variation operators (DE mutation type, Gaussian/Cauchy, PSO velocity, covariance adaptation), selection, parameter adaptation (JADE/SHADE success history, 1/5th rule), restarts, local search and boundary handling.
3. Name the nearest canonical algorithm.
4. Knock out components one at a time and re-measure test AOCC on 24 functions × {5, 10}-D × 10 runs.
5. Report the AST-token similarity to the reference PSO/DE/CMA-ES/GA code as a descriptive statistic only.

The precedent is LLaMEA's admission that ERADS ≈ JADE ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V], and BehaveSim's argument that syntactic similarity misses algorithmic similarity ([arXiv](https://arxiv.org/abs/2603.02787)) [V].

### 4.5 Evaluation compute fits a laptop

In the loop, one candidate costs 720k evaluations. At an observed **13.7 µs per evaluation**, that is **about 10 seconds per candidate**, or about 4.5 CPU-hours for about 1,650 candidates [EST]. The 13.7 µs figure is PSO's total cost per evaluation at 10-D in the repo [R]. LLM-generated code is plausibly 2–5× slower, which gives 10–25 CPU-hours spread across concurrent runs [EST].

In the test phase, about 27 LLM algorithms plus 7 baselines × 24 functions × 30 runs × (10k + 20k + 40k) evaluations come to about **1.7 × 10⁹ evaluations**. That is about 6.5 CPU-hours at PSO-like overhead, so an afternoon on 6–8 cores, with more for slow generated code [EST]. The 300-second limit per candidate in the loop, and a limit per run in the test phase, contain any pathological O(n²) code.

---

## 5. The free-LLM compute plan is built around Cerebras because Groq caps each model at about 50 calls a day

### 5.1 What is actually free on 2026-10-28

Groq's free plan now has **three usable text LLMs: `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` (Preview)**. Each has **30 RPM, 1K RPD, 8K TPM and 200K TPD**, applied per organisation ([Groq rate limits](https://console.groq.com/docs/rate-limits)) [V]. Groq deprecated these for free and developer users ([Groq deprecations](https://console.groq.com/docs/deprecations)) [V]:

| Model | Free/developer shutdown |
|---|---|
| llama-3.1-8b-instant | 2026-08-16 |
| llama-3.3-70b-versatile | 2026-08-16 |
| Llama-4-Scout | 2026-07-17 |
| Qwen3-32B | 2026-07-17 |
| Llama-4-Maverick | 2026-03-09 |
| Kimi K2 (0905) | 2026-04-15 |

The two Llama 3.x models survive only for Enterprise customers ([Groq models](https://console.groq.com/docs/models)) [V].

Preview models "may be discontinued at short notice". Qwen3.6-27B was itself replaced by Qwen3.8-27B on 2026-09-14 ([Groq models](https://console.groq.com/docs/models); [deprecations](https://console.groq.com/docs/deprecations)) [V].

**Cerebras's free trial** gives gpt-oss-120b and qwen-3.8-27b each **5 RPM, 30K uncached TPM and 1M TPD** ([Cerebras rate limits](https://inference-docs.cerebras.ai/support/rate-limits)) [V]. The other options are weaker:

- **OpenRouter `:free`:** 50 requests per day, or 1,000 once 10 credits have been bought ([OpenRouter limits](https://openrouter.ai/docs/api-reference/limits)) [V].
- **GitHub Models:** fully retired on 2026-07-30 ([GitHub Docs](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models)) [V].
- **Hugging Face:** gives free users no inference credits ([HF pricing](https://huggingface.co/docs/inference-providers/pricing)) [V].

The paper must therefore say "open-weight", not "Llama". gpt-oss-120b is Apache-2.0, a mixture-of-experts model with 117B total and 5.1B active parameters ([HF model card](https://huggingface.co/openai/gpt-oss-120b)) [V]. The Qwen3.8 licence was not checked.

### 5.2 Budget arithmetic and allocation

A LLaMEA call is about 2–3K prompt tokens plus 1–2K output tokens, and gpt-oss's hidden reasoning tokens count toward the caps ([Groq API reference](https://console.groq.com/docs/api-reference)) [V]. The plan therefore budgets **4K tokens per Arm A call, 3K per Arm C call and 2.5K per Arm B call** [EST]. The day-0 pilot must replace these figures [MEASURE].

At those rates, Groq allows about **50 calls per model per day** and Cerebras about **250** [EST]. The notes' own figure, about 57 calls per day at 3.5K tokens, gives **8.75 days for 5 × 100 calls on Groq alone**, which misses 2026-11-05 ([Groq rate limits](https://console.groq.com/docs/rate-limits)) [EST].

| Model | Provider (free) | Daily cap | Arms × runs × calls | Planned calls | Tokens [EST] | Quota-days | LLM window |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | Cerebras free trial | 1M TPD, 5 RPM | A 5×50, B 5×80, C 3×50 | 800 | ≈2.45M | ≈2.5 | Oct 29 – Nov 1 |
| qwen-3.8-27b | Cerebras free trial | 1M TPD, 5 RPM | A 5×50, B 5×80 | 650 | ≈2.0M | ≈2.0 | Oct 29 – Oct 31 (Preview, so run it first) |
| gpt-oss-20b | Groq free | 200K TPD | A 3×50 guaranteed (+1 if the pilot comes in under 4K tokens); B 2×80 only if quota remains | 150–360 | 0.6–1.2M | 3–6 | Oct 29 – Nov 1 (the tightest line) |
| gpt-oss-120b, qwen3.8-27b | Groq free | 200K TPD each | Pilot, retries, knock-out/simplify calls, optional one-run provider check | ≤300 | Reserve | — | Oct 28 – Nov 3 |
| qwen2.5-coder 7B/14B | Local Ollama | None | Stretch: A 3×50 | 150 | — | Overnight | Only if GPU throughput ≥ about 1 call/min [MEASURE] |

That is **about 1,600–1,800 calls in total, at $0**. LLaMEA used 4,500 calls on paid GPT models ([arXiv HTML](https://arxiv.org/html/2405.20132)) [V].

Three operational rules follow from the limits:

1. **Throttle each model across processes.** Cerebras allows 5 RPM per model, so 4–5 concurrent runs calling every 30–60 seconds will hit 429s. Put one token-bucket limiter per model in front of all runs, or cap concurrency at 3 runs per model [EST].
2. **Keep a local daily-token ledger.** Groq's headers report requests per day and tokens per minute but not tokens per day ([Groq rate limits](https://console.groq.com/docs/rate-limits)) [V]. When a model nears its cap, stop that arm; do not retry.
3. **Serve each model from one provider.** Groq and Cerebras may run different builds or quantisations, so a model's main runs must not mix providers.

### 5.3 Account hygiene the team must do itself

Pratyush must sign up for the Cerebras free trial and turn on Groq Zero Data Retention in Data Controls. ZDR is free and disables only batch and fine-tuning ([Groq your data](https://console.groq.com/docs/your-data)) [V]. **Do not create extra Groq organisations to multiply quotas.** Limits are per organisation, and splitting accounts may breach the terms (not checked) ([Groq rate limits](https://console.groq.com/docs/rate-limits)) [V]. Neither Cerebras's free-trial duration nor its daily request cap is published, so the plan front-loads Cerebras runs.

---

## 6. The narrative turns a course project into a methodology contribution

For the Bio-Inspired Algorithms course, MetaMind is the field's own critique turned into a measurement. Sörensen argued that metaphor-based methods threaten "to lead the area of metaheuristics away from scientific rigor" ([ITOR record](https://iaorifors.com/paper/87759)) [V]. Camacho-Villalón, Dorigo and Stützle showed that the grey wolf, firefly and bat algorithms reiterate PSO ideas ([Springer](https://link.springer.com/doi/10.1007/978-3-030-60376-2_10)) [V]. One researcher prompted ChatGPT to invent "Zoological Search Optimization" and got the paper published ([arXiv](https://arxiv.org/pdf/2403.16417)) [V]. An LLM loop is a machine for producing exactly such algorithms. The deck's story is that MetaMind builds the machine and then audits it: component tables in place of names, knock-outs in place of metaphors, and a sampling control in place of hype.

For the conference audience, the selling point is **democratised, reproducible automated algorithm design**. Prior LLM studies of algorithm design cost $10–$200 per run on closed APIs ([arXiv HTML](https://arxiv.org/html/2407.10873v1)) [V/S]. MetaMind shows what a student team in Bengaluru can do with $0, three open-weight models and a laptop. It also releases what others need to re-evaluate everything without an LLM: the hardened harness, the exploit canaries and the archived JSONL.

The **three-minute pitch** follows that arc:

1. *Hook (20 s):* LLMs can now write optimisers that beat CMA-ES on a benchmark, but LLaMEA's winner was JADE in disguise, and it cost GPT-4 money.
2. *Question (20 s):* can free open-weight models do it, and is the evolutionary loop doing the work?
3. *Method (60 s):* one loop, three models, a sampling control, a hardened harness.
4. *Findings (60 s), filled from runs:* [MEASURE] loop vs sampling at equal tokens; [MEASURE] exploit attempts caught; [MEASURE] components of the best design.
5. *Takeaway (20 s):* what $0 buys, and what it does not.

---

## 7. The 8-day execution plan puts every LLM-dependent decision before Thursday noon

### 7.1 Conventions and gates

**Conventions.**

- Siddhant makes every git commit. The plan names commits but never assumes anyone else writes git history.
- All runs write JSONL to `results/llm_runs/<model>/<arm>/<run>.jsonl`, and generated code goes to `found/`.
- Dates are IST.
- "Freeze" means no further change to the fitness function, prompts or splits.

**Gates.**

| Gate | When | Exit condition |
|---|---|---|
| G0 | Wed 28 Oct, 23:00 | The three canary exploits are caught. The `cocoex` wrapper returns the correct error f − f_opt on f1/f3 at 5-D. A 10-call pilot per model gives tokens/call (including reasoning) and the valid-code rate. |
| G1 | Thu 29 Oct, 12:00 | Fitness, prompts, splits and hypotheses are frozen and committed, and production runs start. |
| G2 | Sun 1 Nov, 23:59 | LLM cutoff. Whatever has finished is the dataset. |
| G3 | Tue 3 Nov, 12:00 | All test-phase numbers and statistics are frozen. |
| G4 | Wed 4 Nov, 18:00 | Full draft reviewed. |
| G5 | Thu 5 Nov | Submission-ready PDF and reproducibility package. |

### 7.2 Day-by-day, by person

| Day | Siddhant (harness, baselines, comparison) | Pratyush (LLM loop, prompts, Ackley/Griewank) | Om (literature, paper, deck) |
|---|---|---|---|
| **Wed 28 Oct** (today) | Closure-based read-only objective; post-run count and monotonicity checks; static AST scanner; the three probe exploits as `tests/` canaries; explicit disjoint instance lists. `pip install coco-experiment cma` into the existing 3.14 venv and write the BBOB wrapper (settle f_opt and instance numbering). **G0.** | OpenAI-SDK client with a swappable `base_url` (Groq, Cerebras); per-model token-bucket limiter; daily token ledger; JSONL logger; code extraction. Port the verbatim LLaMEA prompt from the `llamea` source and adapt it to `optimize(...)`. Cerebras sign-up and Groq ZDR. 10-call pilot on each model through the Groq reserve. | Verification items 1–4 (§10): Huang et al. Qwen version; LLaMEA TEVC PDF run count and prompt; ICCI template and page limit; the LLaMEA-BO exploit passage. Open the paper skeleton in the venue template. |
| **Thu 29 Oct** | Morning: freeze the in-loop evaluator (**G1**). Afternoon: DE/rand/1/bin, CMA-ES default, BIPOP-CMA-ES and random search behind the wrapper; start the baseline test-phase runs. | Morning: (1+1) loop and Arm B, dry run of 5 calls. **12:00: launch** Qwen3.8 A×5 and gpt-oss-120b A×5 on Cerebras (≤3 concurrent per model) and gpt-oss-20b A run 1 on Groq. Monitor 429s and the ledger. | Write the pre-registration (H1–H5, splits, statistics) for Siddhant to commit before G1. Related work from §1 and the novelty matrix. |
| **Fri 30 Oct** | `report.py` upgrade: AOCC tests, Holm, A12, Wilcoxon over functions, Friedman + CD diagram, ECDF over 51 targets from harness traces. Re-run PSO/GA under the new protocol. | Launch Qwen3.8 Arm B ×5 and gpt-oss-120b Arm B ×5; gpt-oss-20b A run 2. Add shifted and rotated Ackley and Griewank to `benchmarks.py` on their native boxes (out-of-suite test). | Methods section: loop, arms, harness hardening, compute plan. Draft the exploit-taxonomy table. |
| **Sat 31 Oct** | First exploit-audit pass over the archive so far (static scan + self-report mismatch). Test-phase pipeline for the LLM algorithms: 5/10/20-D BBOB on held-out instances plus out-of-suite; dry run on early best-of-run code. | gpt-oss-120b Arm C ×3; gpt-oss-20b A run 3. Qwen should finish (Preview risk retired). Start the error-class breakdown (syntax, interface, runtime, timeout, policy) per model. | Component-table template. Label components of the early best algorithms as they appear. Outline the course deck. |
| **Sun 1 Nov** | Run the test phase on every finished best-of-run algorithm (parallel). Centre-bias and box-hard-coding checks on the current leaders. | gpt-oss-20b A run 4 if quota allows, else Arm B ×2. **G2 at 23:59:** stop all LLM runs; archive and checksum the JSONL. Stretch: local Qwen2.5-Coder overnight only if the pilot showed ≥1 call/min. | Results-section skeleton with placeholder tables and figures matching `report.py` outputs. |
| **Mon 2 Nov** | Finish the test phase; full statistics tables; ECDF and convergence figures; cost/ledger table. | Knock-out ablation of the top 2–3 algorithms (evaluation only). Optional "simplify without changing behaviour" calls from the Groq reserve. Token-vs-fitness anytime curves. | Rediscovery audit write-up: nearest canonical algorithm per top design, with Pratyush's knock-out numbers. |
| **Tue 3 Nov** | **G3 by noon.** Write the results section (baselines, H1–H3, H5). | Write the LLM-behaviour section (H2): valid-code rates against LLaMEA's 18.7%, tokens per improvement, code growth. | Introduction, discussion, limitations, threats to validity (from §9). Siddhant sends the draft to Prof. Pramoda R. |
| **Wed 4 Nov** | Reproducibility package: code, prompts, seeds, JSONL, generated algorithms, pinned `requirements.txt` with Python 3.14.7 and package versions (Siddhant commits; Zenodo/GitHub). Re-run the figure scripts from a clean checkout. | Check every number in the text against the JSON outputs. Second pass on the prompts appendix. | Arxiv scoop check ("LLaMEA", sorted by date, since 2026-10-01). Address faculty comments. Course deck. **G4.** |
| **Thu 5 Nov** | Final PDF build; checklist (anonymisation if required, page limit, references resolve). | Final read of method and appendix. | Final read; cover note; rehearse the pitch and the Q&A in §8. **G5: submission-ready.** |

**Slack is deliberately zero.** The ICCI deadline of 2026-11-13 (as told; check the official site) leaves eight days of conference slack. The firm personal target is 2026-11-05, so the plan does not lean on that slack.

### 7.3 What to cut, in order, if time runs short

Cut from the top of this list first:

1. Local Ollama model (E10).
2. Mixed-LLM and prompt-mix variants (E11).
3. Optuna tuning (E12).
4. The Groq provider-check run.
5. Bayesian signed-rank test.
6. CEC 2022, 40-D and IOHanalyzer/cocopp export.
7. Arm C (E9).
8. gpt-oss-20b reduced to 3 runs with no Arm B.
9. 20-D test reduced to 10-D only.
10. Knock-out reduced to the single best algorithm.

**Never cut:**

- the harness fixes and disjoint split (E1);
- Arm B (E4);
- CMA-ES/BIPOP/DE baselines (E5);
- corrected statistics (E6);
- the JSONL archive.

Without these the paper does not survive review.

**Fallback if `cocoex` is not working by Thursday 09:00:** freeze the training fitness on the team's own suite (Sphere, Rastrigin, Ackley, Griewank, Rosenbrock, Schwefel) with random rotations added, and call it a BBOB-like suite. Run BBOB in the test phase only, if it works by Sunday. This costs comparability with LLaMEA's numbers, so state it as a limitation.

---

## 8. Likely reviewer questions already have answers in the design

**"Isn't this just LLaMEA again?"**
> "The loop is deliberately LLaMEA's (1+1), so the comparison is clean. What's new is the evidence around it: an equal-call and equal-token sampling control, a same-family size pair of current open-weight reasoning models, a per-call cost ledger, an exploit audit in a hardened harness, and a component-level rediscovery audit. LLaMEA reported none of the control, tokens or exploit data ([arXiv HTML](https://arxiv.org/html/2405.20132))."

**"BLADE already showed open models are competitive."**
> "BLADE compared 2024-era coder models (Codestral-22B, Qwen2.5-Coder-14B) against Gemini Flash, and found Gemini significantly better in 2 of 10 setups ([BLADE](https://arxiv.org/html/2504.20183v1)). We test the 2025–26 reasoning generation (gpt-oss, Qwen3.8), and ask a different question: does the loop add value over sampling for these models?"

**"Why not GPT-4 or Llama?"**
> "The study is about what a zero-cost budget buys. Llama 3.x was withdrawn from Groq's free tier on 2026-08-16 ([Groq deprecations](https://console.groq.com/docs/deprecations)). gpt-oss is Apache-2.0 and downloadable, so the work can be re-run locally even if hosted endpoints disappear."

**"Five runs per configuration is too few."**
> "It matches LLaMEA's k = 5. We pre-registered effect sizes with bootstrap CIs as the primary run-level statistic, because with 5 against 5 the smallest attainable Mann-Whitney p is about 0.008. Each discovered algorithm is then tested over 30 runs per function and dimension on held-out instances, with Holm correction."

**"Your algorithm is just DE (or CMA-ES) renamed."**
> "Probably, and we measured it. The rediscovery section of the paper decomposes each top design into components, names the nearest canonical algorithm, runs that algorithm as a baseline, and knocks out each component. We make no novel-algorithm claim."

**"Did the LLM cheat?"**
> "It tried [MEASURE] times. Our first harness let a candidate read the optimum (AOCC 1.0 after one evaluation) and reset the evaluation counter. We fixed both, added canary tests, and report attempt rates per model."

**"The LLM loop spent about 1,600 calls of 'tuning' while the baselines used defaults."**
> "That's why BIPOP-CMA-ES is included, why the selection uses training instances only, and why we report the design-phase cost next to the results. Tuning the baselines at equal budget is listed as future work, or included if time allowed."

**"Why 5-D training? It doesn't generalise."**
> "It matches LLaMEA's protocol, and the generalisation failure is a known LLaMEA finding we test on held-out 10-D and 20-D instances and on out-of-suite functions with different boxes."

**"Results aren't reproducible because hosted models change."**
> "Every prompt, response, `system_fingerprint` and program is archived. The released code re-evaluates every discovered algorithm without any LLM, and the models are open-weight."

**"Why does this belong at a computational intelligence conference?"**
> "It is evolutionary computation with an LLM variation operator, and a study of how much the evolutionary part contributes. The behaviour-space analysis of LLaMEA appeared in a Springer computational intelligence volume ([Springer](https://link.springer.com/chapter/10.1007/978-3-032-15635-8_23))."

---

## 9. Risk register: three confirmed harness holes, one schedule risk, and the usual threats to validity

| # | Risk / critique / threat | Likelihood × impact | Pre-emption |
|---|---|---|---|
| R1 | **Optimum readable through `func.problem.x_opt`** (confirmed: AOCC 1.000 after 1 evaluation) [R] | Certain × fatal | Closure objective, AST scan, canary test (§2.3) |
| R2 | **Budget counter writable** (`func.evals = 0`: 3× overrun, undetected, AOCC inflated) [R] | Certain × fatal | Private counter, post-run checks, canary; parent-side evaluation for the top algorithms in the test phase |
| R3 | Bare `except:` swallows `BudgetExhausted`, so candidates loop until timeout [R] | Likely × wasted compute | Static flag, a prompt rule, the 300 s limit; count these as timeouts |
| R4 | **Train/test instance leakage** (`instance = seed + 1`; final runs reuse instances 1–5) [R] | Certain × invalidates generalisation claims | Explicit disjoint instance lists with an assertion; out-of-suite seeds ≥ 1000 |
| R5 | Separable, unrotated functions let coordinate-wise tricks look strong | Likely × major | BBOB rotations; rotate Ackley and Griewank |
| R6 | Centre bias, or a box hard-coded to ±5 (ERADS initialised in (−5, 5) ([arXiv HTML](https://arxiv.org/html/2405.20132))) | Likely × major | Shifted and boundary-near optima; native-box out-of-suite test; literal-constant flag ([Kůdela](https://www.fme.vutbr.cz/en/veda/publikace/detail/179505)) |
| R7 | Internal `np.random.seed(c)` turns 30 runs into pseudo-replicates | Possible × major (variance understated) | AST flag; reject or re-run with the seed call stripped; report the count |
| R8 | Self-reported results trusted | Low (the harness measures them) × fatal | Keep harness-measured traces; report mismatch rates as an exploit category |
| R9 | AOCC dynamic range: Rastrigin at 30-D gives AOCC 0.008–0.05 against 0.6–0.7 on Sphere [R], so mean fitness rewards unimodal speed | Certain × moderate | Same choice as LLaMEA (mean over 24), but report per-group AOCC and per-group ECDFs; discuss as a threat to validity |
| R10 | Groq free tier limited to about 50 calls per model per day; Llama gone ([Groq](https://console.groq.com/docs/rate-limits)) | Certain × schedule | Cerebras as the main engine; Groq as reserve; T = 50; token ledger |
| R11 | Qwen3.8-27B Preview withdrawn mid-run ([Groq models](https://console.groq.com/docs/models)) | Possible × moderate | Run it first on Cerebras; archive; the paper's core pair is gpt-oss-120b/20b |
| R12 | Cerebras trial expires or has an unpublished daily request cap ([Cerebras](https://inference-docs.cerebras.ai/support/rate-limits)) | Unknown × major | Front-load Cerebras; Groq reserve for gpt-oss-120b; record the provider per call |
| R13 | Hidden reasoning tokens blow the budget ([Groq API](https://console.groq.com/docs/api-reference)) | Likely × moderate | `reasoning_effort="low"`; day-0 pilot; recompute the allocation at G0 |
| R14 | LLM non-determinism and provider drift ([Groq API](https://console.groq.com/docs/api-reference); [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)) | Certain × moderate | Log `system_fingerprint`; archive; reproducibility means re-evaluation |
| R15 | Provider confound (Cerebras vs Groq builds) | Likely × moderate | One provider per model; state it; optional provider-check run |
| R16 | Valid-code rate not comparable to LLaMEA's 18.7% because our interface rules (shape, signature) add failure modes | Certain × minor | Break errors down by class; compare syntax and runtime errors only |
| R17 | "Fake novelty" and metaphor names ([Camacho-Villalón et al.](https://iridia.ulb.ac.be/~mdorigo/Published_papers/All_Dorigo_papers/CamDorStu2022itor.pdf)) | Certain × major | Strip names; component tables; nearest-canonical baseline; no novel-algorithm claim |
| R18 | "The LLM is only a mutation operator" ([Davis](https://cs.nyu.edu/davise/papers/FunSearch.pdf)) | Certain × major | Arm B; report design-phase cost |
| R19 | Weak baselines (GA leaves 7e-4 on 10-D Sphere [R]; untuned PSO) | Certain × major | CMA-ES default and BIPOP, DE; call the GA "textbook" |
| R20 | Underpowered run-level tests (5 vs 5) | Certain × moderate | Pre-registered A12 + bootstrap CI as primary statistic; show all points |
| R21 | Selecting the best algorithm on test data | Possible × fatal | Selection on training fitness only, written into the pre-registration |
| R22 | Scooped by fast-moving 2026 work (SAGE, Huang et al., Code Graph, A2DEPT) | Possible × major | Day-0 Huang check; Nov 4 arXiv scan; claim the combination |
| R23 | `ioh` does not install on Python 3.14 [L]; `llamea` reuse may depend on `ioh` (not checked) | Certain × minor | Use `cocoex` (cp314 wheel) plus the team's own AOCC; vendor the `llamea` prompts, not its runtime |
| R24 | `cocoex` API unknowns (reading f_opt, arbitrary instance numbers) | Possible × schedule | Day-0 spike; the fallback suite in §7.3 |
| R25 | Rate-limit collisions from concurrent runs (Cerebras 5 RPM) | Likely × minor | Per-model limiter; at most 3 concurrent runs per model |
| R26 | **Team bandwidth: TwitchNet Analytics and NaviGuard also need to be submission-ready by 2026-11-05**, and one paper can go to only one conference | Certain × **highest schedule risk** | Assign MetaMind to ICCI 2027 (fits "computational intelligence"); send the others to ICCSMLAI / ICAI-27 (2027-01-04), but keep the 11-05 readiness target; Siddhant's harness work is front-loaded to days 0–2 so his later days can be shared. Confirm deadlines and page limits on the official sites |
| R27 | Windows `spawn` multiprocessing overhead and `__main__` guard errors in the loop [R] | Possible × minor | Keep the loop entry under `if __name__ == "__main__"`; reuse worker pools in the test phase |

---

## 10. Conclusion

The strongest version of MetaMind is a careful look at a celebrated method, not a race to beat it. The free tier makes things harder: about 1,600 calls instead of 4,500, and a 50-call run instead of 100. It also forces the question that richer projects can skip: when compute is scarce, is the evolutionary loop worth its tokens compared with sampling? That question, together with a harness that has been shown to resist its own generated code, gives the paper a result that matters whichever way the numbers fall.

The audit also changed the order of the work. The most valuable work this week is defensive and happens before any LLM call: closing the two holes that let a candidate score a perfect AOCC without optimising, and separating training instances from test instances. A reviewer who finds either of those problems after submission would sink the paper, whatever the LLM results show.

---

## 11. References

Status tags as defined above. "Verify" marks items to check before quoting.

**LLaMEA line**
- van Stein, N., Bäck, T. "LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics." IEEE TEVC 29(2):331–345, 2025. DOI 10.1109/TEVC.2024.3497793. https://arxiv.org/abs/2405.20132 [V]. Run count and verbatim prompt: verify in the PDF.
- van Stein, Vermetten, Bäck. "In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics." ACM TELO 2025. https://arxiv.org/abs/2410.16309 [V]
- Yin, Kononova, Bäck, van Stein. "Controlling the Mutation in Large Language Models for the Efficient Evolution of Algorithms." EvoApplications 2025, LNCS pp. 403–417. https://arxiv.org/abs/2412.03250 [V]
- van Stein, Kononova, Kotthoff, Bäck. "Code Evolution Graphs." GECCO 2025. https://arxiv.org/abs/2503.16668 [V]
- van Stein, Kononova, Yin, Bäck. "BLADE." GECCO 2025 Companion, DOI 10.1145/3712255.3734347. https://arxiv.org/abs/2504.20183 [V]
- van Stein, Yin, Kononova, Bäck, Ochoa. "Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery." Springer 2026. https://arxiv.org/abs/2507.03605 [V/S]
- van Stein, Kononova, Kotthoff, Bäck. "LLaMEA-SAGE." arXiv 2601.21511, 2026. https://arxiv.org/abs/2601.21511 [V]
- Huang, Ye, Shahane, Bäck, van Stein. "From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from Strong Priors." arXiv 2603.02792, 2026. https://arxiv.org/abs/2603.02792 [V/S]. **Verify the model list.**
- Li, van Stein, Bäck, Raponi. "LLaMEA-BO." arXiv 2505.21034. https://arxiv.org/pdf/2505.21034 [S; GECCO 2026 SR]. Verify the exploit passage.
- van Stein, Kononova, Bäck. "From Performance to Understanding: A Vision for Explainable Automated Algorithm Design." arXiv 2511.16201. https://arxiv.org/abs/2511.16201 [V]
- LLaMEA code: https://github.com/XAI-liacs/LLaMEA [V]; PyPI `llamea` 1.4.2, `iohblade` 1.1.1 [L]

**Other LLM algorithm-design work**
- Zhang, Liu, Lin, Wang, Lu, Zhang. "Understanding the Importance of Evolutionary Search in Automated Heuristic Design with LLMs." PPSN 2024. https://arxiv.org/abs/2407.10873 [V/S]
- Liu et al. "LLM4AD." arXiv 2412.17287. https://github.com/Optima-CityU/LLM4AD [V/S]
- Šenkeřík et al. "Open and closed source models for LLM-generated metaheuristics solving engineering optimization problem." EvoApplications 2025. https://badap.agh.edu.pl/publikacja/159982 [V]
- Liu et al. "Evolution of Heuristics." ICML 2024. arXiv 2401.02051 [V via LLM4AD]; Ye et al. "ReEvo." NeurIPS 2024. arXiv 2402.01145 [V/S]; Zheng et al. "MCTS-AHD." ICML 2025. https://arxiv.org/abs/2501.08603 [V]
- Romera-Paredes et al. "FunSearch." Nature 625:468–475, 2024. https://www.nature.com/articles/s41586-023-06924-6 [B]
- Novikov et al. "AlphaEvolve." arXiv 2506.13131 [V blog]; Lange, Imajuku, Cetin. "ShinkaEvolve." arXiv 2509.19349 [V]
- Zhang, Lu. "Rethinking Code Similarity for Automated Algorithm Design with LLMs (BehaveSim)." ICLR 2026. https://arxiv.org/abs/2603.02787 [V]
- "Budget-Efficient Automatic Algorithm Design via Code Graph." arXiv 2605.10598 [S, authors unverified]; "A2DEPT." arXiv 2604.24043 [S, authors unverified]
- "Reward Hacking Benchmark." arXiv 2605.02964 [S]
- Davis, E. Review of FunSearch (2024), https://cs.nyu.edu/davise/papers/FunSearch.pdf; comments on AlphaEvolve (2025), https://cs.nyu.edu/~davise/papers/AlphaEvolveNotes.pdf [V]
- Sakana AI CUDA Engineer walk-back. https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/embed/ [V]

**Metaphor critique and benchmarking methodology**
- Sörensen, K. "Metaheuristics—the metaphor exposed." ITOR 22(1):3–18, 2015. https://iaorifors.com/paper/87759 [V]
- Camacho-Villalón, Dorigo, Stützle. ITOR 30(6), 2023, https://iridia.ulb.ac.be/~mdorigo/Published_papers/All_Dorigo_papers/CamDorStu2022itor.pdf; ANTS 2020, https://link.springer.com/doi/10.1007/978-3-030-60376-2_10 [V]
- Aranha et al. "Metaphor-based metaheuristics, a call for action." Swarm Intelligence 16, 2022. https://link.springer.com/content/pdf/10.1007/s11721-021-00202-9.pdf [V, full text not fetched]
- Kůdela, J. Nature Machine Intelligence 4, 2022, https://www.fme.vutbr.cz/en/veda/publikace/detail/179505; arXiv 2301.01984, https://arxiv.org/abs/2301.01984v1 [V]
- Vermetten et al. "Large-scale Benchmarking of Metaphor-based Optimization Heuristics." https://arxiv.org/abs/2402.09800 [V]
- Hansen et al. "COCO" (OMS 2021), https://arxiv.org/abs/1603.08785; "COCO: Performance Assessment," https://arxiv.org/abs/1605.03560 [V]
- Derrac, García, Molina, Herrera. SWEVO 1(1), 2011. https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/sicidm/2011-Derrac-SWEVO.pdf [V]; Demšar, JMLR 7, 2006 [B]; Vargha & Delaney, 2000 [B]; Benavoli et al., JMLR 18, 2017, https://www.jmlr.org/papers/v18/16-305.html [V]
- pycma docs, https://cma-es.github.io/apidocs-pycma/ [V]; SciPy `differential_evolution` docs [V]; PyPI `coco-experiment` 2.8.3, `ioh` 0.3.22 [L]

**Providers (all read 2026-10-28)**
- Groq rate limits https://console.groq.com/docs/rate-limits; models https://console.groq.com/docs/models; deprecations https://console.groq.com/docs/deprecations; API reference https://console.groq.com/docs/api-reference; your data https://console.groq.com/docs/your-data [V]
- Cerebras rate limits https://inference-docs.cerebras.ai/support/rate-limits [V]; OpenRouter limits https://openrouter.ai/docs/api-reference/limits [V]; GitHub Models retirement [V]; HF Inference pricing [V]; gpt-oss-120b model card https://huggingface.co/openai/gpt-oss-120b [V]

---

## 12. Verification log: what was checked and what was not

**Checked this session (by the research notes, per their tags):**

- LLaMEA method, models, error rate, AOCC figures and limitations (arXiv HTML).
- BLADE setup and LLM comparison.
- LLaMEA-HPO, mutation control, CEG, SAGE and BehaveSim abstracts.
- Groq free-tier limits, models, deprecations and API behaviour (official pages, 2026-10-28).
- Cerebras and OpenRouter limits; GitHub Models retirement.
- Local checks: `coco-experiment` installs on Python 3.14.7 and evaluates in about 2.6 µs; `ioh` fails to build on 3.14; pycma stops at 10.7k evaluations without restarts; `llamea`/`iohblade` AOCC code matches the team's formula; BLADE's default instance-5 overlap.

**Checked by me in the repo today [R]:**

- The venv is Python 3.14.7.
- `benchmarks.py` registers only Sphere and Rastrigin (±5.12, shift fraction 0.8); Ackley and Griewank are not yet implemented.
- `BudgetExhausted` subclasses `BaseException`.
- `CountingFunction` exposes `.problem`, `.evals`, `.trace` and `.best_f` as writable attributes.
- `score()` uses `instance = seed + 1`, and `compare.py` has no seed offset.
- `report.py` tests final error with Mann-Whitney and no multiplicity correction.
- No LLM-loop code exists.
- The PSO/GA numbers quoted in §1.1 come from `results/comparison/tables.md`.
- A probe run in the scratchpad confirmed the read-optimum exploit (AOCC 1.000), the counter-reset overrun (30,000 calls recorded as 3) and that a bare `except` cannot overrun the budget.

**Not checked: must be resolved before submission, in priority order:**

1. Whether Huang et al. (2603.02792) compares Qwen 3.x inside LLaMEA, which would narrow the novelty claim.
2. LLaMEA's exact run count and verbatim prompts (TEVC PDF or `llamea` source).
3. The `cocoex` API for f_opt and for arbitrary instance numbers; whether 30-D is outside COCO's dimensions.
4. ICCI 2027 template, page limit, review model and deadline. The ICCSMLAI and ICAI-27 dates are as told by the user and must be checked on the official sites.
5. The LLaMEA-BO origin-exploit passage.
6. Cerebras trial duration, daily request cap and reasoning-parameter names; whether Cerebras and Groq serve the same weights and quantisation.
7. The team's actual Groq per-account limits (console /settings/limits) and the TPD reset semantics.
8. Real tokens per call including reasoning, and the valid-code rate (day-0 pilot).
9. Local GPU throughput for Ollama.
10. The Qwen3.8 licence; Groq and Cerebras terms on publishing outputs.
11. Authors and venues of the snippet-only 2026 preprints (2605.10598, 2604.24043, 2605.02964).
12. Vargha-Delaney thresholds and the Demšar and Derrac details (background knowledge).
13. Whether the `llamea` package depends on `ioh`.

Every [EST] figure in §4.5 and §5.2 is arithmetic on cited limits or one local timing, not a measurement of the MetaMind loop.
