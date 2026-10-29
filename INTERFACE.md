# Algorithm interface (agreed contract)

Every algorithm, whether a hand-written baseline or LLM-generated, must define:

```python
def optimize(func, dim, bounds, budget):
    ...
    return best_x, best_f, history
```

| Argument | Type | Meaning |
|---|---|---|
| `func` | callable | Takes a numpy array of shape `(dim,)` and returns a float to **minimise**. One vector per call. |
| `dim` | `int` | Number of dimensions. |
| `bounds` | `(lower, upper)` | Two numpy arrays of shape `(dim,)`. |
| `budget` | `int` | Maximum number of calls to `func`. |

| Return | Type | Meaning |
|---|---|---|
| `best_x` | array `(dim,)` | Best point found. |
| `best_f` | `float` | `func(best_x)`. |
| `history` | list of floats | Best-so-far value after each evaluation. |

Rules the harness enforces:

- **Budget.** Calling `func` more than `budget` times stops the run. Whatever was found up to then still counts, so this isn't an error, but well-behaved code should stop on its own.
- **Shape.** Calling `func` with the wrong shape, or with NaN/inf values, is an error and the run counts as failed.
- **Randomness.** Use `np.random` / `random`. The harness seeds both before each run so results are reproducible. (`np.random.default_rng()` without a seed won't be.)
- **Trust.** The harness records the best value itself after every call. The returned `best_f` and `history` are kept for sanity checks only, so an algorithm can't misreport its result.
- **Shifted optima.** The optimum is *not* at the origin (each run uses a randomly shifted instance), so sampling the centre of the box gives no advantage.
- **`func` is a plain function.** It has no attributes to read or change; the counter and the optimum live where the algorithm can't reach them. Don't try to inspect it, and don't catch `BudgetExhausted` with a bare `except:` (it can't overrun the budget anyway; it only wastes time until the timeout).
- **Train/test split.** The LLM loop only ever sees training instances (`TRAIN_INSTANCES`); final results use the disjoint `TEST_INSTANCES`.
- Evaluating outside the bounds is allowed but counted (`out_of_bounds`).

## Scoring LLM-generated code (for the loop)

Use `training_fitness`. It is the one call the loop makes, so the training
suite can change (to BBOB at the fitness freeze) without touching the loop:

```python
from algogenesis.fitness import training_fitness

if __name__ == "__main__":          # required on Windows
    result = training_fitness(code_string)
    result.fitness     # one number, mean AOCC in [0, 1], higher is better; 0 if it crashed
    result.error       # traceback / "timed out" if the code couldn't run, else None
    result.summary()   # per-function stats: mean/median/best error, success rate, runtime
    result.runs        # every run, incl. .history (best-so-far error per evaluation)
```

`evaluate_code` runs the code in a separate process with a time limit, so syntax errors, crashes and infinite loops come back as `error` instead of killing your loop. It is not a security sandbox.

For trusted, in-process code: `run_suite(optimize, name=...)`, or `score(optimize, "rastrigin", dim=10, seed=0)` for a single run.

**Fitness.** AOCC (area over the convergence curve, the same measure LLaMEA uses) gives a score between 0 (error ≥ 1e2 for the whole run) and 1 (error ≤ 1e-8 from the first evaluation), linear in log10 of the error. It rewards both final quality and speed of convergence.
