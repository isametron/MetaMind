"""LLM prompts for the AlgoGenesis LLaMEA loop.

Public names (kept stable — changing them breaks run_llm.py):

    TASK_PROMPT                      str, sent as call 1 (arm A) and every call (arm B)
    build_feedback(task, history, parent_code, parent_mean, parent_std, last_error) -> str

What scripts/run_llm.py passes to build_feedback:

    task         TASK_PROMPT
    history      list of dicts, oldest first, one per earlier call:
                 {"call": int, "name": str, "fitness": float, "error_type": str | None,
                  "accepted": bool,            # did this child become the parent?
                  "summary": list[dict]}       # result.summary(): one row per (function, dim)
    parent_code  source of the current parent (never empty: until a parent exists the runner
                 sends TASK_PROMPT again)
    parent_mean  parent's fitness (mean AOCC over the 72 training runs)
    parent_std   std of AOCC across those runs
    last_error   error text of the most recent call (traceback, trimmed), or None if it ran clean

Attribution
-----------
The opening sentence of TASK_PROMPT ("The optimization algorithm should handle a wide range of
tasks … Give an excellent and novel heuristic algorithm to solve this task.") is taken verbatim
from LLaMEA 1.4.2 (llamea/llamea.py, LLaMEA.__init__, default task_prompt).  Everything else —
the interface specification (optimize signature, bounds usage, budget semantics, return type),
the code rules, and the entire build_feedback function — is our own addition.
"""

# ---------------------------------------------------------------------------
# TASK_PROMPT
# ---------------------------------------------------------------------------
# [LLaMEA verbatim – start]
# Source: LLaMEA 1.4.2, llamea/llamea.py, LLaMEA.__init__, default task_prompt
_LLAMEA_TASK = (
    "The optimization algorithm should handle a wide range of tasks, which is evaluated on the "
    "BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm "
    "in Python code to minimize the function value. "
    "Give an excellent and novel heuristic algorithm to solve this task."
)
# [LLaMEA verbatim – end]

# [Our additions – start]
# Interface specification: adapted from INTERFACE.md to match our optimize() signature.
_INTERFACE = (
    "The algorithm must be a Python function with the exact signature:\n\n"
    "    def optimize(func, dim, bounds, budget):\n"
    "        ...\n"
    "        return best_x, best_f, history\n\n"
    "Arguments:\n"
    "  func    – callable; takes one numpy array of shape (dim,) and returns a float to minimise.\n"
    "            Call it with one vector at a time — never batch.\n"
    "  dim     – int, number of dimensions.\n"
    "  bounds  – tuple (lower, upper) of two numpy arrays of shape (dim,); always use these\n"
    "            for sampling and clipping, never hard-code ±5 or ±5.12.\n"
    "  budget  – int, hard limit on the number of calls to func; stop before exceeding it.\n\n"
    "Return values:\n"
    "  best_x   – numpy array of shape (dim,), the best point found.\n"
    "  best_f   – float, func(best_x).\n"
    "  history  – list of floats, best-so-far value after each evaluation.\n\n"
    "Code rules (statically enforced):\n"
    "  • Use only numpy, math, and random — no scipy, os, sys, inspect, or other imports.\n"
    "  • Call func with one vector per call; respect the budget.\n"
    "  • Read bounds from the arguments; never hard-code ±5 or ±5.12.\n"
    "  • Do not read or set any attribute of func (it is a plain callable).\n"
    "  • Do not reseed the RNG (no np.random.seed(), no random.seed(), no default_rng(seed));\n"
    "    the harness seeds it before each run for reproducibility.\n"
    "  • The optimum is not at the origin — sampling the box centre gives no advantage."
)

# [Our additions – start: output format]
_OUTPUT_FORMAT = (
    "Start your reply with a single line\n\n"
    "    # Name: <short algorithm name>\n\n"
    "then give the complete implementation in one ```python block."
)
# [Our additions – end]

TASK_PROMPT = "\n\n".join([_LLAMEA_TASK, _INTERFACE, _OUTPUT_FORMAT])


# ---------------------------------------------------------------------------
# build_feedback
# ---------------------------------------------------------------------------
def build_feedback(
    task: str,
    history: list,
    parent_code: str,
    parent_mean: float,
    parent_std: float,
    last_error: str | None,
) -> str:
    """Build a feedback prompt for the next LLM call (arm A).

    Structure (all ours):
        1. The task prompt (passed in as `task`).
        2. The last ≤10 (name, fitness) pairs from history.
        3. The current parent's code with its mean/std AOCC.
        4. The last error trimmed to 20 lines, if any.
        5. A closing instruction to refine or redesign.
    """
    parts = [task]

    # [Our addition – recent history]
    recent = history[-10:]
    if recent:
        lines = ["Recent attempts (oldest → newest):"]
        for entry in recent:
            fitness_str = f"{entry['fitness']:.4f}" if entry["fitness"] is not None else "n/a"
            lines.append(f"  call {entry['call']:3d}: {entry['name']!r:30s}  fitness={fitness_str}")
        parts.append("\n".join(lines))

    # [Our addition – current parent]
    parts.append(
        f"The current best algorithm has mean AOCC {parent_mean:.4f} (std {parent_std:.4f}):\n"
        f"```python\n{parent_code}\n```"
    )

    # [Our addition – last error, trimmed to 20 lines]
    if last_error:
        error_lines = last_error.splitlines()
        if len(error_lines) > 20:
            error_lines = error_lines[-20:]
        trimmed = "\n".join(error_lines)
        parts.append(f"Your last submission failed with:\n{trimmed}")

    # [Our addition – closing instruction]
    parts.append("Either refine or redesign to improve the algorithm.")

    return "\n\n".join(parts)
