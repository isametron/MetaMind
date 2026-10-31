"""PLACEHOLDER prompts. Om replaces the text; the names and arguments must stay.

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
"""

TASK_PROMPT = (
    "Write a Python function optimize(func, dim, bounds, budget) that minimises a black-box function "
    "using at most `budget` calls to func, and returns (best_x, best_f, history). "
    "Use only numpy, math and random. Start your reply with a line '# Name: <short name>' and give "
    "the full code in one ```python block."
)


def build_feedback(task, history, parent_code, parent_mean, parent_std, last_error) -> str:
    parts = [
        task,
        f"\nThe current best algorithm scores a mean AOCC of {parent_mean:.4f} (std {parent_std:.4f}):",
        f"```python\n{parent_code}\n```",
    ]
    if last_error:
        parts.append(f"\nYour last attempt failed with:\n{last_error}")
    parts.append("\nImprove the algorithm. Reply with '# Name: <short name>' and the full new code "
                 "in one ```python block.")
    return "\n".join(parts)
