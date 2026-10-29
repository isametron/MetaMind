"""The training fitness the LLM loop optimises.

    from algogenesis.fitness import training_fitness

    if __name__ == "__main__":       # required on Windows
        result = training_fitness(code)
        result.fitness    # mean AOCC in [0, 1], higher is better; 0 if the code failed
        result.error      # traceback / timeout message, or None
        result.summary()  # per-function numbers for the feedback prompt

This is the only call the loop should make, so the evaluation can change
without touching the loop. It only ever uses TRAIN_INSTANCES; final results
are reported on the disjoint TEST_INSTANCES.

PROVISIONAL until the fitness freeze (G1, Sat 31 Oct 12:00): shifted Sphere
and Rastrigin in 10-D. At G1 this switches to BBOB f1-f24 in 5-D with a budget
of 2000 x dim; the call stays the same.
"""
from .harness import TRAIN_INSTANCES, SuiteResult, evaluate_code

TRAINING_SUITE = dict(
    functions=("sphere", "rastrigin"),
    dims=(10,),
    instances=TRAIN_INSTANCES,
    n_runs=len(TRAIN_INSTANCES),  # one seed per training instance
    budget_per_dim=1000,
)


def training_fitness(code: str, timeout: float = 300.0) -> SuiteResult:
    """Score candidate source code on the training suite, in a sandboxed subprocess."""
    return evaluate_code(code, name="candidate", timeout=timeout, **TRAINING_SUITE)
