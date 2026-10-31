"""The training fitness the LLM loop optimises. FROZEN at G1 (Sat 31 Oct 2026).

    from algogenesis.fitness import training_fitness

    if __name__ == "__main__":       # required on Windows
        result = training_fitness(code)
        result.fitness    # mean AOCC in [0, 1], higher is better; 0 if the code failed
        result.error      # traceback / timeout message, or None
        result.summary()  # per-function numbers for the feedback prompt

This is the only call the loop should make. It only ever uses TRAIN_INSTANCES;
final results are reported on the disjoint TEST_INSTANCES, in more dimensions.

Training suite (matches LLaMEA's setup): all 24 BBOB functions in 5-D,
instances 1-3 with one seed each (72 runs per candidate), budget 2000 x dim =
10,000 evaluations per run, 300 s limit per candidate. PSO takes about 5 s.
Do not change these settings after the LLM runs start.
"""
from .bbob import BBOB_FUNCTIONS
from .harness import TRAIN_INSTANCES, SuiteResult, evaluate_code

TRAINING_SUITE = dict(
    functions=BBOB_FUNCTIONS,
    dims=(5,),
    instances=TRAIN_INSTANCES,
    n_runs=len(TRAIN_INSTANCES),  # one seed per training instance
    budget_per_dim=2000,
)


def training_fitness(code: str, timeout: float = 300.0) -> SuiteResult:
    """Score candidate source code on the training suite, in a sandboxed subprocess."""
    return evaluate_code(code, name="candidate", timeout=timeout, **TRAINING_SUITE)
