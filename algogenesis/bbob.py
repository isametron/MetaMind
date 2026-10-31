"""BBOB: the 24 noiseless COCO benchmark functions, adapted to the harness.

Each BBOB problem is identified by function (1-24), dimension and instance.
The instance number sets the optimum's location and value (shifts, rotations
and f_opt), so the harness's TRAIN_INSTANCES / TEST_INSTANCES split applies
here unchanged. The search box is always [-5, 5]^dim and the optimum lies
inside it.

Problems come from ``cocoex.BareProblem`` (pip package ``coco-experiment``).
f_opt is read with ``best_value()`` from a *separate* BareProblem object:
asking COCO for the optimum marks that object as "tainted", so the one we
evaluate is never the one that revealed it. Checked on all 24 functions in
5, 10 and 20-D for instances 1-4, 10 and 18: f(x_opt) - best_value() == 0.
"""
from __future__ import annotations

import numpy as np

BBOB_LOWER, BBOB_UPPER = -5.0, 5.0
BBOB_FUNCTIONS = tuple(f"bbob_f{i}" for i in range(1, 25))

BBOB_NAMES = {
    1: "Sphere", 2: "Separable ellipsoid", 3: "Separable Rastrigin", 4: "Bueche-Rastrigin",
    5: "Linear slope", 6: "Attractive sector", 7: "Step ellipsoid", 8: "Rosenbrock",
    9: "Rotated Rosenbrock", 10: "Ellipsoid", 11: "Discus", 12: "Bent cigar",
    13: "Sharp ridge", 14: "Different powers", 15: "Rastrigin", 16: "Weierstrass",
    17: "Schaffer F7", 18: "Ill-conditioned Schaffer F7", 19: "Griewank-Rosenbrock",
    20: "Schwefel", 21: "Gallagher 101 peaks", 22: "Gallagher 21 peaks", 23: "Katsuura",
    24: "Lunacek bi-Rastrigin",
}

# The five standard BBOB function groups.
BBOB_GROUPS = {
    "separable": (1, 2, 3, 4, 5),
    "low or moderate conditioning": (6, 7, 8, 9),
    "high conditioning, unimodal": (10, 11, 12, 13, 14),
    "multimodal, adequate global structure": (15, 16, 17, 18, 19),
    "multimodal, weak global structure": (20, 21, 22, 23, 24),
}


def parse_bbob_name(name: str) -> int:
    """'bbob_f7' -> 7."""
    if name not in BBOB_FUNCTIONS:
        raise ValueError(f"unknown BBOB function {name!r}; expected bbob_f1 ... bbob_f24")
    return int(name.removeprefix("bbob_f"))


class BBOBProblem:
    """One BBOB function instance, with the same interface as benchmarks.Problem."""

    def __init__(self, function: int, dim: int, instance: int):
        import cocoex  # imported here so the classic benchmarks work without it

        if not 1 <= function <= 24:
            raise ValueError(f"BBOB function must be 1-24, got {function}")
        if instance < 1:
            raise ValueError(f"BBOB instances start at 1, got {instance}")
        self.function = int(function)
        self.dim = int(dim)
        self.instance = int(instance)
        self._f = cocoex.BareProblem("bbob", self.function, self.dim, self.instance)
        self.f_opt = float(cocoex.BareProblem("bbob", self.function, self.dim, self.instance).best_value())

    @property
    def name(self) -> str:
        return f"bbob_f{self.function}"

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(self.dim, BBOB_LOWER), np.full(self.dim, BBOB_UPPER)

    def __call__(self, x) -> float:
        return float(self._f(np.ascontiguousarray(x, dtype=float)))

    def __repr__(self) -> str:
        return f"BBOBProblem(f{self.function} {BBOB_NAMES[self.function]}, dim={self.dim}, instance={self.instance})"


def make_bbob_problem(name: str, dim: int, instance: int) -> BBOBProblem:
    return BBOBProblem(parse_bbob_name(name), dim, instance)
