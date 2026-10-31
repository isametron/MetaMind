"""Benchmark functions for the AlgoGenesis harness.

Every benchmark takes a single 1-D numpy array and returns a float to be
minimised. Each one is registered with its standard search domain and known
global minimum so the harness can report *error* (f - f_opt), which is what we
compare across functions and algorithms.

Problem instances
-----------------
Sphere and Rastrigin both have their optimum at the origin, i.e. the exact
centre of the search box. An algorithm that simply samples the centre would
"solve" them instantly, and an LLM is quite capable of discovering that trick.
To keep the comparison honest, the harness evaluates *shifted* instances by
default: ``f(x - x_opt)`` with ``x_opt`` drawn inside the box. Instance 0 is the
canonical unshifted function; instances 1, 2, ... are shifted and reproducible.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np


def sphere(x) -> float:
    """Sphere: unimodal, separable. f(0) = 0."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def rastrigin(x) -> float:
    """Rastrigin: highly multimodal (~10^dim local minima), separable. f(0) = 0."""
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[np.ndarray], float]
    lower: float
    upper: float
    f_opt: float = 0.0


# Add new benchmarks here (Ackley and Griewank are planned for Phase 2).
BENCHMARKS: dict[str, Benchmark] = {
    "sphere": Benchmark("sphere", sphere, -5.12, 5.12),
    "rastrigin": Benchmark("rastrigin", rastrigin, -5.12, 5.12),
}


class Problem:
    """One concrete instance of a benchmark: a fixed dimension and optimum location."""

    # Keep the shifted optimum away from the box edges so it is always reachable.
    SHIFT_FRACTION = 0.8

    def __init__(self, benchmark: Benchmark, dim: int, instance: int = 1):
        self.benchmark = benchmark
        self.dim = int(dim)
        self.instance = int(instance)
        if instance == 0:
            self.x_opt = np.zeros(self.dim)
        else:
            seed = zlib.crc32(f"{benchmark.name}/{dim}/{instance}".encode())
            rng = np.random.default_rng(seed)
            self.x_opt = self.SHIFT_FRACTION * rng.uniform(benchmark.lower, benchmark.upper, self.dim)

    @property
    def name(self) -> str:
        return self.benchmark.name

    @property
    def f_opt(self) -> float:
        return self.benchmark.f_opt

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.full(self.dim, self.benchmark.lower), np.full(self.dim, self.benchmark.upper))

    def __call__(self, x) -> float:
        return self.benchmark.func(np.asarray(x, dtype=float) - self.x_opt)

    def __repr__(self) -> str:
        return f"Problem({self.name!r}, dim={self.dim}, instance={self.instance})"


def make_problem(name: str, dim: int, instance: int = 1) -> Problem:
    """Build a problem by name: a classic benchmark above, or a BBOB function ('bbob_f1' ... 'bbob_f24')."""
    if name.startswith("bbob_"):
        from .bbob import make_bbob_problem
        return make_bbob_problem(name, dim, instance)
    try:
        benchmark = BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"unknown benchmark {name!r}; available: {sorted(BENCHMARKS)}") from None
    return Problem(benchmark, dim, instance)
