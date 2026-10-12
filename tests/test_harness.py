import numpy as np
import pytest

from algogenesis.baselines import pso
from algogenesis.benchmarks import make_problem, rastrigin, sphere
from algogenesis.harness import aocc, evaluate_code, load_optimizer, run_suite, score


def test_benchmark_values():
    assert sphere(np.zeros(5)) == 0.0
    assert sphere([1, 2, 3]) == 14.0
    assert rastrigin(np.zeros(5)) == pytest.approx(0.0)
    assert rastrigin(np.ones(4)) == pytest.approx(4.0)  # 10n + sum(1 - 10 cos 2π) = n


@pytest.mark.parametrize("name", ["sphere", "rastrigin"])
def test_shifted_instances(name):
    p = make_problem(name, 6, instance=3)
    lower, upper = p.bounds
    assert np.all(p.x_opt > lower) and np.all(p.x_opt < upper)
    assert p(p.x_opt) == pytest.approx(0.0)
    assert p(np.zeros(6)) > 0
    assert np.array_equal(p.x_opt, make_problem(name, 6, instance=3).x_opt)
    assert not np.array_equal(p.x_opt, make_problem(name, 6, instance=4).x_opt)
    assert np.array_equal(make_problem(name, 6, instance=0).x_opt, np.zeros(6))


def test_budget_is_enforced_and_history_recorded():
    def greedy_forever(func, dim, bounds, budget):
        while True:  # ignores the budget; the harness must stop it
            func(np.random.uniform(*bounds))

    r = score(greedy_forever, "sphere", dim=3, budget=500)
    assert r.ok and r.evals == 500 and len(r.history) == 500
    assert np.all(np.diff(r.history) <= 0)
    assert r.best_error == r.history[-1]


def test_crash_is_reported_not_raised():
    def broken(func, dim, bounds, budget):
        func(np.zeros(dim + 1))

    r = score(broken, "sphere", dim=4, budget=100)
    assert not r.ok and "shape" in r.error and r.aocc == 0.0


def test_wrong_return_value_is_a_failure():
    r = score(lambda func, dim, bounds, budget: func(bounds[0]), "sphere", dim=2, budget=10)
    assert not r.ok and "best_x, best_f, history" in r.error


def test_aocc_range():
    assert aocc(np.full(10, 1e3), 10) == 0.0
    assert aocc(np.full(10, 1e-9), 10) == 1.0
    assert aocc(np.array([]), 10) == 0.0


def test_pso_is_reproducible_and_solves_sphere():
    a = score(pso.optimize, "sphere", dim=5, budget=5000, seed=7)
    b = score(pso.optimize, "sphere", dim=5, budget=5000, seed=7)
    assert a.ok and a.evals == 5000
    assert a.best_error == b.best_error
    assert a.best_error < 1e-6
    assert a.reported_best_f == pytest.approx(a.best_error)  # f* = 0, so error == f


def test_suite_summary():
    s = run_suite(pso.optimize, name="PSO", dims=(2,), n_runs=3, budget_per_dim=200)
    rows = s.summary()
    assert {(r["function"], r["dim"]) for r in rows} == {("sphere", 2), ("rastrigin", 2)}
    assert all(r["runs"] == 3 and r["failures"] == 0 for r in rows)
    assert 0 < s.fitness <= 1


CANDIDATE = """
import numpy as np

def optimize(func, dim, bounds, budget):
    lower, upper = bounds
    best_x, best_f, history = None, np.inf, []
    for _ in range(budget):
        x = np.random.uniform(lower, upper)
        f = func(x)
        if f < best_f:
            best_x, best_f = x, f
        history.append(best_f)
    return best_x, best_f, history
"""


def test_load_optimizer_checks_signature():
    assert callable(load_optimizer(CANDIDATE))
    with pytest.raises(ValueError, match="must define"):
        load_optimizer("def run(func): pass")
    with pytest.raises(ValueError, match="must accept"):
        load_optimizer("def optimize(func): pass")


def test_evaluate_code_in_subprocess():
    s = evaluate_code(CANDIDATE, dims=(2,), n_runs=2, budget_per_dim=100, timeout=60)
    assert s.error is None and len(s.runs) == 4 and s.fitness > 0


def test_evaluate_code_reports_syntax_error_and_timeout():
    assert "SyntaxError" in evaluate_code("def optimize(:", timeout=60).error
    hang = "def optimize(func, dim, bounds, budget):\n    while True:\n        pass\n"
    s = evaluate_code(hang, dims=(2,), n_runs=1, timeout=5)
    assert "timed out" in s.error and s.fitness == 0.0
