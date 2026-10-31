import numpy as np
import pytest

from algogenesis.baselines import pso
from algogenesis.bbob import BBOB_FUNCTIONS, BBOB_GROUPS, BBOBProblem
from algogenesis.benchmarks import make_problem
from algogenesis.harness import score

cocoex = pytest.importorskip("cocoex")


def test_registry():
    assert len(BBOB_FUNCTIONS) == 24
    assert sorted(f for group in BBOB_GROUPS.values() for f in group) == list(range(1, 25))
    with pytest.raises(ValueError):
        make_problem("bbob_f25", 5)
    with pytest.raises(ValueError):
        make_problem("bbob_f1", 5, instance=0)


@pytest.mark.parametrize("fid", range(1, 25))
def test_error_is_zero_at_the_optimum_and_positive_elsewhere(fid):
    p = make_problem(f"bbob_f{fid}", 5, instance=4)
    x_opt = np.asarray(cocoex.BareProblem("bbob", fid, 5, 4).best_parameter())
    assert p(x_opt) - p.f_opt == pytest.approx(0.0, abs=1e-8)
    xs = np.random.default_rng(fid).uniform(-5, 5, (20, 5))
    assert all(p(x) - p.f_opt >= -1e-12 for x in xs)
    lower, upper = p.bounds
    assert np.all(lower == -5) and np.all(upper == 5)


def test_instances_move_the_optimum():
    a, b = BBOBProblem(1, 5, 1), BBOBProblem(1, 5, 2)
    assert a.f_opt != b.f_opt


def test_harness_scores_bbob():
    r = score(pso.optimize, "bbob_f1", dim=5, budget=10_000, seed=0, instance=4)
    assert r.ok and r.evals == 10_000 and r.instance == 4
    assert r.history[0] > r.best_error >= 0 and r.best_error < 1e-6
