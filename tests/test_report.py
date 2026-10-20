import numpy as np
import pytest

from algogenesis.baselines import ga, pso
from algogenesis.harness import run_suite, score
from algogenesis.report import (
    BETTER, TIE, WORSE, comparison_rows, latex_error_table, markdown_tables, plot_convergence,
    significance, win_tie_loss,
)


def test_ga_respects_budget_and_is_reproducible():
    for budget in (500, 537):  # 537 ends mid-generation
        a = score(ga.optimize, "rastrigin", dim=4, budget=budget, seed=3)
        b = score(ga.optimize, "rastrigin", dim=4, budget=budget, seed=3)
        assert a.ok and a.evals == budget
        assert a.best_error == b.best_error
        assert a.reported_best_f == pytest.approx(a.best_error)


def test_ga_beats_random_start_on_sphere():
    r = score(ga.optimize, "sphere", dim=5, budget=5000, seed=1)
    assert r.best_error < 1e-2


def test_significance_directions():
    rng = np.random.default_rng(0)
    good, bad = rng.uniform(0, 1, 30), rng.uniform(10, 11, 30)
    assert significance(good, bad)[0] == BETTER
    assert significance(bad, good)[0] == WORSE
    assert significance(good, good)[0] == TIE
    # Both solved to below 1e-8: not "different", even though raw values differ.
    assert significance(np.full(30, 1e-14), np.full(30, 1e-10)) == (TIE, 1.0)


@pytest.fixture(scope="module")
def suites():
    kw = dict(dims=(3,), n_runs=8, budget_per_dim=300)
    return [run_suite(pso.optimize, name="PSO", **kw), run_suite(ga.optimize, name="GA", **kw)]


def test_tables(suites, tmp_path):
    rows = comparison_rows(suites, "PSO")
    assert len(rows) == 4
    assert all(r["vs_reference"] == "" for r in rows if r["algorithm"] == "PSO")
    assert sum(win_tie_loss(rows, "PSO")["GA"]) == 2

    md = markdown_tables(rows, "PSO")
    assert "| sphere 3-D |" in md and "Win / tie / loss against PSO" in md
    tex = latex_error_table(rows, "PSO")
    assert r"\begin{tabular}{lcc}" in tex and "W/T/L vs PSO" in tex

    plot_convergence(suites, "sphere", 3, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0

    with pytest.raises(ValueError):
        comparison_rows(suites, "CMA-ES")
