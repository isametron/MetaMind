"""Real-coded Genetic Algorithm (baseline).

Textbook generational GA for continuous problems:

- tournament selection (size 3)
- BLX-alpha crossover (alpha = 0.5) with probability 0.9: each child gene is
  drawn uniformly from the parents' interval widened by alpha on both sides, so
  the search step shrinks naturally as the population converges
- Gaussian mutation with per-gene probability 1/dim, sigma = 10% of the range
- elitism: the best 2 individuals survive unchanged
- children are clipped to the bounds

Each generation's children are built in one vectorised step so the measured
compute cost reflects the algorithm, not Python loop overhead.
"""
import numpy as np


def optimize(func, dim, bounds, budget, *, pop_size=50, tournament=3, crossover_rate=0.9,
             alpha=0.5, mutation_rate=None, sigma_frac=0.1, elites=2):
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    span = upper - lower
    mutation_rate = 1.0 / dim if mutation_rate is None else mutation_rate
    n = min(pop_size, budget)
    elites = min(elites, n - 1)

    pop = lower + np.random.rand(n, dim) * span
    fit = np.empty(n)
    history = []
    best_f, best_x = np.inf, None
    for i in range(n):
        fit[i] = func(pop[i])
        if fit[i] < best_f:
            best_f, best_x = fit[i], pop[i].copy()
        history.append(best_f)
    evals = n

    while evals < budget:
        m = min(n - elites, budget - evals)  # children this generation

        # Tournament selection: winner = fittest of `tournament` random individuals.
        contestants = np.random.randint(n, size=(2 * m, tournament))
        winners = contestants[np.arange(2 * m), np.argmin(fit[contestants], axis=1)]
        p1, p2 = pop[winners[:m]], pop[winners[m:]]

        # BLX-alpha crossover; rows that skip crossover copy the first parent.
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        d = hi - lo
        children = np.random.uniform(lo - alpha * d, hi + alpha * d)
        no_cross = np.random.rand(m) >= crossover_rate
        children[no_cross] = p1[no_cross]

        # Gaussian mutation.
        mutate = np.random.rand(m, dim) < mutation_rate
        children += mutate * np.random.randn(m, dim) * sigma_frac * span
        children = np.clip(children, lower, upper)

        child_fit = np.empty(m)
        for i in range(m):
            child_fit[i] = func(children[i])
            if child_fit[i] < best_f:
                best_f, best_x = child_fit[i], children[i].copy()
            history.append(best_f)
        evals += m

        if m == n - elites:
            keep = np.argsort(fit)[:elites]
            pop = np.vstack([pop[keep], children])
            fit = np.concatenate([fit[keep], child_fit])
        else:
            # Budget ran out mid-generation: keep the best n of old + new.
            merged, merged_fit = np.vstack([pop, children]), np.concatenate([fit, child_fit])
            keep = np.argsort(merged_fit)[:n]
            pop, fit = merged[keep], merged_fit[keep]

    return best_x, best_f, history
