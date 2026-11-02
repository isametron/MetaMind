# Found by arm B, run 5, cerebras/gpt-oss-120b; training fitness 0.4405 (call 15 of 50); name: AdaptiveDE
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Adaptive Differential Evolution (DE) for BBOB-like problems.

    Parameters
    ----------
    func : callable
        Takes a 1‑D numpy array of length `dim` and returns a float (to minimise).
    dim : int
        Dimensionality of the problem.
    bounds : tuple (lower, upper)
        Two 1‑D numpy arrays of shape (dim,) giving the box constraints.
    budget : int
        Maximum number of calls to `func`.

    Returns
    -------
    best_x : np.ndarray
        Best point found (shape (dim,)).
    best_f : float
        Function value at `best_x`.
    history : list of float
        Best‑so‑far value after each evaluation.
    """
    lower, upper = bounds
    # ------------------------------------------------------------------ #
    # 1. Parameter initialisation
    # ------------------------------------------------------------------ #
    # Population size: moderate size, but never exceed the budget.
    pop_size = min(max(5 * dim, 20), budget // 2)
    # Ensure at least 4 individuals (needed for DE/rand/1)
    pop_size = max(pop_size, 4)

    # Adaptive control parameters (F and CR) – start with typical values
    F = 0.5          # differential weight
    CR = 0.9         # crossover probability

    # History of best‑so‑far values
    history = []

    # ------------------------------------------------------------------ #
    # 2. Initial population (uniform sampling inside the bounds)
    # ------------------------------------------------------------------ #
    pop = np.random.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.empty(pop_size, dtype=float)

    for i in range(pop_size):
        fitness[i] = func(pop[i])
        history.append(fitness[i])
    evals = pop_size

    # Keep track of the best individual
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])

    # ------------------------------------------------------------------ #
    # 3. Main DE loop
    # ------------------------------------------------------------------ #
    while evals < budget:
        # Adaptive scheme: every 10% of remaining budget we slightly
        # perturb F and CR to avoid stagnation.
        if evals % max(1, budget // 10) == 0:
            F = np.clip(np.random.normal(F, 0.1), 0.1, 0.9)
            CR = np.clip(np.random.normal(CR, 0.1), 0.1, 0.9)

        for i in range(pop_size):
            if evals >= budget:
                break

            # ------------------------------------------------------------------
            # 3.1 Mutation: DE/rand/1
            # ------------------------------------------------------------------
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant = pop[a] + F * (pop[b] - pop[c])

            # Clip mutant to bounds
            mutant = np.clip(mutant, lower, upper)

            # ------------------------------------------------------------------
            # 3.2 Crossover (binomial)
            # ------------------------------------------------------------------
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one dimension from mutant
            if not np.any(cross_points):
                cross_points[random.randrange(dim)] = True
            trial = np.where(cross_points, mutant, pop[i])

            # ------------------------------------------------------------------
            # 3.3 Selection
            # ------------------------------------------------------------------
            f_trial = func(trial)
            evals += 1
            history.append(min(best_f, f_trial))

            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = float(f_trial)
                    best_x = trial.copy()

        # End of generation – optional restart if population collapses
        # (all individuals become identical). Detect by variance.
        if np.var(pop, axis=0).max() < 1e-12 and evals < budget:
            # Re‑initialise a fraction of the population
            n_restart = max(1, pop_size // 5)
            new_inds = np.random.uniform(lower, upper, size=(n_restart, dim))
            pop[:n_restart] = new_inds
            for j in range(n_restart):
                fitness[j] = func(pop[j])
                evals += 1
                history.append(min(best_f, fitness[j]))
                if fitness[j] < best_f:
                    best_f = float(fitness[j])
                    best_x = pop[j].copy()

    return best_x, best_f, history
