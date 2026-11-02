# Found by arm B, run 1, cerebras/gpt-oss-120b; training fitness 0.4896 (call 24 of 50); name: AdaptiveSelfAdaptiveDE
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Adaptive Self‑Adaptive Differential Evolution (DE) for noisy‑free BBOB functions.
    Parameters
    ----------
    func : callable
        Takes a 1‑D numpy array of length `dim` and returns a float (to minimise).
    dim : int
        Dimensionality of the problem.
    bounds : tuple (lower, upper)
        Lower and upper bounds as numpy arrays of shape (dim,).
    budget : int
        Maximum number of allowed function evaluations.
    Returns
    -------
    best_x : np.ndarray
        Best point found (shape (dim,)).
    best_f : float
        Function value at best_x.
    history : list of float
        Best‑so‑far value after each evaluation.
    """
    lower, upper = bounds
    # ---- parameters ---------------------------------------------------------
    pop_size = max(4, 5 * dim)                     # population size
    max_iter = max(1, budget // pop_size)          # generations we can afford
    # Ensure we never exceed the budget
    max_iter = min(max_iter, (budget - pop_size) // pop_size + 1)

    # Self‑adaptive control parameters (F and CR) per individual
    # Initialise with random values in sensible ranges
    F = np.full(pop_size, 0.5)                     # scaling factor
    CR = np.full(pop_size, 0.9)                    # crossover probability

    # ---- initialise population ---------------------------------------------
    pop = np.random.rand(pop_size, dim) * (upper - lower) + lower
    fitness = np.empty(pop_size, dtype=float)

    history = []
    for i in range(pop_size):
        fitness[i] = func(pop[i])
        history.append(fitness[i])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    evals = pop_size

    # ---- main DE loop -------------------------------------------------------
    while evals < budget and max_iter > 0:
        max_iter -= 1
        for i in range(pop_size):
            # ---- mutation (current‑to‑pbest with archive) -------------------
            # Choose pbest from top 20% of population
            pbest_size = max(1, pop_size // 5)
            pbest_idx = np.random.choice(np.argsort(fitness)[:pbest_size])
            # Randomly select two distinct individuals different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = random.sample(idxs, 2)

            # Self‑adaptive parameters: with small probability change them
            if random.random() < 0.1:
                F[i] = np.clip(np.random.normal(F[i], 0.1), 0.1, 1.0)
            if random.random() < 0.1:
                CR[i] = np.clip(np.random.normal(CR[i], 0.1), 0.0, 1.0)

            # DE/current‑to‑pbest/1 mutation
            mutant = (pop[i] +
                      F[i] * (pop[pbest_idx] - pop[i]) +
                      F[i] * (pop[a] - pop[b]))

            # Clip mutant to bounds
            mutant = np.clip(mutant, lower, upper)

            # ---- crossover -------------------------------------------------
            cross_mask = np.random.rand(dim) < CR[i]
            # Ensure at least one dimension from mutant
            if not np.any(cross_mask):
                cross_mask[np.random.randint(dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])

            # ---- selection --------------------------------------------------
            trial_f = func(trial)
            evals += 1
            history.append(trial_f)

            if trial_f <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_f
                # Update best
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()
            # stop if budget exhausted
            if evals >= budget:
                break
        # end for each individual
    # end while

    return best_x, best_f, history
