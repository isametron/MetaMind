# Found by arm B, run 4, cerebras/gpt-oss-120b; training fitness 0.4306 (call 48 of 50); name: AdaptiveDE
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Adaptive Differential Evolution (DE/rand/1/bin) with self‑adaptive F and CR.
    Returns best point, its value and a history of best‑so‑far values.
    """
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # --- parameters ---------------------------------------------------------
    pop_size = max(5, 10 + dim)                     # modest population
    max_iter = (budget // pop_size) + 1             # ensure we don't exceed budget
    # minimum and maximum for adaptive parameters
    F_min, F_max = 0.4, 0.9
    CR_min, CR_max = 0.1, 0.9

    # --- initialise population -----------------------------------------------
    pop = np.random.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.empty(pop_size, dtype=float)

    evals = 0
    history = []

    for i in range(pop_size):
        fitness[i] = func(pop[i])
        evals += 1
        history.append(fitness[i])
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])

    # --- main loop ------------------------------------------------------------
    while evals < budget:
        for i in range(pop_size):
            if evals >= budget:
                break

            # --- self‑adaptive parameters ------------------------------------
            F = random.uniform(F_min, F_max)
            CR = random.uniform(CR_min, CR_max)

            # --- mutation: DE/rand/1 -----------------------------------------
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant = pop[a] + F * (pop[b] - pop[c])

            # --- ensure mutant is inside bounds --------------------------------
            mutant = np.clip(mutant, lower, upper)

            # --- binomial crossover -------------------------------------------
            cross_points = np.random.rand(dim) < CR
            # guarantee at least one dimension from mutant
            if not np.any(cross_points):
                cross_points[random.randrange(dim)] = True
            trial = np.where(cross_points, mutant, pop[i])

            # --- selection ------------------------------------------------------
            f_trial = func(trial)
            evals += 1
            history.append(min(best_f, f_trial))

            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = float(f_trial)
                    best_x = trial.copy()
        # optional early exit if optimum found (unlikely for noisy functions)
        if best_f == 0.0:
            break

    return best_x, best_f, history
