# Found by arm B, run 2, cerebras/gpt-oss-120b; training fitness 0.4899 (call 39 of 50); name: AdaptiveHybridDE
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Adaptive Hybrid Differential Evolution with occasional local search.
    Returns best point, its value, and history of best-so-far values.
    """
    lower, upper = bounds
    # Ensure arrays
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Parameters (they adapt during run)
    pop_size = max(5, 2 * dim)          # initial population size
    F = 0.5                             # mutation factor
    CR = 0.9                            # crossover probability
    max_iter = max(1, budget // pop_size)

    # Initialise population uniformly
    pop = lower + (upper - lower) * np.random.rand(pop_size, dim)
    fitness = np.empty(pop_size, dtype=float)
    history = []

    # Evaluate initial population
    evals = 0
    for i in range(pop_size):
        if evals >= budget:
            break
        fitness[i] = func(pop[i])
        evals += 1
        history.append(fitness[i] if i == 0 else min(history[-1], fitness[i]))

    # Keep track of the best
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    if not history:
        history.append(best_f)

    # Main DE loop
    while evals < budget:
        # Adapt parameters slowly
        F = 0.5 + 0.3 * random.random()   # randomise a bit each generation
        CR = 0.7 + 0.3 * random.random()

        for i in range(pop_size):
            if evals >= budget:
                break

            # --- Mutation (DE/rand/1) ---
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant = pop[a] + F * (pop[b] - pop[c])

            # Clip to bounds
            mutant = np.clip(mutant, lower, upper)

            # --- Crossover ---
            cross = np.empty(dim, dtype=float)
            j_rand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    cross[j] = mutant[j]
                else:
                    cross[j] = pop[i][j]

            # --- Selection ---
            f_cross = func(cross)
            evals += 1
            if f_cross < fitness[i]:
                pop[i] = cross
                fitness[i] = f_cross
                if f_cross < best_f:
                    best_f = f_cross
                    best_x = cross.copy()
            # record history
            history.append(best_f)

        # Occasionally perform a tiny local search around the current best
        if evals < budget and random.random() < 0.2:
            # Gaussian perturbation with decreasing step size
            sigma = (upper - lower) * 0.05 * (1.0 - evals / budget)
            candidate = best_x + sigma * np.random.randn(dim)
            candidate = np.clip(candidate, lower, upper)
            f_cand = func(candidate)
            evals += 1
            if f_cand < best_f:
                best_f = f_cand
                best_x = candidate.copy()
            history.append(best_f)

        # If population size is too small relative to remaining budget, enlarge it
        remaining = budget - evals
        if remaining > pop_size * 2 and pop_size < 4 * dim:
            # add new random individuals
            add = min(pop_size, remaining // 2)
            new_inds = lower + (upper - lower) * np.random.rand(add, dim)
            new_fit = np.empty(add, dtype=float)
            for k in range(add):
                if evals >= budget:
                    break
                new_fit[k] = func(new_inds[k])
                evals += 1
                history.append(min(best_f, new_fit[k]))
                if new_fit[k] < best_f:
                    best_f = new_fit[k]
                    best_x = new_inds[k].copy()
            # extend population
            pop = np.vstack((pop, new_inds))
            fitness = np.concatenate((fitness, new_fit))
            pop_size = pop.shape[0]

    return best_x, best_f, history
