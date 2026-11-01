# Found by arm B, run 1, cerebras/gpt-oss-120b; training fitness 0.4584 (call 5 of 7); name: AdaptiveHybridDE
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Adaptive Hybrid Differential Evolution.
    - Uses population of size NP = 10 + dim.
    - Self‑adapts F and CR per individual.
    - Every few generations performs a local Gaussian perturbation.
    - Records the best‑so‑far value after each function evaluation.
    Returns (best_x, best_f, history).
    """
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # population size (minimum 20)
    NP = max(20, 10 + dim)
    # ensure we have at most budget individuals
    NP = min(NP, budget // 2) if budget // 2 > 0 else 1

    # initialise population uniformly in the bounds
    pop = lower + (upper - lower) * np.random.rand(NP, dim)
    # evaluate initial population
    fitness = np.empty(NP, dtype=float)
    history = []
    for i in range(NP):
        fitness[i] = func(pop[i])
        history.append(fitness[i])
    evals = NP

    # best record
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    # parameters (self‑adaptive)
    F = np.full(NP, 0.5)          # differential weight
    CR = np.full(NP, 0.9)         # crossover probability

    # algorithm control
    max_gen = max(1, (budget - evals) // NP)
    gen = 0
    while evals < budget and gen < max_gen:
        for i in range(NP):
            if evals >= budget:
                break

            # --- mutation (DE/current-to-pbest) ---
            # pbest from top 20% of population
            p = max(1, int(0.2 * NP))
            idx_sorted = np.argsort(fitness)
            pbest = pop[random.choice(idx_sorted[:p])]
            # select two distinct random individuals not i nor pbest
            candidates = list(range(NP))
            candidates.remove(i)
            # ensure pbest not selected again
            # (if pbest happens to be i, ignore it)
            # we just pick two distinct others
            a, b = random.sample(candidates, 2)
            xa, xb = pop[a], pop[b]

            # adapt F and CR (jDE style)
            if random.random() < 0.1:
                F[i] = 0.1 + random.random() * 0.9
            if random.random() < 0.1:
                CR[i] = random.random()

            mutant = pop[i] + F[i] * (pbest - pop[i]) + F[i] * (xa - xb)

            # --- crossover (binomial) ---
            cross_mask = np.random.rand(dim) < CR[i]
            # ensure at least one dimension from mutant
            if not np.any(cross_mask):
                cross_mask[random.randrange(dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])

            # clip to bounds
            trial = np.clip(trial, lower, upper)

            # evaluate trial
            f_trial = func(trial)
            evals += 1
            history.append(f_trial)

            # selection
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                # store successful parameters
                # (optional: could average later)
                # update global best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
        # --- occasional local search ---
        # every 5 generations perform a small Gaussian perturbation around best
        if gen % 5 == 0:
            n_local = min(5, budget - evals)
            for _ in range(n_local):
                sigma = 0.1 * (upper - lower)  # 10% of range
                cand = best_x + sigma * np.random.randn(dim)
                cand = np.clip(cand, lower, upper)
                f_cand = func(cand)
                evals += 1
                history.append(f_cand)
                if f_cand < best_f:
                    best_f = f_cand
                    best_x = cand.copy()
        gen += 1

    return best_x, best_f, history
