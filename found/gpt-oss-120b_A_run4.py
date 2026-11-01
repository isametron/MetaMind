# Found by arm A, run 4, cerebras/gpt-oss-120b; training fitness 0.4522 (call 27 of 50); name: RobustHybridDEPSO
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    RobustHybridDEPSO
    --------------------------------------------------------------
    A compact hybrid optimizer that combines three proven ideas
    while staying safe against numerical overflow:

    1. **Latin‑Hypercube Sampling (LHS)** – creates a well‑spread
       initial population and supplies fresh points when stagnation is
       detected.
    2. **Self‑adaptive Differential Evolution (jDE)** – the main
       exploration engine; per‑individual adaptation of F and CR keeps
       diversity high.
    3. **Particle‑Swarm‑style exploitation** – a lightweight velocity
       update nudges individuals toward the global best and an elite
       subgroup, improving local convergence.

    All operations respect the supplied `bounds`, use only numpy,
    random and math, and never exceed the given evaluation budget.
    The algorithm is deliberately robust: every intermediate
    candidate is clipped to the bounds before evaluation, preventing
    overflow that caused earlier failures.
    --------------------------------------------------------------
    """
    lower, upper = bounds
    rng = np.random                 # seeded by the harness

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def clip(x):
        """Clip a vector to the problem bounds."""
        return np.minimum(np.maximum(x, lower), upper)

    def lhs_point():
        """Draw a single LHS point inside the bounds."""
        # stratified coordinate per dimension
        u = (rng.rand(dim) + np.arange(dim)) / dim
        rng.shuffle(u)               # random permutation across dimensions
        return lower + u * (upper - lower)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    pop_size = max(4, 7 * dim)      # moderate size, works for low & high dim
    tau_f = 0.1                     # prob. to change F (jDE)
    tau_cr = 0.1                    # prob. to change CR (jDE)

    # PSO‑like coefficients
    w = 0.35
    c1 = 1.7
    c2 = 1.7

    # ------------------------------------------------------------------
    # 1) Initialise population with LHS
    # ------------------------------------------------------------------
    pop = np.array([lhs_point() for _ in range(pop_size)])
    fitness = np.empty(pop_size, dtype=float)
    for i in range(pop_size):
        fitness[i] = func(pop[i])
    evals = pop_size

    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    history = [best_f]

    # PSO velocities (start at zero)
    vel = np.zeros((pop_size, dim))

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    while evals < budget:
        # --------------------------------------------------------------
        # 2) jDE mutation & crossover (DE/rand/1/bin)
        # --------------------------------------------------------------
        new_pop = np.empty_like(pop)
        new_fit = np.empty_like(fitness)

        for i in range(pop_size):
            # ---- adapt parameters per individual ----
            Fi = 0.1 + random.random() * 0.9 if random.random() < tau_f else 0.5
            CRi = random.random() if random.random() < tau_cr else 0.9

            # ---- mutation (rand/1) ----
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant = pop[a] + Fi * (pop[b] - pop[c])
            mutant = clip(mutant)                     # keep inside bounds

            # ---- binomial crossover ----
            jrand = random.randrange(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]

            trial = clip(trial)

            f_trial = func(trial)
            evals += 1

            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
                if f_trial < best_f:
                    best_f = float(f_trial)
                    best_x = trial.copy()
            else:
                new_pop[i] = pop[i]
                new_fit[i] = fitness[i]

            history.append(best_f)
            if evals >= budget:
                break

        if evals >= budget:
            break

        pop, fitness = new_pop, new_fit

        # --------------------------------------------------------------
        # 3) PSO‑style exploitation on elite individuals
        # --------------------------------------------------------------
        elite_n = max(1, pop_size // 5)
        elite_idx = np.argsort(fitness)[:elite_n]
        elite = pop[elite_idx]

        for i in range(pop_size):
            r1, r2 = random.random(), random.random()
            elite_vec = elite[random.randrange(elite_n)]

            # velocity update with clipping to avoid blow‑up
            vel[i] = w * vel[i] + \
                     c1 * r1 * (best_x - pop[i]) + \
                     c2 * r2 * (elite_vec - pop[i])
            # optional safety clip on velocity magnitude
            # (scale if any component exceeds the bound width)
            width = upper - lower
            mask = np.abs(vel[i]) > width
            if np.any(mask):
                vel[i][mask] = np.sign(vel[i][mask]) * width[mask]

            cand = clip(pop[i] + vel[i])
            f_cand = func(cand)
            evals += 1

            if f_cand < fitness[i]:
                pop[i] = cand
                fitness[i] = f_cand
                if f_cand < best_f:
                    best_f = float(f_cand)
                    best_x = cand.copy()

            history.append(best_f)
            if evals >= budget:
                break

        if evals >= budget:
            break

        # --------------------------------------------------------------
        # 4) Stagnation detection + partial restart
        # --------------------------------------------------------------
        if len(history) > 2 * dim:
            recent_best = min(history[-2 * dim:])
            # if no improvement, inject fresh LHS individuals
            if recent_best >= best_f * (1.0 + 1e-12):
                n_replace = max(1, pop_size // 4)
                worst_idx = np.argsort(fitness)[-n_replace:]
                for wi in worst_idx:
                    if evals >= budget:
                        break
                    pop[wi] = lhs_point()
                    fitness[wi] = func(pop[wi])
                    evals += 1
                    if fitness[wi] < best_f:
                        best_f = float(fitness[wi])
                        best_x = pop[wi].copy()
                    history.append(best_f)

    return best_x, best_f, history
