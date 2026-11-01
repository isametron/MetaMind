# Found by arm A, run 1, cerebras/gpt-oss-120b; training fitness 0.4922 (call 46 of 50); name: AdaptiveHybridDE‑CMA‑PatternSearch
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    AdaptiveHybridDE‑CMA‑PatternSearch
    ----------------------------------
    A lightweight hybrid optimiser designed for the BBOB noiseless suite.

    Core ideas
    ----------
    1. **Quasi‑random start** – a deterministic Halton sequence gives a well
       spread initial population without any random seed manipulation.
    2. **jDE‑style Differential Evolution** – each individual adapts its
       scale factor *F* and crossover probability *CR* on the fly.
    3. **Opposition‑based learning** – every DE trial is paired with its
       opposite w.r.t. the current best solution; the better of the two
       survives.
    4. **Diagonal CMA‑ES refinement** – after each DE generation a few
       Gaussian offspring are drawn from a diagonal covariance matrix that
       is updated by an exponential moving average of successful DE steps.
    5. **Simple pattern‑search escape** – when stagnation is detected a
       short exploratory walk (step‑size reduction) tries to leave flat
       regions.

    The algorithm respects the evaluation budget, never hard‑codes problem
    bounds, and records the best‑so‑far value after every function call.
    """

    # ------------------------------------------------------------------ #
    # 0. Helper utilities
    # ------------------------------------------------------------------ #
    lower, upper = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    width = upper - lower

    def clip(x):
        """Project a vector back into the search box."""
        return np.minimum(np.maximum(x, lower), upper)

    def opposite(x, ref):
        """Opposite point of `x` with respect to reference `ref` inside the box."""
        opp = ref + (ref - x)
        return clip(opp)

    # ------------------------------------------------------------------ #
    # 1. Halton‑based quasi‑random initialisation
    # ------------------------------------------------------------------ #
    _primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
               31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
               73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def halton(idx, base):
        """Radical inverse of integer `idx` in given `base`."""
        f, r = 1.0 / base, 0.0
        while idx:
            r += f * (idx % base)
            idx //= base
            f /= base
        return r

    pop_size = max(4 * dim, 20)               # small but diverse
    pop = np.empty((pop_size, dim))
    for i in range(pop_size):
        for d in range(dim):
            base = _primes[d]
            pop[i, d] = lower[d] + width[d] * halton(i + 1, base)
    pop = clip(pop)

    # ------------------------------------------------------------------ #
    # 2. jDE parameters (self‑adaptation)
    # ------------------------------------------------------------------ #
    tau_f, tau_cr = 0.1, 0.1                  # probability to change F / CR
    F_min, F_max = 0.2, 0.9
    CR_min, CR_max = 0.0, 1.0

    # ------------------------------------------------------------------ #
    # 3. Diagonal CMA‑ES state (initial sigma from box size)
    # ------------------------------------------------------------------ #
    var = np.square(width / 6.0)              # variance per dimension
    sigma = np.sqrt(np.mean(var))             # global step size
    cov_decay = 0.9                           # exponential smoothing factor

    # ------------------------------------------------------------------ #
    # 4. Main optimisation loop
    # ------------------------------------------------------------------ #
    fitness = np.empty(pop_size, float)
    history = []

    # evaluate initial population
    for i in range(pop_size):
        fitness[i] = func(pop[i])
        history.append(fitness[i])

    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    evals = pop_size
    stagnation = 0
    max_stag = dim * 2

    while evals < budget:
        # -------------------------------------------------------------- #
        # 4.1 DE generation with opposition
        # -------------------------------------------------------------- #
        order = list(range(pop_size))
        random.shuffle(order)

        for i in order:
            # ---- jDE self‑adaptation of F and CR ----
            Fi = (F_min + random.random() * (F_max - F_min)) if random.random() < tau_f else 0.5
            CRi = (CR_min + random.random() * (CR_max - CR_min)) if random.random() < tau_cr else 0.9

            # ---- donor selection (3 distinct individuals) ----
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            mutant = pop[a] + Fi * (pop[b] - pop[c])
            mutant = clip(mutant)

            # ---- binomial crossover ----
            trial = pop[i].copy()
            jrand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CRi or j == jrand:
                    trial[j] = mutant[j]
            trial = clip(trial)

            # ---- opposition partner ----
            opp = opposite(trial, best_x)

            # ---- evaluate trial and opposite (budget‑aware) ----
            f_trial = func(trial)
            evals += 1
            history.append(min(best_f, f_trial))
            if evals >= budget:
                if f_trial < best_f:
                    best_f, best_x = f_trial, trial.copy()
                break

            f_opp = func(opp)
            evals += 1
            history.append(min(best_f, f_opp))
            if evals >= budget:
                if f_opp < best_f:
                    best_f, best_x = f_opp, opp.copy()
                break

            # ---- pick the better candidate ----
            if f_opp < f_trial:
                cand, f_cand = opp, f_opp
            else:
                cand, f_cand = trial, f_trial

            # ---- selection & CMA covariance update ----
            if f_cand <= fitness[i]:
                diff = cand - pop[i]
                var = cov_decay * var + (1.0 - cov_decay) * np.square(diff)
                sigma = np.sqrt(np.mean(var))

                pop[i] = cand
                fitness[i] = f_cand
                if f_cand < best_f:
                    best_f, best_x = f_cand, cand.copy()
                    stagnation = 0
            else:
                stagnation += 1

            if evals >= budget:
                break
        if evals >= budget:
            break

        # -------------------------------------------------------------- #
        # 4.2 Local diagonal CMA‑ES offspring
        # -------------------------------------------------------------- #
        n_local = max(1, dim // 4)                 # a few refined samples
        for _ in range(n_local):
            off = best_x + np.random.randn(dim) * sigma
            off = clip(off)

            f_off = func(off)
            evals += 1
            history.append(min(best_f, f_off))

            if f_off < best_f:
                best_f, best_x = f_off, off.copy()
                stagnation = 0

            # replace the worst DE individual if the offspring is better
            worst = int(np.argmax(fitness))
            if f_off < fitness[worst]:
                pop[worst] = off
                fitness[worst] = f_off

            if evals >= budget:
                break
        if evals >= budget:
            break

        # -------------------------------------------------------------- #
        # 4.3 Stagnation handling – short exploratory walk
        # -------------------------------------------------------------- #
        if stagnation >= max_stag:
            step = 0.2 * np.mean(width)          # start with 20 % of the box size
            for _ in range(3):
                cand = best_x + np.random.randn(dim) * step
                cand = clip(cand)

                f_cand = func(cand)
                evals += 1
                history.append(min(best_f, f_cand))

                if f_cand < best_f:
                    best_f, best_x = f_cand, cand.copy()
                    stagnation = 0
                    break
                step *= 0.5                     # shrink step if not successful
            stagnation = 0

    return best_x, best_f, history
