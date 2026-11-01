# Found by arm A, run 3, cerebras/gpt-oss-120b; training fitness 0.4360 (call 48 of 50); name: Chaotic‑Hybrid Lévy‑CMA Memetic Search (CHLCMS)
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Chaotic‑Hybrid Lévy‑CMA Memetic Search (CHLCMS)

    A compact yet powerful optimizer for the BBOB noiseless suite.
    It combines three complementary mechanisms:

    1. **Chaotic global sampling** – a low‑discrepancy sequence (logistic map)
       that quickly spreads points over the whole search box.
    2. **Self‑adaptive Lévy‑flight DE** – DE/rand/1/bin with mutation factor
       drawn from a Lévy distribution, giving occasional long jumps.
    3. **Isotropic CMA‑ES style adaptation** – a simple 1/5‑success rule
       controlling a Gaussian step size `sigma`.

    The algorithm works in generations; each generation performs a DE step
    for the whole population, then a CMA‑ES Gaussian mutation for each
    individual.  Successful individuals drive the adaptation of `F`,
    `CR` and `sigma`.  If a generation yields no improvement, half of the
    population is refreshed with fresh chaotic samples (restart‑like behaviour).

    Parameters
    ----------
    func : callable
        Function to minimise; receives a single ``np.ndarray`` of shape ``(dim,)``.
    dim : int
        Dimensionality of the problem.
    bounds : tuple (lower, upper)
        Lower and upper bound arrays of shape ``(dim,)``.
    budget : int
        Maximum number of allowed function evaluations.

    Returns
    -------
    best_x : ndarray (dim,)
        Best point found.
    best_f : float
        Function value at ``best_x``.
    history : list of float
        Best‑so‑far value after each evaluation.
    """
    lower, upper = bounds
    rng = np.random.default_rng()          # harness seeds globally

    # ------------------- hyper‑parameters -------------------
    pop_size = max(6, int(4 + 6 * np.log(dim)))   # modest size, grows with dim
    # Initial DE parameters (they will be self‑adapted)
    F = 0.5
    CR = 0.9
    # CMA‑ES step size (isotropic)
    sigma = 0.2 * np.mean(upper - lower)
    cs = (dim + 2) / (dim + 5)                     # learning rate for 1/5 rule
    target_success = 0.2

    # ------------------- state -------------------
    # Chaotic sequence (logistic map) for initial sampling
    r = 0.7
    def chaotic():
        nonlocal r
        r = 4.0 * r * (1.0 - r)   # logistic map, r in (0,1)
        return r

    pop = np.empty((pop_size, dim))
    for i in range(pop_size):
        pop[i] = lower + (upper - lower) * chaotic()
    fitness = np.full(pop_size, np.inf)

    best_f = np.inf
    best_x = None
    history = []
    evals = 0

    # ------------------- helpers -------------------
    def clip(x):
        """Reflect out‑of‑bounds coordinates back into the box."""
        over = x > upper
        x[over] = upper[over] - (x[over] - upper[over])
        under = x < lower
        x[under] = lower[under] + (lower[under] - x[under])
        np.clip(x, lower, upper, out=x)
        return x

    def levy_flight(beta=1.5):
        """Draw a Lévy step (scale 1) using Mantegna's algorithm."""
        sigma_u = (math.gamma(1 + beta) *
                   math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) *
                    beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = rng.standard_normal(dim) * sigma_u
        v = rng.standard_normal(dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    # ------------------- initial evaluation -------------------
    for i in range(pop_size):
        if evals >= budget:
            break
        fitness[i] = func(pop[i])
        evals += 1
        if fitness[i] < best_f:
            best_f, best_x = fitness[i], pop[i].copy()
        history.append(best_f)

    # ------------------- main loop -------------------
    while evals < budget:
        # ------- 1. DE mutation & binomial crossover (rand/1) -------
        new_pop = np.empty_like(pop)
        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = rng.choice(idxs, size=3, replace=False)

            # Lévy‑scaled mutation factor
            F_i = np.abs(levy_flight(beta=1.5)).mean()
            mutant = pop[a] + F_i * (pop[b] - pop[c])
            mutant = clip(mutant)

            cross_mask = rng.random(dim) < CR
            if not cross_mask.any():               # guarantee at least one dimension
                cross_mask[rng.integers(dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])
            new_pop[i] = clip(trial)

        # ------- 2. DE selection -------
        succ_de = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            f_trial = func(new_pop[i])
            evals += 1
            if f_trial < fitness[i]:
                pop[i] = new_pop[i]
                fitness[i] = f_trial
                succ_de += 1
                if f_trial < best_f:
                    best_f, best_x = f_trial, new_pop[i].copy()
            history.append(best_f)

        # ------- 3. CMA‑ES style isotropic Gaussian mutation -------
        succ_cma = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            cand = clip(pop[i] + sigma * rng.standard_normal(dim))
            f_cand = func(cand)
            evals += 1
            if f_cand < fitness[i]:
                pop[i] = cand
                fitness[i] = f_cand
                succ_cma += 1
                if f_cand < best_f:
                    best_f, best_x = f_cand, cand.copy()
            history.append(best_f)

        # ------- 4. Parameter adaptation -------
        total_succ = succ_de + succ_cma
        succ_rate = total_succ / (2 * pop_size)

        # Simple self‑adaptation: move F and CR towards values that yielded success
        if succ_de > 0:
            F = np.clip(F * (1 + 0.1 * (succ_de / pop_size)), 0.2, 0.9)
        else:
            F = np.clip(F * 0.9, 0.2, 0.9)

        if succ_de > 0:
            CR = np.clip(CR + 0.05 * (succ_de / pop_size), 0.4, 0.99)
        else:
            CR = np.clip(CR - 0.03, 0.4, 0.99)

        # 1/5‑success rule for sigma
        if succ_rate > target_success:
            sigma *= np.exp(cs)
        else:
            sigma *= np.exp(-cs)
        sigma = np.clip(sigma, 1e-12, np.mean(upper - lower))

        # ------- 5. Restart‑like refresh on stagnation -------
        if total_succ == 0:                # no improvement in this generation
            n_replace = pop_size // 2
            for _ in range(n_replace):
                if evals >= budget:
                    break
                idx = rng.integers(pop_size)
                # fresh chaotic sample
                cand = lower + (upper - lower) * chaotic()
                f_cand = func(cand)
                evals += 1
                pop[idx] = cand
                fitness[idx] = f_cand
                if f_cand < best_f:
                    best_f, best_x = f_cand, cand.copy()
                history.append(best_f)

    return best_x, best_f, history
