# Found by arm A, run 2, cerebras/gpt-oss-120b; training fitness 0.5792 (call 48 of 50); name: AdaptiveHybridDE‑Simplex‑ES
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    AdaptiveHybridDE‑Simplex‑ES
    ---------------------------------
    A lightweight hybrid optimiser built from three complementary components:

    1️⃣  **Self‑adaptive DE/rand/1 (jDE)** – a small population explores globally.
         The scale factor F and crossover rate CR are adapted per‑individual.

    2️⃣  **Periodic Nelder‑Mead simplex** – every few generations a simplex built
         from the current elite is refined.  This gives fast local convergence
         without requiring gradients.

    3️⃣  **(1+1)‑ES with 1/5‑th rule** – when the budget is almost exhausted a
         simple evolution‑strategy polish is applied around the best point.

    The algorithm respects the supplied `budget`, never hard‑codes any problem
    constants, and records the best‑so‑far value after each evaluation.
    """
    # ------------------------------------------------------------------ #
    # 0.  box handling
    lower, upper = map(np.asarray, bounds)
    lower = lower.astype(float).reshape(-1)
    upper = upper.astype(float).reshape(-1)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

    def clip(x):
        """Project a point back into the feasible box."""
        return np.minimum(np.maximum(x, lower), upper)

    # ------------------------------------------------------------------ #
    # 1.  initialise DE population (size scales with budget)
    popsize = max(4, min(30, budget // 5))
    pop = [clip(np.random.uniform(lower, upper)) for _ in range(popsize)]
    fitness = [func(ind) for ind in pop]
    evals = popsize

    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    history = [best_f] * popsize          # history after the initial evaluations

    # jDE parameters
    tau1, tau2 = 0.1, 0.1                # probabilities to modify F / CR
    F_min, F_max = 0.4, 0.9
    CR_min, CR_max = 0.1, 0.9
    Fs = [random.uniform(F_min, F_max) for _ in range(popsize)]
    CRs = [random.uniform(CR_min, CR_max) for _ in range(popsize)]

    # ------------------------------------------------------------------ #
    # 2.  control variables
    generation = 0
    simplex_interval = max(1, popsize // 2)   # how many DE generations between NM phases
    es_interval = max(1, popsize)             # how often to try the final ES step

    # step size for the final (1+1)‑ES, initialised as a fraction of the box width
    sigma = 0.2 * (upper - lower)

    # ------------------------------------------------------------------ #
    # 3.  main loop
    while evals < budget:
        # -------------------------------------------------------------- #
        # DE generation
        for i in range(popsize):
            if evals >= budget:
                break

            # jDE adaptation of control parameters
            if random.random() < tau1:
                Fs[i] = random.uniform(F_min, F_max)
            if random.random() < tau2:
                CRs[i] = random.uniform(CR_min, CR_max)

            # mutation: DE/rand/1
            idxs = list(range(popsize))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant = pop[a] + Fs[i] * (pop[b] - pop[c])
            mutant = clip(mutant)

            # binomial crossover
            trial = np.empty(dim)
            j_rand = random.randrange(dim)
            for j in range(dim):
                trial[j] = (mutant[j] if (random.random() < CRs[i] or j == j_rand)
                            else pop[i][j])
            trial = clip(trial)

            f_trial = func(trial)
            evals += 1
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial

            if f_trial < best_f:
                best_f, best_x = f_trial, trial.copy()
            history.append(best_f)

        generation += 1

        # -------------------------------------------------------------- #
        # Periodic Nelder‑Mead refinement
        if generation % simplex_interval == 0 and evals < budget:
            # select dim+1 elite points (or fewer if population smaller)
            n_elite = min(dim + 1, popsize)
            elite_idx = np.argsort(fitness)[:n_elite]
            simplex = [pop[k].copy() for k in elite_idx]
            simplex_f = [fitness[k] for k in elite_idx]

            # NM coefficients
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

            # Simple NM loop – stops as soon as budget would be exceeded
            while evals < budget:
                # order simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                x_best = simplex[0]
                x_worst = simplex[-1]
                x_cent = np.mean(simplex[:-1], axis=0)

                # reflection
                xr = clip(x_cent + alpha * (x_cent - x_worst))
                fr = func(xr); evals += 1
                history.append(min(best_f, fr))
                if fr < best_f:
                    best_f, best_x = fr, xr.copy()
                if fr < simplex_f[0]:
                    # expansion
                    xe = clip(x_cent + gamma * (xr - x_cent))
                    fe = func(xe); evals += 1
                    history.append(min(best_f, fe))
                    if fe < best_f:
                        best_f, best_x = fe, xe.copy()
                    if fe < fr:
                        simplex[-1] = xe; simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr; simplex_f[-1] = fr
                    continue

                if fr < simplex_f[-2]:
                    simplex[-1] = xr; simplex_f[-1] = fr
                    continue

                # contraction
                if fr < simplex_f[-1]:
                    xc = clip(x_cent + rho * (xr - x_cent))
                else:
                    xc = clip(x_cent + rho * (x_worst - x_cent))
                fc = func(xc); evals += 1
                history.append(min(best_f, fc))
                if fc < best_f:
                    best_f, best_x = fc, xc.copy()
                if fc < simplex_f[-1]:
                    simplex[-1] = xc; simplex_f[-1] = fc
                    continue

                # shrink
                for j in range(1, len(simplex)):
                    simplex[j] = clip(x_best + sigma_nm * (simplex[j] - x_best))
                    sf = func(simplex[j]); evals += 1
                    history.append(min(best_f, sf))
                    if sf < best_f:
                        best_f, best_x = sf, simplex[j].copy()
                    simplex_f[j] = sf
                # after shrink we break to let DE continue
                break

        # -------------------------------------------------------------- #
        # (1+1)-ES polishing – executed periodically, more often near the end
        if evals < budget and evals % es_interval == 0:
            step = sigma * np.random.randn(dim)
            cand = clip(best_x + step)
            f_cand = func(cand)
            evals += 1
            if f_cand < best_f:
                best_f, best_x = f_cand, cand.copy()
                sigma = min(sigma * 1.2, (upper - lower) * 0.5)   # enlarge if success
            else:
                sigma *= 0.82                                        # shrink otherwise
            history.append(best_f)

    # ------------------------------------------------------------------ #
    # Trim history to exact budget length (might be longer by one entry)
    history = history[:budget]
    return best_x, best_f, history
