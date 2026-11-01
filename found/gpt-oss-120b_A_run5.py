# Found by arm A, run 5, cerebras/gpt-oss-120b; training fitness 0.5017 (call 48 of 50); name: SobolCMA‑DEHybrid
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Sobol‑seeded CMA‑ES with a light DE fallback.
    -------------------------------------------------
    1. Sobol (low‑discrepancy) sampling for an initial set of points.
    2. Small‑population CMA‑ES (μ/λ) drives the main search.
    3. For every CMA offspring a DE/rand/1 trial is generated;
       the better of the two is kept (adds diversity at almost no cost).
    4. Simple restart when the step size becomes too small or the
       remaining budget cannot sustain a full generation.
    5. History records the best‑so‑far value after each evaluation.
    """
    lower, upper = bounds
    span = upper - lower

    def clip(x):
        """Clip a vector or an array of vectors to the bounds."""
        return np.minimum(np.maximum(x, lower), upper)

    # ------------------------------------------------------------------ #
    # 1. Sobol low‑discrepancy initial sampling (no external libs)
    # ------------------------------------------------------------------ #
    def sobol_points(n, d):
        """Generate n Sobol points in d dimensions using a simple direction‑number
        generator (good enough for modest d ≤ 20)."""
        # primitive polynomials for first 20 dimensions (seeded from literature)
        primitive = [
            0, 1, 1, 2, 1, 4, 2, 7, 1, 11, 2, 13, 1, 14, 2, 19,
            1, 21, 2, 22, 1, 25, 2, 26, 1, 28, 2, 30, 1, 31, 2, 37,
            1, 38, 2, 40, 1, 41, 2, 42, 1, 44, 2, 45, 1, 46, 2, 49,
            1, 50, 2, 51, 1, 52, 2, 53, 1, 54, 2, 55, 1, 56, 2, 58,
            1, 59, 2, 60, 1, 61, 2, 62, 1, 63, 2, 65, 1, 66, 2, 67,
            1, 68, 2, 69, 1, 70, 2, 71, 1, 72, 2, 73, 1, 74, 2, 75,
            1, 76, 2, 77, 1, 78, 2, 79, 1, 80, 2, 81, 1, 82, 2, 83,
            1, 84, 2, 85, 1, 86, 2, 87, 1, 88, 2, 89, 1, 90, 2, 91,
            1, 92, 2, 93, 1, 94, 2, 95, 1, 96, 2, 97, 1, 98, 2, 99,
        ]
        # direction numbers (only first 30 bits are needed)
        V = np.zeros((d, 30), dtype=np.uint32)
        for i in range(d):
            m = primitive[i] if i < len(primitive) else 1
            for j in range(30):
                V[i, j] = 1 << (29 - j)   # 2^(30‑j‑1)
        points = np.empty((n, d))
        for i in range(1, n + 1):
            gray = i ^ (i >> 1)
            for k in range(d):
                val = 0
                g = gray
                bit = 0
                while g:
                    if g & 1:
                        val ^= V[k, bit]
                    g >>= 1
                    bit += 1
                points[i - 1, k] = val / (1 << 30)
        return points

    init_pop = min(budget, max(5 * dim, 30))
    sobol = sobol_points(init_pop, dim)
    best_x = None
    best_f = float('inf')
    history = []
    evals = 0

    for i in range(init_pop):
        x = clip(lower + sobol[i] * span)
        f = func(x)
        evals += 1
        if f < best_f:
            best_f = float(f)
            best_x = x.copy()
        history.append(best_f)
        if evals >= budget:
            return best_x, best_f, history

    # ------------------------------------------------------------------ #
    # 2. CMA‑ES core (tiny population) + DE fallback
    # ------------------------------------------------------------------ #
    sigma = 0.4 * np.mean(span)          # global step size
    m = best_x.copy()                    # mean starts at best init point

    pop_size = max(4, int(2 * dim))      # λ
    mu = pop_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)

    # CMA‑ES strategy parameters (small‑pop tuned)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1,
              2 * (mu_eff - 2 + 1) / ((dim + 2) ** 2 + mu_eff))
    damps = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

    ps = np.zeros(dim)
    pc = np.zeros(dim)
    C = np.eye(dim)

    # DE parameters
    F_DE = 0.7
    CR_DE = 0.9

    generation = 0
    restarts = 0
    max_restarts = 3

    while evals < budget:
        # ----- restart when step size collapses or not enough evals -----
        if (sigma * np.mean(np.sqrt(np.diag(C))) < 1e-8 or
                evals + 2 * pop_size > budget) and restarts < max_restarts:
            sigma *= 2.0
            pop_size = int(pop_size * 1.5) + 1
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1,
                      2 * (mu_eff - 2 + 1) / ((dim + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            restarts += 1
            continue

        # ----- CMA‑ES sampling -----
        arz = np.random.randn(pop_size, dim)          # N(0,I)
        D, B = np.linalg.eigh(C)                      # eigen decomposition
        D = np.sqrt(np.maximum(D, 1e-20))
        BD = B * D                                    # transformation matrix
        offspring = m + sigma * (arz @ BD.T)          # λ candidates
        offspring = clip(offspring)

        # ----- DE/rand/1 trial for each offspring -----
        de_trials = np.empty_like(offspring)
        for i in range(pop_size):
            idx = list(range(pop_size))
            idx.remove(i)
            r1, r2, r3 = random.sample(idx, 3)
            mutant = offspring[r1] + F_DE * (offspring[r2] - offspring[r3])
            # binomial crossover
            jrand = random.randrange(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if random.random() < CR_DE or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = offspring[i, j]
            de_trials[i] = clip(trial)

        # ----- evaluation, keep the better of the two -----
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            f_cma = func(offspring[i])
            f_de = func(de_trials[i])
            evals += 2
            if f_de < f_cma:
                offspring[i] = de_trials[i]
                fitness[i] = f_de
            else:
                fitness[i] = f_cma

            if fitness[i] < best_f:
                best_f = float(fitness[i])
                best_x = offspring[i].copy()
            history.append(best_f)

            if evals >= budget:
                break
        if evals >= budget:
            break

        # ----- selection & recombination -----
        order = np.argsort(fitness)
        selected = offspring[order[:mu]]
        old_m = m.copy()
        m = np.dot(weights, selected)

        # ----- evolution paths -----
        y = (m - old_m) / sigma
        C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ y)

        norm_ps = np.linalg.norm(ps)
        chi_n = math.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        hsig = 1.0 if norm_ps / math.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) else 0.0

        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mu_eff) * y

        # ----- covariance update -----
        artmp = (selected - old_m) / sigma
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * np.dot((weights * artmp.T), artmp))

        # ----- step‑size control -----
        sigma *= math.exp((cs / damps) * (norm_ps / chi_n - 1))

        generation += 1

    # Ensure history length equals number of evaluations performed
    while len(history) < evals:
        history.append(best_f)

    return best_x, best_f, history
