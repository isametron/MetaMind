# Found by arm A, run 2, cerebras/qwen-3.8-27b; training fitness 0.1351 (call 5 of 5); name: Differential CMA-ES with Adaptive Restarts (DCMA-AR)
def optimize(func, dim, bounds, budget):
    import numpy as np
    import math
    import random

    lower, upper = bounds
    size = upper - lower
    mean_size = float(np.mean(size))

    n_evals = 0
    history = []
    best_f = float('inf')
    best_x = None

    def evaluate(x):
        nonlocal n_evals, best_f, best_x
        f = float(func(x))
        n_evals += 1
        if f < best_f:
            best_f = f
            best_x = x.copy()
        history.append(best_f)
        return f

    def clip(x):
        return np.clip(x, lower, upper)

    # Population size
    lam = max(4, int(4 + 3 * math.log(dim + 1)))
    lam = min(lam, max(4, budget - 1))
    if lam < 2:
        # Budget too small for CMA-ES, just sample randomly
        for _ in range(budget):
            x = lower + size * np.random.rand(dim)
            evaluate(x)
        return best_x, best_f, history

    mu = max(1, lam // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
    weights = weights / weights.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)

    c_sigma = (mu_eff / 2 + 1.0) / (dim + mu_eff / 2 + 5.0)
    c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 4.0 * mu_eff / dim)
    c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    if mu_eff > 2.0:
        c_mu = min(0.5, (1.0 - c_1 + 2.0 * (mu_eff - 2.0 + 2.0 / mu_eff) /
                         ((dim + 2.0) * (mu_eff - 2.0) + 2.0 / mu_eff)) - c_1)
    else:
        c_mu = max(0.0, 1.0 - c_1)

    # Initialize from random point
    x0 = lower + size * np.random.rand(dim)
    sigma = 0.25 * mean_size
    x_mean = x0.copy()
    C = np.eye(dim).astype(float)
    C_sqrt = np.eye(dim).astype(float)
    p_c = np.zeros(dim)
    p_sigma = np.zeros(dim)

    # Initial population
    pop = np.zeros((lam, dim))
    fit = np.zeros(lam)
    actual_lam = 0
    for i in range(lam):
        if n_evals >= budget:
            break
        z = np.random.randn(dim)
        pop[i] = clip(x_mean + sigma * (C_sqrt @ z))
        fit[i] = evaluate(pop[i])
        actual_lam = i + 1

    if actual_lam < 2:
        return best_x, best_f, history

    lam = actual_lam
    mu = max(1, lam // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
    weights = weights / weights.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)

    no_improve = 0
    last_best = float('inf')
    gen = 0
    max_gens = max(100, int(300 * (1 + dim) * math.log(dim + 1) / lam))
    restart_count = 0
    max_restarts = 5

    while n_evals < budget and gen < max_gens:
        gen += 1

        idx = np.argsort(fit[:lam])
        pop = pop[idx].copy()
        fit = fit[idx].copy()

        if fit[0] < last_best - max(1e-14, abs(last_best) * 1e-12):
            last_best = fit[0]
            no_improve = 0
        else:
            no_improve += 1

        # Hard restart if completely stuck
        restart_threshold = max(10 * lam, int(3 * (1 + dim) * lam * math.log(dim + 1)))
        if no_improve > restart_threshold and restart_count < max_restarts:
            restart_count += 1
            x_mean = best_x.copy()
            sigma = 0.4 * mean_size
            C = np.eye(dim).astype(float)
            C_sqrt = np.eye(dim).astype(float)
            p_c = np.zeros(dim)
            p_sigma = np.zeros(dim)
            no_improve = 0
            last_best = fit[0]

            for i in range(lam):
                if n_evals >= budget:
                    break
                z = np.random.randn(dim)
                pop[i] = clip(x_mean + sigma * (C_sqrt @ z))
                fit[i] = evaluate(pop[i])
            continue

        # Update mean
        x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

        # Generate offspring
        n_off = min(lam, budget - n_evals)
        if n_off <= 0:
            break

        offspring = np.empty((n_off, dim))
        off_fit = np.empty(n_off)

        for i in range(n_off):
            if n_evals >= budget:
                n_off = i
                break
            z = np.random.randn(dim)

            # Differential mutation for diversity (applied to ~half the population)
            if i >= lam // 2 and lam >= 4:
                r1 = random.randint(0, lam - 1)
                r2 = random.randint(0, lam - 1)
                while r2 == r1:
                    r2 = random.randint(0, lam - 1)
                F = 0.5 + 0.5 * random.random()
                diff_vec = pop[r1] - pop[r2]
                # Blend CMA-ES step with differential step
                cma_step = x_mean + sigma * (C_sqrt @ z)
                de_step = x_mean + F * diff_vec
                blend = 0.4 + 0.2 * (no_improve / max(lam, 1))
                x_new = (1.0 - blend) * cma_step + blend * de_step
            else:
                x_new = x_mean + sigma * (C_sqrt @ z)

            offspring[i] = clip(x_new)
            off_fit[i] = evaluate(offspring[i])

        if n_off == 0:
            break
        offspring = offspring[:n_off]
        off_fit = off_fit[:n_off]

        # Combined selection (mu+lambda)
        combined = np.vstack([pop[:lam], offspring])
        combined_f = np.concatenate([fit[:lam], off_fit])
        cidx = np.argsort(combined_f)
        new_lam = min(lam, len(combined))
        pop = combined[cidx[:new_lam]].copy()
        fit = combined_f[cidx[:new_lam]].copy()
        lam = new_lam

        # Recompute strategy parameters
        mu = max(1, lam // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

        # Sigma and covariance adaptation
        ref_idx = min(mu, lam - 1)
        ref_fit_val = fit[ref_idx]
        n_success = int(np.sum(off_fit <= ref_fit_val))
        success_rate = n_success / max(n_off, 1)

        if mu >= 2:
            diff = (pop[:mu] - x_mean) / max(sigma, 1e-30)
            c_mean = np.sum(weights[:, None] * diff, axis=0)
            c_mean_norm = np.linalg.norm(c_mean)

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * math.sqrt(mu_eff) * c_mean
            h_sigma = (c_mean_norm / math.sqrt(mu_eff) <
                      (1.4 + 2.0 / (dim + 1)) * (1 - c_sigma))

            p_c = (1 - c_c) * p_c + c_c * math.sqrt(c_c * (2 - c_c) * mu_eff) * c_mean

            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (c_c * (2 - c_c) - 1) * C)
            C += c_mu * (diff.T @ (weights[:, None] * diff))
            C = (C + C.T) / 2.0

            # Periodic eigendecomposition for numerical stability
            if gen % 10 == 0 or gen < 5:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, 1e10)
                C = (eigvecs * eigvals) @ eigvecs.T
                C_sqrt = eigvecs * np.sqrt(eigvals)

            # Path-based sigma update
            if h_sigma:
                sigma *= math.exp(c_sigma * (c_mean_norm / math.sqrt(mu_eff) - 1.0))

        # Success-rate based sigma adjustment
        if success_rate > 0.4:
            sigma *= 1.03
        elif success_rate < 0.1:
            sigma *= 0.9

        sigma = max(sigma, 1e-12)
        sigma = min(sigma, 5.0 * mean_size)

        # Soft diversity injection when progress is slow
        if no_improve > 3 * lam and n_evals < budget - 1:
            worst_idx = lam - 1
            pop[worst_idx] = clip(lower + size * np.random.rand(dim))
            fit[worst_idx] = evaluate(pop[worst_idx])
            no_improve = max(0, no_improve - lam // 2)
            sigma = min(sigma * 1.3, 2.0 * mean_size)

    return best_x, best_f, history
