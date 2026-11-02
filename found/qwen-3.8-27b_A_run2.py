# Found by arm A, run 2, cerebras/qwen-3.8-27b; training fitness 0.4947 (call 43 of 50); name: IPOP-CMA-ES with Adaptive Stagnation and Progressive Sigma (ASPS-CMA-ES)
def optimize(func, dim, bounds, budget):
    import numpy as np
    import math
    import random

    lower, upper = bounds
    size = upper - lower
    box_size = float(np.mean(size))
    if box_size < 1e-10:
        box_size = 1.0

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
            best_x = np.array(x, dtype=float)
        history.append(best_f)
        return f

    def clip(x):
        return np.clip(x, lower, upper)

    if budget <= 0:
        return np.zeros(dim), float('inf'), history

    # Initial random point
    x0 = lower + size * np.random.rand(dim)
    evaluate(x0)

    if n_evals >= budget:
        return best_x, best_f, history

    def cmaes_run(x_center, sigma_init, lam, C_init=None):
        nonlocal n_evals, best_f, best_x

        n = dim
        mu = max(1, lam // 2)
        log_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        w = log_w / log_w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        cc = (4.0 - mu_eff / n) / (n + 4.0 - 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(0.5, (2.0 - c1 + 2.0 * (mu_eff - 2.0 + 2.0 / mu_eff) /
                        ((n + 2.0) * (mu_eff - 2.0) + 2.0 / mu_eff)) - c1)
        ccov = c1 + cmu
        damps = n + 2.0 * math.sqrt(max((mu_eff - 1.0) / (n + 2.0), 0.0)) - 1.0
        damps = max(damps, 1.0)
        D_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) - 1.0 / (25.0 * n ** 2))
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        use_active = (n <= 50 and lam > mu)
        if use_active:
            w_neg = np.ones(lam - mu) / (mu * (1.0 - w[mu - 1]))
            w_neg /= w_neg.sum()
            cmu_neg = min(cmu, cmu * (1.0 - w[mu - 1]) / w_neg[0])
        else:
            w_neg = None
            cmu_neg = 0.0

        x_mean = x_center.copy()
        sigma = sigma_init

        if C_init is not None:
            C = C_init.copy()
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, 1e20)
                C = (eigvecs * eigvals) @ eigvecs.T
                C_sqrt = eigvecs * np.sqrt(eigvals)
            except Exception:
                C = np.eye(n)
                C_sqrt = np.eye(n)
        else:
            C = np.eye(n)
            C_sqrt = np.eye(n)

        p_c = np.zeros(n)
        p_sigma = np.zeros(n)

        sigma_min = 1e-12 * box_size
        sigma_max = 2.0 * box_size

        best_at_start = best_f
        best_local = best_f

        n_off = min(lam, budget - n_evals)
        if n_off <= 0:
            return x_mean, sigma, C, 0, False, False

        # Generate initial population
        pop = np.empty((n_off, n))
        fit = np.empty(n_off)
        for i in range(n_off):
            z = np.random.randn(n)
            pop[i] = clip(x_mean + sigma * (C_sqrt @ z))
            fit[i] = evaluate(pop[i])

        if n_evals >= budget:
            improved = best_f < best_at_start - max(1e-15, abs(best_at_start) * 1e-10)
            return x_mean, sigma, C, 0, improved, False

        actual_lam = n_off
        if actual_lam < 2:
            improved = best_f < best_at_start - max(1e-15, abs(best_at_start) * 1e-10)
            return x_mean, sigma, C, 0, improved, False

        gens = 0
        no_improve = 0
        # Adaptive stagnation: generous threshold based on dim and lam
        stag_thresh = max(25, int(3.0 * lam + 1.5 * dim))
        # Max generations per run to prevent wasting budget
        max_gens = max(40, int(6 * lam + 2 * dim))

        while n_evals < budget and gens < max_gens:
            gens += 1

            idx = np.argsort(fit[:actual_lam])
            pop = pop[idx].copy()
            fit = fit[idx].copy()

            if fit[0] < best_local - max(1e-15, abs(best_local) * 1e-10):
                best_local = fit[0]
                no_improve = 0
            no_improve += 1

            restart = False
            cond_bad = False

            # Condition number check (infrequent to save compute)
            check_freq = 4 if n <= 10 else 6
            if gens % check_freq == 0:
                try:
                    ev = np.linalg.eigvalsh(C)
                    ev = np.clip(ev, 1e-30, None)
                    cond = ev[-1] / ev[0]
                    if cond > 1e14:
                        restart = True
                        cond_bad = True
                except Exception:
                    restart = True
                    cond_bad = True

            # Sigma collapse
            if not restart and sigma < sigma_min * 1e3:
                restart = True

            # Stagnation
            if not restart and no_improve > stag_thresh:
                restart = True

            if restart:
                break

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu][:, None] * pop[:mu], axis=0)

            # Evolution paths
            diff = (pop[:mu] - x_mean_old) / max(sigma, 1e-30)
            c_mean = np.sum(w[:mu][:, None] * diff, axis=0)
            c_mean_norm = np.linalg.norm(c_mean)

            p_sigma = (1.0 - cs) * p_sigma + cs * math.sqrt(mu_eff) * c_mean
            h_sigma = 1.0 if c_mean_norm < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0
            p_c = (1.0 - cc) * p_c + cc * h_sigma * (x_mean - x_mean_old) / max(sigma, 1e-30)

            # Covariance update
            C = (1.0 - ccov) * C
            C += c1 * (np.outer(p_c, p_c) + (cc * (2.0 - cc) - 1.0) * C)
            C += cmu * (diff.T @ (w[:mu][:, None] * diff))

            if use_active and actual_lam > mu:
                n_neg = actual_lam - mu
                diff_neg = (pop[mu:actual_lam] - x_mean_old) / max(sigma, 1e-30)
                wn = w_neg[:n_neg]
                C += cmu_neg * (diff_neg.T @ (wn[:, None] * diff_neg))

            C = (C + C.T) / 2.0

            # Eigendecomposition for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, 1e20)
                C = (eigvecs * eigvals) @ eigvecs.T
                C_sqrt = eigvecs * np.sqrt(eigvals)
            except Exception:
                C = np.eye(n)
                C_sqrt = np.eye(n)

            # Sigma adaptation
            sigma *= math.exp(cs / damps * (c_mean_norm / D_n - 1.0))
            sigma = max(sigma, sigma_min)
            sigma = min(sigma, sigma_max)

            # Generate offspring
            n_off = min(lam, budget - n_evals)
            if n_off <= 0:
                break

            offspring = np.empty((n_off, n))
            off_fit = np.empty(n_off)
            for i in range(n_off):
                z = np.random.randn(n)
                offspring[i] = clip(x_mean + sigma * (C_sqrt @ z))
                off_fit[i] = evaluate(offspring[i])

            # (mu+lambda) selection
            combined = np.vstack([pop[:actual_lam], offspring])
            combined_f = np.concatenate([fit[:actual_lam], off_fit])
            cidx = np.argsort(combined_f)
            new_lam = min(lam, len(combined))
            pop = combined[cidx[:new_lam]].copy()
            fit = combined_f[cidx[:new_lam]].copy()
            actual_lam = new_lam

        improved = best_f < best_at_start - max(1e-15, abs(best_at_start) * 1e-8)
        return x_mean, sigma, C, gens, improved, cond_bad

    # IPOP main loop with progressive sigma adaptation
    lam = max(4, int(4 + 3 * math.log(dim + 1)))
    lam = min(lam, max(4, budget - 1))

    x_mean = x0.copy()
    sigma = 0.35 * box_size
    C_inherit = None
    restart_count = 0
    prev_sigma = sigma

    while n_evals < budget:
        remaining = budget - n_evals
        if remaining < 4:
            break

        lam_eff = min(lam, remaining - 1)
        lam_eff = max(lam_eff, 4)

        x_mean, sigma, C, gens, improved, cond_bad = cmaes_run(
            x_mean, sigma, lam_eff, C_inherit)
        restart_count += 1

        if n_evals >= budget:
            break

        remaining = budget - n_evals
        if remaining < 4:
            break

        # Slower population increase for more restarts
        lam = min(int(math.ceil(lam * 1.25)) + 1, max(4, remaining // 2))
        lam = max(lam, 4)

        # Progressive sigma reset based on progress and sigma trajectory
        sigma_shrank = sigma < prev_sigma * 0.5  # sigma decreased significantly
        if improved and not cond_bad:
            if sigma_shrank:
                # Was converging well; restart slightly wider
                sigma = max(0.15 * box_size, sigma * 1.2)
            else:
                sigma = max(0.12 * box_size, sigma * 0.8)
        elif improved:
            sigma = max(0.2 * box_size, sigma * 0.9)
        else:
            # No improvement: explore much more broadly
            sigma = max(0.45 * box_size, sigma * 2.0)

        prev_sigma = sigma

        # Re-center on best point
        if best_x is not None:
            x_mean = best_x.copy()

        # Covariance inheritance with conservative blending
        if cond_bad:
            C_inherit = None
        else:
            try:
                ev = np.linalg.eigvalsh(C)
                ev = np.clip(ev, 1e-30, None)
                cond = ev[-1] / ev[0]
                if cond < 1e4:
                    C_inherit = C.copy()
                elif cond < 1e7:
                    C_inherit = 0.25 * C + 0.75 * np.eye(dim)
                else:
                    C_inherit = None
            except Exception:
                C_inherit = None

        if restart_count > 500:
            break

    return best_x, best_f, history
