# Found by arm A, run 1, cerebras/qwen-3.8-27b; training fitness 0.5066 (call 46 of 50); name: unnamed
import numpy as np
import math
import random


def optimize(func, dim, bounds, budget):
    lower = np.array(bounds[0], dtype=np.float64)
    upper = np.array(bounds[1], dtype=np.float64)
    range_ = upper - lower
    max_range = float(np.max(range_))

    best_x = None
    best_f = np.inf
    history = []
    calls = 0

    def evaluate(x):
        nonlocal calls, best_x, best_f, history
        x = np.clip(x, lower, upper)
        f = float(func(x))
        calls += 1
        if f < best_f:
            best_f = f
            best_x = x.copy()
        history.append(best_f)
        return f

    def regularize_C(C, max_cond=1e10):
        C = (C + C.T) / 2.0
        try:
            ev, evec = np.linalg.eigh(C)
            ev = np.maximum(ev, 1e-18)
            max_ev = ev[-1]
            min_ev = ev[0]
            if max_ev / max(min_ev, 1e-30) > max_cond:
                target_min = max_ev / max_cond
                ev = np.maximum(ev, target_min)
            C = (evec * ev) @ evec.T
            C = (C + C.T) / 2.0
        except Exception:
            C = np.eye(C.shape[0])
        return C

    def cmaes_run(center, sigma, C_init, max_evals):
        nonlocal calls, best_f, best_x
        n = dim
        lam = max(4, int(4 + 3 * math.log(n)))
        mu = max(1, lam // 2)
        w = np.log(mu + 1.0) - np.log(np.arange(1, mu + 1, dtype=float))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1,
                   2.0 * (mu_eff - 2.0 + 2.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))

        mean = center.copy()
        if C_init is not None:
            C = regularize_C(C_init, max_cond=1e10)
        else:
            C = np.eye(n)

        sigma = max(sigma, 1e-10)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        no_improve = 0
        best_local = best_f
        evals_used = 0
        progress = 0.0

        while calls < budget:
            if evals_used + lam > max_evals:
                break

            try:
                ev, evec = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-18)
                sev = np.sqrt(ev)
            except Exception:
                ev = np.ones(n)
                sev = np.ones(n)
                evec = np.eye(n)

            Z = np.random.randn(lam, n)
            offspring = mean[np.newaxis, :] + sigma * (Z @ (evec * sev).T)
            offspring = np.clip(offspring, lower, upper)

            fits = np.empty(lam)
            for i in range(lam):
                if calls >= budget:
                    return C, sigma, progress
                fits[i] = evaluate(offspring[i])
            evals_used += lam

            idx = np.argsort(fits)

            if fits[idx[0]] < best_local - 1e-12 * max(1.0, abs(best_local)):
                progress += (best_local - fits[idx[0]]) / max(1.0, abs(best_local))
                best_local = fits[idx[0]]
                no_improve = 0
            else:
                no_improve += 1

            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += w[i] * offspring[idx[i]]

            Y = (offspring[idx[:mu]] - mean) / max(sigma, 1e-30)
            y_bar = np.zeros(n)
            for i in range(mu):
                y_bar += w[i] * Y[i]

            ps = (1.0 - c_sigma) * ps + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * y_bar
            ps_norm = np.linalg.norm(ps)
            h_sig = 1.0 if ps_norm < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0

            sigma *= math.exp(c_sigma / chi_n * (ps_norm - 1.0) / chi_n)
            sigma = max(min(sigma, 1e6 * max_range), 1e-30)

            pc = (1.0 - c_c) * pc + h_sig * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_bar

            dh = (h_sig - 1.0) * c_c * (2.0 - c_c)
            C = (1.0 + c_1 * dh - c_1 - c_mu) * C
            C += c_1 * np.outer(pc, pc)
            for i in range(mu):
                C += c_mu * w[i] * np.outer(Y[i], Y[i])
            C = regularize_C(C, max_cond=1e10)

            mean = np.clip(new_mean, lower, upper)

            # Termination criteria
            cond = ev[-1] / max(ev[0], 1e-30)
            if cond > 1e14:
                break
            if sigma < 1e-14 * max_range:
                break
            if no_improve > 15 * (n + 1):
                break
            if no_improve > 8 and sigma < 1e-8 * max_range:
                break

        return C, sigma, progress

    # === Phase 1: Initial diverse sampling ===
    n_init = min(budget, max(4, min(2 * dim, dim + 10)))
    for i in range(n_init):
        if calls >= budget:
            break
        x = lower + np.random.rand(dim) * range_
        evaluate(x)

    if best_x is None:
        return lower.copy(), np.inf, history

    # === Phase 2: Adaptive CMA-ES with covariance-inherited restarts ===
    sigma_init = 0.25 * max_range
    C_memory = None
    sigma_out = sigma_init
    n_restart = 0
    last_progress = 1.0

    while calls < budget and n_restart < 200:
        rem = budget - calls
        if rem < 4:
            break

        if n_restart == 0:
            center = best_x.copy()
            sigma = sigma_init
            C_init = None
            alloc = min(rem, max(20, int(0.5 * rem)))
        else:
            # Progress-gated sigma shrinkage
            if last_progress > 0.5:
                shrink = 0.75
            elif last_progress > 0.1:
                shrink = 0.5
            else:
                shrink = 0.35
            sigma = sigma_init * (shrink ** n_restart)

            # Diversity-gated center selection
            diversity_prob = min(0.35, 0.05 + 0.04 * n_restart)
            if random.random() < diversity_prob:
                r = lower + np.random.rand(dim) * range_
                mix = min(0.5, 0.2 + 0.1 * n_restart)
                center = (1.0 - mix) * best_x + mix * r
            else:
                center = best_x.copy()
            center = np.clip(center, lower, upper)

            # Covariance inheritance with exponential decay
            if C_memory is not None and n_restart <= 6:
                blend = 0.7 ** n_restart
                C_init = blend * C_memory + (1.0 - blend) * np.eye(dim)
            else:
                C_init = None

            # Decreasing budget allocation
            alloc = min(rem, max(8, int(0.25 * rem)))

        C_memory, sigma_out, last_progress = cmaes_run(center, sigma, C_init, alloc)
        n_restart += 1

    # === Phase 3: Stochastic local refinement ===
    if calls < budget and best_x is not None:
        rem = budget - calls
        if rem >= 2:
            step = max(sigma_out * 0.1, 1e-8 * max_range)
            stall = 0
            while calls < budget and stall < 40:
                z = np.random.randn(dim)
                x_try = np.clip(best_x + step * z, lower, upper)
                f_try = evaluate(x_try)
                if f_try < best_f:
                    step *= 1.3
                    stall = 0
                else:
                    step *= 0.7
                    stall += 1
                    if step < 1e-15 * max_range:
                        break

    return best_x, best_f, history
