# Found by arm A, run 4, cerebras/qwen-3.8-27b; training fitness 0.3251 (call 16 of 16); name: unnamed
import numpy as np
import math
import random

def optimize(func, dim, bounds, budget):
    lower, upper = bounds
    range_ = upper - lower
    center = (lower + upper) / 2.0
    max_range = float(np.max(range_))
    if max_range < 1e-10:
        max_range = 1.0

    evals = 0
    history = []

    # Initialize with a random point (not center, per problem statement)
    best_x = np.random.uniform(lower, upper)
    best_f = float('inf')

    def evaluate(x):
        nonlocal evals, best_f, best_x
        x = np.clip(x, lower, upper)
        f = float(func(x))
        evals += 1
        if f < best_f:
            best_f = f
            best_x = x.copy()
        history.append(float(best_f))
        return f

    def cma_params(lam, d):
        mu = max(1, lam // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.dot(w, w)
        c_sig = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        d_sig = 1.0 + 2.0 * max(0.0, (mu_eff - 2.0) / (d + mu_eff - 2.0)) + c_sig
        c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c_cov_pos = (2.0 + 2.0 * min(w[0], 0.5) * mu_eff / d) / (d + 1.0 + 3.0 * mu_eff / d)
        c_cov_neg = (1.0 - c_cov_pos) * 2.0 * min(w[0], 0.5)
        chi_d = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (25.0 * d * d))
        return mu, w, mu_eff, c_sig, d_sig, c_c, c_cov_pos, c_cov_neg, chi_d

    # Population size: standard CMA-ES formula, capped by budget
    base_lam = max(4, int(4 + 3 * math.log(max(dim, 2))))
    lam = base_lam
    max_lam = max(base_lam * 2, min(4 * base_lam, budget // (dim + 3)))

    mu, w, mu_eff, c_sig, d_sig, c_c, c_cov_pos, c_cov_neg, chi_d = cma_params(lam, dim)

    # Initial CMA-ES state
    x_mean = center.copy()
    sigma = 0.25 * max_range
    C = np.eye(dim)
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)

    restart_count = 0
    no_improve = 0
    stall_threshold = max(15, 2 * (dim + 3))

    while evals < budget:
        n_off = min(lam, budget - evals)
        if n_off < 2:
            # Spend remaining budget on local perturbation around best
            step = max(sigma * 0.05, 1e-10 * max_range)
            while evals < budget:
                x_try = best_x + step * np.random.randn(dim)
                evaluate(x_try)
            break

        # CRITICAL FIX: record best before generation for correct stagnation tracking
        best_before_gen = best_f

        # Cholesky decomposition of covariance
        try:
            C_cho = np.linalg.cholesky(C + 1e-16 * np.eye(dim))
        except np.linalg.LinAlgError:
            C_cho = np.eye(dim)

        # Generate and evaluate offspring
        x_off = np.empty((n_off, dim))
        f_off = np.empty(n_off)

        for i in range(n_off):
            if evals >= budget:
                n_off = i
                break
            z = np.random.randn(dim)
            xi = x_mean + sigma * (C_cho @ z)
            f_off[i] = evaluate(xi)
            x_off[i] = np.clip(xi, lower, upper)

        if n_off < 2 or evals >= budget:
            break

        # Sort offspring by fitness (ascending)
        order = np.argsort(f_off[:n_off])
        x_off = x_off[:n_off][order].copy()
        f_off = f_off[:n_off][order].copy()

        # Corrected stagnation counter: compare generation best vs pre-generation best
        if f_off[0] < best_before_gen - 1e-15:
            no_improve = 0
        else:
            no_improve += 1

        # --- CMA-ES Adaptation Step ---
        try:
            # Transform to white space for adaptation
            ys = np.zeros((min(mu, n_off), dim))
            for i in range(min(mu, n_off)):
                ys[i] = np.linalg.solve(C_cho.T, (x_off[i] - x_mean) / sigma)

            y_bar = np.zeros(dim)
            for i in range(min(mu, n_off)):
                y_bar += w[i] * ys[i]

            h_sig = 1.0 if np.linalg.norm(y_bar) < (1.0 + 1.5 / (dim + 1)) * chi_d else 0.0

            # Step-size adaptation (CSA)
            p_sigma = (1 - c_sig) * p_sigma + h_sig * math.sqrt(c_sig * (2 - c_sig) * mu_eff) * y_bar
            sigma *= math.exp(c_sig / d_sig * (np.linalg.norm(p_sigma) / chi_d - 1.0))
            sigma = max(1e-15 * max_range, min(sigma, 10.0 * max_range))

            # Covariance evolution path
            p_c = (1 - c_c) * p_c + h_sig * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_bar

            # Rank-mu + rank-1 covariance matrix update
            delta_h = (h_sig * (2 - c_c) - 1) * c_c * (1 - c_c)
            C *= (1.0 + c_cov_pos * (c_cov_neg + delta_h) - c_cov_pos)
            for i in range(min(mu, n_off)):
                C += c_cov_pos * w[i] * np.outer(ys[i], ys[i])
            C = (C + C.T) * 0.5

            # Enforce positive definiteness
            eig_min = np.linalg.eigvalsh(C)[0]
            if eig_min < 1e-14:
                C += (1e-14 - eig_min + 1e-15) * np.eye(dim)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

        # Update mean via weighted recombination of top-m offspring
        x_mean = np.zeros(dim)
        for i in range(min(mu, n_off)):
            x_mean += w[i] * x_off[i]
        x_mean = np.clip(x_mean, lower, upper)

        # === RESTART AND ADAPTATION DECISION ===
        should_restart = False

        # 1) Primary stagnation criterion
        if no_improve >= stall_threshold:
            should_restart = True

        # 2) Sigma has collapsed below useful range
        elif sigma < 1e-12 * max_range:
            should_restart = True

        # 3) Condition number explosion (checked periodically to save compute)
        if not should_restart and no_improve > 5 and no_improve % 3 == 0:
            try:
                ev = np.linalg.eigvalsh(C)
                if ev[0] > 0 and ev[-1] / ev[0] > 1e8:
                    should_restart = True
            except (np.linalg.LinAlgError, ValueError):
                should_restart = True

        # 4) Budget-aware: become more aggressive in the final 30% of budget
        if not should_restart and evals > 0.7 * budget and no_improve > max(4, dim // 3):
            should_restart = True

        # 5) Hard safeguard: absolute upper limit on consecutive non-improving generations
        if not should_restart and no_improve > max(5 * dim, 30):
            should_restart = True

        # Execute restart if triggered
        if should_restart and evals + 2 < budget:
            restart_count += 1

            # IPOP: double population size (capped)
            new_lam = min(lam * 2, max_lam, max(4, (budget - evals) // (dim + 3)))
            if new_lam > lam:
                lam = new_lam
                mu, w, mu_eff, c_sig, d_sig, c_c, c_cov_pos, c_cov_neg, chi_d = cma_params(lam, dim)

            # Adaptive restart strategy:
            #   - First restart: fully diversified (explore new region)
            #   - Subsequent: mostly focused (exploit around best), occasional diversification
            if restart_count == 1:
                x_mean = np.random.uniform(lower, upper)
                sigma = 0.25 * max_range
            elif random.random() < 0.25:
                # Diversified restart with moderate step
                x_mean = np.random.uniform(lower, upper)
                sigma = 0.15 * max_range
            else:
                # Focused restart: shrink step, center on best known solution
                x_mean = best_x.copy()
                sigma = max(sigma * 0.3, max_range * 1e-8)

            # Reset all adaptation state
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            no_improve = 0
            # Gradually increase patience for later restarts (let exploitation run longer)
            stall_threshold = max(15, 2 * (dim + 3)) + restart_count * 3

    return best_x, best_f, history
