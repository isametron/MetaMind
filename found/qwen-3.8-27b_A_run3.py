# Found by arm A, run 3, cerebras/qwen-3.8-27b; training fitness 0.4725 (call 34 of 50); name: Robust IPOP-CMA-ES with Eigendecomposition (RE-IPOP-CMA-ES)
import numpy as np
import math
import random

def optimize(func, dim, bounds, budget):
    lower, upper = bounds
    span = upper - lower
    min_span = float(np.min(span))

    lambda_init = max(4 + int(3 * math.log(dim + 1)), 4)
    lambda_max = min(10 * dim, 100)

    def get_cma_params(dim, lam):
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt(mu_eff / (dim + 2)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 * (1 - c_c) + mu_eff / dim) / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / (mu_eff + 1)) / ((dim + 2) ** 2 + mu_eff))
        chi_N = math.sqrt(dim) * (1 - 1.0 / (4 * dim) - 1.0 / (25 * dim ** 2))
        return mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_N

    def safe_eigh(C):
        """Robust eigendecomposition ensuring positive eigenvalues."""
        C = (C + C.T) * 0.5
        try:
            ev, evc = np.linalg.eigh(C)
            ev = np.maximum(ev, 1e-18)
            return ev, evc
        except np.linalg.LinAlgError:
            d = np.diag(C).copy()
            d = np.maximum(d, 1e-18)
            return d, np.eye(dim)

    # Initialize
    mean = lower + np.random.rand(dim) * span
    sigma = 0.25 * min_span
    C = np.eye(dim)
    p_c = np.zeros(dim)
    p_sigma = np.zeros(dim)
    lam = lambda_init

    f0 = func(mean)
    best_x = mean.copy()
    best_f = f0
    history = [best_f]
    calls = 1

    if calls >= budget:
        return best_x, best_f, history

    calls_at_best = 1
    stagnation_limit = max(200 * dim, 2 * lambda_init * (dim + 2))
    gen_count = 0

    while calls < budget:
        mu, w, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_N = get_cma_params(dim, lam)

        while calls < budget:
            if calls - calls_at_best > stagnation_limit:
                break

            n_off = min(lam, budget - calls)
            if n_off < 2:
                break

            # Generate offspring using eigendecomposition (robust, no Cholesky)
            ev, evc = safe_eigh(C)
            sqrt_ev = np.sqrt(ev)
            Z = np.random.randn(n_off, dim)
            X = mean[None, :] + sigma * ((Z * sqrt_ev[None, :]) @ evc.T)
            np.clip(X, lower, upper, out=X)

            # Sequential evaluation
            F = np.empty(n_off)
            for i in range(n_off):
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                    calls_at_best = calls
                history.append(min(history[-1], F[i]))

            if calls >= budget:
                break

            # Sort and select parents
            idx = np.argsort(F)
            n_parents = min(mu, n_off)
            w_used = w[:n_parents].copy()
            w_used /= w_used.sum()

            # Weighted mean of selected parents
            y_w = np.zeros(dim)
            for i in range(n_parents):
                y_w += w_used[i] * (X[idx[i]] - mean) / max(sigma, 1e-30)

            # Sigma evolution path
            y_w_norm = np.linalg.norm(y_w)
            if y_w_norm > 1e-30:
                threshold = (1.4 + 2.0 / (dim + 1)) * chi_N
                h_sig = 1.0 if y_w_norm < threshold else 0.0
                p_sigma = (1 - c_sigma) * p_sigma + h_sig * math.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff) * (y_w / y_w_norm)
            else:
                h_sig = 0.0
                p_sigma = (1 - c_sigma) * p_sigma

            # C evolution path
            gen_count += 1
            ev_c, evc_c = safe_eigh(C)
            C_inv_sqrt = evc_c @ np.diag(1.0 / np.sqrt(np.maximum(ev_c, 1e-18)))
            y_c = C_inv_sqrt @ y_w
            p_c = (1 - c_c) * p_c + h_sig * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            # Rank-mu + rank-1 covariance update
            y_w_outer = np.zeros((dim, dim))
            for i in range(n_parents):
                y_i = (X[idx[i]] - mean) / max(sigma, 1e-30)
                y_w_outer += w_used[i] * np.outer(y_i, y_i)

            C = (1.0 - c_1 - c_mu + c_1 * c_sigma * (2.0 - c_sigma)) * C \
                + c_1 * (np.outer(p_sigma, p_sigma) - (1.0 - c_sigma) * C) \
                + c_mu * y_w_outer
            C = (C + C.T) * 0.5
            C += 1e-14 * np.eye(dim)

            # Step size update
            sigma *= math.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_N - 1.0))
            sigma = max(min(sigma, 10.0 * min_span), 1e-30)

            # Mean update
            mean = mean + sigma * y_w
            np.clip(mean, lower, upper, out=mean)

        if calls >= budget:
            break

        # IPOP restart with diagonal covariance inheritance
        lam = min(lam * 2, lambda_max)
        mean = best_x.copy()
        diag_C = np.diag(C).copy()
        diag_C = np.maximum(diag_C, 1e-12)
        C = np.diag(diag_C)
        sigma = 0.25 * min_span
        p_c = np.zeros(dim)
        p_sigma = np.zeros(dim)
        calls_at_best = calls
        gen_count = 0

    return best_x, best_f, history
