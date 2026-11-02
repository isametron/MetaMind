# Found by arm A, run 3, cerebras/qwen-3.8-27b; training fitness 0.4580 (call 2 of 7); name: Covariance Memory Restart CMA-ES (CMR-CMA-ES)
import numpy as np
import math
import random

def optimize(func, dim, bounds, budget):
    lower, upper = bounds
    span = upper - lower
    min_span = float(np.min(span))

    # Population size
    lam = max(4 + int(4 * math.log(dim + 1)), 10)

    # CMA-ES strategy parameters
    mu = lam // 2
    w = np.log(mu + 1.0) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mu_eff = 1.0 / np.sum(w ** 2)
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, math.sqrt(mu_eff / (dim + 2)) - 1) + c_sigma
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c_1 = (2 + 2 * (1 - c_c) + mu_eff / dim) / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / (mu_eff + 1)) / ((dim + 2) ** 2 + mu_eff))
    chi_N = math.sqrt(dim) * (1 - 1.0 / (4 * dim) - 1.0 / (25 * dim ** 2))

    # Initialize state
    mean = lower + np.random.rand(dim) * span
    sigma = 0.4 * min_span
    C = np.eye(dim)
    p_c = np.zeros(dim)
    p_sigma = np.zeros(dim)

    f0 = func(mean)
    best_x = mean.copy()
    best_f = f0
    history = [best_f]
    calls = 1

    if calls >= budget:
        return best_x, best_f, history

    # Stagnation tracking
    stagnation = 0
    prev_best = best_f
    patience = 12 + dim

    # Covariance memory: slowly adapted, preserves useful structure across restarts
    C_memory = np.eye(dim)
    memory_rate = 0.08

    gen = 0

    while calls < budget:
        n_off = min(lam, budget - calls)
        if n_off < 2:
            break

        # Cholesky decomposition with fallback
        try:
            L = np.linalg.cholesky(C + 1e-20 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = C + 1e-10 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C + 1e-20 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

        # Generate and evaluate offspring
        X = np.zeros((n_off, dim))
        F = np.zeros(n_off)

        for i in range(n_off):
            z = np.random.randn(dim)
            x = (mean + sigma * (L @ z)).clip(lower, upper)
            X[i] = x
            F[i] = func(x)
            calls += 1
            history.append(min(history[-1], F[i]))
            if F[i] < best_f:
                best_f = F[i]
                best_x = x.copy()

        if calls >= budget:
            break

        # Sort offspring by fitness
        idx = np.argsort(F)
        n_parents = min(mu, n_off)
        w_used = w[:n_parents].copy()
        w_used /= w_used.sum()

        # Weighted mean of standardized differences and outer product
        y_w = np.zeros(dim)
        y_w_outer = np.zeros((dim, dim))
        for i in range(n_parents):
            y_i = (X[idx[i]] - mean) / max(sigma, 1e-30)
            y_w += w_used[i] * y_i
            y_w_outer += w_used[i] * np.outer(y_i, y_i)

        # Sigma evolution path update
        y_w_norm = np.linalg.norm(y_w)
        if y_w_norm > 1e-30:
            decay = math.sqrt(max(1.0 - (1.0 - c_sigma) ** (2 * (gen + 1)), 1e-15))
            threshold = (1.4 + 2.0 / (dim + 1)) * chi_N
            h_sig = 1.0 if y_w_norm / decay < threshold else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sig * math.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff) * (y_w / y_w_norm)
        else:
            h_sig = 0.0
            p_sigma = (1 - c_sigma) * p_sigma

        # C^{-1/2} via eigendecomposition for the C evolution path
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals_clipped = np.maximum(eigvals, 1e-20)
        C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals_clipped))) @ eigvecs.T
        y_c = C_inv_sqrt @ y_w

        # C evolution path update
        p_c = (1 - c_c) * p_c + h_sig * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

        # Covariance matrix update: rank-mu + rank-1
        C = (1.0 - c_1 - c_mu + c_1 * c_sigma * (2.0 - c_sigma)) * C \
            + c_1 * (np.outer(p_sigma, p_sigma) - (1.0 - c_sigma) * C) \
            + c_mu * y_w_outer
        C = (C + C.T) / 2.0

        # Ensure positive definiteness
        min_eig = np.linalg.eigvalsh(C)[0]
        if min_eig < 1e-18:
            C += (1e-15 - min_eig) * np.eye(dim)

        # Step size update
        sigma *= math.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_N - 1.0))
        sigma = max(min(sigma, 10.0 * min_span), 1e-30)

        # Mean update (correct: includes sigma factor)
        mean = (mean + sigma * y_w).clip(lower, upper)

        # Stagnation detection
        if best_f < prev_best - 1e-10 * max(1.0, abs(prev_best)):
            stagnation = 0
        else:
            stagnation += 1
        prev_best = best_f

        # Update covariance memory (only when making progress)
        if stagnation == 0:
            C_memory = (1.0 - memory_rate) * C_memory + memory_rate * C

        # Stagnation handling: memory-informed restart
        if stagnation >= patience:
            mean = best_x.copy()
            # Blend memory with isotropic component for fresh exploration
            trace_mem = np.trace(C_memory)
            if trace_mem > 1e-20:
                # 60% memory structure, 40% isotropic for exploration
                C = 0.6 * C_memory + 0.4 * (trace_mem / dim) * np.eye(dim)
            else:
                C = np.eye(dim)
            sigma = 0.3 * min_span
            p_c = np.zeros(dim)
            p_sigma = np.zeros(dim)
            stagnation = 0
            C_memory = C.copy()

        gen += 1

    return best_x, best_f, history
