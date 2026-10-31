# Found by arm B, run 1, fake/qwen-3.8-27b; training fitness 0.3321 (call 1 of 2); name: PSOBaseline
"""Standard global-best Particle Swarm Optimization (baseline).

Inertia-weight PSO with the constriction-equivalent parameters of Clerc &
Kennedy (2002): w = 0.7298, c1 = c2 = 1.49618. Velocities are clamped to a
fraction of the search range; particles that leave the box are put back on the
boundary and that velocity component is zeroed.
"""
import numpy as np


def optimize(func, dim, bounds, budget, *, swarm_size=40, w=0.7298, c1=1.49618, c2=1.49618, vmax_frac=0.2):
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    span = upper - lower
    vmax = vmax_frac * span
    n = min(swarm_size, budget)

    x = lower + np.random.rand(n, dim) * span
    v = np.random.uniform(-vmax, vmax, (n, dim))
    f = np.empty(n)
    history = []
    gbest_f, gbest_x = np.inf, None
    for i in range(n):
        f[i] = func(x[i])
        if f[i] < gbest_f:
            gbest_f, gbest_x = f[i], x[i].copy()
        history.append(gbest_f)
    pbest_x, pbest_f = x.copy(), f.copy()
    evals = n

    while evals < budget:
        r1, r2 = np.random.rand(n, dim), np.random.rand(n, dim)
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -vmax, vmax)
        x = x + v
        outside = (x < lower) | (x > upper)
        x = np.clip(x, lower, upper)
        v[outside] = 0.0

        for i in range(n):
            if evals >= budget:
                break
            fi = func(x[i])
            evals += 1
            if fi < pbest_f[i]:
                pbest_f[i], pbest_x[i] = fi, x[i]
                if fi < gbest_f:
                    gbest_f, gbest_x = fi, x[i].copy()
            history.append(gbest_f)

    return gbest_x, gbest_f, history
