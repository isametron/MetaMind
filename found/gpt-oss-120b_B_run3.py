# Found by arm B, run 3, cerebras/gpt-oss-120b; training fitness 0.4779 (call 25 of 50); name: Adaptive-Mean Evolution Strategy (AMES)
import numpy as np
import random
import math

def optimize(func, dim, bounds, budget):
    """
    Adaptive-Mean Evolution Strategy (AMES)

    Parameters
    ----------
    func : callable
        Function to minimise, takes a (dim,) numpy array and returns a float.
    dim : int
        Dimensionality of the problem.
    bounds : tuple (lower, upper)
        Lower and upper bounds as numpy arrays of shape (dim,).
    budget : int
        Maximum number of function evaluations.

    Returns
    -------
    best_x : numpy.ndarray
        Best solution found.
    best_f : float
        Function value at best_x.
    history : list of float
        Best‑so‑far value after each evaluation.
    """

    lower, upper = bounds
    # ----- initialisation -------------------------------------------------
    # Population size (lambda) and number of parents (mu)
    lam = max(4, 2 * dim)           # ensure at least 4 offspring
    mu = lam // 2                   # number of parents
    # Weights for recombination (logarithmic)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)

    # Strategy parameters
    sigma = 0.3 * np.mean(upper - lower)   # initial step size
    cs = (mu_eff + 2) / (dim + mu_eff + 5)  # cumulation for sigma
    ds = 1 + cs + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1)
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)  # cumulation for C
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)                     # rank‑one update
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1) / ((dim + 2) ** 2 + mu_eff))  # rank‑mu

    # Evolution paths
    pc = np.zeros(dim)
    ps = np.zeros(dim)

    # Covariance matrix (diagonal for speed, but we keep full matrix for adaptivity)
    C = np.eye(dim)

    # Starting point – random uniform within bounds
    mean = lower + (upper - lower) * np.random.rand(dim)

    # History tracking
    history = []
    evals = 0

    # Evaluate initial mean
    best_x = np.copy(mean)
    best_f = func(best_x)
    history.append(best_f)
    evals += 1

    # Main loop
    while evals < budget:
        # ------- sample offspring -----------------------------------------
        # Decompose C = B * D^2 * B^T
        try:
            D, B = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            D = np.maximum(D, 1e-12)
            D = np.sqrt(D)
        except np.linalg.LinAlgError:
            # Fallback to diagonal approximation if decomposition fails
            D = np.sqrt(np.diag(C))
            B = np.eye(dim)

        offspring = []
        fitness = []

        for _ in range(lam):
            z = np.random.randn(dim)                     # standard normal
            y = B @ (D * z)                               # transform
            x = mean + sigma * y
            # Clip to bounds
            x = np.clip(x, lower, upper)
            f = func(x)
            evals += 1
            offspring.append((x, y, z))
            fitness.append(f)

            # Record history
            if f < best_f:
                best_f = f
                best_x = x
            history.append(best_f)

            if evals >= budget:
                break

        if evals >= budget:
            break

        # ------- select mu best offspring ---------------------------------
        idx = np.argsort(fitness)[:mu]
        selected = [offspring[i] for i in idx]

        # recombination
        y_mean = np.sum([weights[i] * sel[1] for i, sel in enumerate(selected)], axis=0)

        # ------ update evolution paths ------------------------------------
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * (y_mean / np.linalg.norm(y_mean))
        # sigma control
        sigma = sigma * math.exp((cs / ds) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

        # pc update (only if step is successful)
        hsig = int(np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (evals / lam))) < (1.4 + 2 / (dim + 1)))
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mu_eff) * y_mean

        # ------ rank‑one and rank‑mu updates -------------------------------
        # Outer products
        rank_one = np.outer(pc, pc)
        rank_mu = np.sum([weights[i] * np.outer(sel[1], sel[1]) for i, sel in enumerate(selected)], axis=0)

        C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

        # ------ mean update ------------------------------------------------
        mean = mean + sigma * y_mean

        # Ensure mean stays inside bounds (helps on constrained problems)
        mean = np.clip(mean, lower, upper)

    return best_x, best_f, history
