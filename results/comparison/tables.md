# Comparison results

Settings: {'functions': ['sphere', 'rastrigin'], 'dims': [10, 30], 'n_runs': 30, 'budget_per_dim': 1000, 'reference': 'PSO'}. Shifted problem instances; every algorithm gets 1000 × dim evaluations per run.

## Final error (mean ± std)

Best mean in **bold**. + / ≈ / −: significantly better than, no different from, or worse than **PSO** (Mann-Whitney U, α = 0.05).

| Problem | PSO | GA |
|---|---|---|
| sphere 10-D | **5.17e-13 ± 5.4e-13** | 7.36e-04 ± 2.5e-04 − |
| sphere 30-D | 4.36e-01 ± 1.7e+00 | **2.79e-03 ± 8.1e-04** − |
| rastrigin 10-D | 9.87e+00 ± 5.2e+00 | **2.66e+00 ± 1.3e+00** + |
| rastrigin 30-D | 8.60e+01 ± 2.2e+01 | **1.18e+01 ± 2.7e+00** + |

Win / tie / loss against PSO: GA 2/0/2

## Success rate and AOCC

Success = final error ≤ 1e-8. AOCC = area over the convergence curve, 0 to 1, higher is better (rewards both speed and final quality).

| Problem | PSO | GA |
|---|---|---|
| sphere 10-D | 100% / **0.724** | 0% / 0.421 |
| sphere 30-D | 87% / **0.624** | 0% / 0.367 |
| rastrigin 10-D | 0% / 0.083 | 0% / **0.111** |
| rastrigin 30-D | 0% / 0.008 | 0% / **0.052** |

## Compute cost per run

Wall time in seconds, and in brackets the part spent on the algorithm's own work (excluding benchmark evaluations). Every algorithm gets the same evaluation budget.

| Problem | PSO | GA |
|---|---|---|
| sphere 10-D | 0.137 (0.098) | 0.217 (0.164) |
| sphere 30-D | 0.501 (0.365) | 0.712 (0.548) |
| rastrigin 10-D | 0.222 (0.132) | 0.272 (0.173) |
| rastrigin 30-D | 0.746 (0.449) | 0.860 (0.562) |
