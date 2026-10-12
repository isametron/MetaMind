# Baseline results

Settings: {'functions': ['sphere', 'rastrigin'], 'dims': [10, 30], 'runs': 15, 'budget_per_dim': 1000}. Error = best f − f* at the end of the budget (shifted instances); success = error ≤ 1e-8.

| Algorithm | Function | Dim | Mean error | Std | Median | Best | Success | AOCC | Runtime (s) | Overhead (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSO | sphere | 10 | 3.743e-13 | 4.60e-13 | 1.887e-13 | 2.910e-14 | 100% | 0.7264 | 0.125 | 0.090 |
| PSO | sphere | 30 | 6.172e-01 | 2.31e+00 | 7.719e-14 | 3.886e-16 | 93% | 0.6510 | 0.335 | 0.244 |
| PSO | rastrigin | 10 | 1.157e+01 | 6.01e+00 | 9.950e+00 | 3.031e+00 | 0% | 0.0802 | 0.114 | 0.067 |
| PSO | rastrigin | 30 | 8.733e+01 | 1.97e+01 | 9.054e+01 | 4.975e+01 | 0% | 0.0067 | 0.494 | 0.296 |
