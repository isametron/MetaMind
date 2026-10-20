from . import ga, pso

BASELINES = {
    "PSO": pso.optimize,
    "GA": ga.optimize,
}
