import h5py
import numpy as np
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                                    import ImplicitBayesFitnessFunction as IBFF


PARTICLES = 100
MCMC_STEPS = 10
ESS_THRESHOLD = 0.75


def run_SMC(model):
    
    num_particles = 200
    mcmc_steps = 50
    ess_threshold = 0.75
    data = np.load("../noisycircledata.npy")


    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    import pdb;pdb.set_trace()
    step_list, fit = ibff(model, return_nmll_only=True)
    print(f"-NMLL = {fit}")
    print(str(model))
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    
    circle = AGraph(equation="((X_0 - 1.0) ** 2) + ((X_1 - 1.0) ** 2) - 1")
    str(circle)
    run_SMC(circle)

