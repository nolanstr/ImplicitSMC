import h5py
import numpy as np
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                                    import ImplicitBayesFitnessFunction as IBFF


PARTICLES = 100
MCMC_STEPS = 10
ESS_THRESHOLD = 0.75

def make_random_data(N, std, h=0, k=0):

    theta = np.linspace(0, 2*np.pi, N)
    x = np.cos(theta) + np.random.normal(0, std, size=theta.shape)
    y = np.sin(theta) + np.random.normal(0, std, size=theta.shape)

    data_x = np.zeros((N, 2))
    data_x[:,0] = x + h 
    data_x[:,1] = y + k

    return data_x

def run_SMC(model):
    
    num_particles = 200
    mcmc_steps = 50
    ess_threshold = 0.75
    data = make_random_data(50, 0.01)
    import pdb;pdb.set_trace()


    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    fit, marginal_log_likes, step_list = ibff(model, return_nmll_only=False)
    print(f"-NMLL = {fit}")
    print(str(model))
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    
    circle = PytorchAGraph(equation="(X_2)((-0.9999999999999998 + X_2)^(-1))")
    import pdb;pdb.set_trace()
    str(circle)
    run_SMC(circle)

