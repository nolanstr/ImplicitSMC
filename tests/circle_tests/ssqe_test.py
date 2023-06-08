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

np.random.seed(3)
PARTICLES = 10
MCMC_STEPS = 5
ESS_THRESHOLD = 0.75

def make_data(N, h=0, k=0, r=1):

    theta = np.linspace(0, 2*np.pi, N)
    x = r*np.cos(theta) #+ np.random.normal(0, std, size=theta.shape)
    y = r*np.sin(theta) #+ np.random.normal(0, std, size=theta.shape)

    data_x = np.zeros((N, 2))
    data_x[:,0] = x + h 
    data_x[:,1] = y + k
    print(data_x)
    return data_x

def run_SMC(model):
    
    num_particles = 10
    mcmc_steps = 50
    ess_threshold = 0.75
    data = make_data(10, r=1.2)

    implicit_data = ImplicitTrainingData(data, np.empty_like(data))
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    fit, marginal_log_likes, step_list = ibff(model, return_nmll_only=False)
    print(f"-NMLL = {fit}")
    print(str(model))
    print(step_list[-1].compute_mean())
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    
    circle = PytorchAGraph(equation="((X_0 - 0) ** 2) + ((X_1 - 0) ** 2) - 1")
    str(circle)
    run_SMC(circle)

