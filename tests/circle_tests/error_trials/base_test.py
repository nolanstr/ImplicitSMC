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
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                                    import ImplicitLikelihood as ILike

np.random.seed(3)
PARTICLES = 10
MCMC_STEPS = 5
ESS_THRESHOLD = 0.75

def make_data(N, std=0.01, h=0, k=0, r=1):

    theta = np.linspace(0, 2*np.pi, N)
    noise = np.random.normal(0, std, size=(theta.shape[0], 2))
    x = r*np.cos(theta) + noise[:,0]
    y = r*np.sin(theta) + noise[:,1] 

    data_x = np.zeros((N, 2))
    data_x[:,0] = x + h 
    data_x[:,1] = y + k
    print(data_x)
    return data_x, noise

def find_dx(model, implicit_data, inputs, likelihood, iters=10):
    data = np.expand_dims(np.copy(implicit_data.x), axis=2)
    data = np.repeat(data, inputs.shape[0], axis=2)
    dx = np.zeros_like(data)

    for i in range(0, iters):
        x_pos, x_neg = likelihood.estimate_dx(data, inputs)
        ssqe_pos = np.square(np.linalg.norm(x_pos, axis=0)).sum(axis=0)
        ssqe_neg = np.square(np.linalg.norm(x_neg, axis=0)).sum(axis=0)
        ssqe_pos[np.isnan(ssqe_pos)] = np.inf
        ssqe_neg[np.isnan(ssqe_neg)] = np.inf
        x_pos = np.swapaxes(x_pos, 0, 1)
        x_neg = np.swapaxes(x_neg, 0, 1)
        _dx = np.where(x_pos, x_neg, x_pos<=x_neg)
        dx -= _dx
        data += _dx
    dx = np.mean(dx, axis=2)
    return dx, data

def run_SMC(model):
    
    num_particles = 10
    mcmc_steps = 50
    ess_threshold = 0.75
    data, noise = make_data(10)

    implicit_data = ImplicitTrainingData(data, np.empty_like(data))
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    likelihood = ILike(lambda info: ibff._eval_model(model, info[0], info[1]),
                        implicit_data.x, [(implicit_data.x.shape[0]), None, 10])
    inputs = np.zeros((0,10)).T
    dx, data = find_dx(model, implicit_data, inputs, likelihood, iters=10)
    import pdb;pdb.set_trace()
    fit, marginal_log_likes, step_list = ibff(model, return_nmll_only=False)
    print(f"-NMLL = {fit}")
    print(str(model))
    print(step_list[-1].compute_mean())
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    
    circle = PytorchAGraph(equation="((X_0 - 0) ** 2) + ((X_1 - 0) ** 2) - 1")
    str(circle)
    run_SMC(circle)

