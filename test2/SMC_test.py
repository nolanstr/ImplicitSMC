import os
import sys
import h5py
import scipy
from scipy.stats import uniform
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.bayes_fitness.zero_check_bff \
        import BayesFitnessFunction
from bingo.evaluation.fitness_function import VectorBasedFunction

from smcpy.log_likelihoods import BaseLogLike
from smcpy import AdaptiveSampler
from smcpy import MultiSourceNormal
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from mpi4py import MPI

num_particles = 100
mcmc_steps = 10
ess_threshold = 0.75

class ImplicitLikelihood(BaseLogLike):

    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args

    def __call__(self, inputs):
        inputs = inputs[:,:-1]
        return self.estimate_likelihood(inputs)

    def estimate_likelihood(self, inputs):
        
        Js = np.empty(inputs.shape[0])
        vals = self.model(inputs)
        for i, val in enumerate(vals):
            f, df_dx = val
            J = np.sum(f**2/np.linalg.norm(df_dx, axis=1, ord=2))
            Js[i] = J
        return np.log(Js) 

def eval_model(ind, data, params):
    vals = []

    for param in params:

        ind.set_local_optimization_params(param.T)
        f, df_dx = ind.evaluate_equation_with_x_gradient_at(
                                                x=data.x)
        vals.append([f, df_dx])

    return vals

def run_SMC(ind, training_data):
    
    priors = [uniform(-2,4), uniform(-2,4), uniform(0,100000)]
    param_names = ["P0", "P1", "std_dev"]
    noise = None
    log_like_args = [(training_data.x.shape[0]), noise] 
    log_like_func = ImplicitLikelihood
    vector_mcmc = VectorMCMC(lambda inputs: eval_model(ind, training_data, inputs),
                             np.zeros(training_data.x.shape[0]),
                             priors,
                             log_like_args,
                             log_like_func)

    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
    smc = AdaptiveSampler(mcmc_kernel)
    step_list, marginal_log_likes = \
        smc.sample(num_particles, mcmc_steps,
                   ess_threshold,
                   required_phi=1/np.sqrt(training_data.x.shape[0]))
    print(step_list[-1].compute_mean())
    print(step_list[-1].compute_std_dev())

    import pdb;pdb.set_trace()

PARTICLES=200
MCMC_STEPS=50
hof_index = 22
num_reps = 50
data = np.load("noisycircledata.npy")


implicit_data = ImplicitTrainingData(data)
#fitness = ImplicitRegression(implicit_data)
fitness = MLERegression(implicit_data)
optimizer = ScipyOptimizer(fitness, method='BFGS', 
                param_init_bounds=[-10.,10.], options={'maxiter':1000})
MLEclo = LocalOptFitnessFunction(fitness, optimizer)

eqn_act = AGraph(equation="(X_0 - 1.0) ** 2 + (X_1 - 1.0) ** 2")
MLEclo(eqn_act)
print(str(eqn_act))
run_SMC(eqn_act, implicit_data)

