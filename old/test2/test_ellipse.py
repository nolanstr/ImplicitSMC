import os
import sys
import h5py
import scipy
from scipy.stats import uniform, norm
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
from smcpy import ImproperUniform
from mpi4py import MPI

num_particles = 200
mcmc_steps = 50
ess_threshold = 0.75
data = np.load("noisycircledata.npy")


implicit_data = ImplicitTrainingData(data)
fitness = MLERegression(implicit_data)
optimizer = ScipyOptimizer(fitness, method='BFGS', 
                param_init_bounds=[-1.,1.], options={'maxiter':1000})
MLEclo = LocalOptFitnessFunction(fitness, optimizer)

class ImplicitLikelihood(BaseLogLike):

    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args

    def __call__(self, inputs):
        return self.estimate_likelihood(inputs)

    def estimate_likelihood(self, inputs):
        
        std_dev = inputs[:,-1]
        inputs = inputs[:,:-1]

        L2s = np.empty(inputs.shape[0])
        vals = self.model(inputs)

        for i, val in enumerate(vals):
            f, df_dx = val
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                import pdb;pdb.set_trace()
            L2 = np.sum(np.square(f.flatten())/np.square(np.linalg.norm(df_dx, axis=1, ord=2)))
            L2s[i] = L2

        a = -f.shape[0] * np.log(2*np.pi) / 2
        b = -f.shape[0] * np.log(std_dev**2) / 2
        c = (-1/(2 * np.square(std_dev))) * L2s
        
        LL = a + b + c
        if inputs.shape[0]==1:
            import pdb;pdb.set_trace()
        return LL

def eval_model(ind, data, params):
    vals = []

    for param in params:

        ind.set_local_optimization_params(param.T)
        f, df_dx = ind.evaluate_equation_with_x_gradient_at(
                                                x=data.x)
        vals.append([f, df_dx])

    return vals

def run_SMC(ind, training_data, prop_dists):
    
    n = ind.get_number_local_optimization_params()
    priors = n * [ImproperUniform()] + [ImproperUniform(0, None)]
    param_names = [f"P{i}" for i in range(n)] + ["std_dev"]
    params_dict = dict(zip(param_names, [dist.rvs(num_particles) for dist in \
                                                        prop_dists])) 
    proposal = [params_dict, np.ones(num_particles)/num_particles]
    noise = None
    n = training_data.x.shape[0]
    b = max(1, np.sqrt(n))/n
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
                   required_phi=b,
                   proposal=proposal)
    print(step_list[-1].compute_mean())
    print(len(marginal_log_likes))
    #print(step_list[-1].compute_std_dev())
    nmll = -1 * (marginal_log_likes[-1] - 
                 marginal_log_likes[smc.req_phi_index[0]])

    print(f"-NMLL = {nmll}")
    print(f"final estimate = {marginal_log_likes[-1]}")
    
    inputs_1 = step_list[0].params.mean(axis=0).reshape((1,-1))
    inputs_2 = step_list[-1].params.mean(axis=0).reshape((1,-1))
    vector_mcmc.evaluate_log_likelihood(inputs_1)
    vector_mcmc.evaluate_log_likelihood(inputs_2)
    mean_params = np.array([s.params.mean(axis=0) for s in step_list])
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    
    ellipse = AGraph(equation=
        "((X_0 - 1.0) ** 2) / (1.0**2) + ((X_1 - 1.0) ** 2) / (1.0**2) - 1")
    lower = 10
    n = ellipse.get_number_local_optimization_params()
    MLEclo(ellipse)
    print(str(ellipse))
    ellipse.set_local_optimization_params([0.01, 1, 0.01, 1])

    priors = [norm(loc=mu, scale=abs(mu)) for mu in \
                    ellipse.get_local_optimization_params()] + [uniform(0.001,1)]
    import pdb;pdb.set_trace()
    print(str(ellipse))
    run_SMC(ellipse, implicit_data, priors)

