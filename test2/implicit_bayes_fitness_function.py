import os
import sys
import h5py
import scipy
from scipy.stats import uniform, norm, invgamma
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

from smcpy.log_likelihoods import BaseLogLike
from smcpy import AdaptiveSampler
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import ImproperUniform
from mpi4py import MPI


class ImplicitBayesFitnessFunction:
    
    def __init__(self, num_particles, mcmc_steps, ess_threshold, 
                                            training_data, clo):

        self._eval_count = 0
        self._num_particles = num_particles
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._training_data = training_data
        self._clo = clo

        n = training_data.x.shape[0]
        self._b = max(1, np.sqrt(n))/n


    def __call__(self, ind):
        
        n = ind.get_number_local_optimization_params()
        priors = n * [ImproperUniform()] + [ImproperUniform(0, None)]
        param_names = [f"P{i}" for i in range(n)] + ["std_dev"]
        prop_dists = self._estimate_proposal(ind)
        params_dict = dict(zip(param_names, [dist.rvs(self._num_particles) for dist in \
                                                            prop_dists])) 
        proposal = [params_dict, np.ones(self._num_particles)/self._num_particles]
        noise = None
        log_like_args = [(self._training_data.x.shape[0]), noise] 
        log_like_func = ImplicitLikelihood
        vector_mcmc = VectorMCMC(lambda inputs: self._eval_model(ind, 
                                                self._training_data.x, inputs),
                                 np.zeros(self._training_data.x.shape[0]),
                                 priors,
                                 log_like_args,
                                 log_like_func)

        mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
        smc = AdaptiveSampler(mcmc_kernel)
        step_list, marginal_log_likes = \
            smc.sample(self._num_particles, self._mcmc_steps,
                       self._ess_threshold,
                       required_phi=self._b,
                       proposal=proposal)
        nmll = -1 * (marginal_log_likes[-1] - 
                     marginal_log_likes[smc.req_phi_index[0]])
        mean_params = np.average(step_list[-1].params, 
                             weights=step_list[-1].weights.flatten(), axis=0) 
        print(mean_params)
        ind.set_local_optimization_params(mean_params[:-1])

        return nmll
    
    def _estimate_proposal(self, ind):
        
        self._clo(ind)
        params = ind.get_local_optimization_params()
        print(params)
        ssqe = self._clo._fitness_function.evaluate_fitness_vector(ind)
        ns = 0.01
        n = self._training_data.x.shape[0]
        var = ssqe / n 
        prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
                    [invgamma((ns + n)/2, scale=(ns*var + ssqe)/2)]
        
        return prop_dists

    def _eval_model(self, ind, X, params):
        vals = []
        for param in params:
            ind.set_local_optimization_params(param.T)
            f, df_dx = ind.evaluate_equation_with_x_gradient_at(
                                                    x=X)
            vals.append([f, df_dx])

        return vals

    @property
    def eval_count(self):
        return self._eval_count + self._cont_local_opt.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._eval_count = value - self._cont_local_opt.eval_count

    @property
    def training_data(self):
        return self._cont_local_opt.training_data

    @training_data.setter
    def training_data(self, training_data):
        self._cont_local_opt.training_data = training_data

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
        
        return LL


