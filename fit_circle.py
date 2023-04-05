import glob
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from bingo.local_optimizers.continuous_local_opt \
        import ContinuousLocalOptimization
from bingo.symbolic_regression.explicit_regression \
        import ExplicitRegression, ExplicitTrainingData
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator \
        import ComponentGenerator
from bingo.evaluation.evaluation import Evaluation
from bingo.stats.pareto_front import ParetoFront
from bingo.evolutionary_optimizers.parallel_archipelago import \
                                ParallelArchipelago
from bingo.evolutionary_algorithms.generalized_crowding import \
                                GeneralizedCrowdingEA
from bingo.evolutionary_optimizers.island import Island
from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function import \
                        BayesFitnessFunction
from bingo.symbolic_regression.implicit_regression \
        import ImplicitRegression, ImplicitTrainingData
from bingo.symbolic_regression.agraph.agraph import AGraph

from smcpy import AdaptiveSampler
from smcpy import MultiSourceNormal
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel

def generate_noisy_data(var, center=[0,0]):
    data = np.load('noisycircledata.npy')
    training_data = ImplicitTrainingData(data[:,0:2])#, window_size=7, order=1)
    return training_data

def build_component_generator(training_data, operators):

    for op in operators:
        component_generator.add_operator(int(op))

    return component_generator

def custom_eval(x, model, training_data):
    output = np.empty((training_data.x.shape[0], x.shape[0]))
    for i, x_set in enumerate(x):
        model.set_local_optimization_params(x_set.T)
        f, df = model.evaluate_equation_with_x_gradient_at(training_data.x)
        output[:,i] = df[:,0] / df[:,1]
    return output.T

def fit_model_params(var):
    """BINGO HYPERPARAMS"""
    POPULATION_SIZE=100
    STACK_SIZE=10
    MAX_GENERATIONS=100000
    FITNESS_THRESHOLD=-np.inf
    CHECK_FREQUENCY=int(MAX_GENERATIONS//10)
    MIN_GENERATIONS=int(MAX_GENERATIONS//10)
    CROSSOVER_PROBABILITY=0.4
    MUTATION_PROBABILITY=0.4

    """SMC HYPERPARAMS"""
    PARTICLES=600
    MCMC_STEPS=15
    ESS_THRESHOLD=0.6

    center = [0,0]
    
    training_data = generate_noisy_data(var, center=center)
    training_data.y = np.ones((training_data.x.shape[0],1))
    multisource_num_pts = (training_data.x.shape[0],)

    circle = AGraph()
    circle.command_array = np.array([[ 1,  0,  0],
                                      [ 0,  0,  0],
                                      [ 2,  0,  1],
                                      [-1,  2,  2],
                                      [10,  2,  3],
                                      [ 1,  1,  1],
                                      [ 0,  1,  1],
                                      [ 2,  5,  6],
                                      [10,  7,  3],
                                      [ 2,  4,  8]])
    circle.set_local_optimization_params(center)
    circle._needs_opt = True
    dx = training_data.dx_dt
    finite_dif_dx  = (dx[:,0] / dx[:,1]).flatten()
    fitness = ImplicitRegression(training_data)
    clo = ContinuousLocalOptimization(fitness, algorithm='lm')
    import pdb;pdb.set_trace()
    bff = BayesFitnessFunction(clo)

    norm_phi = 1 / np.sqrt(training_data.x.shape[0])
    param_names, priors = bff._create_priors(circle, multisource_num_pts,
                                                                PARTICLES)
    proposal = bff.generate_proposal_samples(circle, PARTICLES, param_names)
    proposal[0]['p0'] = np.random.normal(loc=0, scale=0.001, size=PARTICLES)
    proposal[0]['p1'] = np.random.normal(loc=0, scale=0.001, size=PARTICLES)
    proposal[0]['std_dev0'] = np.random.normal(loc=0, scale=0.001, size=PARTICLES)

    log_like_args = [multisource_num_pts, tuple([None])]
    log_like_func = MultiSourceNormal
    vector_mcmc = VectorMCMC(lambda x: custom_eval(x, circle, training_data),
                                       finite_dif_dx, 
                                       priors, log_like_args, log_like_func)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
    smc = AdaptiveSampler(mcmc_kernel)

    step_list, marginal_log_likes = smc.sample(PARTICLES, MCMC_STEPS, 
                                               ESS_THRESHOLD, 
                                               proposal=proposal, 
                                               required_phi=norm_phi)
    nmll = -1 * (marginal_log_likes[-1] -
                                 marginal_log_likes[smc.req_phi_index[0]])
    
    
    import pdb;pdb.set_trace()
    plt.scatter(training_data.x[:,0], training_data.x[:,1])

if __name__ == "__main__":
    var = sys.argv[-1]
    fit_model_params(var)
