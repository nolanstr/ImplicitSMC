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

def generate_noisy_data(var, center=[0,0]):
    data = np.load('noisycircledata.npy')
    training_data = ImplicitTrainingData(data[:,0:2])#, window_size=7, order=1)
    return training_data

def build_component_generator(training_data, operators):

    for op in operators:
        component_generator.add_operator(int(op))

    return component_generator

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
    PARTICLES=200
    MCMC_STEPS=5
    
    center = [0,0]
    training_data = generate_noisy_data(var, center=center)
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
    f, df = circle.evaluate_equation_with_x_gradient_at(training_data.x)
    
    fitness = ImplicitRegression(training_data)

    norm_phi = 1 / np.sqrt(training_data.x.shape[0])
    param_names, priors = bff._create_priors(model, multisource_num_pts,
                                                                particles)
    proposal = bff.generate_proposal_samples(model, particles, param_names)
    log_like_args = [multisource_num_pts, 
                                    tuple([None]*len(multisource_num_pts))]
    log_like_func = MultiSourceNormal
    vector_mcmc = VectorMCMC(lambda x: bff.evaluate_model(x, model),
                                       y_noisy.flatten(), 
                                       priors, log_like_args, log_like_func)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
    smc = AdaptiveSampler(mcmc_kernel)

    step_list, marginal_log_likes = smc.sample(particles, mcmc_steps, 
                                               ess_threshold, 
                                               proposal=proposal, 
                                               required_phi=norm_phi)
    nmll = -1 * (marginal_log_likes[-1] -
                                 marginal_log_likes[smc.req_phi_index[0]])
    
    clo = ContinuousLocalOptimization(fitness, algorithm='lm')
    

    plt.scatter(training_data.x[:,0], training_data.x[:,1])
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    var = sys.argv[-1]
    fit_model_params(var)
