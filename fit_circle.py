import glob
import numpy as np
import os
import sys

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

def generate_noisy_data(var):
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

    training_data = generate_noisy_data(var)

    fitness = ImplicitRegression(training_data)

    clo = ContinuousLocalOptimization(fitness, algorithm='lm')
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
    circle.constants.append(-2.3)
    circle.constants.append(3.4)
    
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    var = sys.argv[-1]
    fit_model_params(var)
