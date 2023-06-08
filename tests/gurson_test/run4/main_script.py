import sys
import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import \
                                            ImplicitTrainingData, MLERegression
from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.selection.bayes_crowding import BayesCrowding
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation

from bingo.symbolic_regression.bayes_fitness.alt_implicit_bff \
                                    import ImplicitBayesFitnessFunction as IBFF

POP_SIZE = 100
STACK_SIZE = 24
MAX_GEN = 5000
FIT_THRESH = -np.inf
CHECK_FREQ = 50
MIN_GEN = 500

PARTICLES = 100
MCMC_STEPS = 10
ESS_THRESHOLD = 0.75

def run_implicit_gpsruq():

    data = np.load("../noisy_gurson_data.npy")
    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-10.,10.])
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=5)

    component_generator = ComponentGenerator(implicit_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")
    component_generator.add_operator("pow")
    component_generator.add_operator("cosh")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True)


    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    evaluator = Evaluation(ibff, redundant=False, multiprocess=12)

    selection_phase=BayesCrowding()
    ea = GeneralizedCrowdingEA(evaluator, crossover,
                      mutation, 0.4, 0.4, selection_phase)


    island = Island(ea, agraph_generator, POP_SIZE, hall_of_fame=pareto_front)
    opt_result = island.evolve_until_convergence(max_generations=MAX_GEN,
                                                  fitness_threshold=FIT_THRESH,
                                        convergence_check_frequency=CHECK_FREQ,
                                              checkpoint_base_name='checkpoint')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == '__main__':
    run_implicit_gpsruq()
