import numpy as np
import pickle
import glob
import os

from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.selection.bayes_crowding import BayesCrowding
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.stats.pareto_front import ParetoFront
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bff_laplace \
                import ImplicitLaplaceBayesFitnessFunction as ILBFF

POP_SIZE = 500
STACK_SIZE = 24
MAX_GEN = 10000
FIT_THRESH = -np.inf
CHECK_FREQ = 50
MIN_GEN = 500

def execute_generational_steps():
    cwd = os.getcwd()
    tag = cwd.split("/")[-1].split("_")[-1]
    data = np.load(
        f"../../../../data/spherical_data/spherical_data_{tag}.npy")
    print(data.shape)
    implicit_data = ImplicitTrainingData(data, np.empty_like(data))

    component_generator = ComponentGenerator(implicit_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("pow")
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_pytorch=True)
    fitness = MLERegression(implicit_data, order="first")
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':100})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = ILBFF(implicit_data, MLEclo)

    evaluator = Evaluation(ibff, redundant=False, multiprocess=20)

    selection_phase=BayesCrowding()
    ea = GeneralizedCrowdingEA(evaluator, crossover,
                      mutation, 0.4, 0.4, selection_phase)
    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)
    island = Island(ea, agraph_generator, POP_SIZE, 
                            hall_of_fame=pareto_front)
    opt_result = island.evolve_until_convergence(max_generations=MAX_GEN,
                                                  fitness_threshold=FIT_THRESH,
                                        convergence_check_frequency=CHECK_FREQ,
                                              checkpoint_base_name='checkpoint')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == '__main__':
    execute_generational_steps()

