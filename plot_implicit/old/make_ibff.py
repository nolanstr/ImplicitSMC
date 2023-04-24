import numpy as np

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import \
                                            ImplicitTrainingData, MLERegression
from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization

from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                                    import ImplicitBayesFitnessFunction as IBFF

POP_SIZE = 104
STACK_SIZE = 32
MAX_GEN = 1000
FIT_THRESH = -np.inf
CHECK_FREQ = 10
MIN_GEN = 500

PARTICLES = 100
MCMC_STEPS = 10
ESS_THRESHOLD = 0.75

def make_IBFF(data):

    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-10.,10.])
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)

    return ibff

