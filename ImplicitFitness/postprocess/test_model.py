
import numpy as np
import sys
import pickle
import glob

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
from bingo.symbolic_regression import ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                    import ImplicitBayesFitnessFunction as IBFF

POP_SIZE = 400
STACK_SIZE = 40
MAX_GEN = 10000
FIT_THRESH = -np.inf
CHECK_FREQ = 50
MIN_GEN = 500

PARTICLES = 50
MCMC_STEPS = 7
ESS_THRESHOLD = 0.75

def eval_model(model, dataset):

    data = np.load(f"../../data/circle_data/noisycircledata_{dataset}.npy")[:,:2]
    import pdb;pdb.set_trace()
    implicit_data = ImplicitTrainingData(data, np.empty_like(data))

    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, 
            implicit_data, MLEclo, ensemble=5)
    print(ibff(model))
    print(str(model))

if __name__ == '__main__':
    
    data = sys.argv[-1]
    print(data)

    #string = "(X_0 - C_0)^2 + (X_1 - C_1)^2 - C_2" 
    #string = "(3)(X_1) + (X_1)((-5335494318.986037 + (3)(X_1))^(-1))"
    string = "((((1.0)(1.0))/(X_0) - (1.0) + ((1.0)(1.0))/(X_0) - (1.0) + X_1)(X_1 + X_1))/((X_1)/(1.0) - ((((1.0)(1.0))/(X_0) - (1.0) + ((1.0)(1.0))/(X_0) - (1.0) + X_1)(X_1 + X_1)))"
    model = PytorchAGraph(equation=string)
    print(str(model))
    print(f"Model Complexity: {model.get_complexity()}")
    print(f"Model stack size: {model.command_array.shape[0]}")
    eval_model(model, data)
