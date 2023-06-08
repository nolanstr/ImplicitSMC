import sys
import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import \
                                            ImplicitTrainingData, MLERegression

from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                                    import ImplicitBayesFitnessFunction as IBFF

PARTICLES = 100
MCMC_STEPS = 10
ESS_THRESHOLD = 0.75

hof_index = 22
num_reps = 50


if __name__ == "__main__":

    data = np.load("noisycircledata.npy")
    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-10.,10.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    
    strings = ["((X_0)^(-2))(((0.031168 + (X_0)((X_0)(X_0)))^(-1))((0.031168 + (X_0)((X_0)(X_0)) - ((X_0)(X_0 - ((X_0)(X_0)))))^(-1)))"]

    for string in strings:
        model = AGraph(equation=string)
        str(model)
        ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo)
        print(ibff(model))
        print(str(model))
