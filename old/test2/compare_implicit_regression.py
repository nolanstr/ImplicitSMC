import os
import sys
import h5py
import scipy
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bingo.evolutionary_optimizers.parallel_archipelago \
    import load_parallel_archipelago_from_file
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.bayes_fitness_function import BayesFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.bayes_fitness.zero_check_bff \
        import BayesFitnessFunction
from bingo.evaluation.fitness_function import VectorBasedFunction

PARTICLES=200
MCMC_STEPS=50
hof_index = 22
num_reps = 50
data = np.load("noisycircledata.npy")


implicit_data = ImplicitTrainingData(data)
MLEfitness = MLERegression(implicit_data)
MLEoptimizer = ScipyOptimizer(MLEfitness, method='BFGS', 
                param_init_bounds=[-10.,10.], options={'maxiter':1000})
MLEclo = LocalOptFitnessFunction(MLEfitness, MLEoptimizer)

SLfitness = ImplicitRegression(implicit_data)
SLoptimizer = ScipyOptimizer(SLfitness, method='lm', param_init_bounds=[-10.,10.], options={'maxiter':1000})
SLclo = LocalOptFitnessFunction(SLfitness, SLoptimizer)

N = 50
MLE_params = np.empty((N,2))
MLE_fits = np.empty(N)
SL_params = np.empty((N,2))
SL_fits = np.empty(N)

for j in range(0,50):
    eqn_act = AGraph(equation="(X_0 - 1.0) ** 2 + (X_1 - 1.0) ** 2")
    MLE_fits[j] = MLEclo(eqn_act)
    MLE_params[j] = eqn_act.get_local_optimization_params()
    eqn_act = AGraph(equation="(X_0 - 1.0) ** 2 + (X_1 - 1.0) ** 2")
    SL_fits[j] = SLclo(eqn_act)
    SL_params[j] = eqn_act.get_local_optimization_params()

import pdb;pdb.set_trace()

