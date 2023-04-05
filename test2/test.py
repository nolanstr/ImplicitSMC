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
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, ImplicitTrainingData
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.bayes_fitness.zero_check_bff \
        import BayesFitnessFunction

def get_eqn(filename, hof_index=0):
    archipelago = load_parallel_archipelago_from_file(filename)
    hof = archipelago.hall_of_fame
    equation = hof[hof_index]
    data = archipelago.island._ea.evaluation.fitness_function.training_data
    
    return equation, data.x    

data = np.load("noisycircledata.npy")
implicitdata = ImplicitTrainingData(data)
fitness = ImplicitRegression(implicitdata)
optimizer = ScipyOptimizer(fitness, method='lm', param_init_bounds=[-1.,1.], options={'maxiter':1000})
clo = LocalOptFitnessFunction(fitness, optimizer)
fitcheck_base = ImplicitRegression(implicitdata)

for j in range(0,50):
    eqn_act = AGraph(equation="(X_0 - 1.0) ** 2 + (X_1 - 1.0) ** 2")
    clo(eqn_act)
    print(str(eqn_act))
    fit_base = fitcheck_base.evaluate_fitness_vector(eqn_act)
    mean_const = np.mean(eqn_act.evaluate_equation_at(data))
    y_data = np.ones(86)
    print(np.mean(np.abs(fit_base)))
    import pdb;pdb.set_trace()
mean_const = np.mean(eqn_act.evaluate_equation_at(data))
import pdb;pdb.set_trace()
y_data = np.ones(86)*mean_const
nmll = fbf.__call__(eqn_act)
smc_weight = 1.0
const_thresh = 10
