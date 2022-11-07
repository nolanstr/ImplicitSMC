import os
import sys
import h5py
import scipy
import numpy as np
from sympy import *
from ipywidgets import interact
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Markdown
from bingo.evolutionary_optimizers.parallel_archipelago \
    import load_parallel_archipelago_from_file
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.bayes_fitness_function import BayesFitnessFunction
from bingo.local_optimizers.continuous_local_opt_new import ContinuousLocalOptimization
from bingo.symbolic_regression.bayes_fitness_function_implicit import BayesFitnessFunctionImplicit
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, ImplicitTrainingData

np.set_printoptions(threshold=sys.maxsize)

def get_eqn(filename, hof_index=0):
    archipelago = load_parallel_archipelago_from_file(filename)
    hof = archipelago.hall_of_fame
    equation = hof[hof_index]
    data = archipelago.island._ea.evaluation.fitness_function.training_data
    
    return equation, data.x    

PARTICLES=200
MCMC_STEPS=50
hof_index = 0
eqn, data = get_eqn('smc_multiobject_circle_smoothed1/checkpoint_3450.pkl', hof_index)
##print(eqn)
data = np.load('noisycircledata_01.npy')
##print(data.shape)
equ_eval = eqn.evaluate_equation_at(data[:,0:2])
mean_const = np.mean(equ_eval)
y_data = np.ones(93)*mean_const
#trainingdata = ImplicitTrainingData(data[:,0:2])
#fitness = ImplicitRegression(trainingdata)
trainingdata = ExplicitTrainingData(data[:,0:2],data[:,2])
fitness = ExplicitRegression(trainingdata)
clo = ContinuousLocalOptimization(fitness, algorithm='lm')
#clo.optimization_options = {'options':{'xtol':1e-16, 'ftol':1e-16,
#                                                          'eps':0., 'gtol':1e-16,
#                                                          'maxiter':15000}}
#clo.param_init_bounds = [-1.,1.]
for i in range(1,50):
#     print('Number of Multistarts: ', i)
#     for j in range(5):
    fbf = BayesFitnessFunctionImplicit(clo, y_data, num_particles=PARTICLES,
    mcmc_steps=MCMC_STEPS, return_nmll_only=False, num_multistarts=2)
    nmll, step_list, vector_mcmc = fbf.__call__(eqn)
    print('nmll:',nmll)
    #eqn._needs_opt = True
    #clo(eqn)
    #print(eqn)
