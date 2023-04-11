import numpy as np
import time
import copy
import pickle
from multiprocessing import Pool

from bingo.symbolic_regression import ExplicitTrainingData,\
                                      ExplicitRegression
from bingo.local_optimizers.continuous_local_opt import \
                                      ContinuousLocalOptimization
from bingo.evolutionary_optimizers.get_all_arch import *

import sys;sys.path.append('../');sys.path.append('../../')
sys.path.append("../SMCBingoApplicationPaper/")
from util.population_functions.observe_models import *
from util.vtb_models.sr1_agraph import make_SR1_instance
from util.model_selection.model_selection import *
from util.plot_functions.cred_pred import *
from util.plot_functions.pdf_cred_pred_plots import *
from util.data_functions.get_data import *
from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function import \
                                      BayesFitnessFunction

def make_training_data(n=False):
    
    data, num_points = organize_data([1,2,3])
    if n is False:
        n = min(num_points)
        
    cols = np.append([0], np.arange(2, data.shape[1]))
    x, y = data[:,cols], data[:,1]
    x_new, y_new = [], []

    x_idxs = np.cumsum(np.array([0]+list(num_points)))
    
    for i in range(len(x_idxs)-1):

        x_subset = x[x_idxs[i]:x_idxs[i+1],:]
        y_subset = y[x_idxs[i]:x_idxs[i+1]]
        linear = np.linspace(x_subset[:,0].min(), x_subset[:,0].max(), min(n,
                                        num_points[i]))
        
        local_idxs = np.array([np.argmin(abs(x_subset[:,0]-linear_pt)) for \
                                                        linear_pt in linear])
        x_new.append(x_subset[local_idxs,:])
        y_new.append(y_subset[local_idxs])
    
    x = np.vstack(x_new)
    y = np.hstack(y_new).reshape((-1,1))
    num_points = np.minimum(np.array(num_points), np.array([n]*len(num_points))) 
    training_data = ExplicitTrainingData(x, y)

    return training_data, num_points
def create_bff(n=100, random_sample=25):

    training_data, num_points = make_training_data(n)
    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='SLSQP')

    smc_hyperparams = {'num_particles':5000,
                       'mcmc_steps':10,
                       'ess_threshold':0.75}
    multisource_info = num_points
    random_sample_info = random_sample

    bff = BayesFitnessFunction(local_opt_fitness,
                                   smc_hyperparams=smc_hyperparams,
                                   multisource_info=multisource_info,
                                   random_sample_info=random_sample_info)
    return bff, local_opt_fitness, training_data

def get_cred_pred_intervals_on_subsets(model, bff):

    inputs = []
    models = []
    print(f'run -nmll = {model.fitness}')
    for i in range(10):
        try:
            nmll, step_list, _ = bff(model, return_nmll_only=False)
            break
        except:
            pass
    print(f'-nmll = {nmll}')
    print(f'model = {str(model)}')
    inputs = [bff.estimate_cred_pred(model, step_list, subset=i, \
              linspace=1000) for i in range(len(bff._multisource_num_pts))]
    
    return inputs, bff

def reeval_models(population, bff, nprocs=False):
    
    if nprocs:
        with Pool(nprocs) as p:
            fits = p.map(bff, population)
    else:
        fits = [bff(ind) for ind in population]
    return fits


if __name__ == "__main__":
    DIR = sys.argv[-1]
    if '' in DIR.split('/'):
        fig_name = DIR.split('/')[-2]
    else:
        fig_name = DIR.split('/')[-1]

    SR1 = make_SR1_instance(set_params=True)
    oldest_pickle = get_oldest_pickle(get_all_pickles(DIR))
    try:
        island = oldest_pickle.island
    except:
        island = oldest_pickle
    import pdb;pdb.set_trace()

    for ind in island.hall_of_fame:

        print(str(ind))
        print(f"-NMLL: {ind.fitness}")
