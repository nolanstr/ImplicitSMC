import numpy as np
import pickle
import glob
import time

from bingo.symbolic_regression import ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                    import ImplicitBayesFitnessFunction as IBFF
from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
                    

PARTICLES = 20
MCMC_STEPS = 5
ESS_THRESHOLD = 0.7

def test_true_model():
    noise_levels= ["0", 
                   "002", 
                   "005", 
                   "007", 
                   "01"]
    noise_levels = ["01"]
    for i, noise_level in enumerate(noise_levels):


        data = np.load(
                f"../../data/circle_data/noisycircledata_{noise_level}.npy")[:,:2]
        print(data.shape)
        model = PytorchAGraph(
                equation="(X_0 - 0.0)^2 + (X_1 - 0.0)^2 - (0.0)^2")
        params = np.random.normal(0, 1, size=(100,3))
        model.set_local_optimization_params(params.T)
        model._simplified_constants = params.T
        _f = model.evaluate_equation_at(data)
        df_dx1 = model.evaluate_equation_with_x_partial_at(data, [0] * 2)
        df_dx2 = model.evaluate_equation_with_x_partial_at(data, [1] * 2)
    
        import pdb;pdb.set_trace()

if __name__ == '__main__':
    test_true_model()

