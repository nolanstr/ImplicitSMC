import numpy as np
import pickle
import glob
import time

from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                    import ImplicitBayesFitnessFunction as IBFF
from bingo.symbolic_regression.bayes_fitness.implicit_bff_laplace \
                    import ImplicitLaplaceBayesFitnessFunction as ILBFF
from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
                    

PARTICLES = 30
MCMC_STEPS = 7
ESS_THRESHOLD = 0.7

def test_true_model():
    noise_levels= ["0", 
                   "002", 
                   "005", 
                   "007", 
                   "01"]
    #noise_levels = ["01"]
    for i, noise_level in enumerate(noise_levels):


        data = np.load(
                f"../../../data/circle_data/noisycircledata_{noise_level}.npy")[:,:2]
        implicit_data = ImplicitTrainingData(data, np.empty_like(data))
        fitness = MLERegression(implicit_data, iters=1000, _f_tol=np.inf)
        optimizer = ScipyOptimizer(fitness, method='L-BFGS-B', 
                        param_init_bounds=[-0.1,0.1])
        MLEclo = LocalOptFitnessFunction(fitness, optimizer)
        ilbff = ILBFF(implicit_data, MLEclo)

        print(f"Noise Level: {noise_level}")
        string = "(X_0 - 2.3)^2 + (X_1 + 3.4)^2 - 1.0^2"
        ind = PytorchAGraph(equation=string)
        error, dJ_dc = fitness.get_fitness_vector_and_jacobian(ind)
        print(error)
        print(dJ_dc.mean(axis=0))
        print()

if __name__ == '__main__':
    test_true_model()

