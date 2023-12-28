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
        fitness = MLERegression(implicit_data, order="first")
        optimizer = ScipyOptimizer(fitness, method='lm', 
                        param_init_bounds=[-1.,1.], options={'maxiter':2000})
        MLEclo = LocalOptFitnessFunction(fitness, optimizer)
        
        ilbff = ILBFF(implicit_data, MLEclo)

        print(f"Noise Level: {noise_level}")
        string1 = "(X_0 - 2.3)^2 + (X_1 + 3.4)^2 - 1.0^2"
        string2 = "(X_0 - 2.3)^2 + (X_1 + 3.4)^2 - 1.0"
        string3 = "(X_0 - 2.3)^2 + (X_1 + 3.4)^2 - 1.0^2 + 0.0"
        model1 = PytorchAGraph(equation=string1)
        model2 = PytorchAGraph(equation=string2)
        model3 = PytorchAGraph(equation=string3)
        
        print(fitness.evaluate_fitness_vector(model1))
        print(fitness.evaluate_fitness_vector(model2))
        print(fitness.evaluate_fitness_vector(model3))

        print(MLEclo(model1))
        print(str(model1))
        print(MLEclo(model2))
        print(str(model2))
        print(MLEclo(model3))
        print(str(model3))
if __name__ == '__main__':
    test_true_model()

