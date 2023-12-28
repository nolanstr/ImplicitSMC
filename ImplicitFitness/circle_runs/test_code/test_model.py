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
        fitness = MLERegression(implicit_data, order="first", _f_tol=1)
        optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.])#, options={'maxiter':500})
        MLEclo = LocalOptFitnessFunction(fitness, optimizer)
        
        ilbff = ILBFF(implicit_data, MLEclo)
        #string = "C_0 - ((((X_1)(C_1 + C_1))(C_1 + C_1 - (X_1)))(C_1 + C_1)) + X_0"
        string = "(X_1 + 2.3023120488111837 - (X_1))((2.3023120488111837 + X_1 - (X_1))(X_1 + 2.3023120488111837 - (X_1) - (X_0)))" 

        print(f"Noise Level: {noise_level}")
        model = PytorchAGraph(equation=string)
        print(f"SSQE from MLE Regression: {MLEclo(model)}")

        model = PytorchAGraph(equation=string)
        stime = time.time()
        print(f"\n-NMLL from iSMC(Laplace Approximation): {ilbff(model)}")
        print(f"iSMC(Laplace Approximation) Computation Time = {time.time()-stime}\n")
        print(f"Model: {str(model)}")

if __name__ == '__main__':
    test_true_model()
