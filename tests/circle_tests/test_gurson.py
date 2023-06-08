import numpy as np
import matplotlib.pyplot as plt
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, nsimplify
from sympy.plotting.plot import MatplotlibBackend, Plot

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.alter_implicit_bff \
                                    import ImplicitBayesFitnessFunction as IBFF

def get_sympy_subplots(plot:Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

PARTICLES = 20
MCMC_STEPS = 5
ESS_THRESHOLD = 0.75
data = np.load("../gurson_test/noisy_gurson_data.npy")

def run_SMC(model):
    

    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    fit, marginal_log_likes, step_list = ibff(model, return_nmll_only=False)
    print(f"-NMLL = {fit}")
    print(str(model))
    import pdb;pdb.set_trace()
    
if __name__ == "__main__":
    #[Sp, Sq, VVf]
    #[sigma_h, sigma_vm, f]
    string = "(X_1**2) + (2 * X_2 * cosh(C_0*X_0)) - 1 - (X_2**2)" 
    shape = PytorchAGraph(equation=string)
    str(shape)
    print(shape.get_complexity())
    run_SMC(shape)
