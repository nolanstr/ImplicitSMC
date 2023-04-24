import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, nsimplify
from sympy.plotting.plot import MatplotlibBackend, Plot

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                                    import ImplicitBayesFitnessFunction as IBFF




def get_sympy_subplots(plot:Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

PARTICLES = 100
MCMC_STEPS = 10
ESS_THRESHOLD = 0.75
data = np.load("../noisycircledata.npy")

def run_SMC(model):
    
    num_particles = 200
    mcmc_steps = 50
    ess_threshold = 0.75


    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    MLEclo(model)
    print(str(model))

def plot_model(model):
    x, y = symbols('X_0 X_1')
    string = model.get_formatted_string(format_="sympy")
    eq = Eq(simplify(string))
    pli = plot_implicit(eq, (x,-1.2,1.2), (y,-1.2,1.2), show=False)
    plt = get_sympy_subplots(pli)
    plt.scatter(data[:,0], data[:,1], color='k', label="Noisy Data")
    plt.legend()
    plt.savefig("MLE_approx_solution", dpi=1000)
    plt.show()

if __name__ == "__main__":
    
    shape = AGraph(equation= "((X_0 - 1.0) ** 2) + ((X_1 - 1.0) ** 2) - 1")
    str(shape)
    run_SMC(shape)
    plot_model(shape)
