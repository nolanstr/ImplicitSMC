import numpy as np
import matplotlib.pyplot as plt
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
FILES = ["open_data.npy",
         "seperated_data.npy"]

def run_SMC(model):
    
    num_particles = 200
    mcmc_steps = 50
    ess_threshold = 0.75
    data = np.vstack([np.load(FILE) for FILE in FILES])

    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    step_list, fit = ibff(model, return_nmll_only=True)
    print(f"-NMLL = {fit}")
    print(str(model))
    import pdb;pdb.set_trace()
    
if __name__ == "__main__":
    
    string = \
    "(X_0**2 + X_1**2)**2 - ((2*C_0 ** 2) * ((X_0**2) - (X_1**2))) + (C_0**4) - (C_1**4)"
    x, y = symbols('X_0 X_1')
    shape = AGraph(equation=string)
    data = np.vstack([np.load(FILE) for FILE in FILES])
    str(shape)
    run_SMC(shape)
    string = shape.get_formatted_string(format_="sympy")
    eq = Eq(simplify(string))


    pli = plot_implicit(eq, (x,-1.6,1.6), (y,-1,1), show=False)
    plt = get_sympy_subplots(pli)
    plt.scatter(data[:,0], data[:,1], color='k', label="Noisy Data")
    plt.legend()
    plt.savefig("approx_solution", dpi=1000)
    plt.show()
    
