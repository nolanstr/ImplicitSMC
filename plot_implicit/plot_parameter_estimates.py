import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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
data = np.load("noisycircledata.npy")

def run_SMC(model):
    
    num_particles = 200
    mcmc_steps = 50
    ess_threshold = 0.75


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
    means = list(step_list[-1].compute_mean().values())
    stds = list(step_list[-1].compute_std_dev().values())
    fig, axs = plt.subplots(nrows=3,ncols=1)
    labels = [r"$P_{1}$", r"$P_{2}$", r"$\sigma$"]
    true_val = [0, 0, 0.1]
    for i, (mean, std) in enumerate(zip(means, stds)):
        x = np.linspace(mean-3*std, mean+3*std, 1000)
        y = norm(loc=mean, scale=std).pdf(x)
        axs[i].fill_between(x, y, alpha=0.3)
        axs[i].axvline(true_val[i], color='k', linestyle='--')
    axs[1].set_ylabel(r"Density")
    axs[2].set_xlabel(r"$\theta$")
    plt.show()
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    
    shape = AGraph(equation= "((X_0 - 1.0) ** 2) + ((X_1 - 1.0) ** 2) - 1")
    str(shape)
    run_SMC(shape)
