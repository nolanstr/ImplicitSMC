import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
h, k = 1, 1
data = np.load("noisy_gurson_data.npy")

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
    fit, marginal_log_likes, step_list = ibff(model, return_nmll_only=False)
    print(f"-NMLL = {fit}")
    print(str(model))
    means = list(step_list[-1].compute_mean().values())
    stds = list(step_list[-1].compute_std_dev().values())
    labels = [r"$c_{0}$", r"$\sigma$"]
    fig, axs = plt.subplots(nrows=1,ncols=1)
    true_val = [0.01]

    for i, (mean, std) in enumerate(zip(means, stds)):
        x = np.linspace(mean-3*std, mean+3*std, 1000)
        y = norm(loc=mean, scale=std).pdf(x)
        sns.kdeplot(x=step_list[-1].params[:,i],
                    weights=step_list[-1].weights.flatten(),
                    fill=True, alpha=0.5, ax=axs, palette="crest",
                    label=labels[i])
        axs.axvline(true_val[i], color='k', linestyle='--')
        axs.set_ylabel(r"Density")
        axs.legend(loc="upper right")
    axs.set_xlabel(r"$\theta$")
    plt.tight_layout()
    plt.savefig("all_parameters_gurson", dpi=1000)
    plt.show()
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    
    string = "(X_1**2) + (2 * X_2 * cosh(3*X_0/2)) - 1 - (X_2**2)" 
    shape = AGraph(equation=string)
    str(shape)
    run_SMC(shape)
