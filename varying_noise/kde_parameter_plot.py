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


PARTICLES = 500
MCMC_STEPS = 20
ESS_THRESHOLD = 0.75
h, k = 1, 1
all_data = np.load("noisy_data.npy")
std_vals = [0.0001, 0.001, 0.01, 0.1]

def run_SMC(model, axs, i):
    
    num_particles = 200
    mcmc_steps = 50
    ess_threshold = 0.75
    data = all_data[i]

    implicit_data = ImplicitTrainingData(data)
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    MLEclo(model)
    print(model.constants)
    print(abs(np.mean(np.abs(model.constants)) - 1))
    while abs(np.mean(np.abs(model.constants)) - 1) > 1:
        model._needs_opt = True
        MLEclo(model)
        print('a')
        print(model.constants)
        print(abs(np.mean(np.abs(model.constants)) - 1))
    fit, marginal_log_likes, step_list = ibff(model, return_nmll_only=False)
    print(f"-NMLL = {fit}")
    print(str(model))
    means = list(step_list[-1].compute_mean().values())
    stds = list(step_list[-1].compute_std_dev().values())
    labels = [r"$c_{0}$", r"$c_{1}$", r"$c_{2}$", r"$\sigma$"]
    true_val = [h, k, 1, std_vals[i]]
    axs[i,0].set_title(r"$\sigma$" + f" = {std_vals[i]}")

    for j, (mean, std) in enumerate(zip(means, stds)):
        if j < 3:
            sns.kdeplot(x=step_list[-1].params[:,j],
                        weights=step_list[-1].weights.flatten(),
                        fill=True,alpha=0.5, ax=axs[i,0], palette="crest",
                        label=labels[j])
            axs[i,0].axvline(true_val[j], color='k', linestyle='--')
            axs[i,0].set_ylabel(r"Density")
        else:
            sns.kdeplot(x=abs(step_list[-1].params[:,j]),
                        weights=step_list[-1].weights.flatten(),
                        fill=True,alpha=0.5, ax=axs[i,1], palette="crest",
                        label=labels[j])
            axs[i,1].axvline(true_val[j], color='k', linestyle='--')
            axs[i,1].set_ylabel(r"Density")

        if i == 0:
            axs[i,0].legend(loc="upper right")
            axs[i,1].legend(loc="upper right")


if __name__ == "__main__":
    
    fig, axs = plt.subplots(all_data.shape[0], 2)

    for i in range(4):
        shape = AGraph(equation= "(X_0 - C_0)^2 + (X_1 - C_1)^2 - C_2")
        str(shape)
        run_SMC(shape, axs, i)
    axs[-1,0].set_xlabel(r"$\theta$")
    axs[-1,1].set_xlabel(r"$\sigma$")
    plt.tight_layout()
    plt.savefig("all_parameters_circle", dpi=1000)
    plt.show()
