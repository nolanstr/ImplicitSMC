import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import sympy
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, nsimplify
from sympy.plotting.plot import MatplotlibBackend, Plot

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression
from bingo.symbolic_regression.bayes_fitness.implicit_bayes_fitness_function \
                                    import ImplicitBayesFitnessFunction as IBFF



def make_random_data(N, std, h=0, k=0):

    theta = np.linspace(0, 2*np.pi, N)
    x = np.cos(theta) + np.random.normal(0, std, size=theta.shape)
    y = np.sin(theta) + np.random.normal(0, std, size=theta.shape)

    data_x = np.zeros((N, 2))
    data_x[:,0] = x + h 
    data_x[:,1] = y + k

    return data_x

PARTICLES = 100
MCMC_STEPS = 10
ESS_THRESHOLD = 0.75
h, k = 1, 1
TRUE_STD_DEV = 0.10


def run_SMC(model):
    
    num_particles = 200
    mcmc_steps = 50
    ess_threshold = 0.75
    data = make_random_data(100, TRUE_STD_DEV, h=h, k=k)


    implicit_data = ImplicitTrainingData(data,  np.empty_like(data))
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
    fig, axs = plt.subplots(nrows=4,ncols=1)
    labels = [r"$c_{0}$", r"$c_{1}$", r"$c_{2}$", r"$\sigma$"]
    true_val = [h, k, 1, 0.1]
    for i, (mean, std) in enumerate(zip(means, stds)):
        x = np.linspace(mean-3*std, mean+3*std, 1000)
        y = norm(loc=mean, scale=std).pdf(x)
        sns.kdeplot(x=abs(step_list[-1].params[:,i]),
                    weights=step_list[-1].weights.flatten(),
                    fill=True,alpha=0.5, ax=axs[i], palette="crest",
                    label=labels[i])
        axs[i].axvline(true_val[i], color='k', linestyle='--')
        axs[i].set_ylabel(r"Density")
        axs[i].legend(loc="upper right")
    axs[3].set_xlabel(r"$\theta$")
    plt.tight_layout()
    plt.savefig("all_parameters_circle", dpi=1000)
    plt.show()
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    
    shape = PytorchAGraph(equation= "(X_0 - C_0)^2 + (X_1 - C_1)^2 - C_2")
    str(shape)
    run_SMC(shape)
