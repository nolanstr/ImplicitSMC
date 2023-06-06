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
from bingo.symbolic_regression.bayes_fitness.alter_implicit_bff \
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
    data = make_random_data(1000, TRUE_STD_DEV, h=h, k=k)


    implicit_data = ImplicitTrainingData(data, np.empty_like(data))
    fitness = MLERegression(implicit_data)
    optimizer = ScipyOptimizer(fitness, method='BFGS', 
                    param_init_bounds=[-1.,1.], options={'maxiter':1000})
    MLEclo = LocalOptFitnessFunction(fitness, optimizer)
    ibff = IBFF(PARTICLES, MCMC_STEPS, ESS_THRESHOLD, implicit_data, MLEclo,
                                    ensemble=10)
    fit, marginal_log_likes, step_list = ibff(model, return_nmll_only=False)
    print(f"-NMLL = {fit}")
    print(str(model))
    print(step_list[-1].compute_mean())
    mean = list(step_list[-1].compute_mean().values())[0]
    std = list(step_list[-1].compute_std_dev().values())[0]
    fig, ax = plt.subplots()
    labels = [r"$\sigma$"]
    true_val = [TRUE_STD_DEV]
    x = np.linspace(mean-3*std, mean+3*std, 1000)
    y = norm(loc=mean, scale=std).pdf(x)
    sns.kdeplot(x=abs(step_list[-1].params[:,0]),
                weights=step_list[-1].weights.flatten(),
                fill=True,alpha=0.5, ax=ax, palette="crest",
                label=labels[0])
    ax.axvline(true_val[0], color='k', linestyle='--')
    ax.set_ylabel(r"Density")
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$\theta$")
    plt.tight_layout()
    plt.savefig("base_parameters_circle", dpi=1000)
    plt.show()
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    
    shape = PytorchAGraph(equation= "(X_0 - 1)^2 + (X_1 - 1)^2 - 1")
    str(shape)
    run_SMC(shape)
