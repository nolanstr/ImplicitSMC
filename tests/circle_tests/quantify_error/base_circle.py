import h5py
import numpy as np
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData, MLERegression

PARTICLES = 100
MCMC_STEPS = 5
ESS_THRESHOLD = 0.75

def make_random_data(N, r, h=0, k=0):

    theta = np.linspace(0, 2*np.pi, N)
    x = r*np.cos(theta).reshape((-1,1))
    y = r*np.sin(theta).reshape((-1,1))
    data_x = np.hstack((x, y))
    data_x[:,0] += h 
    data_x[:,1] += k

    return data_x

def estimate_ssqe(model, radius, N):
    
    data = make_random_data(N, radius)

    implicit_data = ImplicitTrainingData(data, np.empty_like(data))
    fitness = MLERegression(implicit_data)
    ssqe = fitness.evaluate_fitness_vector(model)

    return ssqe
if __name__ == "__main__":
    
    circle = PytorchAGraph(equation="((X_0 - 0) ** 2) + ((X_1 - 0) ** 2) - 1")
    min_radius, max_radius, h = 1., 3.0, 0.01
    Ns = [5, 10, 20, 50, 100]
    radii = np.arange(min_radius, max_radius+h, h)
    fig, ax = plt.subplots()

    for z, N in enumerate(Ns):
        ax.grid(zorder=0)
        approx_ssqe = np.empty_like(radii)
        true_ssqe = np.square(radii-min_radius) * N
        for i, radius in enumerate(radii):
            approx_ssqe[i] = estimate_ssqe(circle, radius, N)*np.exp(-N)
        error = approx_ssqe - true_ssqe
        ax.plot(radii-min_radius, error, label=f"N = {N}", zorder=1+z)

    ax.set_xlabel("D(+)")
    ax.set_ylabel(r"$SSQE_{approx} - SSQE_{true}$")
    ax.legend()
    plt.show()
    import pdb;pdb.set_trace() 
