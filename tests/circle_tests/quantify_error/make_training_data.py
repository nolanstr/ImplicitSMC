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
    min_radius, max_radius, h = .8, 1.2, 0.1
    Ns = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
    radii = np.arange(min_radius, max_radius+h, h)
    print(radii)
    data = np.zeros((radii.shape[0]*Ns.shape[0], 3))
    count = 0

    for z, N in enumerate(Ns):
        for i, radius in enumerate(radii):
            approx_ssqe = estimate_ssqe(circle, radius, N)
            true_ssqe = np.square(radius-min_radius) * N
            data[count,:] = [N, approx_ssqe, true_ssqe]
            count += 1
    np.save("N_approx_true_data", data)
    plt.savefig("imputed_data", dpi=300)
    plt.show()
