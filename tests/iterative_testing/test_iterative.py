import h5py
import numpy as np
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData
from bingo.symbolic_regression.implicit_regression import MLERegression

def make_random_data(N, r, h=0, k=0):

    theta = np.linspace(0, 2*np.pi, N)
    x = r*np.cos(theta).reshape((-1,1))
    y = r*np.sin(theta).reshape((-1,1))
    data_x = np.hstack((x, y))
    data_x[:,0] += h 
    data_x[:,1] += k

    return data_x

if __name__ == "__main__":
    
    circle = PytorchAGraph(equation="((X_0 - 0) ** 2) + ((X_1 - 0) ** 2) - 1")
    R = 20
    N = 20

    data = make_random_data(N, R)
    fs = []
    approx_solutions = []
    implicit_data = ImplicitTrainingData(data, np.empty_like(data))

    for i in range(10):
        fitness = MLERegression(implicit_data, iters=i)
        approx_solutions.append(fitness.evaluate_fitness_vector(circle))

    import pdb;pdb.set_trace()
         
    true_ssqe = np.square(1-R) * N
    print(f"True solution: {true_ssqe}")
    print(f"Approx solutions: {approx_solutions}")
    import pdb;pdb.set_trace() 
