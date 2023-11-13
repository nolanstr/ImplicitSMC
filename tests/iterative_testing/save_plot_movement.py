import h5py
import numpy as np
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData
from implicit_regression import MLERegression

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
    N = 10

    data = make_random_data(N, R)
    print(data)
    approx_solutions = []

    plt.plot(data[:,0], data[:,1], color=plt.cm.tab20(0), label=f"iter 0")
    for i in range(10):
        implicit_data = ImplicitTrainingData(data, np.empty_like(data))
        fitness = MLERegression(implicit_data)

        x_pos, x_neg = fitness.estimate_dx(circle, data)
        ssqe_pos = np.square(np.linalg.norm(x_pos, axis=0)).sum(axis=0)
        ssqe_neg = np.square(np.linalg.norm(x_neg, axis=0)).sum(axis=0)
        ssqe_pos[np.isnan(ssqe_pos)] = np.inf
        ssqe_neg[np.isnan(ssqe_neg)] = np.inf
        
        #x_pos = np.sqrt(np.square(np.squeeze(x_pos).T)).astype(np.float64)
        #x_neg = np.sqrt(np.square(np.squeeze(x_neg).T)).astype(np.float64)
        x_pos = np.squeeze(x_pos).T.astype(np.float64)
        x_neg = np.squeeze(x_neg).T.astype(np.float64)

        if i == 0:
            dx = np.zeros_like(x_pos)
        if ssqe_pos[0]<ssqe_neg[0]:
            dx += x_pos
            data += x_pos
        else:
            dx += x_neg
            data += x_neg
        approx_solutions.append(
                np.square(np.linalg.norm(dx, axis=1)).sum())
        plt.plot(data[:,0], data[:,1], color=plt.cm.tab20(i+1), label=f"iter {i}")
    plt.show() 
    true_ssqe = np.square(1-R) * N
    print(f"True solution: {true_ssqe}")
    print(f"Approx solutions: {approx_solutions}")
    import pdb;pdb.set_trace() 
