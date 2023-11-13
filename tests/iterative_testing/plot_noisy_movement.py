import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, \
                                            ImplicitTrainingData
from implicit_regression import MLERegression

def make_random_data(N, r, std=0.2, h=0, k=0):

    theta = np.linspace(0, 2*np.pi, N)
    x = r*np.cos(theta).reshape((-1,1))
    y = r*np.sin(theta).reshape((-1,1))
    x_noise = np.random.normal(0, std, size=x.shape[0])
    y_noise = np.random.normal(0, std, size=y.shape[0])
    data_x_p = np.hstack((x, y))
    data_x_p[:,0] += x_noise 
    data_x_p[:,1] += y_noise

    data_x_n = np.hstack((x, y))
    data_x_n[:,0] -= x_noise 
    data_x_n[:,1] -= y_noise

    return np.hstack((x,y)), data_x_p, data_x_n

if __name__ == "__main__":
    
    circle = PytorchAGraph(equation="((X_0 - 0) ** 2) + ((X_1 - 0) ** 2) - 1")
    R = 0.5
    N = 6

    raw, data_p, data_n  = make_random_data(N, 1)
    labels=["Positive", "Negative"]
    cm_p = matplotlib.colors.LinearSegmentedColormap.from_list("",
                colors=[plt.cm.tab20(0), plt.cm.tab20(1)])
    cm_n = matplotlib.colors.LinearSegmentedColormap.from_list("",
                colors=[plt.cm.tab20(6), plt.cm.tab20(7)])
    cms = [cm_p, cm_n]
    plt.scatter(raw[:,0], raw[:,1], color="k", label="Noiseless data")

    for d, data in enumerate([data_p, data_n]):
        approx_solutions = []
        cm = cms[d]        
        plt.scatter(data[:,0], data[:,1], color=cm(0), label=labels[d],
                edgecolors="k")
        h = 256/11

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
            plt.scatter(data[:,0], data[:,1], color=cm(int((i+1)*h)))
    plt.legend()
    plt.show() 
    true_ssqe = np.square(1-R) * N
    print(f"True solution: {true_ssqe}")
    print(f"Approx solutions: {approx_solutions}")
    import pdb;pdb.set_trace() 
