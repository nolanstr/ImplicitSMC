import matplotlib.pyplot as plt
import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, nsimplify
from sympy.plotting.plot import MatplotlibBackend, Plot

from make_ibff import make_IBFF

def get_sympy_subplots(plot:Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

def plot_implicit_params(string, p1=False):

    model = Eq(simplify(string))
    model = Eq(simplify(sympify(string)))

    if p1:
        p1.append(plot_implicit(model, (x,-1.5,1.5), (y,-1.5,1.5), show=False, 
                                                    label="Approx. Solution")[0])
    else:
        p1 = plot_implicit(model, (x,-1.5,1.5), (y,-1.5,1.5), show=False, 
                                                    label="Approx. Solution")
    return p1

data = np.load("../noisycircledata.npy")
x, y = symbols('X_0 X_1')

string = "(X_1)(((0.000052)((X_0)(X_0) + (X_1)(X_1)) + (X_0)((X_1)(0.873567 - ((X_0)(X_0) + (X_1)(X_1)))))((0.000052)((X_0)(X_0) + (X_1)(X_1)) + (X_0)((X_1)(0.873567 - ((X_0)(X_0) + (X_1)(X_1))))))" 
model = AGraph(equation=string)
print(str(model))
ibff = make_IBFF(data)
step_list, nmll = ibff(model, return_nmll_only=True)
vals = ibff._eval_model(model, ibff.training_data.x, step_list[-1].params)
print()
print(str(model))
fs = [np.mean(val[0]) for val in vals]
import pdb;pdb.set_trace()

ps = model.get_local_optimization_params()
mus = np.array(list(step_list[-1].compute_mean().values()))[:-1]
stds = np.array(list(step_list[-1].compute_std_dev().values()))
targets = [0, 2*stds[-1], -2*stds]
p1 = False

for i in range(3):
    model.set_local_optimization_params(targets[i]) 
    string = model.get_formatted_string(format_="sympy")
    print(string)
    p1 = plot_implicit_params(string, p1=p1) 

plt = get_sympy_subplots(p1)
plt.scatter(data[:,0], data[:,1], color='k', label="Noisy Data")
import pdb;pdb.set_trace()
plt.legend()
plt.show()
