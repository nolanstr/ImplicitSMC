import matplotlib.pyplot as plt
import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, nsimplify
from sympy.plotting.plot import MatplotlibBackend, Plot

def get_sympy_subplots(plot:Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

data = np.load("../noisycircledata.npy")
x, y = symbols('X_0 X_1')

string = "(X_1)(((0.000052)((X_0)(X_0) + (X_1)(X_1)) + (X_0)((X_1)(0.873567 - ((X_0)(X_0) + (X_1)(X_1)))))((0.000052)((X_0)(X_0) + (X_1)(X_1)) + (X_0)((X_1)(0.873567 - ((X_0)(X_0) + (X_1)(X_1))))))" 
model = AGraph(equation=string)
string = model.get_formatted_string(format_="sympy")
model = Eq(simplify(string))
model = Eq(simplify(sympify(string)))
model2 = Eq(nsimplify(sympify(string), tolerance=1e-9))
import pdb;pdb.set_trace()
print(model)
#p0 = plot(show=False)
p1 = plot_implicit(model, (x,-1.5,1.5), (y,-1.5,1.5), show=False, 
                                            label="Approx. Solution")
plt = get_sympy_subplots(p1)
plt.scatter(data[:,0], data[:,1], color='k', label="Noisy Data")
plt.legend()
plt.savefig("approx_solution", dpi=1000)
plt.show()
import pdb;pdb.set_trace()
