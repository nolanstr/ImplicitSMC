import numpy as np

from sympy import plot_implicit, symbols, Eq, solve
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, nsimplify
from sympy.plotting.plot import MatplotlibBackend, Plot

from scipy.stats import norm

def get_sympy_subplots(plot:Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

e = 1.2
a = 1
b = a*e

x, y = symbols('x y')

eq = Eq((x**2 + y**2)**2 - 2*a**2*(x**2-y**2) - b**4 + a**4)
pli = plot_implicit(eq, (x,-1.6,1.6), (y,-1,1), show=False)
series = pli[0]
data, action = series.get_points()
data = np.array([(x_int.mid, y_int.mid) for x_int, y_int in data])
noise = norm(loc=0, scale=0.05)
added_noise = noise.rvs(data.shape)
data += added_noise

#idxs = np.random.choice(np.arange(data.shape[0]), 20, replace=False)
idxs = np.linspace(0, data.shape[0]-1, 20, dtype=int)
data = data[idxs, :]
np.save("noisy_data", data)

plt = get_sympy_subplots(pli)
plt.scatter(data[:,0], data[:,1], color='k', label="Noisy Data")
plt.legend()
plt.savefig("true_model_and_data", dpi=1000)
plt.show()
