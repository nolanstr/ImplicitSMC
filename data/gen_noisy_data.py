import numpy as np
import matplotlib.pyplot as plt
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, diff

#circle
numpts = 100
theta = np.linspace(0, 2*np.pi, numpts)
x = np.cos(theta) + 2.3
y = np.sin(theta) - 3.4
var = 0.1
if var != 0:
	for i in range(theta.shape[0]):
		x[i] += np.random.normal(0, var)
	for i in range(theta.shape[0]):
		y[i] += np.random.normal(0, var)

data_x = np.zeros((numpts, 2))
data_x[:,0] = x
data_x[:,1] = y
# np.save('noisycircledata_'+str(var).replace('.',''), data_x)
counter = np.linspace(1, data_x.shape[0], data_x.shape[0])
c = plt.scatter(data_x[:,0], data_x[:,1], c=counter)
plt.colorbar(c)
plt.show()

#harmonic oscillator
w = np.sqrt(3)
g = -0.1/(2*w)
A = 1/(w*np.sqrt(1-g**2))
t = np.linspace(0, 25, 1000)

# h = th_dprime - 0.1*th_prime + 3*th
z = A*np.exp(-g*w*t)*np.sin(np.sqrt(1-g**2)*w*t) + np.random.normal(0, var, t.shape)
v = -A*w*np.exp(-g*w*t)*(g*np.sin(np.sqrt(1-g**2)*w*t) - np.sqrt(1-g**2)*np.cos(np.sqrt(1-g**2)*w*t)) + np.random.normal(0, var, t.shape)
a = A*w**2*np.exp(-g*w*t)*((2*g**2-1)*np.sin(np.sqrt(1-g**2)*w*t) - 2*g*np.sqrt(1-g**2)*np.cos(np.sqrt(1-g**2)*w*t)) + np.random.normal(0, var, t.shape)
data_store = np.zeros((1000,3))
data_store[:,0] = z
data_store[:,1] = v
data_store[:,2] = a
# np.save('harmonic_data_005',data_store)

#elliptic curve
numpts = 1000
x_vec = np.linspace(0.9, 2, numpts)
y_vec = np.linspace(-3, 3, numpts)
x, y = np.meshgrid(x_vec, y_vec)
F = x**3 - 2*x - y**2 + 1
tol = 1e-4
idx = np.where(np.logical_and(F.flatten()>0-tol,F.flatten()<0+tol))
x_vals = x.flatten()[idx]
y_vals = y.flatten()[idx]
args = y_vals.argsort()
x_vals = x_vals[args]
y_vals = y_vals[args]
i = 1
data = np.zeros((x_vals.shape[0],2))
data[:,0] = x_vals
data[:,1] = y_vals
np.save('arc', data)
for i in range(100):
	plt.plot(x_vals, y_vals, 'k.')
	plt.plot(x_vals[i], y_vals[i], 'r.')
	plt.show()
topcircle = np.load('topcircle.npy')
bottomcircle = np.load('bottomcircle.npy')
arc = np.load('arc.npy')
data_full = np.zeros((topcircle.shape[0]+bottomcircle.shape[0]+arc.shape[0]+1,2))
data_full[:topcircle.shape[0],:] = topcircle
data_full[topcircle.shape[0]:topcircle.shape[0]+bottomcircle.shape[0],:] = bottomcircle
data_full[topcircle.shape[0]+bottomcircle.shape[0],:] = np.nan
data_full[topcircle.shape[0]+bottomcircle.shape[0]+1:,:] = arc
# np.save('elliptic_data_0', data_full)
for i in range(1000):
	plt.plot(data_full[:,0], data_full[:,1], 'k.')
	plt.plot(data_full[i,0], data_full[i,1], 'r.')
	plt.show()
