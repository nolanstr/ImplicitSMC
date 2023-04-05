import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
for i in range(theta.shape[0]):
<<<<<<< HEAD:test2/circle_data.py
	x[i] += np.random.normal(0, 0.05)
for i in range(theta.shape[0]):
	y[i] += np.random.normal(0, 0.05)
=======
	x[i] += np.random.normal(0, 0.1)
for i in range(theta.shape[0]):
	y[i] += np.random.normal(0, 0.1)
>>>>>>> 010b6a811df74945cf4e5e1ebd09ba9528bf9ece:circle_data.py

data_x = np.zeros((100, 3))
data_x[:,0] = x 
data_x[:,1] = y
data_x[:,2] = np.ones(100)
np.save('noisycircledata', data_x)
plt.plot(x, y, '*')
plt.show()
