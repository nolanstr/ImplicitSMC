import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
for i in range(theta.shape[0]):
	x[i] += np.random.normal(0, 0.1)
for i in range(theta.shape[0]):
	y[i] += np.random.normal(0, 0.1)

data_x = np.zeros((100, 2))
data_x[:,0] = x 
data_x[:,1] = y
#data_x[:,2] = np.ones(100)
plt.scatter(data_x[:,0], data_x[:,1], label="Noisy Data")
plt.savefig("data", dpi=1000)
np.save('noisycircledata', data_x)
plt.plot(x, y, '*')
plt.show()
