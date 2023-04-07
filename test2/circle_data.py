import numpy as np
import matplotlib.pyplot as plt

N = 20
std = 0.05

theta = np.linspace(0, 2*np.pi, N)
x = np.cos(theta) + np.random.normal(0, std, size=theta.shape)
y = np.sin(theta) + np.random.normal(0, std, size=theta.shape)

data_x = np.zeros((N, 2))
data_x[:,0] = x 
data_x[:,1] = y
#data_x[:,2] = np.ones(100)
plt.scatter(data_x[:,0], data_x[:,1], label="Noisy Data")
plt.savefig("data", dpi=1000)
np.save('noisycircledata', data_x)
plt.plot(x, y, '*')
plt.show()
