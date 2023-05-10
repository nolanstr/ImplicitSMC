import numpy as np
import matplotlib.pyplot as plt

stds = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
N = 100
h,k = 1, 1
data = np.empty((len(stds), N, 2))

theta = np.linspace(0, 2*np.pi, N)
x = np.cos(theta).reshape((-1,1)) + h 
y = np.sin(theta).reshape((-1,1)) + k
d = np.hstack((x,y))

fig, axs = plt.subplots(2, 3)

for i in range(2):
    for j in range(3):
        std = stds[2*i + j]
        data[2*i+j] = d.copy() + np.random.normal(loc=0, scale=std, size=(N,2))
        axs[i,j].scatter(data[2*i+j][:,0], data[2*i+j][:,1], color='b')
        axs[i,j].plot(x, y, color='k')

plt.savefig("data", dpi=1000)
np.save('noisy_data', data)
plt.show()
