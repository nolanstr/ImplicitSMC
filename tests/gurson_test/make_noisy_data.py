import numpy as np

data = np.load("gurson_data.npy")
nan_idxs = np.where(np.isnan(data))[0]
data = data[~nan_idxs,:]

noisy_data = data + np.random.normal(loc=0, scale=0.01, size=data.shape)
idxs = np.random.choice(np.arange(noisy_data.shape[0]), 100, replace=False)
noisy_data = noisy_data[idxs, :]
np.save("noisy_gurson_data", noisy_data)
import pdb;pdb.set_trace()
