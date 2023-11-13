import numpy as np

if __name__ == "__main__":
    n = 1000
    d = 3
    mean = np.zeros(d)
    covariance = np.eye(d)
    samples = np.random.multivariate_normal(mean, covariance, size=n)
    covariance_matrix = (1 / (n - 1)) * samples.T @ samples
    import pdb;pdb.set_trace()
