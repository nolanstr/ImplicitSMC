import numpy as np


def make_random_dataset(r=2.5, n=200):
    
    theta = 2 * np.pi * np.random.uniform(low=0, high=1, size=(n,1))
    psi = np.pi * np.random.uniform(low=0, high=1, size=(n,1))

    x = r * np.sin(theta) * np.cos(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(theta)
    
    dataset = np.hstack((x,y,z))

    return dataset

def add_noise_to_dataset(dataset, scale):

    noise = np.random.normal(loc=0, scale=scale, size=dataset.shape)

    return dataset + noise

if __name__ == "__main__":
    
    clean_data = make_random_dataset()
    np.save("spherical_data_0.npy", clean_data)
    
    tags = ["002", "005", "007", "01"]
    scales = [0.02, 0.05, 0.07, 0.1]

    for scale, tag in zip(scales, tags):
        clean_data = make_random_dataset()
        noisy_data = add_noise_to_dataset(clean_data, scale)
        np.save(f"spherical_data_{tag}.npy", noisy_data)

