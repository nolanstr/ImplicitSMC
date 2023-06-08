import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

def make_data(N, r=1):

    theta = np.linspace(0, 2*np.pi, N)
    x = r*np.cos(theta) 
    y = r*np.sin(theta)

    return x, y

if __name__ == "__main__":
    
    std_devs = [0.5, 0.2, 0.1, 0.01, 0.001]

    fig, ax = plt.subplots()
    cm = plt.cm.tab10
    X, y = make_data(1000)
    ax.plot(X, y, color="k")
    
    for i, std_dev in enumerate(std_devs):
        X, y = make_data(1000)
        X += np.random.normal(0, std_dev, size=X.shape)
        y += np.random.normal(0, std_dev, size=y.shape)
        ax.scatter(X, y, label=f"$\sigma = {std_dev}$", color=cm(i))
    
    ax.set_ylabel("y", fontsize=16)
    ax.set_xlabel("X", fontsize=16)
    ax.legend(fontsize=16)
    plt.savefig("noise_visualization", dpi=1000)
    plt.show()
