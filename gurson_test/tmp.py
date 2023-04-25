from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import sympy 
from sympy import plot_implicit, symbols, Eq, And, sympify, simplify, nsimplify
from tqdm import tqdm

from bingo.symbolic_regression.agraph.agraph import AGraph

noisy_data = np.load("noisy_gurson_data.npy")

def plot_implicit(fn, _X, _Y, _Z):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax = _X.min(), _X.max()
    ymin, ymax = _Y.min(), _Y.max()
    zmin, zmax = _Z.min(), _Z.max()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 20) # resolution of the contour
    B = np.linspace(ymin, ymax, 20) # number of slices
    C = np.linspace(zmin, zmax, 20) # number of slices
    
    grid_X, grid_Y = np.meshgrid(_X, _Y)
    grid_Z = np.empty(grid_X.shape)
    shape = grid_X.shape

    grid_X, grid_Y, grid_Z = grid_X.flatten(), grid_Y.flatten(), grid_Z.flatten()
    string = "(-0.095137 + X_2)((X_2)^(-2) + X_2)" 
    model = AGraph(equation=string)
    x,y,z = sympy.symbols("X_0 X_1 X_2")
    string = model.get_formatted_string(format_="sympy")
    exp = Eq(simplify(string))

    for i, (xval, yval) in tqdm(enumerate(zip(grid_X, grid_Y)),total=grid_X.shape[0]):
        zval = sympy.solve(exp.subs({x:xval, y:yval}), z)
        zval = zval[np.argmin(abs(np.array(zval)))]
        if zval > zmax:
            zval=np.nan
            pass
        grid_Z[i] = zval
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)
    grid_X = grid_X.reshape(shape)
    grid_Y = grid_Y.reshape(shape)
    grid_Z = grid_Z.reshape(shape)

    ax.plot_surface(grid_X, grid_Y, grid_Z, 
                edgecolor='royalblue', lw=0.5,
                alpha=0.3) 
    ax.contourf(grid_X, grid_Y, grid_Z, 
            zdir='z', offset=zmin, cmap='coolwarm')
    #ax.contourf(grid_X, grid_Y, grid_Z, 
    #        zdir='x', offset=xmin, cmap='coolwarm')
    #ax.contourf(grid_X, grid_Y, grid_Z, 
    #        zdir='y', offset=ymax, cmap='coolwarm')
    #ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax),
    #   xlabel='X', ylabel='Y', zlabel='Z')
    x,y,z = noisy_data[:,0], noisy_data[:,1], noisy_data[:,2]
    ax.scatter(x, y, z, c='k')
    plt.legend()
    plt.show()

def model(x,y,z):
    #[Sp, Sq, VVf]
    #[sigma_h, sigma_vm, f]
    X_0, X_1, X_2 = x, y, z
    return (X_2**2) ** (np.cosh(np.cosh(X_1)))
if __name__ == "__main__":
    N = 20
    maxs = noisy_data.max(axis=0)
    mins = noisy_data.min(axis=0)
    
    p = lambda x: 10*pow((x-0.5),2) + 0
    d = np.append(0, p(np.linspace(0,1,2*N)))
    d /= d.sum()
    d = np.cumsum(d)

    X = mins[0] + d*abs(maxs[0]-mins[0]) 
    Y = np.linspace(mins[1], maxs[1],N) 
    Z = np.linspace(mins[2], maxs[2],N) 
    plot_implicit(model, X, Y, Z)
