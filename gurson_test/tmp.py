import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sym
from sympy import cosh

from skimage.measure import marching_cubes

# Define the equation
x, y, z = sym.symbols('x y z')
eq = (y**2) + (2 * z * cosh(1.5*x)) - 1 - (z**2)

# Create a 3D meshgrid
data = np.load("gurson_data.npy")
idxs = np.linspace(0, data.shape[0]-1, 10, dtype=int)
x_vals = data[idxs,0] 
y_vals = data[idxs,1] 
z_vals = data[idxs,2] 
import pdb;pdb.set_trace()
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)


# Evaluate the equation at each point in the meshgrid
F = sym.lambdify((x, y, z), eq, 'numpy')
S = F(X, Y, Z)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_xlim([x_vals.min(), x_vals.max()])
#ax.set_ylim([y_vals.min(), y_vals.max()])
#ax.set_zlim([z_vals.min(), z_vals.max()])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Implicit equation plot')

# Extract the surface where eq = 0
verts, faces, _, _ = marching_cubes(S, 0)

# Scale and shift the vertices to match the meshgrid
verts *= (x_vals[1] - x_vals[0])
verts += np.array([x_vals[0], y_vals[0], z_vals[0]])

# Plot the surface
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis')
plt.show()
