from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

def plot_implicit(fn, _X, _Y, _Z):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax = _X.min(), _X.max()
    ymin, ymax = _Y.min(), _Y.max()
    zmin, zmax = _Z.min(), _Z.max()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 8) # resolution of the contour
    B = np.linspace(ymin, ymax, 8) # number of slices
    C = np.linspace(zmin, zmax, 8) # number of slices

    A1,A2 = np.meshgrid(A,B) # grid on which the contour is plotted
    coords = []

    for z in C: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        verts = cset.allsegs[0][0]
        xverts = verts[:,0].reshape((-1,1))
        yverts = verts[:,1].reshape((-1,1))
        zverts = np.ones((verts.shape[0],1))*z
        verts = np.hstack((xverts, yverts, zverts))
        coords.append(verts)
        # [z] defines the only level to plot for this contour for this value of z

    A1,A2 = np.meshgrid(A,C) # grid on which the contour is plotted
    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')
        verts = cset.allsegs[0][0]
        xverts = verts[:,0].reshape((-1,1))
        yverts = np.ones((verts.shape[0],1))*y
        zverts = verts[:,1].reshape((-1,1))
        verts = np.hstack((xverts, yverts, zverts))
        coords.append(verts)

    A1,A2 = np.meshgrid(B,C) # grid on which the contour is plotted
    for x in A: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')
        verts = cset.allsegs[0][0]
        xverts = np.ones((verts.shape[0],1))*x
        yverts = verts[:,0].reshape((-1,1))
        zverts = verts[:,1].reshape((-1,1))
        verts = np.hstack((xverts, yverts, zverts))
        coords.append(verts)

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)
    coords = np.vstack(coords)
    #ax.scatter(coords[:,0], coords[:,1], coords[:,2])
    
    for xy in coords[:,:2]:
        idxs = np.where((coords[:,:2] == xy).all(axis=1))[0]
        coords[idxs[1:],:] = np.nan 
    nan_idxs = np.unique(np.where(~np.isnan(coords))[0])
    coords = coords[nan_idxs,:]
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    X_surf, Y_surf = np.meshgrid(x,y)
    Z_surf = np.empty(X_surf.shape)*np.nan
    for j, x_surfs in enumerate(X_surf):
        print(j)
        for i, y_surfs in enumerate(Y_surf):
            xy_data = np.hstack((x_surfs.reshape((-1,1)),
                                y_surfs.reshape((-1,1))))
            idxs = []
            for k, xy in enumerate(xy_data):
                idxs = np.where((coords[:,:2] == xy).all(axis=1))[0]
                ZVAL = coords[idxs][:,-1]
                if len(ZVAL) == 0:
                    ZVAL=np.nan
                Z_surf[i,k] = ZVAL
                if ZVAL < 0:
                    Z_surf[i,k] = np.nan 
    ax.plot_surface(X_surf, Y_surf, Z_surf)
    import pdb;pdb.set_trace()
    plt.show()

def model(x,y,z):
    #[Sp, Sq, VVf]
    #[sigma_h, sigma_vm, f]
    X_0, X_1, X_2 = x, y, z
    C_0 = 1.5
    return (X_1**2) + (2 * X_2 * np.cosh(C_0*X_0)) - 1 - (X_2**2) 

if __name__ == "__main__":
    X, Y, Z = np.linspace(-6,6,10), np.linspace(0.3,1,10), np.linspace(0,0.1,10)
    plot_implicit(model, X, Y, Z)
