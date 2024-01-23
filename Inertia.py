import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

# Assuming this as default frame
Oxyz = np.array([[1, 0 , 0], [0, 1, 0], [0, 0 ,1]], dtype=np.float64)

def getInertiaTensor(points: list):

    # Create the tensot matrix and all the possible plane combinations
    tensor = np.zeros((3, 3), dtype=np.float64)
    planes = itertools.combinations([0, 1, 2], 2)

    for ax in [0, 1, 2]:
        tensor[ax][ax] = getInertia(ax, points)

    for plane in planes:
        tensor[plane[0]][plane[1]] = 0 - getCentrifugalInertia(plane, points)
        tensor[plane[1]][plane[0]] = 0 - getCentrifugalInertia(plane, points)

    return tensor


# Working only with standard Oxyz for now
def getInertia(axis: int, points: list):

    points_inertia = []
    # points_mass = []
    if axis == 0:
        for p in points:
            point_inertia = p[1][0]*((p[0][1])**2 + p[0][2]**2)
            points_inertia.append(point_inertia)
            # points_mass.append(p[1][0])
    elif axis == 1:
        for p in points:
            point_inertia = p[1][0]*((p[0][0])**2 + p[0][2]**2)
            points_inertia.append(point_inertia)
            # points_mass.append(p[1][0])
    elif axis == 2:
        for p in points:
            point_inertia = p[1][0]*((p[0][0])**2 + p[0][1]**2)
            points_inertia.append(point_inertia)
            # points_mass.append(p[1][0])
    else:
        print('Please insert a valid axis! ')

    inertia = sum(points_inertia)
    return inertia

def getCentrifugalInertia(plane: list, points: list):

    points_inertia = []
    if plane == (0, 1):
        for p in points:
            point_inertia = p[1][0]*(p[0][0] * p[0][1])
            points_inertia.append(point_inertia)
    elif plane == (0, 2):
        for p in points:
            point_inertia = p[1][0]*(p[0][0] * p[0][2])
            points_inertia.append(point_inertia)
    elif plane == (1, 2):
        for p in points:
            point_inertia = p[1][0]*(p[0][1] * p[0][2])
            points_inertia.append(point_inertia)
    else:
        print('Please insert a valid plane! ')

    centrifugal_inertia = sum(points_inertia)
    return centrifugal_inertia

def pltElipsoid(tensor: np.ndarray):

    def getDiagonalMatrix():
        if round(np.linalg.det(tensor), 10) == 0:
            print(f'Matrix has rank smaller than 3! ')
        else:
            new_tensor = np.linalg.eig(tensor)[0]
            new_frame = np.linalg.eig(tensor)[1]
            return new_tensor, new_frame

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    theta = np.linspace(0, 2*np.pi, 50)
    etha = np.linspace(0, np.pi, 50)

    A = getDiagonalMatrix()[0][0]
    B = getDiagonalMatrix()[0][1]
    C = getDiagonalMatrix()[0][2]

    X = A*np.outer(np.cos(theta), np.sin(etha))
    Y = B*np.outer(np.sin(theta), np.sin(etha))
    Z = C*np.outer(np.ones(np.size(theta)), np.cos(etha))

    ax.plot_surface(X, Y, Z, cmap=plt.cm.cool, alpha=0.5)

    r = np.linspace(-10, 10, 100)

    ax.plot(r*getDiagonalMatrix()[1][0][0], r*getDiagonalMatrix()[1][0][1], r*getDiagonalMatrix()[1][0][2])
    ax.plot(r*getDiagonalMatrix()[1][1][0], r*getDiagonalMatrix()[1][1][1], r*getDiagonalMatrix()[1][1][2])
    ax.plot(r*getDiagonalMatrix()[1][2][0], r*getDiagonalMatrix()[1][2][1], r*getDiagonalMatrix()[1][2][2])

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))

    plt.show()