import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from scipy.integrate import nquad

# Assuming this as default frame
Oxyz = np.array([[1, 0 , 0], [0, 1, 0], [0, 0 ,1]], dtype=np.float64)


class Continuum:
    '''

    A class representing a plane figure. It's boundaries need to be formalized as if you have to calculate a double integral,
    so the limit of this kind of representation is you have to be able to put down in a suitable form the x and y boundaries.

    :param float y_lower_bound: The y-axis lower bound of integration
    :param list y_upper_bound: The coefficients of the y-axis
    upper bound of integration function (e.g. [3, - 2, 0] -> 3 - (2 * x))
    :param list x_domain: The interval in which varies the x variable
    :param float area: The area of the figure we are considering
    :param float mass: The mass of the figure we are considering

    '''

    def __init__(self, y_lower_bound: float, y_upper_bound: list, x_domain: list, area: float, mass: float):
        self.y_lower = y_lower_bound
        self.y_upper = y_upper_bound
        self.x = x_domain
        self.area = area
        self.mass = mass


def getInertiaTensor(continuum: bool, system: Continuum | list) -> np.ndarray:
    '''

    This function is able to evaluate the Inertia tensor of a system of points,
    either discrete or belonging to a continuum.

    :param bool continuum: If set to True, accepts a continuum body as system,
    otherwise accepts a list of discrete points
    :param Continuum | list system: Either a list of discrete points or a continuum figure
    :return: The 3x3 matrix representing the inertia tensor

    '''

    # Create the tensor matrix and all the possible plane combinations
    tensor = np.zeros((3, 3), dtype=np.float64)
    planes = itertools.combinations([0, 1, 2], 2)

    if continuum:
        for ax in [0, 1, 2]:
            tensor[ax][ax] = getInertiaContinuum(ax, system)

        for plane in planes:
            tensor[plane[0]][plane[1]] = 0 - getCInertiaContinuum(plane, system)
            tensor[plane[1]][plane[0]] = 0 - getCInertiaContinuum(plane, system)

        return tensor

    else:
        for ax in [0, 1, 2]:
            tensor[ax][ax] = getInertia(ax, system)

        for plane in planes:
            tensor[plane[0]][plane[1]] = 0 - getCentrifugalInertia(plane, system)
            tensor[plane[1]][plane[0]] = 0 - getCentrifugalInertia(plane, system)

        return tensor


# Working only with standard Oxyz for now
def getInertia(axis: int, points: list):
    '''

   Calculates the inertia tensor around a given axis of a system of discrete points

    :param int axis: 0 -> x-axis, 1 -> y-axis, 2 -> z-axis
    :param points: A list of discrete points
    :return: The inertia momentum of the given axis

    '''
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
    '''

    Calculates the centrifugal momentum around a given plane of a system made of discrete points

    :param plane: (0, 1) -> xy, (0, 2) -> xz, (1, 2) -> yz
    :param points: A list of discrete points
    :return: The inertia momentum of the given plane

    '''

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

# Inertia of a plane figure
def getInertiaContinuum(axis: int, continuum: Continuum):
    '''

    Calculates the inertia momentum around a given axis of a continuum body

    :param int axis: 0 -> x-axis, 1 -> y-axis, 2 -> z-axis
    :param points: A list of discrete points
    :return: The inertia momentum around the given axis

    '''

    y_lower_bound = continuum.y_lower
    y_upper_bound = continuum.y_upper
    x_domain = continuum.x
    area = continuum.area
    mass = continuum.mass

    if mass > 0:
        if area > 0:
            if len(x_domain) == 2:
                if axis == 0:

                    def f(x, y):
                        return y**2

                    def bounds_y(x):
                        return [y_lower_bound, y_upper_bound[0] + y_upper_bound[1] * x + y_upper_bound[2]* (x**2)]

                    try:
                        fltInertia = nquad(f, [bounds_y, x_domain])
                        return fltInertia[0] * (mass/area)
                    except:
                        print(f'Please check your input variables! ')

                elif axis == 1:

                    def f(x, y):
                        return x**2

                    def bounds_y(x):
                        return [y_lower_bound, y_upper_bound[0] + y_upper_bound[1] * x + y_upper_bound[2] * (x ** 2)]
                    try:
                        fltInertia = nquad(f, [bounds_y, x_domain])
                        return fltInertia[0] * (mass/area)
                    except:
                        print(f'Please check your variables')

                elif axis == 2:

                    def f(x, y):
                        return y**2 + x**2

                    def bounds_y(x):
                        return [y_lower_bound, y_upper_bound[0] + y_upper_bound[1] * x + y_upper_bound[2] * (x ** 2)]
                    try:
                        fltInertia = nquad(f, [bounds_y, x_domain])
                        return fltInertia[0] * (mass/area)
                    except:
                        print('Please check your variables! ')

                else:
                    print('Please insert a valid axis! ')
            else:
                print(f'Please insert a valid y_domain! ')
        else:
            print('Area cannot be negative! ')
    else:
        print('Mass cannot be negative! ')

def getCInertiaContinuum(plane: tuple, continuum: Continuum):
    '''

   Calculates the centrifugal momentum around a given plane of a continuum body.

    :param plane: (0, 1) -> xy, (0, 2) -> xz, (1, 2) -> yz
    :param points: A list of discrete points
    :return: The inertia momentum of the given plane

    '''

    y_lower_bound = continuum.y_lower
    y_upper_bound = continuum.y_upper
    x_domain = continuum.x
    area = continuum.area
    mass = continuum.mass

    if mass > 0:
        if area > 0:
            if len(x_domain) == 2:
                if plane == (0, 1):

                    def f(x, y):
                        return x*y

                    def bounds_y(x):
                        return [y_lower_bound, y_upper_bound[0] + y_upper_bound[1] * x + y_upper_bound[2] * (x ** 2)]

                    try:
                        fltInertia = nquad(f, [bounds_y, x_domain])
                        return fltInertia[0] * (mass/area)
                    except:
                        print(f'Please check your input variables! ')

                elif plane == (0, 2):
                    return 0

                elif plane == (1, 2):
                    return 0

                else:
                    print('Please insert a valid axis! ')
            else:
                print(f'Please insert a valid y_domain! ')
        else:
            print('Area cannot be negative! ')
    else:
        print('Mass cannot be negative! ')

def pltElipsoid(tensor: np.ndarray):
    '''

    Plots in 3D the ellipsoid related to the inertia tensor of a given system.

    :param tensor: The inertia tensor of a body or system of discrete points

    '''

    def getDiagonalMatrix():
        if round(np.linalg.det(tensor), 10) == 0:
            print(f'Matrix has rank smaller than 3! ')
        else:
            new_tensor = np.linalg.eig(tensor)[0]
            new_frame = np.linalg.eig(tensor)[1]
            return new_tensor, new_frame

    new_tensor = getDiagonalMatrix()[0]
    new_frame = getDiagonalMatrix()[1]

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(projection='3d')
    theta = np.linspace(0, 2*np.pi, 100)
    etha = np.linspace(0, np.pi, 100)

    A = new_tensor[0]
    B = new_tensor[1]
    C = new_tensor[2]
    X = A*np.outer(np.cos(theta), np.sin(etha))
    Y = B*np.outer(np.sin(theta), np.sin(etha))
    Z = C*np.outer(np.ones(np.size(theta)), np.cos(etha))

    ax.plot_surface(X, Y, Z, cmap=plt.cm.cool, alpha=0.5, rstride=10, cstride=10, linewidth=0.1, antialiased=False)

    r = np.linspace(-10, 10, 100)

    ax.plot(r*new_frame[0][0], r*new_frame[0][1], r*new_frame[0][2])
    ax.plot(r*new_frame[1][0], r*new_frame[1][1], r*new_frame[1][2])
    ax.plot(r*new_frame[2][0], r*new_frame[2][1], r*new_frame[2][2])

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    RATIO = round(np.max([np.max(X), np.max(Y), np.max(Z)]))
    ax.set_xlim(-RATIO, RATIO)
    ax.set_ylim(-RATIO, RATIO)
    ax.set_zlim(-RATIO, RATIO)

    plt.show()

