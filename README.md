# **Inertia elipsoid**

The inertia elipsoid is the quadric with equation:

$$ Ax^2 + By^2 + Cz^2 - 2Dyz - 2Exz - 2Fxy = \lambda^2$$

where $A$, $B$, $C$, $D$, $E$ and $F$ are the components of the tensor Matrix.

## How to run

After downloading the necessary files follow this [guide](https://github.com/chessparov/chessparov/blob/main/python-setup.md) on how to install python and all the necessary packages.

In the ``Inertia.py`` file are stored all the necessary functions, while the Jupyter Notebook is setup in order to offer an environment where to input data and visualize the output.

## Creating and visualizing the elipsoid

- **Adding particles**
  You can create a new point or modify an existing one; each point is defined by a list containing a float np.ndarray with shape (3, 1) and a list containing a single element.

  The np.ndarray contains the coordinates $x$, $y$, $z$, while the list contains the mass of the particle.
- **Visualizing the tensor matrix**
    In the Jupyter Notebook, after having created the necessary particles, add them to a list and give the list to the `getinertiaTensor(list)` function
- **Visualizing the elipsoid**
    If the matrix is coherent to the results expected, and has no anomalies, plot the elipsoid by running
  `pltElipsoid(list)`
  and the list containing your points as an argument
