import autograd.numpy as np

from surfaces.surfaces import hyperbolic_paraboloid


def hyperbolic_paraboloid_fig(n):
    """
    Generates n elements following a paraboloid's equation
    :param n: Number of elements to generate
    :return: x, y, z
    """
    x1 = np.linspace(-1, 1, n)
    x2 = x1.copy()
    x3 = np.zeros(shape=(x1.shape[0], x2.shape[0]))

    for i, a1 in enumerate(x1):
        for j, a2 in enumerate(x2):
            params = np.array([a2, a1])
            x3[i, j] = hyperbolic_paraboloid(params)

    return x1, x2, x3
