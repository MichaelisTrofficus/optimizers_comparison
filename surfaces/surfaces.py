import autograd.numpy as np

def hyperbolic_paraboloid(params):
    """
    :param params: Params of the solution space
    :return: y = x_1**2 - x_2**2
    """
    squared = np.square(params)
    return squared[0] - squared[1]
