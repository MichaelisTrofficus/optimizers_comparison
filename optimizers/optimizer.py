import abc
import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    """
    Base class for any optimizer
    """
    def __init__(self, gradient, params, step_size=0.01, max_iter=1000):
        self.gradient = gradient
        self.params = params
        self.step_size = step_size
        self.max_iter = max_iter

    @abc.abstractmethod
    def update_params(self):
        raise NotImplementedError

    def get_history(self):
        params_history = [self.params]

        for _ in range(self.max_iter):
            self.update_params()

            if (-1 <= self.params[0] <= 1) and (-1 <= self.params[1] <= 1):
                params_history += [self.params.copy()]
            else:
                new_params = np.array([0., 0.])
                if self.params[0] >= 1:
                    new_params[0] = 1
                if self.params[0] <= -1:
                    new_params[0] = -1
                if self.params[1] >= 1:
                    new_params[1] = 1
                if self.params[1] <= -1:
                    new_params[1] = -1
                params_history += [new_params]

        return np.array(params_history)
