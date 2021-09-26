import numpy as np

from optimizers.optimizer import Optimizer


class Momentum(Optimizer):

    def __init__(self, gradient, params, step_size=0.01, max_iter=1000, beta=0.9):
        super(Momentum, self).__init__(gradient, params, step_size, max_iter)
        self.beta = beta
        self.m = np.array([0, 0])

    def update_params(self):
        gradient_value = self.gradient(self.params)
        self.m = self.beta*self.m + self.step_size*gradient_value
        self.params -= self.m
