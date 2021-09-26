import numpy as np

from optimizers.optimizer import Optimizer


class Adagrad(Optimizer):

    def __init__(self, gradient, params, step_size=0.05, max_iter=1000, epsilon=1e-10):
        super(Adagrad, self).__init__(gradient, params, step_size, max_iter)
        self.epsilon = epsilon
        self.s = np.array([0, 0])

    def update_params(self):
        gradient_value = self.gradient(self.params)
        self.s = self.s + np.square(gradient_value)
        self.params = self.params - self.step_size*(gradient_value / (np.sqrt(self.epsilon + self.s)))
