import numpy as np

from optimizers.optimizer import Optimizer


class RMSProp(Optimizer):

    def __init__(self, gradient, params, step_size=0.01, max_iter=1000, beta=0.9, epsilon=1e-10):
        super(RMSProp, self).__init__(gradient, params, step_size, max_iter)
        self.beta = beta
        self.epsilon = epsilon
        self.s = np.array([0, 0])

    def update_params(self):
        gradient_value = self.gradient(self.params)
        self.s = self.beta*self.s + (1 - self.beta)*np.square(gradient_value)
        self.params = self.params - self.step_size*(gradient_value / (np.sqrt(self.epsilon + self.s)))
