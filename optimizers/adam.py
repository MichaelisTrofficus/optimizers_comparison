import numpy as np

from optimizers.optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self, gradient, params, step_size=0.001, max_iter=1000, beta1=0.93, beta2=0.95, epsilon=1e-08):
        super(Adam, self).__init__(gradient, params, step_size, max_iter)
        self.iteration = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.array([0., 0.], dtype=np.float64)
        self.s = np.array([0., 0.], dtype=np.float64)

    def update_params(self):
        gradient_value = self.gradient(self.params)
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient_value
        self.s = self.beta2*self.s + (1 - self.beta2)*(gradient_value**2)
        self.m = self.m / (1 - self.beta1**self.iteration)
        self.s = self.s / (1 - self.beta2**self.iteration)

        update_step = self.step_size*self.m / (np.sqrt(self.s) + self.epsilon)
        self.params = self.params - update_step
        self.iteration += 1
