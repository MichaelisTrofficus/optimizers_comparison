from optimizers.optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, gradient, params, step_size=0.03, max_iter=1000):
        super(SGD, self).__init__(gradient, params, step_size, max_iter)

    def update_params(self):
        gradient_value = self.gradient(self.params)
        self.params -= self.step_size * gradient_value
