from . import BaseNode
import math


def _f(x):
    return 1 / (1 + math.exp(-x))


class Sigmoid(BaseNode):
    def __init__(self, x):
        """
            Sigmoid activation function
        """
        super(Sigmoid, self).__init__()
        self.x = x

    def forward(self):
        self.forward_result = _f(self.x)
        return self.forward_result

    def backward(self, previous):
        # formula to get the derivative of the Sigmoid function
        self.backward_gradient = (
            (1 - self.forward_result) * self.forward_result
        ) * previous
        return self.backward_gradient
