from . import BaseNode
import numpy as np


def _f(q, z):
    return q * z


class Multiplication(BaseNode):
    def __init__(self, q, z):
        """
            q * y
        """
        super(Multiplication, self).__init__()
        self.q = q
        self.z = z

    def forward(self):
        self.forward_result = _f(self.q, self.z)
        return self.forward_result

    def backward(self, previous):
        # store the partial derivative for each input
        self.local_gradient_q = (_f(self.q + 1e-7, self.z) - self.forward_result) / 1e-7
        self.local_gradient_z = (_f(self.q, self.z + 1e-7) - self.forward_result) / 1e-7
        self.backward_gradient_q = previous * self.local_gradient_q
        self.backward_gradient_z = previous * self.local_gradient_z
