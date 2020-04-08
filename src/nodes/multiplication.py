from . import BaseNode
import numpy as np


class Multiplication(BaseNode):
    def __init__(self, x, y):
        """
            x * y
        """
        super(Multiplication, self).__init__()
        self.x = x
        self.y = y

    def forward(self):
        self.result = self.x * self.y
        return self.result

    def backward(self):
        pass
