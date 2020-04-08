from . import BaseNode
import numpy as np


class Sum(BaseNode):
    def __init__(self, x, y):
        """
            x + y
        """
        super(Sum, self).__init__()
        self.x = x
        self.y = y

    def forward(self):
        self.result = self.x + self.y
        return self.result

    def backward(self):
        pass
