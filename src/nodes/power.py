from . import BaseNode


def _f(x, y):
    return x ** y


class Power(BaseNode):
    def __init__(self, x, y):
        """
            x ** y
        """
        super(Power, self).__init__()
        self.x = x
        self.y = y

    def forward(self):
        self.forward_result = _f(self.x, self.y)
        return self.forward_result

    def backward(self, previous):
        # store the partial derivative for each input
        self.local_gradient_x = (_f(self.x + 1e-7, self.y) - self.forward_result) / 1e-7
        self.local_gradient_y = (_f(self.x, self.y + 1e-7) - self.forward_result) / 1e-7
        self.backward_gradient_x = previous * self.local_gradient_x
        self.backward_gradient_y = previous * self.local_gradient_y
        return [self.backward_gradient_x, self.backward_gradient_y]
