import nodes
import torch

# forward process
s1 = nodes.Add(-2, 5)
m1 = nodes.Multiply(s1.forward(), -4)
result = m1.forward()

# back-propagation process
# 1 is the derivative from the final result to itself
# same as ((result + 1e-7) - result) / 1e-7
b_m1 = m1.backward(1)
# partial derivative of -4
print(b_m1[1])
b_s1 = s1.backward(b_m1[0])
# partial derivative of -2
print(b_s1[0])
# partial derivative of 5
print(b_s1[1])

# validation with PyTorch
t1 = torch.tensor(-2.0, requires_grad=True)
t2 = torch.tensor(5.0, requires_grad=True)
t3 = torch.tensor(-4.0, requires_grad=True)
result = (t1 + t2) * t3
result.backward()
print(t1.grad)
print(t2.grad)
print(t3.grad)
