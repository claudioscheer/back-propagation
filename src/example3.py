import nodes
import math
import torch

# forward process
m1 = nodes.Multiply(2, -1)
m2 = nodes.Multiply(-3, -2)
a1 = nodes.Add(m1.forward(), m2.forward())
a2 = nodes.Add(a1.forward(), -3)
m3 = nodes.Multiply(a2.forward(), -1)
exp1 = nodes.Power(math.e, m3.forward())
a3 = nodes.Add(exp1.forward(), 1)
d1 = nodes.Divide(1, a3.forward())
result = d1.forward()

# back-propagation process
b_d1 = d1.backward(1)
b_a3 = a3.backward(b_d1[1])
b_exp1 = exp1.backward(b_a3[0])
b_m3 = m3.backward(b_exp1[1])
b_a2 = a2.backward(b_m3[0])
# partial derivative of -3
print(b_a2[1])
b_a1 = a1.backward(b_a2[0])
b_m2 = m2.backward(b_a1[1])
b_m1 = m1.backward(b_a1[0])
# b_m1[0] is the partial derivative of 2
# b_m1[1] is the partial derivative of -1
print(b_m1)
# b_m2[0] is the partial derivative of -3
# b_m2[1] is the partial derivative of -2
print(b_m2)

# validation with PyTorch
t1 = torch.tensor(2.0, requires_grad=True)
t2 = torch.tensor(-1.0, requires_grad=True)
t3 = torch.tensor(-3.0, requires_grad=True)
t4 = torch.tensor(-2.0, requires_grad=True)
t5 = torch.tensor(-3.0, requires_grad=True)
result = ((t1 * t2) + (t3 * t4)) + t5
# Sigmoid function
result = 1 / (torch.exp((result * -1)) + 1)
result.backward()
print(t1.grad)
print(t2.grad)
print(t3.grad)
print(t4.grad)
print(t5.grad)
