import nodes

# forward process
s1 = nodes.Sum(-2, 5)
m1 = nodes.Multiplication(s1.forward(), -4)
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
