import nodes

# forward process
m1 = nodes.Multiply(2, -1)
m2 = nodes.Multiply(-3, -2)
s1 = nodes.Add(m1.forward(), m2.forward())
s2 = nodes.Add(s1.forward(), -3)
sig1 = nodes.Sigmoid(s2.forward())
result = sig1.forward()

# back-propagation process
b_sig1 = sig1.backward(1)
b_s2 = s2.backward(b_sig1)
# partial derivative of -3
print(b_s2[1])
b_s1 = s1.backward(b_s2[0])
b_m2 = m2.backward(b_s1[1])
b_m1 = m1.backward(b_s1[0])
# b_m1[0] is the partial derivative of 2
# b_m1[1] is the partial derivative of -1
print(b_m1)
# b_m2[0] is the partial derivative of -3
# b_m2[1] is the partial derivative of -2
print(b_m2)
