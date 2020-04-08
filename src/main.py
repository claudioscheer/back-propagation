import nodes

node_sum = nodes.Sum(-2, 5)
node_multiplication = nodes.Multiplication(node_sum.forward(), -4)
result = node_multiplication.forward()
print(result)
