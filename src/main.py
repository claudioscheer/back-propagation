import nodes

node_sum = nodes.Sum(-2, 5)
node_multiplication = nodes.Multiplication(node_sum.forward(), -4)
result = node_multiplication.forward()

# derivative from the final result to itself
# same as ((result + 1e-7) - result) / 1e-7
# always equal to 1
global_gradient = 1
node_multiplication.backward(global_gradient)
print(node_multiplication.backward_gradient_z)
node_sum.backward(node_multiplication.backward_gradient_q)
print(node_sum.backward_gradient_x)
print(node_sum.backward_gradient_y)
