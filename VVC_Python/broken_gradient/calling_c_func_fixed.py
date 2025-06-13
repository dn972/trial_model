import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper to call c function
def call_c_func(input_image):
    so_file = "/media/EXT0/daole/sd_scripts/VVC_Python/broken_gradient/c_func.so"
    c_func = ctypes.CDLL(so_file)
    return c_func.square(input_image)

# Helper to print model gradient
def print_gradient(model):
    for i, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
            print(f"{i}th layer {m._get_name()}: ", m.weight.grad)

# Input and target tensors of size (1,), (1,)
# input = torch.randint(-10, 10, (1,)).to(torch.float32)
torch.manual_seed(0)
input = torch.randint(-10, 10, (1,)).to(torch.float32).requires_grad_(True)
print('input x', input)
target = input ** 2
print('target z',target)
# A MLP model
model = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1),
)
# The output of model
output = model(input)
print('output x^',output)
# We call the c function whose output cannot be backpropagated
output_c = output.to(torch.int64)
output_c = int(output_c)
output_c = call_c_func(output_c)
output_c = torch.tensor([output_c], dtype=torch.float32, requires_grad=True)
print('output y',output_c)
# Initially the model does not contain gradient
print("Initially the model does not contain gradient")
print_gradient(model)

# We compute the loss between output_c and target
loss = F.mse_loss(output_c, target)

# This will propagate the gradient of loss to output_c
loss.backward()

# This will bypass the gradient of output_c to output !
# print(output_c.grad)
output.backward(output_c.grad)
# output.backward()
print('dx=2x',input.grad) 
# Now the model has gradient. So we can train it :)
print("Now the model has gradient. So we can train it :)")
print_gradient(model)
