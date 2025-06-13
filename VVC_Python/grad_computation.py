# import torch

# x = torch.rand(3, 2, 2)
# print(x)

# net = torch.nn.Conv2d(3, 3, 1)
# # make net to be identity
# net.weight.data.fill_(0)
# net.bias.data.fill_(0)
# net.weight.data[0, 0, 0, 0] = 1
# net.weight.data[1, 1, 0, 0] = 1
# net.weight.data[2, 2, 0, 0] = 1

# y = net(x)

# print(y)
# # check if y is equal to x
# print(torch.allclose(x, y))

import os
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import sys
sys.path.insert(1, '/media/EXT0/daole/sd_scripts/VVC_Python/') 
import VVC_Python
from torchvision import utils as utils_tv
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
# input = torch.randint(-10, 10, (1,)).to(torch.float32).requires_grad_(True)
# input = torch.rand(3, 2, 2).requires_grad_(True)
path = '/media/EXT0/daole/sd_scripts/VVC_Python/000000.png'
input = Image.open(path)
# input = torch.tensor(3.0).unsqueeze(0)
# print('input', input)
# target = input ** 2
target = Image.open("/media/EXT0/daole/sd_scripts/VVC_Python/000000_input.png")
target = TF.to_tensor(target)
# print('target',target)
# A MLP model
# model = nn.Sequential(
#     nn.Linear(1, 1),
#     nn.ReLU(),
#     nn.Linear(1, 1),
# )
input = TF.to_tensor(input).requires_grad_(True)
model = torch.nn.Conv2d(3, 3, 1)
# make net to be identity
model.weight.data.fill_(0)
model.bias.data.fill_(0)
model.weight.data[0, 0, 0, 0] = 1
model.weight.data[1, 1, 0, 0] = 1
model.weight.data[2, 2, 0, 0] = 1

# The output of model
# output = model(input)
utils_tv.save_image(input, os.path.join("/media/EXT0/daole/sd_scripts/VVC_Python/", f"input.png"), nrow=1, normalize=False)
output = model(input)
# print('output',output)
utils_tv.save_image(output, os.path.join("/media/EXT0/daole/sd_scripts/VVC_Python/", f"output_model.png"), nrow=1, normalize=False)

# We call the c function whose output cannot be backpropagated
# output_c = output.to(torch.int64)
# print('output_c',output_c)
# output_c = int(output_c)
# output_c = call_c_func(output_c)
output_c = VVC_Python.vvc_func(output, 1, "/media/EXT0/daole/sd_scripts/VVC_Python/")
output_c.requires_grad_(True)
utils_tv.save_image(output_c, os.path.join("/media/EXT0/daole/sd_scripts/VVC_Python/", f"compressed_image.png"), nrow=1, normalize=False)
# output_c = torch.tensor([output_c], dtype=torch.float32, requires_grad=True)
# print('output_c',output_c)
# Initially the model does not contain gradient
print("Initially the model does not contain gradient")
print_gradient(model)

# We compute the loss between output_c and target
# print(target.size())
# print(output_c.size())
# print(output.size())
loss = F.mse_loss(output_c, target)

# This will propagate the gradient of loss to output_c
loss.backward()

# This will bypass the gradient of output_c to output !
output.backward(output_c.grad)
print('input.grad',input.grad)
# Now the model has gradient. So we can train it :)
print("Now the model has gradient. So we can train it :)")
print_gradient(model)
