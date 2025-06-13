from ctypes import *
import torch
import torch.nn.functional as F
x = torch.tensor(3.0, requires_grad= True)
x_torch = torch.tensor(3.0, requires_grad= True)

# # Compute the loss in pytorch
def square(x):
    return x*x
x_square_torch = square(x_torch)
print("The input x:", x.item()) #print input x
print("The output x^2:",x_square_torch)
y1 = torch.tensor(8, dtype=torch.float, requires_grad=True)
loss_torch = F.mse_loss(x_square_torch, y1)
x_square_torch.backward()
print('dx=2x in pytorch', x_torch.grad)
# Use PyTorch to compute the gradient of the loss with respect to x
# grads_torch = torch.autograd.grad(loss_torch, x_torch, allow_unused=True)
# print('The gradient of loss with respect to x: (pytorch) ',grads_torch)

# x.requires_grad = True
def calling_c_func(input_image):
    so_file = "c_func.so"
    c_func = CDLL(so_file)
    return c_func.square(input_image)
print("The input:", x.item()) #print input x
x_square = calling_c_func(int(x.item()))
print("The output:",x_square)
# Define the target y as a PyTorch tensor
y = torch.tensor(8, dtype=torch.float, requires_grad=True)
x_square =torch.tensor(x_square, dtype=torch.float, requires_grad=True)#convert x_square to tensor
x_square.clone().detach().requires_grad_(True)
# Compute the loss
loss = F.mse_loss(x_square, y)
loss.backward()
x_square.backward(gradient = x_square.grad.clone().detach().requires_grad_(True))
print('dx=2x', x.grad)
# x.backward(gradient=x_square.grad)
# print('x_square.grad', x_square.grad)
# print('x.grad', x.grad)
# Use PyTorch to compute the gradient of the loss with respect to x
# grads = torch.autograd.grad(loss, x, allow_m x: (c function) ',grads)
###########################
# with torch.no_grad():
#     x_square = calling_c_func(int(x.item()))
    

# x_square_detached = torch.tensor(x_square, dtype=torch.float, requires_grad=True).detach()
# x_square_detached.eig()

# # Define the target z as a PyTorch tensor
# y = torch.tensor(3, dtype=torch.float, requires_grad=True)
# # y =torch.tensor(y, dtype=torch.float, requires_grad=True)#convert y to tensor
# # Compute the loss
# loss = F.mse_loss(x_square_detached, y)
# loss.backward()
# # Use PyTorch to compute the gradient of the loss with respect to x
# grads = torch.autograd.grad(loss, x)
# # grads = torch.autograd.grad(loss, x, allow_unused=True)
# print('The gradient of loss with respect to x: ',grads)
#https://www.geeksforgeeks.org/python-pytorch-backward-function/