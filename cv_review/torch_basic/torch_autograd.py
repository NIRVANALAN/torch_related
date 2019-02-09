import torch
from Helper import *

device = torch.device('cpu')
# device = torch.device('cuda')
# torch.cuda.set_device(3)

# dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run GPU

# N is batch size; D_in is input dimenssion
# H is hidden dimenssion; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random input and output data
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
	y_pred = ReLU.apply(x.mm(w1)).mm(w2)
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.item())
	
	loss.backward()
	with torch.no_grad():  # use context manager to prevent PyTorch from building a computational graph
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad
		w1.grad.zero_()
		w2.grad.zero_()

# for t in range(500):
# 	# Forward pss: compute predicted y
# 	h = x.mm(w1)
# 	h_relu = h.clamp(min=0)
# 	y_pred = h_relu.mm(w2)
#
# 	# compute and print loss
# 	loss = (y_pred - y).pow(2).sum()
# 	print(t, loss)
#
# 	# backprop to compute gradients of w1 and w2 with respect to loss
# 	grad_y_pred = 2.0 * (y_pred - y)
# 	grad_w2 = h_relu.t().mm(grad_y_pred)
# 	grad_h_relu = grad_y_pred.mm(w2.t())
# 	grad_h = grad_h_relu.clone()
# 	grad_h[h < 0] = 0
# 	grad_w1 = x.t().mm(grad_h)
#
# 	# update weights using gradient descent
# 	w1 -= learning_rate * grad_w1
# 	w2 -= learning_rate * grad_w2
