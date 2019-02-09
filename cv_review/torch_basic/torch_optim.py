from Helper import *
import torch.nn as nn

# device = torch.device('cpu')
device = torch.device('cuda')
N, D_In, H, D_Out = 64, 1000, 100, 10
model = nn.Sequential(
	nn.Linear(D_In, H),
	nn.ReLU(),
	nn.Linear(H, D_Out),
).to(device)

x = torch.randn(N, D_In, device=device)
y = torch.randn(N, D_Out, device=device)

loss_fn = nn.MSELoss(reduction='sum')
lr = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr)


for t in range(500):
	y_pred = model(x)
	loss = loss_fn(y_pred, y)  # compute loss
	print(t, loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	# with torch.no_grad():
	# 	for param in model.parameters():
	# 		param -= lr * param.grad
