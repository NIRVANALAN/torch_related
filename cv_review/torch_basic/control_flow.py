from Helper import *
import torch.nn as nn
import random

# device = torch.device('cpu')
device = torch.device('cuda')


class DynamicNet(nn.Module):
	
	def __init__(self, D_In, H, D_Out):
		super().__init__()
		self.input_layer = nn.Linear(D_In, H)
		self.middle_layer = nn.Linear(H, H)
		self.output_layer = nn.Linear(H, D_Out)
	
	def forward(self, x):
		h_relu = self.input_layer(x).clamp(min=0)
		for _ in range(random.randint(0, 3)):
			h_relu = self.middle_layer(h_relu).clamp(min=0)
		y_pred = self.output_layer(h_relu)
		return y_pred
		pass


N, D_In, H, D_Out = 64, 1000, 100, 10
# model = nn.Sequential(
# 	nn.Linear(D_In, H),
# 	nn.ReLU(),
# 	nn.Linear(H, D_Out),
# ).to(device)

model = DynamicNet(D_In, H, D_Out).to(device)

x = torch.randn(N, D_In, device=device)
y = torch.randn(N, D_Out, device=device)

criterion = nn.MSELoss(reduction='sum')
# lr = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
	y_pred = model(x)
	loss = criterion(y_pred, y)  # compute loss
	print(t, loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
# with torch.no_grad():
# 	for param in model.parameters():
# 		param -= lr * param.grad
