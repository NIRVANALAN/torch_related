import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super().__init__()
		self.hidden = nn.Linear(n_feature, n_hidden)
		self.predict = nn.Linear(n_hidden, n_output)
	
	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x
		pass


# dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# print(x)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# torch can train on Tensor now

net = Net(1, 10, 1)
optimizer = optim.Adam(net.parameters(), lr=0.1)
criterion = nn.MSELoss()

plt.ion()  # interactive

for t in range(100):
	prediction = net(x)
	loss = criterion(prediction, y)
	optimizer.zero_grad()  # in pytorch, grads are accumulated rather than replaced
	loss.backward()
	optimizer.step()
	
	if t % 5 == 0:
		print(t, loss.data.numpy())
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=4)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.1)  # at least show 0.1 second

plt.ioff()
plt.show()
