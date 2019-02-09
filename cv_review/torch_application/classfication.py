import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt

# device = torch.device('cpu')
device = torch.device('cuda')

torch.manual_seed(1)
t1 = time.time()
#  dataset
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor).to(device)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor).to(device)  # shape (200,) LongTensor = 64-bit integer


# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
		self.out = torch.nn.Linear(n_hidden, n_output)  # output layer
	
	def forward(self, x):
		x = F.relu(self.hidden(x))  # activation function for hidden layer
		x = self.out(x)
		return x


# net = Net(n_feature=2, n_hidden=10, n_output=2)  # define the network
net = nn.Sequential(
	nn.Linear(2, 10),
	nn.ReLU(),
	nn.Linear(10, 2)  # output dimension equals the number of class
).to(device)
# [0, 1]
# [1, 0]
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()  # something about plotting

for t in range(100):
	out = net(x)  # input x and predict based on x
	loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
	
	optimizer.zero_grad()  # clear gradients for next train
	loss.backward()  # backpropagation, compute gradients
	optimizer.step()  # apply gradients
	
	if t % 2 == 0:
		print(t, loss.data.cpu().numpy())
		plt.cla()
		prediction = torch.max(out, 1)[1]  # the second return value is argmax
		predicted_y = prediction.data.cpu().numpy()
		target_y = y.data.cpu().numpy()
		plt.scatter(x.data.cpu().numpy()[:, 0], x.data.cpu().numpy()[:, 1], s=100, c=predicted_y, lw=0, cmap='RdYlGn')
		accuracy = float((predicted_y == target_y).astype(int).sum()) / float(target_y.size)
		plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.1)
torch.save(net, 'net_classification.pkl')
torch.save(net.state_dict(), 'net_classification_params.pkl')
# net = torch.load('net_classification.pkl')
# net.load_state_dict(torch.load('net_classification_params.pkl'))
plt.ioff()
plt.show()

print(time.time() - t1)
